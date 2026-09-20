"""
orcha.integrations.langgraph.checkpoint
=======================================
Durable LangGraph checkpointer backed by the project's own SQLite store.

LangGraph graphs are compiled with a ``BaseCheckpointSaver``; the
default in-memory ``MemorySaver`` loses threads on restart. This module
provides ``OrchaSqliteCheckpointer`` — a LangGraph checkpointer whose
checkpoints and pending writes live in a local SQLite database
(stdlib ``sqlite3``, no new dependency, matching the local-first rule).

Threads survive process restarts: compile a graph with the same
database path, and ``LangGraphEngine`` (which is checkpointer-agnostic —
it only talks to the compiled graph) can resume paused approval threads,
replay history, and read snapshots from disk.

Usage::

    from orcha.integrations.langgraph import build_sqlite_checkpointer
    from orcha.builders import build_agent_runner

    runner = build_agent_runner(
        AgentGraphConfig(...),
        engine="langgraph",
        checkpointer=build_sqlite_checkpointer("~/.orcha/langgraph_threads.db"),
    )

The saver serializes whole checkpoint dicts through the same
OrchaPacket-aware serde as ``MemorySaver``, so packets, tool results and
interrupt payloads round-trip byte-identically.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple

from .prototype import orcha_serde


def build_sqlite_checkpointer(
    path: Optional[str] = None,
    *,
    serde: Any = None,
) -> "OrchaSqliteCheckpointer":
    """
    Build a durable LangGraph checkpointer at ``path`` (default
    ``~/.orcha/langgraph_threads.db``, override with
    ``ORCHA_LANGGRAPH_DB``).
    """
    if path is None:
        path = os.environ.get("ORCHA_LANGGRAPH_DB") or (
            str(Path.home() / ".orcha" / "langgraph_threads.db")
        )
    return OrchaSqliteCheckpointer(path, serde=serde)


class OrchaSqliteCheckpointer(BaseCheckpointSaver):
    """
    LangGraph checkpointer persisted to a single SQLite file.

    Schema mirrors the checkpoint/writes split of LangGraph's own
    ``SqliteSaver``: one row per checkpoint (blob-serialized through
    ``serde``), one row per pending write. All access funnels through a
    thread-local connection guarded by a lock, so the saver is safe under
    both the sync and async LangGraph execution paths.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS checkpoints (
        thread_id           TEXT NOT NULL,
        checkpoint_ns       TEXT NOT NULL DEFAULT '',
        checkpoint_id       TEXT NOT NULL,
        parent_checkpoint_id TEXT,
        checkpoint_type     TEXT NOT NULL,
        checkpoint          BLOB NOT NULL,
        metadata_type       TEXT NOT NULL,
        metadata            BLOB NOT NULL,
        seq                 INTEGER PRIMARY KEY AUTOINCREMENT,
        UNIQUE(thread_id, checkpoint_ns, checkpoint_id)
    );
    CREATE INDEX IF NOT EXISTS idx_cp_thread ON checkpoints(thread_id, checkpoint_ns);
    CREATE TABLE IF NOT EXISTS writes (
        thread_id     TEXT NOT NULL,
        checkpoint_ns TEXT NOT NULL DEFAULT '',
        checkpoint_id TEXT NOT NULL,
        task_id       TEXT NOT NULL,
        idx           INTEGER NOT NULL,
        channel       TEXT NOT NULL,
        value_type    TEXT NOT NULL,
        value         BLOB NOT NULL,
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
    );
    CREATE INDEX IF NOT EXISTS idx_w_thread ON writes(thread_id, checkpoint_ns, checkpoint_id);
    """

    def __init__(self, path: str = "orcha_langgraph.db", *, serde: Any = None) -> None:
        super().__init__(serde=serde or orcha_serde())
        self._path = str(path)
        self._lock = threading.Lock()
        self._tls = threading.local()
        self._async_lock = asyncio.Lock()
        self._conns: set = set()
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn().executescript(self._SCHEMA)

    # ── Connection plumbing ────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._tls, "conn", None)
        if c is None:
            c = sqlite3.connect(self._path, check_same_thread=False)
            c.row_factory = sqlite3.Row
            self._tls.conn = c
            self._conns.add(c)
        return c

    def close(self) -> None:
        """Release every open connection (LangGraph's async path runs the
        saver through worker threads, so each thread owns a connection;
        Windows keeps DB files locked while any connection is open)."""
        with self._lock:
            conns = list(self._conns)
            self._conns.clear()
        for c in conns:
            try:
                c.close()
            except Exception:
                pass
        if getattr(self._tls, "conn", None) is not None:
            self._tls.conn = None

    def _thread_key(self, config: Dict[str, Any]) -> tuple:
        cfg = config.get("configurable", {})
        return (
            cfg.get("thread_id", ""),
            cfg.get("checkpoint_ns", ""),
            cfg.get("checkpoint_id"),
        )

    def _tuple_config(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> Dict[str, Any]:
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            },
        }

    # ── Core reads/writes (sync, thread-safe) ──────────────────────────

    def _get_tuple(
        self, thread_id: str, checkpoint_ns: str = "", checkpoint_id: Optional[str] = None,
    ) -> Optional[CheckpointTuple]:
        with self._lock:
            conn = self._conn()
            if checkpoint_id:
                row = conn.execute(
                    "SELECT * FROM checkpoints "
                    "WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?",
                    (thread_id, checkpoint_ns, checkpoint_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM checkpoints WHERE thread_id=? AND checkpoint_ns=? "
                    "ORDER BY seq DESC LIMIT 1",
                    (thread_id, checkpoint_ns),
                ).fetchone()
            if row is None:
                return None
            return self._row_to_tuple(row)

    def _row_to_tuple(self, row: sqlite3.Row) -> CheckpointTuple:
        thread_id = row["thread_id"]
        checkpoint_ns = row["checkpoint_ns"]
        checkpoint_id = row["checkpoint_id"]
        checkpoint = self.serde.loads_typed(
            (row["checkpoint_type"], row["checkpoint"])
        )
        metadata = self.serde.loads_typed(
            (row["metadata_type"], row["metadata"])
        )
        parent_id = row["parent_checkpoint_id"]
        parent_config = None
        if parent_id:
            parent_config = self._tuple_config(thread_id, checkpoint_ns, parent_id)
        pending_writes: List[tuple] = []
        for w in self._conn().execute(
            "SELECT * FROM writes WHERE thread_id=? AND checkpoint_ns=? "
            "AND checkpoint_id=? ORDER BY idx ASC",
            (thread_id, checkpoint_ns, checkpoint_id),
        ).fetchall():
            pending_writes.append(
                (
                    w["task_id"], w["channel"],
                    self.serde.loads_typed((w["value_type"], w["value"])),
                )
            )
        return CheckpointTuple(
            config=self._tuple_config(thread_id, checkpoint_ns, checkpoint_id),
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def _write(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str,
        parent_checkpoint_id: Optional[str], checkpoint: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        cp_type, cp_bytes = self.serde.dumps_typed(checkpoint)
        meta_type, meta_bytes = self.serde.dumps_typed(metadata)
        with self._lock:
            self._conn().execute(
                "INSERT OR REPLACE INTO checkpoints "
                "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, "
                " checkpoint_type, checkpoint, metadata_type, metadata) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                    cp_type, cp_bytes, meta_type, meta_bytes,
                ),
            )
            self._conn().commit()

    def _write_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str,
        task_id: str, writes: List[tuple],
    ) -> None:
        with self._lock:
            conn = self._conn()
            for idx, (channel, value) in enumerate(writes):
                v_type, v_bytes = self.serde.dumps_typed(value)
                conn.execute(
                    "INSERT OR REPLACE INTO writes "
                    "(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, "
                    " channel, value_type, value) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
                        channel, v_type, v_bytes,
                    ),
                )
            conn.commit()

    def _list_tuples(
        self, thread_id: str, checkpoint_ns: str = "",
        *, before: Optional[str] = None, limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        with self._lock:
            if before:
                rows = self._conn().execute(
                    "SELECT * FROM checkpoints WHERE thread_id=? AND checkpoint_ns=? "
                    "AND checkpoint_id<? ORDER BY seq DESC",
                    (thread_id, checkpoint_ns, before),
                ).fetchall()
            else:
                rows = self._conn().execute(
                    "SELECT * FROM checkpoints WHERE thread_id=? AND checkpoint_ns=? "
                    "ORDER BY seq DESC",
                    (thread_id, checkpoint_ns),
                ).fetchall()
            if limit is not None:
                rows = rows[:limit]
        for row in rows:
            yield self._row_to_tuple(row)

    # ── BaseCheckpointSaver interface ──────────────────────────────────

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        thread_id, checkpoint_ns, checkpoint_id = self._thread_key(config)
        return self._get_tuple(thread_id, checkpoint_ns, checkpoint_id)

    def put(
        self, config: Dict[str, Any], checkpoint: Dict[str, Any],
        metadata: Dict[str, Any], new_versions: Dict[str, Any],
    ) -> Dict[str, Any]:
        thread_id, checkpoint_ns, _ = self._thread_key(config)
        checkpoint_id = checkpoint["id"]
        parent_id = None
        if config.get("configurable", {}).get("checkpoint_id"):
            parent_id = config["configurable"]["checkpoint_id"]
        self._write(
            thread_id, checkpoint_ns, checkpoint_id, parent_id,
            checkpoint, metadata,
        )
        return self._tuple_config(thread_id, checkpoint_ns, checkpoint_id)

    def put_writes(
        self, config: Dict[str, Any], writes: List[tuple],
        task_id: str, task_path: str = "",
    ) -> None:
        thread_id, checkpoint_ns, checkpoint_id = self._thread_key(config)
        self._write_writes(thread_id, checkpoint_ns, checkpoint_id, task_id, writes)

    def list(
        self, config: Dict[str, Any], *, filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None, limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        thread_id, checkpoint_ns, _ = self._thread_key(config)
        before_id = None
        if before is not None:
            before_id = before.get("configurable", {}).get("checkpoint_id")
        yield from self._list_tuples(thread_id, checkpoint_ns, before=before_id, limit=limit)

    # ── Async variants (LangGraph's async execution path) ──────────────

    async def aget_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        return await asyncio.to_thread(self.get_tuple, config)

    async def aput(
        self, config: Dict[str, Any], checkpoint: Dict[str, Any],
        metadata: Dict[str, Any], new_versions: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self.put, config, checkpoint, metadata, new_versions,
        )

    async def aput_writes(
        self, config: Dict[str, Any], writes: List[tuple],
        task_id: str, task_path: str = "",
    ) -> None:
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def alist(
        self, config: Dict[str, Any], *, filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None, limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """Async generator over the thread's checkpoints (newest first)."""
        tuples = await asyncio.to_thread(
            list, self.list(config, filter=filter, before=before, limit=limit),
        )
        for tup in tuples:
            yield tup

    # ── Helpers for the server ─────────────────────────────────────────

    def read_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Latest checkpoint values of a thread, or None when absent.

        A pure-read convenience for surfaces that must not hold a compiled
        graph (e.g. ``GET /v1/run/{id}`` after the runner is gone).
        ``paused`` is True when the thread is genuinely interrupted — the
        approval gate stores an ``__interrupt__`` pending write (and the
        paused superstep also snapshots a ``branch:to:<node>`` channel);
        ``failed`` is True when the thread died mid-run (``__error__``
        pending write). A plain ``branch:to:`` channel alone does NOT
        mean paused: every conditional-edge superstep leaves one, and
        failed runs carry one too.
        """
        tup = self._get_tuple(thread_id)
        if tup is None:
            return None
        values = tup.checkpoint.get("channel_values") or {}
        write_channels = {w[1] for w in (tup.pending_writes or [])}
        paused = bool(values.get("__interrupt__")) or "__interrupt__" in write_channels
        failed = "__error__" in write_channels
        return {
            "values": dict(values),
            "paused": paused,
            "failed": failed,
        }


__all__ = ["OrchaSqliteCheckpointer", "build_sqlite_checkpointer"]