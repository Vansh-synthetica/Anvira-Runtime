"""
orcha.graph.store
=================
Durable checkpoint storage for graph runs.

A checkpoint is a snapshot of the packet at a node boundary: (run_id,
node_id, packet_json, ts). Checkpoints are written after every successful
node transition (subject to a throttle policy) so that a crashed run can
be *resumed* from the latest checkpoint and skipped nodes are not re-run,
and a completed run can be *replayed* deterministically.

The Store is a Protocol so deployments can swap backends. Two concrete
implementations ship:

- FileStore  (default): one JSON file per checkpoint under a runs dir.
              Zero dependencies, human-readable, good for single-host dev.
- SqliteStore (example): a single SQLite database file. Better for many
              concurrent runs / long histories; still single-file & local.

The packet's existing ``to_json`` / ``from_json`` round-trip is the
serialization format — no second schema to maintain.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, runtime_checkable

from ..core.packets import OrchaPacket


# ── Checkpoint record ─────────────────────────────────────────────────────────

@dataclass
class Checkpoint:
    """
    One persisted snapshot of a run at a node boundary.

    Attributes
    ----------
    id         Unique checkpoint id (uuid4 hex).
    run_id     The trace id of the run this checkpoint belongs to.
    node_id    The node that just completed when this was written.
    next_node  The node the run was about to execute (so resume knows where
               to pick up). May be END for a completed run.
    packet     The reconstructed OrchaPacket at this point.
    packet_json  The raw serialized packet (kept for stores that store JSON).
    ts         Epoch seconds the checkpoint was written.
    seq        Monotonic sequence number within the run (0-based).
    """
    id: str
    run_id: str
    node_id: str
    next_node: str
    packet: OrchaPacket
    packet_json: str
    ts: float = field(default_factory=time.time)
    seq: int = 0

    def to_storage_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "node_id": self.node_id,
            "next_node": self.next_node,
            "packet_json": self.packet_json,
            "ts": self.ts,
            "seq": self.seq,
        }

    @classmethod
    def from_storage_dict(cls, d: Dict[str, object]) -> "Checkpoint":
        return cls(
            id=str(d["id"]),
            run_id=str(d["run_id"]),
            node_id=str(d["node_id"]),
            next_node=str(d["next_node"]),
            packet=OrchaPacket.from_json(str(d["packet_json"])),
            packet_json=str(d["packet_json"]),
            ts=float(d.get("ts", 0.0)),
            seq=int(d.get("seq", 0)),
        )


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class Store(Protocol):
    """The durability interface a graph runtime talks to."""

    async def save_checkpoint(
        self, run_id: str, node_id: str, next_node: str, packet: OrchaPacket,
    ) -> Checkpoint:
        """Persist a checkpoint for the given run."""
        ...

    async def load_checkpoint(self, run_id: str) -> Optional[Checkpoint]:
        """Return the latest checkpoint for a run, or None if none exist."""
        ...

    async def list_checkpoints(self, run_id: str) -> List[Checkpoint]:
        """All checkpoints for a run, oldest first."""
        ...

    async def delete_run(self, run_id: str) -> int:
        """Delete all checkpoints for a run. Returns count removed."""
        ...

    async def list_runs(self) -> List[str]:
        """All known run ids."""
        ...


# ── In-memory store (tests / ephemeral runs) ──────────────────────────────────

class MemoryStore:
    """Process-lifetime store. Useful for tests and short-lived runs."""

    def __init__(self) -> None:
        self._runs: Dict[str, List[Checkpoint]] = {}
        self._lock = asyncio.Lock()

    async def save_checkpoint(
        self, run_id: str, node_id: str, next_node: str, packet: OrchaPacket,
    ) -> Checkpoint:
        async with self._lock:
            seq = len(self._runs.get(run_id, []))
            cp = Checkpoint(
                id=uuid.uuid4().hex, run_id=run_id, node_id=node_id,
                next_node=next_node, packet=packet,
                packet_json=packet.to_json(), seq=seq,
            )
            self._runs.setdefault(run_id, []).append(cp)
            return cp

    async def load_checkpoint(self, run_id: str) -> Optional[Checkpoint]:
        cps = self._runs.get(run_id, [])
        return cps[-1] if cps else None

    async def list_checkpoints(self, run_id: str) -> List[Checkpoint]:
        return list(self._runs.get(run_id, []))

    async def delete_run(self, run_id: str) -> int:
        async with self._lock:
            removed = len(self._runs.pop(run_id, []))
            return removed

    async def list_runs(self) -> List[str]:
        return sorted(self._runs.keys())


# ── FileStore (default) ───────────────────────────────────────────────────────

class FileStore:
    """
    Checkpoint store backed by a directory of JSON files.

    Layout::

        <root>/<run_id>/<seq>__<node_id>.json

    One file per checkpoint. ``index.json`` per run records ordering.
    Concurrent writes to the same run are serialized by a per-run lock.
    """

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        max_runs: Optional[int] = None,
    ) -> None:
        if root is None:
            root = Path.home() / ".orcha" / "runs"
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        # When set, oldest run directories (by mtime) are pruned after every
        # checkpoint write once this many runs exist — bounds disk growth.
        self.max_runs = max_runs
        self._locks: Dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _run_lock(self, run_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            if run_id not in self._locks:
                self._locks[run_id] = asyncio.Lock()
            return self._locks[run_id]

    def _run_dir(self, run_id: str) -> Path:
        d = self._root / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def save_checkpoint(
        self, run_id: str, node_id: str, next_node: str, packet: OrchaPacket,
    ) -> Checkpoint:
        lock = await self._run_lock(run_id)
        async with lock:
            d = self._run_dir(run_id)
            existing = sorted(d.glob("*.json"), key=lambda p: p.name)
            seq = len(existing)
            cp = Checkpoint(
                id=uuid.uuid4().hex, run_id=run_id, node_id=node_id,
                next_node=next_node, packet=packet,
                packet_json=packet.to_json(), seq=seq,
            )
            # Atomic write: temp file then rename.
            fname = f"{seq:05d}__{_safe(node_id)}.json"
            tmp = d / (fname + ".tmp")
            tmp.write_text(
                json.dumps(cp.to_storage_dict(), indent=2), encoding="utf-8",
            )
            os.replace(tmp, d / fname)
            await self._prune()
            return cp

    async def _prune(self) -> None:
        """Delete the oldest run directories once ``max_runs`` is exceeded."""
        if self.max_runs is None or not self._root.is_dir():
            return
        runs = [p for p in self._root.iterdir() if p.is_dir()]
        if len(runs) <= self.max_runs:
            return
        # Oldest first (run dirs are written on every checkpoint, so mtime is
        # a reliable recency signal; names are uuids, not chronological).
        runs.sort(key=lambda p: (p.stat().st_mtime, p.name))
        import shutil
        for stale in runs[: len(runs) - self.max_runs]:
            shutil.rmtree(stale, ignore_errors=True)

    async def load_checkpoint(self, run_id: str) -> Optional[Checkpoint]:
        cps = await self.list_checkpoints(run_id)
        return cps[-1] if cps else None

    async def list_checkpoints(self, run_id: str) -> List[Checkpoint]:
        d = self._root / run_id
        if not d.is_dir():
            return []
        files = sorted(d.glob("*.json"))
        out: List[Checkpoint] = []
        for f in files:
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
                out.append(Checkpoint.from_storage_dict(raw))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        return out

    async def delete_run(self, run_id: str) -> int:
        lock = await self._run_lock(run_id)
        async with lock:
            d = self._root / run_id
            if not d.is_dir():
                return 0
            files = list(d.glob("*.json"))
            for f in files:
                try:
                    f.unlink()
                except OSError:
                    pass
            try:
                d.rmdir()
            except OSError:
                pass
            return len(files)

    async def list_runs(self) -> List[str]:
        if not self._root.is_dir():
            return []
        return sorted(
            p.name for p in self._root.iterdir() if p.is_dir()
        )


# ── SqliteStore (example) ─────────────────────────────────────────────────────

class SqliteStore:
    """
    Checkpoint store backed by a single SQLite database file.

    Better suited than FileStore when a deployment accumulates many runs
    or wants cheap indexed queries over history. Still local and
    dependency-free (sqlite3 is in the stdlib).

    All DB access is funneled through a thread-local connection guarded by
    a threading.Lock — SQLite serializes writers anyway, and this keeps the
    async surface honest without an extra dependency.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS checkpoints (
        id          TEXT PRIMARY KEY,
        run_id      TEXT NOT NULL,
        node_id     TEXT NOT NULL,
        next_node   TEXT NOT NULL,
        packet_json TEXT NOT NULL,
        ts          REAL NOT NULL,
        seq         INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_run_seq ON checkpoints(run_id, seq);
    CREATE INDEX IF NOT EXISTS idx_run_only ON checkpoints(run_id);
    """

    def __init__(self, path: Union[str, Path] = "orcha_runs.db") -> None:
        self._path = str(path)
        self._lock = threading.Lock()
        self._tls = threading.local()
        # Initialize schema synchronously once.
        self._conn().executescript(self._SCHEMA)
        self._async_lock = asyncio.Lock()

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._tls, "conn", None)
        if c is None:
            c = sqlite3.connect(self._path, check_same_thread=False)
            c.row_factory = sqlite3.Row
            self._tls.conn = c
        return c

    async def save_checkpoint(
        self, run_id: str, node_id: str, next_node: str, packet: OrchaPacket,
    ) -> Checkpoint:
        async with self._async_lock:
            return await asyncio.to_thread(self._save, run_id, node_id, next_node, packet)

    def _save(
        self, run_id: str, node_id: str, next_node: str, packet: OrchaPacket,
    ) -> Checkpoint:
        with self._lock:
            conn = self._conn()
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM checkpoints WHERE run_id=?", (run_id,),
            ).fetchone()
            seq = int(row["n"]) if row else 0
            cp = Checkpoint(
                id=uuid.uuid4().hex, run_id=run_id, node_id=node_id,
                next_node=next_node, packet=packet,
                packet_json=packet.to_json(), seq=seq,
            )
            conn.execute(
                "INSERT INTO checkpoints(id,run_id,node_id,next_node,packet_json,ts,seq) "
                "VALUES (?,?,?,?,?,?,?)",
                (cp.id, cp.run_id, cp.node_id, cp.next_node, cp.packet_json, cp.ts, cp.seq),
            )
            conn.commit()
            return cp

    async def load_checkpoint(self, run_id: str) -> Optional[Checkpoint]:
        cps = await self.list_checkpoints(run_id)
        return cps[-1] if cps else None

    async def list_checkpoints(self, run_id: str) -> List[Checkpoint]:
        async with self._async_lock:
            return await asyncio.to_thread(self._list, run_id)

    def _list(self, run_id: str) -> List[Checkpoint]:
        with self._lock:
            rows = self._conn().execute(
                "SELECT * FROM checkpoints WHERE run_id=? ORDER BY seq ASC",
                (run_id,),
            ).fetchall()
        return [Checkpoint.from_storage_dict(dict(r)) for r in rows]

    async def delete_run(self, run_id: str) -> int:
        async with self._async_lock:
            return await asyncio.to_thread(self._delete, run_id)

    def _delete(self, run_id: str) -> int:
        with self._lock:
            cur = self._conn().execute(
                "DELETE FROM checkpoints WHERE run_id=?", (run_id,),
            )
            self._conn().commit()
            return cur.rowcount or 0

    async def list_runs(self) -> List[str]:
        async with self._async_lock:
            return await asyncio.to_thread(self._list_runs)

    def _list_runs(self) -> List[str]:
        with self._lock:
            rows = self._conn().execute(
                "SELECT DISTINCT run_id FROM checkpoints ORDER BY run_id",
            ).fetchall()
        return [r["run_id"] for r in rows]


def _safe(s: str) -> str:
    """Make a string safe for use in a filename component."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:80] or "x"


__all__ = [
    "Checkpoint", "Store",
    "MemoryStore", "FileStore", "SqliteStore",
]
