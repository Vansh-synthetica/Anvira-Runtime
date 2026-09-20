"""Shared document index: index text, retrieve the most relevant chunks.

Generalises what Anvira's Notes/Study/chat each did privately
(``utils/notesContext.ts`` + ``bm25.ts``): chunk source text, rank with BM25,
hand back only the relevant pieces. The runtime knows nothing about
notebooks, flashcards or projects — an application indexes *documents* into
its own *collections* and searches them. Each app has a private namespace;
the data lives in one small SQLite file in the runtime data directory
(stdlib only, no vector store or embedding model required).
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .bm25 import chunk_text, rank_chunks
from .errors import RuntimeApiError

MAX_DOC_CHARS = 5_000_000
MAX_SCAN_CHUNKS = 50_000


class ContextIndex:
    def __init__(self, db_path: Path):
        self._path = db_path
        self._db: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    @property
    def opened(self) -> bool:
        return self._db is not None

    def _conn(self) -> sqlite3.Connection:
        """Open the database on first use (an app that never indexes anything costs nothing). Call with the lock held."""
        if self._db is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self._path), check_same_thread=False)
            self._db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS docs(
                    app TEXT NOT NULL, collection TEXT NOT NULL, doc_id TEXT NOT NULL,
                    title TEXT, metadata TEXT, updated_at REAL, chars INTEGER,
                    PRIMARY KEY(app, collection, doc_id));
                CREATE TABLE IF NOT EXISTS chunks(
                    app TEXT NOT NULL, collection TEXT NOT NULL, doc_id TEXT NOT NULL,
                    idx INTEGER NOT NULL, tag TEXT, text TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS chunks_lookup ON chunks(app, collection, doc_id);
            """)
        return self._db

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    # -- sync core (run in a worker thread) ----------------------------------------------
    def _put(self, app: str, collection: str, doc_id: str, text: str, title: str | None,
             metadata: dict[str, Any] | None) -> dict[str, Any]:
        if len(text) > MAX_DOC_CHARS:
            raise RuntimeApiError("document_too_large", f"Documents are limited to {MAX_DOC_CHARS} characters.", 413)
        chunks = chunk_text(title or doc_id, text)
        with self._lock:
            self._conn().execute("DELETE FROM chunks WHERE app=? AND collection=? AND doc_id=?", (app, collection, doc_id))
            self._conn().executemany(
                "INSERT INTO chunks(app,collection,doc_id,idx,tag,text) VALUES(?,?,?,?,?,?)",
                [(app, collection, doc_id, i, c["tag"], c["text"]) for i, c in enumerate(chunks)])
            self._conn().execute(
                "INSERT INTO docs(app,collection,doc_id,title,metadata,updated_at,chars) VALUES(?,?,?,?,?,?,?) "
                "ON CONFLICT(app,collection,doc_id) DO UPDATE SET title=excluded.title, metadata=excluded.metadata, "
                "updated_at=excluded.updated_at, chars=excluded.chars",
                (app, collection, doc_id, title, json.dumps(metadata or {}), time.time(), len(text)))
            self._conn().commit()
        return {"collection": collection, "doc_id": doc_id, "chunks": len(chunks), "chars": len(text)}

    def _delete(self, app: str, collection: str, doc_id: str | None) -> int:
        with self._lock:
            if doc_id is None:
                n = self._conn().execute("DELETE FROM docs WHERE app=? AND collection=?", (app, collection)).rowcount
                self._conn().execute("DELETE FROM chunks WHERE app=? AND collection=?", (app, collection))
            else:
                n = self._conn().execute("DELETE FROM docs WHERE app=? AND collection=? AND doc_id=?",
                                     (app, collection, doc_id)).rowcount
                self._conn().execute("DELETE FROM chunks WHERE app=? AND collection=? AND doc_id=?",
                                 (app, collection, doc_id))
            self._conn().commit()
        return n

    def _search(self, app: str, collection: str, query: str, limit: int, doc_ids: list[str] | None) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn().execute(
                "SELECT c.doc_id, c.idx, c.tag, c.text, d.title, d.metadata FROM chunks c "
                "JOIN docs d ON d.app=c.app AND d.collection=c.collection AND d.doc_id=c.doc_id "
                "WHERE c.app=? AND c.collection=? LIMIT ?", (app, collection, MAX_SCAN_CHUNKS)).fetchall()
        pool = [{"id": f"{r[0]}#{r[1]}", "doc_id": r[0], "index": r[1], "tag": r[2], "text": r[3], "title": r[4],
                 "metadata": json.loads(r[5] or "{}")} for r in rows if not doc_ids or r[0] in doc_ids]
        ranked = rank_chunks(pool, query)
        return [{**c, "score": round(c["score"], 4)} for c in ranked[:limit] if c["score"] > 0]

    def _chunks(self, app: str, collection: str, doc_ids: list[str] | None) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn().execute(
                "SELECT c.doc_id, c.idx, c.tag, c.text, d.title FROM chunks c JOIN docs d ON d.app=c.app AND "
                "d.collection=c.collection AND d.doc_id=c.doc_id WHERE c.app=? AND c.collection=? LIMIT ?",
                (app, collection, MAX_SCAN_CHUNKS)).fetchall()
        return [{"id": f"{r[0]}#{r[1]}", "doc_id": r[0], "index": r[1], "tag": r[2], "text": r[3], "title": r[4]}
                for r in rows if not doc_ids or r[0] in doc_ids]

    def _collections(self, app: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn().execute("SELECT collection, COUNT(*), COALESCE(SUM(chars),0), MAX(updated_at) "
                                    "FROM docs WHERE app=? GROUP BY collection ORDER BY collection", (app,)).fetchall()
        return [{"collection": r[0], "documents": r[1], "chars": r[2], "updated_at": r[3]} for r in rows]

    def _documents(self, app: str, collection: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn().execute("SELECT doc_id,title,metadata,updated_at,chars FROM docs WHERE app=? AND collection=? "
                                    "ORDER BY updated_at DESC", (app, collection)).fetchall()
        return [{"doc_id": r[0], "title": r[1], "metadata": json.loads(r[2] or "{}"), "updated_at": r[3], "chars": r[4]}
                for r in rows]

    # -- async facade ---------------------------------------------------------------------
    async def put(self, app: str, collection: str, doc_id: str, text: str, title: str | None = None,
                  metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self._put, app, collection, doc_id, text, title, metadata)

    async def delete(self, app: str, collection: str, doc_id: str | None = None) -> dict[str, Any]:
        n = await asyncio.to_thread(self._delete, app, collection, doc_id)
        return {"collection": collection, "doc_id": doc_id, "deleted": n}

    async def search(self, app: str, collection: str, query: str, limit: int = 8,
                     doc_ids: list[str] | None = None) -> dict[str, Any]:
        hits = await asyncio.to_thread(self._search, app, collection, query, max(1, min(50, limit)), doc_ids)
        return {"collection": collection, "query": query, "count": len(hits), "items": hits}

    async def chunks(self, app: str, collection: str, doc_ids: list[str] | None = None) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._chunks, app, collection, doc_ids)

    async def collections(self, app: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._collections, app)

    async def documents(self, app: str, collection: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._documents, app, collection)
