"""Shared resources: what apps own, who may use it, and how it is referenced.

The runtime does NOT become a second application and does not copy content. A *resource* is a small record
(`owner app`, `type`, `title`, `workspace`, metadata) that POINTS at content the owning app keeps in the
runtime's context index (`kind=context`: a collection of the owner's namespace), or at a file
(`kind=file`, read-only) or a short text (`kind=text`). Sharing is a grant on that record:

    private (default)  -> only the owner app (and the user via the CLI)
    shared             -> named apps, `read` or `write`
    global             -> every app, read-only; ONLY the user (owner token) can make a resource global

Apps never see each other's databases: a grantee reads/searches/writes *through the runtime*, and the same
underlying collection serves everyone (a reference, not a copy). Content is touched only when someone asks
(on demand); nothing is indexed or loaded on behalf of an app that never reads it. Every grant, revocation,
read, search and write is recorded in a bounded audit log.
"""
from __future__ import annotations

import asyncio
import json
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from ..security.secrets import APP_ID_RE, Principal
from .bm25 import chunk_text, rank_chunks
from .context import ContextIndex
from .errors import RuntimeApiError

KINDS = ("context", "text", "file")
ACCESS_RANK = {"none": 0, "read": 1, "write": 2, "admin": 3}
MAX_FILE_BYTES = 5_000_000
MAX_TEXT_CHARS = 200_000
TEXT_SUFFIXES = {".txt", ".md", ".mdx", ".json", ".csv", ".tsv", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp",
                 ".h", ".cs", ".go", ".rs", ".rb", ".php", ".html", ".css", ".xml", ".yml", ".yaml", ".toml", ".ini",
                 ".sql", ".sh", ".ps1", ".log", ".tex", ".rst"}
REF_RE = re.compile(r"^runtime://(res_[0-9a-f]{12})(?:/(.+))?$")
ID_RE = re.compile(r"^res_[0-9a-f]{12}$")
AUDIT_KEEP = 2000


def _now() -> float:
    return time.time()


class ResourceService:
    """Resource registry + grants + requests + workspaces + audit. Opens its database on first use (lazy)."""

    def __init__(self, db_path: Path, context: ContextIndex):
        self.path, self.context = db_path, context
        self._db: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    @property
    def opened(self) -> bool:
        return self._db is not None

    def _conn(self) -> sqlite3.Connection:
        if self._db is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self.path), check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS resources(
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, kind TEXT NOT NULL, type TEXT NOT NULL, title TEXT NOT NULL,
                    workspace TEXT, metadata TEXT, ref TEXT, created_at REAL, updated_at REAL);
                CREATE TABLE IF NOT EXISTS grants(
                    resource_id TEXT NOT NULL, grantee TEXT NOT NULL, access TEXT NOT NULL, granted_by TEXT, granted_at REAL,
                    PRIMARY KEY(resource_id, grantee));
                CREATE TABLE IF NOT EXISTS requests(
                    id TEXT PRIMARY KEY, resource_id TEXT NOT NULL, requester TEXT NOT NULL, access TEXT NOT NULL,
                    reason TEXT, state TEXT NOT NULL, created_at REAL, decided_at REAL, decided_by TEXT);
                CREATE TABLE IF NOT EXISTS workspaces(
                    id TEXT PRIMARY KEY, name TEXT UNIQUE NOT NULL, created_by TEXT, created_at REAL);
                CREATE TABLE IF NOT EXISTS audit(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, actor TEXT, action TEXT, resource_id TEXT, detail TEXT);
                CREATE INDEX IF NOT EXISTS res_owner ON resources(owner);
            """)
        return self._db

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def actor(pr: Principal) -> str:
        return pr.app_id if pr.kind == "app" and pr.app_id else "user"

    def _audit(self, pr_or_actor: Principal | str, action: str, rid: str | None, detail: str = "") -> None:
        actor = pr_or_actor if isinstance(pr_or_actor, str) else self.actor(pr_or_actor)
        db = self._conn()
        db.execute("INSERT INTO audit(ts,actor,action,resource_id,detail) VALUES(?,?,?,?,?)", (_now(), actor, action, rid, detail[:300]))
        db.execute("DELETE FROM audit WHERE id <= (SELECT MAX(id) FROM audit) - ?", (AUDIT_KEEP,))
        db.commit()

    def _row(self, rid: str) -> sqlite3.Row | None:
        return self._conn().execute("SELECT * FROM resources WHERE id=?", (rid,)).fetchone()

    def _grants(self, rid: str) -> dict[str, str]:
        return {r["grantee"]: r["access"] for r in self._conn().execute("SELECT grantee, access FROM grants WHERE resource_id=?", (rid,))}

    def access_of(self, pr: Principal, row: sqlite3.Row) -> str:
        if pr.kind == "owner":
            return "admin"
        if row["owner"] == pr.app_id:
            return "admin"
        grants = self._grants(row["id"])
        best = ACCESS_RANK[grants.get(pr.app_id or "", "none")]
        best = max(best, ACCESS_RANK[grants.get("*", "none")])
        return {v: k for k, v in ACCESS_RANK.items()}[best]

    @staticmethod
    def _visibility(grants: dict[str, str]) -> str:
        return "global" if "*" in grants else "shared" if grants else "private"

    def _public(self, pr: Principal, row: sqlite3.Row, access: str) -> dict[str, Any]:
        grants = self._grants(row["id"])
        ref = json.loads(row["ref"] or "{}")
        out = {"id": row["id"], "ref": f"runtime://{row['id']}", "owner": row["owner"], "type": row["type"], "kind": row["kind"],
               "title": row["title"], "workspace": row["workspace"], "metadata": json.loads(row["metadata"] or "{}"),
               "visibility": self._visibility(grants), "access": access,
               "created_at": row["created_at"], "updated_at": row["updated_at"]}
        if access == "admin":                      # only the owner sees where content lives and who else has access
            out["content"] = {k: (v if k != "text" else f"<{len(v)} chars>") for k, v in ref.items()}
            out["grants"] = [{"app": g, "access": a} for g, a in sorted(grants.items())]
        return out

    def _require(self, pr: Principal, rid: str, minimum: str = "read") -> tuple[sqlite3.Row, str]:
        if not ID_RE.match(rid or ""):
            raise RuntimeApiError("resource_not_found", f"Resource '{rid}' was not found.", 404)
        row = self._row(rid)
        access = self.access_of(pr, row) if row else "none"
        if row is None or access == "none":
            raise RuntimeApiError("resource_not_found", f"Resource '{rid}' was not found.", 404,
                                  hint="It may not exist, or it has not been shared with this app. Ask the owner, or request access.")
        if ACCESS_RANK[access] < ACCESS_RANK[minimum]:
            raise RuntimeApiError("access_denied", f"This app has '{access}' access to {rid}; '{minimum}' is required.", 403,
                                  hint=f"Ask for more access: POST /v1/resources/{rid}/request.")
        return row, access

    # -------------------------------------------------------------- resources
    def create(self, pr: Principal, *, type: str, title: str, kind: str = "context", collection: str | None = None,
               doc_ids: list[str] | None = None, text: str | None = None, path: str | None = None,
               workspace: str | None = None, metadata: dict | None = None, owner: str | None = None) -> dict[str, Any]:
        pr.require("context.write")
        if kind not in KINDS:
            raise RuntimeApiError("invalid_request", f"kind must be one of {', '.join(KINDS)}.", 400)
        if not (title or "").strip() or not (type or "").strip():
            raise RuntimeApiError("invalid_request", "'type' and 'title' are required.", 400)
        owner_app = owner if pr.kind == "owner" and owner else (pr.app_id or "user")
        ref: dict[str, Any]
        if kind == "context":
            if not collection:
                raise RuntimeApiError("invalid_request", "kind=context needs 'collection' (a collection of the owner's context index).", 400)
            ref = {"collection": collection, **({"doc_ids": doc_ids} if doc_ids else {})}
        elif kind == "text":
            if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
                raise RuntimeApiError("invalid_request", f"kind=text needs 'text' (1..{MAX_TEXT_CHARS} chars).", 400)
            ref = {"text": text}
        else:
            p = Path(path or "").expanduser()
            if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES or p.stat().st_size > MAX_FILE_BYTES:
                raise RuntimeApiError("invalid_request", f"kind=file needs an existing text-like file under {MAX_FILE_BYTES // 1_000_000} MB.", 400)
            ref = {"path": str(p.resolve())}
        with self._lock:
            db = self._conn()
            if workspace:
                self._workspace_id(workspace, create=True, actor=self.actor(pr))
            rid = "res_" + secrets.token_hex(6)
            db.execute("INSERT INTO resources(id,owner,kind,type,title,workspace,metadata,ref,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (rid, owner_app, kind, type.strip(), title.strip(), workspace, json.dumps(metadata or {}), json.dumps(ref), _now(), _now()))
            db.commit()
            self._audit(pr, "create", rid, f"{kind}:{type}:{title}")
            return self._public(pr, self._row(rid), "admin")

    def list(self, pr: Principal, *, type: str | None = None, workspace: str | None = None, owned: bool = False) -> list[dict[str, Any]]:
        pr.require("context.read")
        out = []
        with self._lock:
            for row in self._conn().execute("SELECT * FROM resources ORDER BY created_at DESC"):
                if type and row["type"] != type or workspace and row["workspace"] != workspace:
                    continue
                access = self.access_of(pr, row)
                if access == "none" or (owned and row["owner"] != (pr.app_id or "user")):
                    continue
                out.append(self._public(pr, row, access))
        return out

    def get(self, pr: Principal, rid: str) -> dict[str, Any]:
        pr.require("context.read")
        with self._lock:
            row, access = self._require(pr, rid)
            return self._public(pr, row, access)

    def resolve(self, pr: Principal, ref: str) -> dict[str, Any]:
        m = REF_RE.match(ref or "")
        if not m:
            raise RuntimeApiError("invalid_reference", "A reference looks like runtime://res_<12 hex>[/<document>].", 400)
        res = self.get(pr, m.group(1))
        return {**res, "document": m.group(2)}

    def update(self, pr: Principal, rid: str, *, title: str | None = None, metadata: dict | None = None,
               workspace: str | None = None, text: str | None = None) -> dict[str, Any]:
        pr.require("context.write")
        with self._lock:
            row, access = self._require(pr, rid, "write")
            fields, vals = [], []
            if any(v is not None for v in (title, metadata, workspace)) and access != "admin":
                raise RuntimeApiError("access_denied", "Only the owner can change a resource's title, metadata or workspace.", 403)
            if title is not None:
                fields.append("title=?"); vals.append(title.strip())
            if metadata is not None:
                fields.append("metadata=?"); vals.append(json.dumps(metadata))
            if workspace is not None:
                if workspace:
                    self._workspace_id(workspace, create=True, actor=self.actor(pr))
                fields.append("workspace=?"); vals.append(workspace or None)
            if text is not None:
                if row["kind"] != "text" or len(text) > MAX_TEXT_CHARS:
                    raise RuntimeApiError("invalid_request", "Only kind=text resources hold inline text (max %d chars)." % MAX_TEXT_CHARS, 400)
                fields.append("ref=?"); vals.append(json.dumps({"text": text}))
            if fields:
                self._conn().execute(f"UPDATE resources SET {', '.join(fields)}, updated_at=? WHERE id=?", (*vals, _now(), rid))
                self._conn().commit()
                self._audit(pr, "update", rid, ",".join(f.split("=")[0] for f in fields))
            return self._public(pr, self._row(rid), access)

    async def delete(self, pr: Principal, rid: str, purge: bool = False) -> dict[str, Any]:
        pr.require("context.write")
        with self._lock:
            row, access = self._require(pr, rid, "admin")
            ref = json.loads(row["ref"] or "{}")
            db = self._conn()
            db.execute("DELETE FROM grants WHERE resource_id=?", (rid,))
            db.execute("DELETE FROM requests WHERE resource_id=?", (rid,))
            db.execute("DELETE FROM resources WHERE id=?", (rid,))
            db.commit()
            self._audit(pr, "delete", rid, f"purge={purge}")
        if purge and row["kind"] == "context":
            await self.context.delete(row["owner"], ref["collection"], None)
        return {"id": rid, "deleted": True, "purged": bool(purge and row["kind"] == "context")}

    # ---------------------------------------------------------------- sharing
    def share(self, pr: Principal, rid: str, with_apps: list[str], access: str = "read", *, _by: str | None = None) -> dict[str, Any]:
        pr.require("context.share")
        if access not in ("read", "write"):
            raise RuntimeApiError("invalid_request", "access must be 'read' or 'write'.", 400)
        with self._lock:
            row, _ = self._require(pr, rid, "admin")
            for app in with_apps:
                if app == "*":
                    if pr.kind != "owner":
                        raise RuntimeApiError("user_approval_required", "Only the user can make a resource global.", 403,
                                              hint=f"Run: anvira context share {rid} --global")
                    if access != "read":
                        raise RuntimeApiError("invalid_request", "Global access is always read-only.", 400)
                elif not APP_ID_RE.match(app):
                    raise RuntimeApiError("invalid_request", f"'{app}' is not a valid app id.", 400)
                elif app == row["owner"]:
                    raise RuntimeApiError("invalid_request", "The owner already has full access.", 400)
            db = self._conn()
            for app in with_apps:
                db.execute("INSERT INTO grants(resource_id,grantee,access,granted_by,granted_at) VALUES(?,?,?,?,?) "
                           "ON CONFLICT(resource_id,grantee) DO UPDATE SET access=excluded.access, granted_by=excluded.granted_by, granted_at=excluded.granted_at",
                           (rid, app, access, _by or self.actor(pr), _now()))
                self._audit(_by or self.actor(pr), "share", rid, f"{app}:{access}")
            db.commit()
            return self._public(pr, self._row(rid), "admin")

    def revoke(self, pr: Principal, rid: str, from_apps: list[str] | None = None) -> dict[str, Any]:
        pr.require("context.share")
        with self._lock:
            row, _ = self._require(pr, rid, "admin")
            db = self._conn()
            targets = from_apps or list(self._grants(rid))
            for app in targets:                  # narrowing access is always allowed to the owner, including withdrawing "global"
                db.execute("DELETE FROM grants WHERE resource_id=? AND grantee=?", (rid, app))
                self._audit(pr, "revoke", rid, app)
            db.commit()
            return self._public(pr, self._row(rid), "admin")

    def permissions(self, pr: Principal, rid: str) -> dict[str, Any]:
        pr.require("context.read")
        with self._lock:
            row, access = self._require(pr, rid)
            grants = self._grants(rid)
            out = {"id": rid, "owner": row["owner"], "your_access": access, "visibility": self._visibility(grants)}
            if access == "admin":
                out["grants"] = [{"app": g, "access": a} for g, a in sorted(grants.items())]
            return out

    # --------------------------------------------------------------- requests
    def request_access(self, pr: Principal, rid: str, access: str = "read", reason: str = "") -> dict[str, Any]:
        pr.require("context.read")
        if pr.kind != "app":
            raise RuntimeApiError("invalid_request", "Only applications request access.", 400)
        if access not in ("read", "write"):
            raise RuntimeApiError("invalid_request", "access must be 'read' or 'write'.", 400)
        with self._lock:
            row = self._row(rid) if ID_RE.match(rid or "") else None
            if row is None:
                raise RuntimeApiError("resource_not_found", f"Resource '{rid}' was not found.", 404)
            if row["owner"] == pr.app_id or ACCESS_RANK[self.access_of(pr, row)] >= ACCESS_RANK[access]:
                raise RuntimeApiError("already_allowed", "This app already has that access.", 409)
            db = self._conn()
            dup = db.execute("SELECT id FROM requests WHERE resource_id=? AND requester=? AND state='pending'", (rid, pr.app_id)).fetchone()
            if dup:
                raise RuntimeApiError("request_pending", "A request for this resource is already pending.", 409, details={"request": dup["id"]})
            qid = "req_" + secrets.token_hex(5)
            db.execute("INSERT INTO requests(id,resource_id,requester,access,reason,state,created_at) VALUES(?,?,?,?,?,'pending',?)",
                       (qid, rid, pr.app_id, access, reason[:300], _now()))
            db.commit()
            self._audit(pr, "request", rid, f"{access}: {reason[:100]}")
            return {"id": qid, "resource": rid, "requester": pr.app_id, "access": access, "state": "pending"}

    def list_requests(self, pr: Principal, state: str | None = "pending") -> list[dict[str, Any]]:
        pr.require("context.read")
        out = []
        with self._lock:
            for r in self._conn().execute("SELECT q.*, r.owner, r.title FROM requests q JOIN resources r ON r.id=q.resource_id ORDER BY q.created_at DESC"):
                if state and r["state"] != state:
                    continue
                if pr.kind == "owner" or r["owner"] == pr.app_id or r["requester"] == pr.app_id:
                    out.append({"id": r["id"], "resource": r["resource_id"], "title": r["title"] if pr.kind == "owner" or r["owner"] == pr.app_id else None,
                                "owner": r["owner"], "requester": r["requester"], "access": r["access"], "reason": r["reason"],
                                "state": r["state"], "created_at": r["created_at"]})
        return out

    def decide(self, pr: Principal, qid: str, approve: bool) -> dict[str, Any]:
        pr.require("context.share")
        with self._lock:
            db = self._conn()
            q = db.execute("SELECT q.*, r.owner FROM requests q JOIN resources r ON r.id=q.resource_id WHERE q.id=?", (qid,)).fetchone()
            if q is None or (pr.kind != "owner" and q["owner"] != pr.app_id):
                raise RuntimeApiError("request_not_found", f"Request '{qid}' was not found.", 404)
            if q["state"] != "pending":
                raise RuntimeApiError("request_decided", f"Request already {q['state']}.", 409)
            if approve:
                self.share(pr, q["resource_id"], [q["requester"]], q["access"])
            db.execute("UPDATE requests SET state=?, decided_at=?, decided_by=? WHERE id=?",
                       ("approved" if approve else "denied", _now(), self.actor(pr), qid))
            db.commit()
            self._audit(pr, "approve" if approve else "deny", q["resource_id"], q["requester"])
            return {"id": qid, "state": "approved" if approve else "denied", "resource": q["resource_id"], "requester": q["requester"]}

    # -------------------------------------------------------------- workspaces
    def _workspace_id(self, name: str, create: bool, actor: str) -> str:
        db = self._conn()
        row = db.execute("SELECT id FROM workspaces WHERE name=?", (name,)).fetchone()
        if row:
            return row["id"]
        if not create:
            raise RuntimeApiError("workspace_not_found", f"Workspace '{name}' was not found.", 404)
        wid = "ws_" + secrets.token_hex(4)
        db.execute("INSERT INTO workspaces(id,name,created_by,created_at) VALUES(?,?,?,?)", (wid, name, actor, _now()))
        db.commit()
        return wid

    def workspaces(self, pr: Principal) -> list[dict[str, Any]]:
        pr.require("context.read")
        with self._lock:
            out = []
            for w in self._conn().execute("SELECT * FROM workspaces ORDER BY name"):
                mine = [r for r in self.list(pr, workspace=w["name"])]
                if pr.kind == "owner" or mine:
                    out.append({"id": w["id"], "name": w["name"], "created_by": w["created_by"], "resources": len(mine)})
            return out

    def create_workspace(self, pr: Principal, name: str) -> dict[str, Any]:
        pr.require("context.write")
        if not re.match(r"^[\w .-]{1,60}$", name or ""):
            raise RuntimeApiError("invalid_request", "Workspace names are 1-60 characters (letters, digits, space, . _ -).", 400)
        with self._lock:
            wid = self._workspace_id(name, True, self.actor(pr))
            self._audit(pr, "workspace", None, name)
            return {"id": wid, "name": name}

    def share_workspace(self, pr: Principal, name: str, with_apps: list[str], access: str = "read") -> dict[str, Any]:
        """Share every resource the caller OWNS in a workspace (a convenience over per-resource grants)."""
        pr.require("context.share")
        with self._lock:
            mine = [r for r in self.list(pr, workspace=name) if r["access"] == "admin"]
            for r in mine:
                self.share(pr, r["id"], with_apps, access)
            return {"workspace": name, "shared": [r["id"] for r in mine], "with": with_apps, "access": access}

    # ------------------------------------------------------------------ content
    async def _text_of(self, row: sqlite3.Row) -> list[dict[str, Any]]:
        """Chunks of a resource (on demand; nothing is cached or copied)."""
        ref = json.loads(row["ref"] or "{}")
        if row["kind"] == "context":
            chunks = await self.context.chunks(row["owner"], ref["collection"], ref.get("doc_ids"))
            return [{**c, "resource": row["id"], "title": c.get("title") or row["title"]} for c in chunks]
        if row["kind"] == "text":
            text = ref["text"]
        else:
            try:
                text = await asyncio.to_thread(Path(ref["path"]).read_text, "utf-8", "replace")
            except OSError as exc:
                raise RuntimeApiError("resource_unavailable", f"The file behind this resource is not readable: {exc}", 410) from exc
        return [{**c, "resource": row["id"], "doc_id": "content", "title": row["title"]} for c in chunk_text(row["title"], text)[:2000]]

    async def read(self, pr: Principal, rid: str, doc_id: str | None = None, max_chars: int = 60_000) -> dict[str, Any]:
        pr.require("context.read")
        with self._lock:
            row, access = self._require(pr, rid)
            pub = self._public(pr, row, access)
            ref = json.loads(row["ref"] or "{}")
        if row["kind"] == "context" and not doc_id:
            docs = await self.context.documents(row["owner"], ref["collection"])
            if ref.get("doc_ids"):
                docs = [d for d in docs if d["doc_id"] in ref["doc_ids"]]
            self._audit(pr, "read", rid, "listing")
            return {"resource": pub, "documents": docs}
        chunks = [c for c in await self._text_of(row) if not doc_id or c["doc_id"] == doc_id]
        if not chunks:
            raise RuntimeApiError("document_not_found", f"No document '{doc_id}' in this resource.", 404)
        text = "\n\n".join(c["text"] for c in sorted(chunks, key=lambda c: (c["doc_id"], c.get("index", 0))))[:max_chars]
        self._audit(pr, "read", rid, doc_id or "content")
        return {"resource": pub, "doc_id": doc_id, "text": text, "truncated": len(text) >= max_chars}

    async def write_document(self, pr: Principal, rid: str, doc_id: str, text: str, title: str | None = None) -> dict[str, Any]:
        pr.require("context.write")
        with self._lock:
            row, access = self._require(pr, rid, "write")
            if row["kind"] != "context":
                raise RuntimeApiError("invalid_request", "Documents can only be written to kind=context resources.", 400)
            ref = json.loads(row["ref"] or "{}")
        res = await self.context.put(row["owner"], ref["collection"], doc_id, text, title)   # same collection: a reference, not a copy
        if ref.get("doc_ids") and doc_id not in ref["doc_ids"]:
            with self._lock:
                ref["doc_ids"].append(doc_id)
                self._conn().execute("UPDATE resources SET ref=?, updated_at=? WHERE id=?", (json.dumps(ref), _now(), rid))
                self._conn().commit()
        self._audit(pr, "write", rid, doc_id)
        return {**res, "resource": rid, "by": self.actor(pr)}

    async def search(self, pr: Principal, query: str, *, resources: list[str] | None = None, workspace: str | None = None,
                     types: list[str] | None = None, limit: int = 8) -> dict[str, Any]:
        """Search everything the caller is authorised for (own + shared + global), ranked once over the pooled chunks."""
        pr.require("context.read")
        limit = max(1, min(50, int(limit)))
        with self._lock:
            rows = [r for r in self._conn().execute("SELECT * FROM resources") if self.access_of(pr, r) != "none"]
        rows = [r for r in rows if (not resources or r["id"] in resources) and (not workspace or r["workspace"] == workspace)
                and (not types or r["type"] in types)]
        pool: list[dict[str, Any]] = []
        skipped: list[str] = []
        for r in rows:
            try:
                pool += await self._text_of(r)
            except RuntimeApiError as exc:
                if exc.code != "resource_unavailable":      # a vanished file must not break a search over everything else
                    raise
                skipped.append(r["id"])
            if len(pool) > 50_000:
                break
        ranked = [c for c in rank_chunks(pool, query) if c["score"] > 0][:limit]
        titles = {r["id"]: (r["title"], r["owner"], r["type"]) for r in rows}
        self._audit(pr, "search", None, f"{query[:80]} ({len(rows)} resources)")
        return {"query": query, "searched": len(rows) - len(skipped), "unavailable": skipped, "count": len(ranked), "items": [
            {"resource": c["resource"], "resource_title": titles[c["resource"]][0], "owner": titles[c["resource"]][1],
             "type": titles[c["resource"]][2], "doc_id": c["doc_id"], "text": c["text"], "score": round(c["score"], 4)} for c in ranked]}

    async def context_for(self, pr: Principal, *, query: str, resources: list[str] | None, limit: int = 5) -> list[dict[str, Any]]:
        """Authorised snippets to inject into a prompt (used by chat/ORCHA `context` option)."""
        return (await self.search(pr, query, resources=resources, limit=limit))["items"]

    # ------------------------------------------------------------------ audit / stats
    def audit(self, pr: Principal, limit: int = 50, resource: str | None = None) -> list[dict[str, Any]]:
        pr.require("context.read")
        with self._lock:
            out = []
            for a in self._conn().execute("SELECT * FROM audit ORDER BY id DESC LIMIT 1000"):
                if resource and a["resource_id"] != resource:
                    continue
                if pr.kind != "owner":
                    row = self._row(a["resource_id"]) if a["resource_id"] else None
                    if not (row and row["owner"] == pr.app_id) and a["actor"] != pr.app_id:
                        continue
                out.append({"ts": a["ts"], "actor": a["actor"], "action": a["action"], "resource": a["resource_id"], "detail": a["detail"]})
                if len(out) >= limit:
                    break
            return out

    def stats(self) -> dict[str, Any]:
        if not self.opened and not self.path.exists():
            return {"opened": False, "resources": 0, "grants": 0}
        with self._lock:
            db = self._conn()
            return {"opened": True, "resources": db.execute("SELECT COUNT(*) FROM resources").fetchone()[0],
                    "grants": db.execute("SELECT COUNT(*) FROM grants").fetchone()[0],
                    "pending_requests": db.execute("SELECT COUNT(*) FROM requests WHERE state='pending'").fetchone()[0],
                    "workspaces": db.execute("SELECT COUNT(*) FROM workspaces").fetchone()[0]}
