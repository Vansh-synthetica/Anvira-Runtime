"""Runtime memory API on top of Nomi.

Applications get ``store / search / get / delete`` without knowing Nomi
exists. Nomi has one runtime-owned account; *application boundaries are
enforced here* using reserved tags on every memory:

    app:<app_id>     which application owns it (private to that app)
    scope:shared     visible to apps that hold the ``memory.shared`` permission
    ws:<workspace>   optional workspace boundary inside an app

An app can only read memories tagged with its own ``app:`` tag (or shared
ones if permitted), so Notes cannot read Study's memories by default.
"""
from __future__ import annotations

from typing import Any

from ..security.secrets import Principal
from .bm25 import rank_chunks
from .clients import NomiClient, safe_tag
from .errors import RuntimeApiError

RESERVED = ("app:", "ws:", "scope:")
MAX_SCAN = 2000
PAGE = 100


class MemoryService:
    def __init__(self, nomi: NomiClient, default_scope: str = "app"):
        self.nomi = nomi
        self.default_scope = default_scope

    # -- namespace helpers ---------------------------------------------------------
    @staticmethod
    def _app_id(principal: Principal, as_app: str | None) -> str:
        if principal.kind == "owner":
            return safe_tag(as_app or "owner")
        return principal.app_id or "unknown"

    def _scope_tag(self, principal: Principal, scope: str, as_app: str | None) -> str:
        if scope == "shared":
            principal.require("memory.shared")
            return "scope:shared"
        if scope != "app":
            raise RuntimeApiError("invalid_scope", "scope must be 'app' or 'shared'.", 400)
        return f"app:{self._app_id(principal, as_app)}"

    @staticmethod
    def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
        tags = raw.get("tags") or []
        app = next((t[4:] for t in tags if t.startswith("app:")), None)
        ws = next((t[3:] for t in tags if t.startswith("ws:")), None)
        return {
            "id": raw["id"], "title": raw.get("title"), "content": raw.get("content"),
            "type": raw.get("memory_type"), "importance": raw.get("importance"),
            "tags": [t for t in tags if not t.startswith(RESERVED)],
            "scope": "shared" if "scope:shared" in tags else "app", "app": app, "workspace": ws,
            "extra": {k: v for k, v in (raw.get("extra") or {}).items() if not k.startswith("_")},
            "created_at": raw.get("created_at"), "updated_at": raw.get("updated_at"),
        }

    def _visible(self, principal: Principal, raw: dict[str, Any]) -> bool:
        if principal.kind == "owner":
            return True
        tags = raw.get("tags") or []
        if f"app:{principal.app_id}" in tags:
            return True
        return "scope:shared" in tags and principal.has("memory.shared")

    # -- operations ----------------------------------------------------------------------
    async def store(self, principal: Principal, *, content: str, title: str | None = None,
                    type: str = "note", tags: list[str] | None = None, importance: int = 3,
                    scope: str | None = None, workspace: str | None = None,
                    extra: dict[str, Any] | None = None, app: str | None = None) -> dict[str, Any]:
        principal.require("memory.write")
        if not (content or "").strip():
            raise RuntimeApiError("invalid_request", "content is required.", 400)
        user_tags = [t for t in (tags or [])]
        bad = [t for t in user_tags if t.startswith(RESERVED)]
        if bad:
            raise RuntimeApiError("invalid_tag", f"Tags may not start with {', '.join(RESERVED)} ({bad[0]}).", 400)
        scope = scope or self.default_scope
        scope_tag = self._scope_tag(principal, scope, app)
        app_id = self._app_id(principal, app)
        all_tags = [*user_tags, scope_tag]
        if scope == "shared":
            all_tags.append(f"app:{app_id}")  # remember the author; shared readers still see scope:shared
        if workspace:
            all_tags.append(f"ws:{safe_tag(workspace)}")
        body = {"title": (title or content.strip().splitlines()[0])[:240], "content": content, "memory_type": type,
                "importance": max(1, min(5, int(importance))), "tags": list(dict.fromkeys(all_tags)),
                "source": f"app:{app_id}", "extra": {**(extra or {}), "_app": app_id}}
        created = await self.nomi.request("POST", "/memory", json=body)
        return self._normalize(created)

    async def _scan(self, principal: Principal, scope: str, app: str | None, type: str | None) -> list[dict[str, Any]]:
        tag = self._scope_tag(principal, scope, app)
        items: list[dict[str, Any]] = []
        offset = 0
        while offset < MAX_SCAN:
            params: dict[str, Any] = {"limit": PAGE, "offset": offset, "tag": tag, "sort": "-created_at"}
            if type:
                params["memory_type"] = type
            page = await self.nomi.request("GET", "/memory", params=params)
            items.extend(page.get("items", []))
            offset += PAGE
            if offset >= page.get("total", 0):
                break
        return items

    async def search(self, principal: Principal, *, query: str = "", limit: int = 10, scope: str | None = None,
                     workspace: str | None = None, tags: list[str] | None = None, type: str | None = None,
                     app: str | None = None) -> dict[str, Any]:
        principal.require("memory.read")
        limit = max(1, min(100, int(limit)))
        # Default is the app's own private namespace; shared memory is opt-in (scope="shared" or "all").
        if scope == "all":
            scopes = ["app", "shared"] if principal.has("memory.shared") else ["app"]
        else:
            scopes = [scope or "app"]
        raw: dict[str, dict[str, Any]] = {}
        for sc in scopes:
            for item in await self._scan(principal, sc, app, type):
                raw.setdefault(item["id"], item)
        ws_tag = f"ws:{safe_tag(workspace)}" if workspace else None
        want = set(tags or [])
        pool = [m for m in raw.values()
                if (not ws_tag or ws_tag in (m.get("tags") or [])) and want <= set(m.get("tags") or [])]
        if query.strip():
            docs = [{"id": m["id"], "text": f"{m.get('title', '')}\n{m.get('content', '')}\n{' '.join(m.get('tags') or [])}",
                     "_m": m} for m in pool]
            ranked = [d for d in rank_chunks(docs, query) if d["score"] > 0]
            hits = [{**self._normalize(d["_m"]), "score": round(d["score"], 4)} for d in ranked[:limit]]
        else:
            hits = [{**self._normalize(m), "score": None} for m in pool[:limit]]
        return {"query": query, "count": len(hits), "total_scanned": len(raw), "items": hits}

    async def get(self, principal: Principal, memory_id: str) -> dict[str, Any]:
        principal.require("memory.read")
        raw = await self.nomi.request("GET", f"/memory/{memory_id}")
        if not self._visible(principal, raw):
            raise RuntimeApiError("memory_not_found", f"Memory '{memory_id}' was not found.", 404)
        return self._normalize(raw)

    async def delete(self, principal: Principal, memory_id: str) -> dict[str, Any]:
        principal.require("memory.write")
        raw = await self.nomi.request("GET", f"/memory/{memory_id}")
        if not self._visible(principal, raw):
            raise RuntimeApiError("memory_not_found", f"Memory '{memory_id}' was not found.", 404)
        if principal.kind != "owner" and f"app:{principal.app_id}" not in (raw.get("tags") or []):
            # readers of a shared memory may not delete another app's contribution
            raise RuntimeApiError("memory_not_owned", "Only the app that stored a shared memory (or the owner) can delete it.", 403)
        await self.nomi.request("DELETE", f"/memory/{memory_id}")
        return {"id": memory_id, "deleted": True}

    async def status(self) -> dict[str, Any]:
        info = await self.nomi.health()
        return {"available": True, "service": info.get("service"), "environment": info.get("environment")}
