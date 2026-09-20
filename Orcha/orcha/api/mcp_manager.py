"""
orcha.api.mcp_manager
=====================
Desktop-side management for MCP servers: persisted configuration, explicit
connection states (:class:`~orcha.agent_runtime.mcp.McpState`), backoff
reconnects, hot tool refresh — and one call that hands every currently
connected server's tools to a run's ToolExecutor.

Design notes
------------
- Config lives in ONE human-editable JSON file (``~/.orcha/mcp-servers.json``
  by default, ``ORCHA_MCP_CONFIG`` env override). The file is authoritative
  across restarts; the in-memory registry is rebuilt from it at boot.
- Every network-touching operation is synchronous underneath (the transports
  own their event-loop threads), so the FastAPI layer wraps calls in
  ``asyncio.to_thread`` — the manager itself stays plain and testable.
- Resilience by contract, like everything MCP here: adding a server that
  fails to connect is fine — it lands in ``failed``/``needs_auth`` state,
  shows up in status with the error, and can be retried from the UI.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..agent_runtime.config import McpServerConfig
from ..agent_runtime.mcp import (
    ManagedMcpServer, McpState, qualified_tool_name,
)
from ..agent_runtime.tools import Tool
from ..nodes.tool import ToolSpec

logger = logging.getLogger("orcha.api.mcp_manager")

_DEFAULT_CONFIG_PATH = Path.home() / ".orcha" / "mcp-servers.json"


class McpServerSettings(BaseModel):
    """One persisted MCP server entry (mirrors McpServerConfig fields)."""
    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9 _.-]*$")
    transport: str = "stdio"
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    url: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout_s: float = 30.0
    enabled: bool = True


class McpManager:
    """
    Registry of :class:`ManagedMcpServer` instances backed by one JSON file.
    Thread-safe (a lock guards mutation + persistence); reads are lock-free
    snapshots.
    """

    def __init__(self, config_path: Optional[Path | str] = None) -> None:
        raw = (
            os.environ.get("ORCHA_MCP_CONFIG")
            or (str(config_path) if config_path else None)
            or str(_DEFAULT_CONFIG_PATH)
        )
        self._path = Path(raw)
        self._servers: Dict[str, ManagedMcpServer] = {}
        self._lock = threading.Lock()

    # ── Persistence ─────────────────────────────────────────────────────

    def load(self) -> int:
        """Load + construct servers from disk (no connecting). Returns the
        number of entries found. Best-effort: corrupt files are renamed and
        treated as empty rather than failing boot."""
        try:
            if not self._path.exists():
                return 0
            data = json.loads(self._path.read_text(encoding="utf-8"))
            entries = data.get("servers") if isinstance(data, dict) else data
            if not isinstance(entries, list):
                return 0
            loaded = 0
            for entry in entries:
                try:
                    self.add(McpServerSettings.model_validate(entry), persist=False, connect=False)
                    loaded += 1
                except Exception as exc:
                    logger.warning("skipping MCP config entry: %s", exc)
            return loaded
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("could not read MCP config %s: %s", self._path, exc)
            return 0

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        entries = [
            json.loads(
                McpServerSettings(**_settings_from_server(s)).model_dump_json()
            )
            for s in self._servers.values()
        ]
        payload = {"version": 1, "servers": entries}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    # ── Registry operations ─────────────────────────────────────────────

    def _build(self, settings: McpServerSettings) -> ManagedMcpServer:
        config = McpServerConfig(
            name=settings.name,
            transport=settings.transport,  # type: ignore[arg-type]
            command=settings.command,
            args=settings.args,
            env=settings.env,
            url=settings.url,
            headers=settings.headers,
            timeout_s=settings.timeout_s,
            enabled=settings.enabled,
        )
        return ManagedMcpServer(config, namespace_tools=True)

    def add(
        self,
        settings: McpServerSettings,
        *,
        persist: bool = True,
        connect: bool = True,
    ) -> Dict[str, Any]:
        """Register a server. With ``connect=True`` (default) attempts an
        immediate connection — failures are recorded in state, never raised."""
        key = settings.name.strip()
        with self._lock:
            if key in self._servers:
                raise ValueError(f"An MCP server named {key!r} already exists")
            managed = self._build(settings)
            self._servers[key] = managed
            if persist:
                self._persist()
        if connect and settings.enabled:
            try:
                managed.connect(retry=False)
            except Exception as exc:
                logger.info("MCP server %r added but not connected: %s", key, exc)
        return managed.status()

    def remove(self, name: str) -> bool:
        with self._lock:
            managed = self._servers.pop(name, None)
            if managed is None:
                return False
            self._persist()
        try:
            managed.close()
        except Exception:
            pass
        return True

    def get(self, name: str) -> Optional[ManagedMcpServer]:
        return self._servers.get(name)

    def list_status(self) -> List[Dict[str, Any]]:
        return [s.status() for s in self._servers.values()]

    def reconnect(self, name: str) -> Dict[str, Any]:
        """Force a fresh connection attempt WITH backoff retry."""
        managed = self._servers.get(name)
        if managed is None:
            raise KeyError(name)
        try:
            managed.close()
        except Exception:
            pass
        managed.connect(retry=True)
        return managed.status()

    def refresh(self, name: str) -> Dict[str, Any]:
        """Hot-reload tools if the server's advertised set changed."""
        managed = self._servers.get(name)
        if managed is None:
            raise KeyError(name)
        changed = managed.check_for_changes()
        return {**managed.status(), "tools_changed": changed is not None}

    # ── Tool surface for agent runs ─────────────────────────────────────

    def active_tool_specs(self) -> List[ToolSpec]:
        """
        Specs of every tool exposed by CONNECTED servers, namespaced
        (``mcp__<server>__<tool>``). Dropped into each run's ToolExecutor so
        agents can call them like any local capability.
        """
        out: List[ToolSpec] = []
        for managed in self._servers.values():
            if managed.state != McpState.CONNECTED:
                continue
            for tool in getattr(managed, "_tools", []) or []:
                spec_obj: Optional[ToolSpec] = getattr(tool, "spec", None)
                if spec_obj is not None:
                    out.append(spec_obj)
        return out

    def tool_count(self) -> int:
        return len(self.active_tool_specs())


def _settings_from_server(server: ManagedMcpServer) -> Dict[str, Any]:
    cfg = server.config
    return {
        "name": cfg.name,
        "transport": cfg.transport,
        "command": cfg.command,
        "args": cfg.args,
        "env": cfg.env,
        "url": cfg.url,
        "headers": cfg.headers,
        "timeout_s": cfg.timeout_s,
        "enabled": cfg.enabled,
    }


__all__ = ["McpManager", "McpServerSettings", "qualified_tool_name"]
