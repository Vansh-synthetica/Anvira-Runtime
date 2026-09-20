"""
orcha.agent_runtime.mcp
=======================
MCP client support (OpenHands V1 pattern): connect to MCP servers and
expose their tools through the ToolRegistry as ordinary Tools. The Agent
and the Conversation never know a tool came from an MCP server — an MCP
tool is just another ``Tool`` with a server-provided JSON schema and a
callable that bridges to the server.

Transports
----------
- ``StdioMcpTransport`` — spawn a local command, JSON-RPC over stdin/stdout.
- ``SseMcpTransport``   — remote server over the MCP SSE transport.
- ``McpTransport``      — the minimal sync client protocol; fakes/tests
  implement the same three methods.

Real transports run their own event loop in a background thread (the MCP
SDK sessions are asyncio-bound); the protocol surface they expose is
synchronous, so the rest of the runtime stays sync like Prompt 2.

The official ``mcp`` package is imported lazily inside the real transports:
Orcha boots fine without it — MCP features just need ``pip install mcp``.

Resilience
----------
``register_mcp_server`` catches every failure (unreachable server, bad
handshake, missing fields) and logs a warning; discovery never crashes
AgentRuntime boot.
"""
from __future__ import annotations

import asyncio
import enum
import hashlib
import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..capabilities.base import CAUTIOUS, NETWORK, ToolError, spec
from ..nodes.tool import ToolSchema, ToolSpec
from .config import McpServerConfig
from .tools import Tool, ToolRegistry

logger = logging.getLogger("orcha.agent_runtime.mcp")


# ── Protocol types ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class McpToolInfo:
    """A tool a server advertises: name, description, JSON schema."""
    name: str
    description: str = ""
    input_schema: ToolSchema = field(default_factory=dict)


@dataclass(frozen=True)
class McpToolResult:
    """Rendered result of one tool call."""
    text: str = ""
    is_error: bool = False


class McpTransport(ABC):
    """
    Minimal sync MCP client protocol. Implementations own their own
    concurrency: real transports run a dedicated event loop in a background
    thread; test doubles can be plain objects.
    """

    @property
    def server_name(self) -> str:
        return "mcp-server"

    @abstractmethod
    def connect(self, timeout_s: float = 30.0) -> None:
        """Handshake with the server. Raise on any failure — the caller
        (register_mcp_server) turns failures into warnings."""

    @abstractmethod
    def list_tools(self, timeout_s: float = 30.0) -> List[McpToolInfo]:
        ...

    @abstractmethod
    def call_tool(
        self, name: str, arguments: Dict[str, Any], timeout_s: float = 30.0,
    ) -> McpToolResult:
        ...

    def close(self) -> None:
        """Best-effort teardown; implementations may leave it a no-op."""


# ── Real transports (official `mcp` SDK, lazily imported) ───────────────────

class _LoopRunner:
    """A private event loop in a background daemon thread, so the asyncio
    MCP session can live for the lifetime of the transport while the rest
    of the runtime stays synchronous."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="orcha-mcp-loop",
        )

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        try:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
        except Exception:
            pass
        self._loop.close()

    def start(self) -> None:
        self._thread.start()

    def call(self, awaitable: Any, timeout_s: float) -> Any:
        """Run an awaitable in the transport loop; block until it resolves
        or the timeout elapses (raises asyncio.TimeoutError)."""
        future = asyncio.run_coroutine_threadsafe(awaitable, self._loop)
        return future.result(timeout=timeout_s)

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)


class StdioMcpTransport(McpTransport):
    """
    Local MCP server: spawn ``command args`` and speak JSON-RPC over its
    stdin/stdout via the official MCP SDK.

    The SDK session lives in a worker coroutine on the transport's private
    loop (canonical nested ``async with`` — the SDK's context-manager
    teardown is not re-entrant, so it must all happen inside one task).
    Sync methods (connect/list_tools/call_tool/close) bridge to that task
    through the loop.
    """

    def __init__(
        self, command: str, args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None, *,
        name: str = "mcp-server",
    ) -> None:
        self._command = command
        self._args = list(args or [])
        self._env = dict(env or {})
        self._name = name
        self._runner: Optional[_LoopRunner] = None
        self._ready: Optional[Any] = None
        self._stop: Optional[Any] = None
        self._done: Optional[Any] = None
        self._session: Any = None
        self._error: Optional[BaseException] = None

    @property
    def server_name(self) -> str:
        return self._name

    def connect(self, timeout_s: float = 30.0) -> None:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command=self._command, args=self._args, env=self._env or None,
        )
        self._start_worker(stdio_client, ClientSession, {"server": params})
        try:
            self._runner.call(self._wait_started(), timeout_s)
        except TimeoutError:
            self.close()
            raise ToolError(
                "mcp_connect_timeout",
                f"timed out connecting to MCP server {self._name!r}",
            )
        if self._error is not None:
            exc = self._error
            self.close()
            raise ToolError(
                "mcp_connect_failed",
                f"failed to connect to MCP server {self._name!r}: {exc}",
            ) from exc

    def _start_worker(self, factory: Any, session_cls: Any, params: Dict[str, Any]) -> None:
        self._runner = _LoopRunner()
        self._runner.start()
        self._ready, self._stop, self._done = (
            asyncio.Event(), asyncio.Event(), asyncio.Event(),
        )
        self._error = None
        asyncio.run_coroutine_threadsafe(
            self._worker(factory, session_cls, params), self._runner._loop,
        )

    async def _worker(self, factory: Any, session_cls: Any, params: Dict[str, Any]) -> None:
        try:
            cm = factory(**params)
            async with cm as (read, write):
                async with session_cls(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._stop.wait()
        except Exception as exc:  # handshake/lifecycle failure → surfaced by connect
            self._error = exc
        finally:
            self._session = None
            self._done.set()

    async def _wait_started(self) -> None:
        while not self._ready.is_set() and not self._done.is_set():
            await asyncio.sleep(0.005)

    def list_tools(self, timeout_s: float = 30.0) -> List[McpToolInfo]:
        if self._runner is None or self._session is None:
            raise ToolError("mcp_not_connected", "MCP transport is not connected")
        try:
            result = self._runner.call(self._session.list_tools(), timeout_s)
        except Exception as exc:
            raise ToolError("mcp_list_failed", f"mcp list_tools failed: {exc}") from exc
        return _tools_from_result(result)

    def call_tool(
        self, name: str, arguments: Dict[str, Any], timeout_s: float = 30.0,
    ) -> McpToolResult:
        if self._runner is None or self._session is None:
            raise ToolError("mcp_not_connected", "MCP transport is not connected")
        try:
            result = self._runner.call(
                self._session.call_tool(name, arguments), timeout_s,
            )
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError("mcp_call_failed", f"mcp call to {name!r} failed: {exc}") from exc
        return _render_call_result(result)

    def close(self) -> None:
        if self._runner is None:
            return
        if self._stop is not None:
            self._runner._loop.call_soon_threadsafe(self._stop.set)
        if self._done is not None:
            try:
                self._runner.call(self._done.wait(), 10.0)
            except Exception:
                pass
        self._runner.stop()
        self._runner = None
        self._session = None


class SseMcpTransport(McpTransport):
    """Remote MCP server over the SSE transport (mcp.client.sse)."""

    def __init__(
        self, url: str, headers: Optional[Dict[str, str]] = None, *,
        name: str = "mcp-server",
    ) -> None:
        self._url = url
        self._headers = dict(headers or {})
        self._name = name
        self._runner: Optional[_LoopRunner] = None
        self._ready: Optional[Any] = None
        self._stop: Optional[Any] = None
        self._done: Optional[Any] = None
        self._session: Any = None
        self._error: Optional[BaseException] = None

    @property
    def server_name(self) -> str:
        return self._name

    def connect(self, timeout_s: float = 30.0) -> None:
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        self._start_worker(sse_client, ClientSession, {
            "url": self._url, "headers": self._headers,
        })
        try:
            self._runner.call(self._wait_started(), timeout_s)
        except TimeoutError:
            self.close()
            raise ToolError(
                "mcp_connect_timeout",
                f"timed out connecting to MCP server {self._name!r}",
            )
        if self._error is not None:
            exc = self._error
            self.close()
            raise ToolError(
                "mcp_connect_failed",
                f"failed to connect to MCP server {self._name!r}: {exc}",
            ) from exc

    def _start_worker(self, factory: Any, session_cls: Any, params: Dict[str, Any]) -> None:
        self._runner = _LoopRunner()
        self._runner.start()
        self._ready, self._stop, self._done = (
            asyncio.Event(), asyncio.Event(), asyncio.Event(),
        )
        self._error = None
        asyncio.run_coroutine_threadsafe(
            self._worker(factory, session_cls, params), self._runner._loop,
        )

    async def _worker(self, factory: Any, session_cls: Any, params: Dict[str, Any]) -> None:
        try:
            cm = factory(**params)
            async with cm as (read, write):
                async with session_cls(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._stop.wait()
        except Exception as exc:
            self._error = exc
        finally:
            self._session = None
            self._done.set()

    async def _wait_started(self) -> None:
        while not self._ready.is_set() and not self._done.is_set():
            await asyncio.sleep(0.005)

    def list_tools(self, timeout_s: float = 30.0) -> List[McpToolInfo]:
        if self._runner is None or self._session is None:
            raise ToolError("mcp_not_connected", "MCP transport is not connected")
        try:
            result = self._runner.call(self._session.list_tools(), timeout_s)
        except Exception as exc:
            raise ToolError("mcp_list_failed", f"mcp list_tools failed: {exc}") from exc
        return _tools_from_result(result)

    def call_tool(
        self, name: str, arguments: Dict[str, Any], timeout_s: float = 30.0,
    ) -> McpToolResult:
        if self._runner is None or self._session is None:
            raise ToolError("mcp_not_connected", "MCP transport is not connected")
        try:
            result = self._runner.call(
                self._session.call_tool(name, arguments), timeout_s,
            )
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError("mcp_call_failed", f"mcp call to {name!r} failed: {exc}") from exc
        return _render_call_result(result)

    def close(self) -> None:
        if self._runner is None:
            return
        if self._stop is not None:
            self._runner._loop.call_soon_threadsafe(self._stop.set)
        if self._done is not None:
            try:
                self._runner.call(self._done.wait(), 10.0)
            except Exception:
                pass
        self._runner.stop()
        self._runner = None
        self._session = None


def _tools_from_result(result: Any) -> List[McpToolInfo]:
    tools = []
    for tool in getattr(result, "tools", []) or []:
        name = getattr(tool, "name", "") or ""
        description = getattr(tool, "description", "") or ""
        schema = getattr(tool, "input_schema", None) or getattr(
            tool, "inputSchema", None,
        ) or {}
        tools.append(McpToolInfo(name=name, description=description,
                                 input_schema=schema))
    return tools


def _render_call_result(result: Any) -> McpToolResult:
    """Render an MCP CallToolResult (content blocks) to text."""
    # mcp 2.x uses `is_error`; older SDKs used `isError` — honor both.
    is_error = bool(
        getattr(result, "is_error", None) or getattr(result, "isError", False)
    )
    parts = []
    for block in getattr(result, "content", []) or []:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("text") is not None:
                parts.append(str(block["text"]))
        else:
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(str(text))
            else:
                parts.append(str(block))
    return McpToolResult(text="\n".join(parts), is_error=is_error)


# ── Building Orcha Tools from MCP tools ──────────────────────────────────────

def qualified_tool_name(server_name: str, tool_name: str) -> str:
    """
    Namespace an MCP tool under its server: ``mcp__<server>__<tool>``.
    Prevents cross-server shadowing when several servers expose same-named
    tools (the flat-registry last-wins rule would silently drop one).
    """
    safe_server = re.sub(r"[^A-Za-z0-9_-]+", "_", server_name)
    safe_tool = re.sub(r"[^A-Za-z0-9_-]+", "_", tool_name)
    return f"mcp__{safe_server}__{safe_tool}"


def build_mcp_tool(
    info: McpToolInfo,
    transport: McpTransport,
    *,
    server_name: str = "mcp-server",
    timeout_s: float = 30.0,
    name_override: Optional[str] = None,
) -> Tool:
    """
    An MCP tool is just a Tool: the server's schema becomes the tool's
    JSON-schema parameters, and execution bridges to the transport. The
    Agent/ModelBackend see nothing MCP-specific.

    ``name_override`` lets a manager register a namespaced tool name while
    keeping the server-facing call name intact.
    """
    schema: ToolSchema = info.input_schema or {
        "type": "object", "properties": {}, "required": [],
    }

    def invoke(**kwargs: Any) -> Any:
        result = transport.call_tool(info.name, kwargs, timeout_s)
        if result.is_error:
            raise ToolError(
                "mcp_error", f"mcp tool {info.name!r} reported an error: {result.text}",
            )
        return result.text

    spec_obj = spec(
        name_override or info.name,
        info.description or f"MCP tool exposed by server {server_name}.",
        schema,
        invoke,
        permissions=[NETWORK],
        safety_level=CAUTIOUS,
        capability="mcp",
        result_format={"type": "string"},
        meta={"mcp_server": server_name, "mcp_tool": info.name},
    )
    return Tool(spec_obj)


def build_transport(config: McpServerConfig) -> McpTransport:
    """Instantiate the transport a config describes."""
    if config.transport == "sse":
        return SseMcpTransport(
            config.url or "", headers=config.headers, name=config.name,
        )
    return StdioMcpTransport(
        config.command or "", args=config.args, env=config.env, name=config.name,
    )


class McpServer:
    """
    One connected MCP server. ``connect()`` handshakes and returns the
    server's tools as ordinary Tools; ``close()`` tears the transport down.
    """

    def __init__(
        self, config: McpServerConfig, transport: Optional[McpTransport] = None,
    ) -> None:
        self.config = config
        self._transport = transport or build_transport(config)

    @property
    def transport(self) -> McpTransport:
        return self._transport

    def connect(self) -> List[Tool]:
        missing = self.config.validate_fields()
        if missing:
            raise ToolError(
                "mcp_misconfigured",
                f"missing required field(s) for {self.config.transport} "
                f"transport: {', '.join(missing)}",
            )
        self._transport.connect(self.config.timeout_s)
        infos = self._transport.list_tools(self.config.timeout_s)
        return [
            build_mcp_tool(
                info, self._transport,
                server_name=self.config.name,
                timeout_s=self.config.timeout_s,
            )
            for info in infos
        ]

    def close(self) -> None:
        self._transport.close()


def register_mcp_server(
    registry: ToolRegistry,
    config: McpServerConfig,
    *,
    transport: Optional[McpTransport] = None,
    log: Optional[logging.Logger] = None,
) -> List[Tool]:
    """
    Connect one MCP server and register its tools into ``registry``.

    Resilient by contract: every failure (unreachable, misconfigured, bad
    handshake) is logged as a warning and the server is skipped — this
    function never raises.
    """
    log = log or logger
    if not config.enabled:
        return []
    server = McpServer(config, transport=transport)
    try:
        tools = server.connect()
    except Exception as exc:
        log.warning("skipping MCP server %r: %s", config.name, exc)
        return []
    registered: List[Tool] = []
    for tool in tools:
        if registry.has(tool.name):
            log.warning(
                "MCP tool %r from server %r shadows an existing tool",
                tool.name, config.name,
            )
        registry.register(tool)
        registered.append(tool)
    log.info("registered %d tool(s) from MCP server %r", len(tools), config.name)
    return registered


# ── Managed connections: state machine + reconnect + hot reload ─────────────

class McpState(str, enum.Enum):
    """Explicit lifecycle states for a managed MCP connection."""
    DISABLED = "disabled"    # config.enabled is False — never connect
    FAILED = "failed"        # last connect attempt failed (retry allowed)
    NEEDS_AUTH = "needs_auth"  # failure classified as an auth problem
    CONNECTING = "connecting"
    CONNECTED = "connected"


_AUTH_ERROR_MARKERS = (
    "401", "403", "unauthorized", "forbidden", "auth", "token", "api key",
    "api_key", "permission",
)


def classify_connect_failure(exc: BaseException) -> McpState:
    """
    Best-effort auth classification from error text: lets hosts surface a
    login affordance instead of blind retries. Never raises.
    """
    text = f"{type(exc).__name__} {exc}".lower()
    if any(marker in text for marker in _AUTH_ERROR_MARKERS):
        return McpState.NEEDS_AUTH
    return McpState.FAILED


class ManagedMcpServer:
    """
    A state-managed MCP connection.

    Adds to the raw :class:`McpServer`:

    - **Explicit state** (:class:`McpState`) surfaced via ``status()`` so
      UIs can render connected/failed/needs-auth/disabled servers.
    - **Exponential-backoff reconnect** (``connect(retry=True)``): 1s →
      2s → 4s … capped, bounded attempts; needs-auth failures short-circuit
      the retry loop because retrying cannot fix missing credentials.
    - **Hot tool refresh** (``check_for_changes()``): re-lists tools and
      detects additions/removals/description changes by hash — the host
      polls this cheaply instead of holding SDK notification streams open.
    - **Namespaced registration**: qualified names
      ``mcp__<server>__<tool>`` so multiple servers never shadow each other;
      the server-facing call name stays intact in the spec's meta.

    Resilience contract unchanged: no method raises past ``connect``;
    failures land in state + status.
    """

    def __init__(
        self,
        config: McpServerConfig,
        transport: Optional[McpTransport] = None,
        *,
        namespace_tools: bool = False,
        max_reconnect_attempts: int = 5,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 30.0,
    ) -> None:
        self.config = config
        self._transport = transport or build_transport(config)
        self.namespace_tools = namespace_tools
        self.max_reconnect_attempts = max(1, int(max_reconnect_attempts))
        self.backoff_base_s = max(0.1, float(backoff_base_s))
        self.backoff_cap_s = max(self.backoff_base_s, float(backoff_cap_s))
        self._state = McpState.DISABLED if not config.enabled else McpState.FAILED
        self._last_error: Optional[str] = None
        self._tools: List[Tool] = []
        self._tools_hash: str = ""
        self._connected_at: Optional[float] = None

    # ── Introspection ───────────────────────────────────────────────────

    @property
    def state(self) -> McpState:
        return self._state

    @property
    def transport(self) -> McpTransport:
        return self._transport

    def status(self) -> Dict[str, Any]:
        return {
            "server": self.config.name,
            "state": self._state.value,
            "tools": [t.name for t in self._tools],
            "last_error": self._last_error,
            "connected_at": self._connected_at,
            "enabled": bool(self.config.enabled),
        }

    # ── Connection ──────────────────────────────────────────────────────

    def connect(self, *, retry: bool = False) -> List[Tool]:
        """
        Connect and return the server's Tools (namespaced when enabled).
        With ``retry=True``, transient failures are retried on exponential
        backoff up to ``max_reconnect_attempts``; auth failures are NOT
        retried. Always ends in a defined state; raises only after the
        final attempt fails (the caller decides whether that is fatal).
        """
        if not self.config.enabled:
            self._state = McpState.DISABLED
            return []
        attempts = self.max_reconnect_attempts if retry else 1
        delay = self.backoff_base_s
        last_exc: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            self._state = McpState.CONNECTING
            try:
                self._state_check_connected()
                tools = self._build_tools()
                self._state = McpState.CONNECTED
                self._last_error = None
                self._connected_at = time.time()
                return tools
            except Exception as exc:
                last_exc = exc
                self._last_error = str(exc)
                self._state = classify_connect_failure(exc)
                if self._state == McpState.NEEDS_AUTH:
                    break  # retrying cannot fix credentials
                if attempt < attempts:
                    time.sleep(min(delay, self.backoff_cap_s))
                    delay *= 2
        assert last_exc is not None
        raise last_exc

    def _state_check_connected(self) -> None:
        # Validate config fields first (misconfigured ≠ failed).
        missing = self.config.validate_fields()
        if missing:
            raise ToolError(
                "mcp_misconfigured",
                f"missing required field(s) for {self.config.transport} "
                f"transport: {', '.join(missing)}",
            )
        if self._transport is None:
            raise ToolError("mcp_not_connected", "no transport configured")
        self._transport.connect(self.config.timeout_s)

    def _build_tools(self) -> List[Tool]:
        infos = self._transport.list_tools(self.config.timeout_s)
        self._tools_hash = _hash_tool_infos(infos)
        self._tools = [
            build_mcp_tool(
                info, self._transport,
                server_name=self.config.name,
                timeout_s=self.config.timeout_s,
                name_override=(
                    qualified_tool_name(self.config.name, info.name)
                    if self.namespace_tools else None
                ),
            )
            for info in infos
        ]
        return list(self._tools)

    # ── Hot reload ──────────────────────────────────────────────────────

    def check_for_changes(self) -> Optional[List[Tool]]:
        """
        Re-list tools and compare against the current set. Returns the NEW
        tool list when something changed (added/removed/described), else
        None. A dropped connection flips state to FAILED and returns None —
        call :meth:`connect` to recover.
        """
        if self._state != McpState.CONNECTED:
            return None
        try:
            infos = self._transport.list_tools(self.config.timeout_s)
        except Exception as exc:
            self._state = classify_connect_failure(exc)
            self._last_error = str(exc)
            return None
        new_hash = _hash_tool_infos(infos)
        if new_hash == self._tools_hash:
            return None
        return self._build_tools()

    def close(self) -> None:
        try:
            self._transport.close()
        finally:
            if self._state == McpState.CONNECTED:
                self._state = McpState.FAILED
                self._connected_at = None


def _hash_tool_infos(infos: List[McpToolInfo]) -> str:
    payload = json.dumps(
        [(i.name, i.description, _schema_canonical(i.input_schema)) for i in infos],
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _schema_canonical(schema: Dict[str, Any]) -> Any:
    try:
        return json.dumps(schema, sort_keys=True, default=str)
    except Exception:
        return str(schema)


def register_managed_mcp_server(
    registry: ToolRegistry,
    managed: ManagedMcpServer,
    *,
    log: Optional[logging.Logger] = None,
) -> List[Tool]:
    """
    Connect (with backoff retry) and register a managed server's tools.
    Resilient by contract like ``register_mcp_server``: any failure is a
    warning; the server's state records why it is absent.
    """
    log = log or logger
    try:
        tools = managed.connect(retry=True)
    except Exception as exc:
        log.warning(
            "skipping MCP server %r (%s): %s",
            managed.config.name, managed.state.value, exc,
        )
        return []
    registered: List[Tool] = []
    for tool in tools:
        if registry.has(tool.name):
            log.warning(
                "MCP tool %r from server %r shadows an existing tool",
                tool.name, managed.config.name,
            )
        registry.register(tool)
        registered.append(tool)
    log.info(
        "registered %d tool(s) from MCP server %r", len(tools), managed.config.name,
    )
    return registered


__all__ = [
    "McpToolInfo", "McpToolResult", "McpTransport",
    "StdioMcpTransport", "SseMcpTransport",
    "build_mcp_tool", "build_transport", "McpServer", "register_mcp_server",
    "McpState", "ManagedMcpServer", "register_managed_mcp_server",
    "qualified_tool_name", "classify_connect_failure",
]
