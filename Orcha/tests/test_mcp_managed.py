"""
Tests for Phase-6 MCP management upgrades (all against fake transports —
no real MCP SDK needed):

- qualified_tool_name namespacing
- ManagedMcpServer state machine: disabled → connected, failure states,
  needs-auth classification short-circuits retries
- exponential-backoff reconnect (timing-bounded)
- check_for_changes hot reload (add/remove/describe changes → new tools)
- register_managed_mcp_server resilience
"""
import time

import pytest

from orcha.agent_runtime.config import McpServerConfig
from orcha.agent_runtime.mcp import (
    McpState, ManagedMcpServer, classify_connect_failure,
    qualified_tool_name, register_managed_mcp_server,
)
from orcha.agent_runtime.mcp import McpToolInfo, McpToolResult, McpTransport
from orcha.agent_runtime.tools import ToolRegistry


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeTransport(McpTransport):
    def __init__(self, tools=None, fail_times=0, error="connection refused"):
        self.tools = list(tools or [])
        self.fail_times = fail_times
        self.error = error
        self.connect_calls = 0

    @property
    def server_name(self):
        return "fake"

    def connect(self, timeout_s=30.0):
        self.connect_calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError(self.error)

    def list_tools(self, timeout_s=30.0):
        if not hasattr(self, "_session_up"):
            return list(self.tools)
        return list(self.tools)

    def call_tool(self, name, arguments, timeout_s=30.0):
        return McpToolResult(text="ok")

    # test hook: mutate the advertised tools after connect
    def set_tools(self, tools):
        self.tools = list(tools)


def _info(name, description="", schema=None):
    return McpToolInfo(name=name, description=description,
                       input_schema=schema or {"type": "object", "properties": {}})


# ── Namespacing ───────────────────────────────────────────────────────────────

def test_qualified_tool_name():
    assert qualified_tool_name("github", "create_issue") == "mcp__github__create_issue"
    assert qualified_tool_name("my server.v2", "x/y") == "mcp__my_server_v2__x_y"


# ── State machine ────────────────────────────────────────────────────────────

def _managed(transport, **kwargs):
    config = McpServerConfig(name="srv", transport="stdio", command="echo", enabled=True)
    return ManagedMcpServer(config, transport=transport, max_reconnect_attempts=1, **kwargs)


def test_disabled_config_never_connects():
    config = McpServerConfig(name="srv", command="echo", enabled=False)
    transport = FakeTransport([_info("t")])
    managed = ManagedMcpServer(config, transport=transport)
    tools = managed.connect()
    assert tools == [] and managed.state == McpState.DISABLED
    assert transport.connect_calls == 0


def test_happy_path_reaches_connected_with_tools():
    managed = _managed(FakeTransport([_info("alpha"), _info("beta")]))
    tools = managed.connect()
    assert managed.state == McpState.CONNECTED
    assert [t.name for t in tools] == ["alpha", "beta"]
    assert managed.status()["state"] == "connected"


def test_misconfigured_is_failed_not_connected():
    config = McpServerConfig(name="srv", transport="sse", url=None, enabled=True)
    managed = ManagedMcpServer(config, transport=FakeTransport())
    with pytest.raises(Exception):
        managed.connect()
    assert managed.state == McpState.FAILED


def test_needs_auth_classification_short_circuits_retry():
    config = McpServerConfig(name="srv", command="echo", enabled=True)
    transport = FakeTransport(fail_times=99, error="401 unauthorized")
    managed = ManagedMcpServer(
        config, transport=transport,
        max_reconnect_attempts=4, backoff_base_s=0.01,
    )
    t0 = time.perf_counter()
    with pytest.raises(ConnectionError):
        managed.connect(retry=True)
    elapsed = time.perf_counter() - t0
    assert managed.state == McpState.NEEDS_AUTH
    assert transport.connect_calls == 1  # no blind retries on auth problems
    assert elapsed < 1.0  # did NOT sleep through the backoff ladder


def test_backoff_reconnect_recovers_transient_failure():
    config = McpServerConfig(name="srv", command="echo", enabled=True)
    transport = FakeTransport([_info("t")], fail_times=2, error="ECONNREFUSED")
    managed = ManagedMcpServer(
        config, transport=transport,
        max_reconnect_attempts=5, backoff_base_s=0.01, backoff_cap_s=0.05,
    )
    tools = managed.connect(retry=True)
    assert managed.state == McpState.CONNECTED
    assert transport.connect_calls == 3  # two failures + one success
    assert tools[0].name == "t"


# ── Hot reload ───────────────────────────────────────────────────────────────

def test_check_for_changes_detects_additions_and_removals():
    transport = FakeTransport([_info("a"), _info("b")])
    managed = _managed(transport)
    managed.connect()
    assert managed.check_for_changes() is None  # no change

    transport.set_tools([_info("a"), _info("b"), _info("c", description="new")])
    changed = managed.check_for_changes()
    assert changed is not None and [t.name for t in changed] == ["a", "b", "c"]

    transport.set_tools([_info("a")])
    changed = managed.check_for_changes()
    assert changed is not None and [t.name for t in changed] == ["a"]

    transport.set_tools([_info("a", description="updated description")])
    assert managed.check_for_changes() is not None  # description diff counts


def test_namespaced_registration_avoids_shadowing():
    transport = FakeTransport([_info("search")])
    config = McpServerConfig(name="web", command="echo", enabled=True)
    managed = ManagedMcpServer(config, transport=transport, namespace_tools=True)
    registry = ToolRegistry()
    registered = register_managed_mcp_server(registry, managed)
    assert registered[0].name == "mcp__web__search"
    assert registry.has("mcp__web__search")
    # server-facing name preserved in meta; invocation still calls 'search'
    spec_obj = registry.get("mcp__web__search").spec
    assert spec_obj.meta["mcp_tool"] == "search"


def test_register_managed_resilient_to_failure():
    config = McpServerConfig(name="down", command="echo", enabled=True)
    transport = FakeTransport(fail_times=99, error="nope")
    managed = ManagedMcpServer(
        config, transport=transport,
        max_reconnect_attempts=2, backoff_base_s=0.01,
    )
    registry = ToolRegistry()
    out = register_managed_mcp_server(registry, managed)
    assert out == [] and len(registry) == 0
    assert managed.state == McpState.FAILED


def test_auth_error_classifier():
    assert classify_connect_failure(RuntimeError("HTTP 401")) == McpState.NEEDS_AUTH
    assert classify_connect_failure(RuntimeError("missing api key")) == McpState.NEEDS_AUTH
    assert classify_connect_failure(TimeoutError("timed out")) == McpState.FAILED
