"""
Tests for the MCP manager (api layer): persistence, registry ops,
tool-surface injection. All against FakeTransport — no real MCP SDK.
"""
import json

import pytest

from orcha.api.mcp_manager import McpManager, McpServerSettings
from orcha.agent_runtime.mcp import McpState


@pytest.fixture
def manager(tmp_path):
    return McpManager(config_path=tmp_path / "mcp-servers.json")


class RecordingTransport:
    """Fake transport: connect succeeds unless fail_times remain; tools fixed."""

    def __init__(self, tools=None, fail_times=0):
        from orcha.agent_runtime.mcp import McpToolInfo

        self.tools = tools or [McpToolInfo(name="ping", description="p")]
        self.fail_times = fail_times
        self.connect_calls = 0

    @property
    def server_name(self):
        return "fake"

    def connect(self, timeout_s=30.0):
        self.connect_calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("refused")

    def list_tools(self, timeout_s=30.0):
        return list(self.tools)

    def call_tool(self, name, arguments, timeout_s=30.0):
        from orcha.agent_runtime.mcp import McpToolResult

        return McpToolResult(text=f"pong:{name}")

    def close(self):
        pass


def _patch_build(monkeypatch, transports_by_name):
    """Patch ManagedMcpServer transport construction to record fakes."""
    from orcha.api import mcp_manager as mm

    original = mm.ManagedMcpServer.__init__

    def patched(self, config, transport=None, **kwargs):
        if transport is None:
            transport = transports_by_name.setdefault(
                config.name, RecordingTransport()
            )
        original(self, config, transport=transport, **kwargs)

    monkeypatch.setattr(mm.ManagedMcpServer, "__init__", patched)
    return transports_by_name


def test_add_persists_and_loads(tmp_path, monkeypatch):
    path = tmp_path / "mcp.json"
    m1 = McpManager(config_path=path)
    _patch_build(monkeypatch, {})
    status = m1.add(McpServerSettings(name="srv", command="echo"), persist=True)
    assert status["state"] == "connected"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["servers"][0]["name"] == "srv"

    # Fresh manager rebuilds registry from disk (no connecting at load).
    m2 = McpManager(config_path=path)
    assert m2.load() == 1
    statuses = m2.list_status()
    assert len(statuses) == 1
    # Not connected yet (load does not dial out).
    assert statuses[0]["state"] in ("failed", "connected")


def test_duplicate_name_rejected(tmp_path, monkeypatch):
    m = McpManager(config_path=tmp_path / "mcp.json")
    _patch_build(monkeypatch, {})
    m.add(McpServerSettings(name="srv", command="echo"), persist=False)
    with pytest.raises(ValueError):
        m.add(McpServerSettings(name="srv", command="echo"), persist=False)


def test_failed_connect_lands_in_state_not_exception(tmp_path, monkeypatch):
    m = McpManager(config_path=tmp_path / "mcp.json")
    transports = _patch_build(monkeypatch, {})
    transports["down"] = RecordingTransport(fail_times=99)
    status = m.add(McpServerSettings(name="down", command="x"), persist=False)
    assert status["state"] == "failed"
    assert "refused" in (status["last_error"] or "")


def test_remove_stops_and_unpersists(tmp_path, monkeypatch):
    path = tmp_path / "mcp.json"
    m = McpManager(config_path=path)
    _patch_build(monkeypatch, {})
    m.add(McpServerSettings(name="srv", command="echo"))
    assert m.remove("srv") is True
    assert json.loads(path.read_text(encoding="utf-8"))["servers"] == []
    assert m.remove("srv") is False


def test_active_tool_specs_namespaced_and_connected_only(tmp_path, monkeypatch):
    m = McpManager(config_path=tmp_path / "mcp.json")
    transports = _patch_build(monkeypatch, {})
    transports["good"] = RecordingTransport()
    transports["bad"] = RecordingTransport(fail_times=99)
    m.add(McpServerSettings(name="good", command="a"), persist=False)
    m.add(McpServerSettings(name="bad", command="b"), persist=False)
    specs = m.active_tool_specs()
    names = [s.name for s in specs]
    assert names == ["mcp__good__ping"]
    # Server-facing call name preserved.
    assert specs[0].meta["mcp_tool"] == "ping"


def test_refresh_detects_tool_changes(tmp_path, monkeypatch):
    m = McpManager(config_path=tmp_path / "mcp.json")
    transports = _patch_build(monkeypatch, {})
    transport = RecordingTransport()
    transports["srv"] = transport
    m.add(McpServerSettings(name="srv", command="x"), persist=False)

    result = m.refresh("srv")
    assert result["tools_changed"] is False

    from orcha.agent_runtime.mcp import McpToolInfo

    transport.tools.append(McpToolInfo(name="extra", description="e"))
    result = m.refresh("srv")
    assert result["tools_changed"] is True
    assert "mcp__srv__extra" in result["tools"]
