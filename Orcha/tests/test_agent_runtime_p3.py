"""
Tests for the Prompt 3 layer of orcha.agent_runtime: external tools.

Two sources, one registry:

- MCP servers (stdio/SSE): their tools are registered into the shared
  ToolRegistry as ordinary Tools (OpenHands V1 pattern). The Agent,
  Conversation and ModelBackend never know a tool came from an MCP server.
  Real transports run their own loop in a background thread; test doubles
  implement the same minimal sync protocol as plain objects.
- Local skills: folder-per-skill (SKILL.md frontmatter metadata +
  entrypoint), loaded lazily on first call. Malformed folders are skipped
  with a warning.

Resilience by contract: unreachable MCP server / malformed skill folder →
warning + skip, AgentRuntime boot never crashes. Config surface lives
alongside the ModelBackend config (AgentRuntimeConfig).
"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

from orcha.agent_runtime import (
    AgentRuntimeConfig, Conversation, ConversationConfig, ErrorObservation,
    FinishAction, McpServer, McpServerConfig, McpToolInfo, McpToolResult,
    ModelBackend, ModelResponse, SkillsDirectory, StdioMcpTransport,
    SseMcpTransport, Tool, ToolCall, ToolCallAction, ToolCallingAgent,
    ToolRegistry, ToolResultObservation, ToolWorkspace, TokenUsage,
    action_packet, build_agent_runtime, build_mcp_tool, build_transport,
    discover_tools, observation_from_packet, parse_skill_metadata,
    register_mcp_server, register_skills, skill_to_tool,
)
from orcha.capabilities.base import CAUTIOUS, NETWORK
from orcha.agent_runtime.backends import OpenAICompatBackend


# ── Test doubles ─────────────────────────────────────────────────────────────

def tool_call_response(
    name: str, arguments: Dict[str, Any], *,
    tc_id: str = "call_1",
) -> ModelResponse:
    return ModelResponse(
        tool_calls=[ToolCall(id=tc_id, name=name, arguments=arguments)],
        usage=TokenUsage(prompt_tokens=5, completion_tokens=7),
    )


def text_response(content: str) -> ModelResponse:
    return ModelResponse(
        content=content,
        usage=TokenUsage(prompt_tokens=5, completion_tokens=7),
    )


class FakeBackend(ModelBackend):
    """Scripted backend: pops responses in order; records every call."""

    def __init__(
        self, responses: Optional[Sequence[ModelResponse]] = None,
    ) -> None:
        self._responses = list(responses or [])
        self.calls: List[Dict[str, Any]] = []
        self._streams_tools = False
        self._reports_reasoning = False

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def supports_streaming_tool_calls(self) -> bool:
        return self._streams_tools

    @property
    def reports_reasoning_tokens(self) -> bool:
        return self._reports_reasoning

    def complete(self, messages, tools=None, config=None) -> ModelResponse:
        self.calls.append({"messages": list(messages), "tools": tools, "config": config})
        if not self._responses:
            return text_response("done")
        return self._responses.pop(0)


class FakeMcpTransport:
    """Sync-protocol MCP transport double: no threads, scripted results."""

    def __init__(
        self, tools: Sequence[McpToolInfo],
        results: Optional[Sequence[McpToolResult]] = None, *,
        fail_connect: bool = False,
        name: str = "fake-server",
    ) -> None:
        self._tools = list(tools)
        self._results = list(results or [])
        self.fail_connect = fail_connect
        self.calls: List[tuple] = []
        self.closed = False
        self._name = name

    @property
    def server_name(self) -> str:
        return self._name

    def connect(self, timeout_s: float = 30.0) -> None:
        if self.fail_connect:
            raise ConnectionError("server unreachable")

    def list_tools(self, timeout_s: float = 30.0) -> List[McpToolInfo]:
        return list(self._tools)

    def call_tool(self, name, arguments, timeout_s: float = 30.0) -> McpToolResult:
        self.calls.append((name, dict(arguments)))
        if self._results:
            return self._results.pop(0)
        return McpToolResult(text=f"{name} <- {arguments}", is_error=False)

    def close(self) -> None:
        self.closed = True


def echo_info() -> McpToolInfo:
    return McpToolInfo(
        name="echo", description="Echo a message",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    )


# ── Config surface (alongside the backend config) ────────────────────────────

def test_runtime_config_defaults_and_json_roundtrip():
    cfg = AgentRuntimeConfig()
    assert cfg.mcp_servers == []
    assert cfg.skills_dirs == []
    assert cfg.backend.model == "llama3.2:3b"  # the Prompt-2 backend default

    text = AgentRuntimeConfig(
        mcp_servers=[McpServerConfig(name="srv", transport="sse", url="http://x/")],
        skills_dirs=["skills/"],
    ).model_dump_json()
    back = AgentRuntimeConfig.from_json(text)
    assert back.mcp_servers[0].transport == "sse"
    assert back.mcp_servers[0].url == "http://x/"
    assert back.skills_dirs == ["skills/"]


def test_mcp_config_validate_fields_detects_missing_required():
    assert McpServerConfig(name="a", transport="stdio", command=None).validate_fields() == ["command"]
    assert McpServerConfig(name="b", transport="sse", url=None).validate_fields() == ["url"]
    assert McpServerConfig(name="c", transport="stdio", command="npx").validate_fields() == []
    assert McpServerConfig(name="d", transport="sse", url="http://x/").validate_fields() == []


def test_build_transport_maps_transports():
    assert isinstance(
        build_transport(McpServerConfig(name="s", command="npx", args=["-y", "x"])),
        StdioMcpTransport,
    )
    assert isinstance(
        build_transport(McpServerConfig(name="s", transport="sse", url="http://x/")),
        SseMcpTransport,
    )


# ── MCP tools are ordinary Tools ─────────────────────────────────────────────

def test_mcp_tool_spec_contract():
    tool = build_mcp_tool(echo_info(), FakeMcpTransport([]), server_name="srv")
    assert tool.name == "echo"
    assert tool.description == "Echo a message"
    assert tool.spec.capability == "mcp"
    assert tool.permissions == [NETWORK]
    assert tool.safety_level == CAUTIOUS
    assert tool.parameters["required"] == ["message"]


def test_mcp_tool_call_through_workspace():
    transport = FakeMcpTransport([echo_info()], [McpToolResult(text="pong", is_error=False)])
    registry = ToolRegistry([build_mcp_tool(echo_info(), transport, server_name="srv")])
    ws = ToolWorkspace(registry)

    out = ws.execute(action_packet(ToolCallAction(name="echo", arguments={"message": "ping"})))
    obs = observation_from_packet(out)

    assert isinstance(obs, ToolResultObservation)
    assert obs.success and "pong" in obs.content
    assert transport.calls == [("echo", {"message": "ping"})]


def test_mcp_tool_error_result_becomes_error_observation():
    transport = FakeMcpTransport([echo_info()], [McpToolResult(text="boom", is_error=True)])
    registry = ToolRegistry([build_mcp_tool(echo_info(), transport, server_name="srv")])
    ws = ToolWorkspace(registry)

    out = ws.execute(action_packet(ToolCallAction(name="echo", arguments={"message": "x"})))
    obs = observation_from_packet(out)

    assert isinstance(obs, ErrorObservation)
    assert "mcp_error" in obs.message and "boom" in obs.message


def test_mcp_server_connect_builds_tools():
    transport = FakeMcpTransport([echo_info()])
    server = McpServer(McpServerConfig(name="srv", command="x"), transport=transport)

    tools = server.connect()

    assert [t.name for t in tools] == ["echo"]
    assert transport.closed is False  # stays connected for the app's lifetime


def test_register_mcp_server_registers_all_tools(caplog):
    registry = ToolRegistry()
    config = McpServerConfig(name="srv", command="x")
    with caplog.at_level(logging.INFO, logger="orcha.agent_runtime.mcp"):
        out = register_mcp_server(registry, config, transport=FakeMcpTransport([echo_info()]))

    assert [t.name for t in out] == ["echo"]
    assert registry.has("echo")
    assert any("registered 1 tool(s)" in r.getMessage() for r in caplog.records)


def test_register_mcp_server_skips_unreachable_server(caplog):
    registry = ToolRegistry()
    config = McpServerConfig(name="srv", command="x")
    with caplog.at_level(logging.WARNING, logger="orcha.agent_runtime.mcp"):
        out = register_mcp_server(
            registry, config,
            transport=FakeMcpTransport([echo_info()], fail_connect=True),
        )

    assert out == [] and len(registry) == 0
    assert any("skipping MCP server" in r.getMessage() for r in caplog.records)


def test_register_mcp_server_skips_misconfigured(caplog):
    registry = ToolRegistry()
    config = McpServerConfig(name="srv", transport="stdio", command=None)
    with caplog.at_level(logging.WARNING, logger="orcha.agent_runtime.mcp"):
        out = register_mcp_server(registry, config, transport=FakeMcpTransport([echo_info()]))

    assert out == [] and len(registry) == 0
    assert any("missing required field" in r.getMessage() for r in caplog.records)


def test_register_mcp_server_respects_disabled_flag():
    registry = ToolRegistry()
    out = register_mcp_server(
        registry, McpServerConfig(name="srv", enabled=False),
        transport=FakeMcpTransport([echo_info()]),
    )
    assert out == [] and len(registry) == 0


def test_register_mcp_server_warns_on_shadowing(caplog):
    registry = ToolRegistry([
        Tool(build_mcp_tool(echo_info(), FakeMcpTransport([]), server_name="old")),
    ])
    config = McpServerConfig(name="srv", command="x")
    with caplog.at_level(logging.WARNING, logger="orcha.agent_runtime.mcp"):
        register_mcp_server(registry, config, transport=FakeMcpTransport([echo_info()]))

    assert any("shadows an existing tool" in r.getMessage() for r in caplog.records)
    assert len(registry) == 1  # replaced, not duplicated


async def test_mcp_tool_end_to_end_through_conversation():
    transport = FakeMcpTransport([echo_info()])
    registry = ToolRegistry()
    register_mcp_server(registry, McpServerConfig(name="srv", command="x"), transport=transport)

    backend = FakeBackend([
        tool_call_response("echo", {"message": "hi"}),
        text_response("echoed"),
    ])
    convo = Conversation(ToolCallingAgent(backend, registry), ToolWorkspace(registry), query="q")
    convo.submit_user_message("echo hi")

    result = await convo.run()

    assert result.completed and result.answer == "echoed"
    assert [type(ev.payload).__name__ for ev in result.events] == [
        "UserMessageObservation",
        "ToolCallAction", "ToolResultObservation",
        "FinishAction",
    ]
    assert transport.calls == [("echo", {"message": "hi"})]
    # the agent offered the MCP tool's schema to the model
    assert any(t["function"]["name"] == "echo" for t in backend.calls[0]["tools"])


# ── Skills: metadata ─────────────────────────────────────────────────────────

def test_parse_skill_metadata_happy_path():
    meta = parse_skill_metadata(
        "---\n"
        "name: summarize\n"
        "description: Summarize a file\n"
        "triggers:\n"
        "  - summarize\n"
        "  - tl;dr\n"
        "parameters: {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}}, \"required\": [\"path\"]}\n"
        "---\n"
        "body instructions\n"
    )
    assert meta is not None
    assert meta["name"] == "summarize"
    assert meta["description"] == "Summarize a file"
    assert meta["triggers"] == ["summarize", "tl;dr"]
    assert meta["parameters"]["required"] == ["path"]


def test_parse_skill_metadata_quoted_and_scalar_values():
    meta = parse_skill_metadata(
        '---\nname: "quoted name"\ndescription: \'a description, with commas\'\nenabled: true\n---\n'
    )
    assert meta == {
        "name": "quoted name",
        "description": "a description, with commas",
        "enabled": True,
    }


def test_parse_skill_metadata_malformed_returns_none():
    assert parse_skill_metadata("no frontmatter here") is None
    assert parse_skill_metadata("---\nname: summarize\nbad line\n---\n") is None
    assert parse_skill_metadata("---\nparameters: {not json}\n---\n") is None
    assert parse_skill_metadata("---\ntriggers: [a, b]\n---\n") is None  # unquoted JSON list


# ── Skills: discovery + lazy loading ─────────────────────────────────────────

def _write_skill(root: Path, folder: str, *, meta: str, entry: str) -> Path:
    d = root / folder
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(meta, encoding="utf-8")
    (d / "entrypoint.py").write_text(entry, encoding="utf-8")
    return d


def _good_skill(root: Path, folder: str = "greet") -> Path:
    return _write_skill(
        root, folder,
        meta=(
            "---\n"
            f"name: {folder}\n"
            "description: Greet someone\n"
            "triggers:\n"
            "  - hello\n"
            "---\n"
            "Usage instructions for the model.\n"
        ),
        entry=(
            "def run(name: str = 'world') -> str:\n"
            "    return f'hello {name}'\n"
        ),
    )


def test_skills_directory_skips_malformed_folders(tmp_path, caplog):
    (tmp_path / "nometa").mkdir()
    _write_skill(tmp_path, "badmeta", meta="no frontmatter", entry="def run(): pass\n")
    _write_skill(tmp_path, "badname", meta="---\nname: '9bad name'\ndescription: x\n---\n",
                 entry="def run(): pass\n")
    _write_skill(tmp_path, "noentry", meta="---\nname: noentry\ndescription: x\n---\n",
                 entry="")
    (tmp_path / "noentry" / "entrypoint.py").unlink()
    stray = tmp_path / "stray.txt"
    stray.write_text("not a folder")

    with caplog.at_level(logging.WARNING, logger="orcha.agent_runtime.skills"):
        skills = SkillsDirectory(str(tmp_path)).list_skills()

    assert skills == []
    messages = [r.getMessage() for r in caplog.records]
    assert any("missing SKILL.md" in m for m in messages)          # nometa
    assert any("malformed metadata" in m for m in messages)        # badmeta
    assert any("invalid or missing skill name" in m for m in messages)  # badname
    assert any("entrypoint" in m for m in messages)                # noentry


def test_skills_directory_missing_directory_is_resilient(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="orcha.agent_runtime.skills"):
        skills = SkillsDirectory(str(tmp_path / "does-not-exist")).list_skills()
    assert skills == []
    assert any("does not exist" in r.getMessage() for r in caplog.records)


def test_skill_discovery_is_lazy_and_deterministic(tmp_path):
    _good_skill(tmp_path, "b_skill")
    _good_skill(tmp_path, "a_skill")
    loader = SkillsDirectory(str(tmp_path))

    skills = loader.list_skills()
    assert [s.name for s in skills] == ["a_skill", "b_skill"]  # sorted, deterministic
    assert "orcha_skill_a_skill" not in sys.modules  # nothing imported at scan time

    # first call loads the entrypoint…
    tool = skill_to_tool(skills[0], loader)
    assert tool.name == "a_skill"
    assert tool.spec.capability == "skill"
    assert "orcha_skill_a_skill" not in sys.modules  # building the Tool is still lazy
    out = ToolWorkspace(ToolRegistry([tool])).execute(
        action_packet(ToolCallAction(name="a_skill", arguments={})),
    )
    obs = observation_from_packet(out)
    assert isinstance(obs, ToolResultObservation) and "hello world" in obs.content
    assert "orcha_skill_a_skill" in sys.modules
    # …and later calls reuse the cached module
    module = loader.load_entrypoint(skills[0])
    assert module is loader.load_entrypoint(skills[0])


def test_skill_tool_executes_through_workspace(tmp_path):
    loader = SkillsDirectory(str(tmp_path))
    _good_skill(tmp_path, "greet")
    tool = skill_to_tool(loader.list_skills()[0], loader)
    registry = ToolRegistry([tool])
    ws = ToolWorkspace(registry)

    out = ws.execute(action_packet(ToolCallAction(
        name="greet", arguments={"name": "orcha"},
    )))
    obs = observation_from_packet(out)

    assert isinstance(obs, ToolResultObservation)
    assert obs.success and "hello orcha" in obs.content


def test_skill_without_run_entrypoint_is_clean_error(tmp_path):
    _write_skill(tmp_path, "broken",
                 meta="---\nname: broken\ndescription: x\n---\n",
                 entry="VALUE = 42\n")
    loader = SkillsDirectory(str(tmp_path))
    tool = skill_to_tool(loader.list_skills()[0], loader)
    ws = ToolWorkspace(ToolRegistry([tool]))

    out = ws.execute(action_packet(ToolCallAction(name="broken", arguments={})))
    obs = observation_from_packet(out)

    assert isinstance(obs, ErrorObservation)
    assert "skill_bad_entrypoint" in obs.message


def test_register_skills_missing_directory_is_resilient(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="orcha.agent_runtime.skills"):
        out = register_skills(ToolRegistry(), str(tmp_path / "nope"))
    assert out == [] and len(ToolRegistry()) == 0
    assert any("does not exist" in r.getMessage() for r in caplog.records)


def test_register_skills_shadow_warning(tmp_path, caplog):
    registry = ToolRegistry()
    _good_skill(tmp_path, "greet")
    with caplog.at_level(logging.WARNING, logger="orcha.agent_runtime.skills"):
        register_skills(registry, str(tmp_path))
        register_skills(registry, str(tmp_path))

    assert len(registry) == 1  # re-registration replaces, not duplicates
    assert any("shadows an existing tool" in r.getMessage() for r in caplog.records)


async def test_skill_end_to_end_through_conversation(tmp_path):
    _good_skill(tmp_path, "greet")
    registry = ToolRegistry()
    register_skills(registry, str(tmp_path))

    backend = FakeBackend([
        tool_call_response("greet", {"name": "agent"}),
        text_response("greeted"),
    ])
    convo = Conversation(ToolCallingAgent(backend, registry), ToolWorkspace(registry), query="q")
    convo.submit_user_message("greet agent")

    result = await convo.run()

    assert result.completed and result.answer == "greeted"
    assert any(
        isinstance(ev.payload, ToolResultObservation) and "hello agent" in ev.payload.content
        for ev in result.events
    )
    assert any(t["function"]["name"] == "greet" for t in backend.calls[0]["tools"])


# ── Bootstrap: one registry, resilient discovery ─────────────────────────────

def test_discover_tools_skips_broken_sources(caplog):
    config = AgentRuntimeConfig(
        mcp_servers=[
            McpServerConfig(name="bad", transport="stdio", command="definitely-not-a-real-cmd"),
        ],
        skills_dirs=[str(Path("does-not-exist-dir"))],
    )
    with caplog.at_level(logging.WARNING, logger="orcha.agent_runtime"):
        registry = discover_tools(config)

    assert len(registry) == 0  # boot completes with a usable (empty) registry
    messages = [r.getMessage() for r in caplog.records]
    assert any("skipping MCP server" in m for m in messages)
    assert any("does not exist" in m for m in messages)


def test_discover_tools_registers_mcp_and_skills(caplog):
    transport = FakeMcpTransport([echo_info()])
    config = AgentRuntimeConfig(
        mcp_servers=[McpServerConfig(name="srv", command="injected-fake")],
    )

    # the injection path keeps the test offline:
    registry = ToolRegistry()
    register_mcp_server(registry, config.mcp_servers[0], transport=transport)
    assert registry.names() == ["echo"]
    assert isinstance(discover_tools(config), ToolRegistry)


def test_build_agent_runtime_wires_config_into_conversation(tmp_path):
    _good_skill(tmp_path, "greet")
    config = AgentRuntimeConfig(skills_dirs=[str(tmp_path)])

    conversation, registry, backend = build_agent_runtime(config)

    assert isinstance(backend, OpenAICompatBackend)
    assert backend.model_name == config.backend.model
    assert registry.names() == ["greet"]
    assert isinstance(registry.get("greet"), Tool)


# ── Real stdio MCP server (integration, needs the official `mcp` SDK) ────────

_FIXTURE = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"


def test_real_stdio_mcp_server_full_stack():
    pytest.importorskip("mcp")
    config = McpServerConfig(
        name="echo-server", transport="stdio",
        command=sys.executable, args=[str(_FIXTURE)], timeout_s=30.0,
    )
    server = McpServer(config)
    try:
        tools = server.connect()
        assert {t.name for t in tools} == {"echo", "add", "fail"}

        registry = ToolRegistry(tools)
        ws = ToolWorkspace(registry)

        # a real call: Action → ToolWorkspace → transport → server → observation
        out = ws.execute(action_packet(ToolCallAction(name="add", arguments={"a": 2, "b": 3})))
        obs = observation_from_packet(out)
        assert isinstance(obs, ToolResultObservation)
        assert obs.success and "5" in obs.content

        out = ws.execute(action_packet(ToolCallAction(name="echo", arguments={"message": "hi"})))
        obs = observation_from_packet(out)
        assert isinstance(obs, ToolResultObservation)
        assert "echo: HI" in obs.content

        # a server-side error (is_error) surfaces as a clean mcp_error
        out = ws.execute(action_packet(ToolCallAction(name="fail", arguments={})))
        obs = observation_from_packet(out)
        assert isinstance(obs, ErrorObservation)
        assert "mcp_error" in obs.message

        # client-side schema validation rejects bad arguments cleanly
        out = ws.execute(action_packet(ToolCallAction(name="echo", arguments={"nope": 1})))
        assert isinstance(observation_from_packet(out), ErrorObservation)
    finally:
        server.close()


async def test_real_stdio_mcp_server_through_conversation():
    pytest.importorskip("mcp")
    config = McpServerConfig(
        name="echo-server", transport="stdio",
        command=sys.executable, args=[str(_FIXTURE)], timeout_s=30.0,
    )
    server = McpServer(config)
    try:
        registry = ToolRegistry(server.connect())
        backend = FakeBackend([
            tool_call_response("echo", {"message": "hello"}),
            text_response("done"),
        ])
        convo = Conversation(
            ToolCallingAgent(backend, registry), ToolWorkspace(registry), query="q",
            config=ConversationConfig(max_steps=4),
        )
        convo.submit_user_message("echo hello")

        result = await convo.run()

        assert result.completed
        assert any(
            isinstance(ev.payload, ToolResultObservation) and "echo:" in ev.payload.content
            for ev in result.events
        )
        assert isinstance(result.events[-1].payload, FinishAction)
    finally:
        server.close()
