"""
Tests for the Prompt 2 layer of orcha.agent_runtime:

- Tool system: Tool wraps existing Orcha ToolSpecs (capabilities machinery);
  ToolRegistry; build_default_tools (run_command / read_file / write_file /
  query_experts); ToolWorkspace executes real calls and never crashes on
  unknown tools or schema mismatches.
- Backend seam: ModelBackend protocol; GenerationConfig streaming policy
  (conservative: tool rounds are never streamed unless the backend declares
  support); OpenAICompatBackend parsing + SSE accumulation + malformed-JSON
  ToolCallParseError.
- ToolCallingAgent: valid tool call end-to-end; one repair round for
  malformed tool-call JSON then clean failure; multiple tool calls per turn
  cost one model call; reasoning-token usage folds into the SAME budget
  accounting as ordinary tokens.
- Capability swap: a different backend profile (no reasoning reporting, no
  streaming) requires config-only changes.
"""
from typing import Any, Dict, List, Optional, Sequence

import pytest

from orcha import PacketKind, OrchaPacket
from orcha.agent_runtime import (
    AgentConfig, Conversation, ConversationConfig, ConversationResult,
    ErrorObservation, EventLog, FinishAction,
    GenerationConfig, MessageAction, ModelBackend, ModelResponse,
    StubWorkspace, Tool, ToolCallAction, ToolCallingAgent, ToolRegistry,
    ToolResultObservation, ToolWorkspace, TokenUsage, ToolCall,
    ToolCallParseError, action_packet, build_default_tools,
    observation_from_packet,
)
from orcha.agent_runtime.backends import (
    OpenAICompatBackend, OpenAICompatBackendConfig,
)
from orcha.capabilities.base import READ, SAFE, spec
from orcha.core.packets import ExpertSlot


# ── Test doubles ─────────────────────────────────────────────────────────────

def tool_call_response(
    name: str, arguments: Dict[str, Any], *,
    tc_id: str = "call_1",
    usage: Optional[TokenUsage] = None,
) -> ModelResponse:
    return ModelResponse(
        tool_calls=[ToolCall(id=tc_id, name=name, arguments=arguments)],
        usage=usage or TokenUsage(prompt_tokens=5, completion_tokens=7),
    )


def text_response(
    content: str, *, usage: Optional[TokenUsage] = None,
) -> ModelResponse:
    return ModelResponse(
        content=content,
        usage=usage or TokenUsage(prompt_tokens=5, completion_tokens=7),
    )


class FakeBackend(ModelBackend):
    """
    Scripted backend. Pops responses in order; once empty, always answers
    with plain text "done". Records every call (messages, tools, config)
    so tests can assert what the agent offered the model.
    """

    def __init__(
        self, responses: Optional[Sequence[ModelResponse]] = None, *,
        supports_streaming_tool_calls: bool = False,
        reports_reasoning_tokens: bool = False,
        model: str = "fake-model",
    ) -> None:
        self._responses = list(responses or [])
        self.calls: List[Dict[str, Any]] = []
        self.supports_streaming_tool_calls = supports_streaming_tool_calls
        self.reports_reasoning_tokens = reports_reasoning_tokens
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_streaming_tool_calls(self) -> bool:
        return self._streams_tools

    @supports_streaming_tool_calls.setter
    def supports_streaming_tool_calls(self, value: bool) -> None:
        self._streams_tools = value

    @property
    def reports_reasoning_tokens(self) -> bool:
        return self._reports_reasoning

    @reports_reasoning_tokens.setter
    def reports_reasoning_tokens(self, value: bool) -> None:
        self._reports_reasoning = value

    def complete(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[GenerationConfig] = None,
    ) -> ModelResponse:
        self.calls.append({
            "messages": list(messages),
            "tools": tools,
            "config": config,
        })
        if not self._responses:
            return text_response("done")
        return self._responses.pop(0)


class RepairingFakeBackend(ModelBackend):
    """Raises ToolCallParseError the first ``parse_errors`` calls, then
    behaves like a FakeBackend fed by ``responses``."""

    def __init__(
        self, parse_errors: int, responses: Sequence[ModelResponse], **kwargs: Any,
    ) -> None:
        self._inner = FakeBackend(responses, **kwargs)
        self.parse_errors = parse_errors
        self.calls = 0
        self.call_log: List[Dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def supports_streaming_tool_calls(self) -> bool:
        return self._inner.supports_streaming_tool_calls

    @property
    def reports_reasoning_tokens(self) -> bool:
        return self._inner.reports_reasoning_tokens

    def complete(self, messages, tools=None, config=None) -> ModelResponse:
        self.calls += 1
        self.call_log.append({"messages": list(messages), "tools": tools, "config": config})
        if self.calls <= self.parse_errors:
            raise ToolCallParseError("malformed JSON arguments", raw="{bad")
        return self._inner.complete(messages, tools, config)


class FakeSelector:
    """Duck-typed stand-in for orcha.orchestration.ExpertSelector."""

    def __init__(self, slots: List[ExpertSlot]) -> None:
        self.slots = slots

    def select(self, packet: OrchaPacket) -> OrchaPacket:
        return packet.fork(
            PacketKind.SELECTION, selected_experts=[s.model_dump() for s in self.slots],
        )


# ── Tool system ──────────────────────────────────────────────────────────────

def test_tool_wraps_spec_and_exposes_schema():
    t = Tool(spec(
        "echo_tool", "Echo the text back.", {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        lambda text: {"echo": text}, permissions=[READ], safety_level=SAFE,
    ))
    assert t.name == "echo_tool"
    assert t.parameters["required"] == ["text"]
    schema = t.schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo_tool"


def test_tool_run_returns_observation_and_never_raises():
    t = Tool(spec(
        "echo_tool", "Echo text.", {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        lambda text: {"echo": text}, permissions=[READ], safety_level=SAFE,
    ))
    obs = t.run({"text": "hi"})
    assert isinstance(obs, ToolResultObservation)
    assert obs.success and "hi" in obs.content

    # schema mismatch → ErrorObservation, no exception
    bad = t.run({"missing": 1})
    assert isinstance(bad, ErrorObservation)


def test_tool_registry_flat_registration_and_schemas():
    reg = ToolRegistry()
    reg.register(Tool(spec(
        "a_tool", "A.", {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        lambda x: x, permissions=[READ], safety_level=SAFE,
    )))
    reg.register_spec(spec(
        "b_tool", "B.", {"type": "object", "properties": {"y": {"type": "string"}}, "required": ["y"]},
        lambda y: y, permissions=[READ], safety_level=SAFE,
    ))
    assert set(reg.names()) == {"a_tool", "b_tool"}
    assert len(reg.schemas()) == 2
    assert reg.has("a_tool") and not reg.has("nope")
    assert len(reg) == 2

    with pytest.raises(ValueError):
        reg.register(Tool(spec("", "no name", {"type": "object", "properties": {}}, lambda: 1)))


def test_build_default_tools_include_the_shipped_set(tmp_path):
    tools = build_default_tools([str(tmp_path)], selector=None)
    names = {t.name for t in tools}
    assert {"run_command", "read_file", "write_file"} <= names
    # Loop-guard contract (Prompt 9): without an expert selector the
    # query_experts tool is NOT registered — the model can never loop on it.
    assert "query_experts" not in names


def test_default_tools_read_write_round_trip_through_workspace(tmp_path):
    registry = ToolRegistry(build_default_tools([str(tmp_path)]))
    ws = ToolWorkspace(registry)

    target = tmp_path / "notes.txt"
    out = ws.execute(action_packet(ToolCallAction(
        name="write_file",
        arguments={"path": str(target), "content": "hello prompt 2"},
    )))
    obs = observation_from_packet(out)
    assert isinstance(obs, ToolResultObservation)
    assert obs.success
    assert target.read_text() == "hello prompt 2"

    out = ws.execute(action_packet(ToolCallAction(
        name="read_file",
        arguments={"path": str(target)},
    )))
    obs = observation_from_packet(out)
    assert isinstance(obs, ToolResultObservation)
    assert obs.success and "hello prompt 2" in obs.content


def test_run_command_tool_executes_inside_workspace(tmp_path):
    registry = ToolRegistry(build_default_tools([str(tmp_path)]))
    ws = ToolWorkspace(registry)

    out = ws.execute(action_packet(ToolCallAction(
        name="run_command", arguments={"command": "echo orcha-agent-runtime"},
    )))
    obs = observation_from_packet(out)
    assert isinstance(obs, ToolResultObservation)
    assert obs.success and "orcha-agent-runtime" in obs.content


def test_workspace_unknown_tool_and_schema_mismatch_are_clean_errors(tmp_path):
    registry = ToolRegistry(build_default_tools([str(tmp_path)]))
    ws = ToolWorkspace(registry)

    out = ws.execute(action_packet(ToolCallAction(
        name="does_not_exist", arguments={},
    )))
    obs = observation_from_packet(out)
    assert isinstance(obs, ErrorObservation)
    assert "unknown tool" in obs.message

    # missing required argument → validation error, not a crash
    out = ws.execute(action_packet(ToolCallAction(
        name="write_file", arguments={},
    )))
    obs = observation_from_packet(out)
    assert isinstance(obs, ErrorObservation)
    assert "rejected arguments" in obs.message


def test_query_experts_tool_routes_through_selector(tmp_path):
    slots = [ExpertSlot(name="llama3.2:3b", domain="general", description="default model", score=0.9)]
    registry = ToolRegistry(build_default_tools([str(tmp_path)], selector=FakeSelector(slots)))
    ws = ToolWorkspace(registry)

    out = ws.execute(action_packet(ToolCallAction(
        name="query_experts", arguments={"query": "research pipelines"},
    )))
    obs = observation_from_packet(out)
    assert isinstance(obs, ToolResultObservation)
    assert obs.success and "llama3.2:3b" in obs.content


def test_query_experts_absent_without_selector(tmp_path):
    """Loop-guard contract (Prompt 9): without an expert selector the tool
    is not registered at all, so the model cannot even attempt the call
    that used to degrade to a repeated "no expert selector" note."""
    registry = ToolRegistry(build_default_tools([str(tmp_path)], selector=None))
    assert "query_experts" not in registry.names()

    ws = ToolWorkspace(registry)
    out = ws.execute(action_packet(ToolCallAction(
        name="query_experts", arguments={"query": "anything"},
    )))
    obs = observation_from_packet(out)
    assert isinstance(obs, ErrorObservation)
    assert "unknown tool" in obs.message


# ── GenerationConfig streaming policy ────────────────────────────────────────

def test_streaming_policy_is_conservative_for_tool_rounds():
    cfg = GenerationConfig(stream=True)
    # Tool rounds on a backend that cannot stream them → never streamed.
    assert cfg.should_stream(round_requests_tools=True, backend_streams_tools=False) is False
    # Tool rounds on a capable backend follow stream_tool_calls (default off).
    assert cfg.should_stream(round_requests_tools=True, backend_streams_tools=True) is False
    # Free-text rounds follow stream.
    assert cfg.should_stream(round_requests_tools=False, backend_streams_tools=False) is True

    tool_cfg = GenerationConfig(stream=True, stream_tool_calls=True)
    assert tool_cfg.should_stream(round_requests_tools=True, backend_streams_tools=True) is True
    assert tool_cfg.should_stream(round_requests_tools=True, backend_streams_tools=False) is False


# ── OpenAICompatBackend: parsing (via injected fake transport) ──────────────

class FakeResponse:
    def __init__(self, status: int, text: str, lines: Optional[List[str]] = None) -> None:
        self.status_code = status
        self.text = text
        self._lines = lines or [text]

    def read(self) -> bytes:
        return self.text.encode("utf-8")

    def iter_lines(self):
        return iter(self._lines)


class FakeTransportClient:
    """Stands in for httpx.Client; records requests for assertions."""

    def __init__(self) -> None:
        self.posts: List[Dict[str, Any]] = []
        self.streams: List[Dict[str, Any]] = []
        self.responses: List[FakeResponse] = []
        self.stream_responses: List[FakeResponse] = []

    def post(self, url: str, *, json: Any = None, headers: Any = None) -> FakeResponse:
        self.posts.append({"url": url, "json": json, "headers": headers})
        return self.responses.pop(0)

    def stream(self, method: str, url: str, *, json: Any = None, headers: Any = None):
        self.streams.append({"url": url, "json": json, "headers": headers})
        return _FakeStreamCtx(self.stream_responses.pop(0))

    def close(self) -> None:
        pass


class _FakeStreamCtx:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response

    def __enter__(self):
        return self._response

    def __exit__(self, *exc):
        return False


def _compat_backend(client: FakeTransportClient, **cfg: Any) -> OpenAICompatBackend:
    return OpenAICompatBackend(
        OpenAICompatBackendConfig(base_url="http://fake/v1", model="fake:latest", **cfg),
        client=client,
    )


def test_openai_compat_parses_tool_calls_with_json_string_arguments():
    client = FakeTransportClient()
    client.responses.append(FakeResponse(200, '{"choices":[{"message":{"role":"assistant",'
        '"content":"","tool_calls":[{"id":"c1","type":"function","function":'
        '{"name":"read_file","arguments":"{\\"path\\": \\"/a.txt\\"}"}}]},'
        '"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":20}}'))
    backend = _compat_backend(client)

    resp = backend.complete(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
        config=GenerationConfig(),
    )

    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert tc.id == "c1" and tc.name == "read_file"
    assert tc.arguments == {"path": "/a.txt"}
    assert resp.finish_reason == "tool_calls"
    assert resp.usage.prompt_tokens == 10 and resp.usage.completion_tokens == 20
    # conservative policy: tool round, backend cannot stream → no stream
    assert client.posts[0]["json"]["stream"] is False


def test_openai_compat_maps_reasoning_tokens_when_reported():
    client = FakeTransportClient()
    client.responses.append(FakeResponse(200, '{"choices":[{"message":{"role":"assistant","content":"final answer"},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":5,"completion_tokens":3,"reasoning_tokens":42}}'))
    backend = _compat_backend(client, reports_reasoning_tokens=True)

    resp = backend.complete([{"role": "user", "content": "hi"}], config=GenerationConfig())

    assert resp.usage.reasoning_tokens == 42
    assert resp.usage.step_tokens == 5 + 3 + 42


def test_openai_compat_streams_only_when_policy_allows():
    # Backend that cannot stream tool rounds: even stream=True → no stream.
    client = FakeTransportClient()
    client.responses.append(FakeResponse(200, '{"choices":[{"message":{"role":"assistant","content":"x"},"finish_reason":"stop"}],"usage":{}}'))
    backend = _compat_backend(client)

    backend.complete([{"role": "user", "content": "hi"}], tools=[{"type": "function"}],
                     config=GenerationConfig(stream=True))
    assert client.posts[0]["json"]["stream"] is False  # tool round, no support

    # Capable backend + stream_tool_calls → tool round streams.
    client2 = FakeTransportClient()
    client2.stream_responses.append(FakeResponse(200, ""))
    backend2 = _compat_backend(client2, supports_streaming_tool_calls=True)
    backend2.complete([{"role": "user", "content": "hi"}], tools=[{"type": "function"}],
                      config=GenerationConfig(stream=True, stream_tool_calls=True))
    assert len(client2.streams) == 1
    assert client2.streams[0]["json"]["stream"] is True


def test_openai_compat_streaming_accumulates_delta_chunks():
    client = FakeTransportClient()
    sse = [
        "data: {\"choices\":[{\"delta\":{\"content\":\"hello \"}}]}",
        "data: {\"choices\":[{\"delta\":{\"content\":\"world\"}}]}",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c9\",\"function\":{\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\"}}]}}]}",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"/a.txt\\\"}\"}}]}}]}",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}",
        "data: [DONE]",
    ]
    client.stream_responses.append(FakeResponse(200, "", lines=sse))
    backend = _compat_backend(client, supports_streaming_tool_calls=True)

    resp = backend.complete(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function"}],
        config=GenerationConfig(stream_tool_calls=True),
    )

    assert resp.content == "hello world"
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert tc.id == "c9" and tc.name == "read"
    assert tc.arguments == {"path": "/a.txt"}


def test_openai_compat_malformed_arguments_raise_parse_error():
    client = FakeTransportClient()
    client.responses.append(FakeResponse(200, '{"choices":[{"message":{"role":"assistant",'
        '"tool_calls":[{"id":"c1","function":{"name":"read_file","arguments":"{not json"}}]},'
        '"finish_reason":"tool_calls"}],"usage":{}}'))
    backend = _compat_backend(client)

    with pytest.raises(ToolCallParseError):
        backend.complete([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])


def test_openai_compat_error_status_raises_cleanly():
    client = FakeTransportClient()
    client.responses.append(FakeResponse(400, '{"error":{"message":"model not found"}}'))
    backend = _compat_backend(client)

    with pytest.raises(RuntimeError, match="model not found"):
        backend.complete([{"role": "user", "content": "hi"}])


# ── The ToolCallingAgent ─────────────────────────────────────────────────────

def _registry(tmp_path) -> ToolRegistry:
    return ToolRegistry(build_default_tools([str(tmp_path)]))


async def test_valid_tool_call_executes_end_to_end(tmp_path):
    registry = _registry(tmp_path)
    backend = FakeBackend([
        tool_call_response("write_file", {"path": str(tmp_path / "out.txt"), "content": "hello"}),
        text_response("wrote it"),
    ])
    agent = ToolCallingAgent(backend, registry)
    convo = Conversation(
        agent, ToolWorkspace(registry), query="q",
        agent_config=AgentConfig(generation=GenerationConfig(stream=True)),
    )
    convo.submit_user_message("write a file")

    result: ConversationResult = await convo.run()

    assert (tmp_path / "out.txt").read_text() == "hello"
    assert [type(ev.payload).__name__ for ev in result.events] == [
        "UserMessageObservation",
        "ToolCallAction", "ToolResultObservation",
        "FinishAction",
    ]
    assert result.completed
    assert result.answer == "wrote it"
    assert len(backend.calls) == 2
    # the agent offered the model the registry schemas
    assert any(t["function"]["name"] == "write_file" for t in backend.calls[0]["tools"])
    # usage from the backend landed on the events (budget accounting)
    assert result.tokens_used >= 12
    assert result.events[1].tokens == 12  # 5 prompt + 7 completion


async def test_malformed_tool_call_json_repairs_once_then_fails_cleanly(tmp_path):
    registry = _registry(tmp_path)
    backend = RepairingFakeBackend(parse_errors=1, responses=[
        tool_call_response("read_file", {"path": str(tmp_path / "r.txt")}),
        text_response("recovered"),
    ])
    agent = ToolCallingAgent(backend, registry)
    convo = Conversation(agent, ToolWorkspace(registry), query="q")
    convo.submit_user_message("go")

    result = await convo.run()

    # one repair round happened (2 calls: repair + tool), then a final text
    # call finished the run — 3 model calls total
    assert backend.calls == 3
    # the repair hint was appended after the initial user turn
    assert "malformed" in backend.call_log[1]["messages"][2]["content"].lower()
    assert result.completed

    # persistent malformation → clean ErrorAction, run terminates with error
    broken = RepairingFakeBackend(parse_errors=2, responses=[text_response("never reached")])
    convo2 = Conversation(ToolCallingAgent(broken, _registry(tmp_path)), StubWorkspace(), query="q")
    convo2.submit_user_message("go")
    result2 = await convo2.run()
    assert broken.calls == 2
    assert result2.terminated_by == "error"
    assert result2.completed is False
    assert isinstance(result2.events[-2].payload, ErrorObservation)


async def test_multiple_tool_calls_per_turn_cost_one_model_call(tmp_path):
    registry = _registry(tmp_path)
    backend = FakeBackend([
        ModelResponse(tool_calls=[
            ToolCall(id="a", name="write_file", arguments={"path": str(tmp_path / "a.txt"), "content": "A"}),
            ToolCall(id="b", name="write_file", arguments={"path": str(tmp_path / "b.txt"), "content": "B"}),
        ], usage=TokenUsage(prompt_tokens=10, completion_tokens=10)),
        text_response("wrote both"),
    ])
    agent = ToolCallingAgent(backend, registry)
    convo = Conversation(agent, ToolWorkspace(registry), query="q")
    convo.submit_user_message("write two files")

    result = await convo.run()

    assert (tmp_path / "a.txt").read_text() == "A"
    assert (tmp_path / "b.txt").read_text() == "B"
    assert len(backend.calls) == 2                    # one batch + one final answer
    assert len(result.events) == 6                    # user, 2×(action+obs), finish
    assert result.completed


async def test_intermediate_message_then_final_answer(tmp_path):
    backend = FakeBackend([
        text_response("let me check that"),
        tool_call_response("read_file", {"path": str(tmp_path / "nope.txt")}),
        text_response("it is missing"),
    ])
    agent = ToolCallingAgent(backend, _registry(tmp_path))
    convo = Conversation(agent, ToolWorkspace(_registry(tmp_path)), query="q")
    convo.submit_user_message("check the file")

    result = await convo.run()

    kinds = [ev.payload.kind for ev in result.events]
    assert kinds == ["user_message", "message", "tool_call", "error", "finish"]
    assert isinstance(result.events[1].payload, MessageAction)
    assert isinstance(result.events[-1].payload, FinishAction)
    assert result.events[-1].payload.content == "it is missing"


# ── Reasoning-token accounting ───────────────────────────────────────────────

async def test_reasoning_tokens_fold_into_the_same_budget(tmp_path):
    registry = _registry(tmp_path)
    backend = FakeBackend([
        tool_call_response("read_file", {"path": str(tmp_path / "x.txt")},
                           usage=TokenUsage(prompt_tokens=4, completion_tokens=6, reasoning_tokens=50)),
    ], reports_reasoning_tokens=True)
    convo = Conversation(
        ToolCallingAgent(backend, registry), ToolWorkspace(registry), query="q",
        config=ConversationConfig(max_tokens=60),
    )
    convo.submit_user_message("go")

    result = await convo.run()

    # tokens (10) + reasoning (50) = 60 → max_tokens cutoff fired
    assert result.terminated_by == "max_tokens"
    assert result.reasoning_tokens_used == 50
    assert result.tokens_used >= 60
    assert result.events[1].reasoning_tokens == 50

    # reasoning tokens survive a deterministic replay
    replayed = EventLog.replay(result.events)
    assert replayed == result.log_state
    assert replayed.reasoning_tokens_used == result.log_state.reasoning_tokens_used


async def test_backend_without_usage_reporting_needs_no_special_casing(tmp_path):
    registry = _registry(tmp_path)
    backend = FakeBackend([
        tool_call_response("read_file", {"path": str(tmp_path / "y.txt")}, usage=TokenUsage()),
        text_response("done", usage=TokenUsage()),
    ])  # reports nothing: usage = zeros
    convo = Conversation(
        ToolCallingAgent(backend, registry), ToolWorkspace(registry), query="q",
        config=ConversationConfig(max_tokens=10_000),
    )
    convo.submit_user_message("go")

    result = await convo.run()

    assert result.completed
    assert result.reasoning_tokens_used == 0
    # text actions still carry a deterministic content-based estimate
    assert result.events[-1].tokens >= 1


# ── The capability swap: backend-agnostic by contract ────────────────────────

async def test_swapping_backend_profiles_changes_only_config(tmp_path):
    registry = _registry(tmp_path)
    script = [
        tool_call_response("write_file", {"path": str(tmp_path / "s.txt"), "content": "swap"}),
        text_response("done"),
    ]

    rich = FakeBackend(
        [tool_call_response("write_file", {"path": str(tmp_path / "s.txt"), "content": "swap"},
                            usage=TokenUsage(prompt_tokens=3, completion_tokens=3, reasoning_tokens=9)),
         text_response("done", usage=TokenUsage(prompt_tokens=3, completion_tokens=2, reasoning_tokens=6))],
        supports_streaming_tool_calls=True,
        reports_reasoning_tokens=True,
    )
    poor = FakeBackend(script, supports_streaming_tool_calls=False, reports_reasoning_tokens=False)

    # Identical agent/workspace/conversation wiring — only the backend object
    # and its AgentConfig differ.
    async def make_run(backend: ModelBackend, stream_tool_calls: bool = False) -> ConversationResult:
        convo = Conversation(
            ToolCallingAgent(backend, registry),
            ToolWorkspace(registry),
            query="q",
            agent_config=AgentConfig(generation=GenerationConfig(stream=True, stream_tool_calls=stream_tool_calls)),
        )
        convo.submit_user_message("go")
        return await convo.run()

    run_rich = await make_run(rich, stream_tool_calls=True)
    run_poor = await make_run(poor)

    # Same conversation shape regardless of backend capabilities.
    assert [ev.payload.kind for ev in run_rich.events] == [ev.payload.kind for ev in run_poor.events]
    assert run_rich.completed and run_poor.completed
    assert (tmp_path / "s.txt").read_text() == "swap"

    # The rich backend's reasoning tokens are metered into the log; the poor
    # backend's are absent (= zero), no special-casing anywhere.
    assert run_rich.reasoning_tokens_used == 15
    assert run_poor.reasoning_tokens_used == 0
    assert run_rich.tokens_used > run_poor.tokens_used

    # The poor backend never had to change its call shape.
    assert all(c["tools"] is not None for c in poor.calls)


# ── Streaming-flag respect (fake backend observes the config) ───────────────

async def test_fake_backend_sees_streaming_flags_respected_by_agent(tmp_path):
    registry = _registry(tmp_path)
    backend = FakeBackend([
        tool_call_response("read_file", {"path": str(tmp_path / "z.txt")}),
        text_response("z"),
    ], supports_streaming_tool_calls=False)
    agent = ToolCallingAgent(backend, registry)
    convo = Conversation(agent, ToolWorkspace(registry), query="q",
                         agent_config=AgentConfig(generation=GenerationConfig(stream=True)))
    convo.submit_user_message("go")
    await convo.run()

    gen = backend.calls[0]["config"]
    assert gen.should_stream(round_requests_tools=True, backend_streams_tools=False) is False
    gen2 = backend.calls[1]["config"]
    assert gen2.should_stream(round_requests_tools=False, backend_streams_tools=False) is True
