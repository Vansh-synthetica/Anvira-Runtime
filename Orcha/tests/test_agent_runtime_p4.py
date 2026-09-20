"""
Tests for the Prompt 4 layer of orcha.agent_runtime:

- Working memory: StepMemory projection (action/observation, timing,
  token in/out, optional reasoning share); windowed summarization bounded
  by a token budget on 20+ step sessions — charging reasoning tokens the
  SAME way whether or not the backend reports them.
- Long-term memory: durable facts (Conversation.remember →
  FactObservation) excluded from the turn transcript, injected into every
  step's context, deduplicated, replay-safe.
- Bounded context end-to-end: every message the model receives contains
  exactly the windowed memory section the log prefix dictates, over 25
  steps — with and without backend-reported reasoning tokens.
- Budget via EXISTING machinery: BudgetPlanner decides on the SAME
  BudgetState (wall-clock + completed-iteration accounting); a
  budget-exceeded loop ends in a clean FinishAction, never without end.
- Reuse, not reinvention: real ExpertSelector routing through
  query_experts; the capability sandbox (terminal deny-list, path
  scoping) proven inside the agent loop.
"""
from typing import Any, Dict, List, Optional, Sequence

import pytest

from orcha import BudgetState
from orcha.agent_runtime import (
    Agent, AgentConfig, Conversation, ConversationConfig, ErrorObservation,
    EventLog, FactObservation, FinishAction, ModelBackend, ModelResponse,
    StubAgent, StubWorkspace, Tool, ToolCallAction, ToolCallingAgent,
    ToolRegistry, ToolResultObservation, ToolWorkspace, TokenUsage, ToolCall,
    action_packet, build_default_tools, observation_from_packet,
    ContextMemory, StepMemory, LongTermMemory, render_memory_section,
)
from orcha.agent_runtime.events import Event, EventKind
from orcha.capabilities.base import DANGEROUS_LEVEL, READ, SAFE, spec
from orcha.experts.mock import load_mock_experts
from orcha.orchestration.selector import ExpertSelector


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
    """Scripted backend; records every call so tests can inspect exactly
    what the agent offered the model. Answers "done" when the script is
    exhausted."""

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
        config: Optional[Any] = None,
    ) -> ModelResponse:
        self.calls.append({
            "messages": list(messages),
            "tools": tools,
            "config": config,
        })
        if not self._responses:
            return text_response("done")
        return self._responses.pop(0)


class AlwaysToolAgent(Agent):
    """Never stops on its own: every step asks for another echo."""

    async def step(self, config: AgentConfig, log_slice: Sequence[Event]) -> ToolCallAction:
        return ToolCallAction(name="echo", arguments={"text": "again"})


def echo_tool() -> Tool:
    return Tool(spec(
        "echo", "Echo the text back.",
        {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        lambda text: {"echo": text}, permissions=[READ], safety_level=SAFE,
    ))


def _step_events(n: int, *, reasoning: int = 5, ts_base: float = 1_000.0,
                 tokens_in: int = 10, tokens_out: int = 10) -> List[Event]:
    """Fabricate ``n`` deterministic (action, observation) event pairs."""
    events: List[Event] = []
    seq = 1
    for i in range(1, n + 1):
        events.append(Event(
            seq=seq, kind=EventKind.ACTION, ts=ts_base + i * 2.0,
            payload=ToolCallAction(name=f"t{i}", arguments={"k": "v"}),
            tokens=tokens_in, reasoning_tokens=reasoning,
        ))
        seq += 1
        events.append(Event(
            seq=seq, kind=EventKind.OBSERVATION, ts=ts_base + i * 2.0 + 0.5,
            payload=ToolResultObservation(content=f"r{i}", success=True),
            tokens=tokens_out,
        ))
        seq += 1
    return events


# ── Short-term memory: the projection ────────────────────────────────────────

def test_step_memory_projects_deterministic_records():
    events = _step_events(3)
    # one failing step: action + ErrorObservation
    events.append(Event(
        seq=7, kind=EventKind.ACTION, ts=1_010.0,
        payload=ToolCallAction(name="boom", arguments={}),
        tokens=4, reasoning_tokens=2,
    ))
    events.append(Event(
        seq=8, kind=EventKind.OBSERVATION, ts=1_010.4,
        payload=ErrorObservation(message="exploded"), tokens=6,
    ))
    # one in-flight action with no observation yet
    events.append(Event(
        seq=9, kind=EventKind.ACTION, ts=1_012.0,
        payload=ToolCallAction(name="pending", arguments={}),
        tokens=3, reasoning_tokens=1,
    ))

    mem = StepMemory.from_events(events)

    assert len(mem) == 5
    rec = mem.records[0]
    assert rec.step_number == 1
    assert rec.action_kind == "tool_call"
    assert rec.action == "tool call: t1({'k': 'v'})"
    assert rec.observation_kind == "tool_result"
    assert rec.observation == "r1"
    assert rec.success is True
    assert rec.tokens_in == 10 and rec.reasoning_tokens == 5 and rec.tokens_out == 10
    assert rec.charge == 25
    # wall-clock timing: observation at +0.5s, next event (the following
    # action) at +1.5s → 1500ms
    assert rec.duration_ms == 1500.0

    failed = mem.records[3]
    assert failed.success is False
    assert failed.observation_kind == "error"
    assert failed.error == "exploded"
    assert failed.charge == 12

    pending = mem.records[4]
    assert pending.observation == ""
    assert pending.duration_ms == 0.0  # no next event to time against

    # deterministic projection: same events, same records
    again = StepMemory.from_events(events)
    assert again.records == mem.records

    # compact serialization carries the full accounting
    dumped = mem.records[0].to_dict()
    assert dumped["step"] == 1 and dumped["reasoning_tokens"] == 5
    assert dumped["tokens_in"] == 10 and dumped["tokens_out"] == 10


def test_render_window_is_bounded_newest_first_and_deterministic():
    mem = StepMemory.from_events(_step_events(25))

    w = mem.render_window(max_context_tokens=200)

    # charge per step = 25 → exactly 8 fit; the oldest 17 fold into a digest
    assert w.count("step ") == 8
    assert w.startswith("step 18:")          # oldest INCLUDED first (chronological)
    assert "step 25:" in w and "step 17:" not in w
    assert ("… 17 earlier step(s) omitted (tokens_in=170, "
            "reasoning=85, tokens_out=170)") in w
    assert mem.render_window(200) == w       # deterministic
    # bounded: longer sessions never grow the window beyond the budget
    big = StepMemory.from_events(_step_events(200))
    assert big.render_window(200).count("step ") == 8


def test_render_window_charges_reasoning_tokens_even_when_not_reported():
    with_reasoning = StepMemory.from_events(_step_events(25, reasoning=5))
    without_reasoning = StepMemory.from_events(_step_events(25, reasoning=0))

    w_with = with_reasoning.render_window(max_context_tokens=200)
    w_without = without_reasoning.render_window(max_context_tokens=200)

    # reasoning counts toward the window: with it, charge=25 (8 fit);
    # without, charge=20 (10 fit) — absence simply means zero, never a
    # special-cased "free" step.
    assert w_with.count("step ") == 8
    assert w_without.count("step ") == 10
    assert "reasoning=85" in w_with            # 17 × 5 folded into the digest
    assert "reasoning=0" in w_without          # 15 × 0
    assert w_without.startswith("step 16:")


def test_render_window_always_keeps_the_newest_step():
    mem = StepMemory.from_events(_step_events(25))
    w = mem.render_window(max_context_tokens=1)
    assert "step 25:" in w
    assert "24 earlier step(s) omitted" in w


def test_render_memory_section_assembles_facts_and_steps():
    events = _step_events(5)
    events.insert(0, Event(
        seq=1, kind=EventKind.OBSERVATION, ts=1.0,
        payload=FactObservation(content="user is a data engineer"),
    ))
    mem = ContextMemory.from_events(events)

    section = render_memory_section(mem, max_context_tokens=200)
    assert section.startswith("[Durable facts]\n- user is a data engineer")
    assert "\n\n[Recent steps]\n" in section
    assert "step 5:" in section

    # empty / absent memory → nothing to inject
    assert render_memory_section(None, max_context_tokens=200) == ""
    assert render_memory_section(ContextMemory(), max_context_tokens=200) == ""


# ── Long-term memory: durable facts ──────────────────────────────────────────

def test_long_term_memory_dedups_caps_and_persists():
    dupes = [
        Event(seq=i + 1, kind=EventKind.OBSERVATION, ts=float(i),
              payload=FactObservation(content=f"fact {i % 3}"))
        for i in range(60)
    ]
    mem = LongTermMemory.from_events(dupes)
    assert len(mem) == 3                        # exact repeats collapse
    assert mem.texts == ("fact 0", "fact 1", "fact 2")

    unique = [
        Event(seq=i + 1, kind=EventKind.OBSERVATION, ts=float(i),
              payload=FactObservation(content=f"unique fact {i}"))
        for i in range(60)
    ]
    capped = LongTermMemory.from_events(unique, max_facts=50)
    assert len(capped) == 50

    # JSON round-trip: cross-session persistence with no new file format
    restored = LongTermMemory.from_dict(capped.to_dict())
    assert restored.texts == capped.texts
    assert restored.max_facts == 50

    # non-fact observations are never durable memory
    plain = [
        Event(seq=1, kind=EventKind.OBSERVATION, ts=1.0,
              payload=ToolResultObservation(content="not a fact")),
    ]
    assert len(LongTermMemory.from_events(plain)) == 0


def test_remember_records_durable_facts_and_keeps_them_out_of_the_transcript():
    convo = Conversation(StubAgent(), StubWorkspace(), query="q")
    convo.submit_user_message("go")
    convo.remember("user prefers concise answers")
    convo.remember("user prefers concise answers")   # exact repeat

    facts = [e for e in convo.log.events if e.payload.kind == "fact"]
    assert len(facts) == 2                           # both recorded as events
    assert all(f.payload.content == "user prefers concise answers" for f in facts)

    # facts never reset the turn boundary
    assert convo.state.last_user_seq == 1
    # and never appear in the raw turn transcript
    transcript = ToolCallingAgent.render_history(convo.log.events)
    assert "prefers" not in transcript


async def test_durable_facts_are_injected_into_every_step_context(tmp_path):
    registry = ToolRegistry([echo_tool()])
    backend = FakeBackend([
        tool_call_response("echo", {"text": "hi"}),
        text_response("done"),
    ])
    convo = Conversation(
        ToolCallingAgent(backend, registry), ToolWorkspace(registry), query="q",
    )
    convo.submit_user_message("go")
    convo.remember("workspace is a scratch dir")

    result = await convo.run()

    assert result.completed and result.steps == 1
    for call in backend.calls:
        user = call["messages"][1]["content"]
        assert user.startswith("[Durable facts]\n- workspace is a scratch dir")
    # replay reconstructs the same durable memory
    replay = LongTermMemory.from_events(convo.log.events)
    assert replay.texts == ("workspace is a scratch dir",)


# ── Bounded context end-to-end: 25+ steps ────────────────────────────────────

@pytest.mark.parametrize("reasoning", [0, 5])
async def test_long_session_context_stays_bounded(tmp_path, reasoning):
    registry = ToolRegistry([echo_tool()])
    n_steps = 25
    script = [
        tool_call_response(
            "echo", {"text": f"payload {i}"}, tc_id=f"call_{i}",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=0,
                             reasoning_tokens=reasoning),
        )
        for i in range(n_steps)
    ]
    backend = FakeBackend(script, reports_reasoning_tokens=bool(reasoning))
    convo = Conversation(
        ToolCallingAgent(backend, registry), ToolWorkspace(registry), query="q",
        agent_config=AgentConfig(max_context_tokens=200),
    )
    convo.submit_user_message("go")

    result = await convo.run()

    assert result.completed and result.steps == n_steps
    assert len(backend.calls) == n_steps + 1     # 25 tool turns + 1 final answer
    log = convo.log.events

    # EVERY model call must have received exactly the windowed memory
    # section its log prefix dictates — the model never sees the raw log.
    for k, call in enumerate(backend.calls, start=1):
        prefix_len = 1 + 2 * (k - 1)            # user message + k-1 action/obs pairs
        expected = render_memory_section(
            ContextMemory.from_events(log[:prefix_len]),
            max_context_tokens=200,
        )
        assert call["messages"][1]["content"].startswith(expected)

    last_section = render_memory_section(
        ContextMemory.from_events(log[: 1 + 2 * n_steps]),
        max_context_tokens=200,
    )
    # the window folds the oldest steps away but always keeps the newest
    assert "step 25:" in last_section
    assert "step 1:" not in last_section
    assert "earlier step(s) omitted" in last_section
    # deterministic replay of the whole session
    assert EventLog.replay(result.events) == result.log_state


# ── Budget discipline via the EXISTING machinery ─────────────────────────────

async def test_wall_clock_budget_exceeded_ends_in_clean_finish():
    convo = Conversation(AlwaysToolAgent(), StubWorkspace(), query="q")
    convo.submit_user_message("go")

    budget = BudgetState(max_latency_s=0.0)     # already over budget
    result = await convo.run(budget=budget)

    assert result.steps == 0                    # never even started a step
    assert result.terminated_by == "budget"
    last = result.events[-1]
    assert isinstance(last.payload, FinishAction)
    assert last.payload.reason == "budget"
    assert budget.exhausted


async def test_completed_iterations_count_on_the_shared_budget():
    convo = Conversation(
        AlwaysToolAgent(), StubWorkspace(), query="q",
        config=ConversationConfig(max_tool_repeats=100),
    )
    convo.submit_user_message("go")

    budget = BudgetState(max_iterations=4)
    result = await convo.run(budget=budget)

    assert result.steps == 4
    assert result.terminated_by == "budget"
    # the SAME budget object carries the accounting the planner read
    assert budget.iterations == 4
    assert budget.latency_used_s >= 0.0


async def test_pre_exhausted_budget_stops_without_looping():
    convo = Conversation(AlwaysToolAgent(), StubWorkspace(), query="q")
    convo.submit_user_message("go")

    result = await convo.run(budget=BudgetState(max_iterations=0))

    assert result.steps == 0
    assert result.terminated_by == "budget"
    assert isinstance(result.events[-1].payload, FinishAction)


# ── Reuse, not reinvention: real selector + capability sandbox ───────────────

def _real_toolset(tmp_path, selector: Optional[ExpertSelector] = None):
    return ToolRegistry(build_default_tools([str(tmp_path)], selector=selector))


async def test_query_experts_routes_through_the_real_selector(tmp_path):
    selector = ExpertSelector(load_mock_experts(), seed=7)
    ws = ToolWorkspace(_real_toolset(tmp_path, selector))

    obs = observation_from_packet(ws.execute(action_packet(
        ToolCallAction(name="query_experts",
                       arguments={"query": "explain black holes", "limit": 2}),
        query="q", seq=1,
    )))

    assert isinstance(obs, ToolResultObservation) and obs.success
    assert "experts" in obs.content and "score" in obs.content
    # Loop-guard contract (Prompt 9): no selector configured → the tool is
    # NOT registered, so the model cannot even attempt the call that used
    # to degrade to a repeated "no expert selector" note.
    bare_registry = _real_toolset(tmp_path, None)
    assert "query_experts" not in bare_registry.names()
    bare = ToolWorkspace(bare_registry)
    degraded = observation_from_packet(bare.execute(action_packet(
        ToolCallAction(name="query_experts", arguments={"query": "anything"}),
        query="q", seq=1,
    )))
    assert isinstance(degraded, ErrorObservation)
    assert "unknown tool" in degraded.message


async def test_capability_sandbox_routing_proof(tmp_path):
    tools = build_default_tools([str(tmp_path)], selector=ExpertSelector(load_mock_experts(), seed=7))
    by_name = {t.name: t for t in tools}
    assert by_name["run_command"].safety_level == DANGEROUS_LEVEL
    assert by_name["query_experts"].safety_level == SAFE

    ws = ToolWorkspace(ToolRegistry(tools))

    # deny-list: destructive commands never reach a shell
    denied = observation_from_packet(ws.execute(action_packet(
        ToolCallAction(name="run_command", arguments={"command": "rm -rf /"}),
        query="q", seq=1,
    )))
    assert isinstance(denied, ErrorObservation)
    assert "[denied_command]" in denied.message

    # path scoping: reads outside the workspace roots are refused
    outside = tmp_path.parent / "p4_secret.txt"
    outside.write_text("secret")
    escaped = observation_from_packet(ws.execute(action_packet(
        ToolCallAction(name="read_file", arguments={"path": str(outside)}),
        query="q", seq=1,
    )))
    assert isinstance(escaped, ErrorObservation)
    assert "[path_outside_workspace]" in escaped.message

    # inside the workspace: works
    inside = tmp_path / "ok.txt"
    inside.write_text("fine")
    ok = observation_from_packet(ws.execute(action_packet(
        ToolCallAction(name="read_file", arguments={"path": str(inside)}),
        query="q", seq=1,
    )))
    assert isinstance(ok, ToolResultObservation) and ok.success


async def test_full_session_with_real_capabilities_and_working_memory(tmp_path):
    selector = ExpertSelector(load_mock_experts(), seed=7)
    registry = _real_toolset(tmp_path, selector)
    backend = FakeBackend([
        tool_call_response("write_file",
                           {"path": str(tmp_path / "m.txt"), "content": "hello"},
                           tc_id="call_1"),
        tool_call_response("read_file", {"path": str(tmp_path / "m.txt")}, tc_id="call_2"),
        tool_call_response("query_experts", {"query": "summarize the file", "limit": 2},
                           tc_id="call_3"),
        text_response("done"),
    ])
    convo = Conversation(
        ToolCallingAgent(backend, registry), ToolWorkspace(registry), query="q",
        agent_config=AgentConfig(max_context_tokens=400),
    )
    convo.submit_user_message("go")
    convo.remember("workspace is a scratch dir")

    result = await convo.run()

    assert result.completed and result.steps == 3
    for call in backend.calls:
        content = call["messages"][1]["content"]
        assert "[Durable facts]\n- workspace is a scratch dir" in content
    assert "[Recent steps]" in backend.calls[-1]["messages"][1]["content"]
    assert EventLog.replay(result.events) == result.log_state


# ── The stateless contract survives the memory layer ─────────────────────────

def test_context_memory_is_a_plain_immutable_snapshot():
    events = _step_events(3)
    snap = ContextMemory.from_events(events)
    assert snap.steps[0].step_number == 1 and snap.facts == ()
    # same events → byte-identical snapshot (pure projection)
    assert ContextMemory.from_events(events) == snap
