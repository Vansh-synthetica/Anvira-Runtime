"""
Tests for the LangGraph single-agent graph (orcha.builders.langgraph_agent).

These verify the genuine-LangGraph port of the agent graph: engine
selection through ``build_agent_runner``, topology parity with the native
``Graph``, identical routing (chat / read / tool), and the interrupt-based
approval gate (pause -> ApprovalPending -> resume approve/reject -> answer).
"""
import asyncio

import pytest

from orcha.builders import (
    AgentGraphConfig,
    AgentLangGraphRunner,
    ApprovalPending,
    ComplexityGateConfig,
    build_agent_graph,
    build_agent_graph_langgraph,
    build_agent_runner,
)
from orcha.capabilities.base import CapabilityContext
from orcha.capabilities.registry import CapabilityRegistry
from orcha.graph.runtime import GraphRuntime
from orcha.nodes.agent import AgentConfig
from orcha.nodes.intent import IntentGateConfig


def _filesystem_executor(root):
    registry = CapabilityRegistry().register_defaults()
    ctx = CapabilityContext(roots=[str(root)])
    return registry.build(["filesystem"], ctx=ctx)


def _tool_call(name, arguments, call_id="call_x"):
    return {
        "role": "assistant",
        "content": None,
        "finish_reason": "stop",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }],
    }


def _completion(content=None, tool_calls=None):
    """Deterministic completion fake: an optional tool call, then a final
    plain-text answer (mirrors the test_agent_graph fixtures)."""
    async def fake(messages, system, tool_schemas):
        if tool_calls is not None and not any(
            m.get("role") == "tool" for m in messages
        ):
            return tool_calls
        return {"role": "assistant", "content": content, "finish_reason": "stop"}
    return fake


def _gate_completion(code):
    """Gate completion fake: reply with exactly one intent code (or chat text)."""
    async def fake(messages, system, tool_schemas):
        return {"role": "assistant", "content": code, "finish_reason": "stop"}
    return fake


def _runner(**kwargs):
    return build_agent_runner(**kwargs)


# ── Engine selection ────────────────────────────────────────────────────

def test_build_agent_runner_engine_selection(monkeypatch):
    cfg = AgentGraphConfig(
        agent_config=AgentConfig(completion_fn=_completion("hi")),
    )

    lg = build_agent_runner(cfg, engine="langgraph")
    assert isinstance(lg, AgentLangGraphRunner)
    assert lg.entry == "agent"  # no gate configured

    native = build_agent_runner(cfg, engine="native")
    assert isinstance(native, GraphRuntime)
    assert not isinstance(native, AgentLangGraphRunner)

    defaulted = build_agent_runner(cfg)
    assert isinstance(defaulted, AgentLangGraphRunner)

    env = build_agent_runner(cfg, engine="langgraph")
    assert isinstance(env, AgentLangGraphRunner)

    monkeypatch.setenv("ORCHA_AGENT_ENGINE", "langgraph")
    env_driven = build_agent_runner(cfg)
    assert isinstance(env_driven, AgentLangGraphRunner)

    from_orcha_config = build_agent_runner(
        AgentGraphConfig(
            agent_config=AgentConfig(completion_fn=_completion("hi")),
            engine="langgraph",
        )
    )
    assert isinstance(from_orcha_config, AgentLangGraphRunner)

    with pytest.raises(ValueError, match="Unknown engine"):
        build_agent_runner(cfg, engine="bogus")


def test_build_agent_runner_engine_overrides_config_engine():
    cfg = AgentGraphConfig(
        agent_config=AgentConfig(completion_fn=_completion("hi")),
        engine="langgraph",
    )
    assert isinstance(build_agent_runner(cfg, engine="native"), GraphRuntime)


def test_runner_entry_reflects_gate_config():
    cfg = AgentGraphConfig(
        agent_config=AgentConfig(completion_fn=_completion("hi")),
        gate_config=IntentGateConfig(completion_fn=_gate_completion("CHAT")),
    )
    lg = build_agent_runner(cfg, engine="langgraph")
    assert lg.entry == "intent_gate"


# ── Topology ────────────────────────────────────────────────────────────

def test_langgraph_topology_matches_native():
    cfg = AgentGraphConfig(
        agent_config=AgentConfig(completion_fn=_completion("hi")),
        readonly_agent_config=AgentConfig(completion_fn=_completion("ro")),
        gate_config=IntentGateConfig(completion_fn=_gate_completion("CHAT")),
    )
    native = build_agent_graph(cfg)
    lg = build_agent_graph_langgraph(cfg)

    lg_names = {n for n in lg.nodes if not n.startswith("__")}
    assert set(native._nodes) == lg_names
    assert {"intent_gate", "agent", "agent_readonly", "finalize_agent"} <= lg_names
    assert "approval_gate" not in lg_names


def test_langgraph_topology_approval_gate_only_when_required():
    cfg = AgentGraphConfig(
        agent_config=AgentConfig(completion_fn=_completion("hi")),
        require_approval=True,
    )
    lg = build_agent_graph_langgraph(cfg)
    assert "approval_gate" in lg.nodes

    plain = build_agent_graph_langgraph(AgentGraphConfig(
        agent_config=AgentConfig(completion_fn=_completion("hi")),
    ))
    assert "approval_gate" not in plain.nodes


def test_langgraph_topology_no_gate_legacy():
    cfg = AgentGraphConfig(agent_config=AgentConfig(completion_fn=_completion("hi")))
    lg = build_agent_graph_langgraph(cfg)
    assert "intent_gate" not in lg.nodes
    assert "agent" in lg.nodes
    assert lg.max_steps == 1000


def test_langgraph_max_steps_attribute():
    lg = build_agent_graph_langgraph(max_steps=17)
    assert lg.max_steps == 17


# ── Routing parity (chat / read / tool) ─────────────────────────────────

def test_chat_route_answers_directly():
    chat_text = "Hi there! How can I help?"
    runner = _runner(
        engine="langgraph",
        agent_config=AgentConfig(completion_fn=_completion("agent answer")),
        gate_config=IntentGateConfig(completion_fn=_gate_completion(chat_text)),
    )
    result = asyncio.run(runner.run("hello"))
    assert result.answer == chat_text
    assert result.packet.payload["primary"] == "agent"
    # Chat path: no agent work — no iterations, no steps (agent_completed
    # mirrors native: True when the router produced a response).
    assert result.packet.payload.get("agent_iterations", 0) == 0
    assert result.packet.payload["agent_steps"] == []
    assert result.packet.payload["agent_tool_calls"] == []


def test_read_route_uses_readonly_agent():
    runner = _runner(
        engine="langgraph",
        agent_config=AgentConfig(completion_fn=_completion("full agent answer")),
        readonly_agent_config=AgentConfig(completion_fn=_completion("readonly answer")),
        gate_config=IntentGateConfig(completion_fn=_gate_completion("READ_WORKSPACE")),
    )
    result = asyncio.run(runner.run("look through this folder"))
    assert result.answer == "readonly answer"
    assert result.packet.payload["primary"] == "agent"


def test_read_route_without_readonly_falls_back_to_full_agent():
    runner = _runner(
        engine="langgraph",
        agent_config=AgentConfig(completion_fn=_completion("full agent answer")),
        gate_config=IntentGateConfig(completion_fn=_gate_completion("READ_WORKSPACE")),
    )
    result = asyncio.run(runner.run("look through this folder"))
    assert result.answer == "full agent answer"


def test_tool_route_runs_agent_loop(tmp_path):
    executor = _filesystem_executor(tmp_path)
    runner = _runner(
        engine="langgraph",
        agent_config=AgentConfig(
            completion_fn=_completion(
                content="created the file",
                tool_calls=_tool_call("write_file", {"path": "made.txt", "content": "hi"}),
            ),
            executor=executor,
            max_iterations=4,
        ),
        gate_config=IntentGateConfig(completion_fn=_gate_completion("TOOL_REQUEST")),
    )
    result = asyncio.run(runner.run("create made.txt"))
    p = result.packet.payload

    assert p["answer"] == "created the file"
    assert p["agent_completed"] is True
    assert (tmp_path / "made.txt").read_text() == "hi"
    assert [c["name"] for c in p["agent_tool_calls"]] == ["write_file"]
    assert p["agent_tool_calls"][0]["result_type"] == "ok"
    # One tool-call turn + one final answer = 2 model calls (native semantics:
    # agent_iterations counts model calls).
    assert p["agent_iterations"] == 2


def test_legacy_no_gate_topology_runs():
    runner = _runner(
        engine="langgraph",
        agent_config=AgentConfig(completion_fn=_completion("legacy answer")),
    )
    result = asyncio.run(runner.run("anything"))
    assert result.answer == "legacy answer"


# ── Parity with the native engine ───────────────────────────────────────

def test_parity_native_vs_langgraph_surface(tmp_path):
    executor = _filesystem_executor(tmp_path)
    shared = dict(
        agent_config=AgentConfig(
            completion_fn=_completion(
                content="parity answer",
                tool_calls=_tool_call("write_file", {"path": "p.txt", "content": "p"}),
            ),
            executor=executor,
            max_iterations=4,
        ),
        readonly_agent_config=AgentConfig(completion_fn=_completion("ro")),
        gate_config=IntentGateConfig(completion_fn=_gate_completion("TOOL_REQUEST")),
    )
    native = _runner(engine="native", **shared)
    lg = _runner(engine="langgraph", **shared)

    native_res = asyncio.run(native.run("do it", run_id="parity-1"))
    lg_res = asyncio.run(lg.run("do it", run_id="parity-1"))

    assert lg_res.answer == native_res.answer == "parity answer"
    assert lg_res.confidence == native_res.confidence
    assert lg_res.run_id == native_res.run_id == "parity-1"
    for key in ("answer", "primary", "agg_mode", "agent_completed", "agent_iterations"):
        assert lg_res.packet.payload[key] == native_res.packet.payload[key]
    assert (tmp_path / "p.txt").read_text() == "p"


# ── Approval gate (interrupt-based human-in-the-loop) ───────────────────

def _approval_runner(tmp_path, **overrides):
    executor = _filesystem_executor(tmp_path)
    kwargs = dict(
        engine="langgraph",
        agent_config=AgentConfig(
            completion_fn=_completion(
                content="proposed answer",
                tool_calls=_tool_call("write_file", {"path": "ap.txt", "content": "x"}),
            ),
            executor=executor,
            max_iterations=4,
        ),
        gate_config=IntentGateConfig(completion_fn=_gate_completion("TOOL_REQUEST")),
        require_approval=True,
    )
    kwargs.update(overrides)
    return _runner(**kwargs)


def test_approval_pause_surfaces_payload(tmp_path):
    runner = _approval_runner(tmp_path)
    with pytest.raises(ApprovalPending) as exc_info:
        asyncio.run(runner.run("do it", run_id="approval-pause-1"))
    pending = exc_info.value
    assert pending.run_id == "approval-pause-1"
    assert pending.node == "approval_gate"
    payload = pending.approval_payload
    assert payload["kind"] == "agent_approval"
    assert payload["output"] == "proposed answer"
    # One tool-call turn + one final answer = 2 model calls.
    assert payload["iterations"] == 2
    assert [c["name"] for c in payload["tool_calls"]] == ["write_file"]


def test_approval_resume_approve_true(tmp_path):
    runner = _approval_runner(tmp_path)
    run_id = "approval-approve-1"
    with pytest.raises(ApprovalPending):
        asyncio.run(runner.run("do it", run_id=run_id))
    result = asyncio.run(runner.run(
        "do it", run_id=run_id, resume_from=object(), resume_value=True,
    ))
    assert result.answer == "proposed answer"
    assert result.packet.payload.get("approval_rejected") is not True
    assert (tmp_path / "ap.txt").read_text() == "x"


def test_approval_resume_approve_dict(tmp_path):
    runner = _approval_runner(tmp_path)
    run_id = "approval-approve-2"
    with pytest.raises(ApprovalPending):
        asyncio.run(runner.run("do it", run_id=run_id))
    result = asyncio.run(runner.run(
        "do it", run_id=run_id, resume_from=object(),
        resume_value={"approved": True, "note": "looks good"},
    ))
    assert result.answer == "proposed answer"


def test_approval_resume_reject_with_note(tmp_path):
    runner = _approval_runner(tmp_path)
    run_id = "approval-reject-1"
    with pytest.raises(ApprovalPending):
        asyncio.run(runner.run("do it", run_id=run_id))
    result = asyncio.run(runner.run(
        "do it", run_id=run_id, resume_from=object(),
        resume_value={"approved": False, "note": "nope"},
    ))
    assert result.answer == "The proposed action was not approved. Reason: nope"
    assert result.packet.payload.get("approval_rejected") is True


def test_approval_resume_reject_bare_value(tmp_path):
    runner = _approval_runner(tmp_path)
    run_id = "approval-reject-2"
    with pytest.raises(ApprovalPending):
        asyncio.run(runner.run("do it", run_id=run_id))
    result = asyncio.run(runner.run(
        "do it", run_id=run_id, resume_from=object(), resume_value="no",
    ))
    assert result.answer == "The proposed action was not approved. Reason: no reason given"


def test_approval_resume_completed_thread_starts_fresh(tmp_path):
    runner = _approval_runner(tmp_path)
    run_id = "approval-fresh-1"
    with pytest.raises(ApprovalPending):
        asyncio.run(runner.run("do it", run_id=run_id))
    asyncio.run(runner.run(
        "do it", run_id=run_id, resume_from=object(), resume_value=True,
    ))
    # Thread is now completed: a resume attempt starts a fresh execution
    # (no pending work) — which is a brand-new run and therefore pauses
    # at the approval gate again awaiting a new decision.
    with pytest.raises(ApprovalPending) as exc_info:
        asyncio.run(runner.run(
            "do it", run_id=run_id, resume_from=object(), resume_value=True,
        ))
    assert exc_info.value.approval_payload["kind"] == "agent_approval"


def test_approval_pending_detection_on_engine(tmp_path):
    runner = _approval_runner(tmp_path)
    run_id = "approval-pending-1"
    with pytest.raises(ApprovalPending):
        asyncio.run(runner.run("do it", run_id=run_id))

    async def check():
        return await runner.engine.pending(run_id)
    assert asyncio.run(check()) == ["approval_gate"]

    asyncio.run(runner.run(
        "do it", run_id=run_id, resume_from=object(), resume_value=True,
    ))
    assert asyncio.run(check()) == []


def test_replay_roundtrip_without_approval(tmp_path):
    executor = _filesystem_executor(tmp_path)
    runner = build_agent_runner(
        engine="langgraph",
        agent_config=AgentConfig(
            completion_fn=_completion(
                content="replayed answer",
                tool_calls=_tool_call("write_file", {"path": "rp.txt", "content": "r"}),
            ),
            executor=executor,
            max_iterations=4,
        ),
        gate_config=IntentGateConfig(completion_fn=_gate_completion("TOOL_REQUEST")),
    )
    run_id = "replay-1"
    result = asyncio.run(runner.run("do it", run_id=run_id))
    assert result.answer == "replayed answer"

    replayed = asyncio.run(runner.replay(run_id))
    assert replayed.answer == "replayed answer"
    assert (tmp_path / "rp.txt").read_text() == "r"


def test_replay_of_approval_run_re_pauses(tmp_path):
    runner = _approval_runner(tmp_path)
    run_id = "approval-replay-1"
    with pytest.raises(ApprovalPending):
        asyncio.run(runner.run("do it", run_id=run_id))
    asyncio.run(runner.run(
        "do it", run_id=run_id, resume_from=object(), resume_value=True,
    ))
    # Replay re-executes the thread from its first checkpoint, so an
    # approval-gated run re-enters the gate and pauses again.
    with pytest.raises(ApprovalPending) as exc_info:
        asyncio.run(runner.replay(run_id))
    assert exc_info.value.approval_payload["kind"] == "agent_approval"


# ── Events / knobs ──────────────────────────────────────────────────────

def test_runner_emits_run_events(tmp_path):
    runner = _approval_runner(tmp_path)
    events = []

    async def collect():
        async def on_event(ev):
            events.append(ev)
        try:
            await runner.run("do it", run_id="approval-events-1", on_event=on_event)
        except ApprovalPending:
            pass

    asyncio.run(collect())
    assert "run_start" in [ev.kind for ev in events]

    resumed_events = []

    async def resume_collect():
        async def on_event(ev):
            resumed_events.append(ev)
        await runner.run(
            "do it", run_id="approval-events-1",
            resume_from=object(), resume_value=True, on_event=on_event,
        )

    asyncio.run(resume_collect())
    kinds = [ev.kind for ev in resumed_events]
    assert "run_start" in kinds
    assert "checkpoint" in kinds
    assert resumed_events[0].data.get("action") == "resume"
    assert "run_complete" in kinds


def test_knobs_max_iterations_lands_in_packet_budget(tmp_path):
    executor = _filesystem_executor(tmp_path)
    runner = build_agent_runner(
        engine="langgraph",
        agent_config=AgentConfig(
            completion_fn=_completion(
                content="capped",
                tool_calls=_tool_call("write_file", {"path": "c.txt", "content": "c"}),
            ),
            executor=executor,
            max_iterations=8,
        ),
        gate_config=IntentGateConfig(completion_fn=_gate_completion("TOOL_REQUEST")),
        max_iterations=1,
    )
    result = asyncio.run(runner.run("do it"))
    # The knob lands on the packet budget (graph-level cap); the agent's own
    # model-call budget stays untouched, so the run still completes.
    assert result.packet.budget.max_iterations == 1
    assert result.answer == "capped"
    assert result.packet.payload["agent_completed"] is True


# ── Task-decomposition parity (complexity gate / planner / execution loop) ──
#
# Ports the native complexity-gate topology (orcha/builders/agent.py:606-743)
# into build_agent_graph_langgraph. These tests validate the WIRING — that
# every node native adds is present here too and every route predicate leads
# somewhere real — not the node-level business logic, which is already
# covered by the 100+ tests in test_task_executor.py and test_planner.py
# against the exact same Node classes.

def _complexity_cfg(**overrides):
    """heuristic_threshold=0.0 forces `complexity_gate="complex"` on any
    query without needing a completion_fn (score >= 0.0 is always true —
    see ComplexityGateNode.run in orcha/nodes/planner.py)."""
    kwargs = dict(heuristic_threshold=0.0)
    kwargs.update(overrides)
    return ComplexityGateConfig(**kwargs)


def test_langgraph_topology_matches_native_with_complexity_gate():
    cfg = AgentGraphConfig(
        agent_config=AgentConfig(completion_fn=_completion("hi")),
        gate_config=IntentGateConfig(completion_fn=_gate_completion("TOOL_REQUEST")),
        complexity_gate_config=_complexity_cfg(),
    )
    native = build_agent_graph(cfg)
    lg = build_agent_graph_langgraph(cfg)

    lg_names = {n for n in lg.nodes if not n.startswith("__")}
    assert set(native._nodes) == lg_names
    assert {
        "intent_gate", "agent", "finalize_agent", "complexity_gate",
        "task_planner", "task_context_builder", "step_executor", "observer",
        "replanner", "verifier", "execution_loop", "execution_observer",
        "retry_handler", "final_verifier", "adaptive_replanner",
    } <= lg_names


def test_complexity_gate_simple_query_bypasses_task_planner():
    # heuristic_threshold=1.0 with a trivial query: score is virtually
    # guaranteed to land below threshold*0.5, so the gate short-circuits to
    # "simple" without a completion_fn — same fast path a one-line chat
    # request should take even with the complexity gate configured.
    runner = _runner(
        engine="langgraph",
        agent_config=AgentConfig(completion_fn=_completion("simple answer")),
        gate_config=IntentGateConfig(completion_fn=_gate_completion("TOOL_REQUEST")),
        complexity_gate_config=ComplexityGateConfig(heuristic_threshold=1.0),
    )
    result = asyncio.run(runner.run("hi"))
    assert result.answer == "simple answer"
    assert result.packet.payload["complexity_gate"] == "simple"


def test_complexity_gate_complex_query_reaches_task_planner_and_finalizes(tmp_path):
    # completion_fn=None on TaskPlannerNode deterministically falls back to
    # a single-step "execute_directly" plan (see test_planner.py) — no need
    # to fabricate a JSON plan response. The execution_loop then also gets
    # completion_fn=None (same ComplexityGateConfig.completion_fn reference
    # as task_planner) and returns a clean "no model available" failure per
    # task, rather than raising — this is enough to prove every edge in the
    # new topology actually leads to finalize_agent without a NoRoute/
    # KeyError, which is what this test exists to catch.
    runner = _runner(
        engine="langgraph",
        agent_config=AgentConfig(completion_fn=_completion("unused")),
        gate_config=IntentGateConfig(completion_fn=_gate_completion("TOOL_REQUEST")),
        complexity_gate_config=_complexity_cfg(completion_fn=None),
    )
    result = asyncio.run(runner.run("build a multi-step feature", run_id="complex-1"))
    assert result.packet.payload["complexity_gate"] == "complex"
    assert "execution_plan" in result.packet.payload
    # Reached a real terminal answer — the graph didn't dead-end mid-loop.
    assert isinstance(result.answer, str)


def test_complexity_gate_routes_through_approval_gate_when_required(tmp_path):
    # The approval gate must guard every way out of the complexity-gate
    # pipeline, not just the plain single-turn agent path — see the
    # `after_agent` usage throughout build_agent_graph_langgraph.
    runner = _runner(
        engine="langgraph",
        agent_config=AgentConfig(completion_fn=_completion("unused")),
        gate_config=IntentGateConfig(completion_fn=_gate_completion("TOOL_REQUEST")),
        complexity_gate_config=_complexity_cfg(completion_fn=None),
        require_approval=True,
    )
    with pytest.raises(ApprovalPending) as exc_info:
        asyncio.run(runner.run("build a multi-step feature", run_id="complex-approval-1"))
    assert exc_info.value.node == "approval_gate"
