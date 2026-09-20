"""
tests.test_multi_agent_langgraph
================================
Stage 7: the multi-agent collaboration graph on the LangGraph engine.

Verifies the Send-based scatter/gather port against the native graph:
identical topology, identical packet results (branches, synthesis,
bounded verify retry), identical node/event sequences modulo the native
runtime's fan_out/fan_in metadata events, plus runner-surface parity
(replay / resume / engine switching) and the server-side engine switch.
"""
import asyncio
import os

import pytest

os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "false")

from orcha.builders import (
    MultiAgentLangGraphRunner,
    build_multi_agent_graph,
    build_multi_agent_runner,
)
from orcha.builders.langgraph_multi_agent import build_multi_agent_graph_langgraph
from orcha.builders.multi_agent import MultiAgentGraphConfig
from orcha.graph import GraphRuntime
from orcha.nodes.agent import AgentConfig

pytestmark = pytest.mark.usefixtures("_no_env_engine")


@pytest.fixture
def _no_env_engine(monkeypatch):
    monkeypatch.delenv("ORCHA_AGENT_ENGINE", raising=False)


def _run(rt, query, **kwargs):
    return asyncio.run(rt.run(query, **kwargs))


def _node_start_names(events):
    return [e.node for e in events if e.kind == "node_start"]


def _lg_graph(**cfg_kwargs):
    return build_multi_agent_graph_langgraph(
        MultiAgentGraphConfig(**cfg_kwargs),
    )


def _native_graph(**cfg_kwargs):
    return build_multi_agent_graph(MultiAgentGraphConfig(**cfg_kwargs))


# ── Topology ──────────────────────────────────────────────────────────


def test_langgraph_topology_verify_true():
    g = _lg_graph(verify=True)
    edges = {(e.source, e.target) for e in g.get_graph().edges}

    assert {"decompose", "scatter_agents", "agent_worker", "gather_agents",
            "synthesize", "verify", "verify_retry_gate"} <= set(g.nodes)
    # This langgraph version's static introspection reports only the
    # unconditional linear chain; the fan-in (agent_worker→gather_agents)
    # and the verify-retry loop are exercised by the behavior tests below.
    assert ("__start__", "decompose") in edges
    assert ("decompose", "scatter_agents") in edges


def test_langgraph_topology_verify_false():
    g = _lg_graph(verify=False)
    assert "verify" not in g.nodes
    assert "verify_retry_gate" not in g.nodes
    # Verify-less run: still decompose → scatter → fan-out → gather → synthesize.
    events = []
    rt = build_multi_agent_runner(
        MultiAgentGraphConfig(verify=False), engine="langgraph",
    )
    r = _run(rt, "Explain AI", on_event=events.append)
    assert _node_start_names(events)[-1] == "synthesize"
    assert r.packet.payload["branch_count"] == 5


# ── Behavior parity with the native graph ─────────────────────────────


def test_dry_run_parity_native_vs_langgraph():
    events_lg = []
    rt_lg = build_multi_agent_runner(
        MultiAgentGraphConfig(verify=True), engine="langgraph",
    )
    r_lg = _run(rt_lg, "Explain AI", on_event=events_lg.append)

    events_n = []
    rt_n = GraphRuntime(_native_graph(verify=True))
    r_n = _run(rt_n, "Explain AI", on_event=events_n.append)

    assert r_lg.packet.payload["branch_count"] == r_n.packet.payload["branch_count"] == 5
    assert r_lg.packet.payload["branches_completed"] == r_n.packet.payload["branches_completed"]
    assert r_lg.packet.payload["verify_retries"] == r_n.packet.payload["verify_retries"] == 1
    assert r_lg.packet.payload["answer"] == r_n.packet.payload["answer"]

    # Node execution sequence is identical.
    assert _node_start_names(events_lg) == _node_start_names(events_n) == [
        "decompose", "scatter_agents",
        "agent", "agent", "agent", "agent", "agent",
        "gather_agents", "synthesize",
        "verify", "verify_retry_gate", "synthesize", "verify", "verify_retry_gate",
    ]
    # Event kinds match modulo the native runtime's fan metadata.
    lg_kinds = {e.kind for e in events_lg}
    native_kinds = {e.kind for e in events_n} - {"fan_out", "fan_in"}
    assert lg_kinds == native_kinds


def test_completion_and_synthesis():
    async def fake_completion(messages, system, tool_schemas):
        return {"role": "assistant", "content": "branch answer", "finish_reason": "stop"}

    async def fake_synth(prompt, system):
        return "ANSWER: synthesized answer"

    rt = build_multi_agent_runner(
        MultiAgentGraphConfig(
            agent_config=AgentConfig(completion_fn=fake_completion, max_iterations=2),
            model_fn=fake_synth,
            verify=False,
        ),
        engine="langgraph",
    )
    r = _run(rt, "Explain AI")
    p = r.packet.payload

    assert p["synthesized"] is True
    assert p["answer"] == "synthesized answer"
    assert p["confidence"] == 0.7
    assert p["contributors"] == [
        "branch_0", "branch_1", "branch_2", "branch_3", "branch_4",
    ]


def test_verify_retry_is_bounded():
    rt = build_multi_agent_runner(
        MultiAgentGraphConfig(verify=True), engine="langgraph",
    )
    r = _run(rt, "Explain AI")
    p = r.packet.payload

    assert p.get("verify_retries") == 1
    assert p.get("verify_failures")
    verify_passes = [
        t for t in r.trace if t.get("stage") == "verify"
    ]
    assert len(verify_passes) == 2


def test_max_branches_respected():
    rt = build_multi_agent_runner(
        MultiAgentGraphConfig(verify=False, max_branches=2), engine="langgraph",
    )
    r = _run(rt, "Explain AI")
    p = r.packet.payload

    assert p["branch_count"] == 2
    assert len(p["contributors"]) == 2


# ── Runner surface parity ─────────────────────────────────────────────


def test_replay_matches_run():
    rt = build_multi_agent_runner(
        MultiAgentGraphConfig(verify=False), engine="langgraph",
    )
    r = _run(rt, "Explain AI")

    events = []
    r2 = asyncio.run(rt.replay(r.run_id, on_event=events.append))
    kinds = [e.kind for e in events]

    assert r2.run_id == r.run_id
    assert kinds[0] == "run_start"
    assert kinds[-1] == "run_complete"
    assert events[0].data.get("replay") is True
    assert r2.packet.payload["answer"] == r.packet.payload["answer"]


def test_resume_completed_thread_starts_fresh():
    rt = build_multi_agent_runner(
        MultiAgentGraphConfig(verify=False), engine="langgraph",
    )
    r = _run(rt, "Explain AI", resume_from="some_store")

    # No pending work in the multi-agent graph: the resume behaves like a
    # fresh run on the same thread (matching native resume-from-completed).
    assert r.packet.payload["branch_count"] == 5


def test_engine_switch_surface():
    os.environ["ORCHA_AGENT_ENGINE"] = "langgraph"
    try:
        rt = build_multi_agent_runner(MultiAgentGraphConfig(verify=False))
    finally:
        os.environ.pop("ORCHA_AGENT_ENGINE", None)
    assert isinstance(rt, MultiAgentLangGraphRunner)

    rt_native = build_multi_agent_runner(
        MultiAgentGraphConfig(verify=False), engine="native",
    )
    assert isinstance(rt_native, GraphRuntime)

    with pytest.raises(ValueError, match="Unknown engine"):
        build_multi_agent_runner(MultiAgentGraphConfig(), engine="bogus")


def test_reasoning_knobs_parity():
    # Same config through both builders must produce the same verify
    # topology (the reasoning knobs are shared via _apply_reasoning_knobs).
    config = MultiAgentGraphConfig(verify=True, reasoning_level="fast")
    lg = build_multi_agent_graph_langgraph(config)
    assert config.agent_config.reasoning_level == "fast"

    config2 = MultiAgentGraphConfig(verify=True, reasoning_level="fast")
    native = build_multi_agent_graph(config2)
    assert config2.agent_config.reasoning_level == "fast"

    lg_has_verify = "verify" in lg.nodes
    native_has_verify = "verify" in native.nodes
    assert lg_has_verify == native_has_verify
