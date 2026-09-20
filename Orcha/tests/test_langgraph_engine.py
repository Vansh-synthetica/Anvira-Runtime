"""
Stage 3 tests: LangGraphEngine — LangGraph as the primary execution
engine behind Orcha's ExecutionEngine contract.

Uses the prototype bridge (orcha_node_to_langgraph) with mock Orcha
nodes — no network. Verifies: packet flow, run-context injection,
thread isolation, checkpoint history, interrupts + resume, and the
contract error when a graph does not carry the packet.
"""
import pytest

from orcha.core.packets import OrchaPacket, PacketKind
from orcha.graph.node import to_node
from orcha.integrations import LangGraphBoundary
from orcha.integrations.langgraph import (
    ExecutionInterrupt, LangGraphEngine, build_prototype_graph,
)

pytestmark = pytest.mark.skipif(
    not LangGraphBoundary().status().available,
    reason="orcha[lang] extra not installed",
)


# ── Mock Orcha nodes ─────────────────────────────────────────────────────────

async def _plan(pkt, ctx):
    return pkt.fork(pkt.kind, plan="planned", run_id_ok=str(ctx.run_id) == str(pkt.id))

async def _work(pkt, ctx):
    return pkt.fork(pkt.kind, work="worked")

async def _answer(pkt, ctx):
    return pkt.fork(pkt.kind, answer="final answer")

NODES = [
    to_node(_plan, name="plan"),
    to_node(_work, name="work"),
    to_node(_answer, name="answer"),
]

def _packet(query="benchmark me"):
    return OrchaPacket(kind=PacketKind.QUERY, query=query)


# ── Basic execution ──────────────────────────────────────────────────────────

async def test_engine_runs_packet_through_graph():
    engine = LangGraphEngine(build_prototype_graph(NODES))
    out = await engine.execute(_packet())
    assert isinstance(out, OrchaPacket)
    assert out.payload["plan"] == "planned"
    assert out.payload["work"] == "worked"
    assert out.payload["answer"] == "final answer"

async def test_run_context_is_injected_per_run():
    engine = LangGraphEngine(build_prototype_graph(NODES))
    packet = _packet()
    out = await engine.execute(packet)
    assert out.payload["run_id_ok"] is True

async def test_engine_rejects_graph_without_packet_key():
    from langgraph.graph import END, START, StateGraph

    class _OtherState(dict):
        pass

    graph = StateGraph(_OtherState)
    graph.add_node("noop", lambda state: {"other": 1})
    graph.add_edge(START, "noop")
    graph.add_edge("noop", END)
    compiled = graph.compile()
    engine = LangGraphEngine(compiled)
    with pytest.raises(TypeError, match="OrchaPacket"):
        await engine.execute(_packet())


# ── Threads / checkpoints ────────────────────────────────────────────────────

async def test_thread_isolation_between_runs():
    engine = LangGraphEngine(build_prototype_graph(NODES))
    p1 = _packet("first")
    p2 = _packet("second")
    await engine.execute(p1)
    out2 = await engine.execute(p2)
    assert out2.payload["answer"] == "final answer"
    snap1 = await engine.snapshot(p1.id)
    snap2 = await engine.snapshot(p2.id)
    assert snap1 is not None and snap2 is not None
    assert snap1["packet"].query == "first"
    assert snap2["packet"].query == "second"

async def test_checkpoint_history_records_steps():
    engine = LangGraphEngine(build_prototype_graph(NODES))
    packet = _packet("history")
    await engine.execute(packet)
    history = await engine.replay(packet.id)
    assert len(history) >= 3  # one checkpoint per node


# ── Interrupts / human-in-the-loop ───────────────────────────────────────────

async def test_interrupt_surfaces_and_resume_continues():
    graph = build_prototype_graph(NODES, interrupt_before=["work"])
    engine = LangGraphEngine(graph)
    packet = _packet("needs approval")
    with pytest.raises(ExecutionInterrupt) as exc:
        await engine.execute(packet)
    thread = exc.value.thread_id
    assert thread == packet.id
    snap = await engine.snapshot(thread)
    assert snap is not None and snap["packet"].query == "needs approval"

    out = await engine.resume(packet, thread, value="approved")
    assert out.payload["plan"] == "planned"
    assert out.payload["work"] == "worked"
    assert out.payload["answer"] == "final answer"

async def test_interrupt_does_not_consume_thread():
    engine = LangGraphEngine(build_prototype_graph(NODES, interrupt_before=["work"]))
    packet = _packet("interrupt once")
    with pytest.raises(ExecutionInterrupt):
        await engine.execute(packet)
    # Second attempt on the same thread still interrupts (no partial state leak).
    with pytest.raises(ExecutionInterrupt):
        await engine.execute(packet)


# ── Boundary status ──────────────────────────────────────────────────────────

def test_langgraph_boundary_available():
    assert LangGraphBoundary().status().available is True
    assert LangGraphBoundary().status().version
