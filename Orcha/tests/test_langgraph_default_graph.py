"""
Stage 4 — LangGraph default graph: parity & engine switch tests.

The default ORCHA3 pipeline (decompose → plan → select → execute →
aggregate → evaluate → retry) is now available on two engines:

  - native:      ``GraphRuntime`` + validated ``Graph`` (reference impl)
  - langgraph:   genuine LangGraph ``StateGraph`` reusing the same
                 ORCHA2 stage nodes (``orcha.builders.langgraph_default``)

``build_default_runner`` switches engines; both runners expose the same
surface (``run`` / ``replay`` / ``live_emitter``) and must produce
identical ``RunResult`` objects, traces, and event streams for the same
deterministic inputs.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from orcha.builders import (
    LangGraphDefaultRunner,
    build_default_graph,
    build_default_graph_langgraph,
    build_default_runner,
    DefaultGraphConfig,
)
from orcha.experts.base import BaseExpert, ExpertOutput
from orcha.core.packets import BudgetState, OrchaPacket, PacketKind
from orcha.graph import CancelToken, GraphRuntime
from orcha.graph.edge import END


# ── Deterministic fixtures ────────────────────────────────────────────────────

class FixedExpert(BaseExpert):
    """Fully deterministic expert: fixed confidence and answer text."""

    domain = "general"
    description = "deterministic test expert"
    cost_per_1k_tokens = 0.001

    def __init__(self, name: str, conf: float) -> None:
        self.name = name
        self._conf = conf

    async def execute(self, query: str) -> ExpertOutput:
        long = (
            "This is a complete and detailed answer about the topic. "
            "It covers the main definitions, the key mechanisms, the "
            "most important examples, and the practical implications "
            "for real-world use cases. It is thorough, well-structured, "
            "and directly addresses every part of the question."
        )
        return ExpertOutput(
            answer=f"answer to {query}: {long}", confidence=self._conf,
            tokens_used=10, finish_reason="stop",
        )


def passing_experts() -> dict:
    return {
        "expert_a": FixedExpert("expert_a", 0.9),
        "expert_b": FixedExpert("expert_b", 0.9),
    }


class RaisingExpert(BaseExpert):
    """Always fails — deterministic trigger for the retry loop."""

    domain = "general"
    description = "always raises"
    cost_per_1k_tokens = 0.001

    def __init__(self, name: str) -> None:
        self.name = name

    async def execute(self, query: str) -> ExpertOutput:
        raise RuntimeError("backend down")


def failing_experts() -> dict:
    return {
        "expert_a": RaisingExpert("expert_a"),
        "expert_b": RaisingExpert("expert_b"),
    }


def make_packet(query: str, run_id: str, max_iterations: int = 3) -> OrchaPacket:
    return OrchaPacket(
        id=run_id, kind=PacketKind.QUERY, query=query,
        payload={"force_all_experts": True, "__graph__": "orcha_default"},
        budget=BudgetState(max_iterations=max_iterations),
    )


async def _run_both(engine_args: dict, query: str = "What is machine learning?",
                    run_id: str = "parity") -> dict:
    """Run the same scenario on both engines and return {engine: RunResult}."""
    outs = {}
    for engine in ("native", "langgraph"):
        runner = build_default_runner(engine=engine, **engine_args)
        outs[engine] = await runner.run(
            "", packet=make_packet(query, f"{run_id}-{engine}"),
        )
    return outs


# ── Engine switch / builder surface ──────────────────────────────────────────

class TestEngineSwitch:
    def test_config_engine_field(self):
        cfg = DefaultGraphConfig(engine="langgraph", experts=passing_experts())
        runner = build_default_runner(cfg)
        assert isinstance(runner, LangGraphDefaultRunner)

    def test_env_var_switch(self, monkeypatch):
        monkeypatch.setenv("ORCHA_DEFAULT_ENGINE", "langgraph")
        runner = build_default_runner(experts=passing_experts())
        assert isinstance(runner, LangGraphDefaultRunner)

    def test_explicit_engine_wins_over_config(self):
        cfg = DefaultGraphConfig(engine="langgraph", experts=passing_experts())
        runner = build_default_runner(cfg, engine="native")
        assert isinstance(runner, GraphRuntime)

    def test_default_is_langgraph(self):
        runner = build_default_runner(experts=passing_experts())
        assert isinstance(runner, LangGraphDefaultRunner)

    def test_unknown_engine_raises(self):
        with pytest.raises(ValueError, match="Unknown engine"):
            build_default_runner(experts=passing_experts(), engine="quantum")

    def test_langgraph_runner_surface(self):
        runner = build_default_runner(
            engine="langgraph", experts=passing_experts(),
        )
        assert isinstance(runner, LangGraphDefaultRunner)
        assert callable(runner.run)
        assert callable(runner.replay)
        assert runner.live_emitter is None
        assert runner.graph_name == "orcha_default"

    def test_native_builder_unchanged(self):
        """The native builder still yields the reference Graph topology."""
        graph = build_default_graph(
            experts=passing_experts(), max_iterations=2,
        )
        assert graph.name == "orcha_default"
        assert graph.entry == "decompose"
        expected = {"decompose", "plan", "select", "execute",
                    "aggregate", "evaluate", "retry", "finalize"}
        assert expected.issubset(set(graph.nodes.keys()))

    def test_langgraph_builder_topology(self):
        graph = build_default_graph_langgraph(
            DefaultGraphConfig(experts=passing_experts()),
        )
        nodes = set(graph.get_graph().nodes)
        for expected in ("decompose", "plan", "select", "execute",
                         "aggregate", "evaluate", "retry", "finalize"):
            assert expected in nodes
        # Conditional routing must mirror the native builder.
        cond = {
            (e.source, e.data): e.target
            for e in graph.get_graph().edges if e.conditional
        }
        assert cond[("plan", "stop")] == "finalize"
        assert cond[("plan", "continue")] == "select"
        assert cond[("retry", "retry")] == "plan"
        assert cond[("retry", "stop")] == "finalize"

    def test_same_node_instances_across_engines(self):
        """Both builders drive byte-identical ORCHA2 stage nodes."""
        native = build_default_graph(experts=passing_experts())
        lang = build_default_graph_langgraph(
            DefaultGraphConfig(experts=passing_experts()),
        )
        lg_nodes = {
            name: getattr(n.data, "afunc", n.data).__wrapped_node__
            for name, n in lang.get_graph().nodes.items()
            if name in native.nodes
        }
        for name in native.nodes:
            assert type(lg_nodes[name]).__name__ == type(native.nodes[name]).__name__


# ── Deterministic parity ──────────────────────────────────────────────────────

class TestParity:
    async def test_run_result_parity(self):
        outs = await _run_both({"experts": passing_experts()})
        a, b = outs["native"], outs["langgraph"]
        assert a.answer == b.answer
        assert a.confidence == b.confidence
        assert a.quality_score == b.quality_score
        assert a.synthesized == b.synthesized
        assert a.primary == b.primary
        assert a.domains == b.domains
        assert a.agg_mode == b.agg_mode
        assert a.iterations == b.iterations
        assert a.cost == b.cost

    async def test_trace_parity(self):
        outs = await _run_both({"experts": passing_experts()})
        assert [t["stage"] for t in outs["native"].trace] == \
               [t["stage"] for t in outs["langgraph"].trace]

    async def test_event_stream_parity(self):
        streams = {}
        for engine in ("native", "langgraph"):
            runner = build_default_runner(engine=engine, experts=passing_experts())
            events = []
            await runner.run(
                "", packet=make_packet("What is machine learning?", f"ev-{engine}"),
                on_event=lambda ev, evs=events: evs.append((ev.kind, ev.node)),
            )
            streams[engine] = events
        assert streams["native"] == streams["langgraph"]
        assert ("run_start", "") in streams["langgraph"]
        assert ("run_complete", "") in streams["langgraph"]

    async def test_retry_loop_parity(self):
        """Failing evaluation loops identically and stops gracefully at the
        iteration cap (the retry controller's budget_exhausted stop — not a
        hard BudgetExceeded; that now only fires on cost/latency overspend)."""
        results = {}
        for engine in ("native", "langgraph"):
            runner = build_default_runner(engine=engine, experts=failing_experts())
            result = await runner.run(
                "", packet=make_packet("retry me", f"ret-{engine}", 3),
            )
            results[engine] = result
        a, b = results["native"], results["langgraph"]
        assert a.answer == b.answer
        assert a.iterations == b.iterations == 3
        assert a.packet.payload.get("retry_count") == \
            b.packet.payload.get("retry_count")
        assert a.packet.payload.get("retry_reason") == \
            b.packet.payload.get("retry_reason") == "budget_exhausted"

    async def test_retry_count_and_reason_parity(self):
        """When retries resolve (pass on iteration 2), payload matches."""
        mixed = {
            "expert_a": FixedExpert("expert_a", 0.2),
            # expert_b passes only after expert_a was excluded — force via
            # a selector-free run: use run_all_experts and a passing expert.
            "expert_b": FixedExpert("expert_b", 0.9),
        }
        outs = await _run_both({"experts": mixed}, query="retry then pass",
                               run_id="retpass")
        a, b = outs["native"], outs["langgraph"]
        assert a.packet.payload.get("retry_count") == b.packet.payload.get("retry_count")
        assert a.packet.payload.get("retry_reason") == b.packet.payload.get("retry_reason")
        assert a.answer == b.answer

    async def test_cancel_parity(self):
        for engine in ("native", "langgraph"):
            runner = build_default_runner(engine=engine, experts=passing_experts())
            token = CancelToken()
            token.cancel("stop-now")
            with pytest.raises(Exception) as exc:
                await runner.run(
                    "", packet=make_packet("x", f"cancel-{engine}"),
                    cancel=token,
                )
            assert type(exc.value).__name__ == "Cancelled"
            assert exc.value.node == "decompose"

    async def test_budget_parity(self):
        for engine in ("native", "langgraph"):
            runner = build_default_runner(engine=engine, experts=passing_experts())
            with pytest.raises(Exception) as exc:
                await runner.run(
                    "", packet=make_packet("x", f"budget-{engine}"),
                    budget=BudgetState(max_cost=0.0),
                )
            assert type(exc.value).__name__ == "BudgetExceeded"

    async def test_packet_contract_unchanged(self):
        """Public packet/run contract: same run_id, kind and payload keys."""
        outs = await _run_both({"experts": passing_experts()})
        a, b = outs["native"], outs["langgraph"]
        assert a.packet.kind.value == b.packet.kind.value
        assert set(a.packet.payload) == set(b.packet.payload)


# ── Stage 5: config knobs now reach the run (was: dead) ───────────────────────

class TestConfigKnobs:
    """DefaultGraphConfig.max_iterations / run_all_experts actually reach
    the packet budget/payload on BOTH engines (regression: they used to be
    dead knobs — the run always used BudgetState() defaults)."""

    async def test_config_max_iterations_honored_both_engines(self):
        """max_iterations=1 is a live cap: exactly one pass, graceful stop.
        (With the historical dead knob the budget stayed at the default 3
        and failing experts would have looped three times.)"""
        for engine in ("native", "langgraph"):
            runner = build_default_runner(
                engine=engine, experts=failing_experts(), max_iterations=1,
            )
            result = await runner.run("What is machine learning?")
            assert result.iterations == 1
            assert result.packet.payload.get("retry_reason") == "budget_exhausted"

    async def test_config_run_all_experts_honored_both_engines(self):
        """run_all_experts=True sets payload force_all_experts → selector
        runs the full expert pool (previously never reached the selector)."""
        for engine in ("native", "langgraph"):
            runner = build_default_runner(
                engine=engine, experts=passing_experts(), run_all_experts=True,
            )
            result = await runner.run("What is machine learning?")
            names = {r.name for r in result.packet.get_results()}
            assert {"expert_a", "expert_b"} <= names
            assert result.packet.payload.get("force_all_experts") is True

    async def test_explicit_packet_still_wins_over_config(self):
        """A caller-supplied packet/budget is never overridden by config."""
        runner = build_default_runner(
            engine="native", experts=passing_experts(), max_iterations=1,
        )
        result = await runner.run(
            "", packet=make_packet("q", "explicit", max_iterations=3),
        )
        assert result.packet.budget.max_iterations == 3


class TestResumeParity:
    async def test_resume_completed_run_is_idempotent_both_engines(self):
        """Resuming a COMPLETED run starts fresh on both engines — no spent
        budget, no BudgetExceeded (regression: native reused the final
        checkpoint's spent-budget packet and blew up at the next walk-top)."""
        for engine in ("native", "langgraph"):
            runner = build_default_runner(engine=engine, experts=passing_experts())
            first = await runner.run(
                "", packet=make_packet("What is machine learning?", f"done-{engine}"),
            )
            again = await runner.run(
                "", packet=make_packet("What is machine learning?", f"done-{engine}"),
                resume_from=runner.store,
            )
            assert again.answer == first.answer
            assert again.iterations == first.iterations == 1

    async def test_resume_interrupted_run_continues_from_checkpoint(self):
        """Resuming a run interrupted mid-walk continues from the last
        checkpoint (native semantics preserved for genuinely paused runs)."""
        from orcha.graph.graph import Graph
        from orcha.graph.node import to_node
        from orcha.graph.store import MemoryStore

        async def node_a(pkt, ctx):
            return pkt.fork(pkt.kind, a_ran=True)

        async def node_b(pkt, ctx):
            raise RuntimeError("boom")

        g = Graph(name="interruptible")
        g.add_node(to_node(node_a, name="a"), entry=True)
        g.add_node(to_node(node_b, name="b"))
        g.add_edge("a", "b")
        g.add_edge("b", END)
        g.validate()

        rt = GraphRuntime(g, store=MemoryStore())
        with pytest.raises(Exception):
            await rt.run("q", run_id="inter-1")
        stages = []
        with pytest.raises(Exception):
            await rt.run(
                "q", run_id="inter-1", resume_from=rt.store,
                on_event=lambda ev, s=stages: s.append(ev.node)
                if ev.kind == "node_start" else None,
            )
        assert stages == ["b"]  # resumed directly at the interrupted node


# ── LangGraph-only capabilities (documented divergences) ─────────────────────

class TestLangGraphCapabilities:
    async def test_replay_roundtrip(self):
        runner = build_default_runner(engine="langgraph", experts=passing_experts())
        original = await runner.run(
            "", packet=make_packet("What is machine learning?", "replay-1"),
        )
        events = []
        replayed = await runner.replay(
            "replay-1", on_event=lambda ev: events.append(ev.kind),
        )
        assert replayed.answer == original.answer
        assert replayed.run_id == "replay-1"
        assert events[0] == "run_start"
        assert events[-1] == "run_complete"

    async def test_replay_without_history_raises(self):
        runner = build_default_runner(engine="langgraph", experts=passing_experts())
        with pytest.raises(Exception, match="No checkpoints"):
            await runner.replay("never-ran")

    async def test_resume_without_pending_thread_starts_fresh(self):
        runner = build_default_runner(engine="langgraph", experts=passing_experts())
        first = await runner.run(
            "", packet=make_packet("What is machine learning?", "resume-1"),
        )
        again = await runner.run(
            "", packet=make_packet("What is machine learning?", "resume-1"),
            resume_from=object(),  # no pending work on that thread
        )
        assert again.answer == first.answer

    async def test_run_id_keys_langgraph_thread(self):
        """Each run creates its own LangGraph thread (run_id == thread id)."""
        runner = build_default_runner(engine="langgraph", experts=passing_experts())
        r1 = await runner.run(
            "", packet=make_packet("q1", "thread-1"),
        )
        snap = await runner.engine.snapshot("thread-1")
        assert snap is not None and snap.get("packet") is not None
        assert r1.packet.payload.get("retry") is not True
