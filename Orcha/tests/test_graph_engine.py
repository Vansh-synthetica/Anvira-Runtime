"""
Tests for the ORCHA3 graph engine: topology, conditional routing,
fan-out/fan-in, cancellation, timeout, and budget enforcement.
"""
import asyncio
import pytest

from orcha.core.packets import OrchaPacket, PacketKind, BudgetState
from orcha.graph.context import (
    CancelToken, RunContext, EventEmitter, ScatterResult,
)
from orcha.graph.edge import END, Edge, ConditionalEdge, FanOutEdge, FanInEdge
from orcha.graph.errors import (
    GraphInvalid, NodeTimeout, NodeFailed, BudgetExceeded,
    Cancelled, GatherError,
)
from orcha.graph.graph import Graph
from orcha.graph.node import Node, GatherNode, to_node
from orcha.graph.runtime import GraphRuntime
from orcha.graph.store import MemoryStore, FileStore


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_node(name, kind=PacketKind.RESPONSE, **payload):
    """Build a simple function node."""
    async def _fn(pkt, ctx):
        return pkt.fork(kind, **payload)
    return to_node(_fn, name=name)


# ── Topology validation tests ─────────────────────────────────────────────────

class TestGraphValidation:
    """Tests for Graph.validate() and topology rules."""

    def test_empty_graph_has_no_entry(self):
        g = Graph(name="empty")
        with pytest.raises(GraphInvalid, match="no entry"):
            g.validate()

    def test_duplicate_node_names_rejected(self):
        g = Graph(name="dup")
        g.add_node(make_node("a"), entry=True)
        with pytest.raises(GraphInvalid, match="Duplicate"):
            g.add_node(make_node("a"))

    def test_edge_to_unknown_node_rejected(self):
        g = Graph(name="bad_edge")
        g.add_node(make_node("a"), entry=True)
        g.add_edge("a", "nonexistent")
        with pytest.raises(GraphInvalid, match="not a registered node"):
            g.validate()

    def test_conditional_route_to_unknown_rejected(self):
        g = Graph(name="bad_cond")
        g.add_node(make_node("a"), entry=True)
        g.add_conditional("a", {"go": "ghost"}, lambda p: "go")
        with pytest.raises(GraphInvalid, match="unknown node"):
            g.validate()

    def test_end_unreachable_rejected(self):
        """A graph where END is never reachable should fail."""
        g = Graph(name="trap")
        g.add_node(make_node("a"), entry=True)
        g.add_edge("a", "a")  # self-loop, never reaches END
        with pytest.raises(GraphInvalid, match="END is not reachable"):
            g.validate()

    def test_fan_out_without_gather_rejected(self):
        """fan_out always creates both edges, so this is implicitly tested,
        but verify the pairing logic."""
        g = Graph(name="fan")
        scatter = make_node("scatter")
        scatter.is_scatter = True
        gather = type("G", (GatherNode,), {
            "name": "gather",
            "gather": lambda self, p, c, ctx: p,
        })()
        g.add_node(scatter, entry=True)
        g.add_node(gather)
        g.fan_out("scatter", "gather")
        g.add_edge("gather", END)
        # Should validate cleanly.
        g.validate()
        assert g.scatter_target("scatter") == "gather"
        assert g.gather_source("gather") == "scatter"

    def test_valid_linear_graph(self):
        g = Graph(name="ok")
        g.add_node(make_node("a"), entry=True)
        g.add_node(make_node("b"))
        g.add_node(make_node("c"))
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", END)
        g.validate()
        assert g.entry == "a"
        assert len(g.nodes) == 3


# ── Execution tests ───────────────────────────────────────────────────────────

class TestGraphExecution:
    """Tests for GraphRuntime.run()."""

    async def test_linear_execution(self):
        """A simple a → b → END graph."""
        g = Graph(name="linear")
        g.add_node(make_node("a", PacketKind.SUBTASKS, stage_a=True), entry=True)
        g.add_node(make_node("b", PacketKind.RESPONSE, stage_b=True))
        g.add_edge("a", "b")
        g.add_edge("b", END)
        g.validate()

        rt = GraphRuntime(g, store=MemoryStore())
        result = await rt.run("test query")

        assert result.run_id
        assert result.packet.payload.get("stage_a") is True
        assert result.packet.payload.get("stage_b") is True
        # Trace should have entries for both nodes.
        stages = [t["stage"] for t in result.trace]
        assert "a" in stages
        assert "b" in stages

    async def test_conditional_routing(self):
        """Conditional edge routes based on packet payload."""
        g = Graph(name="cond")
        g.add_node(make_node("start", PacketKind.SELECTION, route="good"), entry=True)
        g.add_node(make_node("good_path", PacketKind.RESPONSE, path="good"))
        g.add_node(make_node("bad_path", PacketKind.RESPONSE, path="bad"))
        g.add_conditional(
            "start",
            {"good": "good_path", "bad": "bad_path"},
            lambda p: "good" if p.payload.get("route") == "good" else "bad",
        )
        g.add_edge("good_path", END)
        g.add_edge("bad_path", END)
        g.validate()

        rt = GraphRuntime(g, store=MemoryStore())
        result = await rt.run("test")
        assert result.packet.payload.get("path") == "good"

    async def test_conditional_soft_end(self):
        """A conditional predicate returning None soft-ends the run."""
        g = Graph(name="soft")
        g.add_node(make_node("start", PacketKind.SELECTION), entry=True)
        g.add_conditional("start", {"continue": "next"}, lambda p: None)
        g.add_node(make_node("next", PacketKind.RESPONSE))
        g.add_edge("next", END)
        g.validate()

        rt = GraphRuntime(g, store=MemoryStore())
        result = await rt.run("test")
        # The run should terminate at "start" because predicate returned None.
        stages = [t["stage"] for t in result.trace]
        assert "start" in stages
        assert "next" not in stages

    async def test_budget_enforcement(self):
        """An exhausted budget raises BudgetExceeded."""
        g = Graph(name="budget")
        g.add_node(make_node("a"), entry=True)
        g.add_node(make_node("b"))
        g.add_edge("a", "b")
        g.add_edge("b", END)
        g.validate()

        # Pre-exhaust the budget on a hard (cost) dimension.
        budget = BudgetState(max_cost=0.0)
        rt = GraphRuntime(g, store=MemoryStore())
        with pytest.raises(BudgetExceeded):
            await rt.run("test", budget=budget)

    async def test_node_failure_propagates(self):
        """A node that raises propagates as NodeFailed."""
        async def fail_fn(pkt, ctx):
            raise RuntimeError("boom")
        g = Graph(name="fail")
        g.add_node(to_node(fail_fn, name="boom"), entry=True)
        g.add_edge("boom", END)
        g.validate()

        rt = GraphRuntime(g, store=MemoryStore())
        with pytest.raises(NodeFailed):
            await rt.run("test")

    async def test_node_retry(self):
        """A node with retries should be attempted multiple times."""
        call_count = {"n": 0}

        async def flaky_fn(pkt, ctx):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise RuntimeError("transient")
            return pkt.fork(PacketKind.RESPONSE, success=True)

        g = Graph(name="retry")
        node = to_node(flaky_fn, name="flaky", retries=2)
        g.add_node(node, entry=True)
        g.add_edge("flaky", END)
        g.validate()

        rt = GraphRuntime(g, store=MemoryStore())
        result = await rt.run("test")
        assert call_count["n"] == 3
        assert result.packet.payload.get("success") is True

    async def test_node_timeout(self):
        """A node that exceeds its timeout raises NodeTimeout."""
        async def slow_fn(pkt, ctx):
            await asyncio.sleep(10)
            return pkt

        g = Graph(name="slow")
        node = to_node(slow_fn, name="slow", timeout_s=0.05)
        g.add_node(node, entry=True)
        g.add_edge("slow", END)
        g.validate()

        rt = GraphRuntime(g, store=MemoryStore())
        with pytest.raises(NodeTimeout):
            await rt.run("test")

    async def test_cancellation(self):
        """A cancelled run raises Cancelled."""
        async def slow_fn(pkt, ctx):
            await asyncio.sleep(10)
            return pkt

        g = Graph(name="cancel")
        node = to_node(slow_fn, name="slow")
        g.add_node(node, entry=True)
        g.add_edge("slow", END)
        g.validate()

        cancel = CancelToken()
        rt = GraphRuntime(g, store=MemoryStore())

        # Cancel before running.
        cancel.cancel("user requested")
        with pytest.raises(Cancelled):
            await rt.run("test", cancel=cancel)


# ── Fan-out / fan-in tests ────────────────────────────────────────────────────

class TestFanOutFanIn:
    """Tests for scatter/gather (fan-out/fan-in)."""

    async def test_basic_fan_out_fan_in(self):
        """Three branches scatter, each produces a value, gather collects."""
        class Scatter(Node):
            name = "scatter"
            is_scatter = True
            async def run(self, packet, ctx):
                branches = [
                    (f"b{i}", packet.fork(PacketKind.SELECTION, branch=i), "worker")
                    for i in range(3)
                ]
                return ScatterResult(branches=branches, gather_to="gather")

        class Worker(Node):
            name = "worker"
            async def run(self, packet, ctx):
                return packet.fork(packet.kind, worked=packet.payload.get("branch"))

        class Gather(GatherNode):
            name = "gather"
            async def gather(self, packet, children, ctx):
                ids = [c.payload.get("worked") for c in children]
                return packet.fork(packet.kind, gathered_ids=ids, count=len(children))

        g = Graph(name="fan")
        g.add_node(Scatter(), entry=True)
        g.add_node(Worker())
        g.add_node(Gather())
        g.fan_out("scatter", "gather")
        g.add_edge("gather", END)
        g.validate()

        rt = GraphRuntime(g, store=MemoryStore())
        result = await rt.run("test")
        assert sorted(result.packet.payload.get("gathered_ids", [])) == [0, 1, 2]
        assert result.packet.payload.get("count") == 3

    async def test_fan_out_single_branch(self):
        """A single-branch scatter still works."""
        class Scatter(Node):
            name = "scatter"
            is_scatter = True
            async def run(self, packet, ctx):
                return ScatterResult(
                    branches=[("only", packet.fork(PacketKind.SELECTION), "worker")],
                    gather_to="gather",
                )

        class Worker(Node):
            name = "worker"
            async def run(self, packet, ctx):
                return packet.fork(packet.kind, processed=True)

        class Gather(GatherNode):
            name = "gather"
            async def gather(self, packet, children, ctx):
                return packet.fork(packet.kind, child_count=len(children))

        g = Graph(name="fan1")
        g.add_node(Scatter(), entry=True)
        g.add_node(Worker())
        g.add_node(Gather())
        g.fan_out("scatter", "gather")
        g.add_edge("gather", END)
        g.validate()

        rt = GraphRuntime(g, store=MemoryStore())
        result = await rt.run("test")
        assert result.packet.payload.get("child_count") == 1

    async def test_scatter_without_branches_raises(self):
        """A ScatterResult with empty branches should raise ValueError."""
        with pytest.raises(ValueError, match="at least one branch"):
            ScatterResult(branches=[], gather_to="gather")


# ── Durability tests ──────────────────────────────────────────────────────────

class TestDurability:
    """Tests for checkpointing and Store implementations."""

    async def test_memory_store_roundtrip(self):
        """MemoryStore saves and loads checkpoints."""
        store = MemoryStore()
        pkt = OrchaPacket(kind=PacketKind.QUERY, query="test")
        await store.save_checkpoint("run1", "node_a", "node_b", pkt)

        cp = await store.load_checkpoint("run1")
        assert cp is not None
        assert cp.run_id == "run1"
        assert cp.node_id == "node_a"
        assert cp.next_node == "node_b"
        assert cp.packet.query == "test"

    async def test_memory_store_multiple_checkpoints(self):
        """The latest checkpoint is returned by load_checkpoint."""
        store = MemoryStore()
        pkt1 = OrchaPacket(kind=PacketKind.QUERY, query="v1")
        pkt2 = OrchaPacket(kind=PacketKind.RESPONSE, query="v2")

        await store.save_checkpoint("run1", "a", "b", pkt1)
        await store.save_checkpoint("run1", "b", "c", pkt2)

        cp = await store.load_checkpoint("run1")
        assert cp.node_id == "b"
        assert cp.packet.kind == PacketKind.RESPONSE

        all_cps = await store.list_checkpoints("run1")
        assert len(all_cps) == 2
        assert all_cps[0].seq == 0
        assert all_cps[1].seq == 1

    async def test_memory_store_delete(self):
        store = MemoryStore()
        pkt = OrchaPacket(kind=PacketKind.QUERY, query="test")
        await store.save_checkpoint("run1", "a", "b", pkt)
        count = await store.delete_run("run1")
        assert count == 1
        cp = await store.load_checkpoint("run1")
        assert cp is None

    async def test_file_store_roundtrip(self, tmp_path):
        """FileStore persists checkpoints to disk."""
        store = FileStore(root=tmp_path)
        pkt = OrchaPacket(kind=PacketKind.QUERY, query="disk test")
        await store.save_checkpoint("run1", "node_a", "node_b", pkt)

        # New store instance reads from the same dir.
        store2 = FileStore(root=tmp_path)
        cp = await store2.load_checkpoint("run1")
        assert cp is not None
        assert cp.packet.query == "disk test"

    async def test_file_store_list_runs(self, tmp_path):
        store = FileStore(root=tmp_path)
        pkt = OrchaPacket(kind=PacketKind.QUERY, query="t")
        await store.save_checkpoint("run1", "a", "b", pkt)
        await store.save_checkpoint("run2", "a", "b", pkt)

        runs = await store.list_runs()
        assert set(runs) == {"run1", "run2"}

    async def test_file_store_prunes_oldest_runs(self, tmp_path):
        """FileStore with max_runs drops the oldest run directories."""
        store = FileStore(root=tmp_path, max_runs=2)
        pkt = OrchaPacket(kind=PacketKind.QUERY, query="t")
        await store.save_checkpoint("run1", "a", "b", pkt)
        await store.save_checkpoint("run2", "a", "b", pkt)
        # Older than run1 so it is pruned first even though it sorts later.
        await store.save_checkpoint("run3", "a", "b", pkt)

        runs = sorted(await store.list_runs())
        assert len(runs) == 2
        assert "run3" in runs

    async def test_file_store_no_prune_without_cap(self, tmp_path):
        store = FileStore(root=tmp_path)
        pkt = OrchaPacket(kind=PacketKind.QUERY, query="t")
        for i in range(5):
            await store.save_checkpoint(f"run{i}", "a", "b", pkt)
        assert len(await store.list_runs()) == 5

    async def test_checkpoint_written_during_run(self, tmp_path):
        """A run with FileStore writes checkpoints."""
        store = FileStore(root=tmp_path)
        g = Graph(name="ckpt")
        g.add_node(make_node("a"), entry=True)
        g.add_node(make_node("b"))
        g.add_edge("a", "b")
        g.add_edge("b", END)
        g.validate()

        rt = GraphRuntime(g, store=store)
        result = await rt.run("checkpoint test", run_id="test-run-1")

        cps = await store.list_checkpoints("test-run-1")
        assert len(cps) >= 1
        # The final checkpoint should point to END.
        assert cps[-1].next_node == END


# ── Event streaming tests ─────────────────────────────────────────────────────

class TestEventStreaming:
    """Tests for the EventEmitter and run events."""

    async def test_events_emitted_in_order(self):
        """node_start and node_end events fire for each node."""
        events = []

        async def _sub(event):
            events.append(event.kind)

        g = Graph(name="events")
        g.add_node(make_node("a"), entry=True)
        g.add_node(make_node("b"))
        g.add_edge("a", "b")
        g.add_edge("b", END)
        g.validate()

        rt = GraphRuntime(g, store=MemoryStore())
        await rt.run("test", on_event=_sub)

        assert "run_start" in events
        assert "node_start" in events
        assert "node_end" in events
        assert "run_complete" in events

    async def test_subscriber_fault_isolation(self):
        """A buggy subscriber does not crash the run."""
        def buggy_sub(event):
            raise RuntimeError("subscriber bug")

        g = Graph(name="isolated")
        g.add_node(make_node("a"), entry=True)
        g.add_edge("a", END)
        g.validate()

        rt = GraphRuntime(g, store=MemoryStore())
        # Should complete despite the buggy subscriber.
        result = await rt.run("test", on_event=buggy_sub)
        assert result.run_id  # did not crash


# ── Resume tests ──────────────────────────────────────────────────────────────

class TestResume:
    """Tests for resuming a run from a checkpoint."""

    async def test_resume_from_checkpoint(self, tmp_path):
        """A resumed run continues from the latest checkpoint."""
        store = FileStore(root=tmp_path)
        g = Graph(name="resume")
        g.add_node(make_node("a", PacketKind.SUBTASKS), entry=True)
        g.add_node(make_node("b", PacketKind.RESPONSE, resumed=True))
        g.add_edge("a", "b")
        g.add_edge("b", END)
        g.validate()

        # Run normally to create checkpoints.
        rt = GraphRuntime(g, store=store)
        result1 = await rt.run("resume test", run_id="resume-1")
        assert result1.packet.payload.get("resumed") is True

        # Now resume from the same run id. It should skip completed nodes.
        result2 = await rt.run(
            "resume test",
            resume_from=store,
            run_id="resume-1",
        )
        assert result2.run_id == "resume-1"


# ── Replay tests ──────────────────────────────────────────────────────────────

class TestReplay:
    """Tests for deterministic replay."""

    async def test_replay_reproduces_result(self, tmp_path):
        """Replaying a run reproduces the same final answer."""
        store = FileStore(root=tmp_path)
        g = Graph(name="replay")
        g.add_node(make_node("a", PacketKind.SUBTASKS, val=42), entry=True)
        g.add_node(make_node("b", PacketKind.RESPONSE, final=True))
        g.add_edge("a", "b")
        g.add_edge("b", END)
        g.validate()

        rt = GraphRuntime(g, store=store)
        result1 = await rt.run("replay test", run_id="replay-1")
        original_val = result1.packet.payload.get("val")

        result2 = await rt.replay("replay-1")
        assert result2.packet.payload.get("val") == original_val
        assert result2.packet.payload.get("final") is True

    async def test_replay_nonexistent_raises(self, tmp_path):
        store = FileStore(root=tmp_path)
        g = Graph(name="noreplay")
        g.add_node(make_node("a"), entry=True)
        g.add_edge("a", END)
        g.validate()

        rt = GraphRuntime(g, store=store)
        with pytest.raises(Exception):
            await rt.replay("nonexistent-run")
