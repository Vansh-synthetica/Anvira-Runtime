"""
Tests for ORCHA3 node types: verify, tool, retrieval, and agent nodes.
Also tests the stage wrappers and graph builders.
"""
import asyncio
import pytest

from orcha.core.packets import OrchaPacket, PacketKind, BudgetState
from orcha.graph.context import RunContext, CancelToken, EventEmitter
from orcha.graph.edge import END
from orcha.graph.graph import Graph
from orcha.graph.node import to_node, GatherNode, Node
from orcha.graph.runtime import GraphRuntime
from orcha.graph.store import MemoryStore
from orcha.observability import get_logger


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_ctx(run_id="test-run"):
    """Build a minimal RunContext for unit-testing nodes directly."""
    return RunContext(
        run_id=run_id,
        store=MemoryStore(),
        cancel=CancelToken(),
        emit=EventEmitter(run_id, logger=get_logger("test")),
        logger=get_logger("test"),
        graph_name="test",
    )


def make_packet(**payload):
    pkt = OrchaPacket(
        kind=PacketKind.AGGREGATION,
        query="test query about machine learning",
        payload=payload,
    )
    return pkt


# ── VerifyNode tests ──────────────────────────────────────────────────────────

class TestVerifyNode:
    """Tests for the VerifyNode."""

    async def test_verify_passes_good_answer(self):
        from orcha.nodes.verify import VerifyNode, VerifyCriteria

        node = VerifyNode(
            criteria=VerifyCriteria(min_confidence=0.5, min_answer_length=10),
        )
        pkt = make_packet(answer="This is a good answer about machine learning.", confidence=0.8)
        result = await node.run(pkt, make_ctx())

        assert result.payload.get("verified") is True
        assert result.payload.get("verify_score") > 0.5
        assert "verify_dimensions" in result.payload

    async def test_verify_fails_low_confidence(self):
        from orcha.nodes.verify import VerifyNode, VerifyCriteria

        node = VerifyNode(criteria=VerifyCriteria(min_confidence=0.9))
        pkt = make_packet(answer="A good long answer here.", confidence=0.3)
        result = await node.run(pkt, make_ctx())

        assert result.payload.get("verified") is False
        assert len(result.payload.get("verify_failures", [])) > 0

    async def test_verify_fails_short_answer(self):
        from orcha.nodes.verify import VerifyNode, VerifyCriteria

        node = VerifyNode(criteria=VerifyCriteria(min_answer_length=100))
        pkt = make_packet(answer="short", confidence=0.9)
        result = await node.run(pkt, make_ctx())

        assert result.payload.get("verified") is False

    async def test_verify_custom_check(self):
        from orcha.nodes.verify import VerifyNode, VerifyCriteria

        # Custom check: answer must contain "learning".
        def contains_learning(text):
            return "learning" in text.lower()

        node = VerifyNode(
            criteria=VerifyCriteria(
                min_confidence=0.0,
                min_answer_length=1,
                custom_checks=[("has_keyword", contains_learning)],
            ),
        )
        pkt = make_packet(answer="Machine learning is great.", confidence=0.5)
        result = await node.run(pkt, make_ctx())
        assert result.payload.get("verified") is True

        pkt2 = make_packet(answer="Nothing relevant here.", confidence=0.5)
        result2 = await node.run(pkt2, make_ctx())
        assert result2.payload.get("verified") is False


# ── CriticNode tests ──────────────────────────────────────────────────────────

class TestCriticNode:
    """Tests for the CriticNode."""

    async def test_critic_produces_dimensions(self):
        from orcha.nodes.verify import CriticNode

        node = CriticNode(threshold=0.3)
        pkt = make_packet(
            answer="Machine learning involves neural networks with 3 layers.",
            confidence=0.7,
        )
        result = await node.run(pkt, make_ctx())

        dims = result.payload.get("critic_dimensions", {})
        assert "completeness" in dims
        assert "relevance" in dims
        assert "coherence" in dims
        assert "specificity" in dims
        assert "safety" in dims
        assert 0.0 <= result.payload.get("critic_score", 0) <= 1.0

    async def test_critic_threshold(self):
        from orcha.nodes.verify import CriticNode

        node = CriticNode(threshold=0.99)  # almost impossible to pass
        pkt = make_packet(answer="ok", confidence=0.1)
        result = await node.run(pkt, make_ctx())
        assert result.payload.get("critic_passed") is False


# ── FactCheckNode tests ───────────────────────────────────────────────────────

class TestFactCheckNode:
    """Tests for the FactCheckNode."""

    async def test_fact_check_no_retrieval_passes(self):
        from orcha.nodes.verify import FactCheckNode

        node = FactCheckNode()
        pkt = make_packet(answer="Some answer.")
        result = await node.run(pkt, make_ctx())
        # No retrieval results → passes by default.
        assert result.payload.get("fact_check_passed") is True

    async def test_fact_check_with_supporting_evidence(self):
        from orcha.nodes.verify import FactCheckNode

        node = FactCheckNode(threshold=0.3)
        pkt = make_packet(
            answer="Python is a programming language. It supports multiple paradigms.",
            retrieval_results=[
                "Python is a popular programming language created by Guido van Rossum.",
                "Python supports object-oriented, functional, and procedural paradigms.",
            ],
        )
        result = await node.run(pkt, make_ctx())
        assert result.payload.get("fact_check_passed") is True
        assert result.payload.get("fact_check_score", 0) > 0

    async def test_fact_check_with_unsupported_claims(self):
        from orcha.nodes.verify import FactCheckNode

        node = FactCheckNode(threshold=0.9)
        # Answer contains claims with NO keyword overlap to the corpus.
        pkt = make_packet(
            answer="Quantum flux capacitors generate tachyon beams via dark energy crystals.",
            retrieval_results=[
                "The weather today is sunny with mild temperatures.",
                "Cooking pasta requires boiling water and salt.",
            ],
        )
        result = await node.run(pkt, make_ctx())
        # The fabricated terms have no overlap → low support score.
        score = result.payload.get("fact_check_score", 1.0)
        assert score < 0.9
        assert result.payload.get("fact_check_passed") is False


# ── ToolNode tests ────────────────────────────────────────────────────────────

class TestToolNode:
    """Tests for the ToolNode."""

    async def test_tool_node_executes_function(self):
        from orcha.nodes.tool import ToolNode, ToolSpec

        def calculator(arg):
            try:
                return str(eval(arg))
            except Exception:
                return "error"

        spec = ToolSpec(name="calc", description="A calculator", fn=calculator)
        node = ToolNode(spec)
        pkt = make_packet(tool_input="2 + 3")
        result = await node.run(pkt, make_ctx())

        assert result.payload.get("tool_output") == "5"
        assert result.payload.get("tool_success") is True
        assert result.payload.get("tool_name") == "calc"

    async def test_tool_node_handles_error(self):
        from orcha.nodes.tool import ToolNode, ToolSpec

        def failing_tool(arg):
            raise ValueError("boom")

        spec = ToolSpec(name="fail", description="Always fails", fn=failing_tool)
        node = ToolNode(spec)
        pkt = make_packet(tool_input="anything")
        result = await node.run(pkt, make_ctx())

        assert result.payload.get("tool_success") is False
        assert "boom" in result.payload.get("tool_output", "")

    async def test_tool_node_falls_back_to_query(self):
        from orcha.nodes.tool import ToolNode, ToolSpec

        def echo(arg):
            return f"echo:{arg}"

        spec = ToolSpec(name="echo", description="Echo", fn=echo)
        node = ToolNode(spec)
        pkt = make_packet()  # no tool_input
        result = await node.run(pkt, make_ctx())
        assert result.payload.get("tool_output") == "echo:test query about machine learning"


# ── RetrievalNode tests ───────────────────────────────────────────────────────

class TestRetrievalNode:
    """Tests for the RetrievalNode."""

    async def test_retrieval_with_keyword_retriever(self):
        from orcha.nodes.retrieval import RetrievalNode, RetrievalConfig, keyword_retriever

        docs = [
            "Machine learning is a subset of artificial intelligence.",
            "Python is a programming language.",
            "Neural networks are used in deep learning.",
        ]
        retriever = keyword_retriever(docs)
        node = RetrievalNode(
            config=RetrievalConfig(retriever=retriever, top_k=2, source_name="keyword"),
        )
        pkt = make_packet()
        # Override the query for retrieval.
        pkt = pkt.model_copy(update={"query": "machine learning"})
        result = await node.run(pkt, make_ctx())

        assert result.payload.get("retrieval_count") == 2
        assert len(result.payload.get("retrieval_results", [])) == 2
        assert result.payload.get("retrieval_source") == "keyword"

    async def test_retrieval_no_retriever_passes_through(self):
        from orcha.nodes.retrieval import RetrievalNode, RetrievalConfig

        node = RetrievalNode(config=RetrievalConfig(retriever=None))
        pkt = make_packet()
        result = await node.run(pkt, make_ctx())
        assert result.payload.get("retrieval_count") == 0
        assert result.payload.get("retrieval_results") == []

    async def test_keyword_retriever_ranks_by_overlap(self):
        from orcha.nodes.retrieval import keyword_retriever

        docs = [
            "completely unrelated text about cooking",
            "machine learning neural networks deep learning AI",
            "some text about data science",
        ]
        retriever = keyword_retriever(docs)
        results = retriever("machine learning", 2)
        assert len(results) == 2
        # The most relevant doc should be first.
        assert "machine learning" in results[0].lower()


# ── AgentNode tests ───────────────────────────────────────────────────────────

class TestAgentNode:
    """Tests for the AgentNode."""

    async def test_agent_dry_run_completes(self):
        from orcha.nodes.agent import AgentNode, AgentConfig

        # No model_fn → dry run mode.
        config = AgentConfig(max_iterations=2)
        node = AgentNode(config=config)
        pkt = make_packet(task="Summarize machine learning")
        result = await node.run(pkt, make_ctx())

        assert result.payload.get("agent_iterations", 0) > 0
        assert "agent_output" in result.payload
        assert "agent_steps" in result.payload

    async def test_agent_with_stop_phrase(self):
        from orcha.nodes.agent import AgentNode, AgentConfig

        async def mock_model(prompt, system):
            return "FINAL ANSWER: The answer is 42."

        config = AgentConfig(max_iterations=5, model_fn=mock_model)
        node = AgentNode(config=config)
        pkt = make_packet(task="What is the answer?")
        result = await node.run(pkt, make_ctx())

        assert result.payload.get("agent_completed") is True
        assert result.payload.get("agent_output") == "The answer is 42."

    async def test_agent_max_iterations(self):
        from orcha.nodes.agent import AgentNode, AgentConfig

        async def mock_model(prompt, system):
            return "Still working..."  # never produces stop phrase

        config = AgentConfig(max_iterations=3, model_fn=mock_model)
        node = AgentNode(config=config)
        pkt = make_packet(task="Do something")
        result = await node.run(pkt, make_ctx())

        assert result.payload.get("agent_iterations") == 3
        assert result.payload.get("agent_completed") is False


# ── Stage wrapper tests ───────────────────────────────────────────────────────

class TestStageWrappers:
    """Tests for the ORCHA2 stage wrappers."""

    def test_decompose_node_wraps_decomposer(self):
        from orcha.nodes.stages import DecomposeNode
        from orcha.orchestration.decomposer import SmartDecomposer

        node = DecomposeNode(SmartDecomposer())
        assert node.name == "decompose"
        assert hasattr(node, "run")

    def test_all_stage_nodes_have_correct_names(self):
        from orcha.nodes.stages import (
            DecomposeNode, PlanNode, SelectNode,
            ExecuteNode, AggregateNode, EvaluateNode, RetryNode,
        )

        class FakeStage:
            pass

        assert DecomposeNode(FakeStage()).name == "decompose"
        assert PlanNode(FakeStage()).name == "plan"
        assert SelectNode(FakeStage()).name == "select"
        assert ExecuteNode(FakeStage()).name == "execute"
        assert AggregateNode(FakeStage()).name == "aggregate"
        assert EvaluateNode(FakeStage()).name == "evaluate"
        assert RetryNode(FakeStage()).name == "retry"


# ── Builder tests ─────────────────────────────────────────────────────────────

class TestBuilders:
    """Tests for the graph builders."""

    def test_build_default_graph_validates(self):
        from orcha.builders import build_default_graph
        from orcha.experts.mock import load_mock_experts

        experts = load_mock_experts()
        graph = build_default_graph(
            experts=experts,
            synthesizer_expert="mock_synthesizer",
        )
        assert graph.name == "orcha_default"
        assert graph.entry == "decompose"
        # Should have all 8 nodes.
        expected = {"decompose", "plan", "select", "execute",
                    "aggregate", "evaluate", "retry", "finalize"}
        assert expected.issubset(set(graph.nodes.keys()))

    def test_build_research_graph_validates(self):
        from orcha.builders import build_research_graph
        from orcha.experts.mock import load_mock_experts

        experts = load_mock_experts()
        graph = build_research_graph(
            experts=experts,
            synthesizer_expert="mock_synthesizer",
        )
        assert graph.name == "orcha_research"
        assert "retrieve" in graph.nodes
        assert "fact_check" in graph.nodes

    def test_build_multi_agent_graph_validates(self):
        from orcha.builders import build_multi_agent_graph

        graph = build_multi_agent_graph(verify=True)
        assert graph.name == "orcha_multi_agent"
        assert "scatter_agents" in graph.nodes
        assert "agent" in graph.nodes
        assert "gather_agents" in graph.nodes
        assert "synthesize" in graph.nodes
        assert "verify" in graph.nodes

    async def test_default_graph_runs_end_to_end(self):
        """The default graph runs with mock experts and produces an answer."""
        from orcha.builders import build_default_graph
        from orcha.experts.mock import load_mock_experts

        experts = load_mock_experts()
        graph = build_default_graph(
            experts=experts,
            synthesizer_expert="mock_synthesizer",
            max_iterations=1,
        )
        rt = GraphRuntime(graph, store=MemoryStore())
        result = await rt.run("What is machine learning?")

        assert result.run_id
        assert isinstance(result.answer, str)
        # Mock experts produce some output.
        assert len(result.answer) > 0

    async def test_research_graph_runs_end_to_end(self):
        """The research graph runs with a simple retriever."""
        from orcha.builders import build_research_graph
        from orcha.nodes.retrieval import keyword_retriever
        from orcha.experts.mock import load_mock_experts

        experts = load_mock_experts()
        docs = ["Machine learning uses data to train models."]
        graph = build_research_graph(
            experts=experts,
            synthesizer_expert="mock_synthesizer",
            retriever=keyword_retriever(docs),
            max_iterations=1,
        )
        rt = GraphRuntime(graph, store=MemoryStore())
        result = await rt.run("What is machine learning?")
        assert result.run_id

    async def test_multi_agent_graph_runs_end_to_end(self):
        """The multi-agent graph runs in dry-run mode."""
        from orcha.builders import build_multi_agent_graph

        graph = build_multi_agent_graph(verify=True)
        rt = GraphRuntime(graph, store=MemoryStore())
        result = await rt.run("Explain AI")
        assert result.run_id
        # Should have gathered branch outputs.
        assert result.packet.payload.get("branch_count", 0) > 0
