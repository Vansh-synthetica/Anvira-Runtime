"""
orcha.builders.research
========================
Research pipeline graph — RAG-augmented retrieval + verification.

This graph is designed for research-oriented queries where factual accuracy
is critical. It retrieves relevant context before expert execution, then
runs a fact-check verification step after aggregation.

Topology::

    decompose ──► retrieve ──► plan ──► select ──► execute ──► aggregate
                                                              │
                                                              ▼
                                                         evaluate
                                                              │
                                                              ▼
                                                        fact_check
                                                              │
                                                              ▼
                                                         retry_decision
                                                         ├─ stop → finalize → END
                                                         └─ retry → plan

The retrieval step enriches the packet with relevant passages from a
document corpus or vector store. The fact-check step validates the
aggregated answer against the retrieval results.

Usage::

    from orcha.builders import build_research_graph
    from orcha.nodes.retrieval import keyword_retriever
    from orcha.graph import GraphRuntime

    retriever = keyword_retriever(documents=my_corpus)
    graph = build_research_graph(
        experts=my_experts,
        retriever=retriever,
    )
    rt = GraphRuntime(graph)
    result = await rt.run("What is the GDP of France?")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..core.packets import PacketKind
from ..experts.base import BaseExpert
from ..graph.edge import END
from ..graph.graph import Graph
from ..nodes.retrieval import RetrievalConfig, RetrievalNode
from ..nodes.stages import (
    AggregateNode,
    DecomposeNode,
    EvaluateNode,
    ExecuteNode,
    PlanNode,
    RetryNode,
    SelectNode,
)
from ..nodes.verify import FactCheckNode
from ..orchestration.aggregator import get_aggregator
from ..orchestration.decomposer import get_decomposer
from ..orchestration.evaluator import get_evaluator
from ..orchestration.executor import get_executor
from ..orchestration.planner import get_planner
from ..orchestration.retry import get_retry_controller
from ..orchestration.selector import get_selector


@dataclass
class ResearchGraphConfig:
    """
    Configuration for the research graph.

    Attributes
    ----------
    experts              Dict of {name: BaseExpert}.
    synthesizer_expert   Name of the synthesizer expert.
    retriever            Retrieval function: (query, top_k) -> list[str].
    retrieval_top_k      Number of documents to retrieve (default 5).
    retrieval_source     Name for the retrieval backend (for tracing).
    run_all_experts      If True, all experts run every iteration.
    max_cost             Budget cap in USD.
    max_latency_s        Maximum wall-clock latency budget.
    max_iterations       Hard cap on pipeline loop count.
    use_embeddings       Use sentence-transformers for decomposition/evaluation.
    fact_check_threshold  Fraction of claims that must be supported (default 0.5).
    """
    experts: Dict[str, BaseExpert] = field(default_factory=dict)
    synthesizer_expert: Optional[str] = None
    retriever: Optional[object] = None
    retrieval_top_k: int = 5
    retrieval_source: str = "default"
    run_all_experts: bool = False
    max_cost: float = 1.0
    max_latency_s: float = 120.0
    max_iterations: int = 3
    use_embeddings: bool = False
    fact_check_threshold: float = 0.5


def build_research_graph(
    config: Optional[ResearchGraphConfig] = None,
    *,
    experts: Optional[Dict[str, BaseExpert]] = None,
    synthesizer_expert: Optional[str] = None,
    retriever: Optional[object] = None,
    retrieval_top_k: int = 5,
    retrieval_source: str = "default",
    run_all_experts: bool = False,
    max_cost: float = 1.0,
    max_latency_s: float = 120.0,
    max_iterations: int = 3,
    use_embeddings: bool = False,
    fact_check_threshold: float = 0.5,
) -> Graph:
    """
    Build the research pipeline graph.

    Accepts either a ``ResearchGraphConfig`` or individual keyword arguments.
    Returns a validated ``Graph`` ready for ``GraphRuntime``.
    """
    if config is None:
        config = ResearchGraphConfig(
            experts=experts or {},
            synthesizer_expert=synthesizer_expert,
            retriever=retriever,
            retrieval_top_k=retrieval_top_k,
            retrieval_source=retrieval_source,
            run_all_experts=run_all_experts,
            max_cost=max_cost,
            max_latency_s=max_latency_s,
            max_iterations=max_iterations,
            use_embeddings=use_embeddings,
            fact_check_threshold=fact_check_threshold,
        )

    # ── Instantiate stages ──────────────────────────────────────────────
    decomposer = get_decomposer(use_embeddings=config.use_embeddings)
    planner = get_planner()
    evaluator = get_evaluator(use_embeddings=config.use_embeddings)
    selector = get_selector(config.experts)
    executor = get_executor(config.experts, selector=selector)
    aggregator = get_aggregator(
        experts=config.experts,
        synthesizer_expert=config.synthesizer_expert,
    )
    retry_ctrl = get_retry_controller(
        synthesizer_expert=config.synthesizer_expert,
    )

    # ── Wrap as graph nodes ────────────────────────────────────────────
    decompose_node = DecomposeNode(decomposer)
    retrieval_node = RetrievalNode(
        name="retrieve",
        config=RetrievalConfig(
            retriever=config.retriever,
            top_k=config.retrieval_top_k,
            source_name=config.retrieval_source,
        ),
    )
    plan_node = PlanNode(planner)
    select_node = SelectNode(selector)
    execute_node = ExecuteNode(executor, timeout_s=180.0)
    aggregate_node = AggregateNode(aggregator, timeout_s=60.0)
    evaluate_node = EvaluateNode(evaluator)
    fact_check_node = FactCheckNode(threshold=config.fact_check_threshold)
    retry_node = RetryNode(retry_ctrl)

    def _finalize(pkt, ctx):
        return pkt.fork(PacketKind.RESPONSE)

    from ..graph.node import to_node
    finalize_node = to_node(_finalize, name="finalize")

    # ── Build topology ──────────────────────────────────────────────────
    g = Graph(name="orcha_research")

    g.add_node(decompose_node, entry=True)
    g.add_node(retrieval_node)
    g.add_node(plan_node)
    g.add_node(select_node)
    g.add_node(execute_node)
    g.add_node(aggregate_node)
    g.add_node(evaluate_node)
    g.add_node(fact_check_node)
    g.add_node(retry_node)
    g.add_node(finalize_node)

    # Linear chain with retrieval pre-pended.
    g.add_edge("decompose", "retrieve")
    g.add_edge("retrieve", "plan")
    g.add_edge("select", "execute")
    g.add_edge("execute", "aggregate")
    g.add_edge("aggregate", "evaluate")
    g.add_edge("evaluate", "fact_check")
    g.add_edge("fact_check", "retry")

    # Plan: stop → finalize, continue → select.
    g.add_conditional(
        "plan",
        {"stop": "finalize", "continue": "select"},
        predicate=lambda pkt: "stop" if pkt.payload.get("stop") else "continue",
    )

    # Retry: stop → finalize, retry → plan.
    g.add_conditional(
        "retry",
        {"stop": "finalize", "retry": "plan"},
        predicate=lambda pkt: "retry" if pkt.payload.get("retry") else "stop",
    )

    g.add_edge("finalize", END)
    g.validate()

    return g


__all__ = ["build_research_graph", "ResearchGraphConfig"]
