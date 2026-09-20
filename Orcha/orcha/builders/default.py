"""
orcha.builders.default
=======================
The default ORCHA3 graph — a faithful reproduction of the ORCHA2 pipeline.

Topology::

    decompose ──► plan ──► [stop?] ──► END
                     │
                     ▼ (continue)
                  select ──► execute ──► aggregate ──► evaluate
                                              │
                                              ▼
                                          retry ──► [retry?] ──► plan
                                                     │
                                                     ▼ (stop)
                                                  finalize ──► END

This builder is the **regression bridge**: users who call
``build_default_graph(experts=...)`` get the exact same behavior as the
ORCHA2 ``Orchestrator`` but running on the graph engine with full
checkpointing, event streaming, and timeout/retry support.

Usage::

    from orcha.builders import build_default_graph
    from orcha.graph import GraphRuntime, MemoryStore

    graph = build_default_graph(experts=my_experts, synthesizer="llama3")
    rt = GraphRuntime(graph, store=MemoryStore())
    result = await rt.run("What causes climate change?")
    print(result.answer)
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.packets import BudgetState, OrchaPacket, PacketKind
from ..experts.base import BaseExpert
from ..graph.edge import END
from ..graph.graph import Graph
from ..graph.runtime import GraphRuntime
from ..nodes.stages import (
    AggregateNode,
    DecomposeNode,
    EvaluateNode,
    ExecuteNode,
    PlanNode,
    RetryNode,
    SelectNode,
)
from ..orchestration.aggregator import get_aggregator
from ..orchestration.decomposer import get_decomposer
from ..orchestration.evaluator import get_evaluator
from ..orchestration.executor import get_executor
from ..orchestration.planner import get_planner
from ..orchestration.retry import get_retry_controller
from ..orchestration.selector import ExpertSelector, get_selector


@dataclass
class DefaultGraphConfig:
    """
    Configuration for the default graph.

    Attributes
    ----------
    experts              Dict of {name: BaseExpert}.
    synthesizer_expert   Name of the synthesizer expert (for aggregation).
    run_all_experts      If True, all experts run every iteration.
    max_cost             Budget cap in USD.
    max_latency_s        Maximum wall-clock latency budget.
    max_iterations       Hard cap on pipeline loop count.
    use_embeddings       Use sentence-transformers for decomposition/evaluation.
    engine               Execution engine: "langgraph" (LangGraph
                         StateGraph; the default) or "native"
                         (GraphRuntime). Used by ``build_default_runner``;
                         overridable per call and via the
                         ORCHA_DEFAULT_ENGINE env var.
    """
    experts: Dict[str, BaseExpert] = field(default_factory=dict)
    synthesizer_expert: Optional[str] = None
    run_all_experts: bool = False
    max_cost: float = 1.0
    max_latency_s: float = 120.0
    max_iterations: int = 3
    use_embeddings: bool = False
    engine: str = "langgraph"


def _build_default_nodes(config: "DefaultGraphConfig") -> Dict[str, Any]:
    """
    Instantiate the ORCHA2 stage nodes for the default graph.

    Shared by the native builder (``build_default_graph``) and the
    LangGraph builder (``orcha.builders.langgraph_default``) so both
    engines execute byte-identical node logic.
    """
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
    from ..graph.node import to_node

    def _finalize(pkt, ctx):
        return pkt.fork(PacketKind.RESPONSE)

    return {
        "decompose": DecomposeNode(decomposer),
        "plan": PlanNode(planner),
        "select": SelectNode(selector),
        "execute": ExecuteNode(executor, timeout_s=180.0),
        "aggregate": AggregateNode(aggregator, timeout_s=60.0),
        "evaluate": EvaluateNode(evaluator),
        "retry": RetryNode(retry_ctrl),
        "finalize": to_node(_finalize, name="finalize"),
    }


def _config_budget(config: "DefaultGraphConfig") -> "BudgetState":
    """
    Map the config's budget knobs onto a packet ``BudgetState``.

    Values of zero mean "not specified" and fall back to the BudgetState
    defaults (e.g. ``max_cost=0`` = local-only / spend nothing must not
    make the run un-started; ``max_iterations=0`` is only meaningful to
    the planner as a stop signal once an iteration completed). This is
    what makes ``DefaultGraphConfig.max_iterations`` / ``max_cost`` /
    ``max_latency_s`` actually reach the run budget — historically they
    were dead knobs that silently never left the config.
    """
    return BudgetState(
        max_cost=config.max_cost if config.max_cost > 0 else 1.0,
        max_latency_s=config.max_latency_s if config.max_latency_s > 0 else 120.0,
        max_iterations=config.max_iterations if config.max_iterations > 0 else 3,
    )


def _initial_packet(
    graph_name: str, run_id: str, query: str, budget: "BudgetState",
    run_all_experts: bool, metadata: Optional[Dict[str, Any]] = None,
) -> "OrchaPacket":
    """Fresh packet exactly like ``GraphRuntime.run`` builds, plus the
    config-driven ``force_all_experts`` payload flag (also historically
    dead: the flag never reached the selector on the graph path)."""
    payload: Dict[str, Any] = {"__graph__": graph_name}
    if run_all_experts:
        payload["force_all_experts"] = True
    return OrchaPacket(
        id=run_id, kind=PacketKind.QUERY, query=query,
        payload=payload, budget=budget, metadata=dict(metadata or {}),
    )


class _NativeDefaultRunner(GraphRuntime):
    """
    Native-engine runner produced by ``build_default_runner``.

    A ``GraphRuntime`` subclass (so ``isinstance(runner, GraphRuntime)``
    keeps working) that applies the ``DefaultGraphConfig`` budget and
    ``run_all_experts`` knobs to the fresh packet when the caller does
    not supply their own packet/budget. Everything else delegates to
    ``GraphRuntime`` verbatim (checkpointing, resume, events, replay).
    """

    def __init__(
        self,
        graph: Graph,
        *,
        config: "DefaultGraphConfig",
        store: Any = None,
        checkpoint_every: Optional[int] = None,
        max_steps: int = 1000,
    ) -> None:
        super().__init__(
            graph, store=store, checkpoint_every=checkpoint_every,
            max_steps=max_steps,
        )
        self._config = config

    async def run(
        self, query: str, *,
        budget: Optional["BudgetState"] = None,
        packet: Optional["OrchaPacket"] = None,
        resume_from: Optional[Any] = None,
        run_id: Optional[str] = None,
        on_event: Optional[Any] = None,
        cancel: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if packet is None and budget is None:
            run_id = run_id or str(uuid.uuid4())
            packet = _initial_packet(
                graph_name=self.graph.name,
                run_id=run_id,
                query=query,
                budget=_config_budget(self._config),
                run_all_experts=self._config.run_all_experts,
                metadata=metadata,
            )
        return await super().run(
            query, budget=budget, packet=packet, resume_from=resume_from,
            run_id=run_id, on_event=on_event, cancel=cancel, metadata=metadata,
        )


def build_default_graph(
    config: Optional[DefaultGraphConfig] = None,
    *,
    experts: Optional[Dict[str, BaseExpert]] = None,
    synthesizer_expert: Optional[str] = None,
    run_all_experts: bool = False,
    max_cost: float = 1.0,
    max_latency_s: float = 120.0,
    max_iterations: int = 3,
    use_embeddings: bool = False,
) -> Graph:
    """
    Build the default ORCHA3 graph (ORCHA2 regression bridge).

    Accepts either a ``DefaultGraphConfig`` or individual keyword arguments
    (keyword args override config values).

    The graph faithfully reproduces the ORCHA2 pipeline loop:
    decompose → plan → select → execute → aggregate → evaluate → retry → (loop)

    Returns a validated ``Graph`` ready for ``GraphRuntime``.
    """
    if config is None:
        config = DefaultGraphConfig(
            experts=experts or {},
            synthesizer_expert=synthesizer_expert,
            run_all_experts=run_all_experts,
            max_cost=max_cost,
            max_latency_s=max_latency_s,
            max_iterations=max_iterations,
            use_embeddings=use_embeddings,
        )

    nodes = _build_default_nodes(config)

    # ── Build topology ──────────────────────────────────────────────────
    g = Graph(name="orcha_default")

    g.add_node(nodes["decompose"], entry=True)
    g.add_node(nodes["plan"])
    g.add_node(nodes["select"])
    g.add_node(nodes["execute"])
    g.add_node(nodes["aggregate"])
    g.add_node(nodes["evaluate"])
    g.add_node(nodes["retry"])
    g.add_node(nodes["finalize"])

    # Linear chain: decompose → plan → select → execute → aggregate → evaluate
    g.add_edge("decompose", "plan")
    g.add_edge("select", "execute")
    g.add_edge("execute", "aggregate")
    g.add_edge("aggregate", "evaluate")
    g.add_edge("evaluate", "retry")

    # Retry: if retry=False → finalize → END. If retry=True → loop back to plan.
    g.add_conditional(
        "retry",
        {"stop": "finalize", "retry": "plan"},
        predicate=lambda pkt: "retry" if pkt.payload.get("retry") else "stop",
    )

    # Plan: if stop=True → finalize → END. Otherwise → select.
    g.add_conditional(
        "plan",
        {"stop": "finalize", "continue": "select"},
        predicate=lambda pkt: "stop" if pkt.payload.get("stop") else "continue",
    )

    g.add_edge("finalize", END)
    g.validate()

    return g


def build_default_runner(
    config: Optional["DefaultGraphConfig"] = None,
    *,
    engine: Optional[str] = None,
    store: Any = None,
    checkpoint_every: Optional[int] = None,
    max_steps: int = 1000,
    checkpointer: Any = None,
    experts: Optional[Dict[str, BaseExpert]] = None,
    synthesizer_expert: Optional[str] = None,
    run_all_experts: bool = False,
    max_cost: float = 1.0,
    max_latency_s: float = 120.0,
    max_iterations: int = 3,
    use_embeddings: bool = False,
) -> Any:
    """
    Build a default-graph runner on the selected execution engine.

    ``engine`` selects the runtime: ``"langgraph"`` (default) returns a
    ``LangGraphDefaultRunner`` driving a genuine LangGraph
    ``StateGraph`` wired with the same ORCHA2 stage nodes; ``"native"``
    returns a ``GraphRuntime`` driving the validated ``Graph``. The
    engine may also be set via the ``ORCHA_DEFAULT_ENGINE`` environment
    variable. Both runners expose the same async surface (``run`` /
    ``replay`` / ``live_emitter``) and return identical ``RunResult``
    objects, so the public contract is unchanged.

    Accepts either a ``DefaultGraphConfig`` or the same keyword
    arguments as ``build_default_graph`` (keyword args win).
    """
    import os

    if config is None:
        config = DefaultGraphConfig(
            experts=experts or {},
            synthesizer_expert=synthesizer_expert,
            run_all_experts=run_all_experts,
            max_cost=max_cost,
            max_latency_s=max_latency_s,
            max_iterations=max_iterations,
            use_embeddings=use_embeddings,
        )
    engine = engine or os.environ.get("ORCHA_DEFAULT_ENGINE") or config.engine

    if engine == "langgraph":
        from ..integrations.langgraph import LangGraphEngine
        from .langgraph_default import LangGraphDefaultRunner, build_default_graph_langgraph

        graph = build_default_graph_langgraph(
            config, max_steps=max_steps, checkpointer=checkpointer,
        )
        return LangGraphDefaultRunner(
            LangGraphEngine(graph), graph_name="orcha_default",
            max_steps=max_steps, config=config,
        )

    if engine != "native":
        raise ValueError(
            f"Unknown engine {engine!r} — expected 'native' or 'langgraph'"
        )

    graph = build_default_graph(config)
    return _NativeDefaultRunner(
        graph, config=config, store=store,
        checkpoint_every=checkpoint_every, max_steps=max_steps,
    )


__all__ = [
    "build_default_graph",
    "build_default_runner",
    "DefaultGraphConfig",
]
