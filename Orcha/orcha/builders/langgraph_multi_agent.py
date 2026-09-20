"""
orcha.builders.langgraph_multi_agent
====================================
Multi-agent collaboration graph as a genuine LangGraph StateGraph.

The native scatter/gather fan-out (decompose → scatter_agents → one
agent branch per subtask → gather_agents → synthesize → verify) is
rebuilt with LangGraph's Send API: the scatter node returns a
``Command`` carrying one ``Send`` per subtask, each branch runs the
same ``_AgentWorker`` on its subtask packet and writes a
``branch_results`` entry (accumulated by a reducer), and the plain edge
``agent_worker → gather_agents`` fans the branches back in — LangGraph
schedules the gather node exactly once, with the merged branch results.

Topology::

    START ─► decompose ─► scatter_agents ──(Send ×N)──► agent_worker
                                                              │
                                                              ▼ (fan-in)
                                                        gather_agents ─► synthesize
                                                                              │
                                                       verify ─► verify_retry_gate ──[retry]──► synthesize
                                                                              │ (pass)
                                                                              ▼
                                                                             END

Every node is wrapped with the same event-emitting adapter as the
default/agent LangGraph ports, so a run streams identical ``RunEvent``
kinds (``node_start`` / ``node_end`` / ``checkpoint`` ...) with the same
node names (``decompose`` / ``scatter_agents`` / ``agent`` /
``gather_agents`` / ``synthesize`` / ``verify`` / ``verify_retry_gate``).

``MultiAgentLangGraphRunner`` mirrors the public surface of
``GraphRuntime`` (``run`` / ``replay`` / ``live_emitter``) and returns
identical ``RunResult`` objects, so callers can switch engines via
``build_multi_agent_runner`` transparently.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import replace
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from ..core.packets import BudgetState, OrchaPacket, PacketKind
from ..graph.context import CancelToken, EventEmitter, RunContext
from ..graph.errors import (
    BudgetExceeded, Cancelled, GraphError, NodeFailed,
)
from ..graph.store import MemoryStore
from ..graph.runtime import CHECKPOINT_ALWAYS
from ..result import RunResult

from .langgraph_default import _lg_wrap
from .multi_agent import (
    MultiAgentGraphConfig, _AgentWorker, _GatherAgents, _ScatterSubtasks,
    _Synthesize, _VerifyRetryGate, _apply_reasoning_knobs,
)


def _merge_branches(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reducer for branch_results: accumulate across the fan-out."""
    return left + right


def _lg_scatter(node: Any) -> Any:
    """
    Wrap the scatter node: emit native events, then fan out via Send.

    The scatter node's ``run`` returns a ``ScatterResult`` of
    (branch_name, child_packet) pairs; the wrapper converts each into a
    ``Send`` targeting the ``agent_worker`` node. Each Send payload
    carries the forked subtask packet plus the branch ``index`` and the
    total ``branch_count`` (used by the branch worker wrapper and the
    verify-retry bookkeeping).
    """
    from langchain_core.runnables.config import RunnableConfig

    async def _run(state: Dict[str, Any], config: RunnableConfig) -> Any:
        ctx: RunContext = config["configurable"]["orcha_ctx"]
        packet: OrchaPacket = state["packet"]

        # ── Guard rails (mirror GraphRuntime._walk) ─────────────────────
        if ctx.cancelled:
            await ctx.emit.emit("cancel", node.name, packet,
                                reason=ctx.cancel.reason or "cancelled")
            raise Cancelled(ctx.cancel.reason or "cancelled", node=node.name)
        if (
            packet.budget.cost_used >= packet.budget.max_cost
            or packet.budget.latency_used_s >= packet.budget.max_latency_s
        ):
            raise BudgetExceeded(
                "budget exhausted before node " + repr(node.name), node=node.name,
            )

        await ctx.emit.emit("node_start", node.name, packet)
        t0 = time.perf_counter()
        try:
            result = await node.run(packet, ctx)
        except Exception as exc:  # noqa: BLE001 — fault isolation parity
            from langgraph.errors import GraphInterrupt
            if isinstance(exc, GraphInterrupt):
                raise
            failure = NodeFailed(
                f"Node {node.name!r} failed after 1 attempt(s): {exc}",
                node=node.name,
            )
            await ctx.emit.emit(
                "error", node.name, packet, error="NodeFailed",
                message=str(failure),
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
            raise failure from exc

        duration_ms = (time.perf_counter() - t0) * 1000
        await ctx.emit.emit(
            "node_end", node.name, packet,
            duration_ms=round(duration_ms, 2), attempt=1,
        )
        await ctx.emit.emit("checkpoint", node.name, packet,
                            checkpoint_id=None, seq=0)

        sends: List[Any] = []
        tasks: List[Dict[str, Any]] = []
        branch_count = len(result.branches)
        for i, (_name, child, _target) in enumerate(result.branches):
            sends.append(Send("agent_worker", {
                "packet": child, "index": i, "branch_count": branch_count,
            }))
            tasks.append({
                "id": child.payload.get("task_id", ""),
                "task": child.payload.get("task", ""),
                "domain": child.payload.get("task_domain", "general"),
            })
        return Command(goto=sends, update={
            "expected_branches": branch_count, "tasks": tasks,
        })

    _run.__name__ = node.name
    _run.__wrapped_node__ = node  # introspection hook for tests/tools
    return _run


def _lg_gather(node: Any) -> Any:
    """
    Wrap the gather node: run it once with the merged branch packets.

    Reuses ``GatherNode.run`` directly by attaching the collected child
    packets to a forked packet under ``__gathered__`` (the native
    runtime's convention), so fan-in behavior is byte-identical to the
    native graph.
    """
    from langchain_core.runnables.config import RunnableConfig

    async def _run(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        ctx: RunContext = config["configurable"]["orcha_ctx"]
        packet: OrchaPacket = state["packet"]

        if ctx.cancelled:
            await ctx.emit.emit("cancel", node.name, packet,
                                reason=ctx.cancel.reason or "cancelled")
            raise Cancelled(ctx.cancel.reason or "cancelled", node=node.name)
        if (
            packet.budget.cost_used >= packet.budget.max_cost
            or packet.budget.latency_used_s >= packet.budget.max_latency_s
        ):
            raise BudgetExceeded(
                "budget exhausted before node " + repr(node.name), node=node.name,
            )

        entries = sorted(
            state.get("branch_results", []), key=lambda e: e["index"],
        )
        children = [e["packet"] for e in entries]

        await ctx.emit.emit("node_start", node.name, packet)
        t0 = time.perf_counter()
        try:
            gathered_packet = packet.fork(packet.kind, __gathered__=children)
            out = await node.run(gathered_packet, ctx)
        except Exception as exc:  # noqa: BLE001 — fault isolation parity
            from langgraph.errors import GraphInterrupt
            if isinstance(exc, GraphInterrupt):
                raise
            failure = NodeFailed(
                f"Node {node.name!r} failed after 1 attempt(s): {exc}",
                node=node.name,
            )
            await ctx.emit.emit(
                "error", node.name, packet, error="NodeFailed",
                message=str(failure),
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
            raise failure from exc

        duration_ms = (time.perf_counter() - t0) * 1000
        out = out.stamp(node.name, duration_ms, attempt=1)
        await node.on_success(packet, out, ctx)
        await ctx.emit.emit(
            "node_end", node.name, out,
            duration_ms=round(duration_ms, 2), attempt=1,
        )
        await ctx.emit.emit("checkpoint", node.name, out,
                            checkpoint_id=None, seq=0)
        return {"packet": out}

    _run.__name__ = node.name
    _run.__wrapped_node__ = node
    return _run


def _verify_retry_route(state: Dict[str, Any]) -> str:
    """Conditional from verify_retry_gate: retry synthesis once, or finish."""
    return "retry" if state["packet"].payload.get("retry_verify") else "pass"


def build_multi_agent_graph_langgraph(
    config: Optional[MultiAgentGraphConfig] = None,
    *,
    checkpointer: Any = None,
    max_steps: int = 1000,
) -> Any:
    """
    Build the multi-agent collaboration graph as a compiled LangGraph.

    Parameters
    ----------
    config           MultiAgentGraphConfig (agent config, verify, budgets...).
    checkpointer     LangGraph checkpointer; defaults to a MemorySaver
                     with the OrchaPacket-safe serde.
    max_steps        Mirror of ``GraphRuntime.max_steps``; mapped onto
                     LangGraph's ``recursion_limit``.

    Returns
    -------
    A compiled LangGraph ready for ``LangGraphEngine`` / the
    ``MultiAgentLangGraphRunner``.
    """
    if config is None:
        config = MultiAgentGraphConfig()
    _apply_reasoning_knobs(config)

    # OrchaPacket/PacketKind are not yet in langgraph's registered msgpack
    # types; keep the deprecation notice non-blocking on future upgrades.
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "false")
    if checkpointer is None:
        from ..integrations.langgraph import orcha_serde
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver(serde=orcha_serde())

    class _MultiAgentState(TypedDict, total=False):
        packet: OrchaPacket
        tasks: List[Dict[str, Any]]
        expected_branches: int
        branch_results: Annotated[List[Dict[str, Any]], _merge_branches]
        # Per-branch channels carried by Send payloads:
        index: int
        branch_count: int

    # ── Instantiate nodes (identical to build_multi_agent_graph) ────────
    from ..nodes.stages import DecomposeNode
    from ..nodes.verify import VerifyNode
    from ..orchestration.decomposer import get_decomposer

    decomposer = get_decomposer(use_embeddings=config.use_embeddings)
    decompose_node = DecomposeNode(decomposer)
    scatter_node = _ScatterSubtasks(max_branches=config.max_branches)
    agent_worker = _AgentWorker(config.agent_config)
    gather_node = _GatherAgents()
    synthesize_node = _Synthesize(model_fn=config.model_fn)

    builder = StateGraph(_MultiAgentState)
    builder.add_node("decompose", _lg_wrap(decompose_node))
    builder.add_node("scatter_agents", _lg_scatter(scatter_node))
    builder.add_node("agent_worker", _lg_wrap(agent_worker, branch=True))
    builder.add_node("gather_agents", _lg_gather(gather_node))
    builder.add_node("synthesize", _lg_wrap(synthesize_node))

    builder.add_edge(START, "decompose")
    builder.add_edge("decompose", "scatter_agents")
    # scatter_agents fans out dynamically via Command(goto=[Send...]);
    # the plain edge below fans the branches back in: LangGraph runs
    # gather_agents exactly once with the merged branch_results.
    builder.add_edge("agent_worker", "gather_agents")
    builder.add_edge("gather_agents", "synthesize")

    if config.verify:
        verify_node = VerifyNode(name="verify", timeout_s=10.0)
        gate = _VerifyRetryGate()
        builder.add_node("verify", _lg_wrap(verify_node))
        builder.add_node("verify_retry_gate", _lg_wrap(gate))
        builder.add_edge("synthesize", "verify")
        builder.add_edge("verify", "verify_retry_gate")
        builder.add_conditional_edges(
            "verify_retry_gate", _verify_retry_route,
            {"retry": "synthesize", "pass": END},
        )
    else:
        builder.add_edge("synthesize", END)

    graph = builder.compile(checkpointer=checkpointer)
    graph.max_steps = max_steps
    return graph


class MultiAgentLangGraphRunner:
    """
    Multi-agent graph runner on the LangGraph engine.

    Exposes the same surface as ``GraphRuntime`` (``run`` / ``replay`` /
    ``live_emitter``) so callers can switch engines transparently through
    ``build_multi_agent_runner``.
    """

    def __init__(
        self,
        engine: Any,
        *,
        graph_name: str = "orcha_multi_agent",
        max_steps: int = 1000,
        logger: Optional[logging.Logger] = None,
        config: Optional[MultiAgentGraphConfig] = None,
    ) -> None:
        self.engine = engine
        self.graph_name = graph_name
        self.max_steps = max_steps
        self.checkpoint_every = CHECKPOINT_ALWAYS  # LangGraph always checkpoints
        self.store = MemoryStore()
        self._config = config
        self._live_emitter: Optional[EventEmitter] = None
        self._logger = logger or logging.getLogger("orcha.builders.langgraph_multi_agent")

    @property
    def live_emitter(self) -> Optional[EventEmitter]:
        """EventEmitter of the most recent run, for late subscribers."""
        return self._live_emitter

    # ── Public entrypoints ─────────────────────────────────────────────

    async def run(
        self,
        query: str,
        *,
        budget: Optional[BudgetState] = None,
        packet: Optional[OrchaPacket] = None,
        resume_from: Any = None,
        run_id: Optional[str] = None,
        on_event: Optional[Any] = None,
        cancel: Optional[CancelToken] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RunResult:
        """
        Execute the multi-agent graph for a query (LangGraph engine).

        Mirrors ``GraphRuntime.run``: identical packet construction,
        resume-from-checkpoint semantics, event stream and ``RunResult``
        shape. LangGraph-specific divergence: threads are keyed by
        ``run_id`` inside the graph's own checkpointer, so ``resume_from``
        continues whatever thread state exists for that id. The
        multi-agent graph has no interrupt nodes, so resuming a completed
        thread starts a fresh run (matching native behavior).
        """
        if packet is None:
            run_id = run_id or str(uuid.uuid4())
            if budget is None and self._config is not None:
                budget = BudgetState(
                    max_cost=self._config.max_cost,
                    max_latency_s=self._config.max_latency_s,
                    max_iterations=self._config.max_iterations,
                )
            packet = OrchaPacket(
                id=run_id, kind=PacketKind.QUERY, query=query,
                payload={"__graph__": self.graph_name},
                budget=budget or BudgetState(),
                metadata=dict(metadata or {}),
            )
        else:
            run_id = run_id or packet.id
            if budget is not None:
                packet = packet.model_copy(update={"budget": budget})

        cancel = cancel or CancelToken()
        emitter = EventEmitter(run_id, logger=self._logger)
        self._live_emitter = emitter
        if on_event is not None:
            emitter.subscribe(on_event)

        ctx = RunContext(
            run_id=run_id, store=self.store, cancel=cancel,
            emit=emitter, logger=self._logger, graph_name=self.graph_name,
        )

        # ── Resume: continue an existing thread with pending work ───────
        resumed = False
        if resume_from is not None:
            pending = await self.engine.pending(run_id)
            if pending:
                self._logger.info(
                    "resume run=%s from pending thread (%s)",
                    run_id[:8], ",".join(pending),
                )
                await emitter.emit(
                    "checkpoint", "", packet, action="resume", seq=0,
                )
                resumed = True
            else:
                self._logger.info(
                    "resume run=%s: no pending thread, starting fresh", run_id[:8],
                )

        await emitter.emit(
            "run_start", "", packet, graph=self.graph_name, entry="decompose",
        )

        final_packet: OrchaPacket
        try:
            if resumed:
                final_packet = await self.engine.resume(
                    packet, run_id, value=None, ctx=ctx,
                )
            else:
                final_packet = await self.engine.execute(
                    packet, ctx,
                    recursion_limit=self.max_steps + 8,
                )
        except GraphError as exc:
            await emitter.emit(
                "error", exc.node or "", packet,
                error=type(exc).__name__, message=str(exc),
            )
            raise

        await emitter.emit("run_complete", "", final_packet)

        # Emit run_completed structured event with final answer
        adapter = final_packet.payload.get("_event_adapter")
        if adapter is not None:
            final_answer = (
                final_packet.payload.get("final_response", {}).get("summary", "")
                if isinstance(final_packet.payload.get("final_response"), dict)
                else ""
            )
            if not final_answer:
                final_answer = final_packet.payload.get("verification_result", "")
            adapter.emit_run_completed(
                summary_text="Run completed successfully",
                plan_progress=1.0,
                final_answer=final_answer,
            )

        return RunResult(
            final_packet, graph_name=self.graph_name, run_id=run_id,
        )

    async def replay(
        self,
        run_id: str,
        on_event: Optional[Any] = None,
    ) -> RunResult:
        """
        Re-emit the event stream of a completed LangGraph thread.

        The thread's earliest checkpoint seeds a fresh ``run_id::replay``
        thread; events are re-emitted as if the run were happening live.
        """
        history = await self.engine.replay(run_id)
        if not history:
            raise GraphError(f"No checkpoints found for run {run_id}")

        # The earliest checkpoint (START) has empty state; seed from the
        # first checkpoint that actually carries a packet.
        packet: Optional[OrchaPacket] = None
        for snap in reversed(history):
            candidate = snap["values"].get("packet")
            if isinstance(candidate, OrchaPacket):
                packet = candidate
                break
        if packet is None:
            raise TypeError(
                f"Replay history for {run_id} has no packet in any checkpoint"
            )

        replay_run_id = f"{run_id}::replay"
        cancel = CancelToken()
        emitter = EventEmitter(replay_run_id, logger=self._logger)
        self._live_emitter = emitter
        if on_event is not None:
            emitter.subscribe(on_event)
        ctx = RunContext(
            run_id=replay_run_id, store=self.store, cancel=cancel,
            emit=emitter, logger=self._logger, graph_name=self.graph_name,
        )

        await emitter.emit(
            "run_start", "", packet, graph=self.graph_name,
            entry="decompose", replay=True,
        )
        try:
            final = await self.engine.execute(
                packet, ctx, recursion_limit=self.max_steps + 8,
            )
        except GraphError as exc:
            await emitter.emit(
                "error", exc.node or "", packet,
                error=type(exc).__name__, message=str(exc),
            )
            raise
        await emitter.emit("run_complete", "", final, replay=True)
        return RunResult(
            final, graph_name=self.graph_name, run_id=run_id,
        )


__all__ = [
    "build_multi_agent_graph_langgraph",
    "MultiAgentLangGraphRunner",
]