"""
orcha.builders.langgraph_agent
===============================
The single-agent graph rebuilt as a genuine LangGraph StateGraph.

The same agent nodes constructed by
``orcha.builders.agent._build_agent_nodes`` (intent gate / full agent /
read-only agent / finalize) are wired into a LangGraph ``StateGraph``
with identical conditional routing:

    START ─► intent_gate ──[chat]─────► finalize_agent ─► END
                  │
                  ├──[read]──► agent_readonly ─► (approval_gate) ─► finalize_agent
                  │
                  └──[tool]──► agent ─────────► (approval_gate) ─► finalize_agent

When ``AgentGraphConfig.require_approval`` is set, an ``approval_gate``
node is inserted before ``finalize_agent`` on the agent paths. The gate
calls LangGraph's ``interrupt()`` with the proposed answer and its tool
calls; the run pauses with ``ApprovalPending`` until the caller resumes
the thread with a decision (``True`` / ``{"approved": True}`` to accept,
anything else to reject). This is a real human-in-the-loop pause point:
the thread is checkpointed and can be resumed later, from another
process, exactly like LangGraph threads.

``AgentLangGraphRunner`` mirrors the public surface of ``GraphRuntime``
(``run`` / ``replay`` / ``live_emitter``) and returns identical
``RunResult`` objects, so switching engines via ``build_agent_runner``
is transparent to callers.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import replace
from typing import Any, Dict, List, Optional

from ..core.packets import BudgetState, OrchaPacket, PacketKind
from ..graph.context import CancelToken, EventEmitter, RunContext
from ..graph.errors import GraphError, NodeFailed, NodeTimeout
from ..graph.node import Node
from ..graph.runtime import CHECKPOINT_ALWAYS
from ..graph.store import MemoryStore
from ..integrations.langgraph.engine import ExecutionInterrupt
from ..result import RunResult

from .agent import (
    AgentGraphConfig,
    _build_agent_nodes,
    _intent_route,
    _complexity_route,
    _task_context_route,
    _execution_observer_route,
    _retry_handler_route,
    _adaptive_replanner_route,
    _replanner_route,
    _observer_route,
)
from .langgraph_default import _lg_wrap


class ApprovalPending(GraphError):
    """
    Raised by ``AgentLangGraphRunner.run`` when the run pauses at the
    approval gate awaiting a human decision.

    Attributes
    ----------
    run_id           The thread/run to resume.
    approval_payload The interrupt payload surfaced to the caller
                     (kind, output, tool_calls, iterations).
    """

    def __init__(
        self, run_id: str, approval_payload: Dict[str, Any],
        *, message: Optional[str] = None,
    ) -> None:
        super().__init__(
            message or "Run paused awaiting approval", node="approval_gate",
        )
        self.run_id = run_id
        self.approval_payload = approval_payload


class _ApprovalGate(Node):
    """
    LangGraph-only node: pauses the run for a final-answer approval.

    The first invocation calls ``interrupt()`` with the proposed output;
    LangGraph checkpoint-pauses the thread and the engine surfaces
    ``ExecutionInterrupt``. On resume the node re-runs and ``interrupt()``
    returns the human's decision: approve keeps the output as-is, reject
    marks the packet with ``approval_rejected`` so finalize answers with
    the refusal.
    """

    name = "approval_gate"

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        from langgraph.types import interrupt

        payload = {
            "kind": "agent_approval",
            "run_id": ctx.run_id,
            "output": packet.payload.get("agent_output", ""),
            "tool_calls": packet.payload.get("agent_tool_calls", []),
            "iterations": packet.payload.get("agent_iterations", 0),
        }
        decision = interrupt(payload)
        approved = decision is True or (
            isinstance(decision, dict) and decision.get("approved") is True
        )
        if approved:
            return packet
        note = decision.get("note") if isinstance(decision, dict) else ""
        return packet.model_copy(update={
            "payload": {
                **packet.payload,
                "approval_rejected": True,
                "approval_note": str(note or ""),
            },
        })


def _intent_route_state(state: Dict[str, Any]) -> str:
    """Conditional-edge label after the gate (LangGraph state wrapper)."""
    return _intent_route(state["packet"])


# ── LangGraph state-wrapper variants of the native task-decomposition
# route predicates (orcha/builders/agent.py). Each just unwraps
# state["packet"] and delegates — same pattern as _intent_route_state
# above — so the routing LOGIC lives in exactly one place and both
# engines stay in sync by construction rather than by duplication.
def _complexity_route_state(state: Dict[str, Any]) -> str:
    return _complexity_route(state["packet"])


def _task_context_route_state(state: Dict[str, Any]) -> str:
    return _task_context_route(state["packet"])


def _execution_observer_route_state(state: Dict[str, Any]) -> str:
    return _execution_observer_route(state["packet"])


def _retry_handler_route_state(state: Dict[str, Any]) -> str:
    return _retry_handler_route(state["packet"])


def _adaptive_replanner_route_state(state: Dict[str, Any]) -> str:
    return _adaptive_replanner_route(state["packet"])


def _replanner_route_state(state: Dict[str, Any]) -> str:
    return _replanner_route(state["packet"])


def _observer_route_state(state: Dict[str, Any]) -> str:
    return _observer_route(state["packet"])


def build_agent_graph_langgraph(
    config: Optional[AgentGraphConfig] = None,
    *,
    checkpointer: Any = None,
    max_steps: int = 1000,
) -> Any:
    """
    Build the single-agent graph as a compiled LangGraph StateGraph.

    Parameters
    ----------
    config           AgentGraphConfig (agent configs, gate, budgets...).
    checkpointer     LangGraph checkpointer; defaults to a MemorySaver
                     with the OrchaPacket-safe serde (a checkpointer is
                     required for the approval gate to pause).
    max_steps        Mirror of ``GraphRuntime.max_steps``; mapped onto
                     LangGraph's ``recursion_limit``.

    Returns
    -------
    A compiled LangGraph ready for ``LangGraphEngine`` / the
    ``AgentLangGraphRunner``.
    """
    if config is None:
        config = AgentGraphConfig()

    from langgraph.graph import END, START, StateGraph
    from langgraph.checkpoint.memory import MemorySaver

    import os
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "false")
    if checkpointer is None:
        from ..integrations.langgraph import orcha_serde
        checkpointer = MemorySaver(serde=orcha_serde())

    from typing import TypedDict

    class _AgentState(TypedDict):
        packet: OrchaPacket

    nodes = _build_agent_nodes(config)
    agent = nodes["agent"]
    finalize = nodes["finalize"]
    gate = nodes["gate"]
    complexity_gate = nodes["complexity_gate"]

    builder = StateGraph(_AgentState)
    builder.add_node("agent", _lg_wrap(agent))
    builder.add_node("finalize_agent", _lg_wrap(finalize))

    after_agent = "approval_gate" if config.require_approval else "finalize_agent"
    if config.require_approval:
        builder.add_node("approval_gate", _lg_wrap(_ApprovalGate()))
        builder.add_edge("agent", "approval_gate")
        builder.add_edge("approval_gate", "finalize_agent")
    else:
        builder.add_edge("agent", "finalize_agent")

    if gate is not None:
        builder.add_node("intent_gate", _lg_wrap(gate))
        builder.add_edge(START, "intent_gate")
        readonly = nodes["readonly"]
        if readonly is not None:
            builder.add_node("agent_readonly", _lg_wrap(readonly))
            builder.add_edge("agent_readonly", after_agent)
            read_dest = "agent_readonly"
        else:
            # Without a read-only node, read intents degrade to the full
            # agent (matches the native builder).
            read_dest = "agent"

        if complexity_gate is not None:
            # Task-decomposition topology — mirrors native build_agent_graph
            # (orcha/builders/agent.py:606-743) node-for-node, so a request
            # that needs multi-step planning/retry/replan/verify gets the
            # same treatment on the LangGraph engine as on native. Every
            # path that reaches a final answer routes through `after_agent`
            # (not a bare "finalize_agent") so the approval gate, when
            # configured, still guards every way out — not just the simple
            # single-turn agent path.
            builder.add_node("complexity_gate", _lg_wrap(complexity_gate))
            builder.add_node("task_planner", _lg_wrap(nodes["task_planner"]))
            builder.add_node("task_context_builder", _lg_wrap(nodes["task_context_builder"]))
            # Native (agent.py:606-613) registers step_executor/observer/
            # replanner/verifier unconditionally whenever complexity_gate is
            # configured, even when the adaptive loop below ends up using
            # them for nothing — mirrored here so a route table can always
            # legally target "verifier"/"replanner" regardless of which
            # branch (adaptive vs legacy) is actually wired below.
            step_executor = nodes["step_executor"]
            observer = nodes["observer"]
            replanner = nodes["replanner"]
            verifier = nodes["verifier"]
            builder.add_node("step_executor", _lg_wrap(step_executor))
            builder.add_node("observer", _lg_wrap(observer))
            builder.add_node("replanner", _lg_wrap(replanner))
            builder.add_node("verifier", _lg_wrap(verifier))

            execution_loop = nodes["execution_loop"]
            execution_observer = nodes["execution_observer"]
            retry_handler = nodes["retry_handler"]
            final_verifier = nodes["final_verifier"]
            adaptive_replanner = nodes["adaptive_replanner"]
            focused_context_builder = nodes["focused_context_builder"]
            if focused_context_builder is not None:
                # Native (agent.py:623-624) registers this node too, even
                # though nothing wires an edge to it in EITHER engine today
                # — TaskExecutionLoop apparently uses it directly rather
                # than as a separate graph step. Registered here only for
                # exact node-set parity with native; genuinely unreachable
                # in both engines, matching existing behavior.
                builder.add_node(
                    "focused_task_context_builder", _lg_wrap(focused_context_builder)
                )

            builder.add_conditional_edges(
                "intent_gate", _intent_route_state,
                {
                    "chat": "finalize_agent",
                    "agent_readonly": read_dest,
                    "agent": "complexity_gate",
                },
            )
            builder.add_conditional_edges(
                "complexity_gate", _complexity_route_state,
                {"simple": "agent", "complex": "task_planner"},
            )
            builder.add_edge("task_planner", "task_context_builder")

            if execution_loop is not None and execution_observer is not None:
                # Adaptive execution loop (the same path native prefers
                # whenever both nodes are available — see agent.py:621).
                builder.add_node("execution_loop", _lg_wrap(execution_loop))
                builder.add_node("execution_observer", _lg_wrap(execution_observer))
                builder.add_conditional_edges(
                    "task_context_builder", _task_context_route_state,
                    {"execute": "execution_loop", "complete": after_agent},
                )
                builder.add_edge("execution_loop", "execution_observer")

                retry_dest = "retry_handler" if retry_handler is not None else after_agent
                replan_dest = "adaptive_replanner" if adaptive_replanner is not None else (
                    "replanner" if replanner is not None else after_agent
                )
                verify_dest = "final_verifier" if final_verifier is not None else (
                    "verifier" if verifier is not None else after_agent
                )
                builder.add_conditional_edges(
                    "execution_observer", _execution_observer_route_state,
                    {
                        "retry": retry_dest,
                        "modify": retry_dest,
                        "replan": replan_dest,
                        "verify": verify_dest,
                        "next_task": "task_context_builder",
                        "finalize": after_agent,
                    },
                )
                if retry_handler is not None:
                    builder.add_node("retry_handler", _lg_wrap(retry_handler))
                    builder.add_conditional_edges(
                        "retry_handler", _retry_handler_route_state,
                        {
                            "retry_same": "execution_loop",
                            "next_task": "task_context_builder",
                            "finalize": after_agent,
                        },
                    )
                if adaptive_replanner is not None:
                    builder.add_node("adaptive_replanner", _lg_wrap(adaptive_replanner))
                    builder.add_conditional_edges(
                        "adaptive_replanner", _adaptive_replanner_route_state,
                        {"next_task": "task_context_builder", "finalize": after_agent},
                    )
                if final_verifier is not None:
                    builder.add_node("final_verifier", _lg_wrap(final_verifier))
            else:
                # Fallback to the legacy step_executor/observer path — only
                # reachable when the adaptive loop nodes aren't configured
                # (matches native's fallback at agent.py:670-682). Nodes
                # were already registered above.
                builder.add_conditional_edges(
                    "task_context_builder", _task_context_route_state,
                    {"execute": "step_executor", "complete": after_agent},
                )
                builder.add_edge("step_executor", "observer")
                verify_dest = "verifier" if verifier is not None else after_agent
                replan_dest = "replanner" if replanner is not None else after_agent
                builder.add_conditional_edges(
                    "observer", _observer_route_state,
                    {
                        "replan": replan_dest,
                        "verify": verify_dest,
                        "next_step": "task_context_builder",
                        "finalize": after_agent,
                    },
                )

            if replanner is not None:
                builder.add_conditional_edges(
                    "replanner", _replanner_route_state,
                    {"replan": "task_planner", "finalize": after_agent},
                )
            # Unconditional, exactly like native (agent.py:692-695): whichever
            # of final_verifier/verifier exists is the one true "answer is
            # ready" edge into after_agent, regardless of which loop branch
            # (adaptive vs legacy) actually produced it.
            if final_verifier is not None:
                builder.add_edge("final_verifier", after_agent)
            else:
                builder.add_edge("verifier", after_agent)
        else:
            builder.add_conditional_edges(
                "intent_gate", _intent_route_state,
                {
                    "chat": "finalize_agent",
                    "agent_readonly": read_dest,
                    "agent": "agent",
                },
            )
    else:
        # No gate configured (legacy direct construction).
        builder.add_edge(START, "agent")

    builder.add_edge("finalize_agent", END)

    graph = builder.compile(checkpointer=checkpointer)
    graph.max_steps = max_steps
    return graph


class AgentLangGraphRunner:
    """
    Single-agent graph runner on the LangGraph engine.

    Exposes the same surface as ``GraphRuntime`` (``run`` / ``replay`` /
    ``live_emitter``) so callers can switch engines transparently through
    ``build_agent_runner``. Divergence: when the graph pauses at the
    approval gate, ``run`` raises ``ApprovalPending``; resume the thread
    with ``run(..., resume_from=..., resume_value=<decision>)``.
    """

    def __init__(
        self,
        engine: Any,
        *,
        graph_name: str = "orcha_agent",
        max_steps: int = 1000,
        logger: Optional[logging.Logger] = None,
        config: Optional[AgentGraphConfig] = None,
    ) -> None:
        self.engine = engine
        self.graph_name = graph_name
        self.max_steps = max_steps
        self.checkpoint_every = CHECKPOINT_ALWAYS  # LangGraph always checkpoints
        self.store = MemoryStore()
        self._config = config
        self._live_emitter: Optional[EventEmitter] = None
        self._logger = logger or logging.getLogger("orcha.builders.langgraph_agent")

    @property
    def live_emitter(self) -> Optional[EventEmitter]:
        """EventEmitter of the most recent run, for late subscribers."""
        return self._live_emitter

    @property
    def entry(self) -> str:
        """Name of the entry node (intent gate when configured)."""
        gate_cfg = self._config.gate_config if self._config is not None else None
        if gate_cfg is not None and gate_cfg.completion_fn is not None:
            return "intent_gate"
        return "agent"

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
        resume_value: Any = None,
    ) -> RunResult:
        """
        Execute the single-agent graph for a query (LangGraph engine).

        Mirrors ``GraphRuntime.run``: identical packet construction,
        resume-from-checkpoint semantics, event stream and ``RunResult``
        shape. ``resume_value`` is delivered to a paused approval gate
        (``True`` / ``{"approved": True}`` approves; anything else
        rejects). A run that pauses raises ``ApprovalPending`` carrying
        the thread id + approval payload instead of returning.
        """
        # ── Resolve starting packet (identical to GraphRuntime.run) ─────
        if packet is None:
            run_id = run_id or str(uuid.uuid4())
            if budget is None and self._config is not None:
                from .default import _config_budget, _initial_packet
                packet = _initial_packet(
                    graph_name=self.graph_name, run_id=run_id, query=query,
                    budget=_config_budget(self._config),
                    run_all_experts=False, metadata=metadata,
                )
            else:
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
            "run_start", "", packet, graph=self.graph_name, entry=self.entry,
        )

        final_packet: OrchaPacket
        try:
            if resumed:
                final_packet = await self.engine.resume(
                    packet, run_id, value=resume_value, ctx=ctx,
                )
            else:
                final_packet = await self.engine.execute(
                    packet, ctx, recursion_limit=self.max_steps + 8,
                )
        except ExecutionInterrupt as exc:
            # The run paused at the approval gate: surface the interrupt
            # payload as ApprovalPending so the caller can decide and
            # resume the thread with resume_value.
            payload: Dict[str, Any] = {}
            snap = exc.snapshot or {}
            interrupts = snap.get("__interrupt__") or ()
            if interrupts:
                first = interrupts[0]
                payload = getattr(first, "value", first)
            raise ApprovalPending(run_id, dict(payload or {})) from None
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
            entry=self.entry, replay=True,
        )
        try:
            final = await self.engine.execute(
                packet, ctx, recursion_limit=self.max_steps + 8,
            )
        except ExecutionInterrupt as exc:
            payload: Dict[str, Any] = {}
            snap = exc.snapshot or {}
            interrupts = snap.get("__interrupt__") or ()
            if interrupts:
                first = interrupts[0]
                payload = getattr(first, "value", first)
            raise ApprovalPending(replay_run_id, dict(payload or {})) from None
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
    "build_agent_graph_langgraph",
    "AgentLangGraphRunner",
    "ApprovalPending",
]
