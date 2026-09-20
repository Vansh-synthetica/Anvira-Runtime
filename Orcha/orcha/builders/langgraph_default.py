"""
orcha.builders.langgraph_default
================================
The default ORCHA3 pipeline rebuilt as a genuine LangGraph StateGraph.

The same ORCHA2 stage nodes (decompose / plan / select / execute /
aggregate / evaluate / retry / finalize) constructed by
``orcha.builders.default._build_default_nodes`` are wired into a
LangGraph ``StateGraph`` with identical conditional routing:

    START ─► decompose ─► plan ──[stop]──► finalize ─► END
                              │
                              ▼ (continue)
                          select ─► execute ─► aggregate ─► evaluate
                                                        │
                                                        ▼
                                                   retry ──[retry]──► plan
                                                          │
                                                          ▼ (stop)
                                                       finalize

Each node is wrapped with an event-emitting adapter so a LangGraph run
streams the same ``RunEvent`` kinds as the native ``GraphRuntime``
(``run_start`` / ``node_start`` / ``node_end`` / ``checkpoint`` /
``run_complete`` / ``error``) with the same payload keys.

``LangGraphDefaultRunner`` mirrors the public surface of
``GraphRuntime`` (``run`` / ``replay`` / ``live_emitter``) and returns
identical ``RunResult`` objects, so switching engines via
``build_default_runner`` is transparent to callers.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import replace
from typing import Any, Dict, List, Optional

from ..core.packets import BudgetState, OrchaPacket, PacketKind
from ..graph.context import CancelToken, EventEmitter, RunContext
from ..graph.errors import (
    BudgetExceeded, Cancelled, GraphError, NodeFailed, NodeTimeout,
)
from ..graph.store import MemoryStore
from ..graph.runtime import CHECKPOINT_ALWAYS
from ..result import RunResult

from .default import DefaultGraphConfig, _build_default_nodes


def _lg_wrap(node: Any, *, branch: bool = False) -> Any:
    """
    Wrap one Orcha node as a LangGraph node that mirrors native events.

    Emits ``node_start`` / ``node_end`` / ``checkpoint`` around the
    node's run, honors ``ctx.cancel`` and the packet budget exactly like
    ``GraphRuntime._exec_node``, and normalizes failures to
    ``NodeFailed`` after ``node.retries`` attempts.

    With ``branch=True`` the wrapper runs inside a dynamic fan-out branch
    (Send) and returns its result as a ``branch_results`` entry carrying
    the branch ``index`` / ``branch_count`` read from the branch state,
    instead of overwriting the shared ``packet`` channel.
    """
    from langchain_core.runnables.config import RunnableConfig

    async def _run(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
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

        # ── Execute with retries (mirror GraphRuntime._exec_node) ───────
        await ctx.emit.emit("node_start", node.name, packet)
        await node.on_enter(packet, ctx)
        t0 = time.perf_counter()
        attempts = 1 + max(0, node.retries)
        out = packet
        last_exc: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            if ctx.cancelled:
                await ctx.emit.emit("cancel", node.name, packet, reason="cancelled")
                raise Cancelled(ctx.cancel.reason or "cancelled", node=node.name)
            try:
                if node.timeout_s is not None:
                    out = await asyncio.wait_for(
                        node.run(packet, replace(ctx, attempt=attempt)),
                        timeout=node.timeout_s,
                    )
                else:
                    out = await node.run(packet, replace(ctx, attempt=attempt))
                if not isinstance(out, OrchaPacket):
                    raise TypeError(
                        f"Node {node.name!r} returned {type(out).__name__}, "
                        f"expected OrchaPacket"
                    )
                duration_ms = (time.perf_counter() - t0) * 1000
                out = out.stamp(node.name, duration_ms, attempt=attempt)
                await node.on_success(packet, out, ctx)
                await ctx.emit.emit(
                    "node_end", node.name, out,
                    duration_ms=round(duration_ms, 2), attempt=attempt,
                )
                # LangGraph persists its own checkpoints; the event only
                # mirrors the native stream's shape (checkpoint_id=None).
                await ctx.emit.emit("checkpoint", node.name, out,
                                    checkpoint_id=None, seq=0)
                # The native engine never serializes packets between nodes,
                # so some task-execution nodes stash a live, non-
                # serializable helper object in payload as a same-process
                # convenience (e.g. task_executor.py's `_event_adapter`, a
                # raw EventStreamAdapter bound to ctx.emit) — every writer
                # already recreates it via `_get_or_create_event_adapter`
                # when it's missing, so it's always safe to drop. LangGraph
                # checkpoints every node's returned state though, and
                # msgpack has no idea how to encode a live adapter object,
                # so the whole run blew up with "Type is not msgpack
                # serializable: OrchaPacket" the first time this wrapper
                # carried one.
                #
                # Blanket-stripping every underscore-prefixed key is NOT
                # safe — the default pipeline's own initial packet sets a
                # plain-string `__graph__` key, and stripping it broke
                # test_packet_contract_unchanged's native/langgraph payload-
                # key parity check. Instead, drop only values ormsgpack
                # actually can't encode: not a JSON-primitive/list/dict, and
                # with none of the shapes JsonPlusSerializer already knows
                # how to convert (pydantic's model_dump, namedtuple's
                # _asdict, dict-likes, etc.) — i.e. an opaque live object.
                def _msgpack_unsafe(v: Any) -> bool:
                    if v is None or isinstance(v, (str, int, float, bool)):
                        return False
                    if isinstance(v, (list, tuple, dict)):
                        return False
                    return not (
                        hasattr(v, "model_dump") or hasattr(v, "dict")
                        or hasattr(v, "_asdict") or hasattr(v, "get_secret_value")
                    )
                checkpoint_out = out
                unsafe_keys = [k for k, v in out.payload.items() if _msgpack_unsafe(v)]
                if unsafe_keys:
                    checkpoint_out = out.model_copy(update={
                        "payload": {
                            k: v for k, v in out.payload.items()
                            if k not in unsafe_keys
                        },
                    })
                if branch:
                    return {"branch_results": [{
                        "index": state["index"],
                        "branch_count": state.get("branch_count", 0),
                        "packet": checkpoint_out,
                    }]}
                return {"packet": checkpoint_out}
            except asyncio.TimeoutError:
                last_exc = NodeTimeout(
                    f"Node {node.name!r} exceeded timeout {node.timeout_s}s",
                    node=node.name,
                )
            except Cancelled:
                raise  # propagate immediately, no retry
            except GraphError:
                raise  # budget/cancellation-style errors are not retryable
            except Exception as exc:  # noqa: BLE001 — fault isolation parity
                # LangGraph interrupts (human-in-the-loop pause points) must
                # bubble up untouched so the thread pauses and the caller
                # can resume with a Command value.
                from langgraph.errors import GraphInterrupt
                if isinstance(exc, GraphInterrupt):
                    raise
                last_exc = exc

        # Exhausted retries.
        duration_ms = (time.perf_counter() - t0) * 1000
        assert last_exc is not None
        await node.on_failure(packet, last_exc, ctx)
        failure = last_exc if isinstance(last_exc, NodeTimeout) else NodeFailed(
            f"Node {node.name!r} failed after {attempts} attempt(s): {last_exc}",
            node=node.name,
        )
        await ctx.emit.emit(
            "error", node.name, packet,
            error=type(failure).__name__, message=str(failure),
            duration_ms=round(duration_ms, 2),
        )
        raise failure from (last_exc if last_exc is not failure else None)

    _run.__name__ = node.name
    _run.__wrapped_node__ = node  # introspection hook for tests/tools
    return _run


def _retry_route(state: Dict[str, Any]) -> str:
    """Conditional from retry: loop back to plan, or finalize."""
    return "retry" if state["packet"].payload.get("retry") else "stop"


def _plan_route(state: Dict[str, Any]) -> str:
    """Conditional from plan: proceed to select, or finalize."""
    return "stop" if state["packet"].payload.get("stop") else "continue"


def build_default_graph_langgraph(
    config: Optional[DefaultGraphConfig] = None,
    *,
    checkpointer: Any = None,
    max_steps: int = 1000,
) -> Any:
    """
    Build the default ORCHA3 pipeline as a compiled LangGraph StateGraph.

    Parameters
    ----------
    config           DefaultGraphConfig (experts, synthesizer, budgets...).
    checkpointer     LangGraph checkpointer; defaults to a MemorySaver
                     with the OrchaPacket-safe serde.
    max_steps        Mirror of ``GraphRuntime.max_steps``; mapped onto
                     LangGraph's ``recursion_limit``.

    Returns
    -------
    A compiled LangGraph ready for ``LangGraphEngine`` / the
    ``LangGraphDefaultRunner``.
    """
    if config is None:
        config = DefaultGraphConfig()

    from langgraph.graph import END, START, StateGraph
    from langgraph.checkpoint.memory import MemorySaver
    import os

    # OrchaPacket/PacketKind are not yet in langgraph's registered msgpack
    # types; keep the deprecation notice non-blocking on future upgrades.
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "false")
    if checkpointer is None:
        from ..integrations.langgraph import orcha_serde
        checkpointer = MemorySaver(serde=orcha_serde())

    from typing import TypedDict

    class _DefaultState(TypedDict):
        packet: OrchaPacket

    nodes = _build_default_nodes(config)

    builder = StateGraph(_DefaultState)
    for name, node in nodes.items():
        builder.add_node(name, _lg_wrap(node))

    builder.add_edge(START, "decompose")
    builder.add_edge("decompose", "plan")
    builder.add_edge("select", "execute")
    builder.add_edge("execute", "aggregate")
    builder.add_edge("aggregate", "evaluate")
    builder.add_edge("evaluate", "retry")
    builder.add_conditional_edges(
        "retry", _retry_route, {"retry": "plan", "stop": "finalize"},
    )
    builder.add_conditional_edges(
        "plan", _plan_route, {"stop": "finalize", "continue": "select"},
    )
    builder.add_edge("finalize", END)

    graph = builder.compile(
        checkpointer=checkpointer, interrupt_before=[],
    )
    graph.max_steps = max_steps
    return graph


class LangGraphDefaultRunner:
    """
    Default-graph runner on the LangGraph engine.

    Exposes the same surface as ``GraphRuntime`` (``run`` / ``replay`` /
    ``live_emitter``) so callers can switch engines transparently through
    ``build_default_runner``.
    """

    def __init__(
        self,
        engine: Any,
        *,
        graph_name: str = "orcha_default",
        max_steps: int = 1000,
        logger: Optional[logging.Logger] = None,
        config: Optional["DefaultGraphConfig"] = None,
    ) -> None:
        self.engine = engine
        self.graph_name = graph_name
        self.max_steps = max_steps
        self.checkpoint_every = CHECKPOINT_ALWAYS  # LangGraph always checkpoints
        self.store = MemoryStore()
        self._config = config
        self._live_emitter: Optional[EventEmitter] = None
        self._logger = logger or logging.getLogger("orcha.builders.langgraph_default")

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
        Execute the default graph for a query (LangGraph engine).

        Mirrors ``GraphRuntime.run``: identical packet construction,
        resume-from-checkpoint semantics, event stream and ``RunResult``
        shape. LangGraph-specific divergence: threads are keyed by
        ``run_id`` inside the graph's own checkpointer, so ``resume_from``
        continues whatever thread state exists for that id.
        """
        # ── Resolve starting packet (identical to GraphRuntime.run) ─────
        if packet is None:
            run_id = run_id or str(uuid.uuid4())
            if budget is None and self._config is not None:
                from .default import _config_budget, _initial_packet
                packet = _initial_packet(
                    graph_name=self.graph_name, run_id=run_id, query=query,
                    budget=_config_budget(self._config),
                    run_all_experts=self._config.run_all_experts,
                    metadata=metadata,
                )
            else:
                payload: Dict[str, Any] = {"__graph__": self.graph_name}
                if self._config is not None and self._config.run_all_experts:
                    payload["force_all_experts"] = True
                packet = OrchaPacket(
                    id=run_id, kind=PacketKind.QUERY, query=query,
                    payload=payload, budget=budget or BudgetState(),
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
            # Emit run_failed structured event
            adapter = packet.payload.get("_event_adapter")
            if adapter is not None:
                adapter.emit_run_failed(error=f"{type(exc).__name__}: {exc}")
                final_packet = packet.fork(
                    packet.kind,
                    _event_adapter=adapter,
                    _event_stream_snapshot=adapter.get_snapshot(),
                )
            raise
        # ExecutionInterrupt is langgraph-specific (no native equivalent);
        # it propagates as-is so callers can resume with a Command value.

        await emitter.emit("run_complete", "", final_packet)

        # Emit run_completed structured event with final answer
        adapter = final_packet.payload.get("_event_adapter")
        if adapter is not None:
            # Extract final answer from the packet
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
    "build_default_graph_langgraph",
    "LangGraphDefaultRunner",
]
