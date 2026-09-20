"""
orcha.graph.runtime
===================
GraphRuntime — the engine that walks a Graph's topology to completion.

Responsibilities (one cohesive traversal loop)
----------------------------------------------
1. Node execution: wrap every ``node.run`` in cancellation-check, timeout
   enforcement (``asyncio.wait_for``), retry (``node.retries``), and
   fault normalization (any raised exception → ``NodeFailed``).
2. Edge resolution: unconditional Edge → fixed destination; ConditionalEdge
   → predicate(packet) picks a label; FanOutEdge → scatter; FanInEdge →
   the gather node consumes gathered children.
3. Fan-out / fan-in: a scatter node returns a ScatterResult; the runtime
   runs every branch concurrently, then hands the gathered child packets
   to the paired gather node via ``packet.metadata['__gathered__']``.
4. Checkpointing: after every successful node transition, write a
   checkpoint to the Store (throttled by a configurable policy). Resume
   loads the latest checkpoint and skips already-completed nodes.
5. Event streaming: emit node_start / node_end / fan_out / fan_in /
   checkpoint / cancel / error / run_complete events to all subscribers.
6. Budget enforcement: before each node, check the packet's BudgetState;
   raise BudgetExceeded (recoverable) if exhausted.
7. Termination: reaching END, a conditional predicate returning None, the
   budget exhausting, cancellation, or a control-node failure all end the
   run and yield a RunResult.

The runtime is reentrant per Graph instance: each ``run()`` call gets its
own packet, context, and event emitter, so concurrent runs on the same
graph are safe.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from ..core.packets import BudgetState, OrchaPacket, PacketKind
from ..observability import get_logger
from .context import (
    CancelToken, EventEmitter, EventSubscriber, RunContext, RunEvent,
    ScatterResult, current_trace_id, trace_id_var,
)
from .edge import (
    ConditionalEdge, Edge, END, FanInEdge, FanOutEdge,
)
from .errors import (
    BudgetExceeded, Cancelled, GatherError, GraphError, GraphInvalid,
    NodeFailed, NodeTimeout, NoRoute,
)
from .graph import Graph
from .node import GatherNode, Node
from .store import Checkpoint, MemoryStore, Store

# Sentinel for "no throttle": checkpoint after every node.
CHECKPOINT_ALWAYS: Optional[int] = None


class GraphRuntime:
    """
    Executes a Graph. One runtime can run the same graph many times.

    Parameters
    ----------
    graph          The validated Graph to execute.
    store          Where to persist checkpoints. MemoryStore by default;
                   pass a FileStore / SqliteStore for durability.
    checkpoint_every
                   Throttle: write a checkpoint every N successful node
                   transitions (None = every node). Larger values reduce
                   disk traffic at the cost of more re-work on resume.
    max_steps      Hard cap on total node transitions across the whole run
                   (safety net against pathological cycles). Default 1000.
    tracker        Optional RunStateTracker. When provided, its snapshot is
                   persisted into the checkpoint after each node transition,
                   enabling full state reconstruction on resume.
    """

    def __init__(
        self,
        graph: Graph,
        store: Optional[Store] = None,
        checkpoint_every: Optional[int] = CHECKPOINT_ALWAYS,
        max_steps: int = 1000,
        tracker: Optional[Any] = None,
    ) -> None:
        self.graph = graph
        self.graph.validate()
        self.store: Store = store if store is not None else MemoryStore()
        self.checkpoint_every = checkpoint_every
        self.max_steps = max_steps
        self._tracker = tracker
        self._live_emitter: Optional[EventEmitter] = None
        self._logger = get_logger("orcha.graph.runtime")
        # Cache node lookups to avoid repeated graph traversal per step.
        # Nodes are immutable after graph construction, so this is safe.
        self._node_cache: Dict[str, Node] = dict(self.graph.nodes)

    @property
    def live_emitter(self) -> Optional[EventEmitter]:
        """The EventEmitter of the most recent run() on this runtime.

        Lets an external consumer (e.g. the SSE endpoint) attach late
        subscribers and receive events that are not persisted as
        checkpoints (node_start/node_end/fan_out/…).
        """
        return self._live_emitter

    # ── Public entrypoints ─────────────────────────────────────────────

    async def run(
        self,
        query: str,
        *,
        budget: Optional[BudgetState] = None,
        packet: Optional[OrchaPacket] = None,
        resume_from: Optional[Store] = None,
        run_id: Optional[str] = None,
        on_event: Optional[EventSubscriber] = None,
        cancel: Optional[CancelToken] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RunResult":
        """
        Execute the graph for a query.

        Either supply ``query`` (a fresh packet is created) or ``packet``
        (a pre-built packet, e.g. from a previous stage). ``budget``
        overrides the packet's BudgetState when given.

        To resume a crashed run, pass ``resume_from=store``; the runtime
        loads the latest checkpoint for the run id and continues.

        ``on_event`` registers a streaming subscriber (sync or async) that
        receives a RunEvent per node transition.

        Returns a RunResult (terminal projection of the final packet).
        """
        from ..result import RunResult

        # ── Resolve starting packet ────────────────────────────────────
        if packet is None:
            run_id = run_id or str(uuid.uuid4())
            packet = OrchaPacket(
                id=run_id, kind=PacketKind.QUERY, query=query,
                payload={"__graph__": self.graph.name},
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
            emit=emitter, logger=self._logger, graph_name=self.graph.name,
        )

        # ── Resume: skip to last checkpoint's next_node ────────────────
        start_node = self.graph.entry
        if resume_from is not None:
            cp = await resume_from.load_checkpoint(run_id)
            if cp is not None:
                if cp.next_node == END:
                    # The run already completed. Resuming it must behave
                    # like a fresh attempt on the same thread (idempotent),
                    # not re-execute with a spent budget — a completed
                    # checkpoint packet carries consumed iterations/cost
                    # that would immediately re-trip the budget.
                    if packet is None:
                        packet = OrchaPacket(
                            id=run_id, kind=PacketKind.QUERY, query=query,
                            payload={"__graph__": self.graph.name},
                            budget=budget or BudgetState(),
                            metadata=dict(metadata or {}),
                        )
                    self._logger.info(
                        "resume run=%s: already completed, starting fresh",
                        run_id[:8],
                    )
                else:
                    packet = cp.packet
                    start_node = cp.next_node
                    self._logger.info(
                        "resume run=%s from node=%s seq=%d",
                        run_id[:8], start_node, cp.seq,
                    )
                    # Restore the tracker from the persisted snapshot so
                    # sequence numbering and event history survive restarts.
                    if self._tracker is not None:
                        snapshot = packet.payload.get("_event_stream_snapshot")
                        if snapshot is not None:
                            from .context import RunStateTracker
                            restored = RunStateTracker.from_snapshot(snapshot)
                            # Copy restored state into the live tracker
                            self._tracker._seq = restored._seq
                            self._tracker._events = list(restored._events)
                            self._tracker._status = restored._status
                            self._tracker._current_task_id = restored._current_task_id
                            self._tracker._current_task_title = restored._current_task_title
                            self._tracker._completed_task_ids = list(restored._completed_task_ids)
                            self._tracker._failed_task_ids = list(restored._failed_task_ids)
                            self._tracker._plan = restored._plan
                            self._tracker._verification_result = restored._verification_result
                            self._tracker._final_response = restored._final_response
                            self._tracker._error = restored._error
                            self._tracker._created_at = restored._created_at
                            self._logger.info(
                                "restored tracker run=%s seq=%d status=%s completed=%d",
                                run_id[:8], restored._seq, restored._status,
                                len(restored._completed_task_ids),
                            )
                    await emitter.emit(
                        "checkpoint", "", packet, action="resume", seq=cp.seq,
                    )
            else:
                self._logger.info("resume run=%s: no checkpoint, starting fresh", run_id[:8])

        await emitter.emit("run_start", "", packet, graph=self.graph.name, entry=start_node)

        # ── Walk ───────────────────────────────────────────────────────
        token = trace_id_var.set(run_id)
        final_packet: OrchaPacket
        try:
            final_packet = await self._walk(packet, start_node, ctx)
        except GraphError as exc:
            await emitter.emit("error", exc.node or "", packet, error=type(exc).__name__, message=str(exc))
            raise
        finally:
            trace_id_var.reset(token)

        await emitter.emit("run_complete", "", final_packet)
        return RunResult(final_packet, graph_name=self.graph.name, run_id=run_id)

    async def replay(
        self,
        run_id: str,
        store: Optional[Store] = None,
        on_event: Optional[EventSubscriber] = None,
    ) -> "RunResult":
        """
        Deterministically re-execute a run from its first checkpoint.

        Replay re-emits every event as if the run were happening live,
        making it invaluable for debugging and benchmarking. The run
        re-uses the original packet (so budgets and ids are identical)
        and walks from the entry node, but the Store is consulted as
        read-only history — no new checkpoints are written.
        """
        from ..result import RunResult

        store = store or self.store
        cps = await store.list_checkpoints(run_id)
        if not cps:
            raise GraphError(f"No checkpoints found for run {run_id}")
        first = cps[0]
        packet = first.packet

        cancel = CancelToken()
        emitter = EventEmitter(run_id, logger=self._logger)
        self._live_emitter = emitter
        if on_event is not None:
            emitter.subscribe(on_event)
        ctx = RunContext(
            run_id=run_id, store=store, cancel=cancel,
            emit=emitter, logger=self._logger, graph_name=self.graph.name,
        )
        await emitter.emit("run_start", "", packet, graph=self.graph.name, entry=self.graph.entry, replay=True)
        token = trace_id_var.set(run_id)
        try:
            final = await self._walk(packet, self.graph.entry, ctx, replay=True)
        finally:
            trace_id_var.reset(token)
        await emitter.emit("run_complete", "", final, replay=True)
        return RunResult(final, graph_name=self.graph.name, run_id=run_id)

    # ── The traversal loop ─────────────────────────────────────────────

    async def _walk(
        self,
        packet: OrchaPacket,
        start_node: str,
        ctx: RunContext,
        replay: bool = False,
    ) -> OrchaPacket:
        """
        Walk node-to-node from ``start_node`` until termination.

        Termination conditions:
          - reach END
          - a conditional predicate returns None (soft END)
          - budget exhausted (BudgetExceeded)
          - cancellation (Cancelled)
          - control-node failure after retries (NodeFailed)
          - max_steps exceeded (GraphError)
        """
        current = packet
        node_name = start_node
        steps = 0
        transitions_since_ckpt = 0

        while True:
            # Guard rails checked at the top of every iteration.
            if ctx.cancelled:
                await ctx.emit.emit("cancel", node_name, current, reason=ctx.cancel.reason or "cancelled")
                raise Cancelled(ctx.cancel.reason or "cancelled", node=node_name)
            # Resource overspend (cost/latency) is a hard stop: the run cannot
            # continue spending. The iteration cap is deliberately NOT raised
            # here — the planner and retry controller turn it into a graceful
            # loop stop (stop -> finalize), so a run that completes within its
            # cap is a success, not an error.
            if (
                current.budget.cost_used >= current.budget.max_cost
                or current.budget.latency_used_s >= current.budget.max_latency_s
            ):
                raise BudgetExceeded("budget exhausted before node " + repr(node_name), node=node_name)
            steps += 1
            if steps > self.max_steps:
                raise GraphError(
                    f"Exceeded max_steps={self.max_steps} — possible cycle",
                    node=node_name,
                )

            if node_name == END:
                return current

            node = self._node_cache.get(node_name) or self.graph.get_node(node_name)

            # ── Execute the node (timeout, retry, fault isolation) ────
            result = await self._exec_node(node, current, ctx)
            current = result.packet
            transitions_since_ckpt += 1

            # ── Checkpoint (throttled; skipped during replay) ─────────
            if not replay and self._should_checkpoint(transitions_since_ckpt):
                next_after = self._peek_next(node_name, current)
                # Persist the event adapter's tracker snapshot into the packet
                # so it survives server restarts and enables full state
                # reconstruction on resume.
                adapter = current.payload.get("_event_adapter")
                if adapter is not None and hasattr(adapter, "_tracker"):
                    tracker = adapter._tracker
                    if hasattr(tracker, "persist_to_packet"):
                        tracker.persist_to_packet(current)
                cp = await self.store.save_checkpoint(
                    ctx.run_id, node_name, next_after, current,
                )
                transitions_since_ckpt = 0
                await ctx.emit.emit(
                    "checkpoint", node_name, current,
                    checkpoint_id=cp.id, seq=cp.seq, next_node=next_after,
                )

            # ── Resolve the next node ─────────────────────────────────
            nxt = self._resolve_next(node_name, current, ctx)
            if nxt is None:
                # Soft END: predicate returned None or no outgoing edge.
                return current
            if isinstance(nxt, _ScatterPlan):
                # Fan-out: run branches, gather, continue from gather node.
                current = await self._run_scatter(nxt, current, ctx)
                node_name = nxt.gather_node
                continue
            node_name = nxt

    # ── Node execution with timeout + retry ────────────────────────────

    async def _exec_node(
        self, node: Node, packet: OrchaPacket, ctx: RunContext,
    ) -> "_NodeExecResult":
        """
        Run one node with full safety wrapping.

        Returns a _NodeExecResult carrying the (possibly forked) output
        packet and the wall-clock duration. On failure after retries,
        raises NodeFailed (control nodes) — but gather/leaf fault isolation
        is handled at the branch level in _run_scatter.
        """
        await ctx.emit.emit("node_start", node.name, packet)
        await node.on_enter(packet, ctx)
        t0 = time.perf_counter()
        attempts = 1 + max(0, node.retries)
        last_exc: Optional[BaseException] = None

        for attempt in range(1, attempts + 1):
            if ctx.cancelled:
                await ctx.emit.emit("cancel", node.name, packet, reason="cancelled")
                raise Cancelled(ctx.cancel.reason or "cancelled", node=node.name)
            attempt_ctx = self._ctx_with_attempt(ctx, attempt)
            try:
                if node.timeout_s is not None:
                    out = await asyncio.wait_for(
                        node.run(packet, attempt_ctx), timeout=node.timeout_s,
                    )
                else:
                    out = await node.run(packet, attempt_ctx)
                # A scatter node returns a ScatterResult, not a packet.
                # Translate it into a packet carrying the branches under
                # '__scatter__' so the fan-out machinery can read them.
                # ScatterResult.branches is a list of (key, packet, entry_node).
                if isinstance(out, ScatterResult):
                    out = packet.fork(
                        packet.kind,
                        __scatter__={
                            "branches": list(out.branches),
                            "gather_to": out.gather_to,
                        },
                    )
                if not isinstance(out, OrchaPacket):
                    raise TypeError(
                        f"Node {node.name!r} returned {type(out).__name__}, "
                        f"expected OrchaPacket"
                    )
                duration_ms = (time.perf_counter() - t0) * 1000
                # Stamp a trace step on the output packet so the run's trace
                # captures every node transition uniformly.
                out = out.stamp(node.name, duration_ms, attempt=attempt)
                await node.on_success(packet, out, ctx)
                await ctx.emit.emit(
                    "node_end", node.name, out,
                    duration_ms=round(duration_ms, 2), attempt=attempt,
                )
                return _NodeExecResult(packet=out, duration_ms=duration_ms)
            except asyncio.TimeoutError:
                last_exc = NodeTimeout(
                    f"Node {node.name!r} exceeded timeout {node.timeout_s}s",
                    node=node.name,
                )
                self._logger.warning(
                    "node_timeout node=%s timeout=%s attempt=%d",
                    node.name, node.timeout_s, attempt,
                )
            except Cancelled:
                raise  # propagate immediately, no retry
            except GraphError:
                raise  # budget/cancellation-style errors are not retryable
            except Exception as exc:  # noqa: BLE001 — normalized below
                last_exc = exc
                self._logger.warning(
                    "node_failed node=%s attempt=%d error=%s",
                    node.name, attempt, exc,
                )

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

    # ── Edge resolution ────────────────────────────────────────────────

    def _resolve_next(
        self, node_name: str, packet: OrchaPacket, ctx: RunContext,
    ) -> Optional[Union[str, "_ScatterPlan"]]:
        """
        Decide which node runs next, or return None for soft-END.

        Precedence at a node with multiple outgoing edges:
          1. FanOutEdge (scatter) — if present, it wins.
          2. ConditionalEdge — predicate picks the label.
          3. Edge — unconditional destination.

        A node with no outgoing edges at all terminates the run (soft END).
        """
        outs = self.graph.outgoing(node_name)

        # Fan-out takes precedence.
        for e in outs:
            if isinstance(e, FanOutEdge):
                return _ScatterPlan(scatter_node=node_name, gather_node=e.to_node)

        # Conditional.
        for e in outs:
            if isinstance(e, ConditionalEdge):
                try:
                    label = e.predicate(packet)
                except Exception as exc:
                    raise NoRoute(
                        f"Conditional predicate at {node_name!r} raised: {exc}",
                        node=node_name,
                    ) from exc
                if label is None:
                    return None  # soft END
                if label not in e.routes:
                    raise NoRoute(
                        f"Conditional predicate at {node_name!r} returned "
                        f"label {label!r} which has no route (known: "
                        f"{sorted(e.routes)})", node=node_name,
                    )
                dest = e.routes[label]
                return None if dest == END else dest

        # Unconditional.
        for e in outs:
            if isinstance(e, Edge):
                return None if e.to_node == END else e.to_node

        # No outgoing edges: soft END.
        return None

    def _peek_next(self, node_name: str, packet: OrchaPacket) -> str:
        """Best-effort next-node guess for checkpoint metadata."""
        nxt = self._resolve_next(node_name, packet, _NULL_CTX)
        if nxt is None:
            return END
        if isinstance(nxt, _ScatterPlan):
            return nxt.gather_node
        return nxt

    # ── Fan-out / fan-in ───────────────────────────────────────────────

    async def _run_scatter(
        self, plan: "_ScatterPlan", packet: OrchaPacket, ctx: RunContext,
    ) -> OrchaPacket:
        """
        Execute a scatter→gather cycle.

        The scatter node has already run (in _exec_node); its output packet
        carries the branches under payload['__scatter__'] as a list of
        (key, child_packet, entry_node) triples.

        Steps:
          1. Read branches from packet.payload['__scatter__'].
          2. Run each branch concurrently from its entry node to termination.
          3. Collect the terminal child packets.
          4. Build a gather input packet with payload['__gathered__'] set.
          5. Return it — the walk loop will execute the gather node next.

        The gather node is NOT executed here; the walk loop picks it up
        naturally via ``node_name = gather_node`` on the next iteration.
        """
        scatter_payload = packet.payload.get("__scatter__")
        if scatter_payload is None:
            raise GatherError(
                f"Scatter node {plan.scatter_node!r} did not produce branches",
                node=plan.scatter_node,
            )
        branches: List[tuple] = scatter_payload.get("branches", [])
        gather_to = scatter_payload.get("gather_to", plan.gather_node)

        await ctx.emit.emit(
            "fan_out", plan.scatter_node, packet,
            branches=[k for k, _, _ in branches], gather_to=gather_to,
        )

        # Run every branch concurrently. Each branch is itself a sub-walk
        # from the branch's nominated entry node to termination.
        async def _run_branch(key: str, child: OrchaPacket, entry: str) -> OrchaPacket:
            branch_cancel = CancelToken()
            # Propagate parent cancellation into branches.
            if ctx.cancelled:
                branch_cancel.cancel(ctx.cancel.reason or "parent cancelled")
            try:
                return await self._walk_branch(child, entry, ctx, branch_cancel)
            except Cancelled:
                return child.error_packet(f"branch {key!r} cancelled")

        tasks = [_run_branch(key, child, entry) for key, child, entry in branches]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        await ctx.emit.emit("fan_in", gather_to, packet, count=len(results))

        # Build the gather input packet: carry the gathered children in payload
        # so the GatherNode.run() can read them. The walk loop will execute the
        # gather node on the next iteration — we do NOT run it here to avoid
        # double execution. Strip __scatter__ from the forked payload so it
        # does not leak downstream.
        clean_payload = {
            k: v for k, v in packet.payload.items() if k != "__scatter__"
        }
        gather_packet = OrchaPacket(
            parent_id=packet.id,
            kind=packet.kind,
            query=packet.query,
            payload={**clean_payload, "__gathered__": [r for r in results]},
            budget=packet.budget.model_copy(deep=True),
            trace=list(packet.trace),
            metadata=dict(packet.metadata),
            tags=list(packet.tags),
        )

        # If a merge_fn was registered on the FanInEdge, apply it first.
        fin = self._find_fan_in_edge(plan.scatter_node, gather_to)
        if fin is not None and fin.merge_fn is not None:
            gather_packet = fin.merge_fn(gather_packet, list(results))

        return gather_packet

    async def _walk_branch(
        self,
        packet: OrchaPacket,
        entry: str,
        parent_ctx: RunContext,
        cancel: CancelToken,
    ) -> OrchaPacket:
        """
        Walk a single fan-out branch to termination.

        Branches share the parent's store and emitter (so events stream
        under the same run id) but get their own cancellation token so a
        slow branch can be killed without taking down siblings.
        """
        branch_ctx = RunContext(
            run_id=parent_ctx.run_id, store=parent_ctx.store,
            cancel=cancel, emit=parent_ctx.emit, logger=parent_ctx.logger,
            graph_name=parent_ctx.graph_name,
        )
        # Branch budget is inherited from the packet (already forked).
        # Limit branch steps to avoid runaway sub-walks.
        return await self._walk_with_limit(packet, entry, branch_ctx, max_steps=200)

    async def _walk_with_limit(
        self, packet: OrchaPacket, entry: str, ctx: RunContext, max_steps: int,
    ) -> OrchaPacket:
        """A bounded _walk used for fan-out branches."""
        saved = self.max_steps
        self.max_steps = max_steps
        try:
            return await self._walk(packet, entry, ctx)
        finally:
            self.max_steps = saved

    def _find_fan_in_edge(self, scatter: str, gather: str) -> Optional[FanInEdge]:
        for e in self.graph.edges:
            if isinstance(e, FanInEdge) and e.from_node == scatter and e.to_node == gather:
                return e
        return None

    # ── Checkpoint policy ──────────────────────────────────────────────

    def _should_checkpoint(self, transitions: int) -> bool:
        if self.checkpoint_every is None:
            return True
        return transitions >= self.checkpoint_every

    # ── Context helpers ────────────────────────────────────────────────

    def _ctx_with_attempt(self, ctx: RunContext, attempt: int) -> RunContext:
        return RunContext(
            run_id=ctx.run_id, store=ctx.store, cancel=ctx.cancel,
            emit=ctx.emit, logger=ctx.logger, graph_name=ctx.graph_name,
            attempt=attempt,
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

class _NodeExecResult:
    __slots__ = ("packet", "duration_ms")

    def __init__(self, packet: OrchaPacket, duration_ms: float) -> None:
        self.packet = packet
        self.duration_ms = duration_ms


class _ScatterPlan:
    """Internal marker: the next 'step' is a scatter→gather cycle."""
    __slots__ = ("scatter_node", "gather_node")

    def __init__(self, scatter_node: str, gather_node: str) -> None:
        self.scatter_node = scatter_node
        self.gather_node = gather_node


# A null context used only by _peek_next when we need to call _resolve_next
# for checkpoint metadata but have no live RunContext. _resolve_next does
# not actually use the ctx argument, so a placeholder is safe.
_NULL_CTX = None  # type: Optional[RunContext]


__all__ = ["GraphRuntime", "CHECKPOINT_ALWAYS"]
