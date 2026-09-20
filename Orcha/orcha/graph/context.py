"""
orcha.graph.context
====================
The per-run ambient context handed to every node during execution.

This module is the *only* place that owns:
  - the cancellation primitive      (CancelToken)
  - the streaming event primitive   (EventEmitter / RunEvent)
  - the trace-id propagation        (contextvars token for log correlation)
  - the ambient run handle          (RunContext)

Design
------
- Cancellation is **cooperative**: a node that respects the contract checks
  ``ctx.cancelled`` at natural boundaries and exits cleanly; the runtime
  also checks between nodes so a node that never polls still terminates
  promptly after the next transition.
- Event emission is **async, typed, and fault-isolated**: a buggy subscriber
  cannot crash the run (every subscriber callback is wrapped, exceptions
  are logged and dropped).
- RunContext is a frozen dataclass: nodes receive it but must not mutate
  the ambient state. Per-run mutable state lives on the packet.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

from ..core.packets import OrchaPacket, TraceStep
from ..observability import get_logger

# ── Trace-id propagation across fan-out ───────────────────────────────────────
#
# Each graph run has exactly one trace id (== packet.id). Child nodes spawned
# by fan-out inherit it automatically via contextvars, so log lines emitted
# from any branch can be correlated without explicit plumbing.

trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "orcha_trace_id", default=None
)


def current_trace_id() -> Optional[str]:
    """Return the trace id of the graph run we are currently inside, if any."""
    return trace_id_var.get()


# ── Cancellation ──────────────────────────────────────────────────────────────

class CancelToken:
    """
    Cooperative cancellation primitive.

    A node polls ``token.cancelled`` at natural boundaries; the runtime also
    checks between node transitions. Raising inside ``cancel()`` is forbidden
    — cancellation must be graceful so checkpoints and budgets stay consistent.

    Thread-safe enough for single-loop asyncio use (the only model ORCHA
    supports). The underlying event is set under the loop's mutex.
    """

    __slots__ = ("_event", "_reason")

    def __init__(self) -> None:
        self._event: asyncio.Event = asyncio.Event()
        self._reason: Optional[str] = None

    @property
    def cancelled(self) -> bool:
        """True once ``cancel()`` has been called."""
        return self._event.is_set()

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    def cancel(self, reason: str = "cancelled") -> None:
        """Mark the run as cancelled. Idempotent and non-raising."""
        if not self._event.is_set():
            self._reason = reason
            self._event.set()

    async def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until cancelled (or timeout). Returns True if cancelled."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def raise_if_cancelled(self) -> None:
        """Convenience for nodes: raise Cancelled if already cancelled."""
        if self._event.is_set():
            from .errors import Cancelled
            raise Cancelled(self._reason or "cancelled")


# ── Streaming events ──────────────────────────────────────────────────────────

# Event kinds emitted by the runtime during a graph run.
EVT_NODE_START    = "node_start"
EVT_NODE_END      = "node_end"
EVT_FAN_OUT       = "fan_out"
EVT_FAN_IN        = "fan_in"
EVT_CHECKPOINT    = "checkpoint"
EVT_CANCEL        = "cancel"
EVT_ERROR         = "error"
EVT_RUN_START     = "run_start"
EVT_RUN_COMPLETE  = "run_complete"
EVT_TEXT_DELTA    = "text_delta"

# Task orchestration events
EVT_TASK_START    = "task_start"
EVT_TASK_COMPLETE = "task_complete"
EVT_PLAN_CREATED  = "plan_created"
EVT_PLAN_UPDATED  = "plan_updated"
EVT_REPLAN        = "replan"
EVT_VERIFY        = "verify"
EVT_TASK_MODIFIED = "task_modified"
EVT_TASK_SKIPPED  = "task_skipped"
EVT_BLOCKER       = "blocker"
EVT_ARTIFACT      = "artifact"
EVT_OBJECTIVE_MET = "objective_met"

# Verification layer events
EVT_VERIFICATION_START = "verification_start"
EVT_VERIFICATION_COMPLETE = "verification_complete"
EVT_VERIFICATION_SIGNAL = "verification_signal"
EVT_RECOVERY_TASK_CREATED = "recovery_task_created"
EVT_FINAL_RESPONSE = "final_response"

# A subscriber is any callable taking a RunEvent. May be sync or async.
EventSubscriber = Callable[["RunEvent"], Union[None, Awaitable[None]]]


@dataclass
class RunEvent:
    """
    One typed event emitted during a graph run.

    Attributes
    ----------
    run_id   The trace id of the run (== packet.id).
    node     Name of the node this event concerns ("" for run-level events).
    kind     One of the EVT_* constants.
    packet   The packet state at the time of the event (forked snapshot;
             mutating it does not affect the run).
    ts       Wall-clock epoch seconds.
    data     Free-form payload (e.g. scatter keys, checkpoint path, error msg).
    """
    run_id: str
    node: str
    kind: str
    packet: OrchaPacket
    ts: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    trace_step: Optional[TraceStep] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "node": self.node,
            "kind": self.kind,
            "ts": self.ts,
            "data": self.data,
            "packet_kind": self.packet.kind.value if self.packet else None,
        }


class EventEmitter:
    """
    Fan-out event dispatcher: one emitter, many subscribers, no crashes.

    Every subscriber invocation is wrapped so that an exception in one
    subscriber is logged and dropped — observability must never break the
    pipeline (this invariant is inherited from ORCHA2's ``_emit``).

    Subscribers may be sync or async; both are awaited correctly.
    """

    def __init__(self, run_id: str, logger: Optional[logging.Logger] = None) -> None:
        self._run_id = run_id
        self._subs: List[EventSubscriber] = []
        self._logger = logger or get_logger("orcha.graph.events")

    def subscribe(self, fn: EventSubscriber) -> Callable[[], None]:
        """Register a subscriber. Returns an unsubscribe callable."""
        self._subs.append(fn)

        def _unsub() -> None:
            try:
                self._subs.remove(fn)
            except ValueError:
                pass

        return _unsub

    async def emit(self, kind: str, node: str, packet: OrchaPacket,
                   **data: Any) -> None:
        """Emit one event to all subscribers, fault-isolated."""
        if not self._subs:
            return
        event = RunEvent(
            run_id=self._run_id, node=node, kind=kind, packet=packet, data=data,
        )
        # Snapshot the subscriber list so unsubscribing during emit is safe.
        for fn in list(self._subs):
            try:
                result = fn(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001 — observability must not crash
                self._logger.exception(
                    "event_subscriber_failed kind=%s node=%s", kind, node,
                )


# ── Scatter result ────────────────────────────────────────────────────────────

@dataclass
class ScatterResult:
    """
    Output shape of a fan-out (scatter) node.

    A scatter node returns this instead of a single packet; the runtime
    spawns one child branch per entry in ``branches``, runs them concurrently,
    then hands the list of completed child packets to the matching fan-in
    (gather) node.

    Attributes
    ----------
    branches  Ordered list of (key, packet, entry_node). Each branch starts
              executing at ``entry_node`` (which must be a registered node)
              and runs to termination (END / soft-END). Keys identify
              branches so the gather node can correlate results and traces
              stay readable.
    gather_to Name of the gather node that will collect these branches.
              Must match a FanInEdge registered on the graph.
    """
    branches: List[tuple]
    gather_to: str

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError("ScatterResult must contain at least one branch")
        for b in self.branches:
            if not (isinstance(b, tuple) and len(b) == 3):
                raise TypeError(
                    "ScatterResult.branches must be a list of "
                    "(key, packet, entry_node) tuples"
                )


# ── Run context ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunContext:
    """
    The ambient per-run handle passed into every node.

    Frozen so nodes cannot accidentally mutate the run's ambient state —
    per-run mutable state lives on the packet, which a node forks.
    """
    run_id: str            # == packet.id == trace id
    store: Any             # Store (typed loosely to avoid import cycle)
    cancel: CancelToken
    emit: EventEmitter
    logger: logging.Logger
    # Read-only graph metadata available to nodes that want it.
    graph_name: str = ""
    attempt: int = 0       # which attempt of a retrying node this is (1-based)

    @property
    def cancelled(self) -> bool:
        return self.cancel.cancelled


# ── Execution Event Stream ───────────────────────────────────────────────────

# Frontend-safe event kinds — these are the semantic, observable events
# that form the execution timeline. They are a subset/abstraction of the
# internal EVT_* constants, designed for clean frontend consumption.
EVT_EXEC_RUN_CREATED        = "run_created"
EVT_EXEC_INTENT_ANALYZED    = "intent_analyzed"
EVT_EXEC_COMPLEXITY_DETERMINED = "complexity_determined"
EVT_EXEC_PLANNING_STARTED   = "planning_started"
EVT_EXEC_PLAN_CREATED       = "plan_created"
EVT_EXEC_TASK_READY         = "task_ready"
EVT_EXEC_TASK_STARTED       = "task_started"
EVT_EXEC_MODEL_STARTED      = "model_started"
EVT_EXEC_TOOL_STARTED       = "tool_started"
EVT_EXEC_TOOL_COMPLETED     = "tool_completed"
EVT_EXEC_TASK_OBSERVED      = "task_observed"
EVT_EXEC_TASK_COMPLETED     = "task_completed"
EVT_EXEC_TASK_FAILED        = "task_failed"
EVT_EXEC_RETRY_STARTED      = "retry_started"
EVT_EXEC_REPLANNING_STARTED = "replanning_started"
EVT_EXEC_PLAN_UPDATED       = "plan_updated"
EVT_EXEC_VERIFICATION_STARTED   = "verification_started"
EVT_EXEC_VERIFICATION_COMPLETED = "verification_completed"
EVT_EXEC_RUN_COMPLETED      = "run_completed"
EVT_EXEC_RUN_FAILED         = "run_failed"
EVT_EXEC_RUN_CANCELLED      = "run_cancelled"
EVT_EXEC_RUN_RESUMED        = "run_resumed"


class RunStateTracker:
    """
    Tracks run state and emits structured, frontend-safe ExecutionEvents.

    This class bridges the internal EventEmitter (which emits raw EVT_* events)
    to the structured ExecutionEvent stream that the frontend consumes.

    It maintains:
    - A sequence counter for deterministic event ordering
    - A RunStateSnapshot for state reconstruction after reconnect
    - A list of all emitted ExecutionEvents for replay

    Usage:
        tracker = RunStateTracker(run_id="abc", objective="Build feature X")
        tracker.subscribe_to(ctx.emit)  # attach to internal event emitter
        # ... run executes, tracker auto-emits structured events ...
        snapshot = tracker.snapshot()   # for get-run API
        events = tracker.events         # for event replay
    """

    def __init__(
        self,
        run_id: str,
        objective: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_id = run_id
        self.objective = objective
        self._seq = 0
        self._events: List[Any] = []  # List[ExecutionEvent] — avoid import cycle
        self._status = "created"
        self._current_task_id: Optional[str] = None
        self._current_task_title: Optional[str] = None
        self._completed_task_ids: List[str] = []
        self._failed_task_ids: List[str] = []
        self._plan: Optional[Any] = None  # ExecutionPlan
        self._verification_result: Optional[Any] = None
        self._final_response: Optional[Any] = None
        self._error: Optional[str] = None
        self._metadata = metadata or {}
        self._created_at = time.time()
        self._unsub: Optional[Callable[[], None]] = None

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def emit_event(
        self,
        kind: str,
        summary: str = "",
        task_id: Optional[str] = None,
        task_title: Optional[str] = None,
        task_status: Optional[str] = None,
        plan_progress: Optional[float] = None,
        plan_step_count: Optional[int] = None,
        plan_completed_count: Optional[int] = None,
        verification_status: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Emit a structured ExecutionEvent. Returns the event for immediate use.

        This is the primary API for emitting frontend-safe events.
        """
        from ..core.packets import ExecutionEvent, ExecutionEventKind

        # Validate kind
        try:
            event_kind = ExecutionEventKind(kind)
        except ValueError:
            event_kind = None  # Allow unknown kinds for extensibility

        event = ExecutionEvent(
            run_id=self.run_id,
            seq=self._next_seq(),
            kind=event_kind if event_kind else kind,
            ts=time.time(),
            summary=summary,
            task_id=task_id,
            task_title=task_title,
            task_status=task_status,
            plan_progress=plan_progress,
            plan_step_count=plan_step_count,
            plan_completed_count=plan_completed_count,
            verification_status=verification_status,
            error=error,
            metadata=metadata or {},
        )

        self._events.append(event)
        return event

    def subscribe_to(self, emitter: Any) -> None:
        """
        Subscribe to an EventEmitter and auto-emit structured events
        for relevant internal events.

        This bridges internal EVT_* events to ExecutionEvent stream.
        """
        async def _on_event(event: Any) -> None:
            await self._handle_internal_event(event)

        self._unsub = emitter.subscribe(_on_event)

    def unsubscribe(self) -> None:
        """Unsubscribe from the internal event emitter."""
        if self._unsub:
            self._unsub()
            self._unsub = None

    async def _handle_internal_event(self, event: Any) -> None:
        """Translate internal RunEvents to structured ExecutionEvents."""
        kind = event.kind
        data = event.data if hasattr(event, 'data') else {}

        if kind == EVT_RUN_START:
            self._status = "running"
            self.emit_event(
                EVT_EXEC_RUN_CREATED,
                summary="Execution run started",
                metadata={"graph_name": data.get("graph_name", "")},
            )

        elif kind == EVT_TASK_START:
            task_id = data.get("task_id", "")
            task_title = data.get("task_title", "")
            self._current_task_id = task_id
            self._current_task_title = task_title
            self.emit_event(
                EVT_EXEC_TASK_STARTED,
                summary=f"Starting task: {task_title}",
                task_id=task_id,
                task_title=task_title,
                task_status="in_progress",
                plan_progress=data.get("plan_progress"),
            )

        elif kind == EVT_TASK_COMPLETE:
            task_id = data.get("task_id", "")
            task_title = data.get("task_title", "")
            success = data.get("success", True)
            if success:
                self._completed_task_ids.append(task_id)
                self.emit_event(
                    EVT_EXEC_TASK_COMPLETED,
                    summary=f"Completed task: {task_title}",
                    task_id=task_id,
                    task_title=task_title,
                    task_status="completed",
                    plan_progress=data.get("plan_progress"),
                    metadata={
                        "duration_s": data.get("duration_s"),
                        "tool_calls_made": data.get("tool_calls_made"),
                        "iterations_used": data.get("iterations_used"),
                    },
                )
            else:
                self._failed_task_ids.append(task_id)
                self.emit_event(
                    EVT_EXEC_TASK_FAILED,
                    summary=f"Task failed: {task_title}",
                    task_id=task_id,
                    task_title=task_title,
                    task_status="failed",
                    error=data.get("exhaustion_reason", "unknown"),
                    plan_progress=data.get("plan_progress"),
                )
            self._current_task_id = None
            self._current_task_title = None

        elif kind == EVT_PLAN_CREATED:
            self.emit_event(
                EVT_EXEC_PLAN_CREATED,
                summary="Execution plan created",
                plan_step_count=data.get("step_count", 0),
                plan_progress=0.0,
                metadata={"objective": data.get("objective", "")},
            )

        elif kind == EVT_PLAN_UPDATED:
            self.emit_event(
                EVT_EXEC_PLAN_UPDATED,
                summary="Plan updated",
                plan_step_count=data.get("step_count"),
                plan_completed_count=data.get("completed_count"),
                plan_progress=data.get("progress"),
            )

        elif kind == EVT_REPLAN:
            self._status = "replanning"
            self.emit_event(
                EVT_EXEC_REPLANNING_STARTED,
                summary="Replanning triggered",
                metadata={"reason": data.get("reason", "")},
            )

        elif kind == EVT_TASK_MODIFIED:
            self.emit_event(
                EVT_EXEC_PLAN_UPDATED,
                summary=f"Task modified: {data.get('task_id', '')}",
                task_id=data.get("task_id"),
                metadata={"reason": data.get("reason", "")},
            )

        elif kind == EVT_TASK_SKIPPED:
            self.emit_event(
                EVT_EXEC_TASK_COMPLETED,
                summary=f"Task skipped: {data.get('task_id', '')}",
                task_id=data.get("task_id"),
                task_status="skipped",
            )

        elif kind == EVT_VERIFICATION_START:
            self.emit_event(
                EVT_EXEC_VERIFICATION_STARTED,
                summary=f"Verifying task: {data.get('step_id', '')}",
                task_id=data.get("step_id"),
                task_title=data.get("step_title"),
            )

        elif kind == EVT_VERIFICATION_COMPLETE:
            verified = data.get("verified", False)
            self.emit_event(
                EVT_EXEC_VERIFICATION_COMPLETED,
                summary=f"Verification {'passed' if verified else 'failed'}",
                task_id=data.get("step_id"),
                verification_status="verified" if verified else "failed",
                metadata={
                    "pass_rate": data.get("pass_rate"),
                    "signal_count": data.get("signal_count"),
                    "used_llm": data.get("used_llm"),
                },
            )

        elif kind == EVT_OBJECTIVE_MET:
            self.emit_event(
                EVT_EXEC_VERIFICATION_COMPLETED,
                summary="Objective verified as satisfied",
                verification_status="verified",
            )

        elif kind == EVT_RUN_COMPLETE:
            self._status = "completed"
            self.emit_event(
                EVT_EXEC_RUN_COMPLETED,
                summary="Run completed successfully",
                plan_progress=1.0,
            )

        elif kind == EVT_CANCEL:
            self._status = "cancelled"
            self._error = data.get("reason", "cancelled")
            self.emit_event(
                EVT_EXEC_RUN_CANCELLED,
                summary="Run cancelled",
                error=self._error,
            )

        elif kind == EVT_ERROR:
            self._status = "failed"
            self._error = data.get("error", "unknown error")
            self.emit_event(
                EVT_EXEC_RUN_FAILED,
                summary=f"Run failed: {self._error}",
                error=self._error,
            )

    def update_plan(self, plan: Any) -> None:
        """Update the tracked plan state."""
        self._plan = plan

    def set_verification_result(self, result: Any) -> None:
        """Set the verification result."""
        self._verification_result = result

    def set_final_response(self, response: Any) -> None:
        """Set the final response."""
        self._final_response = response

    def set_status(self, status: str) -> None:
        """Manually set run status."""
        self._status = status

    def snapshot(self) -> Any:
        """
        Create a RunStateSnapshot for state reconstruction.

        This is used by the get-run API to allow the frontend to
        reconstruct the full timeline after reconnecting.
        """
        from ..core.packets import RunStateSnapshot

        return RunStateSnapshot(
            run_id=self.run_id,
            status=self._status,
            objective=self.objective,
            created_at=self._created_at,
            updated_at=time.time(),
            plan=self._plan,
            current_task_id=self._current_task_id,
            current_task_title=self._current_task_title,
            completed_task_ids=list(self._completed_task_ids),
            failed_task_ids=list(self._failed_task_ids),
            verification_result=self._verification_result,
            final_response=self._final_response,
            events=list(self._events),
            error=self._error,
            metadata=dict(self._metadata),
        )

    def persist_to_packet(self, packet: Any) -> None:
        """
        Persist the tracker's full state into the packet payload.

        This is called after each node transition so that the tracker state
        survives server restarts. The snapshot is stored under
        ``_event_stream_snapshot`` in the packet payload, which is then
        serialized as part of the checkpoint.
        """
        snapshot = self.snapshot()
        # RunStateSnapshot is a Pydantic model — use model_dump for serialization
        if hasattr(snapshot, "model_dump"):
            packet.payload["_event_stream_snapshot"] = snapshot.model_dump()
        else:
            packet.payload["_event_stream_snapshot"] = snapshot

    @classmethod
    def from_snapshot(cls, snapshot: Any) -> "RunStateTracker":
        """
        Restore a RunStateTracker from a persisted RunStateSnapshot.

        This reconstructs the tracker's internal state from the snapshot
        stored in a checkpoint, allowing the run to continue with correct
        sequence numbering, task tracking, and event history.
        """
        # RunStateSnapshot can be a dict (deserialized) or a Pydantic model
        if hasattr(snapshot, "model_dump"):
            data = snapshot.model_dump()
        elif isinstance(snapshot, dict):
            data = snapshot
        else:
            # Fallback: create a minimal tracker
            run_id = getattr(snapshot, "run_id", "unknown")
            return cls(run_id=run_id)

        tracker = cls(
            run_id=data.get("run_id", "unknown"),
            objective=data.get("objective", ""),
            metadata=data.get("metadata"),
        )
        tracker._status = data.get("status", "created")
        tracker._current_task_id = data.get("current_task_id")
        tracker._current_task_title = data.get("current_task_title")
        tracker._completed_task_ids = list(data.get("completed_task_ids", []))
        tracker._failed_task_ids = list(data.get("failed_task_ids", []))
        tracker._error = data.get("error")
        tracker._created_at = data.get("created_at", 0.0)

        # Restore events
        events_data = data.get("events", [])
        from ..core.packets import ExecutionEvent
        for ev_data in events_data:
            if isinstance(ev_data, dict):
                try:
                    tracker._events.append(ExecutionEvent(**ev_data))
                except Exception:
                    pass  # Skip malformed events
            elif hasattr(ev_data, "kind"):
                tracker._events.append(ev_data)

        # Restore sequence counter from the last event
        if tracker._events:
            last_seq = max(
                getattr(ev, "seq", 0) for ev in tracker._events
            )
            tracker._seq = last_seq

        # Restore plan (if it's a dict, try to reconstruct ExecutionPlan)
        plan_data = data.get("plan")
        if plan_data is not None:
            if isinstance(plan_data, dict):
                try:
                    from ..core.packets import ExecutionPlan
                    tracker._plan = ExecutionPlan(**plan_data)
                except Exception:
                    tracker._plan = plan_data
            else:
                tracker._plan = plan_data

        # Restore verification result
        ver_data = data.get("verification_result")
        if ver_data is not None:
            if isinstance(ver_data, dict):
                try:
                    from ..core.packets import ObjectiveVerificationResult
                    tracker._verification_result = ObjectiveVerificationResult(**ver_data)
                except Exception:
                    tracker._verification_result = ver_data
            else:
                tracker._verification_result = ver_data

        # Restore final response
        resp_data = data.get("final_response")
        if resp_data is not None:
            if isinstance(resp_data, dict):
                try:
                    from ..core.packets import FinalResponse
                    tracker._final_response = FinalResponse(**resp_data)
                except Exception:
                    tracker._final_response = resp_data
            else:
                tracker._final_response = resp_data

        return tracker

    @property
    def events(self) -> List[Any]:
        """All emitted ExecutionEvents in order."""
        return list(self._events)

    @property
    def status(self) -> str:
        return self._status

    @property
    def seq(self) -> int:
        return self._seq
