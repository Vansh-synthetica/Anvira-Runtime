"""Tests for the execution event stream infrastructure."""
import os
import sys
import json
import time
import pytest

# Bypass heavy imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import types
_fake = types.ModuleType("orcha")
_fake.__path__ = [os.path.join(os.path.dirname(__file__), "..", "orcha")]
# sys.modules["orcha"] = _fake

from orcha.core.packets import (
    ExecutionEvent, ExecutionEventKind, RunStateSnapshot,
    TaskStep, TaskResult, TaskArtifact,
)
from orcha.graph.context import (
    EVT_EXEC_RUN_CREATED, EVT_EXEC_TASK_STARTED, EVT_EXEC_TASK_COMPLETED,
    EVT_EXEC_VERIFICATION_STARTED, EVT_EXEC_VERIFICATION_COMPLETED,
    EVT_EXEC_RUN_COMPLETED, EVT_EXEC_RUN_FAILED, EVT_EXEC_RUN_CANCELLED,
    RunStateTracker, EventEmitter,
)


# ── ExecutionEvent Model ────────────────────────────────────────────────────

def test_execution_event_creation():
    event = ExecutionEvent(
        run_id="test-run",
        seq=1,
        kind=ExecutionEventKind.RUN_CREATED,
        ts=1234567890.0,
        summary="Test run created",
        metadata={"graph_name": "default"},
    )
    assert event.run_id == "test-run"
    assert event.seq == 1
    assert event.kind == ExecutionEventKind.RUN_CREATED


def test_execution_event_with_task_fields():
    event = ExecutionEvent(
        run_id="test-run",
        seq=2,
        kind=ExecutionEventKind.TASK_STARTED,
        ts=1234567891.0,
        summary="Starting task: Build feature",
        task_id="t1",
        task_title="Build feature",
        task_status="in_progress",
        plan_progress=0.25,
    )
    assert event.task_id == "t1"
    assert event.task_title == "Build feature"
    assert event.task_status == "in_progress"
    assert event.plan_progress == 0.25


def test_execution_event_serialization():
    event = ExecutionEvent(
        run_id="test-run",
        seq=3,
        kind=ExecutionEventKind.VERIFICATION_COMPLETED,
        ts=1234567892.0,
        summary="Verification passed",
        task_id="t1",
        verification_status="verified",
        metadata={"pass_rate": 1.0},
    )
    data = event.model_dump()
    assert data["run_id"] == "test-run"
    assert data["kind"] == "verification_completed"
    assert data["verification_status"] == "verified"
    # Reconstruct from dict
    event2 = ExecutionEvent(**data)
    assert event2.seq == event.seq


# ── RunStateSnapshot Model ──────────────────────────────────────────────────

def test_run_state_snapshot_creation():
    events = [
        ExecutionEvent(
            run_id="test-run", seq=i, kind="run_created",
            ts=time.time(), summary=f"Event {i}",
        )
        for i in range(5)
    ]
    snapshot = RunStateSnapshot(
        run_id="test-run",
        status="completed",
        objective="Test objective",
        created_at=1234567890.0,
        updated_at=1234567900.0,
        events=events,
        completed_task_ids=["t1", "t2"],
        failed_task_ids=[],
    )
    assert snapshot.run_id == "test-run"
    assert snapshot.status == "completed"
    assert len(snapshot.events) == 5
    assert len(snapshot.completed_task_ids) == 2


def test_run_state_snapshot_serialization():
    snapshot = RunStateSnapshot(
        run_id="test-run",
        status="running",
        objective="Test",
        created_at=1234567890.0,
        updated_at=1234567890.0,
    )
    data = snapshot.model_dump()
    assert data["run_id"] == "test-run"
    assert data["status"] == "running"
    # Reconstruct
    snapshot2 = RunStateSnapshot(**data)
    assert snapshot2.run_id == snapshot.run_id


# ── RunStateTracker ─────────────────────────────────────────────────────────

def test_tracker_emit_event():
    tracker = RunStateTracker(run_id="test-run", objective="Test objective")
    event = tracker.emit_event("run_created", summary="Run started")
    assert len(tracker.events) == 1
    assert event.seq == 1
    assert event.kind == "run_created"


def test_tracker_sequential_ordering():
    tracker = RunStateTracker(run_id="test-run", objective="Test")
    for i in range(10):
        tracker.emit_event("run_created", summary=f"Event {i}")
    seqs = [e.seq for e in tracker.events]
    assert seqs == list(range(1, 11))


def test_tracker_snapshot():
    tracker = RunStateTracker(run_id="test-run", objective="Test")
    tracker.emit_event("run_created", summary="Run started")
    tracker.emit_event("task_started", summary="Task started", task_id="t1")
    snap = tracker.snapshot()
    assert snap.run_id == "test-run"
    assert len(snap.events) == 2
    assert snap.objective == "Test"


def test_tracker_status():
    tracker = RunStateTracker(run_id="test-run", objective="Test")
    assert tracker.status == "created"
    tracker.set_status("running")
    assert tracker.status == "running"
    tracker.set_status("completed")
    assert tracker.status == "completed"


def test_tracker_plan_and_verification():
    from orcha.core.packets import ObjectiveVerificationResult, FinalResponse
    tracker = RunStateTracker(run_id="test-run", objective="Test")
    # Plan is stored as-is (any type), but snapshot requires ExecutionPlan or None
    tracker.update_plan(None)
    # Set proper typed objects
    tracker.set_verification_result(ObjectiveVerificationResult(verified=True))
    tracker.set_final_response(FinalResponse(summary="Done"))
    snap = tracker.snapshot()
    assert snap.verification_result is not None
    assert snap.verification_result.verified is True
    assert snap.final_response is not None
    assert snap.final_response.summary == "Done"


# ── EventStreamAdapter ──────────────────────────────────────────────────────

def test_adapter_emit_run_created():
    from orcha.nodes.task_executor import EventStreamAdapter
    adapter = EventStreamAdapter(run_id="test-run", objective="Test")
    adapter.emit_run_created(objective="Test", graph_name="default")
    events = adapter.get_events()
    assert len(events) == 1
    assert events[0].kind == "run_created"


def test_adapter_emit_task_lifecycle():
    from orcha.nodes.task_executor import EventStreamAdapter
    adapter = EventStreamAdapter(run_id="test-run", objective="Test")
    adapter.emit_task_started(task_id="t1", task_title="Build feature")
    adapter.emit_tool_started(task_id="t1", tool_name="write_file")
    adapter.emit_tool_completed(task_id="t1", tool_name="write_file", success=True)
    adapter.emit_task_completed(task_id="t1", task_title="Build feature")
    events = adapter.get_events()
    assert len(events) == 4
    kinds = [e.kind for e in events]
    assert "task_started" in kinds
    assert "tool_started" in kinds
    assert "tool_completed" in kinds
    assert "task_completed" in kinds


def test_adapter_emit_verification():
    from orcha.nodes.task_executor import EventStreamAdapter
    adapter = EventStreamAdapter(run_id="test-run", objective="Test")
    adapter.emit_verification_started(task_id="t1", task_title="Build feature")
    adapter.emit_verification_completed(task_id="t1", verified=True, pass_rate=1.0)
    events = adapter.get_events()
    assert len(events) == 2
    assert events[0].kind == "verification_started"
    assert events[1].kind == "verification_completed"
    assert events[1].verification_status == "verified"


def test_adapter_emit_run_completed_with_answer():
    from orcha.nodes.task_executor import EventStreamAdapter
    adapter = EventStreamAdapter(run_id="test-run", objective="Test")
    adapter.emit_run_completed(
        summary_text="Run completed successfully",
        final_answer="The feature was built successfully.",
    )
    events = adapter.get_events()
    assert len(events) == 1
    assert events[0].kind == "run_completed"
    assert events[0].metadata.get("final_answer") == "The feature was built successfully."


def test_adapter_emit_retry_and_replan():
    from orcha.nodes.task_executor import EventStreamAdapter
    adapter = EventStreamAdapter(run_id="test-run", objective="Test")
    adapter.emit_retry_started(
        task_id="t1", task_title="Build feature",
        retry_reason="timeout", retry_count=1,
    )
    adapter.emit_replanning_started(reason="structural_change", replan_count=2)
    events = adapter.get_events()
    assert len(events) == 2
    assert events[0].kind == "retry_started"
    assert events[0].metadata["retry_count"] == 1
    assert events[1].kind == "replanning_started"
    assert events[1].metadata["replan_count"] == 2


def test_adapter_emit_run_failed():
    from orcha.nodes.task_executor import EventStreamAdapter
    adapter = EventStreamAdapter(run_id="test-run", objective="Test")
    adapter.emit_run_failed(error="Model timeout")
    events = adapter.get_events()
    assert len(events) == 1
    assert events[0].kind == "run_failed"
    assert events[0].error == "Model timeout"


def test_adapter_emit_run_cancelled():
    from orcha.nodes.task_executor import EventStreamAdapter
    adapter = EventStreamAdapter(run_id="test-run", objective="Test")
    adapter.emit_run_cancelled(reason="User cancelled")
    events = adapter.get_events()
    assert len(events) == 1
    assert events[0].kind == "run_cancelled"
    assert events[0].error == "User cancelled"


def test_adapter_get_snapshot():
    from orcha.nodes.task_executor import EventStreamAdapter
    adapter = EventStreamAdapter(run_id="test-run", objective="Test")
    adapter.emit_run_created(objective="Test")
    adapter.emit_task_started(task_id="t1", task_title="Task 1")
    snap = adapter.get_snapshot()
    assert isinstance(snap, RunStateSnapshot)
    assert snap.run_id == "test-run"
    assert len(snap.events) == 2
