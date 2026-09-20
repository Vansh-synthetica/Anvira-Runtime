"""
Tests for run interruption and resume.

Verifies that:
1. RunStateTracker snapshot persists into packet payload
2. RunStateTracker can be restored from a persisted snapshot
3. Event history, task tracking, and sequence numbers survive restore
4. Completed tasks are preserved across resume
5. The RUN_RESUMED event kind exists and serializes correctly
6. Checkpoint store round-trips packet with tracker snapshot
7. GraphRuntime persists tracker snapshot during checkpoint
"""
from __future__ import annotations

import os
import sys
import time
import json
import asyncio
import pytest

# Bypass orcha/__init__.py heavy imports by importing submodules directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import types
_fake = types.ModuleType("orcha")
_fake.__path__ = [os.path.join(os.path.dirname(__file__), "..", "orcha")]
# sys.modules["orcha"] = _fake

from orcha.core.packets import (  # noqa: E402
    ExecutionEvent, ExecutionEventKind, OrchaPacket, PacketKind,
    RunStateSnapshot, ExecutionPlan, TaskStep, TaskResult,
    ObjectiveVerificationResult, FinalResponse,
)
from orcha.graph.context import (  # noqa: E402
    RunStateTracker, EventEmitter,
    EVT_EXEC_RUN_CREATED, EVT_EXEC_TASK_STARTED, EVT_EXEC_TASK_COMPLETED,
    EVT_EXEC_RUN_COMPLETED, EVT_EXEC_RUN_RESUMED,
)
from orcha.graph.store import MemoryStore, Checkpoint  # noqa: E402


# ── 1. RunStateTracker snapshot persistence ──────────────────────────────────

class TestTrackerSnapshotPersistence:
    """Verify that persist_to_packet serializes tracker state into packet."""

    def test_persist_creates_snapshot_in_payload(self):
        tracker = RunStateTracker("run-1", objective="Build feature X")
        tracker.emit_event(EVT_EXEC_RUN_CREATED, summary="Run started")
        tracker.emit_event(EVT_EXEC_TASK_STARTED, summary="Task 1 started",
                           task_id="t1", task_title="Implement auth")
        tracker._completed_task_ids.append("t1")

        packet = OrchaPacket(
            id="run-1", kind=PacketKind.QUERY, query="Build feature X",
        )
        tracker.persist_to_packet(packet)

        assert "_event_stream_snapshot" in packet.payload
        snap = packet.payload["_event_stream_snapshot"]
        assert isinstance(snap, dict)
        assert snap["run_id"] == "run-1"
        assert snap["status"] == "created"
        assert snap["objective"] == "Build feature X"
        assert len(snap["events"]) == 2
        assert snap["completed_task_ids"] == ["t1"]

    def test_persist_preserves_event_history(self):
        tracker = RunStateTracker("run-2")
        kinds = [
            EVT_EXEC_RUN_CREATED, EVT_EXEC_TASK_STARTED,
            EVT_EXEC_TASK_COMPLETED, EVT_EXEC_RUN_COMPLETED,
            EVT_EXEC_TASK_STARTED,
        ]
        for i, kind in enumerate(kinds):
            tracker.emit_event(kind, summary=f"Event {i}")

        packet = OrchaPacket(id="run-2", kind=PacketKind.QUERY, query="test")
        tracker.persist_to_packet(packet)

        snap = packet.payload["_event_stream_snapshot"]
        assert len(snap["events"]) == 5
        seqs = [e["seq"] for e in snap["events"]]
        assert seqs == sorted(seqs), "Events must be in order"


# ── 2. RunStateTracker restore from snapshot ─────────────────────────────────

class TestTrackerRestore:
    """Verify that from_snapshot reconstructs tracker state correctly."""

    def test_basic_restore(self):
        original = RunStateTracker("run-3", objective="Test restore")
        original.emit_event(EVT_EXEC_RUN_CREATED, summary="Run started")
        original.emit_event(EVT_EXEC_TASK_STARTED, summary="Task started",
                           task_id="t1")
        original._completed_task_ids.append("t1")
        original._failed_task_ids.append("t2")
        original._status = "running"
        original._current_task_id = "t3"
        original._current_task_title = "Current task"

        packet = OrchaPacket(id="run-3", kind=PacketKind.QUERY, query="test")
        original.persist_to_packet(packet)

        snapshot = packet.payload["_event_stream_snapshot"]
        restored = RunStateTracker.from_snapshot(snapshot)

        assert restored.run_id == "run-3"
        assert restored.objective == "Test restore"
        assert restored._status == "running"
        assert restored._current_task_id == "t3"
        assert restored._current_task_title == "Current task"
        assert restored._completed_task_ids == ["t1"]
        assert restored._failed_task_ids == ["t2"]
        assert len(restored._events) == 2

    def test_sequence_counter_restores(self):
        original = RunStateTracker("run-4")
        for i in range(10):
            original.emit_event(EVT_EXEC_TASK_STARTED, summary=f"Event {i}")

        packet = OrchaPacket(id="run-4", kind=PacketKind.QUERY, query="test")
        original.persist_to_packet(packet)

        snapshot = packet.payload["_event_stream_snapshot"]
        restored = RunStateTracker.from_snapshot(snapshot)

        assert restored._seq == 10, "Sequence counter should restore from last event seq"
        # New events should continue from the restored seq
        new_event = restored.emit_event(EVT_EXEC_TASK_COMPLETED, summary="New")
        assert new_event.seq == 11

    def test_event_history_preserved(self):
        original = RunStateTracker("run-5")
        for i in range(20):
            original.emit_event(EVT_EXEC_TASK_STARTED, summary=f"Event {i}",
                               task_id=f"t{i % 3}")

        packet = OrchaPacket(id="run-5", kind=PacketKind.QUERY, query="test")
        original.persist_to_packet(packet)

        snapshot = packet.payload["_event_stream_snapshot"]
        restored = RunStateTracker.from_snapshot(snapshot)

        assert len(restored._events) == 20
        for i, ev in enumerate(restored._events):
            assert ev.seq == i + 1
            assert ev.kind == EVT_EXEC_TASK_STARTED

    def test_plan_restored(self):
        original = RunStateTracker("run-6")
        plan = ExecutionPlan(
            objective="Build feature",
            steps=[
                TaskStep(id="s1", title="Step 1", objective="Do thing 1"),
                TaskStep(id="s2", title="Step 2", objective="Do thing 2"),
            ],
        )
        plan.steps[0].status = "completed"
        original.update_plan(plan)

        packet = OrchaPacket(id="run-6", kind=PacketKind.QUERY, query="test")
        original.persist_to_packet(packet)

        snapshot = packet.payload["_event_stream_snapshot"]
        restored = RunStateTracker.from_snapshot(snapshot)

        assert restored._plan is not None
        assert restored._plan.objective == "Build feature"
        assert len(restored._plan.steps) == 2
        assert restored._plan.steps[0].status == "completed"

    def test_from_dict_snapshot(self):
        """Restore works from a plain dict (deserialized JSON)."""
        original = RunStateTracker("run-7", objective="Dict test")
        original.emit_event(EVT_EXEC_RUN_CREATED, summary="Start")
        original._completed_task_ids.append("t1")

        packet = OrchaPacket(id="run-7", kind=PacketKind.QUERY, query="test")
        original.persist_to_packet(packet)

        # Simulate round-trip through JSON
        snapshot_dict = packet.payload["_event_stream_snapshot"]
        json_str = json.dumps(snapshot_dict)
        roundtripped = json.loads(json_str)

        restored = RunStateTracker.from_snapshot(roundtripped)
        assert restored.run_id == "run-7"
        assert restored._completed_task_ids == ["t1"]
        assert len(restored._events) == 1


# ── 3. RUN_RESUMED event kind ───────────────────────────────────────────────

class TestRunResumedEventKind:
    """Verify the RUN_RESUMED event kind exists and serializes."""

    def test_run_resumed_kind_exists(self):
        assert hasattr(ExecutionEventKind, "RUN_RESUMED")
        assert ExecutionEventKind.RUN_RESUMED.value == "run_resumed"

    def test_run_resumed_event_creation(self):
        event = ExecutionEvent(
            run_id="run-8",
            seq=1,
            kind=ExecutionEventKind.RUN_RESUMED,
            ts=time.time(),
            summary="Run resumed from checkpoint",
            metadata={"checkpoint_node": "task_executor", "resume_node": "observer"},
        )
        assert event.kind == ExecutionEventKind.RUN_RESUMED
        assert event.metadata["checkpoint_node"] == "task_executor"

    def test_run_resumed_event_serialization(self):
        event = ExecutionEvent(
            run_id="run-9",
            seq=1,
            kind=ExecutionEventKind.RUN_RESUMED,
            ts=time.time(),
            summary="Run resumed",
        )
        data = event.model_dump()
        assert data["kind"] == "run_resumed"

        restored = ExecutionEvent(**data)
        assert restored.kind == ExecutionEventKind.RUN_RESUMED

    def test_run_resumed_constant_exists(self):
        assert EVT_EXEC_RUN_RESUMED == "run_resumed"


# ── 4. Checkpoint round-trip with snapshot ──────────────────────────────────

class TestCheckpointRoundTrip:
    """Verify that checkpoints with tracker snapshots survive serialization."""

    def test_memory_store_round_trip(self):
        store = MemoryStore()

        # Create a packet with a persisted snapshot
        tracker = RunStateTracker("run-10", objective="Checkpoint test")
        tracker.emit_event(EVT_EXEC_RUN_CREATED, summary="Start")
        tracker._completed_task_ids.append("t1")

        packet = OrchaPacket(id="run-10", kind=PacketKind.QUERY, query="test")
        tracker.persist_to_packet(packet)

        # Save checkpoint
        async def _run():
            cp = await store.save_checkpoint("run-10", "node_a", "node_b", packet)
            assert cp is not None
            assert "_event_stream_snapshot" in cp.packet.payload

            # Load and verify
            loaded = await store.load_checkpoint("run-10")
            assert loaded is not None
            snap = loaded.packet.payload["_event_stream_snapshot"]
            assert snap["run_id"] == "run-10"
            assert snap["completed_task_ids"] == ["t1"]
            assert len(snap["events"]) == 1

        asyncio.run(_run())

    def test_multiple_checkpoints_preserve_latest_snapshot(self):
        store = MemoryStore()

        tracker = RunStateTracker("run-11", objective="Multi-ckpt test")
        packet = OrchaPacket(id="run-11", kind=PacketKind.QUERY, query="test")

        async def _run():
            # Checkpoint 1: one event
            tracker.emit_event(EVT_EXEC_TASK_STARTED, summary="Event 1")
            tracker.persist_to_packet(packet)
            await store.save_checkpoint("run-11", "node_a", "node_b", packet)

            # Checkpoint 2: two events
            tracker.emit_event(EVT_EXEC_TASK_COMPLETED, summary="Event 2")
            tracker._completed_task_ids.append("t1")
            tracker.persist_to_packet(packet)
            await store.save_checkpoint("run-11", "node_b", "node_c", packet)

            # Latest checkpoint should have both events
            loaded = await store.load_checkpoint("run-11")
            snap = loaded.packet.payload["_event_stream_snapshot"]
            assert len(snap["events"]) == 2
            assert snap["completed_task_ids"] == ["t1"]

        asyncio.run(_run())


# ── 5. Completed tasks not repeated on restore ─────────────────────────────

class TestCompletedTasksPreserved:
    """Verify that completed tasks survive restore and aren't re-executed."""

    def test_completed_tasks_in_snapshot(self):
        tracker = RunStateTracker("run-12")
        # Simulate completing 3 tasks
        for i in range(3):
            tracker.emit_event(EVT_EXEC_TASK_STARTED, task_id=f"t{i}")
            tracker.emit_event(EVT_EXEC_TASK_COMPLETED, task_id=f"t{i}")
            tracker._completed_task_ids.append(f"t{i}")

        packet = OrchaPacket(id="run-12", kind=PacketKind.QUERY, query="test")
        tracker.persist_to_packet(packet)

        restored = RunStateTracker.from_snapshot(
            packet.payload["_event_stream_snapshot"]
        )

        assert len(restored._completed_task_ids) == 3
        assert restored._completed_task_ids == ["t0", "t1", "t2"]

    def test_failed_tasks_in_snapshot(self):
        tracker = RunStateTracker("run-13")
        tracker._failed_task_ids.append("t_fail")
        tracker._completed_task_ids.append("t_ok")

        packet = OrchaPacket(id="run-13", kind=PacketKind.QUERY, query="test")
        tracker.persist_to_packet(packet)

        restored = RunStateTracker.from_snapshot(
            packet.payload["_event_stream_snapshot"]
        )

        assert restored._failed_task_ids == ["t_fail"]
        assert restored._completed_task_ids == ["t_ok"]


# ── 6. Tracker state fields survive round-trip ──────────────────────────────

class TestTrackerStateRoundTrip:
    """Verify all tracker state fields survive persist → restore."""

    def test_error_field(self):
        tracker = RunStateTracker("run-14")
        tracker._error = "Something went wrong"
        tracker._status = "failed"

        packet = OrchaPacket(id="run-14", kind=PacketKind.QUERY, query="test")
        tracker.persist_to_packet(packet)
        restored = RunStateTracker.from_snapshot(
            packet.payload["_event_stream_snapshot"]
        )

        assert restored._error == "Something went wrong"
        assert restored._status == "failed"

    def test_metadata_field(self):
        tracker = RunStateTracker("run-15", metadata={"key": "value"})
        packet = OrchaPacket(id="run-15", kind=PacketKind.QUERY, query="test")
        tracker.persist_to_packet(packet)
        restored = RunStateTracker.from_snapshot(
            packet.payload["_event_stream_snapshot"]
        )

        assert restored._metadata == {"key": "value"}

    def test_created_at_preserved(self):
        tracker = RunStateTracker("run-16")
        original_time = tracker._created_at

        packet = OrchaPacket(id="run-16", kind=PacketKind.QUERY, query="test")
        tracker.persist_to_packet(packet)
        restored = RunStateTracker.from_snapshot(
            packet.payload["_event_stream_snapshot"]
        )

        assert restored._created_at == original_time
