"""
Tests for execution metrics tracking.
"""
from __future__ import annotations

import os
import sys
import types

# Bypass orcha's heavy __init__.py — same approach as test_new_signals.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_fake = types.ModuleType("orcha")
_fake.__path__ = [os.path.join(os.path.dirname(__file__), "..", "orcha")]
# sys.modules["orcha"] = _fake

from orcha.nodes.execution_metrics import RunMetrics, TaskMetrics, estimate_tokens


def test_estimate_tokens():
    """Rough token estimation."""
    assert estimate_tokens("") >= 1
    assert estimate_tokens("hello") >= 1
    assert estimate_tokens("x" * 100) == 25
    assert estimate_tokens("x" * 400) == 100


def test_run_metrics_create():
    """Can create a RunMetrics instance."""
    m = RunMetrics(run_id="test-123", objective="Build feature")
    assert m.run_id == "test-123"
    assert m.objective == "Build feature"
    assert m.tasks == []
    assert m.status == "running"


def test_run_metrics_record_task_start():
    """Record task start creates a TaskMetrics."""
    m = RunMetrics(run_id="r1")
    tm = m.record_task_start("t1", title="Inspect repo")
    assert tm.task_id == "t1"
    assert tm.title == "Inspect repo"
    assert tm.started_at > 0
    assert len(m.tasks) == 1


def test_run_metrics_record_task_complete():
    """Record task completion fills in metrics."""
    m = RunMetrics(run_id="r1")
    m.record_task_start("t1", title="Inspect")
    m.record_task_complete(
        "t1",
        tokens_in=100,
        tokens_out=50,
        tool_calls=3,
        tool_names=["read_file", "list_directory"],
        model_calls=2,
        iterations=3,
        success=True,
    )
    tm = m.tasks[0]
    assert tm.tokens_in == 100
    assert tm.tokens_out == 50
    assert tm.total_tokens == 150
    assert tm.tool_calls == 3
    assert tm.tool_names == ["read_file", "list_directory"]
    assert tm.model_calls == 2
    assert tm.iterations == 3
    assert tm.success is True
    assert tm.duration_s >= 0


def test_run_metrics_record_task_complete_auto_creates():
    """Record complete auto-creates task if not found."""
    m = RunMetrics(run_id="r1")
    m.record_task_complete("t_new", success=True)
    assert len(m.tasks) == 1
    assert m.tasks[0].task_id == "t_new"
    assert m.tasks[0].success is True


def test_run_metrics_record_context_estimate():
    """Record context token estimates."""
    m = RunMetrics(run_id="r1")
    m.record_task_start("t1")
    m.record_context_estimate(
        "t1",
        system_prompt_tokens=200,
        user_message_tokens=100,
        prior_results_tokens=50,
    )
    tm = m.tasks[0]
    assert tm.system_prompt_tokens_est == 200
    assert tm.user_message_tokens_est == 100
    assert tm.prior_results_tokens_est == 50


def test_run_metrics_finalize():
    """Finalize returns complete report."""
    m = RunMetrics(run_id="r1", objective="Test objective")
    m.record_task_start("t1", title="Task 1")
    m.record_task_complete("t1", tokens_in=100, tokens_out=50, success=True)
    m.record_task_start("t2", title="Task 2")
    m.record_task_complete("t2", tokens_in=80, tokens_out=30, success=False, error="failed")
    m.planner_calls = 1
    m.observer_calls = 2

    report = m.finalize(status="completed")
    assert report["run_id"] == "r1"
    assert report["status"] == "completed"
    assert report["task_count"] == 2
    assert report["successful_tasks"] == 1
    assert report["failed_tasks"] == 1
    assert report["total_tokens"] == 260
    assert report["planner"]["calls"] == 1
    assert report["observer"]["calls"] == 2
    assert len(report["tasks"]) == 2
    assert report["duration_s"] >= 0


def test_run_metrics_summary():
    """Summary returns a readable string."""
    m = RunMetrics(run_id="test-run-id", objective="Test")
    m.record_task_start("t1")
    m.record_task_complete("t1", tokens_in=100, tokens_out=50, success=True)
    summary = m.summary()
    assert "test-run" in summary
    assert "1/1 tasks OK"
    assert "150 tokens" in summary


def test_run_metrics_multiple_tasks():
    """Multiple tasks track correctly."""
    m = RunMetrics(run_id="r1")
    for i in range(5):
        m.record_task_start(f"t{i}", title=f"Task {i}")
        m.record_task_complete(
            f"t{i}",
            tokens_in=100 * (i + 1),
            tokens_out=50 * (i + 1),
            success=i < 4,
            error=None if i < 4 else "failed",
            tool_calls=i + 1,
        )
    report = m.finalize()
    assert report["task_count"] == 5
    assert report["successful_tasks"] == 4
    assert report["failed_tasks"] == 1
    assert report["total_tokens"] == sum(150 * (i + 1) for i in range(5))


def test_task_metrics_total_tokens():
    """TaskMetrics total_tokens sums in and out."""
    tm = TaskMetrics(task_id="t1", tokens_in=100, tokens_out=50)
    assert tm.total_tokens == 150


def test_task_metrics_tokens_per_iteration():
    """tokens_per_iteration divides correctly."""
    tm = TaskMetrics(task_id="t1", tokens_in=100, tokens_out=50, iterations=3)
    assert tm.tokens_per_iteration == 50.0


def test_task_metrics_tokens_per_iteration_zero():
    """tokens_per_iteration is 0 when no iterations."""
    tm = TaskMetrics(task_id="t1")
    assert tm.tokens_per_iteration == 0.0


def test_task_metrics_to_dict():
    """TaskMetrics serializes to dict."""
    tm = TaskMetrics(
        task_id="t1",
        title="Test task",
        tokens_in=100,
        tokens_out=50,
        tool_calls=2,
        tool_names=["read_file"],
        iterations=3,
        success=True,
    )
    d = tm.to_dict()
    assert d["task_id"] == "t1"
    assert d["title"] == "Test task"
    assert d["total_tokens"] == 150
    assert d["tool_calls"] == 2
    assert d["tool_names"] == ["read_file"]
    assert d["iterations"] == 3
    assert d["success"] is True


def test_run_metrics_planner_costs():
    """Planner costs are tracked."""
    m = RunMetrics(run_id="r1")
    m.planner_calls = 2
    m.planner_tokens_in = 500
    m.planner_tokens_out = 200
    report = m.finalize()
    assert report["planner"]["calls"] == 2
    assert report["planner"]["tokens_in"] == 500
    assert report["planner"]["tokens_out"] == 200


def test_run_metrics_replanner_costs():
    """Replanner costs are tracked."""
    m = RunMetrics(run_id="r1")
    m.replanner_calls = 1
    m.replanner_tokens_in = 300
    m.replanner_tokens_out = 100
    m.replan_count = 1
    report = m.finalize()
    assert report["replanner"]["calls"] == 1
    assert report["replanner"]["tokens_in"] == 300
    assert report["replanner"]["tokens_out"] == 100
    assert report["replan_count"] == 1
