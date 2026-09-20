"""
Regression tests for the plan-routing pass.

Every case here is a way a run used to end EARLY while reporting success —
the hardest class of bug in this pipeline, because each node reports "ok"
and the only visible symptom is a final answer about work that never
happened.

  R1  — a dependency on a *skipped* step no longer blocks its dependents.
        The retry handler and the replanner both convert exhausted steps to
        "skipped" and then route next_task; treating that as unresolved
        stranded every downstream step.
  R2  — a dependency on a step id that does not exist (small planning models
        invent them routinely) no longer blocks forever.
  R3  — "no runnable step" with work remaining is a STALL, routed to the
        replanner, not a completion routed to the final answer.
  R4  — the stall flag does not stick: once the plan is repaired, the very
        next pass executes instead of replanning again.
  R5  — an observer COMPLETE ("the whole objective is already satisfied")
        is rejected while steps remain unfinished.
  R6  — decision parsing never infers COMPLETE from prose. Every decision
        value is also an ordinary English word and the eval prompt lists
        all six by name, so a substring scan fired on "did not complete".
"""
import asyncio

import pytest

from orcha.builders.agent import _execution_observer_route, _task_context_route
from orcha.core.packets import (
    ExecutionPlan,
    OrchaPacket,
    PacketKind,
    TaskExecutionResult,
    TaskObservationDecision,
    TaskStep,
)
from orcha.nodes.planner import TaskContextBuilderNode
from orcha.nodes.task_executor import TaskExecutionObserver

from test_nodes import make_ctx


def _plan4() -> ExecutionPlan:
    """The shape that failed live: four chained steps, one per file/action."""
    return ExecutionPlan(
        objective="write a stats module with tests and run them",
        steps=[
            TaskStep(id="s1", title="create stats.py", objective="o1"),
            TaskStep(id="s2", title="create test_stats.py", objective="o2",
                     dependencies=["s1"]),
            TaskStep(id="s3", title="run the tests", objective="o3",
                     dependencies=["s2"]),
            TaskStep(id="s4", title="summarise", objective="o4",
                     dependencies=["s3"]),
        ],
    )


def _packet(plan: ExecutionPlan, **extra) -> OrchaPacket:
    return OrchaPacket(
        kind=PacketKind.EXECUTION,
        query=plan.objective,
        payload={"execution_plan": plan.model_dump(), **extra},
    )


# ── R1/R2: dependency resolution ─────────────────────────────────────────────

def test_skipped_dependency_does_not_strand_downstream_steps():
    plan = _plan4()
    plan.steps[0].status = "completed"
    plan.steps[1].status = "skipped"        # retry budget exhausted

    nxt = plan.next_pending_step()
    assert nxt is not None, "a skipped dependency must not block its dependents"
    assert nxt.id == "s3"
    assert not plan.is_complete
    assert not plan.is_stalled()


def test_dangling_dependency_id_does_not_block_forever():
    plan = _plan4()
    plan.steps[0].status = "completed"
    plan.steps[1].dependencies = ["s1", "step_0_setup"]   # invented by planner

    nxt = plan.next_pending_step()
    assert nxt is not None and nxt.id == "s2"


def test_is_stalled_is_the_exact_complement_of_is_complete():
    plan = _plan4()
    for step in plan.steps:
        step.status = "completed"
    assert plan.is_complete and not plan.is_stalled()
    assert plan.unfinished_steps == []


# ── R3/R4: a stall is not a completion, and does not stick ───────────────────

def test_stalled_plan_routes_to_replan_not_to_the_final_answer():
    plan = _plan4()
    plan.steps[0].status = "completed"
    plan.steps[1].status = "in_progress"    # nothing runnable, 3 unfinished

    out = asyncio.run(TaskContextBuilderNode().run(_packet(plan), make_ctx()))

    assert out.payload.get("task_blocked") is True
    assert not out.payload.get("task_complete")
    assert _task_context_route(out) == "replan"
    assert "stalled" in (out.payload.get("replan_reason") or "").lower()


def test_stall_flag_clears_once_a_step_is_runnable_again():
    builder = TaskContextBuilderNode()
    plan = _plan4()
    plan.steps[0].status = "completed"
    plan.steps[1].status = "in_progress"

    stalled = asyncio.run(builder.run(_packet(plan), make_ctx()))
    assert _task_context_route(stalled) == "replan"

    # The replanner repairs the graph. fork() inherits the whole payload, so
    # a flag left set here would route past execution on every later pass.
    repaired = ExecutionPlan(**stalled.payload["execution_plan"])
    repaired.steps[1].status = "pending"
    nxt = stalled.fork(stalled.kind, execution_plan=repaired.model_dump())

    out = asyncio.run(builder.run(nxt, make_ctx()))
    assert out.payload.get("task_blocked") is False
    assert _task_context_route(out) == "execute"
    assert out.payload.get("task_step_id") == "s2"


def test_genuinely_finished_plan_still_completes():
    plan = _plan4()
    for step in plan.steps:
        step.status = "completed"

    out = asyncio.run(TaskContextBuilderNode().run(_packet(plan), make_ctx()))
    assert out.payload.get("task_complete") is True
    assert _task_context_route(out) == "complete"


# ── R5: early-completion claims are checked against the plan ─────────────────

def test_observer_complete_is_rejected_while_steps_remain():
    plan = _plan4()
    plan.steps[0].status = "completed"
    plan.steps[1].status = "in_progress"
    observer = TaskExecutionObserver()
    result = TaskExecutionResult(
        step_id="s2", success=False, output="CANNOT COMPLETE: no such file"
    )

    out = asyncio.run(observer._handle_decision(
        TaskObservationDecision.COMPLETE, plan.step_by_id("s2"),
        result, plan, [], make_ctx(), _packet(plan),
    ))

    assert out.payload["task_action"] == "next_task"
    assert out.payload["objective_met"] is False
    assert _execution_observer_route(out) == "next_task"
    assert "remain unfinished" in out.payload["early_complete_rejected"]


def test_observer_complete_is_honoured_on_the_last_step():
    plan = _plan4()
    for step in plan.steps[:3]:
        step.status = "completed"
    plan.steps[3].status = "in_progress"
    observer = TaskExecutionObserver()
    result = TaskExecutionResult(step_id="s4", success=True, output="done")

    out = asyncio.run(observer._handle_decision(
        TaskObservationDecision.COMPLETE, plan.step_by_id("s4"),
        result, plan, [], make_ctx(), _packet(plan),
    ))

    assert out.payload["task_action"] == "verify"
    assert out.payload["objective_met"] is True
    assert _execution_observer_route(out) == "verify"


# ── R6: decision parsing ─────────────────────────────────────────────────────

@pytest.mark.parametrize("content,expected", [
    ("DECISION: MODIFY\nREASON: wrong file", TaskObservationDecision.MODIFY),
    ("DECISION: COMPLETE\nREASON: all done", TaskObservationDecision.COMPLETE),
    ("- **DECISION:** RETRY - transient error", TaskObservationDecision.RETRY),
    ("The step succeeded, so we should CONTINUE.", TaskObservationDecision.CONTINUE),
    # Prose that used to be misparsed, every instance of which ended the run:
    ("The task did not complete successfully; no file was created.", None),
    ("This is not COMPLETE. The step failed.", None),
    ("Work is incomplete and the objective is not complete.", None),
    # Ambiguous or malformed answers fall through to the rule-based path.
    ("I would CONTINUE, though some may REPLAN here.", None),
    ("DECISION: unclear", None),
])
def test_decision_parsing_never_guesses_a_run_ending_decision(content, expected):
    assert TaskExecutionObserver()._parse_decision(content) is expected
