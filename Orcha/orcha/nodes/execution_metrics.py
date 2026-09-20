"""
Execution metrics for comparing Orcha decomposed vs direct execution.

Tracks per-task and per-run metrics to enable comparison between:
- Direct single-shot execution (model solves everything at once)
- Orcha decomposed execution (Orcha breaks into tasks, model executes each)

Usage:
    metrics = RunMetrics(run_id="abc", objective="Build feature X")
    metrics.record_task_start("t1", title="Inspect repo")
    metrics.record_task_complete("t1", tokens_used=1200, tool_calls=3)
    report = metrics.finalize()
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskMetrics:
    """Metrics for a single task execution."""
    task_id: str
    title: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    duration_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    tool_calls: int = 0
    tool_names: List[str] = field(default_factory=list)
    model_calls: int = 0
    iterations: int = 0
    success: bool = False
    error: Optional[str] = None
    exhausted: bool = False
    output_length: int = 0
    malformed_outputs: int = 0
    recovery_prompts: int = 0
    observer_decision: str = ""
    # Context metrics
    system_prompt_tokens_est: int = 0
    user_message_tokens_est: int = 0
    prior_results_tokens_est: int = 0

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def tokens_per_iteration(self) -> float:
        if self.iterations == 0:
            return 0.0
        return self.total_tokens / self.iterations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "duration_s": round(self.duration_s, 2),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
            "tool_names": self.tool_names,
            "model_calls": self.model_calls,
            "iterations": self.iterations,
            "success": self.success,
            "error": self.error,
            "exhausted": self.exhausted,
            "output_length": self.output_length,
            "malformed_outputs": self.malformed_outputs,
            "recovery_prompts": self.recovery_prompts,
            "observer_decision": self.observer_decision,
            "system_prompt_tokens_est": self.system_prompt_tokens_est,
            "user_message_tokens_est": self.user_message_tokens_est,
            "prior_results_tokens_est": self.prior_results_tokens_est,
        }


@dataclass
class RunMetrics:
    """Metrics for an entire Orcha run."""
    run_id: str
    objective: str = ""
    graph_name: str = ""
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    status: str = "running"
    tasks: List[TaskMetrics] = field(default_factory=list)
    replan_count: int = 0
    verification_performed: bool = False
    final_answer_length: int = 0
    # Pipeline-level metrics
    planner_calls: int = 0
    planner_tokens_in: int = 0
    planner_tokens_out: int = 0
    observer_calls: int = 0
    observer_tokens_in: int = 0
    observer_tokens_out: int = 0
    replanner_calls: int = 0
    replanner_tokens_in: int = 0
    replanner_tokens_out: int = 0

    def record_task_start(self, task_id: str, title: str = "") -> TaskMetrics:
        """Record the start of a task execution."""
        tm = TaskMetrics(
            task_id=task_id,
            title=title,
            started_at=time.time(),
        )
        self.tasks.append(tm)
        return tm

    def record_task_complete(
        self,
        task_id: str,
        *,
        tokens_in: int = 0,
        tokens_out: int = 0,
        tool_calls: int = 0,
        tool_names: Optional[List[str]] = None,
        model_calls: int = 0,
        iterations: int = 0,
        success: bool = True,
        error: Optional[str] = None,
        exhausted: bool = False,
        output_length: int = 0,
        malformed_outputs: int = 0,
        recovery_prompts: int = 0,
        observer_decision: str = "",
    ) -> None:
        """Record task completion with metrics."""
        tm = self._find_task(task_id)
        if tm is None:
            tm = self.record_task_start(task_id)
        tm.completed_at = time.time()
        tm.duration_s = tm.completed_at - tm.started_at
        tm.tokens_in = tokens_in
        tm.tokens_out = tokens_out
        tm.tool_calls = tool_calls
        tm.tool_names = tool_names or []
        tm.model_calls = model_calls
        tm.iterations = iterations
        tm.success = success
        tm.error = error
        tm.exhausted = exhausted
        tm.output_length = output_length
        tm.malformed_outputs = malformed_outputs
        tm.recovery_prompts = recovery_prompts
        tm.observer_decision = observer_decision

    def record_context_estimate(
        self,
        task_id: str,
        *,
        system_prompt_tokens: int = 0,
        user_message_tokens: int = 0,
        prior_results_tokens: int = 0,
    ) -> None:
        """Record estimated token counts for context construction."""
        tm = self._find_task(task_id)
        if tm:
            tm.system_prompt_tokens_est = system_prompt_tokens
            tm.user_message_tokens_est = user_message_tokens
            tm.prior_results_tokens_est = prior_results_tokens

    def finalize(self, status: str = "completed") -> Dict[str, Any]:
        """Finalize the run and return the complete metrics report."""
        self.completed_at = time.time()
        self.status = status
        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full metrics report."""
        total_duration = self.completed_at - self.started_at if self.completed_at else 0
        total_tokens = sum(t.total_tokens for t in self.tasks)
        total_tool_calls = sum(t.tool_calls for t in self.tasks)
        total_model_calls = sum(t.model_calls for t in self.tasks)
        total_iterations = sum(t.iterations for t in self.tasks)
        successful_tasks = sum(1 for t in self.tasks if t.success)
        failed_tasks = sum(1 for t in self.tasks if not t.success and t.error)
        exhausted_tasks = sum(1 for t in self.tasks if t.exhausted)
        total_malformed = sum(t.malformed_outputs for t in self.tasks)
        total_recovery = sum(t.recovery_prompts for t in self.tasks)

        return {
            "run_id": self.run_id,
            "objective": self.objective[:200],
            "graph_name": self.graph_name,
            "status": self.status,
            "duration_s": round(total_duration, 2),
            "task_count": len(self.tasks),
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "exhausted_tasks": exhausted_tasks,
            "total_tokens": total_tokens,
            "total_tool_calls": total_tool_calls,
            "total_model_calls": total_model_calls,
            "total_iterations": total_iterations,
            "tokens_per_task": round(total_tokens / max(1, len(self.tasks))),
            "tokens_per_iteration": round(total_tokens / max(1, total_iterations)),
            "malformed_outputs": total_malformed,
            "recovery_prompts": total_recovery,
            "replan_count": self.replan_count,
            "verification_performed": self.verification_performed,
            "final_answer_length": self.final_answer_length,
            # Pipeline costs
            "planner": {
                "calls": self.planner_calls,
                "tokens_in": self.planner_tokens_in,
                "tokens_out": self.planner_tokens_out,
            },
            "observer": {
                "calls": self.observer_calls,
                "tokens_in": self.observer_tokens_in,
                "tokens_out": self.observer_tokens_out,
            },
            "replanner": {
                "calls": self.replanner_calls,
                "tokens_in": self.replanner_tokens_in,
                "tokens_out": self.replanner_tokens_out,
            },
            # Per-task breakdown
            "tasks": [t.to_dict() for t in self.tasks],
        }

    def _find_task(self, task_id: str) -> Optional[TaskMetrics]:
        for t in reversed(self.tasks):
            if t.task_id == task_id:
                return t
        return None

    def summary(self) -> str:
        """One-line human-readable summary."""
        d = self.to_dict()
        return (
            f"Run {self.run_id[:8]}: "
            f"{d['successful_tasks']}/{d['task_count']} tasks OK, "
            f"{d['total_tokens']} tokens, "
            f"{d['total_tool_calls']} tool calls, "
            f"{d['duration_s']}s"
        )


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (4 chars ≈ 1 token for English)."""
    return max(1, len(text) // 4)
