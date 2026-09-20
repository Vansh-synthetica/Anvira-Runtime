"""
orcha.core.packets
==================
The foundational data model for Orcha. Every component in the pipeline
receives an OrchaPacket, reads from it, writes results back, stamps a
trace entry, and returns it. This makes the entire orchestration loop a
typed, inspectable, serialisable chain of transformations.

Design goals
------------
- One type flows everywhere: no ad-hoc dicts passed between stages.
- Full observability: every decision is recorded in the trace.
- Budget enforcement: cost, latency, and iteration limits are tracked
  centrally and checked before each stage.
- Forkable: child packets inherit parent state so branches never share
  mutable references.
- Serialisable: the whole packet round-trips through JSON for logging,
  caching, and replay.
"""
from __future__ import annotations

import copy
import json
import re
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _is_json_shaped(value: Any, _depth: int = 0) -> bool:
    """
    True when ``value`` is composed only of JSON-primitive types (dict,
    list/tuple, str, int, float, bool, None) all the way down — i.e. it's
    plain data, not a live Python object (a class instance, bound method,
    function, etc.) that got stashed in a packet's free-form ``payload``
    dict and was never meant to survive a checkpoint write. Depth-capped so
    a pathological structure can't cause unbounded recursion.
    """
    if _depth > 20:
        return True
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, dict):
        return all(
            isinstance(k, str) and _is_json_shaped(v, _depth + 1)
            for k, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_is_json_shaped(v, _depth + 1) for v in value)
    return False


# ── Enumerations ─────────────────────────────────────────────────────────────

class PacketKind(str, Enum):
    """Lifecycle state of an OrchaPacket as it moves through the pipeline."""
    QUERY       = "query"       # initial user request
    SUBTASKS    = "subtasks"    # after decomposition
    PLAN        = "plan"        # after planning
    SELECTION   = "selection"   # after expert selection
    EXECUTION   = "execution"   # after parallel model execution
    AGGREGATION = "aggregation" # after answer combination / synthesis
    EVALUATION  = "evaluation"  # after quality assessment
    RETRY       = "retry"       # after retry decision
    RESPONSE    = "response"    # final output
    ERROR       = "error"       # unrecoverable failure
    ACTION      = "action"      # agent-runtime: an Action in flight on the bus
    OBSERVATION = "observation" # agent-runtime: an Observation in flight on the bus


class Domain(str, Enum):
    """Known expert domains used for routing."""
    GENERAL   = "general"
    REASONING = "reasoning"
    FINANCE   = "finance"
    CODE      = "code"
    SCIENCE   = "science"
    CREATIVE  = "creative"
    MATH      = "math"
    MEDICAL   = "medical"
    LEGAL     = "legal"
    UNKNOWN   = "unknown"


class AggregationMode(str, Enum):
    """How the aggregator combined expert outputs."""
    SYNTHESIS         = "synthesis"          # synthesizer model rewrote everything
    CONFIDENCE_WEIGHT = "confidence_weighted" # highest confidence answer wins
    VOTE              = "vote"               # majority or plurality answer
    SINGLE            = "single"             # only one expert answered
    EMPTY             = "empty"             # no expert answered


class RetryReason(str, Enum):
    """Why the retry controller decided to loop (or stop)."""
    PASSED            = "passed"
    BUDGET_EXHAUSTED  = "budget_exhausted"
    RETRYING          = "retrying"
    LOW_CONFIDENCE    = "low_confidence"
    EMPTY_ANSWER      = "empty_answer"
    ALL_EXPERTS_FAILED = "all_experts_failed"
    # A single-expert pool (the common BYOK setup: one connection registered)
    # has no alternative model to route to on retry — re-asking the exact
    # same model the exact same question again is a pure latency cost with
    # no realistic chance of a meaningfully different answer, so this stops
    # immediately on a below-threshold score instead of spending a second
    # (or third) full round-trip hoping for a better roll.
    NO_ALTERNATIVE_EXPERT = "no_alternative_expert"


# ── Sub-models ────────────────────────────────────────────────────────────────

class TraceStep(BaseModel):
    """One audit entry stamped by a pipeline stage."""
    stage:       str
    ts:          float = Field(default_factory=time.time)
    duration_ms: float = 0.0
    data:        Dict[str, Any] = {}

    def summary(self) -> str:
        extras = "  ".join(f"{k}={v}" for k, v in self.data.items())
        return f"[{self.stage:>16}] {self.duration_ms:7.2f}ms  {extras}"


class BudgetState(BaseModel):
    """
    Tracks and enforces resource limits across the whole orchestration run.

    Any of the three limits being hit marks the budget as exhausted, which
    causes the planner to stop the loop regardless of answer quality.
    """
    # ── Limits ────────────────────────────────────────────────────────
    max_cost:       float = 1.0    # USD, summed across all expert calls
    max_latency_s:  float = 120.0  # wall-clock seconds (max-latency per iter)
    max_iterations: int   = 3      # hard cap on pipeline loop count

    # ── Consumed ──────────────────────────────────────────────────────
    cost_used:      float = 0.0
    latency_used_s: float = 0.0
    iterations:     int   = 0

    # ── Derived ───────────────────────────────────────────────────────
    @property
    def exhausted(self) -> bool:
        return (
            self.cost_used      >= self.max_cost
            or self.latency_used_s >= self.max_latency_s
            or self.iterations     >= self.max_iterations
        )

    @property
    def remaining_cost(self) -> float:
        return max(0.0, self.max_cost - self.cost_used)

    @property
    def remaining_latency_s(self) -> float:
        return max(0.0, self.max_latency_s - self.latency_used_s)

    @property
    def remaining_iterations(self) -> int:
        return max(0, self.max_iterations - self.iterations)

    def utilisation(self) -> Dict[str, float]:
        """Fraction of each budget dimension consumed (0–1)."""
        return {
            "cost":       self.cost_used / self.max_cost if self.max_cost else 0,
            "latency":    self.latency_used_s / self.max_latency_s if self.max_latency_s else 0,
            "iterations": self.iterations / self.max_iterations if self.max_iterations else 0,
        }

    def most_constrained_dimension(self) -> str:
        u = self.utilisation()
        return max(u, key=u.__getitem__)


class SubTask(BaseModel):
    """A structured sub-goal produced by the decomposer."""
    id:          str
    description: str
    domain:      str   = Domain.GENERAL
    priority:    float = Field(default=0.5, ge=0.0, le=1.0)
    metadata:    Dict[str, Any] = {}


class ExpertSlot(BaseModel):
    """A selected expert with routing metadata set by the selector."""
    name:        str
    domain:      str
    description: str
    score:       float = Field(ge=0.0, le=1.0)
    excluded:    bool  = False   # True if circuit-breaker tripped this expert
    metadata:    Dict[str, Any] = {}


class ExpertResult(BaseModel):
    """
    Output from a single expert execution, fully self-describing.
    A failed result carries an error string and zero confidence.
    """
    name:           str
    output:         str
    confidence:     float = Field(default=0.0, ge=0.0, le=1.0)
    tokens:         int   = 0
    latency_s:      float = 0.0
    cost:           float = 0.0
    success:        bool  = True
    error:          Optional[str] = None
    finish_reason:  Optional[str] = None   # "stop", "length", "timeout", …
    model_name:     Optional[str] = None   # underlying model identifier
    metadata:       Dict[str, Any] = {}

    @property
    def failed(self) -> bool:
        return not self.success or not self.output.strip()

    def quality_score(self) -> float:
        """
        Simple composite quality signal used by the aggregator for ranking.
        Combines confidence with a small bonus for longer, complete answers.
        """
        if self.failed:
            return 0.0
        length_bonus = min(0.05, len(self.output.split()) / 2000)
        finish_bonus = 0.05 if self.finish_reason in ("stop", None) else 0.0
        return min(1.0, self.confidence + length_bonus + finish_bonus)


# ── Task orchestration types ──────────────────────────────────────────────────

class PlannerRequest(BaseModel):
    """Structured input for the task planner."""
    original_request:    str                    # verbatim user message
    normalized_objective: str                   # cleaned-up goal statement
    constraints:         List[str] = []         # e.g. "use Python 3.11", "no external deps"
    context:             str = ""               # relevant existing context (file listings, errors, etc.)
    tools:               List[str] = []         # available tool/capability names
    workspace_state:     str = ""               # current workspace description when available


class TaskStep(BaseModel):
    """A single task in an execution plan graph."""
    id:                    str
    title:                 str
    objective:             str
    execution_instructions: str = ""
    dependencies:          List[str] = []        # IDs of tasks that must complete first
    expected_outcome:      str = ""
    required_context:      List[str] = []        # IDs of tasks whose output is needed
    likely_tools:          List[str] = []        # tools/capabilities likely needed
    verification_criteria: str = ""              # how to verify this task succeeded
    action:                str   = "tool_use"    # "tool_use", "search", "create", "modify", "execute", "think"
    status:                str   = "pending"     # pending, in_progress, completed, failed, skipped
    result:                Optional["TaskResult"] = None
    metadata:              Dict[str, Any] = {}


class TaskResult(BaseModel):
    """Result of executing a single TaskStep."""
    step_id:     str
    output:      str
    success:     bool  = True
    error:       Optional[str] = None
    tool_calls:  List[Dict[str, Any]] = []
    tokens_used: int   = 0
    latency_s:   float = 0.0
    metadata:    Dict[str, Any] = {}


_PATH_ARG_KEYS = ("path", "file_path", "filepath", "target", "destination", "dest")
_PATH_ARG_RE = re.compile(
    r'"(?:path|file_path|filepath|target|destination|dest)"'
    r'\s*:\s*"((?:[^"\\]|\\.)*)"'
)


# apply_patch carries paths inside its patch text ("*** Update File: src/a.py"), possibly JSON-escaped.
_PATCH_PATH_RE = re.compile(r"\*\*\* (?:Add File|Update File|Delete File|Move to): ?(.+?)(?=\\n|\r?\n|\\?\"|$)")


def _paths_in_tool_args(tool_args: str) -> List[str]:
    """File paths named by a tool call, from its RAW argument string.

    Parses leniently on purpose. ``strict=False`` because a model writing a
    file embeds real newlines in the JSON string, and a regex fallback
    because the arguments may be malformed or already truncated. Reporting
    what was touched must not depend on the model's JSON being perfect.
    """
    if not tool_args:
        return []
    found: List[str] = []
    try:
        args = json.loads(tool_args, strict=False)
    except (json.JSONDecodeError, TypeError, ValueError):
        args = None
    if isinstance(args, dict):
        for key in _PATH_ARG_KEYS:
            value = args.get(key)
            if isinstance(value, str) and value and value not in found:
                found.append(value)
    if not found and "*** Begin Patch" in tool_args:
        for match in _PATCH_PATH_RE.finditer(tool_args):
            value = match.group(1).strip()
            if value and value not in found:
                found.append(value)
    if not found:
        for match in _PATH_ARG_RE.finditer(tool_args):
            value = match.group(1)
            if value and value not in found:
                found.append(value)
    return found


class TaskExecutionState(BaseModel):
    """Tracks the execution state of a single task through the loop."""
    step_id:                    str
    status:                     str   = "pending"  # pending, running, completed, failed, retrying, replanning
    iteration:                  int   = 0
    max_iterations:             int   = 5
    tool_call_count:            int   = 0
    repeated_tool_calls:        int   = 0
    empty_model_outputs:        int   = 0
    malformed_model_outputs:    int   = 0
    consecutive_errors:         int   = 0
    last_tool_call:             Optional[str] = None
    last_tool_args:             Optional[str] = None
    tool_call_history:          List[Dict[str, Any]] = []
    start_time:                 float = Field(default_factory=time.time)
    last_activity_time:         float = Field(default_factory=time.time)
    timeout_s:                  float = 300.0
    result:                     Optional[TaskResult] = None
    error_history:              List[str] = []
    metadata:                   Dict[str, Any] = {}

    @property
    def is_timed_out(self) -> bool:
        return (time.time() - self.last_activity_time) > self.timeout_s

    @property
    def should_retry(self) -> bool:
        return (
            self.consecutive_errors < 3
            and self.repeated_tool_calls < 3
            and self.empty_model_outputs < 3
            and self.malformed_model_outputs < 3
            and not self.is_timed_out
        )

    @property
    def exhaustion_reason(self) -> Optional[str]:
        if self.is_timed_out:
            return "timeout"
        if self.consecutive_errors >= 3:
            return "consecutive_model_errors"
        if self.repeated_tool_calls >= 3:
            return "repeated_tool_calls"
        if self.empty_model_outputs >= 3:
            return "empty_model_outputs"
        if self.malformed_model_outputs >= 3:
            return "malformed_model_outputs"
        if self.iteration >= self.max_iterations:
            return "max_iterations_exceeded"
        return None

    def record_tool_call(self, tool_name: str, tool_args: str) -> None:
        """Record a tool call and detect repeats."""
        self.tool_call_count += 1
        self.last_activity_time = time.time()
        call_key = f"{tool_name}:{tool_args}"
        if call_key == f"{self.last_tool_call}:{self.last_tool_args}":
            self.repeated_tool_calls += 1
        else:
            self.repeated_tool_calls = 0
        self.last_tool_call = tool_name
        self.last_tool_args = tool_args
        self.tool_call_history.append({
            "tool": tool_name,
            "args_preview": tool_args[:200],
            # Paths extracted from the FULL arguments, before the 200-char
            # truncation above. Everything downstream — the artifact
            # collector, the "files already written" context, the final
            # progress report — used to re-parse args_preview as JSON, which
            # silently found nothing for exactly the calls that matter most:
            # a create_file carrying a real file's contents blows past 200
            # chars, so the preview is cut mid-string and never parses. A run
            # that wrote two files reported zero.
            "files": _paths_in_tool_args(tool_args),
            "iteration": self.iteration,
            "timestamp": time.time(),
        })

    def record_model_output(self, content: str) -> None:
        """Record model output and detect empty/malformed."""
        self.last_activity_time = time.time()
        if not content or not content.strip():
            self.empty_model_outputs += 1
            self.consecutive_errors += 1
        else:
            self.empty_model_outputs = 0
            self.consecutive_errors = 0

    def record_error(self, error: str) -> None:
        """Record an error."""
        self.last_activity_time = time.time()
        self.consecutive_errors += 1
        self.error_history.append(error)


class TaskExecutionResult(BaseModel):
    """Structured result from executing a single task through the loop."""
    step_id:                str
    success:                bool
    output:                 str
    tool_calls_made:        int   = 0
    iterations_used:        int   = 0
    duration_s:             float = 0.0
    exhaustion_reason:      Optional[str] = None
    tool_summaries:         List[Dict[str, Any]] = []
    error:                  Optional[str] = None
    metadata:               Dict[str, Any] = {}

    @classmethod
    def from_state(cls, state: "TaskExecutionState", output: str = "", success: bool = True) -> "TaskExecutionResult":
        # Use provided output, or fall back to state result output
        final_output = output
        if not final_output and state.result:
            final_output = state.result.output or ""
        return cls(
            step_id=state.step_id,
            success=success,
            output=final_output,
            tool_calls_made=state.tool_call_count,
            iterations_used=state.iteration,
            duration_s=time.time() - state.start_time,
            exhaustion_reason=state.exhaustion_reason,
            tool_summaries=state.tool_call_history[-10:],
            error=state.error_history[-1] if state.error_history else None,
        )


class TaskObservationDecision(str, Enum):
    """Decision made by the observer after evaluating a task result."""
    CONTINUE = "continue"    # Success, plan remains valid
    RETRY    = "retry"       # Transient failure, retry same task
    MODIFY   = "modify"      # Wrong approach, change instructions and retry
    REPLAN   = "replan"      # Structural change needed in the plan graph
    BLOCK    = "block"       # External blocker prevents progress
    COMPLETE = "complete"    # Objective already satisfied


class TaskArtifact(BaseModel):
    """An artifact produced during task execution."""
    step_id:          str
    kind:             str   = "output"  # "file", "output", "observation", "command"
    description:      str   = ""
    location:         Optional[str] = None  # file path, URL, etc.
    content_preview:  Optional[str] = None
    timestamp:        float = Field(default_factory=time.time)
    metadata:         Dict[str, Any] = {}


class TaskObservation(BaseModel):
    """Observation from the Observer node after each step execution."""
    step_id:           str
    step_result:       TaskResult
    decision:          TaskObservationDecision = TaskObservationDecision.CONTINUE
    objective_met:     Optional[bool] = None  # None = not yet assessable
    issues:            List[str] = []
    suggestions:       List[str] = []
    feedback_for_retry: Optional[str] = None  # context injected on retry/modify
    artifacts:         List[TaskArtifact] = []
    metadata:          Dict[str, Any] = {}


class ReplanRequest(BaseModel):
    """Structured input for the adaptive replanner."""
    original_objective: str
    current_plan:       ExecutionPlan
    completed_results:  List[TaskResult] = []
    current_result:     Optional[TaskResult] = None
    errors:             List[str] = []
    artifacts:          List[TaskArtifact] = []
    workspace_state:    str = ""
    decision:           TaskObservationDecision = TaskObservationDecision.REPLAN
    feedback:           str = ""


class ReplanResult(BaseModel):
    """Output from the adaptive replanner — incremental plan modifications."""
    modified_steps:   List[TaskStep] = []   # Steps that changed
    added_steps:      List[TaskStep] = []   # New steps added
    removed_step_ids: List[str] = []        # Steps removed (pending/failed only)
    kept_step_ids:    List[str] = []        # Steps unchanged
    reason:           str = ""
    replan_count:     int = 0


class FailedAttempt(BaseModel):
    """One concrete thing that was tried and did not work."""
    step_id:  str
    tool:     str = ""
    args:     str = ""      # short preview, enough to recognise a repeat
    error:    str = ""
    attempts: int = 1       # how many times this exact thing was retried


class WorkingMemory(BaseModel):
    """The run's external brain: what is known, and what has already failed.

    The plan already records structure — steps, statuses, dependencies,
    results. This records the things that structure does not capture and
    that a small model cannot hold across calls:

    - ``facts``: durable findings later steps need (a path, a version, a
      command that worked).
    - ``failed_attempts``: the exact actions already proven not to work.

    The second is the one with measured value. Without it a step repeats
    an identical failing call until its budget is gone: observed live, a
    3B model passed ``cwd="stats.py"`` — a file, not a directory — to
    run_command on four consecutive attempts, receiving the same error
    each time, because nothing in its context said it had already tried
    that. Each retry began from a blank slate.
    """
    facts:           List[str] = []
    failed_attempts: List[FailedAttempt] = []

    def record_failure(
        self, step_id: str, tool: str, args: str, error: str
    ) -> None:
        """Record a failed action, collapsing exact repeats into a count."""
        args = (args or "")[:200]
        error = (error or "")[:300]
        for existing in self.failed_attempts:
            if (existing.step_id == step_id and existing.tool == tool
                    and existing.args == args):
                existing.attempts += 1
                existing.error = error or existing.error
                return
        self.failed_attempts.append(FailedAttempt(
            step_id=step_id, tool=tool, args=args, error=error,
        ))
        # Bounded: this is injected into prompts, so it cannot grow without
        # limit. The most recent failures are the relevant ones.
        if len(self.failed_attempts) > 40:
            self.failed_attempts[:] = self.failed_attempts[-40:]

    def record_fact(self, fact: str) -> None:
        fact = (fact or "").strip()
        if fact and fact not in self.facts:
            self.facts.append(fact)
            if len(self.facts) > 60:
                self.facts[:] = self.facts[-60:]

    def failures_for(self, step_id: str) -> List[FailedAttempt]:
        """Failures relevant to a step: its own, most recent first.

        Selective retrieval, not the whole transcript — the point of the
        store is that a micro-prompt stays small.
        """
        return [f for f in self.failed_attempts if f.step_id == step_id][-6:]


class ExecutionPlan(BaseModel):
    """A complete execution plan for a complex query."""
    id:              str = Field(default_factory=lambda: str(uuid.uuid4()))
    request:         Optional[PlannerRequest] = None
    objective:       str  # the original user objective
    steps:           List[TaskStep] = []
    current_step_idx: int = 0
    status:          str  = "pending"   # pending, in_progress, completed, failed, replanning
    observations:    List[TaskObservation] = []
    replan_count:    int  = 0
    max_replans:     int  = 3
    created_at:      float = Field(default_factory=time.time)
    # The persistent working memory that cuts across planning, execution
    # and synthesis. Lives on the plan so it survives every fork() and
    # checkpoint the plan already survives.
    working_memory:  WorkingMemory = Field(default_factory=WorkingMemory)
    metadata:        Dict[str, Any] = {}

    @property
    def current_step(self) -> Optional[TaskStep]:
        if 0 <= self.current_step_idx < len(self.steps):
            return self.steps[self.current_step_idx]
        return None

    @property
    def completed_steps(self) -> List[TaskStep]:
        return [s for s in self.steps if s.status == "completed"]

    @property
    def failed_steps(self) -> List[TaskStep]:
        return [s for s in self.steps if s.status == "failed"]

    @property
    def is_complete(self) -> bool:
        return all(s.status in ("completed", "skipped") for s in self.steps)

    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        return len(self.completed_steps) / len(self.steps)

    def _dependency_resolved(self, dep_id: str) -> bool:
        """Whether a single dependency id no longer blocks its dependents.

        Three cases count as resolved, and the last two are the reason this
        is not a one-line ``status == "completed"`` check:

        - ``completed`` — the ordinary case, the dependency ran and produced
          a result downstream steps can consume.
        - ``skipped`` — a terminal state ``is_complete`` already accepts.
          The retry handler (``_handle_replan``) and the replanner BOTH
          convert exhausted/failed steps to ``skipped`` and then route
          ``next_task``, so treating it as unresolved here strands every
          dependent step permanently: the plan has pending work, nothing is
          runnable, and the context builder reports the plan finished. That
          is the observed "run stops with steps still pending" stall.
        - dangling — no step in the plan carries this id at all. A
          dependency on a step that does not exist can never be satisfied
          by execution, so blocking on it is an infinite wait. Small
          planning models emit these routinely (invented ids, ids from a
          discarded draft of the plan, ids renamed by a replan). Treat it
          as planner noise rather than a real ordering constraint.
        """
        for s in self.steps:
            if s.id == dep_id:
                return s.status in ("completed", "skipped")
        return True  # dangling dependency — no such step to wait for

    def next_pending_step(self) -> Optional[TaskStep]:
        """Return the next step whose dependencies are all resolved."""
        for step in self.steps:
            if step.status != "pending":
                continue
            if all(self._dependency_resolved(dep) for dep in step.dependencies):
                return step
        return None

    @property
    def unfinished_steps(self) -> List[TaskStep]:
        """Steps that have not reached a terminal state.

        Distinct from ``get_remaining_steps`` only in intent: this is the
        exact complement of ``is_complete``, so ``is_complete`` is true iff
        this is empty. Callers use it to tell "the plan is genuinely done"
        apart from "nothing is runnable right now", which are very
        different situations that were previously indistinguishable.
        """
        return [s for s in self.steps if s.status not in ("completed", "skipped")]

    def is_stalled(self) -> bool:
        """True when work remains but no step can currently be run.

        A stall is never a completion. It means the plan graph itself is
        broken (a cycle, a dependency on a step stuck ``in_progress``, or a
        dependent of something that failed without being marked terminal)
        and needs replanning, not a final answer claiming the objective was
        met.
        """
        return bool(self.unfinished_steps) and self.next_pending_step() is None

    def step_by_id(self, step_id: str) -> Optional[TaskStep]:
        """Find a step by its ID."""
        for s in self.steps:
            if s.id == step_id:
                return s
        return None

    def get_step_ids(self) -> List[str]:
        """Return all step IDs in order."""
        return [s.id for s in self.steps]

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Return {step_id: [dependency_ids]}."""
        return {s.id: list(s.dependencies) for s in self.steps}

    def get_completed_results(self) -> List[TaskResult]:
        """Return results from all completed steps."""
        return [s.result for s in self.steps if s.status == "completed" and s.result is not None]

    def get_remaining_steps(self) -> List[TaskStep]:
        """Return steps that are not yet completed or skipped."""
        return [s for s in self.steps if s.status in ("pending", "in_progress", "failed")]

    def apply_replan(self, replan: "ReplanResult") -> None:
        """
        Apply an incremental replan result to this plan.
        Completed steps are NEVER removed or modified.
        """
        # Remove steps marked for removal (only pending/failed)
        removable_ids = set(replan.removed_step_ids)
        self.steps = [
            s for s in self.steps
            if s.id not in removable_ids or s.status in ("completed", "skipped")
        ]

        # Apply modifications to existing pending/failed steps
        mod_map = {s.id: s for s in replan.modified_steps}
        for i, step in enumerate(self.steps):
            if step.id in mod_map and step.status not in ("completed", "skipped"):
                modified = mod_map[step.id]
                self.steps[i] = modified

        # Add new steps
        existing_ids = {s.id for s in self.steps}
        for new_step in replan.added_steps:
            if new_step.id not in existing_ids:
                self.steps.append(new_step)

        self.replan_count = replan.replan_count


# ── The packet ────────────────────────────────────────────────────────────────

class OrchaPacket(BaseModel):
    """
    The universal typed message that flows through every Orcha stage.

    Contract
    --------
    - Every stage has the signature:
        def stage(self, packet: OrchaPacket) -> OrchaPacket
      (async for executor and aggregator).

    - A stage MUST NOT mutate the incoming packet's payload directly —
      it calls fork() to produce a child, then stamps the child.

    - The payload dict accumulates data as the packet traverses the pipeline.
      Later stages can read data written by earlier ones (e.g. the aggregator
      reads `results` written by the executor).

    Payload keys by stage
    ---------------------
    decompose  → subtasks: list[SubTask], domains: list[str], complexity: float
    plan       → stop: bool, parallel_width: int, quality_threshold: float,
                 force_all_experts: bool, excluded_experts: list[str]
    select     → selected_experts: list[ExpertSlot]
    execute    → results: list[ExpertResult]
    aggregate  → answer: str, confidence: float, contributors: list[str],
                 primary: str, synthesized: bool, agg_mode: str
    evaluate   → passed: bool, confidence: float, quality_dimensions: dict
    retry      → retry: bool, reason: str
    """
    id:         str  = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id:  Optional[str] = None          # set when forked from another packet
    created_at: float = Field(default_factory=time.time)
    kind:       PacketKind
    query:      str
    payload:    Dict[str, Any]  = {}
    budget:     BudgetState     = Field(default_factory=BudgetState)
    trace:      List[TraceStep] = []
    metadata:   Dict[str, Any]  = {}
    tags:       List[str]       = []          # free-form labels for filtering

    model_config = {"arbitrary_types_allowed": True}

    # ── Mutation helpers ──────────────────────────────────────────────

    def stamp(self, stage: str, duration_ms: float = 0.0, **data: Any) -> "OrchaPacket":
        """Append a trace entry. Mutates in-place; returns self for chaining."""
        self.trace.append(TraceStep(stage=stage, duration_ms=duration_ms, data=data))
        return self

    def fork(self, kind: PacketKind, **payload_updates: Any) -> "OrchaPacket":
        """
        Produce a child packet that inherits this packet's query, budget,
        trace, metadata, and tags, then merges payload_updates on top of
        the existing payload.

        The budget AND the payload are deep-copied so the child and parent
        never share mutable state — list/dict payload values written by an
        earlier stage cannot be mutated in place by a later one. This makes
        the "stages MUST NOT mutate the incoming packet's payload" contract
        enforced rather than hoped.
        """
        new_payload = copy.deepcopy(self.payload)
        new_payload.update(payload_updates)
        return OrchaPacket(
            parent_id=self.id,
            kind=kind,
            query=self.query,
            payload=new_payload,
            budget=self.budget.model_copy(deep=True),
            trace=list(self.trace),
            metadata=dict(self.metadata),
            tags=list(self.tags),
        )

    def error_packet(self, message: str, **extra: Any) -> "OrchaPacket":
        """Convenience: produce an ERROR-kind packet with an error message."""
        return self.fork(PacketKind.ERROR, error=message, **extra)

    def tag(self, *labels: str) -> "OrchaPacket":
        """Add free-form labels. Mutates in-place; returns self."""
        self.tags.extend(labels)
        return self

    # ── Read helpers ──────────────────────────────────────────────────

    def get_results(self) -> List[ExpertResult]:
        """Parse the results list from payload into typed ExpertResult objects."""
        raw = self.payload.get("results", [])
        return [ExpertResult(**r) if isinstance(r, dict) else r for r in raw]

    def get_subtasks(self) -> List[SubTask]:
        raw = self.payload.get("subtasks", [])
        return [SubTask(**t) if isinstance(t, dict) else t for t in raw]

    def get_selected_experts(self) -> List[ExpertSlot]:
        raw = self.payload.get("selected_experts", [])
        return [ExpertSlot(**e) if isinstance(e, dict) else e for e in raw]

    def successful_results(self) -> List[ExpertResult]:
        return [r for r in self.get_results() if not r.failed]

    def get_execution_plan(self) -> Optional[ExecutionPlan]:
        raw = self.payload.get("execution_plan")
        if raw is None:
            return None
        return ExecutionPlan(**raw) if isinstance(raw, dict) else raw

    def set_execution_plan(self, plan: ExecutionPlan) -> None:
        self.payload["execution_plan"] = plan.model_dump()

    # ── Serialisation ─────────────────────────────────────────────────

    def to_json(self, indent: int = 2) -> str:
        try:
            return self.model_dump_json(indent=indent)
        except Exception:
            # `payload` is a free-form dict, and a node occasionally stashes
            # a live, in-memory-only helper in it to survive across
            # node-to-node forks within one run (confirmed so far:
            # nodes/task_executor.py's EventStreamAdapter, and a bound
            # method) — never meant to be persisted, and never exhaustively
            # enumerable up front. Rather than allowlisting specific keys
            # one at a time as new ones turn up, drop whichever top-level
            # payload entries aren't plain JSON-shaped data and keep
            # everything else, so a checkpoint write can never hard-crash a
            # run over an internal helper value.
            sanitized = self.model_copy(deep=False)
            sanitized.payload = {
                k: v for k, v in self.payload.items() if _is_json_shaped(v)
            }
            return sanitized.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "OrchaPacket":
        return cls.model_validate_json(text)

    # ── Introspection ─────────────────────────────────────────────────

    def age_s(self) -> float:
        return time.time() - self.created_at

    def stage_duration(self, stage: str) -> Optional[float]:
        """Return the duration_ms for the first trace entry with the given stage."""
        for step in self.trace:
            if step.stage == stage:
                return step.duration_ms
        return None

    def explain(self) -> str:
        """Human-readable pipeline trace."""
        lines = [f"Packet {self.id[:8]}  kind={self.kind}  age={self.age_s():.2f}s"]
        lines.append(f"Query: {self.query[:120]}")
        lines.append("")
        for step in self.trace:
            lines.append(step.summary())
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"OrchaPacket(id={self.id[:8]}, kind={self.kind}, "
            f"query={self.query[:40]!r})"
        )


# ── Verification Layer Models ────────────────────────────────────────────────


class VerificationSignal(BaseModel):
    """A deterministic verification signal extracted from execution results."""
    signal_type: str           # "file_exists", "file_content", "command_exit_code",
                               # "tool_result", "artifact_exists", "schema_valid",
                               # "test_result", "build_result", "diagnostic"
    description: str = ""
    passed: bool = False
    expected: Optional[str] = None   # what was expected
    actual: Optional[str] = None     # what was actually observed
    source_step_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


class VerificationResult(BaseModel):
    """Verification result for a single task step."""
    step_id: str
    verified: bool = False
    signals: List[VerificationSignal] = []
    llm_evaluation: Optional[str] = None    # LLM verdict if deterministic is insufficient
    llm_confidence: Optional[float] = None  # 0.0-1.0
    issues: List[str] = []
    summary: str = ""
    deterministic_pass_rate: float = 0.0    # ratio of deterministic signals that passed
    used_llm_fallback: bool = False
    metadata: Dict[str, Any] = {}


class ObjectiveVerificationResult(BaseModel):
    """Final structured result of objective verification across all tasks."""
    status: str = "unverified"              # "verified", "partially_verified", "failed", "unverified"
    objective_satisfied: bool = False
    verified: bool = False
    objective: str = ""                     # original user objective
    task_results: List[VerificationResult] = []
    incomplete_items: List[str] = []        # task IDs or descriptions of what's incomplete
    unresolved_errors: List[str] = []       # errors that could not be resolved
    recovery_tasks: List["RecoveryTask"] = []  # tasks created to fix verification failures
    summary: str = ""
    artifacts_verified: List[str] = []      # artifact locations that were verified
    artifacts_missing: List[str] = []       # expected artifacts not found
    verification_duration_s: float = 0.0
    metadata: Dict[str, Any] = {}


class RecoveryTask(BaseModel):
    """A recovery task created when verification fails."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    objective: str = ""
    execution_instructions: str = ""
    verification_criteria: str = ""
    recovery_for: str = ""                  # step_id that failed verification
    reason: str = ""
    priority: int = 0                       # lower = higher priority
    dependencies: List[str] = []
    metadata: Dict[str, Any] = {}


class FinalResponse(BaseModel):
    """User-facing response generated from verified execution state."""
    status: str = "unverified"              # "success", "partial", "failure"
    objective: str = ""
    completed_items: List[str] = []         # what was completed
    changed_items: List[str] = []           # what was changed (files, etc.)
    verified_items: List[str] = []          # what was verified with evidence
    failed_items: List[str] = []            # failures with reasons
    limitations: List[str] = []             # known limitations
    verification_summary: str = ""
    errors: List[str] = []
    summary: str = ""
    metadata: Dict[str, Any] = {}


# ── Execution Event Stream Models ────────────────────────────────────────────


class ExecutionEventKind(str, Enum):
    """Structured event kinds for the observable execution stream."""
    # Run lifecycle
    RUN_CREATED        = "run_created"
    RUN_COMPLETED      = "run_completed"
    RUN_FAILED         = "run_failed"
    RUN_CANCELLED      = "run_cancelled"
    RUN_RESUMED        = "run_resumed"

    # Intent & planning
    INTENT_ANALYZED    = "intent_analyzed"
    COMPLEXITY_DETERMINED = "complexity_determined"
    PLANNING_STARTED   = "planning_started"
    PLAN_CREATED       = "plan_created"
    PLAN_UPDATED       = "plan_updated"

    # Task lifecycle
    TASK_READY         = "task_ready"
    TASK_STARTED       = "task_started"
    TASK_COMPLETED     = "task_completed"
    TASK_FAILED        = "task_failed"
    TASK_OBSERVED      = "task_observed"

    # Model & tool execution
    MODEL_STARTED      = "model_started"
    TOOL_STARTED       = "tool_started"
    TOOL_COMPLETED     = "tool_completed"

    # Retry & replanning
    RETRY_STARTED      = "retry_started"
    REPLANNING_STARTED = "replanning_started"

    # Verification
    VERIFICATION_STARTED   = "verification_started"
    VERIFICATION_COMPLETED = "verification_completed"


class ExecutionEvent(BaseModel):
    """
    A structured, frontend-safe event for the observable execution stream.

    This event is safe to send to the frontend — it contains no internal secrets,
    private prompts, API keys, or unsafe internal data.

    Events are strictly ordered by (run_id, seq). The sequence number is
    assigned by the RunStateTracker at emit time.
    """
    run_id: str
    seq: int = 0
    kind: ExecutionEventKind
    ts: float = Field(default_factory=time.time)

    # Human-readable summary for frontend timeline
    summary: str = ""

    # Task context (populated for task-related events)
    task_id: Optional[str] = None
    task_title: Optional[str] = None
    task_status: Optional[str] = None     # "pending", "in_progress", "completed", "failed", "skipped"

    # Plan context (populated for plan-related events)
    plan_progress: Optional[float] = None  # 0.0-1.0
    plan_step_count: Optional[int] = None
    plan_completed_count: Optional[int] = None

    # Verification context
    verification_status: Optional[str] = None  # "verified", "partially_verified", "failed"

    # Error context (safe error messages only)
    error: Optional[str] = None

    # Additional safe metadata for frontend rendering
    metadata: Dict[str, Any] = {}


class RunStateSnapshot(BaseModel):
    """
    Complete, serializable snapshot of run state for reconstruction after reconnect.

    The frontend can call get-run to fetch this and reconstruct the timeline.
    Backend can also persist this for resume support.
    """
    run_id: str
    status: str = "created"            # created, running, completed, failed, cancelled
    objective: str = ""
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    # Current plan state (if applicable)
    plan: Optional[ExecutionPlan] = None

    # Task-level state
    current_task_id: Optional[str] = None
    current_task_title: Optional[str] = None
    completed_task_ids: List[str] = []
    failed_task_ids: List[str] = []

    # Verification state
    verification_result: Optional[ObjectiveVerificationResult] = None
    final_response: Optional[FinalResponse] = None

    # Event history (compact, frontend-safe)
    events: List[ExecutionEvent] = []

    # Safe metadata for frontend
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}
