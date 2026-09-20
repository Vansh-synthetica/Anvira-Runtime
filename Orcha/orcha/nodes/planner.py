"""
orcha.nodes.planner
===================
Complex task orchestration components.

This module implements the Intent and Complexity Analysis Layer for complex
task orchestration. It contains:

- ComplexityGate: decides whether a query is simple (single agent call) or
  complex (needs decomposition, execution, observation, replanning).
- TaskPlanner: decomposes complex queries into structured execution plans
  with dependency tracking, verification criteria, and strict validation.
- TaskContextBuilder: builds the context for each step from the execution plan.
- TaskSummaryMemory: maintains a rolling summary of completed work for
  replanning and final verification.
- OrchaTaskPlanner: the core planner that receives structured input and
  produces a minimal executable task graph with strict validation.
- TaskPlanBuilder: helper for constructing validated task graphs.

Design principles
-----------------
- AgentNode stays the tool-using agent (no changes to agent.py core logic).
- GraphRuntime/LangGraph stays durable execution (checkpoint, resume).
- State lives in OrchaPacket.payload["execution_plan"] (no new DBs).
- Simple requests remain unchanged (zero overhead).
- Existing tools/ToolRegistry reused.
"""
from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..core.packets import (
    ExecutionPlan,
    OrchaPacket,
    PacketKind,
    PlannerRequest,
    TaskExecutionResult,
    TaskExecutionState,
    TaskObservation,
    TaskResult,
    TaskStep,
)
from ..graph.context import (
    EVT_NODE_END,
    EVT_NODE_START,
    EVT_PLAN_CREATED,
    EVT_PLAN_UPDATED,
    EVT_REPLAN,
    EVT_TASK_COMPLETE,
    EVT_TASK_START,
    EVT_VERIFY,
    RunContext,
)
from ..graph.node import Node


# ── Task type classification ─────────────────────────────────────────────────

class TaskType(str, Enum):
    """Classification of task types in a plan graph."""
    INSPECT     = "inspect"      # read/examine existing state
    IDENTIFY    = "identify"     # analyze and determine patterns/framework
    UNDERSTAND  = "understand"   # clarify requirements/spec
    CREATE      = "create"       # create new files/code
    MODIFY      = "modify"       # modify existing files/code
    EXECUTE     = "execute"      # run commands/tests
    VERIFY      = "verify"       # validate results
    DIAGNOSE    = "diagnose"     # analyze failures
    RECOVER     = "recover"      # fix failures
    THINK       = "think"        # reasoning/planning step


# ── Task plan builder ────────────────────────────────────────────────────────

class PlanValidationError(Exception):
    """Raised when a plan fails validation."""
    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__(f"Plan validation failed: {'; '.join(errors)}")


class TaskPlanBuilder:
    """
    Helper for constructing validated task graphs.

    Provides a fluent API for building plans with proper dependency tracking.
    Validates the graph structure at build time.
    """

    def __init__(self, objective: str) -> None:
        self._objective = objective
        self._steps: List[TaskStep] = []
        self._ids: Set[str] = set()

    def add_step(
        self,
        task_id: str,
        title: str,
        objective: str,
        *,
        dependencies: Optional[List[str]] = None,
        execution_instructions: str = "",
        expected_outcome: str = "",
        required_context: Optional[List[str]] = None,
        likely_tools: Optional[List[str]] = None,
        verification_criteria: str = "",
        action: str = "tool_use",
    ) -> "TaskPlanBuilder":
        """Add a step to the plan. Raises ValueError on duplicate ID."""
        if task_id in self._ids:
            raise ValueError(f"Duplicate task ID: {task_id}")
        self._ids.add(task_id)
        step = TaskStep(
            id=task_id,
            title=title,
            objective=objective,
            execution_instructions=execution_instructions,
            dependencies=dependencies or [],
            expected_outcome=expected_outcome,
            required_context=required_context or [],
            likely_tools=likely_tools or [],
            verification_criteria=verification_criteria,
            action=action,
        )
        self._steps.append(step)
        return self

    def build(self) -> ExecutionPlan:
        """Build and validate the execution plan."""
        plan = ExecutionPlan(
            objective=self._objective,
            steps=self._steps,
            status="pending",
        )
        errors = validate_plan(plan)
        if errors:
            raise PlanValidationError(errors)
        return plan

    def get_step_ids(self) -> List[str]:
        """Return all step IDs in order."""
        return [s.id for s in self._steps]


# ── Complexity scoring ────────────────────────────────────────────────────────

# Keywords indicating multi-step or complex requests
_COMPLEXITY_SIGNALS = {
    "high": [
        re.compile(r'\b(and\s+then|after\s+that|once\s+(?:you|that)|followed\s+by)\b', re.I),
        re.compile(r'\b(all\s+files|every\s+file|all\s+folders|recursively)\b', re.I),
        re.compile(r'\b(compare|contrast|analyze|summarize|review|audit)\s+.*\b(and|plus|also|with)\b', re.I),
        re.compile(r'\b(step\s+by\s+step|sequential|in\s+order)\b', re.I),
        # Whole-app/whole-project build specs ("Build a Snake Game with a
        # clean GUI...", "Create a REST API for..."). Distinct shape from
        # the above: not a chained instruction, but a single build request
        # that implies many files/steps once actually attempted.
        re.compile(
            r'\b(build|create|make|develop|implement)\s+(a|an)\s+'
            r'[\w\s,.\'"-]{0,80}?'
            r'\b(app|application|game|website|dashboard|api|service|'
            r'server|backend|frontend)\b',
            re.I,
        ),
    ],
    "medium": [
        re.compile(r'\b(find\s+and\s+(?:edit|update|fix|change|modify))\b', re.I),
        re.compile(r'\b(search\s+and\s+(?:replace|rename|move))\b', re.I),
        re.compile(r'\b(read\s+.*\s+and\s+(?:write|create|summarize))\b', re.I),
        re.compile(r'\b(extract\s+.*\s+from\s+.*\s+and\s+(?:save|write|create))\b', re.I),
    ],
}

# A long bulleted/numbered requirements list ("Gameplay\n* Snake grows...\n*
# Collision detection...") is itself a strong complexity signal regardless
# of which words appear in it — it's a feature spec, not a single request.
_BULLET_LINE_RE = re.compile(r'^\s*[-*•]\s+\S', re.M)
_MIN_BULLETS_FOR_COMPLEX = 4

# Indicators of a simple, single-step request
_SIMPLE_SIGNALS = [
    re.compile(r'^(?:what|how|why|when|where|who|which|can|could|should|would|is|are|does|do|did|will|shall|may)\b', re.I),
    re.compile(r'\b(explain|describe|tell\s+me|show\s+me|list|read|open|show)\b', re.I),
]

# Explicit sequencing ("... Then ... Then ...") is a multi-step request by
# construction, even when no individual clause trips a signal above.
# Measured gap: "Create stats.py ... Then create test_stats.py ... Then run
# it and tell me the output" scored only 0.2 against a 0.5 threshold, so it
# was handed to the single-shot agent — which then wrote one of the two
# files and never ran anything. Each additional step past the first adds
# weight rather than a flat bump, so a two-step request stays cheap while a
# genuinely long chain escalates.
_SEQUENCE_STEP_RE = re.compile(
    r'(?:^|[.;,\n]|\band\b)\s*(?:then|after\s+that|next|finally|afterwards)\b',
    re.I,
)
# Weighted so sequencing ALONE does not cross the 0.5 default threshold:
# three chained steps reach 0.45, enough to make a genuinely multi-step
# request outrank a flat one and to tip it over when combined with any
# other signal, without a single "do X then Y then Z" phrasing unilaterally
# diverting traffic into the decomposition pipeline. That restraint is
# deliberate while the pipeline's execution stall is unfixed — see
# ENABLE_SMALL_MODEL_DECOMPOSITION in orcha/api/server.py.
_PER_STEP_WEIGHT = 0.15
_MAX_SEQUENCE_BONUS = 0.45


@dataclass
class ComplexityGateConfig:
    """Configuration for the complexity gate."""
    completion_fn: Optional[Callable] = None
    system_prompt: str = (
        "You are a task complexity analyzer. Analyze the user's request and "
        "determine if it is SIMPLE (single action, single file, direct answer) "
        "or COMPLEX (multi-step, multi-file, requires planning). "
        "Reply with exactly one line: SIMPLE or COMPLEX."
    )
    seed_messages: List[Dict[str, Any]] = field(default_factory=list)
    # Heuristic threshold: if heuristic score >= this, treat as complex
    # even without a model call.
    #
    # This is deliberately model-size dependent (set by the API layer from
    # the active model's name — see SMALL_MODEL_COMPLEXITY_THRESHOLD).
    # "Complex" is not a property of the request alone: it is whether THIS
    # model can hold the whole thing in one loop. A request a 70B one-shots
    # comfortably is exactly the request a 3B drops half of — measured, not
    # assumed: a two-file + run + report task scored 0.2 here, went to the
    # single-shot agent, and the 3B wrote one file and never ran it.
    heuristic_threshold: float = 0.5


def _heuristic_complexity_score(query: str) -> float:
    """Fast heuristic complexity score (0.0 = simple, 1.0 = complex)."""
    text = (query or "").strip()
    if not text:
        return 0.0
    score = 0.0
    # High-complexity signals (strong indicators of multi-step work)
    for pattern in _COMPLEXITY_SIGNALS["high"]:
        if pattern.search(text):
            score += 0.5
    # Medium-complexity signals
    for pattern in _COMPLEXITY_SIGNALS["medium"]:
        if pattern.search(text):
            score += 0.3
    # A long requirements list is complex on its own, independent of the
    # word-based signals above (a feature-by-feature spec rarely phrases
    # itself as "do X and then Y").
    if len(_BULLET_LINE_RE.findall(text)) >= _MIN_BULLETS_FOR_COMPLEX:
        score += 0.5
    # Explicit sequencing — see _SEQUENCE_STEP_RE. Scored BEFORE the
    # simple-signal penalty below, because that penalty is gated on
    # `score == 0.0` precisely so it only ever applies to requests nothing
    # else marked complex. Scoring sequencing afterwards defeated that
    # guard: "Create stats.py. Then create test_stats.py. Then run it and
    # show output." was penalised -0.3 for containing "show" and landed at
    # 0.10 instead of 0.40, i.e. back on the single-shot path despite
    # being explicitly three steps.
    steps = len(_SEQUENCE_STEP_RE.findall(text))
    if steps:
        score += min(steps * _PER_STEP_WEIGHT, _MAX_SEQUENCE_BONUS)
    # Simple signals reduce complexity — but only for genuinely short
    # messages. These patterns match common bare words ("show", "list",
    # "read", "open") that legitimately signal a simple one-line request
    # ("show me config.json") but are equally likely to just appear once,
    # incidentally, inside an otherwise long and clearly complex spec (e.g.
    # a feature list bullet like "Show current score and level"). Gating on
    # length keeps the penalty for what it was designed for.
    if score == 0.0 and len(text) <= 200:
        for pattern in _SIMPLE_SIGNALS:
            if pattern.search(text):
                score -= 0.3
    # Length-based: very long queries are often complex
    if len(text) > 300:
        score += 0.2
    elif len(text) > 150:
        score += 0.1
    return max(0.0, min(1.0, score))


class ComplexityGateNode(Node):
    """
    Decides whether a query is simple or complex.

    Writes ``complexity_gate`` ("simple" or "complex") and
    ``complexity_score`` (float) into the packet payload.
    """

    name = "complexity_gate"

    def __init__(
        self,
        name: str = "complexity_gate",
        config: Optional[ComplexityGateConfig] = None,
        timeout_s: Optional[float] = 120.0,
        retries: int = 0,
    ) -> None:
        self.name = name
        self.timeout_s = timeout_s
        self.retries = retries
        self.config = config or ComplexityGateConfig()

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        query = packet.payload.get("task", packet.query)
        # A request that names its files is built file by file by the agent node (agent_runtime/scaffold.py); the general
        # planner over-decomposes those on small models (17 overlapping steps, stub files), so do not send it there.
        from ..agent_runtime.scaffold import plan_files
        if plan_files(query):
            return packet.fork(packet.kind, complexity_gate="simple", complexity_score=0.0, scaffold=True)
        # Fast path: heuristic scoring (no model call)
        score = _heuristic_complexity_score(query)
        if score >= self.config.heuristic_threshold:
            return packet.fork(
                packet.kind,
                complexity_gate="complex",
                complexity_score=score,
            )
        if score < self.config.heuristic_threshold * 0.5:
            # Very low heuristic score — definitely simple
            return packet.fork(
                packet.kind,
                complexity_gate="simple",
                complexity_score=score,
            )
        # Ambiguous score — ask the model
        if self.config.completion_fn is None:
            # No model available, default to simple
            return packet.fork(
                packet.kind,
                complexity_gate="simple",
                complexity_score=score,
            )
        messages = list(self.config.seed_messages or [])
        messages.append({"role": "user", "content": query})
        try:
            message = await self.config.completion_fn(
                messages, self.config.system_prompt, None
            )
            content = (message.get("content") or "").strip().upper()
            if "COMPLEX" in content:
                return packet.fork(
                    packet.kind,
                    complexity_gate="complex",
                    complexity_score=max(score, self.config.heuristic_threshold),
                )
        except Exception:
            pass
        return packet.fork(
            packet.kind,
            complexity_gate="simple",
            complexity_score=score,
        )


# ── Task planner ──────────────────────────────────────────────────────────────

_TASK_PLANNER_PROMPT = """You are a task planner. Given a user request and its context, produce a minimal executable task graph.

Do NOT blindly create a long checklist. Produce ONLY the tasks required to achieve the objective.

Each task must be a JSON object with these fields:
- "id": stable unique string identifier (e.g. "inspect_repo", "identify_framework")
- "title": concise 3-8 word title
- "objective": what this task accomplishes
- "execution_instructions": step-by-step instructions for the agent executing this task
- "dependencies": list of task IDs that must complete before this task starts
- "expected_outcome": what success looks like
- "required_context": list of task IDs whose output this task needs as input
- "likely_tools": list of tool names likely needed (e.g. "read_file", "list_directory", "search_code", "execute_command")
- "verification_criteria": how to verify this task succeeded
- "action": one of "tool_use", "search", "create", "modify", "execute", "think"

IMPORTANT: when an "Available tools" list is provided below, "likely_tools" MUST contain ONLY names copied exactly from that list — never invent a plausible-sounding tool name (the example names above, like "search_code" or "execute_command", are illustrative only and may not be real). The agent executing each task can only call tools that actually exist; a made-up name wastes the task's first attempt and can derail the whole step. If you are unsure which real tool applies, leave "likely_tools" empty rather than guessing a name.

File rules:
- A file you are CREATING must be written COMPLETE in ONE task, with all of its
  functions, classes and imports in that single write. NEVER create a file in one
  task and add to it in later tasks: editing requires matching the file's exact
  existing text, which fails far more often than writing the finished file once.
- Only use a "modify" task for a file that already existed BEFORE this plan started.
- State the file path in execution_instructions, and reuse the SAME path and
  directory across every task that refers to that file.

Dependency rules:
- Task A depends on Task B means A cannot start until B completes.
- Dependencies form a DAG (no circular dependencies).
- Independent tasks can run in parallel (no dependencies between them).
- Recovery tasks depend on the failed task and may also depend on diagnostic tasks.
- Each task MUST have a non-empty "dependencies" list unless it has no prerequisites.

Output ONLY a JSON array of task objects. No explanation, no markdown, no extra text.

Example for "inspect this project, add a feature, run tests, fix failures":

[
  {
    "id": "inspect_repo",
    "title": "Inspect repository structure",
    "objective": "Understand the project layout and existing code",
    "execution_instructions": "List all files, read key configuration files, identify the main entry points",
    "dependencies": [],
    "expected_outcome": "Clear understanding of project structure, tech stack, and architecture",
    "required_context": [],
    "likely_tools": ["list_directory", "read_file", "search_code"],
    "verification_criteria": "Can describe the project structure, tech stack, and main entry points",
    "action": "search"
  },
  {
    "id": "identify_framework",
    "title": "Identify framework and patterns",
    "objective": "Determine the framework, patterns, and conventions used",
    "execution_instructions": "Analyze package.json/requirements.txt, read config files, examine existing code patterns",
    "dependencies": ["inspect_repo"],
    "expected_outcome": "Documented framework choice, coding patterns, and conventions",
    "required_context": ["inspect_repo"],
    "likely_tools": ["read_file", "search_code"],
    "verification_criteria": "Can list the framework, testing approach, and coding conventions",
    "action": "tool_use"
  }
]"""

_STEP_RE = re.compile(r'"id"\s*:\s*"([^"]+)"')
_TITLE_RE = re.compile(r'"title"\s*:\s*"([^"]+)"')
_OBJECTIVE_RE = re.compile(r'"objective"\s*:\s*"([^"]+)"')
_ACTION_RE = re.compile(r'"action"\s*:\s*"([^"]+)"')
_TOOLS_RE = re.compile(r'"likely_tools"\s*:\s*\[([^\]]*)\]')
_DEPS_RE = re.compile(r'"dependencies"\s*:\s*\[([^\]]*)\]')
_EXEC_RE = re.compile(r'"execution_instructions"\s*:\s*"([^"]+)"')
_EXPECTED_RE = re.compile(r'"expected_outcome"\s*:\s*"([^"]+)"')
_REQUIRED_CTX_RE = re.compile(r'"required_context"\s*:\s*\[([^\]]*)\]')
_VERIFICATION_RE = re.compile(r'"verification_criteria"\s*:\s*"([^"]+)"')


def _extract_json_array(content: str) -> Optional[list]:
    """Extract a JSON array from model output, handling markdown fences."""
    text = content.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines if they're fence markers
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            return arr
    except Exception:
        pass
    # Try to find the array in the text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            arr = json.loads(text[start:end + 1])
            if isinstance(arr, list):
                return arr
        except Exception:
            pass
    return None


def merge_create_then_modify(steps: List[TaskStep]) -> List[TaskStep]:
    """Fold "modify X" steps back into the step that CREATES X.

    A plan that creates a file in one task and then fills it in over
    several later tasks is a plan no small model can execute. Filling in
    means ``edit_file``, ``edit_file`` demands an ``old_string`` that
    matches the file byte-for-byte, and a 3B model cannot reliably produce
    one — it has not read the file, and it guesses. Measured on the live
    task: a 9-step plan (create_stats_module -> add_mean_function ->
    add_median_function -> ...) left ``stats.py`` containing ``mean`` and
    never ``median``, because every add-a-function step failed on exactly
    that argument.

    Writing the whole file once has none of that failure surface, so a
    modify step whose only dependency is a create step is merged into it:
    its instructions are appended to the create step's, and every reference
    to the merged id is repointed at the create step.

    Deliberately conservative. Only a modify step that depends on exactly
    one step, that step being a create, is merged — a modify that also
    depends on an inspection or a test run is doing something this rule
    knows nothing about, and is left alone.
    """
    by_id = {s.id: s for s in steps}
    merged_into: Dict[str, str] = {}

    for step in steps:
        if step.action != "modify" or len(step.dependencies) != 1:
            continue
        target_id = step.dependencies[0]
        # Follow an earlier merge so a create -> modify -> modify chain
        # collapses into the single create.
        while target_id in merged_into:
            target_id = merged_into[target_id]
        target = by_id.get(target_id)
        if target is None or target.action != "create":
            continue

        detail = " ".join(
            part for part in (step.execution_instructions, step.objective) if part
        ).strip()
        if detail:
            target.execution_instructions = (
                f"{target.execution_instructions.rstrip()} "
                f"Also, in the same file and the same write: {detail}"
            ).strip()
        if step.expected_outcome:
            target.expected_outcome = (
                f"{target.expected_outcome.rstrip()} "
                f"{step.expected_outcome}"
            ).strip()
        if step.verification_criteria:
            target.verification_criteria = (
                f"{target.verification_criteria.rstrip()} "
                f"{step.verification_criteria}"
            ).strip()
        merged_into[step.id] = target_id

    if not merged_into:
        return steps

    def repoint(ids: List[str]) -> List[str]:
        out: List[str] = []
        for i in ids:
            while i in merged_into:
                i = merged_into[i]
            if i not in out:
                out.append(i)
        return out

    kept = [s for s in steps if s.id not in merged_into]
    for step in kept:
        step.dependencies = [d for d in repoint(step.dependencies) if d != step.id]
        step.required_context = [
            c for c in repoint(step.required_context) if c != step.id
        ]
    return kept


def _parse_tasks_from_model(content: str) -> List[TaskStep]:
    """Parse task objects from model output with robust fallbacks."""
    arr = _extract_json_array(content)
    if arr is not None:
        steps: List[TaskStep] = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            task_id = (item.get("id") or "").strip()
            title = (item.get("title") or "").strip()
            objective = (item.get("objective") or "").strip()
            if not task_id or not title or not objective:
                continue
            steps.append(TaskStep(
                id=task_id,
                title=title,
                objective=objective,
                execution_instructions=item.get("execution_instructions", ""),
                dependencies=[d.strip() for d in (item.get("dependencies") or []) if isinstance(d, str)],
                expected_outcome=item.get("expected_outcome", ""),
                required_context=[c.strip() for c in (item.get("required_context") or []) if isinstance(c, str)],
                likely_tools=[t.strip() for t in (item.get("likely_tools") or []) if isinstance(t, str)],
                verification_criteria=item.get("verification_criteria", ""),
                action=item.get("action", "tool_use"),
            ))
        if steps:
            return merge_create_then_modify(steps)
    # Regex fallback for weak models
    steps = []
    ids = _STEP_RE.findall(content)
    titles = _TITLE_RE.findall(content)
    objectives = _OBJECTIVE_RE.findall(content)
    actions = _ACTION_RE.findall(content)
    exec_ins = _EXEC_RE.findall(content)
    expecteds = _EXPECTED_RE.findall(content)
    verifications = _VERIFICATION_RE.findall(content)
    all_deps = _DEPS_RE.findall(content)
    all_tools = _TOOLS_RE.findall(content)
    all_ctx = _REQUIRED_CTX_RE.findall(content)
    for i, task_id in enumerate(ids):
        if not task_id.strip():
            continue
        deps = []
        if i < len(all_deps):
            deps = [d.strip().strip('"') for d in all_deps[i].split(",") if d.strip().strip('"')]
        tools = []
        if i < len(all_tools):
            tools = [t.strip().strip('"') for t in all_tools[i].split(",") if t.strip().strip('"')]
        ctx = []
        if i < len(all_ctx):
            ctx = [c.strip().strip('"') for c in all_ctx[i].split(",") if c.strip().strip('"')]
        steps.append(TaskStep(
            id=task_id,
            title=titles[i] if i < len(titles) else f"Task {i + 1}",
            objective=objectives[i] if i < len(objectives) else task_id,
            execution_instructions=exec_ins[i] if i < len(exec_ins) else "",
            dependencies=deps,
            expected_outcome=expecteds[i] if i < len(expecteds) else "",
            required_context=ctx,
            likely_tools=tools,
            verification_criteria=verifications[i] if i < len(verifications) else "",
            action=actions[i] if i < len(actions) else "tool_use",
        ))
    return merge_create_then_modify(steps)


# ── Plan validation ──────────────────────────────────────────────────────────

def validate_plan(plan: ExecutionPlan) -> List[str]:
    """
    Validate an execution plan. Returns a list of error messages.
    An empty list means the plan is valid.

    Checks:
    1. No empty task list
    2. No duplicate task IDs
    3. No empty/meaningless objectives
    4. No circular dependencies
    5. All dependency references point to existing tasks
    6. No self-dependencies
    """
    errors: List[str] = []
    if not plan.steps:
        errors.append("Plan has no tasks")
        return errors
    seen_ids: Set[str] = set()
    all_ids: Set[str] = {s.id for s in plan.steps}
    for step in plan.steps:
        # Duplicate IDs
        if step.id in seen_ids:
            errors.append(f"Duplicate task ID: {step.id}")
        seen_ids.add(step.id)
        # Empty/meaningless objective
        obj = (step.objective or "").strip()
        if not obj or len(obj) < 5:
            errors.append(f"Task {step.id} has an empty or too-short objective")
        # Empty title
        if not (step.title or "").strip():
            errors.append(f"Task {step.id} has an empty title")
        # Self-dependency
        if step.id in step.dependencies:
            errors.append(f"Task {step.id} depends on itself")
        # Non-existent dependencies
        for dep in step.dependencies:
            if dep not in all_ids:
                errors.append(f"Task {step.id} depends on non-existent task: {dep}")
    # Circular dependency detection via topological sort
    if not errors:
        cycle = _detect_cycle(plan.steps)
        if cycle:
            errors.append(f"Circular dependency detected: {' -> '.join(cycle)}")
    return errors


def _detect_cycle(steps: List[TaskStep]) -> Optional[List[str]]:
    """Detect circular dependencies using Kahn's algorithm. Returns cycle path or None."""
    adj: Dict[str, List[str]] = {s.id: [] for s in steps}
    in_degree: Dict[str, int] = {s.id: 0 for s in steps}
    for s in steps:
        for dep in s.dependencies:
            if dep in adj:
                adj[dep].append(s.id)
                in_degree[s.id] += 1
    queue: deque = deque()
    for node, deg in in_degree.items():
        if deg == 0:
            queue.append(node)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    if visited == len(steps):
        return None
    # Find a cycle for error reporting
    cycle_nodes = [nid for nid, deg in in_degree.items() if deg > 0]
    if cycle_nodes:
        # Trace one cycle
        start = cycle_nodes[0]
        path = [start]
        visited_cycle: Set[str] = set()
        current = start
        while current not in visited_cycle:
            visited_cycle.add(current)
            found_next = False
            for dep in steps:
                if dep.id == current:
                    for nxt in dep.dependencies:
                        if nxt in {n for n in cycle_nodes}:
                            path.append(nxt)
                            current = nxt
                            found_next = True
                            break
            if not found_next:
                break
        return path[:10]  # Limit path length
    return None


def build_planner_prompt(request: PlannerRequest) -> str:
    """Build the system prompt for the task planner from a PlannerRequest."""
    parts = [_TASK_PLANNER_PROMPT]
    parts.append(f"\nOriginal request: {request.original_request}")
    parts.append(f"Objective: {request.normalized_objective}")
    if request.constraints:
        parts.append(f"Constraints: {'; '.join(request.constraints)}")
    if request.context:
        parts.append(f"\nExisting context:\n{request.context}")
    if request.tools:
        parts.append(f"\nAvailable tools: {', '.join(request.tools)}")
    if request.workspace_state:
        parts.append(f"\nWorkspace state:\n{request.workspace_state}")
    return "\n".join(parts)


class TaskPlannerNode(Node):
    """
    Decomposes a complex query into a validated ExecutionPlan.

    Receives a PlannerRequest in the packet payload and produces an
    ExecutionPlan with strict validation. Rejects plans with duplicate IDs,
    circular dependencies, impossible dependencies, empty tasks, or
    meaningless objectives.

    Writes ``execution_plan`` (ExecutionPlan) into the packet payload.
    """

    name = "task_planner"

    def __init__(
        self,
        name: str = "task_planner",
        completion_fn: Optional[Callable] = None,
        timeout_s: Optional[float] = 180.0,
        retries: int = 1,
        executor: Optional[Any] = None,  # ToolExecutor
    ) -> None:
        self.name = name
        self.timeout_s = timeout_s
        self.retries = retries
        self.completion_fn = completion_fn
        self.executor = executor

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        # Build PlannerRequest from packet payload
        request_raw = packet.payload.get("planner_request")
        if request_raw is not None:
            request = PlannerRequest(**request_raw) if isinstance(request_raw, dict) else request_raw
        else:
            query = packet.payload.get("task", packet.query)
            request = PlannerRequest(
                original_request=query,
                normalized_objective=query,
                constraints=packet.payload.get("constraints", []),
                context=packet.payload.get("context", ""),
                tools=packet.payload.get("tools", []),
                workspace_state=packet.payload.get("workspace_state", ""),
            )
        # `request.tools` was previously always empty here — nothing upstream
        # ever populated it — which meant build_planner_prompt's "Available
        # tools: ..." line was a permanent no-op and the model planned
        # completely blind to what tools actually exist. It then filled
        # each step's `likely_tools` with plausible-sounding but fabricated
        # names (e.g. "write_code" instead of the real "write_file"), which
        # get quoted verbatim into the execution prompt by
        # TaskContextBuilderNode and mislead the executing model into never
        # calling a real tool at all. Ground the plan in reality instead.
        if not request.tools and self.executor is not None:
            try:
                request.tools = list(self.executor.names())
            except Exception:
                pass
        plan = ExecutionPlan(
            request=request,
            objective=request.normalized_objective,
            status="pending",
        )
        if self.completion_fn is None:
            plan.steps = [TaskStep(
                id="execute_directly",
                title="Execute request directly",
                objective=request.normalized_objective,
                execution_instructions=request.original_request,
                dependencies=[],
                expected_outcome="Request completed",
                required_context=[],
                likely_tools=[],
                verification_criteria="Request is fulfilled",
                action="tool_use",
            )]
            plan.status = "in_progress"
            return packet.fork(
                packet.kind,
                execution_plan=plan.model_dump(),
                plan_created=True,
            )
        prompt = build_planner_prompt(request)
        messages = [{"role": "user", "content": request.original_request}]
        best_plan: Optional[ExecutionPlan] = None
        last_error: Optional[str] = None
        for attempt in range(self.retries + 1):
            try:
                message = await self.completion_fn(
                    messages, prompt, None
                )
                content = (message.get("content") or "").strip()
                steps = _parse_tasks_from_model(content)
            except Exception as exc:
                last_error = str(exc)
                steps = []
            if not steps:
                last_error = "Model produced no parseable tasks"
                continue
            plan.steps = steps
            validation_errors = validate_plan(plan)
            if not validation_errors:
                best_plan = plan
                break
            last_error = f"Validation errors: {'; '.join(validation_errors)}"
        if best_plan is None:
            # Fallback: single-step plan
            plan.steps = [TaskStep(
                id="execute_directly",
                title="Execute request directly",
                objective=request.normalized_objective,
                execution_instructions=request.original_request,
                dependencies=[],
                expected_outcome="Request completed",
                required_context=[],
                likely_tools=[],
                verification_criteria="Request is fulfilled",
                action="tool_use",
            )]
        best_plan = best_plan or plan
        best_plan.status = "in_progress"
        return packet.fork(
            packet.kind,
            execution_plan=best_plan.model_dump(),
            plan_created=True,
            plan_validation_errors=last_error if last_error else None,
        )


# ── Task context builder ──────────────────────────────────────────────────────

class TaskContextBuilderNode(Node):
    """
    Builds the context for each step from the execution plan.

    Reads execution_plan from the packet, selects the next pending step,
    and constructs a focused context (instructions, available tools, prior
    step results) for the agent to execute.
    """

    name = "task_context_builder"

    def __init__(
        self,
        name: str = "task_context_builder",
        timeout_s: Optional[float] = 30.0,
        retries: int = 0,
    ) -> None:
        self.name = name
        self.timeout_s = timeout_s
        self.retries = retries

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        plan_raw = packet.payload.get("execution_plan")
        if plan_raw is None:
            return packet.fork(packet.kind)
        plan = ExecutionPlan(**plan_raw) if isinstance(plan_raw, dict) else plan_raw
        # Find next pending step
        step = plan.next_pending_step()
        if step is None:
            # No runnable step. That is only a completion when nothing is
            # left unfinished — otherwise it is a STALL: work remains but
            # the graph cannot make progress (a dependent of a skipped or
            # in_progress step, a cycle, a dependency the replanner
            # renamed). Reporting a stall as "completed" is how a run ends
            # after two of four steps while cheerfully claiming success,
            # and it is unrecoverable because finalize is a terminal node.
            # Route stalls at the replanner instead, which is bounded by
            # plan.max_replans and finalizes on its own once exhausted.
            unfinished = plan.unfinished_steps
            if unfinished:
                plan.status = "replanning"
                blocked = ", ".join(f"{s.id}({s.status})" for s in unfinished[:8])
                return packet.fork(
                    packet.kind,
                    execution_plan=plan.model_dump(),
                    task_action="replan",
                    task_blocked=True,
                    replan_reason=(
                        "Plan stalled: no runnable step remains but "
                        f"{len(unfinished)} step(s) are unfinished [{blocked}]"
                    ),
                )
            plan.status = "completed"
            return packet.fork(
                packet.kind,
                execution_plan=plan.model_dump(),
                task_complete=True,
                task_blocked=False,
            )
        step.status = "in_progress"
        # Build context for this step. The original user request (plan.objective,
        # set verbatim from the request in TaskPlannerNode) is included first and
        # always — not just the per-step objective/execution_instructions the
        # planner wrote. Those per-step fields come from the PLANNING model's own
        # summary of the step, and a weaker/smaller model compresses them far
        # enough that concrete requirements (e.g. "canvas, arrow keys, collision
        # detection, score") get lost entirely. Confirmed live: a real run asking
        # for a Snake game decomposed into a single step titled "Create Snake game
        # HTML file" with no such detail, and the executing model correctly
        # replied "CANNOT COMPLETE: No specific instructions or code provided" —
        # it genuinely never saw them, because this method never showed the
        # original request to begin with. Keeping the step's own fields alongside
        # this still matters for genuinely multi-step plans, where a step is a
        # narrower slice of the overall objective, not the whole thing.
        context_parts = [
            f"=== ORIGINAL REQUEST ===\n{plan.objective}",
            f"=== TASK: {step.title} ===",
            f"ID: {step.id}",
            f"Objective: {step.objective}",
            f"Execution instructions: {step.execution_instructions}",
            f"Expected outcome: {step.expected_outcome}",
            f"Verification criteria: {step.verification_criteria}",
            f"Likely tools: {', '.join(step.likely_tools) if step.likely_tools else 'any available tool'}",
        ]
        # Add results from dependency steps
        if step.dependencies:
            context_parts.append("\n--- Prior step outputs ---")
            for dep_id in step.dependencies:
                dep_step = plan.step_by_id(dep_id)
                if dep_step and dep_step.result:
                    context_parts.append(f"  [{dep_id}] {dep_step.title}: {dep_step.result.output[:300]}")
        # Add required context
        if step.required_context:
            context_parts.append("\n--- Required context from prior steps ---")
            for ctx_id in step.required_context:
                ctx_step = plan.step_by_id(ctx_id)
                if ctx_step and ctx_step.result:
                    context_parts.append(f"  [{ctx_id}] {ctx_step.title}: {ctx_step.result.output[:300]}")
        # Files this run has actually written, by real path, taken from the
        # tool calls rather than from the model's narration of them.
        #
        # Without this, each step picks a directory on its own and they
        # disagree. Confirmed live on a 3B model: step 1 wrote src/stats.py,
        # step 2 wrote test_stats.py at the workspace root, and step 3's
        # "run the tests" then failed with ModuleNotFoundError: stats —
        # three steps that individually did the right thing and collectively
        # produced a broken tree, purely because step 2 was never told where
        # step 1 put its file.
        #
        # Drawn from EVERY step carrying a result, not just completed ones:
        # a step can write its file and still be marked failed (it ran out of
        # iterations afterwards), and that file is on disk regardless.
        from .task_executor import _files_written_so_far
        written = _files_written_so_far(plan, exclude_step_id=step.id)
        if written:
            context_parts.append("\n--- Files already written in this run ---")
            for path in written:
                context_parts.append(f"  {path}")
            context_parts.append(
                "Use these EXACT paths when referring to or importing from "
                "these files. Put new related files in the same directory "
                "unless the task says otherwise."
            )
        # Update plan
        plan.current_step_idx = plan.steps.index(step)
        return packet.fork(
            packet.kind,
            execution_plan=plan.model_dump(),
            task_step_context="\n".join(context_parts),
            task_step_id=step.id,
            # fork() inherits the parent payload wholesale, so this flag
            # has to be cleared explicitly on the path that DID find
            # work - a stall recorded on an earlier pass would otherwise
            # stay true forever and keep routing to the replanner even
            # though a runnable step exists.
            task_blocked=False,
        )


# ── Task summary memory ───────────────────────────────────────────────────────

class TaskSummaryMemory:
    """Maintains a rolling summary of completed work for replanning."""

    def __init__(self, max_summary_tokens: int = 1000) -> None:
        self.max_summary_tokens = max_summary_tokens
        self.entries: List[Dict[str, Any]] = []

    def add(self, step: TaskStep, result: TaskResult) -> None:
        self.entries.append({
            "step_id": step.id,
            "title": step.title,
            "objective": step.objective,
            "output": result.output[:500],
            "success": result.success,
            "error": result.error,
            "timestamp": time.time(),
        })

    def build_summary(self) -> str:
        if not self.entries:
            return "No steps completed yet."
        lines = ["Execution summary:"]
        for entry in self.entries:
            status = "✓" if entry["success"] else "✗"
            lines.append(f"{status} {entry['step_id']}: {entry['title']}")
            lines.append(f"  Objective: {entry['objective']}")
            if entry.get("error"):
                lines.append(f"  Error: {entry['error']}")
            elif entry.get("output"):
                lines.append(f"  Output: {entry['output'][:150]}...")
        return "\n".join(lines)

    def get_context_for_replan(self) -> str:
        """Build context for replanning decisions."""
        summary = self.build_summary()
        failed = [e for e in self.entries if not e["success"]]
        if failed:
            summary += f"\n\nFailed steps: {', '.join(e['step_id'] for e in failed)}"
        return summary


# ── Observer node ──────────────────────────────────────────────────────────────

class ObserverNode(Node):
    """
    Observes step results and decides whether the objective is met.

    Writes ``observations`` (list of TaskObservation) and
    ``objective_met`` (bool) into the packet payload.
    """

    name = "observer"

    def __init__(
        self,
        name: str = "observer",
        completion_fn: Optional[Callable] = None,
        timeout_s: Optional[float] = 120.0,
        retries: int = 0,
    ) -> None:
        self.name = name
        self.timeout_s = timeout_s
        self.retries = retries
        self.completion_fn = completion_fn

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        plan_raw = packet.payload.get("execution_plan")
        if plan_raw is None:
            return packet.fork(packet.kind, objective_met=True)
        plan = ExecutionPlan(**plan_raw) if isinstance(plan_raw, dict) else plan_raw
        # Find the step that just completed
        for step in plan.steps:
            if step.status == "in_progress" and step.result:
                # Create observation
                obs = TaskObservation(
                    step_id=step.id,
                    step_result=step.result,
                    objective_met=None,
                )
                # Simple heuristic: if the step succeeded, objective might be met
                if step.result.success:
                    obs.objective_met = plan.is_complete
                else:
                    obs.objective_met = False
                    obs.issues.append(f"Step {step.id} failed: {step.result.error}")
                plan.observations.append(obs)
                step.status = "completed" if step.result.success else "failed"
                break
        return packet.fork(
            packet.kind,
            execution_plan=plan.model_dump(),
            objective_met=plan.is_complete,
        )


# ── Replanner node ────────────────────────────────────────────────────────────

class ReplannerNode(Node):
    """
    Decides whether to replan when a step fails.

    Writes ``replan_needed`` (bool) and optionally updates the execution plan.
    """

    name = "replanner"

    def __init__(
        self,
        name: str = "replanner",
        completion_fn: Optional[Callable] = None,
        timeout_s: Optional[float] = 120.0,
        retries: int = 0,
    ) -> None:
        self.name = name
        self.timeout_s = timeout_s
        self.retries = retries
        self.completion_fn = completion_fn

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        plan_raw = packet.payload.get("execution_plan")
        if plan_raw is None:
            return packet.fork(packet.kind, replan_needed=False)
        plan = ExecutionPlan(**plan_raw) if isinstance(plan_raw, dict) else plan_raw
        # Check if replanning is needed
        if plan.replan_count >= plan.max_replans:
            return packet.fork(packet.kind, replan_needed=False)
        failed = plan.failed_steps
        if not failed:
            return packet.fork(packet.kind, replan_needed=False)
        # Mark failed steps as skipped, continue with remaining
        for step in failed:
            step.status = "skipped"
        plan.replan_count += 1
        plan.status = "replanning"
        # Update plan with skipped steps
        return packet.fork(
            packet.kind,
            execution_plan=plan.model_dump(),
            replan_needed=True,
            replan_count=plan.replan_count,
        )


# ── Verifier node ─────────────────────────────────────────────────────────────

class VerifierNode(Node):
    """
    Verifies that the objective is met after all steps complete.

    Writes ``verification_result`` (str) and ``objective_met`` (bool).
    """

    name = "verifier"

    def __init__(
        self,
        name: str = "verifier",
        completion_fn: Optional[Callable] = None,
        timeout_s: Optional[float] = 120.0,
        retries: int = 0,
    ) -> None:
        self.name = name
        self.timeout_s = timeout_s
        self.retries = retries
        self.completion_fn = completion_fn

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        plan_raw = packet.payload.get("execution_plan")
        if plan_raw is None:
            return packet.fork(
                packet.kind,
                verification_result="No execution plan found",
                objective_met=False,
            )
        plan = ExecutionPlan(**plan_raw) if isinstance(plan_raw, dict) else plan_raw
        if not plan.is_complete:
            return packet.fork(
                packet.kind,
                verification_result="Plan not complete",
                objective_met=False,
            )
        # Simple verification: all steps completed
        completed = len(plan.completed_steps)
        total = len(plan.steps)
        skipped = len([s for s in plan.steps if s.status == "skipped"])
        if completed + skipped == total:
            return packet.fork(
                packet.kind,
                verification_result=f"All {total} steps completed ({completed} successful, {skipped} skipped)",
                objective_met=True,
            )
        return packet.fork(
            packet.kind,
            verification_result=f"Only {completed}/{total} steps completed",
            objective_met=False,
        )


# ── Step executor node (delegates to TaskExecutionLoop) ──────────────────────

class StepExecutorNode(Node):
    """
    Executes a single step from the execution plan.

    This node delegates to the TaskExecutionLoop for actual execution,
    but provides a simpler interface for the graph topology.
    """

    name = "step_executor"

    def __init__(
        self,
        name: str = "step_executor",
        completion_fn: Optional[Callable] = None,
        executor: Optional[Any] = None,  # ToolExecutor
        timeout_s: Optional[float] = 300.0,
        max_task_iterations: int = 5,
        retries: int = 0,
    ) -> None:
        self.name = name
        self.timeout_s = timeout_s
        self.retries = retries
        self.completion_fn = completion_fn
        self.executor = executor
        self.max_task_iterations = max_task_iterations

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        from .task_executor import TaskExecutionLoop

        # Create and run the execution loop
        loop = TaskExecutionLoop(
            name="task_execution_loop",
            completion_fn=self.completion_fn,
            executor=self.executor,
            timeout_s=self.timeout_s,
            max_task_iterations=self.max_task_iterations,
        )
        return await loop.run(packet, ctx)


# ── Orcha Task Planner ────────────────────────────────────────────────────────
#
# The core planner that receives structured input and produces a minimal
# executable task graph. It works both with and without a model:
#
# - With a model: delegates to the model for plan generation, then validates.
# - Without a model: uses heuristic analysis to build a structured plan.

# Keywords indicating different task phases
_INSPECT_KEYWORDS = re.compile(
    r'\b(inspect|examine|review|audit|look\s+at|check\s+out|explore|scan)\b', re.I
)
_IDENTIFY_KEYWORDS = re.compile(
    r'\b(identify|determine|find\s+out|discover|detect|recognize|figure\s+out)\b', re.I
)
_UNDERSTAND_KEYWORDS = re.compile(
    r'\b(understand|analyze|comprehend|grasp|learn|study)\b', re.I
)
_IMPLEMENT_KEYWORDS = re.compile(
    r'\b(implement|add|create|build|write|develop|introduce|insert|compose)\b', re.I
)
_MODIFY_KEYWORDS = re.compile(
    r'\b(modify|update|change|fix|repair|refactor|edit|adjust|patch)\b', re.I
)
_TEST_KEYWORDS = re.compile(
    r'\b(test|run\s+(?:the\s+)?tests?|execute\s+(?:the\s+)?tests?|lint|check)\b', re.I
)
_FIX_KEYWORDS = re.compile(
    r'\b(fix|repair|resolve|correct|debug|troubleshoot)\b', re.I
)
_VERIFY_KEYWORDS = re.compile(
    r'\b(verify|validate|confirm|ensure|final\s+check|final\s+verification)\b', re.I
)


def _classify_task_type(text: str) -> TaskType:
    """Classify a text fragment into a task type."""
    # Check more specific patterns first to avoid false positives
    if _FIX_KEYWORDS.search(text):
        return TaskType.RECOVER
    if _VERIFY_KEYWORDS.search(text):
        return TaskType.VERIFY
    if _TEST_KEYWORDS.search(text):
        return TaskType.EXECUTE
    if _INSPECT_KEYWORDS.search(text):
        return TaskType.INSPECT
    if _IDENTIFY_KEYWORDS.search(text):
        return TaskType.IDENTIFY
    if _UNDERSTAND_KEYWORDS.search(text):
        return TaskType.UNDERSTAND
    if _MODIFY_KEYWORDS.search(text):
        return TaskType.MODIFY
    if _IMPLEMENT_KEYWORDS.search(text):
        return TaskType.CREATE
    return TaskType.THINK


def _split_clauses(request: str) -> List[str]:
    """Split a request into logical clauses."""
    # Split on common connectors
    parts = re.split(
        r',\s*|\s+and\s+(?:then\s+)?|\s*;\s*|\s*followed\s+by\s+',
        request,
    )
    return [p.strip() for p in parts if p.strip()]


def _tools_for_task_type(task_type: TaskType) -> List[str]:
    """Return likely tools for a given task type."""
    mapping = {
        TaskType.INSPECT: ["list_directory", "read_file", "search_code"],
        TaskType.IDENTIFY: ["read_file", "search_code"],
        TaskType.UNDERSTAND: ["read_file", "search_code"],
        TaskType.CREATE: ["create_file", "edit_file", "read_file"],
        TaskType.MODIFY: ["edit_file", "read_file", "search_code"],
        TaskType.EXECUTE: ["execute_command"],
        TaskType.VERIFY: ["execute_command", "read_file"],
        TaskType.DIAGNOSE: ["read_file", "search_code", "execute_command"],
        TaskType.RECOVER: ["edit_file", "read_file", "execute_command"],
        TaskType.THINK: [],
    }
    return mapping.get(task_type, [])


def _action_for_task_type(task_type: TaskType) -> str:
    """Return the action string for a given task type."""
    mapping = {
        TaskType.INSPECT: "search",
        TaskType.IDENTIFY: "tool_use",
        TaskType.UNDERSTAND: "tool_use",
        TaskType.CREATE: "create",
        TaskType.MODIFY: "modify",
        TaskType.EXECUTE: "execute",
        TaskType.VERIFY: "execute",
        TaskType.DIAGNOSE: "search",
        TaskType.RECOVER: "modify",
        TaskType.THINK: "think",
    }
    return mapping.get(task_type, "tool_use")


class OrchaTaskPlanner:
    """
    The core Orcha Task Planner.

    Receives:
    - original user request
    - normalized objective
    - detected constraints
    - relevant existing context
    - available tools/capabilities
    - current workspace/application state when available

    Produces a minimal executable task graph with:
    - stable task IDs
    - concise titles
    - clear objectives
    - execution instructions
    - dependency edges
    - expected outcomes
    - required context
    - likely tools
    - verification criteria

    The planner does NOT execute tasks. It only creates the execution plan.
    """

    def __init__(
        self,
        completion_fn: Optional[Callable] = None,
        max_retries: int = 1,
        small_model: bool = False,
    ) -> None:
        self.completion_fn = completion_fn
        self.max_retries = max_retries
        self.small_model = small_model

    def plan(self, request: PlannerRequest) -> ExecutionPlan:
        """
        Create an execution plan from a PlannerRequest.

        This is the synchronous entry point. For model-based planning,
        use plan_async() instead.

        Returns a validated ExecutionPlan. Raises PlanValidationError
        if the plan cannot be constructed.
        """
        # Try model-based planning first if available
        if self.completion_fn is not None:
            # Model-based planning requires async; fall back to heuristic
            return self._heuristic_plan(request)
        return self._heuristic_plan(request)

    async def plan_async(self, request: PlannerRequest) -> ExecutionPlan:
        """
        Create an execution plan, using the model if available.

        Falls back to heuristic planning if the model fails or produces
        an invalid plan.
        """
        if self.completion_fn is not None:
            plan = await self._model_plan(request)
            if plan is not None:
                return plan
        return self._heuristic_plan(request)

    def _heuristic_plan(self, request: PlannerRequest) -> ExecutionPlan:
        """
        Build a plan using heuristic analysis of the request.

        This works without a model and produces structured task graphs
        based on keyword analysis and request decomposition.
        """
        text = request.original_request
        objective = request.normalized_objective or text

        # Split the request into logical clauses
        clauses = _split_clauses(text)
        if not clauses:
            clauses = [text]

        # Build task steps from clauses
        steps: List[TaskStep] = []
        prev_id: Optional[str] = None
        seen_ids: Set[str] = set()

        for i, clause in enumerate(clauses):
            task_type = _classify_task_type(clause)

            # Generate a stable task ID
            task_id = self._generate_task_id(task_type, clause, seen_ids)
            seen_ids.add(task_id)

            # Build the step
            deps = [prev_id] if prev_id else []
            ctx = [prev_id] if prev_id else []

            step = TaskStep(
                id=task_id,
                title=self._generate_title(task_type, clause),
                objective=self._generate_objective(task_type, clause, objective),
                execution_instructions=self._generate_instructions(task_type, clause),
                dependencies=deps,
                expected_outcome=self._generate_outcome(task_type, clause),
                required_context=ctx,
                likely_tools=_tools_for_task_type(task_type),
                verification_criteria=self._generate_verification(task_type, clause),
                action=_action_for_task_type(task_type),
            )
            steps.append(step)
            prev_id = task_id

        # If no steps were created, create a single direct execution step
        if not steps:
            steps.append(TaskStep(
                id="execute_directly",
                title="Execute request directly",
                objective=objective,
                execution_instructions=text,
                dependencies=[],
                expected_outcome="Request completed",
                required_context=[],
                likely_tools=[],
                verification_criteria="Request is fulfilled",
                action="tool_use",
            ))

        plan = ExecutionPlan(
            request=request,
            objective=objective,
            steps=steps,
            status="pending",
        )

        # Validate
        errors = validate_plan(plan)
        if errors:
            # If validation fails, fall back to single-step plan
            plan.steps = [TaskStep(
                id="execute_directly",
                title="Execute request directly",
                objective=objective,
                execution_instructions=text,
                dependencies=[],
                expected_outcome="Request completed",
                required_context=[],
                likely_tools=[],
                verification_criteria="Request is fulfilled",
                action="tool_use",
            )]

        return plan

    async def _model_plan(self, request: PlannerRequest) -> Optional[ExecutionPlan]:
        """Try to get a plan from the model. Returns None on failure.

        When small_model=True, uses the compact planner prompt optimized
        for 7B-class local models with limited context windows.
        """
        if self.small_model:
            from .small_model_prompts import PLANNER_SMALL
            prompt = PLANNER_SMALL.format(request=request.original_request[:500])
        else:
            prompt = build_planner_prompt(request)
        messages = [{"role": "user", "content": request.original_request}]

        for attempt in range(self.max_retries + 1):
            try:
                message = await self.completion_fn(messages, prompt, None)
                content = (message.get("content") or "").strip()
                steps = _parse_tasks_from_model(content)
                if not steps:
                    continue

                # Small models: enforce max 5 tasks
                if self.small_model and len(steps) > 5:
                    steps = steps[:5]

                plan = ExecutionPlan(
                    request=request,
                    objective=request.normalized_objective,
                    steps=steps,
                    status="pending",
                )
                errors = validate_plan(plan)
                if not errors:
                    return plan
            except Exception:
                continue
        return None

    def _generate_task_id(
        self, task_type: TaskType, clause: str, existing_ids: Set[str],
    ) -> str:
        """Generate a stable, unique task ID."""
        # Use task type as prefix
        prefix = task_type.value
        # Add a numeric suffix for uniqueness
        counter = 1
        while True:
            task_id = f"{prefix}_{counter}"
            if task_id not in existing_ids:
                return task_id
            counter += 1

    def _generate_title(self, task_type: TaskType, clause: str) -> str:
        """Generate a concise title for the task."""
        # Use the task type as a base, capitalize first letter
        base = task_type.value.capitalize()
        # Truncate clause for the title
        words = clause.split()[:6]
        detail = " ".join(words)
        if len(detail) > 40:
            detail = detail[:37] + "..."
        return f"{base}: {detail}" if detail else base

    def _generate_objective(
        self, task_type: TaskType, clause: str, overall_objective: str,
    ) -> str:
        """Generate a clear objective for the task."""
        return clause.strip().rstrip(".")

    def _generate_instructions(self, task_type: TaskType, clause: str) -> str:
        """Generate execution instructions."""
        instructions = {
            TaskType.INSPECT: f"Examine the codebase to: {clause}",
            TaskType.IDENTIFY: f"Analyze and determine: {clause}",
            TaskType.UNDERSTAND: f"Study and comprehend: {clause}",
            TaskType.CREATE: f"Create new code to: {clause}",
            TaskType.MODIFY: f"Modify existing code to: {clause}",
            TaskType.EXECUTE: f"Run the following: {clause}",
            TaskType.VERIFY: f"Validate that: {clause}",
            TaskType.DIAGNOSE: f"Analyze failures to: {clause}",
            TaskType.RECOVER: f"Fix issues to: {clause}",
            TaskType.THINK: f"Reason about: {clause}",
        }
        return instructions.get(task_type, f"Execute: {clause}")

    def _generate_outcome(self, task_type: TaskType, clause: str) -> str:
        """Generate expected outcome."""
        return f"Successfully completed: {clause.strip().rstrip('.')}"

    def _generate_verification(self, task_type: TaskType, clause: str) -> str:
        """Generate verification criteria."""
        verifications = {
            TaskType.INSPECT: "Project structure and patterns are documented",
            TaskType.IDENTIFY: "Framework and conventions are identified",
            TaskType.UNDERSTAND: "Requirements and affected areas are clear",
            TaskType.CREATE: "New code is written and follows conventions",
            TaskType.MODIFY: "Changes are applied correctly",
            TaskType.EXECUTE: "Commands complete successfully",
            TaskType.VERIFY: "Validation criteria are met",
            TaskType.DIAGNOSE: "Root causes are identified",
            TaskType.RECOVER: "Fixes resolve the issues",
            TaskType.THINK: "Analysis is complete",
        }
        return verifications.get(task_type, "Task is completed successfully")


__all__ = [
    "ComplexityGateNode",
    "ComplexityGateConfig",
    "TaskPlannerNode",
    "TaskContextBuilderNode",
    "TaskSummaryMemory",
    "ObserverNode",
    "ReplannerNode",
    "VerifierNode",
    "StepExecutorNode",
    "OrchaTaskPlanner",
    "TaskPlanBuilder",
    "TaskType",
    "PlanValidationError",
    "validate_plan",
    "build_planner_prompt",
    "TaskExecutionResult",
    "TaskExecutionState",
]
