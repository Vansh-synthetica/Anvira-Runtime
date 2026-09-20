"""
orcha.nodes.task_executor
=========================
Adaptive Task Execution Loop for complex task orchestration.

This module implements the core execution engine that:

1. Selects the next ready task from the execution plan
2. Builds a focused execution context (not the entire history)
3. Invokes the local model with task-specific instructions
4. Allows tool execution via existing Orcha tool system
5. Collects structured results
6. Observes state and evaluates outcomes
7. Updates task state
8. Can retry, replan, or modify tasks

Key architectural rule:
    The model receives ONLY what is relevant for the current task:
    - Original objective
    - Current task details
    - Required constraints
    - Necessary previous results
    - Relevant artifacts
    - Relevant tool outputs
    - Current execution state
    - Applicable tools
    - Execution rules

    The model is NEVER dumped the entire conversation/task history.

Safeguards:
    - Repeated identical tool calls detection
    - Empty model output detection
    - Malformed model output detection
    - Task loop detection
    - Failed tool execution handling
    - Task timeout
    - Cancellation support
    - Model refusal/failure handling
    - Task exhaustion detection

Every execution step is observable via events.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..core.packets import (
    ExecutionPlan,
    FinalResponse,
    ObjectiveVerificationResult,
    OrchaPacket,
    PacketKind,
    RecoveryTask,
    ReplanRequest,
    ReplanResult,
    TaskArtifact,
    TaskExecutionResult,
    TaskExecutionState,
    TaskObservation,
    TaskObservationDecision,
    TaskResult,
    TaskStep,
    VerificationResult,
    VerificationSignal,
)
from ..graph.context import (
    EVT_NODE_END,
    EVT_NODE_START,
    EVT_OBJECTIVE_MET,
    EVT_PLAN_UPDATED,
    EVT_RECOVERY_TASK_CREATED,
    EVT_REPLAN,
    EVT_TASK_COMPLETE,
    EVT_TASK_MODIFIED,
    EVT_TASK_SKIPPED,
    EVT_TASK_START,
    EVT_TEXT_DELTA,
    EVT_VERIFICATION_COMPLETE,
    EVT_VERIFICATION_SIGNAL,
    EVT_VERIFICATION_START,
    EVT_FINAL_RESPONSE,
    EVT_EXEC_RUN_CREATED,
    EVT_EXEC_INTENT_ANALYZED,
    EVT_EXEC_COMPLEXITY_DETERMINED,
    EVT_EXEC_PLANNING_STARTED,
    EVT_EXEC_PLAN_CREATED,
    EVT_EXEC_TASK_READY,
    EVT_EXEC_TASK_STARTED,
    EVT_EXEC_MODEL_STARTED,
    EVT_EXEC_TOOL_STARTED,
    EVT_EXEC_TOOL_COMPLETED,
    EVT_EXEC_TASK_OBSERVED,
    EVT_EXEC_TASK_COMPLETED,
    EVT_EXEC_TASK_FAILED,
    EVT_EXEC_RETRY_STARTED,
    EVT_EXEC_REPLANNING_STARTED,
    EVT_EXEC_PLAN_UPDATED,
    EVT_EXEC_VERIFICATION_STARTED,
    EVT_EXEC_VERIFICATION_COMPLETED,
    EVT_EXEC_RUN_COMPLETED,
    EVT_EXEC_RUN_FAILED,
    EVT_EXEC_RUN_CANCELLED,
    EVT_EXEC_RUN_RESUMED,
    EventEmitter,
    RunStateTracker,
)
import logging as _logging

# This pipeline previously reported success at every node while executing
# nothing, with no trace of why — diagnosing it required temporarily
# instrumenting the loop by hand. One INFO line per step outcome keeps that
# from ever being invisible again.
_exec_log = _logging.getLogger("orcha.nodes.task_executor")

from .small_model_prompts import (
    build_small_exec_system,
    detect_task_completion,
    build_malformed_recovery,
    EMPTY_OUTPUT_RECOVERY,
    REPEATED_TOOL_RECOVERY,
    DONE_MARKER,
    STUCK_MARKER,
    TASK_COMPLETE_MARKER,
    CANNOT_COMPLETE_MARKER,
    NEED_REPLAN_MARKER,
)
from .execution_metrics import RunMetrics, estimate_tokens
from ..graph.node import Node


_FILE_PATH_ARG_KEYS = ("path", "src", "dst", "source", "destination", "file_path")


_LARGE_ARG_KEYS = ("content", "old_string", "new_string", "append_content")


def _sanitize_tool_args_for_event(tool_args: Dict[str, Any]) -> Dict[str, Any]:
    """Cap known large string args before they go out over SSE — the model
    already saw the full content; the UI only needs enough to preview it."""
    if not isinstance(tool_args, dict):
        return {}
    out = dict(tool_args)
    for key in _LARGE_ARG_KEYS:
        value = out.get(key)
        if isinstance(value, str) and len(value) > 2000:
            out[key] = value[:2000] + f"\n… ({len(value) - 2000} more chars)"
    return out


def _extract_affected_files(tool_args: Dict[str, Any]) -> List[str]:
    """
    Best-effort extraction of file/directory paths a tool call named, so the
    activity timeline can show real affected-file chips per tool call
    instead of forcing the user to trust the model's own narration of what
    it touched.
    """
    if not isinstance(tool_args, dict):
        return []
    files: List[str] = []
    for key in _FILE_PATH_ARG_KEYS:
        value = tool_args.get(key)
        if isinstance(value, str) and value:
            files.append(value)
    return files


# ── Event Stream Adapter helper ───────────────────────────────────────────────


def _get_or_create_event_adapter(
    packet: OrchaPacket,
    ctx: RunContext,
    objective: str = "",
) -> EventStreamAdapter:
    """
    Get the event adapter from packet payload, or create and attach a new one.

    The adapter is persisted in the packet payload so it survives across graph
    node invocations. This allows the event stream to be maintained throughout
    the entire execution lifecycle.

    On resume, the tracker is restored from the persisted ``_event_stream_snapshot``
    so that sequence numbering, event history, and task tracking survive server
    restarts. Completed tasks are not re-executed.

    On first creation, emits a run_created event so the frontend sees the
    very start of the run timeline.
    """
    adapter: Optional[EventStreamAdapter] = packet.payload.get("_event_adapter")
    if adapter is not None:
        return adapter

    run_id = packet.id or ctx.run_id

    # On resume: restore the tracker from the persisted snapshot so that
    # sequence numbering, event history, and task tracking survive restarts.
    snapshot = packet.payload.get("_event_stream_snapshot")
    if snapshot is not None:
        from ..graph.context import RunStateTracker
        restored_tracker = RunStateTracker.from_snapshot(snapshot)
        adapter = EventStreamAdapter(run_id=run_id, objective=objective)
        adapter._tracker = restored_tracker
        adapter.attach(ctx.emit)
        # Emit a RUN_RESUMED event so the frontend knows this is a
        # continuation. EventStreamAdapter has no generic emit_event of its
        # own (only the explicit emit_<kind> methods below) — the actual
        # emit machinery lives on the tracker, same as every one of those.
        adapter._tracker.emit_event(
            EVT_EXEC_RUN_RESUMED,
            summary=f"Run resumed — {len(restored_tracker._completed_task_ids)} tasks already completed",
            metadata={
                "restored_seq": restored_tracker._seq,
                "restored_status": restored_tracker._status,
                "completed_tasks": list(restored_tracker._completed_task_ids),
                "failed_tasks": list(restored_tracker._failed_task_ids),
            },
        )
        return adapter

    # Fresh run: create a new adapter
    adapter = EventStreamAdapter(run_id=run_id, objective=objective)
    adapter.attach(ctx.emit)
    adapter.emit_run_created(
        objective=objective,
        graph_name=getattr(ctx, "graph_name", ""),
    )
    return adapter


# ── Execution rules ──────────────────────────────────────────────────────────

# A suffix appended to the system prompt to tell the completion_fn's
# strict-tools envelope builder (orcha/api/server.py's
# _build_agent_completion_fn) to constrain the model's JSON schema so
# "action" can ONLY be "tool" — the "final"/describe-instead-of-act escape
# hatch is structurally removed, not just discouraged in prose.
#
# Why this exists: confirmed live that prompting alone doesn't reliably
# fix this. Small/quantized local models (Qwen2.5-Coder 3B and 7B q4
# observed directly) consistently chose action="final" with a prose
# description in "answer" instead of action="tool" — even under the
# schema-constrained envelope (so the JSON was always syntactically valid),
# and even after an explicit corrective retry naming the exact mistake and
# telling the model to call the tool. The model was making a genuine
# CHOICE to describe rather than act; wording could not close that gap
# because the schema itself still let it choose. Removing "final" from the
# enum via a stricter schema variant closes it structurally instead:
# grammar-constrained decoding makes the response literally incapable of
# being anything but a real tool call.
#
# Applied only while a step that expects a tool call hasn't made one yet
# (see step_expects_tool / state.tool_call_count in _execute_task_loop) —
# once real progress has happened, or if the step turns out not to need a
# tool after all, normal calls (marker absent) allow a text final answer
# again so a genuinely text-only step, or a task's closing summary after
# its tool calls ran, isn't forced into a tool call it doesn't need.
FORCE_TOOL_CALL_MARKER = "\n\n<<ORCHA_FORCE_TOOL_CALL>>"

TASK_EXECUTION_SYSTEM_PROMPT = """You are executing a specific task within a larger project. You are NOT working on the entire project — you are focused ONLY on this single task.

OVERALL PROJECT REQUEST (for reference — the planner split this into steps; if the step summary below is thinner than this, treat this as the source of truth for concrete requirements):
{original_request}

YOUR TASK:
{task_objective}

EXECUTION INSTRUCTIONS:
{execution_instructions}

EXPECTED OUTCOME:
{expected_outcome}

VERIFICATION CRITERIA:
{verification_criteria}

CONSTRAINTS:
{constraints}

WORKSPACE STATE:
{workspace_state}

IMPORTANT RULES:
1. You are executing THIS specific task, not the entire overall project.
2. Use ONLY the tools listed as available for this task.
3. Do NOT modify files or run commands outside the scope of this task.
4. If a tool call is rejected with a validation error (wrong/missing argument
   name, wrong type, etc.), that is NOT a reason to give up — the error
   tells you exactly what's wrong. Fix the arguments and call the tool
   again. A single error may name only ONE problem at a time (e.g. it says
   "content" is missing after you already fixed "path") — that's normal,
   not a new failure; fix that one too and retry again. Keep retrying,
   fixing one issue per error, until the call succeeds or you have made
   several genuine attempts. Only report "CANNOT COMPLETE" after actually
   exhausting reasonable attempts to fix the arguments, or when the task is
   genuinely impossible for a reason no argument change fixes.
5. Every tool call must be justified by the task requirements.
6. Never fabricate successful execution — report actual tool outputs.
7. If the task requires information from prior steps, use only what is provided in the context.
8. When done, reply with "TASK COMPLETE:" followed by what you accomplished.

PROHIBITED:
- Working on tasks outside your scope
- Modifying files not related to this task
- Running commands not needed for this task
- Fabricating tool outputs
- Claiming success without verification"""

TASK_REPLAN_PROMPT = """The previous task execution failed. Analyze the failure and suggest how to recover.

FAILED TASK:
{failed_task}

ERROR:
{error}

PRIOR SUCCESSFUL TASKS:
{prior_results}

SUGGEST a recovery approach:
1. Can the failed task be retried with different instructions?
2. Should the task be split into smaller subtasks?
3. Should the task be skipped entirely?
4. Should the plan be modified?

Reply with a JSON object:
{{
    "action": "retry" | "skip" | "modify_plan" | "abort",
    "reason": "explanation",
    "modified_task": {{optional modified task if action=modify_plan}},
    "additional_context": {{optional extra context for retry}}
}}"""

TASK_OBSERVER_EVAL_PROMPT = """You are evaluating whether a completed task actually satisfies its objective.

TASK OBJECTIVE:
{task_objective}

EXPECTED OUTCOME:
{expected_outcome}

VERIFICATION CRITERIA:
{verification_criteria}

ACTUAL OUTPUT:
{actual_output}

TOOL CALLS MADE:
{tool_calls}

Does this output satisfy the objective? Consider:
1. Is the output complete and correct?
2. Does it match the expected outcome?
3. Are the verification criteria met?

Reply with EXACTLY one of these decisions:
- CONTINUE: The task succeeded and the plan remains valid
- RETRY: The task failed transiently (e.g. model error, timeout) and can be retried
- MODIFY: The task output is wrong/incomplete but the approach can be changed
- REPLAN: New information means the plan graph needs structural changes
- BLOCK: An external condition prevents progress
- COMPLETE: The overall objective has already been satisfied

Then explain briefly.

Format:
DECISION: <one of CONTINUE/RETRY/MODIFY/REPLAN/BLOCK/COMPLETE>
REASON: <brief explanation>
FEEDBACK: <optional feedback to inject on retry/modify>"""


# Every registered tool that can put bytes on disk. This list was
# previously just ("write_file", "edit_file"), which silently excluded
# create_file — the tool a model actually reaches for when the task is
# "create a new file", and the one used in every observed run. The cost of
# missing it is not cosmetic: a file written by create_file produced no
# artifact, so nothing downstream could learn where it landed, and the next
# step guessed a different directory.
_FILE_WRITING_TOOLS = frozenset({
    "create_file",
    "write_file",
    "edit_file",
    "apply_patch",
    "append_file",
    "replace_text",
    "copy_file",
    "move_file",
    "rename_file",
})


def _files_written_so_far(
    plan: ExecutionPlan,
    exclude_step_id: Optional[str] = None,
) -> List[str]:
    """Real paths this run has written, in the order they were written.

    Reads the recorded tool calls rather than any step's prose output, so
    it reports what actually happened on disk instead of what the model
    said it did. Deduplicated, and a path rewritten later keeps its first
    position so the listing stays stable across steps.
    """
    seen: List[str] = []
    for step in plan.steps:
        if step.id == exclude_step_id or step.result is None:
            continue
        for call in step.result.tool_calls or []:
            if call.get("tool") not in _FILE_WRITING_TOOLS:
                continue
            # `files` is extracted at record time from the FULL arguments
            # (see TaskExecutionState.record_tool_call). Prefer it: the
            # `args_preview` fallback below is truncated to 200 chars, which
            # is shorter than any real file write, so re-parsing it as JSON
            # finds nothing for exactly the calls worth reporting.
            for path in call.get("files") or []:
                if path and path not in seen:
                    seen.append(path)
            if call.get("files"):
                continue
            args_preview = call.get("args_preview") or ""
            try:
                args = json.loads(args_preview, strict=False) if args_preview else {}
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(args, dict):
                continue
            path = args.get("path") or args.get("file_path") or ""
            if path and path not in seen:
                seen.append(path)
    return seen


class TaskArtifactCollector:
    """Collects artifacts from tool call history and task results."""

    @staticmethod
    def collect_from_tool_history(
        tool_history: List[Dict[str, Any]],
        step_id: str,
    ) -> List[TaskArtifact]:
        """Extract artifacts from tool call history."""
        artifacts: List[TaskArtifact] = []
        for call in tool_history:
            tool_name = call.get("tool", "")
            args_preview = call.get("args_preview", "")

            # File write operations
            if tool_name in _FILE_WRITING_TOOLS:
                try:
                    args = json.loads(args_preview) if args_preview else {}
                    path = args.get("path", args.get("file_path", ""))
                    if path:
                        artifacts.append(TaskArtifact(
                            step_id=step_id,
                            kind="file",
                            description=f"Modified file: {path}",
                            location=path,
                            content_preview=args_preview[:200],
                        ))
                except (json.JSONDecodeError, AttributeError):
                    pass

            # Command execution
            elif tool_name in ("execute_command", "run_command", "shell"):
                try:
                    args = json.loads(args_preview) if args_preview else {}
                    cmd = args.get("command", args.get("cmd", ""))
                    if cmd:
                        artifacts.append(TaskArtifact(
                            step_id=step_id,
                            kind="command",
                            description=f"Executed: {cmd[:100]}",
                            content_preview=args_preview[:200],
                        ))
                except (json.JSONDecodeError, AttributeError):
                    pass

            # Search operations
            elif tool_name in ("search_files", "grep", "find"):
                try:
                    args = json.loads(args_preview) if args_preview else {}
                    query = args.get("query", args.get("pattern", ""))
                    if query:
                        artifacts.append(TaskArtifact(
                            step_id=step_id,
                            kind="observation",
                            description=f"Search for: {query[:100]}",
                            content_preview=args_preview[:200],
                        ))
                except (json.JSONDecodeError, AttributeError):
                    pass

        return artifacts

    @staticmethod
    def collect_from_result(result: TaskExecutionResult) -> List[TaskArtifact]:
        """Extract artifacts from a task execution result."""
        artifacts: List[TaskArtifact] = []

        # Add output as an observation artifact
        if result.output:
            artifacts.append(TaskArtifact(
                step_id=result.step_id,
                kind="output",
                description=f"Task output ({len(result.output)} chars)",
                content_preview=result.output[:300],
            ))

        # Collect from tool summaries
        artifacts.extend(
            TaskArtifactCollector.collect_from_tool_history(
                result.tool_summaries, result.step_id
            )
        )

        return artifacts


# ── Faked tool-call recovery ──────────────────────────────────────────────────
#
# Some models — especially weaker/free ones accessed via an OpenAI-compatible
# proxy — don't reliably use the API's native function-calling protocol even
# when given a tool schema. Instead of populating the structured `tool_calls`
# field, they write a JSON-shaped envelope describing the call as plain text
# content (e.g. {"action": "tool", "name": "write_file", "arguments": {...}}).
# That JSON is inert prose from the executor's point of view: nothing runs,
# no file gets written, yet the response looks like real progress. Treating
# this purely as a "the model didn't use a tool" case (the corrective-retry
# path below) wastes the one extra round-trip a model like this may not
# recover from. Since the model already committed to a specific, structured
# tool name and arguments — it just sent them over the wrong channel — this
# recovers that intent directly rather than blaming the model tier for what
# is really our pipeline not accepting a call it can plainly identify.
#
# Deliberately conservative: only a small set of common envelope shapes are
# recognized, the tool name must resolve (directly or via one explicit alias)
# to a tool actually registered for this task, and arguments must themselves
# parse as a JSON object. Anything else (a bare code fence with no structured
# call, unparseable JSON, an unknown tool name) falls through unchanged to
# the existing corrective-retry/acceptance logic — this never guesses at a
# file path or action that wasn't explicitly named by the model.
_FAKE_TOOL_CALL_NAME_ALIASES = {
    "create_file": "write_file",
    "new_file": "write_file",
    "make_file": "write_file",
    "edit_file": "write_file",
    "update_file": "write_file",
    "run_command": "execute_command",
    "shell": "execute_command",
    "run_shell": "execute_command",
}


def _extract_json_object(text: str) -> Optional[str]:
    """Pull the first balanced top-level JSON object out of `text`, unwrapping
    a ```json fence if present. The scan is string-aware (tracks whether it's
    inside a JSON string literal, respecting `\\"` escapes) so a `{`/`}`
    appearing literally inside a "content" field's own code (e.g. a `{ ... }`
    destructure or function body in the file being written) doesn't throw off
    the brace count — a naive counter would terminate on the first internal
    `}` and either return a truncated, unparseable fragment or the wrong
    object entirely. Also tolerates leading junk before the first `{` (some
    models leak raw template/special-token text ahead of an otherwise-valid
    JSON payload), by starting the scan at the first `{` found anywhere.
    Returns None if no balanced object is found."""
    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*\n(.*?)\n?```$", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    start = stripped.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : i + 1]
    return None


_INVOKE_NAME_RE = re.compile(r'(?:invoke|tool_call|function)[^>]*?\bname\s*=\s*"([a-zA-Z_][\w.-]*)"')


def _resolve_tool_name(name: str, valid_tool_names: "set[str]") -> Optional[str]:
    if name in valid_tool_names:
        return name
    alias = _FAKE_TOOL_CALL_NAME_ALIASES.get(name)
    return alias if alias in valid_tool_names else None


def try_parse_faked_tool_call(
    content: str, valid_tool_names: "set[str]"
) -> Optional[Dict[str, Any]]:
    """If `content` is really a tool call the model wrote as JSON text instead
    of a real tool_calls entry, return an OpenAI-style call dict
    ({"id", "function": {"name", "arguments"}}) ready to execute. Returns
    None when the content doesn't confidently match this pattern.

    Handles two shapes seen in practice:
    1. A self-contained JSON envelope naming the tool itself, e.g.
       {"action": "tool", "name": "write_file", "arguments": {...}}.
    2. A leaked/garbled pseudo-XML wrapper (some models' native tool-call
       template tokens showing up as literal text instead of being consumed,
       e.g. `...<invoke name="write_file">...<arguments>: {...}`) where the
       tool name lives in an `name="..."` attribute outside the JSON, and the
       JSON object itself IS the arguments rather than wrapping them.
    """
    raw = _extract_json_object(content)
    if raw is None:
        return None
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(envelope, dict):
        return None

    nested_function = envelope.get("function")
    envelope_name = (
        envelope.get("name")
        or envelope.get("tool")
        or envelope.get("tool_name")
        or (nested_function.get("name") if isinstance(nested_function, dict) else None)
    )

    resolved_name: Optional[str] = None
    args: Any = None

    if envelope_name and isinstance(envelope_name, str):
        resolved_name = _resolve_tool_name(envelope_name, valid_tool_names)
        if resolved_name:
            args = (
                envelope.get("arguments")
                if "arguments" in envelope
                else envelope.get("tool_input") if "tool_input" in envelope
                else envelope.get("parameters") if "parameters" in envelope
                else envelope.get("args")
            )

    if resolved_name is None:
        # Shape 2: name isn't a JSON key at all — look for it as a tag
        # attribute elsewhere in the text, and treat the extracted object as
        # the arguments directly (it wasn't wrapped in an envelope).
        match = _INVOKE_NAME_RE.search(content)
        if match:
            resolved_name = _resolve_tool_name(match.group(1), valid_tool_names)
            if resolved_name:
                args = envelope

    if resolved_name is None:
        return None

    if isinstance(args, str):
        try:
            args = json.loads(args, strict=False)
        except json.JSONDecodeError:
            return None
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None

    return {
        "id": "recovered-fake-tool-call-0",
        "function": {"name": resolved_name, "arguments": json.dumps(args)},
    }


# ── Task Execution Loop ──────────────────────────────────────────────────────

class TaskExecutionLoop(Node):
    """
    Adaptive task execution loop that orchestrates the execution of a single
    task through the model, with tool execution, observation, and safeguards.

    This node is the core of the execution engine. It:
    1. Reads the execution plan from the packet
    2. Selects the next ready task
    3. Builds focused context for that task
    4. Invokes the model with task-specific instructions
    5. Allows tool execution via existing Orcha tool system
    6. Collects structured results
    7. Observes state and evaluates outcomes
    8. Updates task state
    9. Can retry, replan, or modify tasks

    The model receives ONLY what is relevant for the current task.
    """

    name = "task_execution_loop"

    def __init__(
        self,
        name: str = "task_execution_loop",
        completion_fn: Optional[Callable] = None,
        executor: Optional[Any] = None,  # ToolExecutor
        timeout_s: Optional[float] = 600.0,
        max_task_iterations: int = 5,
        max_tool_repeats: int = 2,
        max_empty_outputs: int = 2,
        max_consecutive_errors: int = 2,
        retries: int = 0,
        small_model: bool = False,
        workspace_roots: Optional[List[str]] = None,
        metrics: Optional[RunMetrics] = None,
    ) -> None:
        self.name = name
        self.timeout_s = timeout_s
        self.retries = retries
        self.completion_fn = completion_fn
        self.executor = executor
        self.small_model = small_model
        self.workspace_roots = list(workspace_roots or [])
        self.metrics = metrics

        # Small-model optimizations: stricter limits
        if small_model:
            self.max_task_iterations = min(max_task_iterations, 3)
            self.max_tool_repeats = min(max_tool_repeats, 1)
            self.max_empty_outputs = min(max_empty_outputs, 1)
            self.max_consecutive_errors = min(max_consecutive_errors, 1)
        else:
            self.max_task_iterations = max_task_iterations
            self.max_tool_repeats = max_tool_repeats
            self.max_empty_outputs = max_empty_outputs
            self.max_consecutive_errors = max_consecutive_errors

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        plan_raw = packet.payload.get("execution_plan")
        if plan_raw is None:
            return packet.fork(
                packet.kind,
                task_execution_result=TaskExecutionResult(
                    step_id="unknown",
                    success=False,
                    output="No execution plan found",
                ).model_dump(),
            )
        plan = ExecutionPlan(**plan_raw) if isinstance(plan_raw, dict) else plan_raw

        # Get or create the event adapter for this run
        adapter = _get_or_create_event_adapter(packet, ctx, plan.objective)

        # Use the step TaskContextBuilderNode already selected (and marked
        # in_progress) one hop earlier, rather than independently re-querying
        # next_pending_step() — that only matches status == "pending"
        # (core/packets.py ExecutionPlan.next_pending_step), which the
        # already-selected step no longer is, so re-deriving here always
        # found nothing and silently no-opped the entire task (zero tool
        # calls, no task_execution_result, straight to "task_complete").
        # Falls back to next_pending_step() when no task_step_id is present
        # (e.g. this node used standalone, without TaskContextBuilderNode
        # having run first).
        step_id = packet.payload.get("task_step_id")
        step = plan.step_by_id(step_id) if step_id else None
        if step is None:
            step = plan.next_pending_step()
        if step is None:
            # Genuinely nothing left to run is a normal finish. But a plan
            # that is NOT complete and still yields no runnable step is the
            # silent no-op this pipeline's worst failure mode: every node
            # reports success, zero tools run, and the user gets a bare
            # "couldn't complete that" with nothing to debug from.
            # Confirmed live: a valid 4-step plan executed nothing and the
            # run ended with no files written and no error text anywhere.
            # Carry the reason forward instead of returning empty-handed.
            if not plan.is_complete:
                statuses = ", ".join(
                    f"{s.id}={s.status}" for s in plan.steps
                ) or "no steps"
                return packet.fork(
                    packet.kind,
                    execution_plan=plan.model_dump(),
                    task_execution_result=TaskExecutionResult(
                        step_id=step_id or "unknown",
                        success=False,
                        output="",
                        error=(
                            "No runnable step: the plan is incomplete but no "
                            "step was selectable. "
                            f"task_step_id={step_id!r}; step statuses: {statuses}. "
                            "next_pending_step() only matches status "
                            "'pending', so a step left 'in_progress' by an "
                            "earlier node with a task_step_id that no longer "
                            "resolves strands the run here."
                        ),
                    ).model_dump(),
                    _event_adapter=adapter,
                )
            return packet.fork(
                packet.kind,
                execution_plan=plan.model_dump(),
                task_complete=True,
                _event_adapter=adapter,
            )

        # Emit task start event (both internal + structured)
        await ctx.emit.emit(
            EVT_TASK_START,
            self.name,
            packet,
            task_id=step.id,
            task_title=step.title,
            task_objective=step.objective,
            plan_progress=plan.progress,
        )
        adapter.emit_task_started(
            task_id=step.id,
            task_title=step.title,
            plan_progress=plan.progress,
        )

        # Build execution state
        state = TaskExecutionState(
            step_id=step.id,
            max_iterations=self.max_task_iterations,
            timeout_s=300.0,
        )

        # Build focused execution context
        context = self._build_focused_context(plan, step, packet)

        # Execute the task through the model loop
        result = await self._execute_task_loop(step, state, context, ctx, packet, adapter)

        # Update step with result
        step.result = TaskResult(
            step_id=step.id,
            output=result.output,
            success=result.success,
            error=result.error,
            tool_calls=result.tool_summaries,
            latency_s=result.duration_s,
        )
        step.status = "completed" if result.success else "failed"
        _exec_log.info(
            "step %s -> %s (tools=%d, iterations=%s, error=%s)",
            step.id, step.status, len(result.tool_summaries or []),
            getattr(result, "iterations", "?"),
            (result.error or "")[:200] or "none",
        )

        # Update plan
        plan.status = "in_progress"

        # Emit task complete event (both internal + structured)
        await ctx.emit.emit(
            EVT_TASK_COMPLETE,
            self.name,
            packet,
            task_id=step.id,
            task_title=step.title,
            success=result.success,
            duration_s=result.duration_s,
            tool_calls_made=result.tool_calls_made,
            iterations_used=result.iterations_used,
            exhaustion_reason=result.exhaustion_reason,
            plan_progress=plan.progress,
        )
        if result.success:
            adapter.emit_task_completed(
                task_id=step.id,
                task_title=step.title,
                plan_progress=plan.progress,
                duration_s=result.duration_s,
                tool_calls_made=result.tool_calls_made,
            )
        else:
            adapter.emit_task_failed(
                task_id=step.id,
                task_title=step.title,
                error=result.exhaustion_reason or result.error or "unknown",
                plan_progress=plan.progress,
            )

        # Update adapter state
        adapter.update_plan(plan)

        return packet.fork(
            packet.kind,
            execution_plan=plan.model_dump(),
            task_execution_result=result.model_dump(),
            task_step_id=step.id,
            task_step_status=step.status,
            _event_adapter=adapter,
            _event_stream_snapshot=adapter.get_snapshot(),
        )

    def _build_focused_context(
        self,
        plan: ExecutionPlan,
        step: TaskStep,
        packet: OrchaPacket,
    ) -> Dict[str, Any]:
        """
        Build focused execution context for a single task.

        Architectural rule: Only include what is relevant for THIS task.
        Do NOT include entire plan history or all prior results.
        """
        # Collect ONLY results from dependencies and required context
        # Truncate to prevent context overflow
        # Small models: tighter truncation to fit context window
        truncation = 200 if self.small_model else 500
        prior_results: List[str] = []
        for dep_id in step.dependencies + step.required_context:
            dep_step = plan.step_by_id(dep_id)
            if dep_step and dep_step.result:
                # Truncate output to keep context focused
                output = dep_step.result.output[:truncation]
                prior_results.append(
                    f"[{dep_id}] {dep_step.title}:\n{output}"
                )

        # Collect constraints from plan request
        constraints: List[str] = []
        if plan.request:
            constraints = plan.request.constraints
            workspace_state = plan.request.workspace_state
            original_request = plan.request.original_request
        else:
            workspace_state = ""
            original_request = plan.objective

        # Count plan progress for context
        completed_count = len([s for s in plan.steps if s.status == "completed"])
        total_count = len(plan.steps)

        return {
            "original_request": original_request,
            "objective": plan.objective,
            "step_id": step.id,
            "step_title": step.title,
            "step_objective": step.objective,
            "execution_instructions": step.execution_instructions,
            "expected_outcome": step.expected_outcome,
            "verification_criteria": step.verification_criteria,
            "likely_tools": step.likely_tools,
            "action": step.action,
            "prior_results": prior_results,
            "constraints": constraints,
            "workspace_state": workspace_state,
            "plan_progress": f"{completed_count}/{total_count} tasks completed",
            # Real paths this run has already written, read off the recorded
            # tool calls. TaskContextBuilderNode assembles a similar string
            # into `task_step_context`, but NOTHING reads that payload key —
            # the execution loop builds its own context from the plan, which
            # is why every step was choosing its own directory. The facts
            # have to live here to reach the model at all.
            "files_written": _files_written_so_far(plan, exclude_step_id=step.id),
            # The directory the workspace tools actually operate on. Stating
            # it removes the last "where am I" guess: relative paths in the
            # step's own instructions are anchored, and the model stops
            # inventing subfolders to cd into.
            "workspace_root": self.workspace_roots[0] if self.workspace_roots else "",
            # Selective retrieval from working memory: this step's own prior
            # failures and the run's durable facts. Not the transcript — the
            # micro-prompt has to stay small to be worth having.
            "failed_attempts": plan.working_memory.failures_for(step.id),
            "known_facts": plan.working_memory.facts[-8:],
        }

    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build the system prompt for the model from focused context.

        When small_model=True, uses the compact prompt template designed for
        7B-class local models with limited context windows.
        """
        if self.small_model:
            return build_small_exec_system(
                step_title=context["step_title"],
                step_objective=context["step_objective"],
                tools=context.get("likely_tools", []),
                execution_instructions=context.get("execution_instructions", ""),
                verification_criteria=context.get("verification_criteria", ""),
                constraints=context.get("constraints"),
                prior_results=context.get("prior_results"),
            )
        return TASK_EXECUTION_SYSTEM_PROMPT.format(
            original_request=context["original_request"],
            task_objective=context["step_objective"],
            execution_instructions=context["execution_instructions"],
            expected_outcome=context["expected_outcome"],
            verification_criteria=context["verification_criteria"],
            constraints="\n".join(context["constraints"]) if context["constraints"] else "None",
            workspace_state=context["workspace_state"] or "Not specified",
        )

    def _build_user_message(
        self,
        context: Dict[str, Any],
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build the step prompt: ONE target, named precisely.

        A decomposed step is only worth decomposing if the prompt that
        carries it is narrower than the prompt for the whole objective. The
        previous version was not: it gave a title, a one-line objective, a
        bare list of tool NAMES, and nothing else — no file path, no
        argument shape, and it silently dropped ``execution_instructions``
        (the planner's actual step-by-step detail) on the floor.

        A 3B model handed "Execute the task: Add mean function / Available
        tools: edit_file" has to guess the path, the call shape and what
        finished looks like. Observed consequences, all from live runs on
        the same task: ``create_file`` invoked with ``{}``, ``edit_file``
        invoked with no ``old_string``, one step writing ``src/stats.py``
        while the next wrote ``test_stats.py`` at the root, and a test file
        importing ``mean_median`` — a module no step ever created.

        So this states, in order: the single action, the exact target, the
        exact call shape for the tool that action needs, the paths this run
        has already written, and the done condition. Every one of those is
        a guess removed.
        """
        parts = [
            "Do exactly ONE thing: the task below. "
            "Not the overall project, not the next task.",
            f"\nTASK: {context['step_title']}",
            f"GOAL: {context['step_objective']}",
        ]

        # The planner's own step-by-step detail. Previously computed into
        # the context dict and then never used in the prompt.
        instructions = (context.get("execution_instructions") or "").strip()
        if instructions:
            parts.append(f"HOW: {instructions}")

        expected = (context.get("expected_outcome") or "").strip()
        if expected:
            parts.append(f"DONE WHEN: {expected}")

        root = (context.get("workspace_root") or "").strip()
        if root:
            parts.append(
                f"\nWORKING DIRECTORY: {root}\n"
                "Tools already run there. Use plain relative paths like "
                "'stats.py'. Do NOT cd into a subfolder and do NOT invent "
                "one — anything you name must already exist."
            )

        # Concrete paths beat prose about paths. This is what stops step N+1
        # from inventing a directory step N never used.
        files_written = context.get("files_written") or []
        if files_written:
            parts.append(
                "\nFILES THIS RUN HAS ALREADY WRITTEN — use these EXACT "
                "paths; put related new files in the same directory:"
            )
            parts.extend(f"  {p}" for p in files_written)

        if context["prior_results"]:
            parts.append("\nOutput from the steps this one depends on:")
            results = context["prior_results"]
            # Small models: limit prior results to most recent 1, truncated
            if self.small_model and len(results) > 1:
                results = results[-1:]
            parts.extend(results)

        known_facts = context.get("known_facts") or []
        if known_facts:
            parts.append("\nKNOWN (established earlier in this run):")
            parts.extend(f"  {f}" for f in known_facts)

        # What has already been ruled out. This is the half of working
        # memory that changes behaviour: a retry that cannot see its own
        # previous attempt will make the identical call again, and did —
        # four times, with the same error each time.
        failed = context.get("failed_attempts") or []
        if failed:
            parts.append(
                "\nALREADY TRIED AND FAILED — do NOT repeat any of these, "
                "change your approach:"
            )
            for attempt in failed:
                tool = getattr(attempt, "tool", "") or "?"
                args = getattr(attempt, "args", "") or ""
                err = getattr(attempt, "error", "") or ""
                times = getattr(attempt, "attempts", 1)
                suffix = f" (tried {times}x)" if times > 1 else ""
                parts.append(f"  {tool}({args[:120]}){suffix}")
                parts.append(f"    -> failed: {err[:160]}")

        # The exact call shape, quoted from the live schema rather than
        # described. Malformed arguments were the single largest source of
        # wasted iterations in every measured run.
        tool_block = self._describe_tools(
            context.get("likely_tools") or [], tool_schemas or []
        )
        if tool_block:
            parts.append(tool_block)

        parts.append(
            "\nCall a tool to do the work — do not describe what you would "
            "do, and do not answer in prose alone. When the task is done, "
            "reply 'TASK COMPLETE:' followed by what you did. Only if it is "
            "genuinely impossible, reply 'CANNOT COMPLETE:' followed by why."
        )
        return "\n".join(parts)

    @staticmethod
    def _describe_tools(
        likely_tools: List[str],
        tool_schemas: List[Dict[str, Any]],
    ) -> str:
        """Render the exact argument shape of this step's tools.

        Names alone ("Available tools: create_file") tell a small model
        nothing about how to call it, and it fills the gap by guessing —
        usually with an empty argument object. Quoting the required keys
        straight from the registered schema removes the guess.
        """
        if not likely_tools:
            return ""
        by_name = {
            s["function"]["name"]: s["function"]
            for s in tool_schemas
            if isinstance(s.get("function"), dict) and s["function"].get("name")
        }
        lines: List[str] = []
        for name in likely_tools:
            fn = by_name.get(name)
            if fn is None:
                # The planner named a tool that does not exist. Say so here
                # rather than letting the step burn its attempts on it.
                lines.append(f'  {name} — NOT AVAILABLE, do not call it')
                continue
            params = fn.get("parameters") or {}
            required = params.get("required") or []
            props = params.get("properties") or {}
            keys = required or list(props)[:4]
            shape = ", ".join(f'"{k}": <{k}>' for k in keys)
            lines.append(f"  {name} — call with exactly {{{shape}}}")
        if not lines:
            return ""
        return (
            "\nTOOLS FOR THIS TASK (every argument below is required):\n"
            + "\n".join(lines)
        )

    async def _execute_task_loop(
        self,
        step: TaskStep,
        state: TaskExecutionState,
        context: Dict[str, Any],
        ctx: RunContext,
        packet: OrchaPacket,
        adapter: Optional[EventStreamAdapter] = None,
    ) -> TaskExecutionResult:
        """
        Execute a single task through the model loop with safeguards.

        Architectural rule: The model receives ONLY what is relevant for the
        current task. We never dump the entire conversation/task history.

        Message management: We use a focused message window that contains only:
        1. The initial user message (task description)
        2. The most recent assistant + tool exchange (for continuity)
        3. A summary of prior tool results (not full history)

        This prevents context window overflow and keeps the model focused.
        """
        if self.completion_fn is None:
            return TaskExecutionResult(
                step_id=step.id,
                success=False,
                output="No model available for execution",
                error="No completion_fn configured",
            )

        # Tool schemas are resolved BEFORE the prompts are built, not after:
        # the step prompt quotes the real argument shape of the tool this
        # step is meant to call, and it can only do that if the schemas are
        # already in hand.
        tool_schemas: List[Dict[str, Any]] = []
        if self.executor is not None:
            tool_schemas = self.executor.schemas()

        system_prompt = self._build_system_prompt(context)
        user_message = self._build_user_message(context, tool_schemas)

        # Metrics tracking
        task_metrics = None
        if self.metrics:
            task_metrics = self.metrics.record_task_start(step.id, title=step.title)
            task_metrics.system_prompt_tokens_est = estimate_tokens(system_prompt)
            task_metrics.user_message_tokens_est = estimate_tokens(user_message)

        # Initial messages: just the task description
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_message}]

        # Track tool results for summary (not full history)
        tool_results_summary: List[str] = []
        tool_names_used: List[str] = []

        valid_tool_names = {
            s["function"]["name"]
            for s in tool_schemas
            if isinstance(s.get("function"), dict) and s["function"].get("name")
        }
        # "The model said it had no way to create a file" and "the model was
        # genuinely handed no tools" look identical from the outside, and
        # only one of them is the model's fault. Record which, per step.
        _exec_log.info(
            "step %s starting: executor=%s schemas=%d likely_tools=%s",
            step.id,
            type(self.executor).__name__ if self.executor is not None else None,
            len(tool_schemas),
            ",".join(step.likely_tools) or "-",
        )

        final_output = ""
        completed = False
        malformed_count = 0
        model_calls = 0
        # Tool calls that actually EXECUTED successfully, as opposed to
        # state.tool_call_count which counts every call the model emitted —
        # including ones rejected for bad arguments before they ever ran.
        # The distinction matters: a malformed first call must not be able
        # to satisfy "you have used a tool", or the forcing below releases
        # after a call that accomplished nothing.
        successful_tool_calls = 0
        # One corrective round-trip for a premature "CANNOT COMPLETE".
        refused_without_tool = False
        # Counts corrective round-trips issued for a text-only response when
        # a tool was clearly expected but never called (see the `else`
        # branch below). Capped at one so a model that simply won't
        # tool-call can't loop forever.
        text_only_without_tool_count = 0

        # Whether this step is the kind that should be doing something to
        # the workspace at all — mirrors the same two signals used by the
        # corrective-retry gate below (context["likely_tools"] is the
        # PLANNING model's own guess and can be empty even when the step
        # genuinely needs a tool; step.action has a firmer default).
        # Computed once — it doesn't change turn to turn — and used to
        # decide whether FORCE_TOOL_CALL_MARKER applies to THIS call (see
        # its own module-level comment for why prompting alone wasn't
        # enough and a schema-level constraint is used instead).
        step_expects_tool = bool(context["likely_tools"]) or context.get("action") in (
            "tool_use", "create", "modify", "execute", "search",
        )

        for iteration in range(1, state.max_iterations + 1):
            if ctx.cancelled:
                state.record_error("Cancelled by user")
                await ctx.emit.emit(
                    EVT_TASK_MODIFIED, self.name, packet,
                    task_id=step.id, reason="cancelled",
                )
                break

            state.iteration = iteration
            state.last_activity_time = time.time()

            # Check timeout
            if state.is_timed_out:
                state.record_error("Task timed out")
                await ctx.emit.emit(
                    EVT_TASK_MODIFIED, self.name, packet,
                    task_id=step.id, reason="timeout",
                    duration_s=time.time() - state.start_time,
                )
                break

            # Build focused messages for this iteration
            # Rule: Never dump entire history. Use focused window.
            focused_messages = self._build_focused_messages(
                messages, tool_results_summary, iteration
            )

            # Call model with focused context. Force the tool-only schema
            # (see FORCE_TOOL_CALL_MARKER) until this task has made at
            # least one real tool call — after that, later turns (a
            # multi-call task's next step, or its closing summary) are
            # free to answer in text again.
            call_system_prompt = system_prompt
            if step_expects_tool and successful_tool_calls == 0:
                call_system_prompt = system_prompt + FORCE_TOOL_CALL_MARKER
            try:
                message = await self.completion_fn(
                    focused_messages, call_system_prompt, tool_schemas or None
                )
                model_calls += 1
                if task_metrics:
                    task_metrics.model_calls = model_calls
            except Exception as exc:
                state.record_error(f"Model call failed: {exc}")
                await ctx.emit.emit(
                    EVT_TASK_MODIFIED, self.name, packet,
                    task_id=step.id, reason="model_error",
                    error=str(exc)[:200],
                )
                if not state.should_retry:
                    break
                # Inject retry prompt into base messages
                messages.append({
                    "role": "user",
                    "content": "The model call failed. Please retry with a different approach.",
                })
                continue

            content = (message.get("content") or "").strip()
            calls = message.get("tool_calls") or []

            # Recover a tool call the model wrote as JSON text instead of
            # using the real function-calling protocol (see
            # try_parse_faked_tool_call above). Only attempts this when the
            # model made no real tool call this turn — a genuine tool_calls
            # entry always wins.
            recovered_call = None
            if not calls and content:
                recovered_call = try_parse_faked_tool_call(content, valid_tool_names)
                if recovered_call is not None:
                    calls = [recovered_call]
                    await ctx.emit.emit(
                        EVT_TASK_MODIFIED, self.name, packet,
                        task_id=step.id, reason="recovered_faked_tool_call",
                        tool_name=recovered_call["function"]["name"],
                    )
                    if task_metrics:
                        task_metrics.recovery_prompts += 1

            # Record model output
            state.record_model_output(content)

            # Emit observable event for model response
            await ctx.emit.emit(
                EVT_TEXT_DELTA, self.name, packet,
                task_id=step.id,
                iteration=iteration,
                has_content=bool(content),
                has_tool_calls=bool(calls),
                content_preview=content[:200] if content else "",
            )

            # Check for empty output — use optimized recovery for small models
            if not content and not calls:
                if not state.should_retry:
                    break
                recovery_msg = EMPTY_OUTPUT_RECOVERY if self.small_model else (
                    "You provided an empty response. "
                    "Please either use a tool or provide a text response."
                )
                messages.append({"role": "user", "content": recovery_msg})
                if task_metrics:
                    task_metrics.malformed_outputs += 1
                    task_metrics.recovery_prompts += 1
                continue

            # A "done" marker (TASK COMPLETE:/DONE:/the small-model detector)
            # asserts success in prose. Trusting that at face value for a
            # step that expected a tool call and never got one is exactly
            # the false-positive confirmed live: the model wrote "TASK
            # COMPLETE: created snake.html" without ever calling write_file,
            # the loop accepted it on iteration 1 (tool_calls_made: 0), and
            # the run reported "VERIFIED... Objective achieved" with nothing
            # on disk. These markers must go through the exact same one-
            # corrective-retry gate as the generic prose-with-no-tool-call
            # path below, not bypass it by arriving prefixed differently.
            expects_action = context.get("action") in (
                "tool_use", "create", "modify", "execute", "search",
            )
            expected_tool_use = (
                not calls
                and (bool(context["likely_tools"]) or expects_action)
                and successful_tool_calls == 0
            )

            # Check for task completion — use optimized detection for small models
            completion = None
            if self.small_model:
                completion = detect_task_completion(content)

            claims_done = (
                (completion and completion["status"] == "done")
                or content.startswith("TASK COMPLETE:")
                or content.startswith("DONE:")
            )
            if claims_done and expected_tool_use and text_only_without_tool_count == 0:
                text_only_without_tool_count += 1
                messages.append({
                    "role": "user",
                    "content": (
                        "You reported the task as complete, but no tool was "
                        "actually called — nothing was created or changed. "
                        "If this task requires creating or modifying a file, "
                        "call write_file now with the complete content "
                        "before reporting completion again."
                    ),
                })
                if task_metrics:
                    task_metrics.recovery_prompts += 1
                continue

            if completion and completion["status"] == "done":
                final_output = completion["output"]
                completed = True
                break
            elif (
                completion
                and completion["status"] in ("stuck", "replan")
                and not refused_without_tool
            ):
                # Same recovery as the CANNOT COMPLETE branch below, which
                # this path would otherwise pre-empt entirely: with
                # small_model on, detect_task_completion() classifies
                # "CANNOT COMPLETE: ..." as `stuck` and breaks here first,
                # so a model that merely botched its arguments never got
                # the corrective retry. Deliberately NOT gated on "no tool
                # has succeeded": measured, the model made one successful
                # read, then gave up on edit_file against a file that did
                # not exist yet — one successful call is not evidence the
                # step was accomplished. Bounded to a single nudge instead,
                # so a genuine dead end still terminates.
                refused_without_tool = True
                state.record_error(content[:500])
                messages.append({
                    "role": "user",
                    "content": (
                        "Giving up is not accepted yet — this task has not "
                        "been accomplished. A tool rejecting your arguments "
                        "is recoverable, not a dead end: it told you exactly "
                        "which arguments it needs. If you were trying to "
                        "change a file that does not exist yet, CREATE it "
                        "with write_file and the COMPLETE content instead of "
                        "editing it. Issue that tool call now."
                    ),
                })
                if task_metrics:
                    task_metrics.recovery_prompts += 1
                continue
            elif completion and completion["status"] in ("stuck", "replan"):
                # The model explicitly said it's stuck/needs a replan — that
                # reason (in `content`) was previously discarded here, so
                # every downstream error report (the task-failed event, the
                # "unknown" decision reason surfaced in the UI) had nothing
                # to show. record_error() feeds the same field
                # `error=result.exhaustion_reason or result.error or
                # "unknown"` already reads from, so the model's own stated
                # reason shows up instead of a dead-end "unknown".
                state.record_error(content[:500] or "Model reported it was stuck")
                final_output = content
                completed = False
                break

            # Legacy completion detection (for standard models)
            if content.startswith("TASK COMPLETE:"):
                final_output = content[len("TASK COMPLETE:"):].strip()
                completed = True
                break

            if content.startswith("CANNOT COMPLETE:") or content.startswith("NEED REPLAN:"):
                # A give-up before ANY tool has successfully run is almost
                # never a real dead end — measured on a 3B local model: it
                # emitted create_file with bad arguments, read the
                # validation error, and immediately answered "CANNOT
                # COMPLETE: the provided arguments ... are incorrect",
                # which killed the whole step and wrote zero files. The
                # arguments being wrong is precisely the recoverable case,
                # so push the corrective shape back at it and let the
                # forced-tool schema above make the retry a real call
                # rather than more prose. Bounded: only while nothing has
                # executed, and only once, so a genuine dead end still
                # terminates instead of looping.
                if not refused_without_tool:
                    refused_without_tool = True
                    state.record_error(content[:500])
                    messages.append({
                        "role": "user",
                        "content": (
                            "Giving up is not accepted yet — this task has "
                            "not been accomplished. A tool rejecting your "
                            "arguments is recoverable, not a dead end: it "
                            "told you exactly which arguments it needs. If "
                            "you were trying to change a file that does not "
                            "exist yet, CREATE it with write_file and the "
                            "COMPLETE content instead of editing it. Issue "
                            "that tool call now."
                        ),
                    })
                    continue
                state.record_error(content[:500])
                final_output = content
                completed = False
                break

            # Also check the new markers
            if content.startswith("DONE:"):
                final_output = content[len("DONE:"):].strip()
                completed = True
                break
            if content.startswith("STUCK:"):
                state.record_error(content[:500])
                final_output = content
                completed = False
                break

            # Execute tool calls
            if calls:
                for call in calls:
                    tool_name = call.get("function", {}).get("name", "")
                    tool_args_str = call.get("function", {}).get("arguments", "")

                    # Track tool names
                    if tool_name and tool_name not in tool_names_used:
                        tool_names_used.append(tool_name)

                    # Check for repeated tool calls — use optimized recovery for small models
                    state.record_tool_call(tool_name, tool_args_str)
                    if state.repeated_tool_calls >= self.max_tool_repeats:
                        # Emit observable event for repeated calls
                        await ctx.emit.emit(
                            EVT_TASK_MODIFIED, self.name, packet,
                            task_id=step.id,
                            reason="repeated_tool_calls",
                            tool_name=tool_name,
                            repeat_count=state.repeated_tool_calls,
                        )
                        # Inject a message telling the model to try something else
                        if self.small_model:
                            recovery = REPEATED_TOOL_RECOVERY.format(
                                tool_name=tool_name,
                                count=state.repeated_tool_calls,
                            )
                        else:
                            recovery = (
                                f"You have called {tool_name} multiple times with the same arguments. "
                                "This is not productive. Please try a completely different approach "
                                "or report that you cannot complete the task."
                            )
                        messages.append({"role": "user", "content": recovery})
                        if task_metrics:
                            task_metrics.recovery_prompts += 1
                        break

                    # Execute the tool
                    try:
                        tool_args = json.loads(tool_args_str, strict=False) if tool_args_str else {}
                    except json.JSONDecodeError:
                        tool_args = {}
                        state.record_error(f"Malformed tool arguments for {tool_name}")

                    tool_start = time.time()
                    tool_result = await self._execute_tool(tool_name, tool_args, ctx)
                    tool_duration_ms = int((time.time() - tool_start) * 1000)
                    tool_success = getattr(tool_result, 'success', True)
                    if tool_success:
                        successful_tool_calls += 1

                    if adapter is not None:
                        adapter.emit_tool_started(
                            task_id=step.id,
                            tool_name=tool_name,
                            tool_args_preview=str(tool_args)[:200],
                            tool_args=_sanitize_tool_args_for_event(tool_args),
                        )
                        adapter.emit_tool_completed(
                            tool_name=tool_name,
                            task_id=step.id,
                            success=tool_success,
                            duration_ms=tool_duration_ms,
                            affected_files=_extract_affected_files(tool_args),
                        )

                    # Record tool result in summary (not full message history)
                    # Small models: tighter truncation
                    preview_limit = 150 if self.small_model else 300
                    result_preview = tool_result.to_message()[:preview_limit]
                    tool_results_summary.append(
                        f"[{tool_name}] {result_preview}"
                    )

                    # Add tool result to base messages for model continuity
                    messages.append({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [call],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": tool_result.to_message(),
                    })

                    # Emit observable event for tool execution
                    await ctx.emit.emit(
                        EVT_TASK_MODIFIED, self.name, packet,
                        task_id=step.id,
                        reason="tool_executed",
                        tool_name=tool_name,
                        tool_success=tool_success,
                        duration_ms=tool_duration_ms,
                    )
            else:
                # No tool calls — check for text-only completion
                # Use optimized detection for small models
                if self.small_model:
                    completion = detect_task_completion(content)
                    if completion and completion["status"] == "done":
                        final_output = completion["output"]
                        completed = True
                        break
                    elif completion and completion["status"] in ("stuck", "replan"):
                        final_output = content
                        completed = False
                        break

                # No tool calls, just text response
                messages.append({
                    "role": "assistant",
                    "content": content,
                })
                # If model just answered without tools, check if it's substantial
                # Small models: lower threshold for "substantial"
                substantial_threshold = 20 if self.small_model else 50
                if len(content) > substantial_threshold:
                    # A model that never once called a tool for a task that
                    # clearly expected one (it has likely_tools) is very
                    # often just describing the answer in prose/a code fence
                    # instead of acting on it — accepting that as "done" is
                    # exactly what let weaker/free models silently skip real
                    # file writes while still reporting success (the caller
                    # then finds nothing on disk and fails much later with a
                    # confusing "couldn't complete" message, instead of the
                    # model getting a chance to fix its own mistake here).
                    # Give it ONE corrective round-trip naming the mistake
                    # before accepting a text-only answer as final.
                    #
                    # `likely_tools` alone under-triggers this: it's the
                    # PLANNING model's own guess at which tools a step needs,
                    # and a live run confirmed a real planner can leave it
                    # empty on a step whose `action` is still "tool_use" —
                    # that step then sailed through as "completed" with
                    # tool_calls_made=0 (TaskExecutionObserver's fast path
                    # trusts `result.success` and never itself checks tool
                    # count). `step.action` has a firmer default ("tool_use")
                    # per its own field comment in orcha/core/packets.py, so
                    # OR it in as a second, more reliable signal.
                    expects_action = context.get("action") in (
                        "tool_use", "create", "modify", "execute", "search",
                    )
                    expected_tool_use = (
                        (bool(context["likely_tools"]) or expects_action)
                        and state.tool_call_count == 0
                    )
                    if expected_tool_use and text_only_without_tool_count == 0:
                        text_only_without_tool_count += 1
                        # Some weaker/free models, when they don't reliably use the
                        # API's native function-calling, fake a tool call by writing
                        # a JSON-shaped envelope (e.g. {"action": "tool", "name": ...,
                        # "arguments": ...}) as plain text instead. That JSON is never
                        # executed — it just looks like progress — so call this out
                        # explicitly rather than relying on the generic message below,
                        # which this exact failure mode has already ignored once.
                        looks_like_fake_tool_json = (
                            content.lstrip().startswith("{")
                            and '"action"' in content[:400]
                            and ('"name"' in content[:400] or '"tool"' in content[:400])
                        )
                        if looks_like_fake_tool_json:
                            corrective = (
                                "The JSON you just wrote as text (with \"action\"/\"name\"/\"arguments\") "
                                "is not a real tool call — writing JSON in your reply does not execute "
                                "anything, so nothing was created or changed. You must use the actual "
                                "function-calling mechanism to call write_file, not describe the call as "
                                "text. Call write_file now, for real, with the complete file content."
                            )
                        else:
                            corrective = (
                                "You responded with an explanation and/or code instead of calling a "
                                "tool. If this task requires creating or changing a file, call "
                                "write_file now with the complete content — do not describe it in "
                                "prose or a code block. If no tool is actually needed to complete "
                                "this task, say so explicitly."
                            )
                        messages.append({"role": "user", "content": corrective})
                        if task_metrics:
                            task_metrics.recovery_prompts += 1
                        continue
                    final_output = content
                    completed = True
                    break

        # Build result
        result = TaskExecutionResult.from_state(state, final_output, completed)
        result.success = completed and not state.exhaustion_reason
        if state.exhaustion_reason:
            result.exhaustion_reason = state.exhaustion_reason
        if state.error_history:
            result.error = state.error_history[-1]

        # Finalize metrics
        if task_metrics:
            task_metrics.tool_calls = state.tool_call_count
            task_metrics.tool_names = tool_names_used
            task_metrics.iterations = state.iteration
            task_metrics.success = result.success
            task_metrics.error = result.error
            task_metrics.exhausted = bool(state.exhaustion_reason)
            task_metrics.output_length = len(final_output)
            task_metrics.malformed_outputs = malformed_count
            task_metrics.prior_results_tokens_est = estimate_tokens(
                str(context.get("prior_results", ""))
            )

        return result

    def _build_focused_messages(
        self,
        base_messages: List[Dict[str, Any]],
        tool_results_summary: List[str],
        current_iteration: int,
    ) -> List[Dict[str, Any]]:
        """
        Build focused message list for the model.

        Architectural rule: Do NOT dump entire conversation history.
        Instead, use a focused window:
        1. The initial user message (task description) - always included
        2. Recent exchanges (last N turns) for continuity
        3. Tool results summary (not full outputs)

        This prevents context overflow and keeps the model task-focused.

        Small models: even tighter window — first message + last 2 messages.
        """
        if len(base_messages) <= 3:
            # Few messages: use them all
            return list(base_messages)

        # Keep: first message (task) + last N messages (recent context)
        # Small models: tighter window to fit context
        recent_count = 2 if self.small_model else 4
        first_message = base_messages[0]
        recent_messages = base_messages[-recent_count:]

        focused = [first_message]

        # Add tool results summary if we have any
        if tool_results_summary:
            # Small models: only last 2 tool results
            recent_tools = tool_results_summary[-(2 if self.small_model else 5):]
            summary = "Tool results so far:\n" + "\n".join(recent_tools)
            focused.append({"role": "user", "content": summary})

        focused.extend(recent_messages)

        return focused

    async def _execute_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        ctx: RunContext,
    ) -> Any:
        """Execute a tool through the existing Orcha tool system."""
        if self.executor is None:
            from ..capabilities.base import ToolResult
            return ToolResult.failure("no_executor", "No tool executor configured")

        try:
            # ToolExecutor.invoke is synchronous and takes the arguments as
            # kwargs, not a single dict — `.execute(name, args_dict)` doesn't
            # exist on it at all and would raise AttributeError on every
            # single call through this loop (research/multi-step tasks),
            # always falling into the except branch below.
            result = self.executor.invoke(tool_name, **(tool_args or {}))
            return result
        except Exception as exc:
            from ..capabilities.base import ToolResult
            return ToolResult.failure(
                "tool_execution_error",
                f"Tool {tool_name} failed: {exc}",
            )


# ── Task Context Builder (focused) ───────────────────────────────────────────

class FocusedTaskContextBuilder(Node):
    """
    Builds focused execution context for each task.

    Unlike the previous TaskContextBuilderNode, this one constructs a
    minimal, task-specific context that contains ONLY what is relevant
    for the current task. No unnecessary history is included.
    """

    name = "focused_task_context_builder"

    def __init__(
        self,
        name: str = "focused_task_context_builder",
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
            plan.status = "completed"
            return packet.fork(
                packet.kind,
                execution_plan=plan.model_dump(),
                task_complete=True,
            )

        step.status = "in_progress"
        plan.current_step_idx = plan.steps.index(step)

        # Build minimal context
        context_parts = [
            f"=== EXECUTING TASK: {step.title} ===",
            f"Task ID: {step.id}",
            f"Objective: {step.objective}",
            f"Instructions: {step.execution_instructions}",
            f"Expected outcome: {step.expected_outcome}",
            f"Verification: {step.verification_criteria}",
        ]

        # Add only necessary prior results (dependencies + required context)
        needed_ids = set(step.dependencies + step.required_context)
        if needed_ids:
            context_parts.append("\n--- Required prior outputs ---")
            for dep_id in step.dependencies:
                dep_step = plan.step_by_id(dep_id)
                if dep_step and dep_step.result:
                    context_parts.append(
                        f"[{dep_id}] {dep_step.title}: {dep_step.result.output[:300]}"
                    )

        # Add available tools
        if step.likely_tools:
            context_parts.append(f"\nAvailable tools: {', '.join(step.likely_tools)}")

        return packet.fork(
            packet.kind,
            execution_plan=plan.model_dump(),
            task_step_context="\n".join(context_parts),
            task_step_id=step.id,
        )


# ── Task Execution Observer ──────────────────────────────────────────────────

class TaskExecutionObserver(Node):
    """
    Observes task execution results and decides the next action.

    This observer evaluates whether the task completed successfully,
    whether to continue, retry, modify, replan, block, or complete.

    Decision logic:
    1. Fast path: success + no exhaustion -> CONTINUE
    2. LLM evaluation for ambiguous cases
    3. Rule-based fallback when no LLM available
    """

    name = "task_execution_observer"

    def __init__(
        self,
        name: str = "task_execution_observer",
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
        result_raw = packet.payload.get("task_execution_result")

        if plan_raw is None:
            return packet.fork(
                packet.kind,
                task_action="finalize",
                objective_met=False,
            )

        if result_raw is None:
            # A missing execution_plan means genuinely nothing to observe —
            # legitimate "we're done" (or never started). But a PRESENT,
            # not-yet-complete plan with no result means the execution loop
            # ran and produced nothing for a task that should have run (the
            # exact silent no-op this class of bug produces — see
            # TaskExecutionLoop's task_step_id fix). Surface that plainly
            # instead of finalizing as if it were an ordinary, expected
            # empty-plan case.
            plan_check = ExecutionPlan(**plan_raw) if isinstance(plan_raw, dict) else plan_raw
            if not plan_check.is_complete:
                return packet.fork(
                    packet.kind,
                    task_action="finalize",
                    objective_met=False,
                    error=(
                        "Task execution produced no result for an "
                        "incomplete plan — the run stopped without "
                        "actually executing the pending task."
                    ),
                )
            return packet.fork(
                packet.kind,
                task_action="finalize",
                objective_met=False,
            )

        plan = ExecutionPlan(**plan_raw) if isinstance(plan_raw, dict) else plan_raw
        result = TaskExecutionResult(**result_raw) if isinstance(result_raw, dict) else result_raw

        # Find the step that just executed
        step = plan.step_by_id(result.step_id)
        if step is None:
            return packet.fork(
                packet.kind,
                task_action="finalize",
                objective_met=False,
            )

        # Collect artifacts from this execution
        artifacts = TaskArtifactCollector.collect_from_result(result)

        # The step verifier runs immediately before this node and checks the
        # workspace, not the model's account of itself. Where the two
        # disagree, evidence wins:
        #
        #   reported failure + verified  -> the work landed; a malformed
        #       trailing tool call must not erase a file that is on disk.
        #       This is why a step that wrote stats.py was still marked
        #       failed and burned its whole retry budget re-writing it.
        #   reported success  + unverified -> the model claimed completion
        #       it cannot evidence. Under the free-choice tool envelope a 3B
        #       answered "stats.py created successfully" having called no
        #       tool at all; accepting that propagates a phantom result into
        #       every dependent step.
        verification = packet.payload.get("verification_result")
        if isinstance(verification, dict) and verification.get("step_id") == result.step_id:
            verified = bool(verification.get("verified"))
            if verified and not result.success:
                _exec_log.info(
                    "observer override step=%s: reported failure but "
                    "verification passed (%s) - treating as success",
                    result.step_id, verification.get("summary", ""),
                )
                result.success = True
                result.exhaustion_reason = None
                result.error = None
                if step.result is not None:
                    step.result.success = True
            elif not verified and result.success:
                # Only CONTRARY evidence may overturn a reported success —
                # not merely absent evidence. A step whose whole job is to
                # report a result writes no file, runs no command and
                # touches no tool, so the generic signals ("output matched
                # expected keywords", "expected tools were called") all miss
                # and the step looks unverified while being perfectly fine.
                # Observed: a completed 4-step run was reported "3/4 tasks
                # verified" purely because its final reporting step had
                # nothing deterministic to show.
                hard_fail_signals = {
                    "file_written", "file_exists", "test_result",
                    "build_result", "command_exit_code", "schema_valid",
                    "no_result",
                }
                contradicted = [
                    s for s in (verification.get("signals") or [])
                    if not s.get("passed")
                    and s.get("signal_type") in hard_fail_signals
                ]
                if contradicted:
                    issues = "; ".join(
                        s.get("description", "") for s in contradicted
                    ) or "unverified"
                    _exec_log.info(
                        "observer override step=%s: reported success but "
                        "evidence contradicts it (%s) - treating as failure",
                        result.step_id, issues,
                    )
                    result.success = False
                    result.error = f"Verification failed: {issues}"
                    if step.result is not None:
                        step.result.success = False
                else:
                    _exec_log.info(
                        "observer step=%s: unverified but nothing contradicts "
                        "the reported success - accepting (soft signals only)",
                        result.step_id,
                    )

        # Write what just failed into the run's working memory, so the next
        # attempt at this step starts knowing what has already been ruled
        # out instead of from a blank slate.
        if not result.success:
            last_call = (result.tool_summaries or [{}])[-1]
            plan.working_memory.record_failure(
                step_id=result.step_id,
                tool=str(last_call.get("tool") or ""),
                args=str(last_call.get("args_preview") or ""),
                error=str(result.error or result.exhaustion_reason or ""),
            )
        else:
            for path in _files_written_so_far(plan):
                plan.working_memory.record_fact(f"File exists: {path}")

        # Fast path: success with no exhaustion -> CONTINUE
        if result.success and not result.exhaustion_reason:
            step.status = "completed"
            observation = TaskObservation(
                step_id=step.id,
                step_result=TaskResult(
                    step_id=step.id,
                    output=result.output,
                    success=True,
                ),
                decision=TaskObservationDecision.CONTINUE,
                objective_met=True if plan.is_complete else None,
                artifacts=artifacts,
            )
            plan.observations.append(observation)

            # Emit task_observed event
            adapter = _get_or_create_event_adapter(packet, ctx, plan.objective)
            adapter.emit_task_observed(
                task_id=step.id,
                task_title=step.title,
                decision=TaskObservationDecision.CONTINUE.value,
                plan_progress=plan.progress,
            )

            if plan.is_complete:
                return packet.fork(
                    packet.kind,
                    execution_plan=plan.model_dump(),
                    task_action="verify",
                    objective_met=True,
                    task_observation=observation.model_dump(),
                    _event_adapter=adapter,
                )
            else:
                return packet.fork(
                    packet.kind,
                    execution_plan=plan.model_dump(),
                    task_action="next_task",
                    objective_met=False,
                    task_observation=observation.model_dump(),
                    _event_adapter=adapter,
                )

        # Ambiguous case: use LLM evaluation if available
        if self.completion_fn is not None:
            eval_observation = await self._llm_evaluate(step, result, plan)
            if eval_observation is not None:
                return await self._handle_decision(
                    eval_observation.decision, step, result, plan, artifacts, ctx, packet
                )

        # Rule-based fallback
        return await self._rule_based_evaluate(
            step, result, plan, artifacts, ctx, packet
        )

    async def _llm_evaluate(
        self,
        step: TaskStep,
        result: TaskExecutionResult,
        plan: ExecutionPlan,
    ) -> Optional[TaskObservation]:
        """Use LLM to evaluate task result quality."""
        try:
            # Format tool calls for the prompt
            tool_calls_str = "\n".join(
                f"- {t.get('tool', 'unknown')}: {t.get('args_preview', '')[:100]}"
                for t in result.tool_summaries
            ) or "No tool calls"

            prompt = TASK_OBSERVER_EVAL_PROMPT.format(
                task_objective=step.objective,
                expected_outcome=step.expected_outcome,
                verification_criteria=step.verification_criteria,
                actual_output=result.output[:2000] if result.output else "No output",
                tool_calls=tool_calls_str,
            )

            response = await self.completion_fn(
                [{"role": "user", "content": prompt}],
                "You are an objective evaluator. Be strict and precise.",
                None,
            )

            content = (response.get("content") or "").strip()

            # Parse decision
            decision = self._parse_decision(content)
            _exec_log.info(
                "observer llm-eval step=%s parsed=%s raw=%r",
                step.id,
                decision.value if decision else None,
                content[:200],
            )
            if decision is None:
                return None

            # Parse feedback
            feedback = None
            for line in content.split("\n"):
                if line.upper().startswith("FEEDBACK:"):
                    feedback = line[9:].strip() or None
                    break

            # Parse reason
            reason = ""
            for line in content.split("\n"):
                if line.upper().startswith("REASON:"):
                    reason = line[7:].strip()
                    break

            return TaskObservation(
                step_id=step.id,
                step_result=TaskResult(
                    step_id=step.id,
                    output=result.output,
                    success=result.success,
                    error=result.error,
                ),
                decision=decision,
                issues=[reason] if reason else [],
                feedback_for_retry=feedback,
            )

        except Exception:
            # LLM evaluation failed, fall through to rule-based
            return None

    # Decisions that only ever come from an EXPLICIT, well-formed answer.
    # COMPLETE is a claim about the WHOLE objective, not this one step, and
    # acting on it skips every remaining step in the plan - far too much
    # authority to hand to a word that appears in a sentence.
    _EXPLICIT_ONLY_DECISIONS = frozenset({TaskObservationDecision.COMPLETE})

    def _parse_decision(self, content: str) -> Optional[TaskObservationDecision]:
        """Parse the decision from an LLM evaluation response.

        Returns None when the response does not clearly name exactly one
        decision, which sends the caller to the deterministic rule-based
        path. That is the right outcome for unparseable output: a wrong
        decision here is strictly worse than no decision, because several of
        them (COMPLETE, BLOCK) end the run outright.

        Why this is stricter than the substring scan it replaces: every
        decision value is also an ordinary English word, and the evaluation
        prompt lists all six of them by name. A small model restating the
        menu, hedging ("this is not COMPLETE"), or writing "the file was
        never completed" all matched. Scanning in enum order made it worse -
        whichever decision comes first in the enum won, regardless of what
        the model meant. Observed live: a four-step run jumped straight to
        final verification after two steps, because the prose around a
        failed step contained the word "complete".
        """
        # 1. Explicit, well-formed answer: "DECISION: <VALUE>". Tolerate
        #    list markers, markdown emphasis and trailing prose, but require
        #    the value itself to be the first token after the colon.
        by_value = {d.value.upper(): d for d in TaskObservationDecision}
        for line in content.split("\n"):
            line_upper = line.upper().strip().lstrip("-*# ").strip()
            if not line_upper.startswith("DECISION:"):
                continue
            tail = line_upper[len("DECISION:"):].strip().strip("*`_ ")
            match = re.match(r"[A-Z_]+", tail)
            if match and match.group(0) in by_value:
                return by_value[match.group(0)]
            # A DECISION: line naming nothing recognizable is a malformed
            # answer, not an invitation to guess from the surrounding prose.
            return None

        # 2. No DECISION: line at all. Infer only from a whole-word mention,
        #    only when exactly one decision is named, and never for the
        #    decisions that terminate the run.
        content_upper = content.upper()
        found = {
            d
            for d in TaskObservationDecision
            if d not in self._EXPLICIT_ONLY_DECISIONS
            and re.search(
                r"(?<![A-Z])" + d.value.upper() + r"(?![A-Z])", content_upper
            )
        }
        if len(found) == 1:
            return next(iter(found))

        return None


    # Exhaustion reasons that describe something going wrong INSIDE this
    # pipeline - the model timed out, produced garbage, looped, or ran out of
    # iterations. None of them is an external blocker, and all of them are
    # what the retry/modify/replan budget exists to absorb.
    _TRANSIENT_EXHAUSTION = frozenset({
        "timeout",
        "consecutive_model_errors",
        "empty_model_outputs",
    })
    _MODEL_SHAPED_EXHAUSTION = frozenset({
        "repeated_tool_calls",
        "malformed_model_outputs",
        "max_iterations_exceeded",
    })

    def _internal_failure_decision(
        self,
        result: TaskExecutionResult,
        plan: ExecutionPlan,
    ) -> Optional[TaskObservationDecision]:
        """The recovery decision an internal failure warrants, or None.

        None means "no grounds to override a BLOCK": either the recovery
        budget is spent, or the failure carries no recognized internal
        exhaustion reason and so might genuinely be external. Never returns
        BLOCK, which is what makes the BLOCK downgrade non-recursive.
        """
        if plan.replan_count >= plan.max_replans:
            return None
        if result.exhaustion_reason in self._TRANSIENT_EXHAUSTION:
            return TaskObservationDecision.RETRY
        if result.exhaustion_reason in self._MODEL_SHAPED_EXHAUSTION:
            return TaskObservationDecision.MODIFY
        return None

    async def _rule_based_evaluate(
        self,
        step: TaskStep,
        result: TaskExecutionResult,
        plan: ExecutionPlan,
        artifacts: List[TaskArtifact],
        ctx: RunContext,
        packet: OrchaPacket,
    ) -> OrchaPacket:
        """Rule-based evaluation when LLM is not available."""
        decision = TaskObservationDecision.CONTINUE
        feedback = None

        if result.success:
            decision = TaskObservationDecision.CONTINUE
        elif result.exhaustion_reason in self._TRANSIENT_EXHAUSTION:
            if plan.replan_count < plan.max_replans:
                decision = TaskObservationDecision.RETRY
                feedback = f"Previous attempt failed due to: {result.exhaustion_reason}"
            else:
                decision = TaskObservationDecision.REPLAN
        elif result.exhaustion_reason in self._MODEL_SHAPED_EXHAUSTION:
            if plan.replan_count < plan.max_replans:
                decision = TaskObservationDecision.MODIFY
                feedback = (
                    f"Task failed with: {result.exhaustion_reason}. "
                    "Try a different approach or break into smaller steps."
                )
            else:
                decision = TaskObservationDecision.REPLAN
        else:
            # Default: replan if budget allows
            if plan.replan_count < plan.max_replans:
                decision = TaskObservationDecision.REPLAN
            else:
                decision = TaskObservationDecision.BLOCK

        observation = TaskObservation(
            step_id=step.id,
            step_result=TaskResult(
                step_id=step.id,
                output=result.output,
                success=result.success,
                error=result.error,
            ),
            decision=decision,
            issues=[result.error] if result.error else [],
            feedback_for_retry=feedback,
            artifacts=artifacts,
        )
        plan.observations.append(observation)

        return await self._handle_decision(
            decision, step, result, plan, artifacts, ctx, packet
        )

    async def _handle_decision(
        self,
        decision: TaskObservationDecision,
        step: TaskStep,
        result: TaskExecutionResult,
        plan: ExecutionPlan,
        artifacts: List[TaskArtifact],
        ctx: RunContext,
        packet: OrchaPacket,
    ) -> OrchaPacket:
        """Route to the appropriate action based on the decision."""
        observation = TaskObservation(
            step_id=step.id,
            step_result=TaskResult(
                step_id=step.id,
                output=result.output,
                success=result.success,
                error=result.error,
            ),
            decision=decision,
            artifacts=artifacts,
        )

        # Emit task_observed event
        adapter = _get_or_create_event_adapter(packet, ctx, plan.objective)
        adapter.emit_task_observed(
            task_id=step.id,
            task_title=step.title,
            decision=observation.decision.value,
            plan_progress=plan.progress,
        )

        # The observer's decision is the single most consequential routing
        # choice in the pipeline — BLOCK and COMPLETE both end the run — and
        # it was previously invisible, so "the run stopped after step N" gave
        # no way to tell a deliberate decision from a misparse. One INFO line
        # per decision, alongside the per-step outcome line above.
        _exec_log.info(
            "observer decision step=%s -> %s (success=%s, exhaustion=%s, "
            "replans=%d/%d, progress=%.2f)",
            step.id, decision.value, result.success,
            result.exhaustion_reason, plan.replan_count, plan.max_replans,
            plan.progress,
        )

        if decision == TaskObservationDecision.CONTINUE:
            step.status = "completed"
            if plan.is_complete:
                return packet.fork(
                    packet.kind,
                    execution_plan=plan.model_dump(),
                    task_action="verify",
                    objective_met=True,
                    task_observation=observation.model_dump(),
                )
            else:
                return packet.fork(
                    packet.kind,
                    execution_plan=plan.model_dump(),
                    task_action="next_task",
                    objective_met=False,
                    task_observation=observation.model_dump(),
                )

        elif decision == TaskObservationDecision.RETRY:
            return packet.fork(
                packet.kind,
                execution_plan=plan.model_dump(),
                task_action="retry",
                objective_met=False,
                retry_reason=result.exhaustion_reason or "transient_failure",
                task_observation=observation.model_dump(),
            )

        elif decision == TaskObservationDecision.MODIFY:
            return packet.fork(
                packet.kind,
                execution_plan=plan.model_dump(),
                task_action="modify",
                objective_met=False,
                task_observation=observation.model_dump(),
            )

        elif decision == TaskObservationDecision.REPLAN:
            return packet.fork(
                packet.kind,
                execution_plan=plan.model_dump(),
                task_action="replan",
                objective_met=False,
                replan_reason=result.error or result.exhaustion_reason or "structural_change_needed",
                task_observation=observation.model_dump(),
            )

        elif decision == TaskObservationDecision.BLOCK:
            # BLOCK asserts an EXTERNAL condition prevents progress - no
            # network, missing credentials, a permission denial. It ends the
            # run immediately, so like COMPLETE it is checked against the
            # evidence rather than taken on the evaluator's word.
            #
            # The failure that prompted this: a 3B evaluator returned BLOCK
            # for a step whose real problem was "the model called edit_file
            # with no arguments and then ran out of iterations". That is
            # internal and recoverable - exactly what MODIFY and the retry
            # budget exist for - but BLOCK bypassed both and finalized the
            # run with replans at 0/3, every recovery mechanism unused.
            internal = self._internal_failure_decision(result, plan)
            if internal is not None:
                _exec_log.info(
                    "observer BLOCK downgraded to %s step=%s "
                    "(exhaustion=%s, replans=%d/%d) - internal failure, "
                    "recovery budget still available",
                    internal.value, step.id, result.exhaustion_reason,
                    plan.replan_count, plan.max_replans,
                )
                # internal is never BLOCK, so this cannot recurse further.
                return await self._handle_decision(
                    internal, step, result, plan, artifacts, ctx, packet
                )
            return packet.fork(
                packet.kind,
                execution_plan=plan.model_dump(),
                task_action="finalize",
                objective_met=False,
                block_reason=step.metadata.get("blocker", "External blocker"),
                task_observation=observation.model_dump(),
            )

        elif decision == TaskObservationDecision.COMPLETE:
            # An early-completion claim is the evaluator asserting that the
            # WHOLE objective is already satisfied, which skips every
            # remaining step and jumps to final verification. Honour it only
            # when the plan agrees there is nothing left to do.
            #
            # This is deliberately defence in depth, not redundancy with the
            # stricter _parse_decision. That parser reduces how often a
            # bogus COMPLETE is produced; this gate bounds the damage when
            # one is produced anyway - including by a large model that
            # genuinely means it but is simply wrong about the plan state.
            # The observer sees one step's output; the plan knows what is
            # still pending, and the plan wins.
            step.status = "completed"
            unfinished = plan.unfinished_steps
            if unfinished:
                remaining = ", ".join(s.id for s in unfinished[:8])
                return packet.fork(
                    packet.kind,
                    execution_plan=plan.model_dump(),
                    task_action="next_task",
                    objective_met=False,
                    early_complete_rejected=(
                        "Evaluator reported COMPLETE for step "
                        f"{step.id}, but {len(unfinished)} step(s) remain "
                        f"unfinished [{remaining}] - continuing execution"
                    ),
                    task_observation=observation.model_dump(),
                )
            return packet.fork(
                packet.kind,
                execution_plan=plan.model_dump(),
                task_action="verify",
                objective_met=True,
                early_complete=True,
                task_observation=observation.model_dump(),
            )

        # Default: finalize
        return packet.fork(
            packet.kind,
            execution_plan=plan.model_dump(),
            task_action="finalize",
            objective_met=False,
            task_observation=observation.model_dump(),
        )


# ── Task Retry Handler ───────────────────────────────────────────────────────

class TaskRetryHandler(Node):
    """
    Handles task retries by modifying the task context or plan.

    When a task fails, this node decides whether to:
    1. Retry the same task with additional context (RETRY)
    2. Modify the task instructions and retry (MODIFY)
    3. Skip the task and continue (REPLAN)
    4. Delegate to adaptive replanner for structural changes
    """

    name = "task_retry_handler"

    def __init__(
        self,
        name: str = "task_retry_handler",
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
        action = packet.payload.get("task_action", "retry")
        retry_reason = packet.payload.get("retry_reason", "")
        observation_raw = packet.payload.get("task_observation")

        if plan_raw is None:
            return packet.fork(packet.kind, task_action="finalize")

        plan = ExecutionPlan(**plan_raw) if isinstance(plan_raw, dict) else plan_raw

        # Extract feedback from observation if available
        feedback = None
        if observation_raw:
            observation = TaskObservation(**observation_raw) if isinstance(observation_raw, dict) else observation_raw
            feedback = observation.feedback_for_retry

        # Get adapter for event emission
        adapter = _get_or_create_event_adapter(packet, ctx, plan.objective)

        # Find the step being retried for event emission
        retry_step_id = ""
        retry_step_title = ""
        for step in plan.steps:
            if step.status == "failed":
                retry_step_id = step.id
                retry_step_title = step.title
                break

        if action == "retry":
            adapter.emit_retry_started(
                task_id=retry_step_id,
                task_title=retry_step_title,
                retry_reason=retry_reason,
                retry_count=plan.replan_count,
            )
            return await self._handle_retry(plan, retry_reason, feedback, packet, adapter)

        elif action == "modify":
            adapter.emit_retry_started(
                task_id=retry_step_id,
                task_title=retry_step_title,
                retry_reason=feedback or "modify",
                retry_count=plan.replan_count,
            )
            return await self._handle_modify(plan, feedback, packet, adapter)

        elif action == "replan":
            adapter.emit_replanning_started(
                reason=retry_reason or "structural_change_needed",
                replan_count=plan.replan_count,
            )
            return await self._handle_replan(plan, packet, adapter)

        # Default: finalize
        return packet.fork(
            packet.kind,
            execution_plan=plan.model_dump(),
            task_action="finalize",
            _event_adapter=adapter,
        )

    async def _handle_retry(
        self,
        plan: ExecutionPlan,
        retry_reason: str,
        feedback: Optional[str],
        packet: OrchaPacket,
        adapter: Optional[EventStreamAdapter] = None,
    ) -> OrchaPacket:
        """Retry the same task with additional context."""
        # Find the failed step
        failed_step = None
        for step in plan.steps:
            if step.status == "failed":
                failed_step = step
                break

        if failed_step and plan.replan_count < plan.max_replans:
            # Reset the step for retry
            failed_step.status = "pending"
            failed_step.result = None

            # Inject feedback into execution instructions
            if feedback:
                original_instructions = failed_step.execution_instructions
                failed_step.execution_instructions = (
                    f"{original_instructions}\n\n"
                    f"ADDITIONAL CONTEXT FROM OBSERVER:\n"
                    f"{feedback}"
                )
                failed_step.metadata["retry_feedback_injected"] = True
                failed_step.metadata["original_instructions"] = original_instructions

            plan.replan_count += 1

            return packet.fork(
                packet.kind,
                execution_plan=plan.model_dump(),
                task_action="retry_same",
                retry_count=plan.replan_count,
                _event_adapter=adapter,
            )

        # Cannot retry, finalize
        return packet.fork(
            packet.kind,
            execution_plan=plan.model_dump(),
            task_action="finalize",
            _event_adapter=adapter,
        )

    async def _handle_modify(
        self,
        plan: ExecutionPlan,
        feedback: Optional[str],
        packet: OrchaPacket,
        adapter: Optional[EventStreamAdapter] = None,
    ) -> OrchaPacket:
        """Modify the task instructions and retry."""
        # Find the failed step
        failed_step = None
        for step in plan.steps:
            if step.status == "failed":
                failed_step = step
                break

        if failed_step and plan.replan_count < plan.max_replans:
            # Store original instructions for reference
            original_instructions = failed_step.execution_instructions

            # Build modified instructions
            if feedback:
                failed_step.execution_instructions = (
                    f"{original_instructions}\n\n"
                    f"MODIFIED APPROACH (previous attempt failed):\n"
                    f"{feedback}\n\n"
                    f"Try a different approach than before."
                )
            else:
                failed_step.execution_instructions = (
                    f"{original_instructions}\n\n"
                    f"The previous approach failed. Try a different approach."
                )

            failed_step.status = "pending"
            failed_step.result = None
            failed_step.metadata["modified"] = True
            failed_step.metadata["original_instructions"] = original_instructions
            plan.replan_count += 1

            return packet.fork(
                packet.kind,
                execution_plan=plan.model_dump(),
                task_action="retry_same",
                retry_count=plan.replan_count,
                task_modified=True,
                _event_adapter=adapter,
            )

        # Cannot modify, finalize
        return packet.fork(
            packet.kind,
            execution_plan=plan.model_dump(),
            task_action="finalize",
            _event_adapter=adapter,
        )

    async def _handle_replan(
        self,
        plan: ExecutionPlan,
        packet: OrchaPacket,
        adapter: Optional[EventStreamAdapter] = None,
    ) -> OrchaPacket:
        """Skip failed steps and continue."""
        for step in plan.steps:
            if step.status == "failed":
                step.status = "skipped"
        plan.replan_count += 1

        return packet.fork(
            packet.kind,
            execution_plan=plan.model_dump(),
            task_action="next_task",
            replan_count=plan.replan_count,
            _event_adapter=adapter,
        )


# ── Task Final Verifier ──────────────────────────────────────────────────────

class TaskFinalVerifier(Node):
    """
    Final verification of the complete execution plan.

    Uses the enhanced verification layer:
    1. Runs per-step deterministic verification
    2. Falls back to LLM evaluation when deterministic is insufficient
    3. Performs final objective verification
    4. Creates recovery tasks if verification fails
    5. Generates the user-facing response

    This distinguishes "the model said it completed" from "the objective was actually verified."
    """

    name = "task_final_verifier"

    def __init__(
        self,
        name: str = "task_final_verifier",
        completion_fn: Optional[Callable] = None,
        timeout_s: Optional[float] = 120.0,
        retries: int = 0,
    ) -> None:
        self.name = name
        self.timeout_s = timeout_s
        self.retries = retries
        self.completion_fn = completion_fn
        self._step_verifier = TaskStepVerifier(
            completion_fn=completion_fn, timeout_s=timeout_s
        )
        self._objective_verifier = ObjectiveVerifier(
            completion_fn=completion_fn, timeout_s=timeout_s
        )
        self._recovery_planner = RecoveryPlanner(
            completion_fn=completion_fn, timeout_s=timeout_s
        )
        self._response_generator = FinalResponseGenerator(
            completion_fn=completion_fn, timeout_s=timeout_s
        )

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        plan_raw = packet.payload.get("execution_plan")
        if plan_raw is None:
            return packet.fork(
                packet.kind,
                verification_result="No execution plan found",
                objective_met=False,
            )

        plan = ExecutionPlan(**plan_raw) if isinstance(plan_raw, dict) else plan_raw

        # Get adapter for event emission
        adapter = _get_or_create_event_adapter(packet, ctx, plan.objective)
        adapter.emit_verification_started(
            task_id="objective",
            task_title="Final Objective Verification",
        )

        # Phase 1: Per-step verification for completed steps
        verification_results: List[VerificationResult] = []
        for step in plan.steps:
            if step.status in ("completed", "failed") and step.result is not None:
                # Run per-step verification
                step_packet = packet.fork(
                    packet.kind,
                    execution_plan=plan.model_dump(),
                    step_id=step.id,
                )
                step_response = await self._step_verifier.run(step_packet, ctx)
                vr_raw = step_response.payload.get("verification_result")
                if vr_raw:
                    vr = VerificationResult(**vr_raw) if isinstance(vr_raw, dict) else vr_raw
                    verification_results.append(vr)

                    # Update step verification metadata
                    step.metadata["verified"] = vr.verified
                    step.metadata["verification_summary"] = vr.summary

        # Phase 2: Final objective verification
        obj_packet = packet.fork(
            packet.kind,
            execution_plan=plan.model_dump(),
            verification_results=[vr.model_dump() for vr in verification_results],
        )
        obj_response = await self._objective_verifier.run(obj_packet, ctx)
        obj_verification_raw = obj_response.payload.get("objective_verification")
        obj_verification = (
            ObjectiveVerificationResult(**obj_verification_raw)
            if isinstance(obj_verification_raw, dict)
            else obj_verification_raw
        )

        # Phase 3: Recovery planning (if verification failed)
        recovery_tasks = []
        if obj_verification and not obj_verification.objective_satisfied:
            recovery_packet = packet.fork(
                packet.kind,
                execution_plan=plan.model_dump(),
                objective_verification=obj_verification_raw,
            )
            recovery_response = await self._recovery_planner.run(recovery_packet, ctx)
            recovery_raw = recovery_response.payload.get("recovery_tasks", [])
            recovery_tasks = [
                RecoveryTask(**rt) if isinstance(rt, dict) else rt
                for rt in recovery_raw
            ]
            obj_verification.recovery_tasks = recovery_tasks

        # Phase 4: Generate final response
        response_packet = packet.fork(
            packet.kind,
            execution_plan=plan.model_dump(),
            objective_verification=obj_verification_raw,
            recovery_tasks=[rt.model_dump() for rt in recovery_tasks],
        )
        response = await self._response_generator.run(response_packet, ctx)

        # Build the final verification result
        completed = [s for s in plan.steps if s.status == "completed"]
        failed = [s for s in plan.steps if s.status == "failed"]
        skipped = [s for s in plan.steps if s.status == "skipped"]
        total = len(plan.steps)

        # Determine objective_met from objective verification
        objective_met = obj_verification.objective_satisfied if obj_verification else False

        # Build verification result string
        if objective_met:
            verification_result_str = (
                f"VERIFIED: All {total} tasks verified successfully "
                f"({len(completed)} completed, {len(skipped)} skipped). "
                f"Objective achieved."
            )
        else:
            verified_count = sum(1 for vr in verification_results if vr.verified)
            verification_result_str = (
                f"NOT VERIFIED: {verified_count}/{len(verification_results)} tasks verified. "
                f"{len(failed)} failed, {len(skipped)} skipped. "
                f"Objective NOT achieved."
            )
            if recovery_tasks:
                verification_result_str += (
                    f" {len(recovery_tasks)} recovery task(s) created."
                )

        # Emit verification_completed event
        adapter.emit_verification_completed(
            task_id="objective",
            verified=objective_met,
            pass_rate=sum(1 for vr in verification_results if vr.verified) / max(len(verification_results), 1),
            summary_text=verification_result_str,
        )

        # Update adapter with final state
        adapter.set_verification_result(obj_verification)
        adapter.set_final_response(response.payload.get("final_response"))
        adapter.update_plan(plan)

        return packet.fork(
            packet.kind,
            verification_result=verification_result_str,
            objective_met=objective_met,
            completed_tasks=len(completed),
            failed_tasks=len(failed),
            skipped_tasks=len(skipped),
            failed_task_ids=[s.id for s in failed],
            verification_results=[vr.model_dump() for vr in verification_results],
            objective_verification=obj_verification_raw,
            recovery_tasks=[rt.model_dump() for rt in recovery_tasks],
            final_response=response.payload.get("final_response"),
            _event_adapter=packet.payload.get("_event_adapter"),
            _event_stream_snapshot=(
                packet.payload.get("_event_adapter").get_snapshot()
                if packet.payload.get("_event_adapter") is not None
                else None
            ),
        )


# ── Adaptive Replanner ───────────────────────────────────────────────────────

ADAPTIVE_REPLAN_PROMPT = """You are re-planning a task graph based on execution results.

ORIGINAL OBJECTIVE:
{original_objective}

COMPLETED TASKS:
{completed_results}

CURRENT TASK RESULT:
{current_result}

ERRORS:
{errors}

ARTIFACTS:
{artifacts}

REMAINING TASKS:
{remaining_tasks}

Based on this information, modify ONLY the necessary portion of the plan.
NEVER remove or modify completed tasks — they are immutable.

Return a JSON object with:
{{
    "modified_steps": [<list of steps that need modification>],
    "added_steps": [<list of new steps to add>],
    "removed_step_ids": [<list of step IDs to remove — only pending/failed>],
    "kept_step_ids": [<list of step IDs that remain unchanged>],
    "reason": "explanation of changes"
}}

Rules:
1. Only modify steps that are pending or failed
2. Never modify completed steps
3. Add steps only if needed to achieve the objective
4. Remove steps only if they are no longer needed
5. Keep dependency graph valid (no cycles, all deps exist)
6. Be minimal — change as little as possible"""


DOWNSTREAM_VALIDATION_PROMPT = """You are evaluating whether a completed task's result affects downstream tasks.

COMPLETED TASK:
{completed_task_id}: {completed_task_title}
Objective: {completed_task_objective}
Result: {task_result}

REMAINING TASKS THAT DEPEND ON THIS TASK:
{dependent_tasks}

ORIGINAL OBJECTIVE:
{original_objective}

Consider:
1. Does the result match what downstream tasks expect?
2. Do downstream tasks need modification based on this result?
3. Are there new constraints or information that changes downstream?
4. Should any downstream task be skipped, modified, or replaced?

Reply with EXACTLY one of:
- VALID: All downstream tasks remain valid
- MODIFY_DOWNSTREAM: Some downstream tasks need modification
- SKIP_DOWNSTREAM: Some downstream tasks should be skipped
- BLOCKED: A dependency or external condition prevents progress

Then explain briefly.

Format:
DECISION: <one of VALID/MODIFY_DOWNSTREAM/SKIP_DOWNSTREAM/BLOCKED>
REASON: <brief explanation>
AFFECTED_TASKS: <comma-separated task IDs that need changes, or "none">
FEEDBACK: <optional feedback for affected tasks>"""


class AdaptiveReplanner(Node):
    """
    Adaptive replanner that modifies the plan graph based on execution results.

    Unlike the simple replanner that just skips failed tasks, this node:
    1. Receives the full context: objective, completed results, current result, errors, artifacts
    2. Uses LLM to decide what to modify
    3. Produces incremental plan modifications (add/modify/remove steps)
    4. Preserves completed task history
    5. Emits events describing plan changes
    """

    name = "adaptive_replanner"

    def __init__(
        self,
        name: str = "adaptive_replanner",
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
        observation_raw = packet.payload.get("task_observation")

        if plan_raw is None:
            return packet.fork(
                packet.kind,
                task_action="finalize",
                replan_success=False,
            )

        plan = ExecutionPlan(**plan_raw) if isinstance(plan_raw, dict) else plan_raw

        # Check replan budget
        if plan.replan_count >= plan.max_replans:
            return packet.fork(
                packet.kind,
                execution_plan=plan.model_dump(),
                task_action="finalize",
                replan_success=False,
                replan_exhausted=True,
            )

        # Extract context from observation
        observation = None
        if observation_raw:
            observation = (
                TaskObservation(**observation_raw)
                if isinstance(observation_raw, dict)
                else observation_raw
            )

        # Build replan request
        replan_request = self._build_replan_request(plan, observation, packet)

        # Try LLM-assisted replanning
        if self.completion_fn is not None:
            replan_result = await self._llm_replan(replan_request)
            if replan_result is not None:
                return await self._apply_replan(replan_result, plan, packet, ctx)

        # Fallback: simple skip of failed tasks
        return await self._simple_replan(plan, packet)

    def _build_replan_request(
        self,
        plan: ExecutionPlan,
        observation: Optional[TaskObservation],
        packet: OrchaPacket,
    ) -> ReplanRequest:
        """Build a structured replan request from the current state."""
        # Collect completed results
        completed_results = plan.get_completed_results()

        # Collect errors
        errors = []
        for step in plan.failed_steps:
            if step.result and step.result.error:
                errors.append(f"[{step.id}] {step.result.error}")

        # Collect artifacts
        artifacts = []
        for obs in plan.observations:
            artifacts.extend(obs.artifacts)

        # Current result from observation
        current_result = None
        if observation:
            current_result = observation.step_result

        return ReplanRequest(
            original_objective=plan.objective,
            current_plan=plan,
            completed_results=completed_results,
            current_result=current_result,
            errors=errors,
            artifacts=artifacts,
            workspace_state=(
                plan.request.workspace_state if plan.request else ""
            ),
            decision=observation.decision if observation else TaskObservationDecision.REPLAN,
            feedback=observation.feedback_for_retry or "" if observation else "",
        )

    async def _llm_replan(
        self, request: ReplanRequest
    ) -> Optional[ReplanResult]:
        """Use LLM to generate incremental plan modifications."""
        try:
            # Format context for the prompt
            completed_str = "\n".join(
                f"- [{r.step_id}] {'SUCCESS' if r.success else 'FAILED'}: {r.output[:200]}"
                for r in request.completed_results
            ) or "None"

            current_str = (
                f"{'SUCCESS' if request.current_result.success else 'FAILED'}: "
                f"{request.current_result.output[:300]}"
                if request.current_result
                else "None"
            )

            errors_str = "\n".join(f"- {e}" for e in request.errors) or "None"

            artifacts_str = "\n".join(
                f"- [{a.kind}] {a.description}" for a in request.artifacts
            ) or "None"

            remaining = request.current_plan.get_remaining_steps()
            remaining_str = "\n".join(
                f"- [{s.id}] {s.title} (status={s.status}, deps={s.dependencies})"
                for s in remaining
            ) or "None"

            prompt = ADAPTIVE_REPLAN_PROMPT.format(
                original_objective=request.original_objective,
                completed_results=completed_str,
                current_result=current_str,
                errors=errors_str,
                artifacts=artifacts_str,
                remaining_tasks=remaining_str,
            )

            response = await self.completion_fn(
                [{"role": "user", "content": prompt}],
                "You are a task planner. Output valid JSON only.",
                None,
            )

            content = (response.get("content") or "").strip()
            return self._parse_replan_result(content, request.current_plan)

        except Exception:
            return None

    def _parse_replan_result(
        self, content: str, plan: ExecutionPlan
    ) -> Optional[ReplanResult]:
        """Parse the LLM response into a ReplanResult."""
        try:
            # Try to extract JSON from the response
            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            elif "{" in content:
                # Try to find the JSON object
                start = content.index("{")
                # Find matching closing brace
                depth = 0
                for i in range(start, len(content)):
                    if content[i] == "{":
                        depth += 1
                    elif content[i] == "}":
                        depth -= 1
                        if depth == 0:
                            json_str = content[start : i + 1]
                            break

            data = json.loads(json_str)

            # Parse modified steps
            modified_steps = []
            for step_data in data.get("modified_steps", []):
                if isinstance(step_data, dict):
                    step_data.setdefault("status", "pending")
                    modified_steps.append(TaskStep(**step_data))

            # Parse added steps
            added_steps = []
            for step_data in data.get("added_steps", []):
                if isinstance(step_data, dict):
                    step_data.setdefault("status", "pending")
                    added_steps.append(TaskStep(**step_data))

            # Parse removed IDs
            removed_ids = [
                str(sid) for sid in data.get("removed_step_ids", [])
            ]

            # Parse kept IDs
            kept_ids = [
                str(sid) for sid in data.get("kept_step_ids", [])
            ]

            return ReplanResult(
                modified_steps=modified_steps,
                added_steps=added_steps,
                removed_step_ids=removed_ids,
                kept_step_ids=kept_ids,
                reason=data.get("reason", ""),
                replan_count=plan.replan_count + 1,
            )

        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    async def _apply_replan(
        self,
        replan_result: ReplanResult,
        plan: ExecutionPlan,
        packet: OrchaPacket,
        ctx: RunContext,
    ) -> OrchaPacket:
        """Apply the replan result to the plan."""
        # Apply incremental changes
        plan.apply_replan(replan_result)

        # Emit event
        await ctx.emit.emit(
            EVT_REPLAN,
            self.name,
            packet,
            replan_reason=replan_result.reason,
            modified_count=len(replan_result.modified_steps),
            added_count=len(replan_result.added_steps),
            removed_count=len(replan_result.removed_step_ids),
            replan_count=plan.replan_count,
        )

        return packet.fork(
            packet.kind,
            execution_plan=plan.model_dump(),
            task_action="next_task",
            replan_success=True,
            replan_count=plan.replan_count,
            replan_changes={
                "modified": [s.id for s in replan_result.modified_steps],
                "added": [s.id for s in replan_result.added_steps],
                "removed": replan_result.removed_step_ids,
                "kept": replan_result.kept_step_ids,
                "reason": replan_result.reason,
            },
        )

    async def _simple_replan(
        self,
        plan: ExecutionPlan,
        packet: OrchaPacket,
    ) -> OrchaPacket:
        """Fallback: skip failed tasks and continue."""
        for step in plan.steps:
            if step.status == "failed":
                step.status = "skipped"
        plan.replan_count += 1

        return packet.fork(
            packet.kind,
            execution_plan=plan.model_dump(),
            task_action="next_task",
            replan_count=plan.replan_count,
            replan_success=False,
        )


# ── Plan Integrity Checker ───────────────────────────────────────────────────

class PlanIntegrityChecker:
    """
    Validates plan graph integrity after replanning.

    Ensures:
    1. No circular dependencies
    2. All dependency references point to existing tasks
    3. Completed tasks are not modified
    4. No duplicate task IDs
    5. All tasks have valid objectives
    """

    @staticmethod
    def validate_after_replan(
        plan: ExecutionPlan,
        original_plan: Optional[ExecutionPlan] = None,
    ) -> List[str]:
        """
        Validate plan integrity after replanning.

        Returns list of error messages. Empty list means valid.
        """
        errors: List[str] = []

        # Check for empty plan
        if not plan.steps:
            errors.append("Plan has no tasks after replanning")
            return errors

        # Check for duplicate IDs
        seen_ids: Set[str] = set()
        all_ids: Set[str] = {s.id for s in plan.steps}
        for step in plan.steps:
            if step.id in seen_ids:
                errors.append(f"Duplicate task ID after replanning: {step.id}")
            seen_ids.add(step.id)

        # Check completed tasks are preserved
        if original_plan is not None:
            for orig_step in original_plan.completed_steps:
                new_step = plan.step_by_id(orig_step.id)
                if new_step is None:
                    errors.append(
                        f"Completed task {orig_step.id} was removed during replanning"
                    )
                elif new_step.status != "completed":
                    errors.append(
                        f"Completed task {orig_step.id} status changed to {new_step.status}"
                    )

        # Check dependencies
        for step in plan.steps:
            # Self-dependency
            if step.id in step.dependencies:
                errors.append(f"Task {step.id} depends on itself")

            # Non-existent dependencies
            for dep in step.dependencies:
                if dep not in all_ids:
                    errors.append(f"Task {step.id} depends on non-existent task: {dep}")

        # Check for circular dependencies
        cycle = PlanIntegrityChecker._detect_cycle(plan.steps)
        if cycle:
            errors.append(f"Circular dependency detected: {' -> '.join(cycle)}")

        # Check task objectives
        for step in plan.steps:
            obj = (step.objective or "").strip()
            if not obj or len(obj) < 5:
                errors.append(f"Task {step.id} has empty or too-short objective")
            if not (step.title or "").strip():
                errors.append(f"Task {step.id} has an empty title")

        return errors

    @staticmethod
    def _detect_cycle(steps: List[TaskStep]) -> Optional[List[str]]:
        """Detect circular dependencies using Kahn's algorithm."""
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
            start = cycle_nodes[0]
            path = [start]
            visited_cycle: Set[str] = set()
            current = start
            while current not in visited_cycle:
                visited_cycle.add(current)
                found_next = False
                for dep_step in steps:
                    if dep_step.id == current:
                        for nxt in dep_step.dependencies:
                            if nxt in {n for n in cycle_nodes}:
                                path.append(nxt)
                                current = nxt
                                found_next = True
                                break
                if not found_next:
                    break
            return path[:10]
        return None

    @staticmethod
    def check_replan_loop(
        plan: ExecutionPlan,
        max_replans: int = 3,
        recent_changes: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if replanning is in a loop.

        Returns (is_loop, reason).
        """
        # Check basic replan count
        if plan.replan_count >= max_replans:
            return True, f"Replan count ({plan.replan_count}) exceeded max ({max_replans})"

        # Check for repeated changes to same tasks
        if recent_changes and len(recent_changes) >= 3:
            # Get last 3 changes
            last_3 = recent_changes[-3:]
            affected_tasks = []
            for change in last_3:
                affected = set()
                affected.update(change.get("modified", []))
                affected.update(change.get("added", []))
                affected_tasks.append(frozenset(affected))

            # If same tasks affected 3 times in a row, it's a loop
            if len(affected_tasks) == 3 and affected_tasks[0] == affected_tasks[1] == affected_tasks[2]:
                return True, f"Repeated changes to tasks: {affected_tasks[0]}"

        return False, None


# ── Enhanced Task Execution Observer ─────────────────────────────────────────

class AdaptiveTaskExecutionObserver(TaskExecutionObserver):
    """
    Enhanced observer that:
    1. Validates downstream task viability after each result
    2. Uses LLM evaluation for both success and failure cases
    3. Emits explicit observation events
    4. Prevents infinite replanning loops

    Extends TaskExecutionObserver with adaptive capabilities.
    """

    name = "adaptive_task_execution_observer"

    def __init__(
        self,
        name: str = "adaptive_task_execution_observer",
        completion_fn: Optional[Callable] = None,
        timeout_s: Optional[float] = 120.0,
        retries: int = 0,
        max_replans: int = 3,
        recent_changes: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(name=name, completion_fn=completion_fn, timeout_s=timeout_s, retries=retries)
        self.max_replans = max_replans
        self.recent_changes = recent_changes or []

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        """
        Enhanced observation with downstream validation and loop detection.
        """
        plan_raw = packet.payload.get("execution_plan")
        result_raw = packet.payload.get("task_execution_result")

        if plan_raw is None or result_raw is None:
            return packet.fork(
                packet.kind,
                task_action="finalize",
                objective_met=False,
            )

        plan = ExecutionPlan(**plan_raw) if isinstance(plan_raw, dict) else plan_raw
        result = TaskExecutionResult(**result_raw) if isinstance(result_raw, dict) else result_raw

        # Find the step that just executed
        step = plan.step_by_id(result.step_id)
        if step is None:
            return packet.fork(
                packet.kind,
                task_action="finalize",
                objective_met=False,
            )

        # Check for replan loop BEFORE making any decision
        is_loop, loop_reason = PlanIntegrityChecker.check_replan_loop(
            plan, self.max_replans, self.recent_changes
        )
        if is_loop:
            await ctx.emit.emit(
                EVT_TASK_MODIFIED, self.name, packet,
                task_id=step.id,
                reason="replan_loop_detected",
                loop_reason=loop_reason,
            )
            # Force finalization
            return packet.fork(
                packet.kind,
                execution_plan=plan.model_dump(),
                task_action="finalize",
                objective_met=False,
                replan_loop_detected=True,
                loop_reason=loop_reason,
            )

        # Collect artifacts from this execution
        artifacts = TaskArtifactCollector.collect_from_result(result)

        # Fast path: success with no exhaustion -> validate downstream
        if result.success and not result.exhaustion_reason:
            # Validate downstream tasks
            downstream_valid = await self._validate_downstream(
                step, result, plan, ctx, packet
            )

            if downstream_valid == "VALID":
                # Standard CONTINUE
                step.status = "completed"
                observation = TaskObservation(
                    step_id=step.id,
                    step_result=TaskResult(
                        step_id=step.id,
                        output=result.output,
                        success=True,
                    ),
                    decision=TaskObservationDecision.CONTINUE,
                    objective_met=True if plan.is_complete else None,
                    artifacts=artifacts,
                )
                plan.observations.append(observation)

                await ctx.emit.emit(
                    EVT_TASK_COMPLETE, self.name, packet,
                    task_id=step.id,
                    decision="continue",
                    plan_progress=plan.progress,
                )

                if plan.is_complete:
                    return packet.fork(
                        packet.kind,
                        execution_plan=plan.model_dump(),
                        task_action="verify",
                        objective_met=True,
                        task_observation=observation.model_dump(),
                    )
                else:
                    return packet.fork(
                        packet.kind,
                        execution_plan=plan.model_dump(),
                        task_action="next_task",
                        objective_met=False,
                        task_observation=observation.model_dump(),
                    )
            elif downstream_valid == "MODIFY_DOWNSTREAM":
                # Downstream tasks need modification
                return packet.fork(
                    packet.kind,
                    execution_plan=plan.model_dump(),
                    task_action="modify_downstream",
                    objective_met=False,
                    task_observation=observation.model_dump() if 'observation' in dir() else None,
                )
            elif downstream_valid == "SKIP_DOWNSTREAM":
                # Some downstream tasks should be skipped
                return packet.fork(
                    packet.kind,
                    execution_plan=plan.model_dump(),
                    task_action="skip_downstream",
                    objective_met=False,
                )
            elif downstream_valid == "BLOCKED":
                return packet.fork(
                    packet.kind,
                    execution_plan=plan.model_dump(),
                    task_action="finalize",
                    objective_met=False,
                    block_reason="Downstream validation blocked progress",
                )

        # For failures or ambiguous cases, use parent logic
        return await super().run(packet, ctx)

    async def _validate_downstream(
        self,
        completed_step: TaskStep,
        result: TaskExecutionResult,
        plan: ExecutionPlan,
        ctx: RunContext,
        packet: OrchaPacket,
    ) -> str:
        """
        Validate that downstream tasks remain valid after this result.

        Returns: "VALID", "MODIFY_DOWNSTREAM", "SKIP_DOWNSTREAM", or "BLOCKED"
        """
        # Find tasks that depend on this one
        dependent_tasks = [
            s for s in plan.steps
            if completed_step.id in s.dependencies or completed_step.id in s.required_context
        ]

        if not dependent_tasks:
            return "VALID"

        # If no LLM available, use rule-based validation
        if self.completion_fn is None:
            return self._rule_based_downstream_validation(
                completed_step, result, dependent_tasks
            )

        # Use LLM for downstream validation
        try:
            dependent_str = "\n".join(
                f"- [{s.id}] {s.title}: {s.objective} (deps={s.dependencies})"
                for s in dependent_tasks
            ) or "None"

            prompt = DOWNSTREAM_VALIDATION_PROMPT.format(
                completed_task_id=completed_step.id,
                completed_task_title=completed_step.title,
                completed_task_objective=completed_step.objective,
                task_result=result.output[:1000] if result.output else "No output",
                dependent_tasks=dependent_str,
                original_objective=plan.objective,
            )

            response = await self.completion_fn(
                [{"role": "user", "content": prompt}],
                "You are a task graph validator. Be precise.",
                None,
            )

            content = (response.get("content") or "").strip()
            return self._parse_downstream_decision(content)

        except Exception:
            return self._rule_based_downstream_validation(
                completed_step, result, dependent_tasks
            )

    def _rule_based_downstream_validation(
        self,
        completed_step: TaskStep,
        result: TaskExecutionResult,
        dependent_tasks: List[TaskStep],
    ) -> str:
        """Rule-based downstream validation when LLM is not available."""
        # If the task succeeded, downstream is likely valid
        if result.success:
            return "VALID"

        # If the task failed, downstream may be affected
        if result.error and "not found" in result.error.lower():
            return "MODIFY_DOWNSTREAM"

        return "VALID"

    def _parse_downstream_decision(self, content: str) -> str:
        """Parse the downstream validation decision."""
        content_upper = content.upper()

        # Try explicit DECISION: line first
        for line in content.split("\n"):
            line_upper = line.upper().strip()
            if line_upper.startswith("DECISION:"):
                decision_str = line_upper[9:].strip()
                for valid in ("VALID", "MODIFY_DOWNSTREAM", "SKIP_DOWNSTREAM", "BLOCKED"):
                    if valid in decision_str:
                        return valid

        # Fallback: search for keywords
        if "MODIFY_DOWNSTREAM" in content_upper:
            return "MODIFY_DOWNSTREAM"
        if "SKIP_DOWNSTREAM" in content_upper:
            return "SKIP_DOWNSTREAM"
        if "BLOCKED" in content_upper:
            return "BLOCKED"
        if "VALID" in content_upper:
            return "VALID"

        return "VALID"


# ── Verification Layer Prompts ───────────────────────────────────────────────

TASK_STEP_VERIFICATION_PROMPT = """You are verifying whether a completed task actually achieved its objective.

TASK:
{step_id}: {step_title}
Objective: {step_objective}
Expected Outcome: {expected_outcome}
Verification Criteria: {verification_criteria}

ACTUAL RESULT:
Output: {task_output}
Tool Calls: {tool_calls}

DETERMINISTIC SIGNALS:
{deterministic_signals}

Did this task achieve its objective? Consider:
1. Does the output match the expected outcome?
2. Are the verification criteria satisfied?
3. Do the deterministic signals confirm success?
4. Are there any gaps between what was expected and what occurred?

Reply with EXACTLY one of:
- VERIFIED: The task objective was fully achieved
- PARTIALLY_VERIFIED: Some criteria met, some not
- NOT_VERIFIED: The task objective was not achieved

Format:
DECISION: <VERIFIED/PARTIALLY_VERIFIED/NOT_VERIFIED>
CONFIDENCE: <0.0-1.0>
REASON: <brief explanation>
ISSUES: <comma-separated list of issues, or "none">
SUGGESTIONS: <optional suggestions for improvement>"""


OBJECTIVE_VERIFICATION_PROMPT = """You are performing final verification of whether the overall objective was achieved.

ORIGINAL OBJECTIVE:
{objective}

TASK GRAPH SUMMARY:
{task_graph_summary}

TASK RESULTS:
{task_results}

COMPLETED ARTIFACTS:
{artifacts}

UNRESOLVED ERRORS:
{unresolved_errors}

Has the original objective been fully satisfied? Consider:
1. Are all required tasks completed?
2. Do the results match the original objective?
3. Are there any outstanding issues or limitations?
4. Would the user consider this work complete?

Reply with EXACTLY one of:
- OBJECTIVE_SATISFIED: The original objective has been fully achieved
- OBJECTIVE_PARTIALLY_SATISFIED: Some aspects achieved, some not
- OBJECTIVE_NOT_SATISFIED: The original objective was not achieved

Format:
DECISION: <OBJECTIVE_SATISFIED/OBJECTIVE_PARTIALLY_SATISFIED/OBJECTIVE_NOT_SATISFIED>
CONFIDENCE: <0.0-1.0>
REASON: <brief explanation>
COMPLETED_ITEMS: <comma-separated list of what was completed>
INCOMPLETE_ITEMS: <comma-separated list of what remains>
LIMITATIONS: <comma-separated list of known limitations>"""


# ── Deterministic Verifier ───────────────────────────────────────────────────


class DeterministicVerifier:
    """
    Verifies task results using deterministic signals first.

    Extracts signals from tool calls, artifact metadata, and result content
    to determine whether the task objective was achieved without needing LLM evaluation.
    """

    @staticmethod
    def extract_signals(
        step: TaskStep,
        result: TaskResult,
        artifacts: Optional[List[TaskArtifact]] = None,
    ) -> List[VerificationSignal]:
        """Extract deterministic verification signals from a completed task."""
        signals: List[VerificationSignal] = []

        if result is None:
            signals.append(VerificationSignal(
                signal_type="no_result",
                description="Task produced no result",
                passed=False,
                source_step_id=step.id,
            ))
            return signals

        # Signal 1: Task reported success
        signals.append(VerificationSignal(
            signal_type="task_success_flag",
            description="Task execution reported success",
            passed=result.success,
            expected="success=True",
            actual=f"success={result.success}",
            source_step_id=step.id,
        ))

        # Signal 2: Output is non-empty
        has_output = bool(result.output and result.output.strip())
        signals.append(VerificationSignal(
            signal_type="output_nonempty",
            description="Task produced non-empty output",
            passed=has_output,
            expected="non-empty output",
            actual=f"{len(result.output)} chars" if result.output else "empty",
            source_step_id=step.id,
        ))

        # Signal 3: No errors reported
        has_error = bool(result.error)
        signals.append(VerificationSignal(
            signal_type="no_error",
            description="Task completed without errors",
            passed=not has_error,
            expected="no error",
            actual=result.error or "none",
            source_step_id=step.id,
        ))

        # Signal 4: Tool calls were made (if task expected tools)
        if step.likely_tools:
            tool_names_used = set()
            for tc in result.tool_calls:
                tool_name = tc.get("tool", "")
                if tool_name:
                    tool_names_used.add(tool_name)
            expected_tools = set(step.likely_tools)
            tools_match = expected_tools.issubset(tool_names_used) if tool_names_used else False
            signals.append(VerificationSignal(
                signal_type="tools_used",
                description=f"Expected tools {sorted(expected_tools)} were called",
                passed=tools_match,
                expected=str(sorted(expected_tools)),
                actual=str(sorted(tool_names_used)),
                source_step_id=step.id,
            ))

        # Signal 5: File artifacts exist (from tool call history)
        if artifacts:
            file_artifacts = [a for a in artifacts if a.kind == "file" and a.location]
            for fa in file_artifacts:
                signals.append(VerificationSignal(
                    signal_type="file_artifact_recorded",
                    description=f"File artifact recorded: {fa.location}",
                    passed=True,  # If recorded, the tool reported success
                    expected=f"file at {fa.location}",
                    actual="recorded in artifacts",
                    source_step_id=step.id,
                    metadata={"location": fa.location},
                ))

        # Signal 6: Command artifacts recorded
        if artifacts:
            cmd_artifacts = [a for a in artifacts if a.kind == "command"]
            for ca in cmd_artifacts:
                signals.append(VerificationSignal(
                    signal_type="command_artifact_recorded",
                    description=f"Command artifact recorded: {ca.description}",
                    passed=True,
                    expected=f"command execution",
                    actual=ca.description,
                    source_step_id=step.id,
                ))

        # Signal 7: Output matches expected outcome (basic content check)
        if step.expected_outcome:
            expected_lower = step.expected_outcome.lower()
            output_lower = (result.output or "").lower()
            # Check if key words from expected outcome appear in output
            expected_words = [
                w.strip()
                for w in re.split(r'\s+', expected_lower)
                if len(w.strip()) > 3
            ]
            if expected_words:
                matched = sum(1 for w in expected_words if w in output_lower)
                match_ratio = matched / len(expected_words)
                signals.append(VerificationSignal(
                    signal_type="outcome_content_match",
                    description="Output content matches expected outcome keywords",
                    passed=match_ratio >= 0.5,
                    expected=step.expected_outcome[:200],
                    actual=result.output[:200] if result.output else "",
                    source_step_id=step.id,
                    metadata={"match_ratio": match_ratio},
                ))

        # Signal 8: TASK COMPLETE marker in output (model self-report)
        output_upper = (result.output or "").upper()
        has_complete_marker = "TASK COMPLETE" in output_upper
        signals.append(VerificationSignal(
            signal_type="complete_marker",
            description="Output contains TASK COMPLETE marker",
            passed=has_complete_marker,
            expected="TASK COMPLETE: in output",
            actual="found" if has_complete_marker else "not found",
            source_step_id=step.id,
        ))

        # Signal 9: Filesystem existence check (actual os.path.exists)
        if artifacts:
            file_artifacts = [a for a in artifacts if a.kind == "file" and a.location]
            for fa in file_artifacts:
                file_path = fa.location
                exists = False
                try:
                    exists = os.path.exists(file_path)
                except (OSError, ValueError):
                    pass
                signals.append(VerificationSignal(
                    signal_type="file_exists",
                    description=f"File exists on disk: {file_path}",
                    passed=exists,
                    expected=f"file at {file_path}",
                    actual="exists" if exists else "not found on disk",
                    source_step_id=step.id,
                    metadata={"location": file_path},
                ))

        # Signal 10: Command exit code (parsed from output or tool calls)
        exit_code = DeterministicVerifier._extract_exit_code(result)
        if exit_code is not None:
            signals.append(VerificationSignal(
                signal_type="command_exit_code",
                description=f"Command exit code: {exit_code}",
                passed=exit_code == 0,
                expected="exit code 0",
                actual=str(exit_code),
                source_step_id=step.id,
                metadata={"exit_code": exit_code},
            ))

        # Signal 11: Build result (detected from tool calls and output)
        build_result = DeterministicVerifier._extract_build_result(result)
        if build_result is not None:
            signals.append(VerificationSignal(
                signal_type="build_result",
                description=f"Build result: {build_result['status']}",
                passed=build_result["passed"],
                expected="successful build",
                actual=build_result["detail"],
                source_step_id=step.id,
                metadata=build_result.get("metadata", {}),
            ))

        # Signal 12: Test result (detected from tool calls and output)
        test_result = DeterministicVerifier._extract_test_result(result)
        if test_result is not None:
            signals.append(VerificationSignal(
                signal_type="test_result",
                description=f"Test result: {test_result['status']}",
                passed=test_result["passed"],
                expected="all tests passing",
                actual=test_result["detail"],
                source_step_id=step.id,
                metadata=test_result.get("metadata", {}),
            ))

        # Signal 13: Schema validation (JSON validity check)
        schema_result = DeterministicVerifier._extract_schema_result(result)
        if schema_result is not None:
            signals.append(VerificationSignal(
                signal_type="schema_valid",
                description=f"Schema validation: {schema_result['status']}",
                passed=schema_result["passed"],
                expected="valid schema/JSON",
                actual=schema_result["detail"],
                source_step_id=step.id,
            ))

        return signals

    @staticmethod
    def _extract_exit_code(result: TaskResult) -> Optional[int]:
        """Extract command exit code from tool calls or output."""
        # Check tool calls for exit_code in result metadata
        for tc in result.tool_calls:
            tool_name = tc.get("tool", "")
            if tool_name in ("run_command", "stream_output", "git"):
                # Check if exit_code is stored in the tool call record
                exit_code = tc.get("exit_code")
                if exit_code is not None:
                    return int(exit_code)
                # Check result string for exit code patterns
                result_str = tc.get("result", "")
                match = re.search(r'exit[_\s]?code[:\s]*(\d+)', result_str, re.I)
                if match:
                    return int(match.group(1))
                match = re.search(r'returncode[:\s]*(\d+)', result_str, re.I)
                if match:
                    return int(match.group(1))

        # Check output for exit code patterns
        output = result.output or ""
        match = re.search(r'exit[_\s]?code[:\s]*(\d+)', output, re.I)
        if match:
            return int(match.group(1))
        match = re.search(r'returncode[:\s]*(\d+)', output, re.I)
        if match:
            return int(match.group(1))

        return None

    @staticmethod
    def _extract_build_result(result: TaskResult) -> Optional[Dict[str, Any]]:
        """Extract build result from tool calls and output."""
        build_tools = {
            "npm", "yarn", "pnpm", "make", "cargo", "go", "gradle",
            "mvn", "dotnet", "cmake", "meson", "bun", "esbuild",
        }
        build_output_tools = {"run_command", "stream_output"}

        # Check if any build tool was used
        build_tool_used = False
        for tc in result.tool_calls:
            tool_name = tc.get("tool", "")
            args_preview = tc.get("args_preview", "")
            if tool_name in build_output_tools:
                # Check if the command is a build command
                args_lower = args_preview.lower()
                if any(bt in args_lower for bt in build_tools):
                    build_tool_used = True
                    break
            elif tool_name in build_tools:
                build_tool_used = True
                break

        if not build_tool_used:
            return None

        output = (result.output or "").lower()

        # Check for build failure patterns
        failure_patterns = [
            r'build\s+failed',
            r'compilation\s+error',
            r'error\[', r'error:',
            r'fatal\s+error',
            r'make:\s*\*\*\*',
            r'exit\s+status\s+[1-9]',
            r'unresolved\s+reference',
            r'type\s+error',
            r'syntax\s+error',
        ]
        for pattern in failure_patterns:
            if re.search(pattern, output):
                return {
                    "passed": False,
                    "status": "build_failed",
                    "detail": f"Build output contains failure pattern: {pattern}",
                    "metadata": {"pattern": pattern},
                }

        # Check for build success patterns
        success_patterns = [
            r'build\s+successful',
            r'build\s+complete',
            r'compiled?\s+successfully',
            r'successfully\s+built',
            r'ok\b.*\d+\s+(packages?\s+)?built',
            r'finished\s+in',
            r'done\s+in',
            r'build\s+time',
        ]
        for pattern in success_patterns:
            if re.search(pattern, output):
                return {
                    "passed": True,
                    "status": "build_succeeded",
                    "detail": f"Build output matches success pattern: {pattern}",
                    "metadata": {"pattern": pattern},
                }

        # If build tool was used but no clear success/failure pattern,
        # infer from task success flag
        return {
            "passed": result.success,
            "status": "build_inferred",
            "detail": f"Build tool used, inferred from task success={result.success}",
            "metadata": {"inferred": True},
        }

    @staticmethod
    def _extract_test_result(result: TaskResult) -> Optional[Dict[str, Any]]:
        """Extract test result from tool calls and output."""
        test_tools = {
            "pytest", "jest", "mocha", "vitest", "cargo", "go",
            "dotnet", "gradle", "mvn", "phpunit", "rspec", "junit",
            "nose", "unittest", "bun", "ava", "tap",
        }
        test_output_tools = {"run_command", "stream_output"}

        # Check if any test tool was used
        test_tool_used = False
        for tc in result.tool_calls:
            tool_name = tc.get("tool", "")
            args_preview = tc.get("args_preview", "")
            if tool_name in test_output_tools:
                args_lower = args_preview.lower()
                if any(tt in args_lower for tt in test_tools) or "test" in args_lower:
                    test_tool_used = True
                    break
            elif tool_name in test_tools:
                test_tool_used = True
                break

        if not test_tool_used:
            return None

        output = result.output or ""
        output_lower = output.lower()

        # Parse test results from output
        passed_count = 0
        failed_count = 0
        error_count = 0
        skipped_count = 0

        # pytest pattern: "X passed, Y failed, Z errors, W skipped"
        pytest_match = re.search(
            r'(\d+)\s+passed.*?(\d+)\s+failed.*?(\d+)\s+error', output, re.I
        )
        if pytest_match:
            passed_count = int(pytest_match.group(1))
            failed_count = int(pytest_match.group(2))
            error_count = int(pytest_match.group(3))
        else:
            pytest_match = re.search(r'(\d+)\s+passed', output, re.I)
            if pytest_match:
                passed_count = int(pytest_match.group(1))
            pytest_match = re.search(r'(\d+)\s+failed', output, re.I)
            if pytest_match:
                failed_count = int(pytest_match.group(1))
            pytest_match = re.search(r'(\d+)\s+errors?', output, re.I)
            if pytest_match:
                error_count = int(pytest_match.group(1))

        # Jest pattern: "Tests: X passed, Y total"
        jest_match = re.search(r'Tests:\s*(\d+)\s+passed.*?(\d+)\s+total', output, re.I)
        if jest_match:
            passed_count = int(jest_match.group(1))
            total = int(jest_match.group(2))
            # Jest doesn't always show failed count separately

        # Go test pattern: "ok" or "FAIL"
        go_match = re.search(r'^(ok|FAIL)\s+', output, re.M)
        if go_match:
            if go_match.group(1) == "ok":
                passed_count = max(passed_count, 1)
            else:
                failed_count = max(failed_count, 1)

        # Cargo test pattern
        cargo_match = re.search(r'test\s+result:\s*(\d+)\s+passed.*?(\d+)\s+failed', output, re.I)
        if cargo_match:
            passed_count = int(cargo_match.group(1))
            failed_count = int(cargo_match.group(2))

        # Check for test failure patterns
        # NOTE: Use case-SENSITIVE matching for FAIL/FAILED to avoid matching
        # count strings like "0 failed". Only match uppercase "FAIL" or "FAILED"
        # which indicates an actual test failure status, not a count.
        failure_patterns = [
            r'(?<![0-9])FAIL\b', r'(?<![0-9])FAILED\b', r'failed\s+test',
            r'assertion\s+error', r'expected\s+.*but\s+got',
            r'test\s+failures?', r'all\s+tests?\s+failed',
        ]
        has_failure_pattern = any(re.search(p, output, re.I) for p in failure_patterns)

        # Check for test success patterns
        success_patterns = [
            r'ALL\s+TESTS?\s+PASSED', r'all\s+tests?\s+passed',
            r'\d+\s+passed.*0\s+failed',
            r'OK\b.*\(\d+\s+tests?\)',
        ]
        has_success_pattern = any(re.search(p, output, re.I) for p in success_patterns)

        total_tests = passed_count + failed_count + error_count + skipped_count
        all_passed = failed_count == 0 and error_count == 0 and total_tests > 0

        # Priority: parsed counts take precedence over pattern matching.
        # If we parsed counts showing 0 failures, that's definitive success.
        if all_passed or has_success_pattern:
            return {
                "passed": True,
                "status": "tests_passed",
                "detail": f"Tests: {passed_count} passed, {failed_count} failed, {error_count} errors",
                "metadata": {
                    "passed": passed_count, "failed": failed_count,
                    "errors": error_count, "skipped": skipped_count,
                },
            }
        elif has_failure_pattern or failed_count > 0 or error_count > 0:
            return {
                "passed": False,
                "status": "tests_failed",
                "detail": f"Tests: {passed_count} passed, {failed_count} failed, {error_count} errors",
                "metadata": {
                    "passed": passed_count, "failed": failed_count,
                    "errors": error_count, "skipped": skipped_count,
                },
            }
        elif total_tests > 0:
            return {
                "passed": all_passed,
                "status": "tests_completed",
                "detail": f"Tests: {passed_count} passed, {failed_count} failed, {error_count} errors",
                "metadata": {
                    "passed": passed_count, "failed": failed_count,
                    "errors": error_count, "skipped": skipped_count,
                },
            }

        # Test tool was used but no parseable output - infer from task success
        return {
            "passed": result.success,
            "status": "tests_inferred",
            "detail": f"Test tool used, inferred from task success={result.success}",
            "metadata": {"inferred": True},
        }

    @staticmethod
    def _extract_schema_result(result: TaskResult) -> Optional[Dict[str, Any]]:
        """Validate output as JSON if it appears to be structured data."""
        output = (result.output or "").strip()
        if not output:
            return None

        # Only check if output looks like JSON
        if not (output.startswith("{") or output.startswith("[")):
            return None

        try:
            parsed = json.loads(output)
            # Basic schema validation: check it's a dict or list
            if isinstance(parsed, dict):
                return {
                    "passed": True,
                    "status": "valid_json_object",
                    "detail": f"Valid JSON object with {len(parsed)} keys",
                }
            elif isinstance(parsed, list):
                return {
                    "passed": True,
                    "status": "valid_json_array",
                    "detail": f"Valid JSON array with {len(parsed)} items",
                }
            else:
                return {
                    "passed": True,
                    "status": "valid_json",
                    "detail": f"Valid JSON value: {type(parsed).__name__}",
                }
        except json.JSONDecodeError as e:
            return {
                "passed": False,
                "status": "invalid_json",
                "detail": f"JSON parse error: {e.msg} at line {e.lineno}",
            }

    @staticmethod
    def compute_pass_rate(signals: List[VerificationSignal]) -> float:
        """Compute the ratio of signals that passed."""
        if not signals:
            return 0.0
        passed = sum(1 for s in signals if s.passed)
        return passed / len(signals)

    @staticmethod
    def has_blocking_failure(signals: List[VerificationSignal]) -> bool:
        """Check if any signal indicates a hard failure (not just incomplete)."""
        for s in signals:
            if s.signal_type == "task_success_flag" and not s.passed:
                return True
            if s.signal_type == "no_error" and not s.passed:
                return True
        return False


# ── Task Step Verifier ───────────────────────────────────────────────────────


class TaskStepVerifier(Node):
    """
    Per-step verification node that uses deterministic signals first,
    then falls back to LLM evaluation when deterministic signals are insufficient.

    This distinguishes "the model said it completed" from "the objective was actually verified."
    """

    name = "task_step_verifier"

    def __init__(
        self,
        name: str = "task_step_verifier",
        completion_fn: Optional[Callable] = None,
        timeout_s: Optional[float] = 120.0,
        retries: int = 0,
        workspace_roots: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.timeout_s = timeout_s
        self.retries = retries
        self.completion_fn = completion_fn
        self.workspace_roots = list(workspace_roots or [])

    def _file_signals(
        self,
        step: TaskStep,
        result: TaskResult,
    ) -> List[VerificationSignal]:
        """Ground truth: did the files this step claimed to write appear?

        Every other deterministic signal reads the step's own account of
        itself — "did it report success", "was the output non-empty". Those
        are self-reports, and a step that wrote a real file while ending on
        a malformed follow-up call reports failure. This checks the disk,
        which is the only signal in the set the model cannot be wrong about.
        """
        signals: List[VerificationSignal] = []
        if not self.workspace_roots:
            return signals
        root = self.workspace_roots[0]
        for call in result.tool_calls or []:
            if call.get("tool") not in _FILE_WRITING_TOOLS:
                continue
            for path in call.get("files") or []:
                candidate = path if os.path.isabs(path) else os.path.join(root, path)
                exists = os.path.isfile(candidate)
                signals.append(VerificationSignal(
                    signal_type="file_written",
                    description=f"File {path} exists on disk",
                    passed=exists,
                    expected=f"{path} exists",
                    actual="exists" if exists else "missing",
                    source_step_id=step.id,
                ))
        return signals

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        plan_raw = packet.payload.get("execution_plan")
        # The execution loop publishes `task_step_id`; this node was written
        # against `step_id` and never wired, so the mismatch was never hit.
        # Accept both rather than renaming a key other nodes already read.
        step_id = packet.payload.get("step_id") or packet.payload.get("task_step_id")

        if plan_raw is None or step_id is None:
            return packet.fork(
                packet.kind,
                verification_result=VerificationResult(
                    step_id="unknown",
                    verified=False,
                    issues=["Missing execution plan or step_id"],
                ).model_dump(),
            )

        plan = ExecutionPlan(**plan_raw) if isinstance(plan_raw, dict) else plan_raw
        step = plan.step_by_id(step_id)

        if step is None:
            return packet.fork(
                packet.kind,
                verification_result=VerificationResult(
                    step_id=step_id,
                    verified=False,
                    issues=[f"Step {step_id} not found in plan"],
                ).model_dump(),
            )

        result = step.result
        if result is None:
            return packet.fork(
                packet.kind,
                verification_result=VerificationResult(
                    step_id=step_id,
                    verified=False,
                    issues=["No result available for verification"],
                ).model_dump(),
            )

        # Collect artifacts for this step
        step_artifacts = []
        for obs in plan.observations:
            if obs.step_id == step_id:
                step_artifacts.extend(obs.artifacts)

        # Emit verification start
        await ctx.emit.emit(
            EVT_VERIFICATION_START, self.name, packet,
            step_id=step_id,
            step_title=step.title,
        )

        # Phase 1: Deterministic verification
        signals = DeterministicVerifier.extract_signals(step, result, step_artifacts)
        file_signals = self._file_signals(step, result)
        signals.extend(file_signals)
        pass_rate = DeterministicVerifier.compute_pass_rate(signals)
        has_blocking = DeterministicVerifier.has_blocking_failure(signals)
        # A step that put its files on disk did the work, whatever it said
        # about itself afterwards. Treat that as decisive so a malformed
        # trailing tool call cannot erase a completed write.
        files_all_present = bool(file_signals) and all(s.passed for s in file_signals)

        # Emit individual signals
        for signal in signals:
            await ctx.emit.emit(
                EVT_VERIFICATION_SIGNAL, self.name, packet,
                step_id=step_id,
                signal_type=signal.signal_type,
                passed=signal.passed,
                description=signal.description,
            )

        # Phase 2: Decide if LLM fallback is needed
        needs_llm = False
        llm_reason = ""

        if pass_rate < 0.5:
            needs_llm = True
            llm_reason = "Low deterministic pass rate"
        elif has_blocking:
            needs_llm = True
            llm_reason = "Blocking failure detected"
        elif step.verification_criteria and pass_rate < 1.0:
            needs_llm = True
            llm_reason = "Task has verification criteria that need qualitative assessment"
        elif not step.verification_criteria and not step.expected_outcome:
            # No verification criteria defined - trust deterministic signals
            pass  # Don't need LLM

        # Phase 3: LLM evaluation (if needed and available)
        llm_evaluation = None
        llm_confidence = None
        used_llm = False

        if needs_llm and self.completion_fn is not None:
            llm_result = await self._llm_evaluate(step, result, signals, pass_rate)
            if llm_result is not None:
                llm_evaluation = llm_result.get("evaluation", "")
                llm_confidence = llm_result.get("confidence")
                used_llm = True

        # Phase 4: Compute final verification verdict
        issues = [s.description for s in signals if not s.passed]

        if files_all_present and not has_blocking:
            # Disk evidence outranks every self-report and the LLM's opinion
            # of them: the artifacts this step exists to produce are there.
            verified = True
        elif has_blocking:
            verified = False
        elif used_llm and llm_confidence is not None:
            verified = llm_confidence >= 0.7
        elif pass_rate >= 0.7:
            verified = True
        else:
            verified = pass_rate >= 0.5 and not has_blocking

        # Build summary
        if verified:
            summary = f"Task {step_id} verified: {len(signals)} signals, {pass_rate:.0%} pass rate"
        else:
            summary = f"Task {step_id} NOT verified: {len(issues)} issues, {pass_rate:.0%} pass rate"

        verification_result = VerificationResult(
            step_id=step_id,
            verified=verified,
            signals=signals,
            llm_evaluation=llm_evaluation,
            llm_confidence=llm_confidence,
            issues=issues,
            summary=summary,
            deterministic_pass_rate=pass_rate,
            used_llm_fallback=used_llm,
        )

        # Emit verification complete
        await ctx.emit.emit(
            EVT_VERIFICATION_COMPLETE, self.name, packet,
            step_id=step_id,
            verified=verified,
            pass_rate=pass_rate,
            signal_count=len(signals),
            used_llm=used_llm,
        )

        return packet.fork(
            packet.kind,
            verification_result=verification_result.model_dump(),
            objective_met=verified,
        )

    async def _llm_evaluate(
        self,
        step: TaskStep,
        result: TaskResult,
        signals: List[VerificationSignal],
        pass_rate: float,
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to evaluate task verification when deterministic signals are insufficient."""
        if self.completion_fn is None:
            return None

        # Format deterministic signals for the prompt
        signal_lines = []
        for s in signals:
            status = "PASS" if s.passed else "FAIL"
            signal_lines.append(f"  [{status}] {s.signal_type}: {s.description}")
            if s.expected:
                signal_lines.append(f"         Expected: {s.expected}")
            if s.actual:
                signal_lines.append(f"         Actual: {s.actual}")
        signals_text = "\n".join(signal_lines) if signal_lines else "  No deterministic signals available"

        prompt = TASK_STEP_VERIFICATION_PROMPT.format(
            step_id=step.id,
            step_title=step.title,
            step_objective=step.objective,
            expected_outcome=step.expected_outcome or "Not specified",
            verification_criteria=step.verification_criteria or "Not specified",
            task_output=result.output[:1000] if result.output else "No output",
            tool_calls=json.dumps(result.tool_calls[:5], indent=2) if result.tool_calls else "None",
            deterministic_signals=signals_text,
        )

        try:
            response = await self.completion_fn(
                messages=[{"role": "user", "content": prompt}],
                system="You are a verification expert. Evaluate task completion objectively.",
                tools=None,
            )
            content = response.get("content", "")

            # Parse LLM response
            confidence = self._parse_confidence(content)
            evaluation = content

            return {
                "evaluation": evaluation,
                "confidence": confidence,
            }
        except Exception:
            return None

    @staticmethod
    def _parse_confidence(content: str) -> float:
        """Parse confidence value from LLM response."""
        match = re.search(r'CONFIDENCE:\s*(\d+\.?\d*)', content, re.I)
        if match:
            try:
                val = float(match.group(1))
                return max(0.0, min(1.0, val))
            except ValueError:
                pass

        content_upper = content.upper()
        if "VERIFIED" in content_upper and "NOT_VERIFIED" not in content_upper:
            return 0.9
        elif "NOT_VERIFIED" in content_upper:
            return 0.2
        elif "PARTIALLY_VERIFIED" in content_upper:
            return 0.5
        return 0.5


# ── Objective Verifier ───────────────────────────────────────────────────────


class ObjectiveVerifier(Node):
    """
    Final objective verification at the end of a full execution run.

    Distinguishes "all tasks completed" from "the objective was actually achieved."
    Uses both deterministic task results and optional LLM evaluation.
    """

    name = "objective_verifier"

    def __init__(
        self,
        name: str = "objective_verifier",
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
        verification_results_raw = packet.payload.get("verification_results", [])

        if plan_raw is None:
            return packet.fork(
                packet.kind,
                objective_verification=ObjectiveVerificationResult(
                    status="failed",
                    objective="unknown",
                    summary="No execution plan found",
                ).model_dump(),
            )

        plan = ExecutionPlan(**plan_raw) if isinstance(plan_raw, dict) else plan_raw
        verification_results = [
            VerificationResult(**vr) if isinstance(vr, dict) else vr
            for vr in verification_results_raw
        ]

        start_time = time.time()

        # Collect all artifacts
        all_artifacts = []
        for obs in plan.observations:
            all_artifacts.extend(obs.artifacts)

        # Collect unresolved errors
        unresolved_errors = []
        for step in plan.failed_steps:
            if step.result and step.result.error:
                unresolved_errors.append(f"[{step.id}] {step.result.error}")

        # Build task graph summary
        task_graph_summary = self._build_task_graph_summary(plan)

        # Build task results summary
        task_results_text = self._build_task_results_summary(plan, verification_results)

        # Artifacts summary
        artifact_locations = [a.location for a in all_artifacts if a.location]

        # Phase 1: Deterministic objective check
        deterministic_satisfied = self._deterministic_objective_check(plan, verification_results)

        # Phase 2: LLM evaluation (if available and deterministic is ambiguous)
        llm_satisfied = None
        llm_confidence = None
        completed_items = []
        incomplete_items = []
        limitations = []

        if self.completion_fn is not None:
            llm_result = await self._llm_evaluate(
                plan, task_graph_summary, task_results_text,
                artifact_locations, unresolved_errors
            )
            if llm_result is not None:
                llm_satisfied = llm_result.get("satisfied")
                llm_confidence = llm_result.get("confidence")
                completed_items = llm_result.get("completed_items", [])
                incomplete_items = llm_result.get("incomplete_items", [])
                limitations = llm_result.get("limitations", [])

        # Phase 3: Compute final verdict.
        # A concrete negative signal (a failed step, an incomplete plan, or a
        # per-step verifier that actually found problems) is ground truth and
        # must act as a hard veto — the objective-level LLM call above has no
        # access to the real artifacts (it only sees a text summary), so its
        # optimism can confirm an ambiguous deterministic pass but must never
        # overturn a deterministic failure into "satisfied". This is what let
        # a run with zero real tool calls (the model faked one in prose) get
        # reported as "VERIFIED: Objective achieved" purely because the final
        # verifier guessed from output length ("no artifacts to inspect...
        # but the output length strongly suggests all features are included").
        if not deterministic_satisfied:
            objective_satisfied = False
        elif llm_satisfied is not None:
            objective_satisfied = llm_satisfied
        else:
            objective_satisfied = deterministic_satisfied

        # Determine status
        if objective_satisfied:
            status = "verified"
            verified = True
        elif deterministic_satisfied and not objective_satisfied:
            status = "partially_verified"
            verified = False
        else:
            status = "failed"
            verified = False

        # Build incomplete items from failed steps if not provided by LLM
        if not incomplete_items:
            for step in plan.failed_steps:
                incomplete_items.append(f"{step.id}: {step.title} - {step.result.error if step.result else 'no result'}")

        # Build completed items from completed steps if not provided by LLM
        if not completed_items:
            for step in plan.completed_steps:
                completed_items.append(f"{step.id}: {step.title}")

        verification_duration = time.time() - start_time

        # Verify artifact existence on disk
        artifacts_verified = []
        artifacts_missing = []
        for loc in artifact_locations:
            try:
                if os.path.exists(loc):
                    artifacts_verified.append(loc)
                else:
                    artifacts_missing.append(loc)
            except (OSError, ValueError):
                artifacts_missing.append(loc)

        # Check file_exists signals from step verification results for additional info
        for vr in verification_results:
            for signal in vr.signals:
                if signal.signal_type == "file_exists" and signal.metadata.get("location"):
                    loc = signal.metadata["location"]
                    if signal.passed and loc not in artifacts_verified:
                        artifacts_verified.append(loc)
                    elif not signal.passed and loc not in artifacts_missing:
                        artifacts_missing.append(loc)

        # Build summary
        summary = self._build_summary(
            plan, status, objective_satisfied, completed_items,
            incomplete_items, unresolved_errors, limitations
        )

        result = ObjectiveVerificationResult(
            status=status,
            objective_satisfied=objective_satisfied,
            verified=verified,
            objective=plan.objective,
            task_results=verification_results,
            incomplete_items=incomplete_items,
            unresolved_errors=unresolved_errors,
            summary=summary,
            artifacts_verified=artifacts_verified,
            artifacts_missing=artifacts_missing,
            verification_duration_s=verification_duration,
        )

        # Emit verification complete
        await ctx.emit.emit(
            EVT_VERIFICATION_COMPLETE, self.name, packet,
            status=status,
            objective_satisfied=objective_satisfied,
            verified=verified,
            task_count=len(plan.steps),
            completed_count=len(plan.completed_steps),
            failed_count=len(plan.failed_steps),
        )

        # Emit objective met/not met
        if objective_satisfied:
            await ctx.emit.emit(
                EVT_OBJECTIVE_MET, self.name, packet,
                objective=plan.objective,
            )

        return packet.fork(
            packet.kind,
            objective_verification=result.model_dump(),
            objective_met=objective_satisfied,
        )

    def _deterministic_objective_check(
        self,
        plan: ExecutionPlan,
        verification_results: List[VerificationResult],
    ) -> bool:
        """Determine objective satisfaction using deterministic signals."""
        # If any step failed, objective is not satisfied
        if plan.failed_steps:
            return False

        # If not all steps completed or skipped, not satisfied
        if not plan.is_complete:
            return False

        # If we have verification results, check them
        if verification_results:
            all_verified = all(vr.verified for vr in verification_results)
            return all_verified

        # All steps completed and no verification results = satisfied by default
        return True

    async def _llm_evaluate(
        self,
        plan: ExecutionPlan,
        task_graph_summary: str,
        task_results_text: str,
        artifact_locations: List[str],
        unresolved_errors: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Use LLM for qualitative objective evaluation."""
        if self.completion_fn is None:
            return None

        prompt = OBJECTIVE_VERIFICATION_PROMPT.format(
            objective=plan.objective,
            task_graph_summary=task_graph_summary,
            task_results=task_results_text,
            artifacts="\n".join(f"  - {loc}" for loc in artifact_locations) if artifact_locations else "  None",
            unresolved_errors="\n".join(f"  - {e}" for e in unresolved_errors) if unresolved_errors else "  None",
        )

        try:
            response = await self.completion_fn(
                messages=[{"role": "user", "content": prompt}],
                system="You are a verification expert. Determine if the objective was achieved.",
                tools=None,
            )
            content = response.get("content", "")

            # Parse response
            satisfied = self._parse_satisfied(content)
            confidence = self._parse_confidence(content)
            completed_items = self._parse_list(content, "COMPLETED_ITEMS")
            incomplete_items = self._parse_list(content, "INCOMPLETE_ITEMS")
            limitations = self._parse_list(content, "LIMITATIONS")

            return {
                "satisfied": satisfied,
                "confidence": confidence,
                "completed_items": completed_items,
                "incomplete_items": incomplete_items,
                "limitations": limitations,
            }
        except Exception:
            return None

    @staticmethod
    def _parse_satisfied(content: str) -> bool:
        content_upper = content.upper()
        if "OBJECTIVE_SATISFIED" in content_upper and "NOT_SATISFIED" not in content_upper:
            return True
        return False

    @staticmethod
    def _parse_confidence(content: str) -> float:
        match = re.search(r'CONFIDENCE:\s*(\d+\.?\d*)', content, re.I)
        if match:
            try:
                val = float(match.group(1))
                return max(0.0, min(1.0, val))
            except ValueError:
                pass
        return 0.5

    @staticmethod
    def _parse_list(content: str, field_name: str) -> List[str]:
        match = re.search(rf'{field_name}:\s*(.+?)(?:\n|$)', content, re.I)
        if match:
            text = match.group(1).strip()
            if text.lower() in ("none", "n/a", "nothing"):
                return []
            return [item.strip() for item in text.split(",") if item.strip()]
        return []

    @staticmethod
    def _build_task_graph_summary(plan: ExecutionPlan) -> str:
        lines = []
        for step in plan.steps:
            status_marker = {
                "completed": "[DONE]",
                "failed": "[FAIL]",
                "skipped": "[SKIP]",
                "pending": "[TODO]",
                "in_progress": "[...]",
            }.get(step.status, "[???]")
            deps = f" (deps: {', '.join(step.dependencies)})" if step.dependencies else ""
            lines.append(f"  {status_marker} {step.id}: {step.title}{deps}")
        return "\n".join(lines)

    @staticmethod
    def _build_task_results_summary(
        plan: ExecutionPlan,
        verification_results: List[VerificationResult],
    ) -> str:
        lines = []
        vr_map = {vr.step_id: vr for vr in verification_results}
        for step in plan.steps:
            result = step.result
            vr = vr_map.get(step.id)
            if result:
                verified_str = "verified" if (vr and vr.verified) else ("unverified" if vr else "not checked")
                lines.append(
                    f"  {step.id}: success={result.success}, "
                    f"output_len={len(result.output)}, "
                    f"errors={'yes' if result.error else 'none'}, "
                    f"verification={verified_str}"
                )
            else:
                lines.append(f"  {step.id}: no result")
        return "\n".join(lines)

    @staticmethod
    def _build_summary(
        plan: ExecutionPlan,
        status: str,
        objective_satisfied: bool,
        completed_items: List[str],
        incomplete_items: List[str],
        unresolved_errors: List[str],
        limitations: List[str],
    ) -> str:
        parts = [f"Verification status: {status}"]
        parts.append(f"Objective satisfied: {objective_satisfied}")
        parts.append(f"Completed: {len(completed_items)} items")
        if incomplete_items:
            parts.append(f"Incomplete: {len(incomplete_items)} items")
        if unresolved_errors:
            parts.append(f"Unresolved errors: {len(unresolved_errors)}")
        if limitations:
            parts.append(f"Limitations: {len(limitations)}")
        return ". ".join(parts)


# ── Recovery Planner ─────────────────────────────────────────────────────────


class RecoveryPlanner(Node):
    """
    Creates recovery tasks when verification fails.

    Given verification failures, generates minimal recovery tasks
    that can be added to the execution plan to achieve the objective.
    """

    name = "recovery_planner"

    def __init__(
        self,
        name: str = "recovery_planner",
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
        objective_verification_raw = packet.payload.get("objective_verification")

        if plan_raw is None or objective_verification_raw is None:
            return packet.fork(
                packet.kind,
                recovery_tasks=[],
                recovery_needed=False,
            )

        plan = ExecutionPlan(**plan_raw) if isinstance(plan_raw, dict) else plan_raw
        obj_ver = (
            ObjectiveVerificationResult(**objective_verification_raw)
            if isinstance(objective_verification_raw, dict)
            else objective_verification_raw
        )

        # If objective is satisfied, no recovery needed
        if obj_ver.objective_satisfied:
            return packet.fork(
                packet.kind,
                recovery_tasks=[],
                recovery_needed=False,
            )

        # Build recovery tasks from incomplete items
        recovery_tasks = []

        # If we have incomplete items, create recovery tasks for each
        for idx, item in enumerate(obj_ver.incomplete_items):
            recovery_tasks.append(RecoveryTask(
                title=f"Recovery: {item[:60]}",
                objective=f"Resolve verification failure: {item}",
                execution_instructions=(
                    f"The following item was identified as incomplete during verification:\n"
                    f"{item}\n\n"
                    f"Original objective: {plan.objective}\n\n"
                    f"Take whatever actions are needed to complete this item."
                ),
                verification_criteria="The item should be completed and verifiable",
                recovery_for=item.split(":")[0] if ":" in item else "",
                reason=f"Verification identified this as incomplete",
                priority=idx,
            ))

        # If we have unresolved errors, create recovery tasks for those
        for idx, error in enumerate(obj_ver.unresolved_errors):
            recovery_tasks.append(RecoveryTask(
                title=f"Error recovery: {error[:60]}",
                objective=f"Resolve error: {error}",
                execution_instructions=(
                    f"The following error was unresolved:\n"
                    f"{error}\n\n"
                    f"Original objective: {plan.objective}\n\n"
                    f"Diagnose and fix this error."
                ),
                verification_criteria="The error should be resolved",
                recovery_for=error.split("]")[0].strip("[") if "]" in error else "",
                reason=f"Unresolved error from verification",
                priority=len(obj_ver.incomplete_items) + idx,
            ))

        # Emit recovery task creation events
        for rt in recovery_tasks:
            await ctx.emit.emit(
                EVT_RECOVERY_TASK_CREATED, self.name, packet,
                recovery_task_id=rt.id,
                title=rt.title,
                recovery_for=rt.recovery_for,
                reason=rt.reason,
            )

        return packet.fork(
            packet.kind,
            recovery_tasks=[rt.model_dump() for rt in recovery_tasks],
            recovery_needed=len(recovery_tasks) > 0,
        )


# ── Final Response Generator ────────────────────────────────────────────────


class FinalResponseGenerator(Node):
    """
    Generates the final user-facing response from the verified execution state.

    Clearly states:
    - What was completed
    - What was changed
    - What was verified
    - Any failures or limitations

    Never fabricates actions.
    """

    name = "final_response_generator"

    def __init__(
        self,
        name: str = "final_response_generator",
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
        objective_verification_raw = packet.payload.get("objective_verification")
        recovery_tasks_raw = packet.payload.get("recovery_tasks", [])

        if plan_raw is None:
            return packet.fork(
                packet.kind,
                final_response=FinalResponse(
                    status="failure",
                    summary="No execution plan found",
                ).model_dump(),
            )

        plan = ExecutionPlan(**plan_raw) if isinstance(plan_raw, dict) else plan_raw
        obj_ver = (
            ObjectiveVerificationResult(**objective_verification_raw)
            if isinstance(objective_verification_raw, dict)
            else objective_verification_raw
        ) if objective_verification_raw else None

        recovery_tasks = [
            RecoveryTask(**rt) if isinstance(rt, dict) else rt
            for rt in recovery_tasks_raw
        ]

        # Build structured response
        completed_items = []
        changed_items = []
        verified_items = []
        failed_items = []
        limitations = []

        # Collect completed steps
        for step in plan.completed_steps:
            completed_items.append(f"Completed: {step.title}")
            if step.result and step.result.output:
                # Extract what changed from tool calls
                for tc in step.result.tool_calls:
                    tool_name = tc.get("tool", "")
                    if tool_name in ("write_file", "edit_file", "create_file"):
                        args_preview = tc.get("args_preview", "")
                        try:
                            args = json.loads(args_preview) if args_preview else {}
                            path = args.get("path", args.get("file_path", ""))
                            if path:
                                changed_items.append(f"Modified: {path}")
                        except (json.JSONDecodeError, AttributeError):
                            pass

        # Collect verified items from verification results
        if obj_ver and obj_ver.task_results:
            for vr in obj_ver.task_results:
                if vr.verified:
                    verified_items.append(f"Verified: {vr.step_id} ({vr.summary})")
                elif vr.issues:
                    failed_items.append(f"Issue in {vr.step_id}: {'; '.join(vr.issues)}")

        # Collect failed steps
        for step in plan.failed_steps:
            error_msg = step.result.error if step.result else "No result"
            failed_items.append(f"Failed: {step.title} - {error_msg}")

        # Collect limitations
        if obj_ver:
            if obj_ver.unresolved_errors:
                limitations.extend(obj_ver.unresolved_errors)
            if obj_ver.incomplete_items:
                limitations.extend([f"Incomplete: {item}" for item in obj_ver.incomplete_items])

        # Add recovery task limitations
        if recovery_tasks:
            limitations.append(f"{len(recovery_tasks)} recovery task(s) needed to complete objective")

        # Determine status
        if obj_ver and obj_ver.objective_satisfied:
            status = "success"
        elif completed_items and failed_items:
            status = "partial"
        elif failed_items:
            status = "failure"
        else:
            status = "partial"

        # Build verification summary
        verification_summary = ""
        if obj_ver:
            verification_summary = obj_ver.summary

        # Build overall summary
        summary = self._build_summary(
            plan, status, completed_items, changed_items,
            verified_items, failed_items, limitations, obj_ver
        )

        response = FinalResponse(
            status=status,
            objective=plan.objective,
            completed_items=completed_items,
            changed_items=changed_items,
            verified_items=verified_items,
            failed_items=failed_items,
            limitations=limitations,
            verification_summary=verification_summary,
            errors=[f for f in failed_items],
            summary=summary,
        )

        # Emit final response event
        await ctx.emit.emit(
            EVT_FINAL_RESPONSE, self.name, packet,
            status=status,
            completed_count=len(completed_items),
            changed_count=len(changed_items),
            verified_count=len(verified_items),
            failed_count=len(failed_items),
        )

        return packet.fork(
            packet.kind,
            final_response=response.model_dump(),
        )

    @staticmethod
    def _build_summary(
        plan: ExecutionPlan,
        status: str,
        completed_items: List[str],
        changed_items: List[str],
        verified_items: List[str],
        failed_items: List[str],
        limitations: List[str],
        obj_ver: Optional[ObjectiveVerificationResult],
    ) -> str:
        parts = []

        if status == "success":
            parts.append(f"Objective achieved successfully.")
        elif status == "partial":
            parts.append(f"Objective partially achieved.")
        else:
            parts.append(f"Objective NOT achieved.")

        parts.append(f"{len(completed_items)} task(s) completed.")

        if changed_items:
            unique_changes = list(dict.fromkeys(changed_items))  # deduplicate
            parts.append(f"{len(unique_changes)} file(s) changed.")

        if verified_items:
            parts.append(f"{len(verified_items)} item(s) verified.")

        if failed_items:
            parts.append(f"{len(failed_items)} failure(s).")

        if limitations:
            parts.append(f"{len(limitations)} limitation(s).")

        if obj_ver and obj_ver.recovery_tasks:
            parts.append(f"{len(obj_ver.recovery_tasks)} recovery task(s) recommended.")

        return " ".join(parts)


# ── Event Stream Adapter ─────────────────────────────────────────────────────


class EventStreamAdapter:
    """
    Bridges the internal EventEmitter to a structured, frontend-safe event stream.

    This adapter:
    1. Attaches to the internal EventEmitter via RunStateTracker
    2. Provides explicit emit methods for each semantic event kind
    3. Maintains a RunStateSnapshot for state reconstruction
    4. Ensures deterministic event ordering via sequence numbers
    5. Sanitizes all data before emitting (no secrets, no private prompts)

    Usage:
        adapter = EventStreamAdapter(run_id="abc", objective="Build feature X")
        adapter.attach(ctx.emit)
        # ... adapter.emit_run_created(), emit_task_started(), etc. ...
        snapshot = adapter.get_snapshot()  # for get-run API
        events = adapter.get_events()      # for event replay / SSE
    """

    def __init__(
        self,
        run_id: str,
        objective: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._tracker = RunStateTracker(run_id, objective, metadata)

    def attach(self, emitter: EventEmitter) -> None:
        """Attach to the internal EventEmitter for automatic event translation."""
        self._tracker.subscribe_to(emitter)

    def detach(self) -> None:
        """Detach from the internal EventEmitter."""
        self._tracker.unsubscribe()

    # ── Explicit emit methods for each event kind ────────────────────────

    def emit_run_created(
        self,
        objective: str = "",
        graph_name: str = "",
    ) -> Any:
        """Emit run_created event."""
        self._tracker.objective = objective or self._tracker.objective
        self._tracker.set_status("running")
        return self._tracker.emit_event(
            EVT_EXEC_RUN_CREATED,
            summary=f"Run started: {objective[:80]}",
            metadata={"graph_name": graph_name},
        )

    def emit_intent_analyzed(
        self,
        intent: str = "",
        complexity: str = "",
    ) -> Any:
        """Emit intent_analyzed event."""
        return self._tracker.emit_event(
            EVT_EXEC_INTENT_ANALYZED,
            summary=f"Intent analyzed: {intent[:80]}",
            metadata={"complexity": complexity},
        )

    def emit_complexity_determined(
        self,
        complexity: str = "",
        needs_planning: bool = False,
    ) -> Any:
        """Emit complexity_determined event."""
        return self._tracker.emit_event(
            EVT_EXEC_COMPLEXITY_DETERMINED,
            summary=f"Complexity: {complexity}",
            metadata={"needs_planning": needs_planning},
        )

    def emit_planning_started(self, objective: str = "") -> Any:
        """Emit planning_started event."""
        return self._tracker.emit_event(
            EVT_EXEC_PLANNING_STARTED,
            summary="Planning execution steps",
        )

    def emit_plan_created(
        self,
        step_count: int = 0,
        objective: str = "",
    ) -> Any:
        """Emit plan_created event."""
        return self._tracker.emit_event(
            EVT_EXEC_PLAN_CREATED,
            summary=f"Plan created with {step_count} steps",
            plan_step_count=step_count,
            plan_progress=0.0,
            metadata={"objective": objective[:200]},
        )

    def emit_task_ready(
        self,
        task_id: str = "",
        task_title: str = "",
        plan_progress: Optional[float] = None,
    ) -> Any:
        """Emit task_ready event."""
        return self._tracker.emit_event(
            EVT_EXEC_TASK_READY,
            summary=f"Task ready: {task_title}",
            task_id=task_id,
            task_title=task_title,
            task_status="pending",
            plan_progress=plan_progress,
        )

    def emit_task_started(
        self,
        task_id: str = "",
        task_title: str = "",
        plan_progress: Optional[float] = None,
    ) -> Any:
        """Emit task_started event."""
        return self._tracker.emit_event(
            EVT_EXEC_TASK_STARTED,
            summary=f"Starting task: {task_title}",
            task_id=task_id,
            task_title=task_title,
            task_status="in_progress",
            plan_progress=plan_progress,
        )

    def emit_model_started(
        self,
        task_id: str = "",
        model_name: str = "",
    ) -> Any:
        """Emit model_started event."""
        return self._tracker.emit_event(
            EVT_EXEC_MODEL_STARTED,
            summary=f"Model invoked for task {task_id}",
            task_id=task_id,
            metadata={"model": model_name},
        )

    def emit_tool_started(
        self,
        task_id: str = "",
        tool_name: str = "",
        tool_args_preview: str = "",
        tool_args: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Emit tool_started event."""
        return self._tracker.emit_event(
            EVT_EXEC_TOOL_STARTED,
            summary=f"Executing tool: {tool_name}",
            task_id=task_id,
            metadata={
                "tool": tool_name,
                "args_preview": tool_args_preview[:200],
                # Full args (not just the truncated preview string) so the
                # UI can render a real ToolCallCard instead of a plain
                # summary line — this is what makes a claimed action
                # ("copied the PDFs") checkable against what was actually
                # requested of the tool.
                "args": tool_args if isinstance(tool_args, dict) else {},
            },
        )

    def emit_tool_completed(
        self,
        task_id: str = "",
        tool_name: str = "",
        success: bool = True,
        duration_ms: Optional[float] = None,
        affected_files: Optional[List[str]] = None,
    ) -> Any:
        """Emit tool_completed event."""
        return self._tracker.emit_event(
            EVT_EXEC_TOOL_COMPLETED,
            summary=f"Tool {tool_name} {'completed' if success else 'failed'}",
            task_id=task_id,
            error=None if success else f"Tool {tool_name} failed",
            metadata={
                "tool": tool_name,
                "success": success,
                "duration_ms": duration_ms,
                "affected_files": affected_files or [],
            },
        )

    def emit_task_observed(
        self,
        task_id: str = "",
        task_title: str = "",
        decision: str = "",
        plan_progress: Optional[float] = None,
    ) -> Any:
        """Emit task_observed event."""
        return self._tracker.emit_event(
            EVT_EXEC_TASK_OBSERVED,
            summary=f"Task observed: {decision}",
            task_id=task_id,
            task_title=task_title,
            plan_progress=plan_progress,
            metadata={"decision": decision},
        )

    def emit_task_completed(
        self,
        task_id: str = "",
        task_title: str = "",
        plan_progress: Optional[float] = None,
        duration_s: Optional[float] = None,
        tool_calls_made: Optional[int] = None,
    ) -> Any:
        """Emit task_completed event."""
        if task_id and task_id not in self._tracker._completed_task_ids:
            self._tracker._completed_task_ids.append(task_id)
        self._tracker._current_task_id = None
        self._tracker._current_task_title = None
        return self._tracker.emit_event(
            EVT_EXEC_TASK_COMPLETED,
            summary=f"Completed task: {task_title}",
            task_id=task_id,
            task_title=task_title,
            task_status="completed",
            plan_progress=plan_progress,
            metadata={
                "duration_s": duration_s,
                "tool_calls_made": tool_calls_made,
            },
        )

    def emit_task_failed(
        self,
        task_id: str = "",
        task_title: str = "",
        error: str = "",
        plan_progress: Optional[float] = None,
    ) -> Any:
        """Emit task_failed event."""
        if task_id and task_id not in self._tracker._failed_task_ids:
            self._tracker._failed_task_ids.append(task_id)
        self._tracker._current_task_id = None
        self._tracker._current_task_title = None
        return self._tracker.emit_event(
            EVT_EXEC_TASK_FAILED,
            summary=f"Task failed: {task_title}",
            task_id=task_id,
            task_title=task_title,
            task_status="failed",
            error=error,
            plan_progress=plan_progress,
        )

    def emit_retry_started(
        self,
        task_id: str = "",
        task_title: str = "",
        retry_reason: str = "",
        retry_count: int = 0,
    ) -> Any:
        """Emit retry_started event."""
        return self._tracker.emit_event(
            EVT_EXEC_RETRY_STARTED,
            summary=f"Retrying task: {task_title} (attempt {retry_count + 1})",
            task_id=task_id,
            task_title=task_title,
            task_status="in_progress",
            metadata={
                "retry_reason": retry_reason,
                "retry_count": retry_count,
            },
        )

    def emit_replanning_started(
        self,
        reason: str = "",
        replan_count: int = 0,
    ) -> Any:
        """Emit replanning_started event."""
        return self._tracker.emit_event(
            EVT_EXEC_REPLANNING_STARTED,
            summary=f"Replanning (attempt {replan_count + 1})",
            metadata={"reason": reason, "replan_count": replan_count},
        )

    def emit_plan_updated(
        self,
        step_count: Optional[int] = None,
        completed_count: Optional[int] = None,
        plan_progress: Optional[float] = None,
        reason: str = "",
    ) -> Any:
        """Emit plan_updated event."""
        return self._tracker.emit_event(
            EVT_EXEC_PLAN_UPDATED,
            summary=f"Plan updated: {reason}" if reason else "Plan updated",
            plan_step_count=step_count,
            plan_completed_count=completed_count,
            plan_progress=plan_progress,
        )

    def emit_verification_started(
        self,
        task_id: str = "",
        task_title: str = "",
    ) -> Any:
        """Emit verification_started event."""
        return self._tracker.emit_event(
            EVT_EXEC_VERIFICATION_STARTED,
            summary=f"Verifying task: {task_title}",
            task_id=task_id,
            task_title=task_title,
        )

    def emit_verification_completed(
        self,
        task_id: str = "",
        verified: bool = True,
        pass_rate: Optional[float] = None,
        summary_text: str = "",
    ) -> Any:
        """Emit verification_completed event."""
        return self._tracker.emit_event(
            EVT_EXEC_VERIFICATION_COMPLETED,
            summary=summary_text or (
                f"Verification {'passed' if verified else 'failed'} for {task_id}"
            ),
            task_id=task_id,
            verification_status="verified" if verified else "failed",
            metadata={"pass_rate": pass_rate},
        )

    def emit_run_completed(
        self,
        summary_text: str = "",
        plan_progress: float = 1.0,
        final_answer: str = "",
    ) -> Any:
        """Emit run_completed event with optional final answer reference."""
        self._tracker.set_status("completed")
        metadata = {}
        if final_answer:
            # Store a truncated version of the final answer for frontend reference
            metadata["final_answer"] = final_answer[:500]
            metadata["final_answer_length"] = len(final_answer)
        return self._tracker.emit_event(
            EVT_EXEC_RUN_COMPLETED,
            summary=summary_text or "Run completed successfully",
            plan_progress=plan_progress,
            metadata=metadata,
        )

    def emit_run_failed(
        self,
        error: str = "",
    ) -> Any:
        """Emit run_failed event."""
        self._tracker.set_status("failed")
        return self._tracker.emit_event(
            EVT_EXEC_RUN_FAILED,
            summary=f"Run failed: {error}",
            error=error,
        )

    def emit_run_cancelled(
        self,
        reason: str = "",
    ) -> Any:
        """Emit run_cancelled event."""
        self._tracker.set_status("cancelled")
        return self._tracker.emit_event(
            EVT_EXEC_RUN_CANCELLED,
            summary=f"Run cancelled: {reason}",
            error=reason,
        )

    # ── State access ─────────────────────────────────────────────────────

    def get_snapshot(self) -> Any:
        """
        Get a RunStateSnapshot for state reconstruction.

        Use this for the get-run API to allow the frontend to
        reconstruct the full timeline after reconnecting.
        """
        return self._tracker.snapshot()

    def get_events(self) -> List[Any]:
        """Get all emitted ExecutionEvents in order."""
        return self._tracker.events

    def get_status(self) -> str:
        """Get current run status."""
        return self._tracker.status

    def update_plan(self, plan: Any) -> None:
        """Update tracked plan state."""
        self._tracker.update_plan(plan)

    def set_verification_result(self, result: Any) -> None:
        """Set verification result."""
        self._tracker.set_verification_result(result)

    def set_final_response(self, response: Any) -> None:
        """Set final response."""
        self._tracker.set_final_response(response)


__all__ = [
    "TaskExecutionLoop",
    "FocusedTaskContextBuilder",
    "TaskExecutionObserver",
    "AdaptiveTaskExecutionObserver",
    "TaskRetryHandler",
    "TaskFinalVerifier",
    "AdaptiveReplanner",
    "PlanIntegrityChecker",
    "TaskArtifactCollector",
    "DeterministicVerifier",
    "TaskStepVerifier",
    "ObjectiveVerifier",
    "RecoveryPlanner",
    "FinalResponseGenerator",
    "EventStreamAdapter",
    "TASK_EXECUTION_SYSTEM_PROMPT",
    "TASK_REPLAN_PROMPT",
    "TASK_OBSERVER_EVAL_PROMPT",
    "ADAPTIVE_REPLAN_PROMPT",
    "DOWNSTREAM_VALIDATION_PROMPT",
    "TASK_STEP_VERIFICATION_PROMPT",
    "OBJECTIVE_VERIFICATION_PROMPT",
]
