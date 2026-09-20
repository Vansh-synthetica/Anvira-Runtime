"""
Small-model-optimized prompt templates for Orcha task execution.

Design principles for 7B-class local models:
1. Short, imperative prompts — every token counts
2. Structured markers the model can reliably parse
3. One clear objective per prompt — no ambiguity
4. Deterministic output format — minimize free-form generation
5. Context budget: system prompt < 400 tokens, user message < 300 tokens
6. Explicit STOP conditions — small models tend to over-generate
7. Recovery from malformed output via re-prompting with examples
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# ── Task Execution ───────────────────────────────────────────────────────────

TASK_EXEC_SYSTEM_SMALL = """TASK: {task_title}

GOAL: {task_objective}

RULES:
- Use ONLY these tools: {tools}
- Do NOT touch files outside scope
- When done output: DONE: <what you did>
- If stuck output: STUCK: <why>

{verification_block}
{constraints_block}
{context_block}"""

TASK_EXEC_SYSTEM_SMALL_WITH_TOOLS = """TASK: {task_title}

GOAL: {task_objective}

TOOLS: {tools}

INSTRUCTIONS:
{execution_instructions}

DONE when: {verification_criteria}

OUTPUT FORMAT:
- Use a tool, OR
- Reply DONE: <result>, OR
- Reply STUCK: <reason>

DO NOT:
- Modify files outside this task
- Run unnecessary commands
- Fabricate tool results"""


def build_small_exec_system(
    step_title: str,
    step_objective: str,
    tools: List[str],
    execution_instructions: str = "",
    verification_criteria: str = "",
    constraints: Optional[List[str]] = None,
    prior_results: Optional[List[str]] = None,
) -> str:
    """Build a compact system prompt optimized for small models.

    Target: < 400 tokens. Every word earns its place.
    """
    tools_str = ", ".join(tools[:8]) if tools else "none available"

    verification_block = ""
    if verification_criteria:
        verification_block = f"VERIFY: {verification_criteria[:200]}"

    constraints_block = ""
    if constraints:
        short = [c[:80] for c in constraints[:3]]
        constraints_block = "RULES: " + "; ".join(short)

    context_block = ""
    if prior_results:
        # Only include the most recent, truncated to 150 chars each
        recent = prior_results[-2:]
        context_block = "PRIOR:\n" + "\n".join(
            f"- {r[:150]}" for r in recent
        )

    if execution_instructions and len(execution_instructions) < 300:
        return TASK_EXEC_SYSTEM_SMALL_WITH_TOOLS.format(
            task_title=step_title[:60],
            task_objective=step_objective[:200],
            tools=tools_str,
            execution_instructions=execution_instructions[:300],
            verification_criteria=verification_criteria[:200],
        )

    return TASK_EXEC_SYSTEM_SMALL.format(
        task_title=step_title[:60],
        task_objective=step_objective[:200],
        tools=tools_str,
        verification_block=verification_block,
        constraints_block=constraints_block,
        context_block=context_block,
    )


# ── Task Completion Detection ────────────────────────────────────────────────

DONE_MARKER = "DONE:"
STUCK_MARKER = "STUCK:"
TASK_COMPLETE_MARKER = "TASK COMPLETE:"
CANNOT_COMPLETE_MARKER = "CANNOT COMPLETE:"
NEED_REPLAN_MARKER = "NEED REPLAN:"

# All recognized completion markers, ordered by specificity
COMPLETION_MARKERS = [
    DONE_MARKER,
    TASK_COMPLETE_MARKER,
    STUCK_MARKER,
    CANNOT_COMPLETE_MARKER,
    NEED_REPLAN_MARKER,
]


def detect_task_completion(content: str) -> Optional[Dict[str, str]]:
    """Detect if the model's output indicates task completion or failure.

    Returns {"status": "done"|"stuck"|"replan", "output": "..."} or None.
    Optimized for small models that may not follow the exact format.
    """
    text = content.strip()
    if not text:
        return None

    # Check markers in order of specificity
    for marker in COMPLETION_MARKERS:
        if marker in text:
            idx = text.index(marker)
            output = text[idx + len(marker):].strip()
            if marker in (DONE_MARKER, TASK_COMPLETE_MARKER):
                return {"status": "done", "output": output}
            elif marker in (STUCK_MARKER, CANNOT_COMPLETE_MARKER):
                return {"status": "stuck", "output": output}
            elif marker == NEED_REPLAN_MARKER:
                return {"status": "replan", "output": output}

    # Fuzzy detection for small models that may miss the colon or space
    lower = text.lower()
    if "done" in lower[:30] and len(text) > 10:
        # Could be a DONE variant — extract after "done"
        for sep in [":", ":", "-"]:
            if sep in lower:
                idx = lower.index(sep)
                return {"status": "done", "output": text[idx + 1:].strip()}
    if "stuck" in lower[:30] or "cannot" in lower[:30]:
        return {"status": "stuck", "output": text}
    if "replan" in lower[:30]:
        return {"status": "replan", "output": text}

    return None


# ── Malformed Output Recovery ────────────────────────────────────────────────

MALFORMED_RECOVERY_PROMPT = """Your last response was not in the correct format.

You MUST reply with exactly one of:
1. Use a tool (via tool call)
2. DONE: <what you accomplished>
3. STUCK: <why you cannot continue>

Your last response was:
---
{previous_response}
---

Try again. Use a tool, say DONE:, or say STUCK:"""


def build_malformed_recovery(previous_response: str) -> str:
    """Build a recovery prompt for malformed model output.

    Keeps the prompt short — small models get confused by long corrections.
    """
    return MALFORMED_RECOVERY_PROMPT.format(
        previous_response=previous_response[:300]
    )


# ── Empty Output Recovery ────────────────────────────────────────────────────

EMPTY_OUTPUT_RECOVERY = """You provided an empty response.

Use a tool to make progress, or say:
- DONE: <what you accomplished>
- STUCK: <why you cannot continue>

What will you do?"""


# ── Repeated Tool Call Recovery ──────────────────────────────────────────────

REPEATED_TOOL_RECOVERY = """You called {tool_name} {count} times with similar arguments.

This is not making progress. Try a DIFFERENT approach:
1. Use a different tool
2. Change the arguments significantly
3. Say DONE: if the task is actually complete
4. Say STUCK: if you cannot proceed

What will you do differently?"""


# ── Observer Evaluation (Small Model) ────────────────────────────────────────

OBSERVER_EVAL_SMALL = """Rate this task result.

TASK: {task_objective}
OUTPUT: {actual_output[:500]}

Reply ONE word:
- PASS (task succeeded)
- FAIL (task failed, can retry)
- CHANGE (wrong approach, modify plan)"""


def parse_observer_decision(content: str) -> str:
    """Parse observer decision from small model output.

    Returns: "CONTINUE", "RETRY", "MODIFY", "REPLAN", or "BLOCK"
    """
    text = content.strip().upper()
    if "PASS" in text[:20]:
        return "CONTINUE"
    if "FAIL" in text[:20]:
        return "RETRY"
    if "CHANGE" in text[:20]:
        return "MODIFY"
    # Fallback: check for standard markers
    for word in ["CONTINUE", "RETRY", "MODIFY", "REPLAN", "BLOCK", "COMPLETE"]:
        if word in text[:30]:
            return word
    return "CONTINUE"  # Default to continue on ambiguous output


# ── Planner Prompt (Small Model) ─────────────────────────────────────────────

PLANNER_SMALL = """Break this into tasks. Output a JSON array.

REQUEST: {request}

Each task needs: id, title, objective, dependencies, likely_tools

Keep it SIMPLE:
- 2-5 tasks max
- Linear dependencies (A->B->C) preferred
- One clear action per task

Output ONLY the JSON array:"""


# ── Replanner Prompt (Small Model) ──────────────────────────────────────────

REPLANNER_SMALL = """The current task failed. Suggest a fix.

FAILED: {failed_task}
REASON: {error}
COMPLETED: {completed_count} tasks done

Reply JSON: {{"action": "retry"|"skip"|"modify", "reason": "..."}}

Keep it brief:"""
