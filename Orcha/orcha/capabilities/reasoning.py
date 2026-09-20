"""
orcha.capabilities.reasoning
============================
Reasoning levels (Fast → Max) as orchestration configuration — not prompt
engineering. Each level maps to concrete builder knobs that change how many
planning steps run, how much retrieval/verification happens, whether
multi-agent execution is used, and how failures are retried.

The orchestrator reads :func:`reasoning_config` and wires the knobs into the
graph builder (see ``orcha/builders/multi_agent.py``).
"""
from __future__ import annotations

import enum
from typing import Any, Dict, Optional


class ReasoningLevel(str, enum.Enum):
    """The five supported reasoning levels."""

    FAST = "fast"
    LIGHT = "light"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


LEVELS = [level.value for level in ReasoningLevel]


def _level(value: Optional[str]) -> ReasoningLevel:
    if value is None:
        return ReasoningLevel.MEDIUM
    try:
        return ReasoningLevel(value)
    except ValueError:
        raise ValueError(f"Invalid reasoning level '{value}'. Use one of: {', '.join(LEVELS)}")


def reasoning_config(level: Optional[str]) -> Dict[str, Any]:
    """
    Return the orchestration configuration for a reasoning level.

    Knobs
    -----
    planner_steps    how many planning iterations run before acting
    retrieval        whether workspace/context retrieval happens
    verify           whether results are verified before finalizing
    synthesize       whether a synthesis pass aggregates sub-results
    multi_agent      whether parallel sub-agents execute subtasks
    max_iterations   agent tool-call iteration budget
    retries          retry budget for transient failures
    description      human-readable summary for UI/prompts
    """
    config = {
        "fast": {
            "planner_steps": 1, "retrieval": False, "verify": False,
            "synthesize": False, "multi_agent": False, "max_iterations": 2,
            "retries": 0,
            "description": "One-pass direct answer; minimal tool use.",
        },
        "light": {
            "planner_steps": 2, "retrieval": True, "verify": False,
            "synthesize": False, "multi_agent": False, "max_iterations": 4,
            "retries": 1,
            "description": "Brief planning plus minimal retrieval.",
        },
        "medium": {
            "planner_steps": 3, "retrieval": True, "verify": False,
            "synthesize": True, "multi_agent": False, "max_iterations": 8,
            "retries": 1,
            "description": "Planning, retrieval and optional tool calls.",
        },
        "high": {
            "planner_steps": 4, "retrieval": True, "verify": True,
            "synthesize": True, "multi_agent": False, "max_iterations": 12,
            "retries": 2,
            "description": "Planning, retrieval, verification and multiple tool calls.",
        },
        "max": {
            "planner_steps": 6, "retrieval": True, "verify": True,
            "synthesize": True, "multi_agent": True, "max_iterations": 16,
            "retries": 3,
            "description": "Full planner, parallel sub-agents, verification and synthesis.",
        },
    }
    return config[_level(level).value]


# ── Model capability mapping ─────────────────────────────────────────────────

_NATIVE_REASONING_HINTS = (
    "reasoning", "r1", "qwq", "o1", "o3", "o4", "think", "deepthink",
    "deepseek-r1", "sky-t1", "gpt-oss", "kimi-k2",
)


def model_supports_native_reasoning(model_name: Optional[str]) -> bool:
    """
    Does this model family produce a chain-of-thought natively (e.g.
    DeepSeek-R1, QwQ, OpenAI o-series, Kimi K2)? Used by the streaming
    path to decide whether to hand the selected reasoning level to the
    model as a native ``reasoning_effort`` field, or emulate it with
    orchestration knobs (output budget + sampling) instead.
    """
    if not model_name:
        return False
    name = model_name.lower()
    return any(hint in name for hint in _NATIVE_REASONING_HINTS)


# ── Pipeline / generation knobs ──────────────────────────────────────────────

_EFFORT = {"fast": "low", "light": "low", "medium": "medium", "high": "high", "max": "high"}

_PIPELINE = {
    # max_iterations   pipeline loop cap (None = server default)
    # run_all_experts  run every expert every iteration (None = leave as-is)
    # width            exact planner parallel width (None = default planner logic)
    # threshold        quality threshold to pass evaluation (None = default logic)
    # output_scale     streaming output-token budget multiplier (emulation)
    # temperature_delta  streaming sampling shift (emulation)
    # frequency_penalty / presence_penalty  anti-repetition penalties
    "fast": {
        "max_iterations": 1, "run_all_experts": False,
        "width": 1, "threshold": 0.50,
        "output_scale": 0.75, "temperature_delta": 0.10,
        "frequency_penalty": 0.5, "presence_penalty": 0.3,
        "description": "One-pass direct answer; minimal tool use.",
    },
    "light": {
        "max_iterations": 2, "run_all_experts": False,
        "width": 2, "threshold": 0.55,
        "output_scale": 0.90, "temperature_delta": 0.05,
        "frequency_penalty": 0.4, "presence_penalty": 0.2,
        "description": "Brief planning plus minimal retrieval.",
    },
    "medium": {
        "max_iterations": None, "run_all_experts": None,
        "width": None, "threshold": None,
        "output_scale": 1.0, "temperature_delta": 0.0,
        "frequency_penalty": 0.3, "presence_penalty": 0.15,
        "description": "Planning, retrieval and optional tool calls.",
    },
    "high": {
        "max_iterations": 4, "run_all_experts": True,
        "width": 4, "threshold": 0.70,
        "output_scale": 1.30, "temperature_delta": -0.10,
        "frequency_penalty": 0.2, "presence_penalty": 0.1,
        "description": "Planning, retrieval, verification and multiple tool calls.",
    },
    "max": {
        "max_iterations": 5, "run_all_experts": True,
        "width": 5, "threshold": 0.80,
        "output_scale": 1.60, "temperature_delta": -0.15,
        "frequency_penalty": 0.15, "presence_penalty": 0.05,
        "description": "Full planner, parallel sub-agents, verification and synthesis.",
    },
}


def pipeline_config(level: Optional[str]) -> Dict[str, Any]:
    """
    Orchestration knobs for the multi-expert pipeline (POST /v1/query).

    Unlike :func:`reasoning_config` (which targets the multi-agent graph
    builder), this maps a level to the classic planner/budget pipeline:
    iteration cap, expert fan-out, the planner's parallel width, and the
    quality threshold that an answer must clear to avoid a retry.

    ``output_scale``/``temperature_delta`` are used by the streaming path
    (POST /v1/query/stream) to emulate the level on models without native
    chain-of-thought.
    """
    return _PIPELINE[_level(level).value]


def effort_for_level(level: Optional[str]) -> str:
    """
    Native-reasoning effort string for OpenAI-compatible servers that
    expose a ``reasoning_effort`` field: low | medium | high.
    """
    return _EFFORT[_level(level).value]


__all__ = [
    "ReasoningLevel",
    "LEVELS",
    "reasoning_config",
    "pipeline_config",
    "effort_for_level",
    "model_supports_native_reasoning",
]
