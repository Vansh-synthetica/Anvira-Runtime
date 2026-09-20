"""
Tests for small-model-optimized prompt templates and completion detection.
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

from orcha.nodes.small_model_prompts import (
    build_small_exec_system,
    build_malformed_recovery,
    detect_task_completion,
    parse_observer_decision,
    build_small_exec_system,
    EMPTY_OUTPUT_RECOVERY,
    REPEATED_TOOL_RECOVERY,
    DONE_MARKER,
    STUCK_MARKER,
    TASK_COMPLETE_MARKER,
    CANNOT_COMPLETE_MARKER,
    NEED_REPLAN_MARKER,
)


def test_build_small_exec_system_basic():
    """Basic prompt generation with minimal fields."""
    prompt = build_small_exec_system(
        step_title="Inspect repo",
        step_objective="List files and identify framework",
        tools=["list_directory", "read_file"],
    )
    assert "Inspect repo" in prompt
    assert "List files and identify framework" in prompt
    assert "list_directory" in prompt
    assert "read_file" in prompt
    # Should be short — target < 400 tokens (~1600 chars)
    assert len(prompt) < 2000


def test_build_small_exec_system_with_instructions():
    """Prompt generation with execution instructions uses compact template."""
    prompt = build_small_exec_system(
        step_title="Build feature",
        step_objective="Add a login button",
        tools=["read_file", "write_file"],
        execution_instructions="Read the current HTML, add a button element",
        verification_criteria="Button exists in HTML",
    )
    assert "Build feature" in prompt
    assert "Add a login button" in prompt
    assert "DONE" in prompt
    assert "STUCK" in prompt


def test_build_small_exec_system_with_constraints():
    """Prompt includes truncated constraints."""
    constraints = [
        "Do not modify test files",
        "Use TypeScript",
        "Follow existing code style",
    ]
    prompt = build_small_exec_system(
        step_title="Code",
        step_objective="Write code",
        tools=["write_file"],
        constraints=constraints,
    )
    assert "Do not modify test files" in prompt


def test_build_small_exec_system_with_prior_results():
    """Prompt includes truncated prior results."""
    prior = ["[t1] Inspect: Found React app", "[t2] Plan: Added button spec"]
    prompt = build_small_exec_system(
        step_title="Build",
        step_objective="Build the feature",
        tools=["write_file"],
        prior_results=prior,
    )
    assert "PRIOR" in prompt


def test_build_small_exec_system_truncates_long_objectives():
    """Long objectives are truncated to 200 chars."""
    long_obj = "x" * 500
    prompt = build_small_exec_system(
        step_title="Task",
        step_objective=long_obj,
        tools=[],
    )
    assert long_obj not in prompt
    assert len(prompt) < 2000


def test_detect_task_completion_done():
    """Detect DONE: marker."""
    result = detect_task_completion("DONE: Created the file successfully")
    assert result is not None
    assert result["status"] == "done"
    assert "Created the file" in result["output"]


def test_detect_task_completion_task_complete():
    """Detect TASK COMPLETE: marker."""
    result = detect_task_completion("TASK COMPLETE: Feature implemented")
    assert result is not None
    assert result["status"] == "done"
    assert "Feature implemented" in result["output"]


def test_detect_task_completion_stuck():
    """Detect STUCK: marker."""
    result = detect_task_completion("STUCK: Missing required dependency")
    assert result is not None
    assert result["status"] == "stuck"
    assert "Missing required dependency" in result["output"]


def test_detect_task_completion_cannot_complete():
    """Detect CANNOT COMPLETE: marker."""
    result = detect_task_completion("CANNOT COMPLETE: No access to database")
    assert result is not None
    assert result["status"] == "stuck"
    assert "No access to database" in result["output"]


def test_detect_task_completion_need_replan():
    """Detect NEED REPLAN: marker."""
    result = detect_task_completion("NEED REPLAN: Requirements changed")
    assert result is not None
    assert result["status"] == "replan"
    assert "Requirements changed" in result["output"]


def test_detect_task_completion_empty():
    """Empty input returns None."""
    assert detect_task_completion("") is None
    assert detect_task_completion("  ") is None


def test_detect_task_completion_none():
    """Non-completion text returns None."""
    assert detect_task_completion("I will use read_file to inspect the directory") is None
    assert detect_task_completion("The file contains the following code...") is None


def test_detect_task_completion_fuzzy_done():
    """Fuzzy detection for small models that miss exact format."""
    result = detect_task_completion("done - created the file")
    assert result is not None
    assert result["status"] == "done"


def test_detect_task_completion_fuzzy_stuck():
    """Fuzzy detection for stuck variants."""
    result = detect_task_completion("stuck because of missing tools")
    assert result is not None
    assert result["status"] == "stuck"


def test_parse_observer_decision_pass():
    """Parse PASS from observer."""
    assert parse_observer_decision("PASS") == "CONTINUE"
    assert parse_observer_decision("pass - task succeeded") == "CONTINUE"


def test_parse_observer_decision_fail():
    """Parse FAIL from observer."""
    assert parse_observer_decision("FAIL") == "RETRY"
    assert parse_observer_decision("fail - timeout") == "RETRY"


def test_parse_observer_decision_change():
    """Parse CHANGE from observer."""
    assert parse_observer_decision("CHANGE") == "MODIFY"
    assert parse_observer_decision("change approach") == "MODIFY"


def test_parse_observer_decision_standard():
    """Parse standard decision keywords."""
    assert parse_observer_decision("CONTINUE") == "CONTINUE"
    assert parse_observer_decision("RETRY") == "RETRY"
    assert parse_observer_decision("MODIFY") == "MODIFY"
    assert parse_observer_decision("REPLAN") == "REPLAN"
    assert parse_observer_decision("BLOCK") == "BLOCK"


def test_parse_observer_decision_ambiguous():
    """Ambiguous input defaults to CONTINUE."""
    assert parse_observer_decision("I think the task is done") == "CONTINUE"


def test_build_malformed_recovery():
    """Malformed recovery prompt includes the previous response."""
    prompt = build_malformed_recovery("I did the thing")
    assert "I did the thing" in prompt
    assert "DONE" in prompt
    assert "STUCK" in prompt
    assert len(prompt) < 500


def test_build_malformed_recovery_truncates():
    """Long previous responses are truncated."""
    long_response = "x" * 1000
    prompt = build_malformed_recovery(long_response)
    assert long_response not in prompt
    # Template itself is ~560 chars; truncation brings total under 600
    assert len(prompt) < 600


def test_empty_output_recovery_constant():
    """EMPTY_OUTPUT_RECOVERY is a useful prompt."""
    assert "DONE" in EMPTY_OUTPUT_RECOVERY
    assert "STUCK" in EMPTY_OUTPUT_RECOVERY


def test_repeated_tool_recovery_format():
    """REPEATED_TOOL_RECOVERY formats correctly."""
    prompt = REPEATED_TOOL_RECOVERY.format(tool_name="read_file", count=3)
    assert "read_file" in prompt
    assert "3" in prompt
    assert "DONE" in prompt


def test_markers_are_strings():
    """All markers are non-empty strings."""
    assert isinstance(DONE_MARKER, str) and DONE_MARKER
    assert isinstance(STUCK_MARKER, str) and STUCK_MARKER
    assert isinstance(TASK_COMPLETE_MARKER, str) and TASK_COMPLETE_MARKER
    assert isinstance(CANNOT_COMPLETE_MARKER, str) and CANNOT_COMPLETE_MARKER
    assert isinstance(NEED_REPLAN_MARKER, str) and NEED_REPLAN_MARKER
