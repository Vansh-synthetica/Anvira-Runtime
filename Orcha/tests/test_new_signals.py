"""Tests for the new deterministic verification signals."""
import os
import sys
import json
import tempfile
import pytest

# Bypass orcha/__init__.py heavy imports by importing submodules directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch out heavy imports that aren't needed for verification
import types
_fake = types.ModuleType("orcha")
_fake.__path__ = [os.path.join(os.path.dirname(__file__), "..", "orcha")]
# sys.modules["orcha"] = _fake

from orcha.core.packets import (
    TaskStep, TaskResult, TaskArtifact, VerificationSignal,
    VerificationResult, ObjectiveVerificationResult,
)
from orcha.nodes.task_executor import DeterministicVerifier


def _step(task_id="s1"):
    return TaskStep(id=task_id, title="Test", objective="Test objective")


# ── File Existence ──────────────────────────────────────────────────────────

def test_file_exists_on_disk():
    step = _step()
    result = TaskResult(step_id="s1", output="Done", success=True)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = f.name
    try:
        artifacts = [TaskArtifact(step_id="s1", kind="file", location=temp_path)]
        signals = DeterministicVerifier.extract_signals(step, result, artifacts)
        file_exists = [s for s in signals if s.signal_type == "file_exists"]
        assert len(file_exists) == 1
        assert file_exists[0].passed is True
    finally:
        os.unlink(temp_path)


def test_file_not_exists_on_disk():
    step = _step()
    result = TaskResult(step_id="s1", output="Done", success=True)
    artifacts = [TaskArtifact(step_id="s1", kind="file", location="/nonexistent/file.txt")]
    signals = DeterministicVerifier.extract_signals(step, result, artifacts)
    file_exists = [s for s in signals if s.signal_type == "file_exists"]
    assert len(file_exists) == 1
    assert file_exists[0].passed is False


# ── Exit Code ───────────────────────────────────────────────────────────────

def test_exit_code_success():
    step = _step()
    result = TaskResult(
        step_id="s1",
        output="Command completed. exit_code: 0",
        success=True,
        tool_calls=[{"tool": "run_command", "args_preview": "ls"}],
    )
    signals = DeterministicVerifier.extract_signals(step, result)
    exit_codes = [s for s in signals if s.signal_type == "command_exit_code"]
    assert len(exit_codes) == 1
    assert exit_codes[0].passed is True
    assert exit_codes[0].metadata["exit_code"] == 0


def test_exit_code_failure():
    step = _step()
    result = TaskResult(
        step_id="s1",
        output="Error: exit_code: 1",
        success=False,
        tool_calls=[{"tool": "run_command", "args_preview": "failing_cmd"}],
    )
    signals = DeterministicVerifier.extract_signals(step, result)
    exit_codes = [s for s in signals if s.signal_type == "command_exit_code"]
    assert len(exit_codes) == 1
    assert exit_codes[0].passed is False
    assert exit_codes[0].metadata["exit_code"] == 1


def test_exit_code_returncode_format():
    step = _step()
    result = TaskResult(
        step_id="s1",
        output="returncode: 0",
        success=True,
        tool_calls=[{"tool": "run_command", "args_preview": "test"}],
    )
    signals = DeterministicVerifier.extract_signals(step, result)
    exit_codes = [s for s in signals if s.signal_type == "command_exit_code"]
    assert len(exit_codes) == 1
    assert exit_codes[0].passed is True


def test_exit_code_no_command_tool():
    step = _step()
    result = TaskResult(
        step_id="s1",
        output="Done",
        success=True,
        tool_calls=[{"tool": "read_file", "args_preview": "{}"}],
    )
    signals = DeterministicVerifier.extract_signals(step, result)
    exit_codes = [s for s in signals if s.signal_type == "command_exit_code"]
    assert len(exit_codes) == 0


def test_exit_code_from_metadata():
    step = _step()
    result = TaskResult(
        step_id="s1",
        output="Done",
        success=True,
        tool_calls=[{"tool": "run_command", "args_preview": "ls", "exit_code": 0}],
    )
    signals = DeterministicVerifier.extract_signals(step, result)
    exit_codes = [s for s in signals if s.signal_type == "command_exit_code"]
    assert len(exit_codes) == 1
    assert exit_codes[0].passed is True


# ── Build Result ────────────────────────────────────────────────────────────

def test_build_success():
    step = _step()
    result = TaskResult(
        step_id="s1",
        output="Build successful. 5 files compiled.",
        success=True,
        tool_calls=[{"tool": "run_command", "args_preview": "npm run build"}],
    )
    signals = DeterministicVerifier.extract_signals(step, result)
    build = [s for s in signals if s.signal_type == "build_result"]
    assert len(build) == 1
    assert build[0].passed is True


def test_build_failure():
    step = _step()
    result = TaskResult(
        step_id="s1",
        output="Build failed. Compilation error in main.ts",
        success=False,
        tool_calls=[{"tool": "run_command", "args_preview": "cargo build"}],
    )
    signals = DeterministicVerifier.extract_signals(step, result)
    build = [s for s in signals if s.signal_type == "build_result"]
    assert len(build) == 1
    assert build[0].passed is False


def test_build_inferred_from_success():
    step = _step()
    result = TaskResult(
        step_id="s1",
        output="Some output from make",
        success=True,
        tool_calls=[{"tool": "run_command", "args_preview": "make"}],
    )
    signals = DeterministicVerifier.extract_signals(step, result)
    build = [s for s in signals if s.signal_type == "build_result"]
    assert len(build) == 1
    assert build[0].passed is True


def test_no_build_tool_no_signal():
    step = _step()
    result = TaskResult(
        step_id="s1",
        output="Done",
        success=True,
        tool_calls=[{"tool": "read_file", "args_preview": "{}"}],
    )
    signals = DeterministicVerifier.extract_signals(step, result)
    build = [s for s in signals if s.signal_type == "build_result"]
    assert len(build) == 0


def test_build_direct_tool_name():
    step = _step()
    result = TaskResult(
        step_id="s1",
        output="Build successful",
        success=True,
        tool_calls=[{"tool": "cargo", "args_preview": "build"}],
    )
    signals = DeterministicVerifier.extract_signals(step, result)
    build = [s for s in signals if s.signal_type == "build_result"]
    assert len(build) == 1
    assert build[0].passed is True


# ── Test Result ─────────────────────────────────────────────────────────────

def test_test_success():
    step = _step()
    result = TaskResult(
        step_id="s1",
        output="10 passed, 0 failed, 0 errors",
        success=True,
        tool_calls=[{"tool": "run_command", "args_preview": "pytest"}],
    )
    signals = DeterministicVerifier.extract_signals(step, result)
    test_r = [s for s in signals if s.signal_type == "test_result"]
    assert len(test_r) == 1
    assert test_r[0].passed is True
    assert test_r[0].metadata["passed"] == 10
    assert test_r[0].metadata["failed"] == 0


def test_test_failure():
    step = _step()
    result = TaskResult(
        step_id="s1",
        output="8 passed, 2 failed, 0 errors",
        success=False,
        tool_calls=[{"tool": "run_command", "args_preview": "pytest"}],
    )
    signals = DeterministicVerifier.extract_signals(step, result)
    test_r = [s for s in signals if s.signal_type == "test_result"]
    assert len(test_r) == 1
    assert test_r[0].passed is False
    assert test_r[0].metadata["failed"] == 2


def test_test_jest_format():
    step = _step()
    result = TaskResult(
        step_id="s1",
        output="Tests: 15 passed, 15 total",
        success=True,
        tool_calls=[{"tool": "run_command", "args_preview": "npx jest"}],
    )
    signals = DeterministicVerifier.extract_signals(step, result)
    test_r = [s for s in signals if s.signal_type == "test_result"]
    assert len(test_r) == 1
    assert test_r[0].passed is True


def test_test_go_ok_format():
    step = _step()
    result = TaskResult(
        step_id="s1",
        output="ok  \tpackage/name\t0.5s",
        success=True,
        tool_calls=[{"tool": "run_command", "args_preview": "go test"}],
    )
    signals = DeterministicVerifier.extract_signals(step, result)
    test_r = [s for s in signals if s.signal_type == "test_result"]
    assert len(test_r) == 1
    assert test_r[0].passed is True


def test_test_cargo_format():
    step = _step()
    result = TaskResult(
        step_id="s1",
        output="test result: 5 passed; 0 failed; 0 ignored",
        success=True,
        tool_calls=[{"tool": "run_command", "args_preview": "cargo test"}],
    )
    signals = DeterministicVerifier.extract_signals(step, result)
    test_r = [s for s in signals if s.signal_type == "test_result"]
    assert len(test_r) == 1
    assert test_r[0].passed is True
    assert test_r[0].metadata["passed"] == 5


def test_no_test_tool_no_signal():
    step = _step()
    result = TaskResult(
        step_id="s1",
        output="Done",
        success=True,
        tool_calls=[{"tool": "write_file", "args_preview": "{}"}],
    )
    signals = DeterministicVerifier.extract_signals(step, result)
    test_r = [s for s in signals if s.signal_type == "test_result"]
    assert len(test_r) == 0


def test_test_direct_tool_name():
    step = _step()
    result = TaskResult(
        step_id="s1",
        output="ALL TESTS PASSED",
        success=True,
        tool_calls=[{"tool": "pytest", "args_preview": ""}],
    )
    signals = DeterministicVerifier.extract_signals(step, result)
    test_r = [s for s in signals if s.signal_type == "test_result"]
    assert len(test_r) == 1
    assert test_r[0].passed is True


# ── Schema Validation ───────────────────────────────────────────────────────

def test_valid_json_object():
    step = _step()
    result = TaskResult(step_id="s1", output='{"key": "value", "count": 42}', success=True)
    signals = DeterministicVerifier.extract_signals(step, result)
    schema = [s for s in signals if s.signal_type == "schema_valid"]
    assert len(schema) == 1
    assert schema[0].passed is True


def test_valid_json_array():
    step = _step()
    result = TaskResult(step_id="s1", output="[1, 2, 3]", success=True)
    signals = DeterministicVerifier.extract_signals(step, result)
    schema = [s for s in signals if s.signal_type == "schema_valid"]
    assert len(schema) == 1
    assert schema[0].passed is True


def test_invalid_json():
    step = _step()
    result = TaskResult(step_id="s1", output='{"key": "value", invalid', success=True)
    signals = DeterministicVerifier.extract_signals(step, result)
    schema = [s for s in signals if s.signal_type == "schema_valid"]
    assert len(schema) == 1
    assert schema[0].passed is False


def test_non_json_no_signal():
    step = _step()
    result = TaskResult(step_id="s1", output="This is plain text", success=True)
    signals = DeterministicVerifier.extract_signals(step, result)
    schema = [s for s in signals if s.signal_type == "schema_valid"]
    assert len(schema) == 0


def test_empty_output_no_signal():
    step = _step()
    result = TaskResult(step_id="s1", output="", success=True)
    signals = DeterministicVerifier.extract_signals(step, result)
    schema = [s for s in signals if s.signal_type == "schema_valid"]
    assert len(schema) == 0
