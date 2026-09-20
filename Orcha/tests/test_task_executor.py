"""Tests for the Orcha adaptive Task Execution Loop."""
import asyncio
import json
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from orcha.core.packets import (
    ExecutionPlan,
    OrchaPacket,
    PacketKind,
    PlannerRequest,
    TaskExecutionResult,
    TaskExecutionState,
    TaskObservation,
    TaskObservationDecision,
    TaskResult,
    TaskStep,
)
from orcha.nodes.task_executor import (
    TaskExecutionLoop,
    FocusedTaskContextBuilder,
    TaskExecutionObserver,
    TaskRetryHandler,
    TaskFinalVerifier,
    AdaptiveReplanner,
    TASK_EXECUTION_SYSTEM_PROMPT,
)
from orcha.graph.context import RunContext


# ── Helper: create a valid TaskStep ──────────────────────────────────────────

def _make_step(
    task_id: str,
    title: str = "Test task",
    objective: str = "Test objective",
    deps: list = None,
    **kwargs,
) -> TaskStep:
    return TaskStep(
        id=task_id,
        title=title,
        objective=objective,
        execution_instructions=kwargs.get("execution_instructions", "Do the thing"),
        dependencies=deps or [],
        expected_outcome=kwargs.get("expected_outcome", "Done"),
        required_context=kwargs.get("required_context", []),
        likely_tools=kwargs.get("likely_tools", []),
        verification_criteria=kwargs.get("verification_criteria", "Verified"),
        action=kwargs.get("action", "tool_use"),
    )


def _make_plan(steps: list = None, **kwargs) -> ExecutionPlan:
    if steps is None:
        steps = [_make_step("step_1")]
    return ExecutionPlan(
        objective=kwargs.get("objective", "Test objective"),
        steps=steps,
        **{k: v for k, v in kwargs.items() if k != "objective" and k != "steps"},
    )


def _make_ctx(cancelled: bool = False) -> MagicMock:
    """Create a mock RunContext with emit support."""
    ctx = MagicMock(spec=RunContext)
    ctx.cancelled = cancelled
    ctx.emit = MagicMock()
    ctx.emit.emit = AsyncMock()
    ctx.run_id = "test_run"
    return ctx


# ── TaskExecutionState ───────────────────────────────────────────────────────

def test_task_execution_state_defaults():
    state = TaskExecutionState(step_id="test_step")
    assert state.step_id == "test_step"
    assert state.status == "pending"
    assert state.iteration == 0
    assert state.max_iterations == 5
    assert state.tool_call_count == 0
    assert state.repeated_tool_calls == 0
    assert state.empty_model_outputs == 0
    assert state.consecutive_errors == 0
    assert state.exhaustion_reason is None


def test_task_execution_state_timeout():
    state = TaskExecutionState(
        step_id="test_step",
        timeout_s=0.1,
        last_activity_time=time.time() - 1.0,
    )
    assert state.is_timed_out is True


def test_task_execution_state_should_retry():
    state = TaskExecutionState(step_id="test_step")
    assert state.should_retry is True

    state.consecutive_errors = 3
    assert state.should_retry is False


def test_task_execution_state_exhaustion_reasons():
    state = TaskExecutionState(step_id="test_step")
    assert state.exhaustion_reason is None

    state.consecutive_errors = 3
    assert state.exhaustion_reason == "consecutive_model_errors"

    state.consecutive_errors = 0
    state.repeated_tool_calls = 3
    assert state.exhaustion_reason == "repeated_tool_calls"

    state.repeated_tool_calls = 0
    state.empty_model_outputs = 3
    assert state.exhaustion_reason == "empty_model_outputs"

    state.empty_model_outputs = 0
    state.malformed_model_outputs = 3
    assert state.exhaustion_reason == "malformed_model_outputs"

    state.malformed_model_outputs = 0
    state.iteration = 5
    state.max_iterations = 5
    assert state.exhaustion_reason == "max_iterations_exceeded"


def test_task_execution_state_record_tool_call():
    state = TaskExecutionState(step_id="test_step")
    state.record_tool_call("read_file", '{"path": "test.py"}')
    assert state.tool_call_count == 1
    assert state.last_tool_call == "read_file"
    assert len(state.tool_call_history) == 1

    # Same call again
    state.record_tool_call("read_file", '{"path": "test.py"}')
    assert state.repeated_tool_calls == 1

    # Different call
    state.record_tool_call("write_file", '{"path": "test.py"}')
    assert state.repeated_tool_calls == 0


def test_task_execution_state_record_model_output():
    state = TaskExecutionState(step_id="test_step")
    state.record_model_output("Hello world")
    assert state.empty_model_outputs == 0
    assert state.consecutive_errors == 0

    state.record_model_output("")
    assert state.empty_model_outputs == 1
    assert state.consecutive_errors == 1

    state.record_model_output("Hello again")
    assert state.empty_model_outputs == 0
    assert state.consecutive_errors == 0


# ── TaskExecutionResult ──────────────────────────────────────────────────────

def test_task_execution_result_from_state():
    state = TaskExecutionState(step_id="test_step")
    state.iteration = 3
    state.tool_call_count = 5
    state.start_time = time.time() - 10.0

    result = TaskExecutionResult.from_state(state, "Task completed", True)
    assert result.step_id == "test_step"
    assert result.success is True
    assert result.output == "Task completed"
    assert result.tool_calls_made == 5
    assert result.iterations_used == 3
    assert result.duration_s > 0


def test_task_execution_result_from_state_failure():
    state = TaskExecutionState(step_id="test_step")
    state.record_error("Model failed")

    result = TaskExecutionResult.from_state(state, "", False)
    assert result.success is False
    assert result.error == "Model failed"


# ── TaskExecutionLoop ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execution_loop_no_plan():
    loop = TaskExecutionLoop()
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    ctx = _make_ctx()
    result = await loop.run(packet, ctx)
    assert "task_execution_result" in result.payload
    assert result.payload["task_execution_result"]["success"] is False


@pytest.mark.asyncio
async def test_execution_loop_no_model():
    plan = _make_plan([_make_step("step_1")])
    loop = TaskExecutionLoop(completion_fn=None)
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = _make_ctx()
    result = await loop.run(packet, ctx)
    assert "task_execution_result" in result.payload
    assert result.payload["task_execution_result"]["success"] is False


@pytest.mark.asyncio
async def test_execution_loop_task_complete():
    async def mock_completion(messages, system, tools):
        return {"content": "TASK COMPLETE: Done what was asked"}

    plan = _make_plan([_make_step("step_1", title="Simple task")])
    loop = TaskExecutionLoop(completion_fn=mock_completion)
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = _make_ctx()
    ctx.cancelled = False
    result = await loop.run(packet, ctx)
    assert result.payload["task_step_status"] == "completed"
    assert "Done what was asked" in result.payload["task_execution_result"]["output"]


@pytest.mark.asyncio
async def test_execution_loop_tool_execution():
    call_count = 0

    async def mock_completion(messages, system, tools):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: request tool use
            return {
                "content": "Let me read the file",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "test.py"}),
                    },
                }],
            }
        else:
            # Second call: task complete
            return {"content": "TASK COMPLETE: Read the file successfully"}

    plan = _make_plan([_make_step("step_1", likely_tools=["read_file"])])
    loop = TaskExecutionLoop(completion_fn=mock_completion)
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = _make_ctx()
    ctx.cancelled = False
    result = await loop.run(packet, ctx)
    assert result.payload["task_step_status"] == "completed"
    assert call_count == 2


@pytest.mark.asyncio
async def test_execution_loop_repeated_tool_calls():
    call_count = 0

    async def mock_completion(messages, system, tools):
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            # Keep calling same tool
            return {
                "content": "Let me try again",
                "tool_calls": [{
                    "id": f"call_{call_count}",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "test.py"}),
                    },
                }],
            }
        else:
            return {"content": "TASK COMPLETE: Finally done"}

    plan = _make_plan([_make_step("step_1")])
    loop = TaskExecutionLoop(
        completion_fn=mock_completion,
        max_tool_repeats=2,
    )
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = _make_ctx()
    ctx.cancelled = False
    result = await loop.run(packet, ctx)
    # Should have stopped after repeated calls
    assert "task_execution_result" in result.payload


@pytest.mark.asyncio
async def test_execution_loop_empty_output():
    call_count = 0

    async def mock_completion(messages, system, tools):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return {"content": ""}
        else:
            return {"content": "TASK COMPLETE: Done"}

    plan = _make_plan([_make_step("step_1")])
    loop = TaskExecutionLoop(
        completion_fn=mock_completion,
        max_empty_outputs=2,
    )
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = _make_ctx()
    ctx.cancelled = False
    result = await loop.run(packet, ctx)
    assert "task_execution_result" in result.payload


@pytest.mark.asyncio
async def test_execution_loop_inability():
    async def mock_completion(messages, system, tools):
        return {"content": "CANNOT COMPLETE: Missing required permissions"}

    plan = _make_plan([_make_step("step_1")])
    loop = TaskExecutionLoop(completion_fn=mock_completion)
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = _make_ctx()
    ctx.cancelled = False
    result = await loop.run(packet, ctx)
    assert result.payload["task_step_status"] == "failed"
    assert "Missing required permissions" in result.payload["task_execution_result"]["output"]
    # Regression: the model's own stated reason must reach `error` too, not
    # just `output` — otherwise every downstream failure report (the
    # task-failed event, the UI's "unknown" decision reason) has nothing
    # to show, since they read `exhaustion_reason or error or "unknown"`
    # and `output` isn't part of that chain. Confirmed as a real bug via a
    # live run: a local model that reported "stuck" via this exact
    # CANNOT COMPLETE marker surfaced as a bare "unknown" error with no
    # indication of what actually went wrong.
    assert "Missing required permissions" in result.payload["task_execution_result"]["error"]


@pytest.mark.asyncio
async def test_execution_loop_appends_force_tool_marker_until_first_tool_call():
    """FORCE_TOOL_CALL_MARKER must be present on the system prompt for
    every call until this task has made a real tool call, then absent
    afterward — that's what lets the model relax back to a normal
    (unconstrained) response for a task's closing turn after real
    progress has already happened."""
    from orcha.nodes.task_executor import FORCE_TOOL_CALL_MARKER

    systems_seen = []

    async def mock_completion(messages, system, tools):
        systems_seen.append(system)
        if len(systems_seen) == 1:
            # First turn: actually call the tool.
            return {
                "content": "",
                "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "write_file", "arguments": '{"path": "x.txt", "content": "hi"}'},
                }],
            }
        return {"content": "TASK COMPLETE: wrote the file"}

    step = _make_step("step_1", likely_tools=["write_file"], action="tool_use")
    plan = _make_plan([step])
    loop = TaskExecutionLoop(completion_fn=mock_completion, max_task_iterations=5)
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = _make_ctx()
    await loop.run(packet, ctx)

    assert len(systems_seen) >= 2
    assert systems_seen[0].endswith(FORCE_TOOL_CALL_MARKER)
    assert not systems_seen[-1].endswith(FORCE_TOOL_CALL_MARKER)


@pytest.mark.asyncio
async def test_execution_loop_task_complete_marker_without_tool_call_gets_corrected():
    """Regression: a real live run confirmed the model can write 'TASK
    COMPLETE: created snake.html' without ever calling write_file, and the
    loop accepted it immediately (tool_calls_made=0, iterations_used=1) —
    the "TASK COMPLETE:"/"DONE:" markers bypassed the corrective-retry gate
    entirely (that gate only covered the generic no-marker prose path).
    Downstream, the run reported 'VERIFIED... Objective achieved' with
    nothing on disk. The marker must get the same one corrective round-trip
    before being trusted, when the step expected a tool call and none has
    happened yet."""
    calls = []

    async def mock_completion(messages, system, tools):
        calls.append(messages)
        return {"content": "TASK COMPLETE: created the file with the requested content."}

    step = _make_step("step_1", likely_tools=[], action="tool_use")
    plan = _make_plan([step])
    loop = TaskExecutionLoop(completion_fn=mock_completion, max_task_iterations=5)
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = _make_ctx()
    result = await loop.run(packet, ctx)
    assert len(calls) >= 2, "the false TASK COMPLETE claim must trigger a corrective retry, not be accepted on turn 1"
    last_messages = calls[-1]
    assert any(
        m.get("role") == "user" and "no tool was" in m.get("content", "")
        for m in last_messages
    )


@pytest.mark.asyncio
async def test_execution_loop_corrective_retry_without_likely_tools():
    """Regression: the corrective-retry (one nudge to actually call a tool
    before accepting a text-only answer) used to gate solely on
    `step.likely_tools` being non-empty. A live run confirmed a real
    planner can leave `likely_tools` empty on a step whose `action` is
    still the default "tool_use" — that step then sailed straight through
    as "completed" on the model's very first prose response, zero tool
    calls made, TaskExecutionObserver's fast path accepting it as success
    with nothing to show for it. `step.action` must also gate this, not
    just `likely_tools`."""
    calls = []

    async def mock_completion(messages, system, tools):
        calls.append(messages)
        return {"content": "Here is a description of what the file would contain, in prose."}

    step = _make_step("step_1", likely_tools=[], action="tool_use")
    plan = _make_plan([step])
    loop = TaskExecutionLoop(completion_fn=mock_completion, max_task_iterations=5)
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = _make_ctx()
    result = await loop.run(packet, ctx)
    # At least 2 completion calls: the corrective nudge must have fired
    # instead of accepting the very first prose response as done.
    assert len(calls) >= 2
    last_messages = calls[-1]
    assert any(
        m.get("role") == "user" and "instead of calling a tool" in m.get("content", "")
        for m in last_messages
    )


@pytest.mark.asyncio
async def test_execution_loop_cancellation():
    async def mock_completion(messages, system, tools):
        return {"content": "TASK COMPLETE: Done"}

    plan = _make_plan([_make_step("step_1")])
    loop = TaskExecutionLoop(completion_fn=mock_completion)
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = _make_ctx()
    ctx.cancelled = True  # Cancel immediately
    result = await loop.run(packet, ctx)
    assert "task_execution_result" in result.payload


# ── FocusedTaskContextBuilder ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_focused_context_builder():
    builder = FocusedTaskContextBuilder()
    step = _make_step(
        "step_1",
        title="Find files",
        objective="Find all Python files",
        execution_instructions="List directory and filter",
        expected_outcome="List of Python files",
        verification_criteria="Found files match *.py",
        likely_tools=["list_directory"],
    )
    plan = _make_plan([step])
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = _make_ctx()
    result = await builder.run(packet, ctx)
    assert result.payload["task_step_id"] == "step_1"
    context = result.payload["task_step_context"]
    assert "Find files" in context
    assert "Find all Python files" in context
    assert "list_directory" in context


@pytest.mark.asyncio
async def test_focused_context_builder_with_prior_results():
    builder = FocusedTaskContextBuilder()
    step1 = _make_step("step_1", title="Step 1")
    step1.status = "completed"
    step1.result = TaskResult(step_id="step_1", output="Found 5 files", success=True)
    step2 = _make_step("step_2", title="Step 2", deps=["step_1"], required_context=["step_1"])
    plan = _make_plan([step1, step2])
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = _make_ctx()
    result = await builder.run(packet, ctx)
    context = result.payload["task_step_context"]
    assert "Found 5 files" in context


@pytest.mark.asyncio
async def test_focused_context_builder_no_plan():
    builder = FocusedTaskContextBuilder()
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    ctx = _make_ctx()
    result = await builder.run(packet, ctx)
    assert "task_step_id" not in result.payload


# ── TaskExecutionObserver ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execution_observer_success():
    observer = TaskExecutionObserver()
    step = _make_step("step_1")
    step.status = "in_progress"
    plan = _make_plan([step])
    result = TaskExecutionResult(
        step_id="step_1",
        success=True,
        output="Task completed",
    )
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    packet.payload["task_execution_result"] = result.model_dump()
    ctx = _make_ctx()
    response = await observer.run(packet, ctx)
    assert response.payload["task_action"] == "verify"


@pytest.mark.asyncio
async def test_execution_observer_failure_retry():
    observer = TaskExecutionObserver()
    step = _make_step("step_1")
    step.status = "in_progress"
    plan = _make_plan([step])
    result = TaskExecutionResult(
        step_id="step_1",
        success=False,
        output="",
        exhaustion_reason="timeout",
        error="Timed out",
    )
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    packet.payload["task_execution_result"] = result.model_dump()
    ctx = _make_ctx()
    response = await observer.run(packet, ctx)
    assert response.payload["task_action"] == "retry"


@pytest.mark.asyncio
async def test_execution_observer_failure_modify():
    observer = TaskExecutionObserver()
    step = _make_step("step_1")
    step.status = "in_progress"
    plan = _make_plan([step], max_replans=3)
    result = TaskExecutionResult(
        step_id="step_1",
        success=False,
        output="",
        exhaustion_reason="repeated_tool_calls",
        error="Tool calls repeated",
    )
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    packet.payload["task_execution_result"] = result.model_dump()
    ctx = _make_ctx()
    response = await observer.run(packet, ctx)
    # repeated_tool_calls triggers MODIFY when replan budget available
    assert response.payload["task_action"] == "modify"


# ── TaskRetryHandler ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retry_handler_retry():
    handler = TaskRetryHandler()
    step = _make_step("step_1")
    step.status = "failed"
    plan = _make_plan([step], max_replans=3)
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    packet.payload["task_action"] = "retry"
    ctx = _make_ctx()
    result = await handler.run(packet, ctx)
    assert result.payload["task_action"] == "retry_same"
    assert result.payload["retry_count"] == 1


@pytest.mark.asyncio
async def test_retry_handler_replan():
    handler = TaskRetryHandler()
    step = _make_step("step_1")
    step.status = "failed"
    plan = _make_plan([step], max_replans=3)
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    packet.payload["task_action"] = "replan"
    ctx = _make_ctx()
    result = await handler.run(packet, ctx)
    assert result.payload["task_action"] == "next_task"


# ── TaskFinalVerifier ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_final_verifier_all_complete():
    verifier = TaskFinalVerifier()
    step1 = _make_step("step_1")
    step1.status = "completed"
    step1.result = TaskResult(step_id="step_1", output="TASK COMPLETE: Done", success=True)
    step2 = _make_step("step_2")
    step2.status = "completed"
    step2.result = TaskResult(step_id="step_2", output="TASK COMPLETE: Done", success=True)
    plan = _make_plan([step1, step2])
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = _make_ctx()
    result = await verifier.run(packet, ctx)
    assert result.payload["objective_met"] is True
    assert result.payload["completed_tasks"] == 2
    assert result.payload["failed_tasks"] == 0


@pytest.mark.asyncio
async def test_final_verifier_some_failed():
    verifier = TaskFinalVerifier()
    step1 = _make_step("step_1")
    step1.status = "completed"
    step1.result = TaskResult(step_id="step_1", output="TASK COMPLETE: Done", success=True)
    step2 = _make_step("step_2")
    step2.status = "failed"
    step2.result = TaskResult(step_id="step_2", output="", success=False, error="Failed")
    plan = _make_plan([step1, step2])
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = _make_ctx()
    result = await verifier.run(packet, ctx)
    assert result.payload["objective_met"] is False
    assert result.payload["completed_tasks"] == 1
    assert result.payload["failed_tasks"] == 1


@pytest.mark.asyncio
async def test_final_verifier_no_plan():
    verifier = TaskFinalVerifier()
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    ctx = _make_ctx()
    result = await verifier.run(packet, ctx)
    assert result.payload["objective_met"] is False


# ── System prompt ────────────────────────────────────────────────────────────

def test_system_prompt_format():
    prompt = TASK_EXECUTION_SYSTEM_PROMPT.format(
        original_request="Build a CLI tool that finds and lists all Python files in a repo",
        task_objective="Find all Python files",
        execution_instructions="List directory and filter",
        expected_outcome="List of Python files",
        verification_criteria="Found files match *.py",
        constraints="No external dependencies",
        workspace_state="Git repo with 50 files",
    )
    assert "Find all Python files" in prompt
    assert "List directory and filter" in prompt
    assert "No external dependencies" in prompt
    assert "You are executing a specific task" in prompt
    assert "You are NOT working on the entire project" in prompt
    # Regression: the step-level fields alone can be thinner than the real
    # requirements a weaker planner model compressed away — the original
    # request must always be present as a fallback source of truth.
    assert "Build a CLI tool that finds and lists all Python files" in prompt


# ══════════════════════════════════════════════════════════════════════════════
# Focused message building tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFocusedMessageBuilding:
    """Tests for focused message building that prevents context overflow."""

    def test_build_focused_messages_few_messages(self):
        """With few messages, use them all."""
        loop = TaskExecutionLoop()
        messages = [
            {"role": "user", "content": "Do task"},
            {"role": "assistant", "content": "I'll do it"},
        ]
        result = loop._build_focused_messages(messages, [], 1)
        assert len(result) == 2

    def test_build_focused_messages_many_messages(self):
        """With many messages, use focused window."""
        loop = TaskExecutionLoop()
        messages = [
            {"role": "user", "content": "Do task"},
            {"role": "assistant", "content": "Step 1"},
            {"role": "tool", "content": "Result 1"},
            {"role": "assistant", "content": "Step 2"},
            {"role": "tool", "content": "Result 2"},
            {"role": "assistant", "content": "Step 3"},
            {"role": "tool", "content": "Result 3"},
            {"role": "assistant", "content": "Step 4"},
        ]
        result = loop._build_focused_messages(messages, [], 4)
        # Should have first message + last 4
        assert result[0]["content"] == "Do task"
        assert len(result) <= 6  # First + recent

    def test_build_focused_messages_with_tool_summary(self):
        """Tool results summary is included."""
        loop = TaskExecutionLoop()
        messages = [
            {"role": "user", "content": "Do task"},
            {"role": "assistant", "content": "Step 1"},
            {"role": "tool", "content": "Result 1"},
            {"role": "assistant", "content": "Step 2"},
            {"role": "tool", "content": "Result 2"},
            {"role": "assistant", "content": "Step 3"},
            {"role": "tool", "content": "Result 3"},
            {"role": "assistant", "content": "Step 4"},
        ]
        tool_summary = ["[read_file] Found 5 files", "[edit_file] Modified test.py"]
        result = loop._build_focused_messages(messages, tool_summary, 4)
        # Should include tool summary
        has_summary = any("Tool results so far" in m.get("content", "") for m in result)
        assert has_summary

    def test_build_focused_messages_tool_summary_limited(self):
        """Tool results summary is limited to last 5."""
        loop = TaskExecutionLoop()
        messages = [
            {"role": "user", "content": "Do task"},
            {"role": "assistant", "content": "Done"},
        ]
        tool_summary = [f"[tool_{i}] result_{i}" for i in range(10)]
        result = loop._build_focused_messages(messages, tool_summary, 1)
        # Find the summary message
        for m in result:
            if "Tool results so far" in m.get("content", ""):
                # Should only have 5 tool results
                assert m["content"].count("[tool_") <= 5
                break


# ══════════════════════════════════════════════════════════════════════════════
# Focused context building tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFocusedContextBuilding:
    """Tests for focused context building."""

    def test_build_focused_context_includes_only_relevant_results(self):
        """Only dependency results are included, not all prior results."""
        loop = TaskExecutionLoop()
        step1 = _make_step("step_1", title="Step 1")
        step1.status = "completed"
        step1.result = TaskResult(step_id="step_1", output="Result 1", success=True)
        step2 = _make_step("step_2", title="Step 2")
        step2.status = "completed"
        step2.result = TaskResult(step_id="step_2", output="Result 2", success=True)
        # step_3 depends on step_1 only (not step_2)
        step3 = _make_step("step_3", title="Step 3", deps=["step_1"])
        plan = _make_plan([step1, step2, step3])
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")

        context = loop._build_focused_context(plan, step3, packet)
        # Should only include step_1 result, not step_2
        assert len(context["prior_results"]) == 1
        assert "Result 1" in context["prior_results"][0]

    def test_build_focused_context_truncates_output(self):
        """Prior results are truncated to prevent overflow."""
        loop = TaskExecutionLoop()
        step1 = _make_step("step_1", title="Step 1")
        step1.status = "completed"
        step1.result = TaskResult(
            step_id="step_1",
            output="x" * 1000,  # Long output
            success=True,
        )
        step2 = _make_step("step_2", title="Step 2", deps=["step_1"])
        plan = _make_plan([step1, step2])
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")

        context = loop._build_focused_context(plan, step2, packet)
        # Output should be truncated to 500 chars
        assert len(context["prior_results"][0]) < 600

    def test_build_focused_context_includes_plan_progress(self):
        """Context includes plan progress information."""
        loop = TaskExecutionLoop()
        step1 = _make_step("step_1", title="Step 1")
        step1.status = "completed"
        step2 = _make_step("step_2", title="Step 2")
        plan = _make_plan([step1, step2])
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")

        context = loop._build_focused_context(plan, step2, packet)
        assert "plan_progress" in context
        assert "1/2" in context["plan_progress"]


# ══════════════════════════════════════════════════════════════════════════════
# User message building tests
# ══════════════════════════════════════════════════════════════════════════════

class TestUserMessageBuilding:
    """Tests for user message building."""

    def test_build_user_message_includes_task_focus(self):
        """User message explicitly tells model it's executing current task."""
        loop = TaskExecutionLoop()
        context = {
            "step_title": "Find files",
            "step_objective": "Find all Python files",
            "prior_results": [],
            "likely_tools": ["list_directory"],
        }
        message = loop._build_user_message(context)
        assert "current task" in message.lower()
        assert "not the entire overall project" in message.lower()

    def test_build_user_message_includes_prior_results(self):
        """User message includes required prior results."""
        loop = TaskExecutionLoop()
        context = {
            "step_title": "Process files",
            "step_objective": "Process the found files",
            "prior_results": ["[step_1] Found 5 files"],
            "likely_tools": [],
        }
        message = loop._build_user_message(context)
        assert "Found 5 files" in message

    def test_build_user_message_includes_completion_instructions(self):
        """User message includes how to report completion."""
        loop = TaskExecutionLoop()
        context = {
            "step_title": "Do task",
            "step_objective": "Do something",
            "prior_results": [],
            "likely_tools": [],
        }
        message = loop._build_user_message(context)
        assert "TASK COMPLETE:" in message
        assert "CANNOT COMPLETE:" in message


# ══════════════════════════════════════════════════════════════════════════════
# Observable events tests
# ══════════════════════════════════════════════════════════════════════════════

class TestObservableEvents:
    """Tests for observable execution events."""

    @pytest.mark.asyncio
    async def test_execution_emits_task_start_event(self):
        """Execution emits task_start event."""
        async def mock_completion(messages, system, tools):
            return {"content": "TASK COMPLETE: Done"}

        plan = _make_plan([_make_step("step_1")])
        loop = TaskExecutionLoop(completion_fn=mock_completion)
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        ctx = _make_ctx()
        ctx.cancelled = False
        await loop.run(packet, ctx)

        # Check that emit was called with task_start
        calls = ctx.emit.emit.call_args_list
        event_types = [call[0][0] for call in calls]
        assert "task_start" in event_types

    @pytest.mark.asyncio
    async def test_execution_emits_task_complete_event(self):
        """Execution emits task_complete event."""
        async def mock_completion(messages, system, tools):
            return {"content": "TASK COMPLETE: Done"}

        plan = _make_plan([_make_step("step_1")])
        loop = TaskExecutionLoop(completion_fn=mock_completion)
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        ctx = _make_ctx()
        ctx.cancelled = False
        await loop.run(packet, ctx)

        calls = ctx.emit.emit.call_args_list
        event_types = [call[0][0] for call in calls]
        assert "task_complete" in event_types

    @pytest.mark.asyncio
    async def test_execution_emits_text_delta_for_model_response(self):
        """Execution emits text_delta event for model responses."""
        async def mock_completion(messages, system, tools):
            return {"content": "Let me think about this..."}

        plan = _make_plan([_make_step("step_1")])
        loop = TaskExecutionLoop(completion_fn=mock_completion)
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        ctx = _make_ctx()
        ctx.cancelled = False
        await loop.run(packet, ctx)

        calls = ctx.emit.emit.call_args_list
        event_types = [call[0][0] for call in calls]
        assert "text_delta" in event_types

    @pytest.mark.asyncio
    async def test_execution_emits_events_for_tool_execution(self):
        """Execution emits events for tool execution."""
        call_count = 0

        async def mock_completion(messages, system, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "content": "Let me read the file",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "test.py"}),
                        },
                    }],
                }
            else:
                return {"content": "TASK COMPLETE: Done"}

        plan = _make_plan([_make_step("step_1", likely_tools=["read_file"])])
        loop = TaskExecutionLoop(completion_fn=mock_completion)
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        ctx = _make_ctx()
        ctx.cancelled = False
        await loop.run(packet, ctx)

        calls = ctx.emit.emit.call_args_list
        event_types = [call[0][0] for call in calls]
        assert "task_modified" in event_types


# ══════════════════════════════════════════════════════════════════════════════
# Safeguard tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSafeguards:
    """Tests for execution safeguards."""

    @pytest.mark.asyncio
    async def test_repeated_tool_calls_stops_execution(self):
        """Repeated identical tool calls trigger safeguard."""
        call_count = 0

        async def mock_completion(messages, system, tools):
            nonlocal call_count
            call_count += 1
            # Keep calling same tool with same args
            return {
                "content": "Let me try again",
                "tool_calls": [{
                    "id": f"call_{call_count}",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "test.py"}),
                    },
                }],
            }

        plan = _make_plan([_make_step("step_1")])
        loop = TaskExecutionLoop(
            completion_fn=mock_completion,
            max_tool_repeats=2,
            max_task_iterations=5,
        )
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        ctx = _make_ctx()
        ctx.cancelled = False
        result = await loop.run(packet, ctx)

        # Should have stopped due to repeated calls
        exec_result = result.payload["task_execution_result"]
        # The loop should have terminated
        assert call_count <= 5  # Not infinite

    @pytest.mark.asyncio
    async def test_empty_output_triggers_retry(self):
        """Empty model output triggers retry prompt."""
        call_count = 0

        async def mock_completion(messages, system, tools):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return {"content": ""}
            else:
                return {"content": "TASK COMPLETE: Done"}

        plan = _make_plan([_make_step("step_1")])
        loop = TaskExecutionLoop(
            completion_fn=mock_completion,
            max_empty_outputs=2,
        )
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        ctx = _make_ctx()
        ctx.cancelled = False
        result = await loop.run(packet, ctx)

        # Should have eventually completed
        assert result.payload["task_step_status"] == "completed"

    @pytest.mark.asyncio
    async def test_cancellation_stops_execution(self):
        """Cancellation immediately stops execution."""
        async def mock_completion(messages, system, tools):
            return {"content": "TASK COMPLETE: Done"}

        plan = _make_plan([_make_step("step_1")])
        loop = TaskExecutionLoop(completion_fn=mock_completion)
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        ctx = _make_ctx()
        ctx.cancelled = True  # Cancel immediately
        result = await loop.run(packet, ctx)

        exec_result = result.payload["task_execution_result"]
        assert exec_result["success"] is False

    @pytest.mark.asyncio
    async def test_timeout_stops_execution(self):
        """Timeout stops execution."""
        import time

        async def mock_completion(messages, system, tools):
            # Simulate slow model that takes time
            await asyncio.sleep(0.05)
            return {"content": "TASK COMPLETE: Done"}

        plan = _make_plan([_make_step("step_1")])
        # Create state with already-timed-out timestamp
        loop = TaskExecutionLoop(
            completion_fn=mock_completion,
            timeout_s=0.01,
        )
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        ctx = _make_ctx()
        ctx.cancelled = False

        # Pre-set the timeout by modifying the state after creation
        # We'll test the timeout check logic directly
        from orcha.core.packets import TaskExecutionState
        state = TaskExecutionState(
            step_id="step_1",
            timeout_s=0.01,
            last_activity_time=time.time() - 1.0,  # Already timed out
        )
        assert state.is_timed_out is True

    @pytest.mark.asyncio
    async def test_model_error_triggers_retry(self):
        """Model error triggers retry."""
        call_count = 0

        async def mock_completion(messages, system, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Model unavailable")
            else:
                return {"content": "TASK COMPLETE: Done after retry"}

        plan = _make_plan([_make_step("step_1")])
        loop = TaskExecutionLoop(
            completion_fn=mock_completion,
            max_consecutive_errors=2,
        )
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        ctx = _make_ctx()
        ctx.cancelled = False
        result = await loop.run(packet, ctx)

        # Should have retried and succeeded
        assert result.payload["task_step_status"] == "completed"

    @pytest.mark.asyncio
    async def test_inability_reported(self):
        """Model can report inability to complete task."""
        async def mock_completion(messages, system, tools):
            return {"content": "CANNOT COMPLETE: Missing required file access"}

        plan = _make_plan([_make_step("step_1")])
        loop = TaskExecutionLoop(completion_fn=mock_completion)
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        ctx = _make_ctx()
        ctx.cancelled = False
        result = await loop.run(packet, ctx)

        assert result.payload["task_step_status"] == "failed"
        assert "Missing required file access" in result.payload["task_execution_result"]["output"]

    @pytest.mark.asyncio
    async def test_replanning_request(self):
        """Model can request replanning."""
        async def mock_completion(messages, system, tools):
            return {"content": "NEED REPLAN: The task scope has changed"}

        plan = _make_plan([_make_step("step_1")])
        loop = TaskExecutionLoop(completion_fn=mock_completion)
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        ctx = _make_ctx()
        ctx.cancelled = False
        result = await loop.run(packet, ctx)

        assert result.payload["task_step_status"] == "failed"
        assert "task_execution_result" in result.payload


# ══════════════════════════════════════════════════════════════════════════════
# Multi-step execution tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiStepExecution:
    """Tests for multi-step execution scenarios."""

    @pytest.mark.asyncio
    async def test_sequential_task_execution(self):
        """Execute multiple tasks in sequence."""
        call_count = 0

        async def mock_completion(messages, system, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"content": "TASK COMPLETE: Step 1 done"}
            elif call_count == 2:
                return {"content": "TASK COMPLETE: Step 2 done"}
            else:
                return {"content": "TASK COMPLETE: Step 3 done"}

        step1 = _make_step("step_1", title="Step 1")
        step2 = _make_step("step_2", title="Step 2", deps=["step_1"])
        step3 = _make_step("step_3", title="Step 3", deps=["step_2"])
        plan = _make_plan([step1, step2, step3])

        loop = TaskExecutionLoop(completion_fn=mock_completion)
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        ctx = _make_ctx()
        ctx.cancelled = False

        # Execute first task
        result = await loop.run(packet, ctx)
        assert result.payload["task_step_status"] == "completed"

        # Update plan in packet for next execution
        packet.payload["execution_plan"] = result.payload["execution_plan"]

        # Execute second task
        result = await loop.run(packet, ctx)
        assert result.payload["task_step_status"] == "completed"

        # Update plan and execute third
        packet.payload["execution_plan"] = result.payload["execution_plan"]
        result = await loop.run(packet, ctx)
        assert result.payload["task_step_status"] == "completed"

    @pytest.mark.asyncio
    async def test_plan_completion_detection(self):
        """Plan completion is detected when all tasks complete."""
        async def mock_completion(messages, system, tools):
            return {"content": "TASK COMPLETE: Done"}

        step1 = _make_step("step_1")
        plan = _make_plan([step1])

        loop = TaskExecutionLoop(completion_fn=mock_completion)
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        ctx = _make_ctx()
        ctx.cancelled = False

        result = await loop.run(packet, ctx)
        # After completing the only step, plan should be in progress
        assert "execution_plan" in result.payload


# ══════════════════════════════════════════════════════════════════════════════
# Tool execution tests
# ══════════════════════════════════════════════════════════════════════════════

class TestToolExecution:
    """Tests for tool execution integration."""

    @pytest.mark.asyncio
    async def test_tool_execution_with_no_executor(self):
        """Tool execution fails gracefully with no executor."""
        call_count = 0

        async def mock_completion(messages, system, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "content": "Let me use a tool",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "test.py"}),
                        },
                    }],
                }
            else:
                return {"content": "TASK COMPLETE: Done"}

        plan = _make_plan([_make_step("step_1")])
        loop = TaskExecutionLoop(
            completion_fn=mock_completion,
            executor=None,  # No executor
        )
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        ctx = _make_ctx()
        ctx.cancelled = False

        result = await loop.run(packet, ctx)
        # Should handle gracefully
        assert "task_execution_result" in result.payload

    @pytest.mark.asyncio
    async def test_tool_execution_with_mock_executor(self):
        """Tool execution works with mock executor."""
        from unittest.mock import MagicMock
        from orcha.capabilities.base import ToolResult

        call_count = 0

        async def mock_completion(messages, system, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "content": "Let me read the file",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "test.py"}),
                        },
                    }],
                }
            else:
                return {"content": "TASK COMPLETE: File read successfully"}

        # Create mock executor — TaskExecutionLoop calls the real
        # ToolExecutor's synchronous `invoke(tool_name, **kwargs)`, not
        # `execute(tool_name, args_dict)` (that method doesn't exist on the
        # real class at all; see capabilities/base.py's ToolExecutor).
        mock_executor = MagicMock()
        mock_executor.schemas.return_value = [
            {"name": "read_file", "description": "Read a file"}
        ]
        mock_executor.invoke = MagicMock(
            return_value=ToolResult.success("file content here")
        )

        plan = _make_plan([_make_step("step_1", likely_tools=["read_file"])])
        loop = TaskExecutionLoop(
            completion_fn=mock_completion,
            executor=mock_executor,
        )
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        ctx = _make_ctx()
        ctx.cancelled = False

        result = await loop.run(packet, ctx)
        assert result.payload["task_step_status"] == "completed"
        # Verify executor was called
        mock_executor.invoke.assert_called_once_with("read_file", path="test.py")

    @pytest.mark.asyncio
    async def test_execution_loop_uses_step_already_selected_by_context_builder(self):
        """
        Regression test for a status-mutation race: TaskContextBuilderNode
        (planner.py) selects the next pending step, flips it to
        "in_progress", and writes task_step_id into the packet — one hop
        before TaskExecutionLoop runs. TaskExecutionLoop used to ignore
        task_step_id and independently re-call next_pending_step(), which
        only matches status == "pending" and so always found nothing (the
        step was already "in_progress"), silently no-opping the entire task:
        zero tool calls, no task_execution_result, straight to
        task_complete=True. This runs the two nodes in the same sequence the
        real graph does and asserts the task actually gets executed.
        """
        from orcha.nodes.planner import TaskContextBuilderNode
        from orcha.capabilities.base import ToolResult

        async def mock_completion(messages, system, tools):
            return {"content": "TASK COMPLETE: done"}

        mock_executor = MagicMock()
        mock_executor.schemas.return_value = []
        mock_executor.invoke = MagicMock(return_value=ToolResult.success("ok"))

        plan = _make_plan([_make_step("step_1")])
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        ctx = _make_ctx()
        ctx.cancelled = False

        context_builder = TaskContextBuilderNode()
        after_context = await context_builder.run(packet, ctx)
        # Confirm the race precondition: the step really is "in_progress"
        # and task_step_id really is set, exactly as TaskExecutionLoop
        # receives it in the real graph.
        built_plan = ExecutionPlan(**after_context.payload["execution_plan"])
        assert built_plan.step_by_id("step_1").status == "in_progress"
        assert after_context.payload["task_step_id"] == "step_1"

        loop = TaskExecutionLoop(completion_fn=mock_completion, executor=mock_executor)
        result = await loop.run(after_context, ctx)

        assert "task_execution_result" in result.payload, (
            "TaskExecutionLoop silently no-opped instead of executing the "
            "step TaskContextBuilderNode had already selected"
        )
        exec_result = TaskExecutionResult(**result.payload["task_execution_result"])
        assert exec_result.step_id == "step_1"
        assert exec_result.success is True


# ══════════════════════════════════════════════════════════════════════════════
# PlanIntegrityChecker tests
# ══════════════════════════════════════════════════════════════════════════════

from orcha.nodes.task_executor import PlanIntegrityChecker


class TestPlanIntegrityChecker:
    """Tests for plan integrity validation after replanning."""

    def test_validate_valid_plan(self):
        """Valid plan passes validation."""
        plan = _make_plan([
            _make_step("a"),
            _make_step("b", deps=["a"]),
            _make_step("c", deps=["a", "b"]),
        ])
        errors = PlanIntegrityChecker.validate_after_replan(plan)
        assert errors == []

    def test_validate_preserves_completed_tasks(self):
        """Completed tasks are preserved after replanning."""
        original = _make_plan([
            _make_step("a"),
            _make_step("b", deps=["a"]),
        ])
        original.steps[0].status = "completed"

        # New plan keeps step_a completed
        new_plan = _make_plan([
            _make_step("a"),
            _make_step("b", deps=["a"]),
            _make_step("c", deps=["b"]),  # Added new step
        ])
        new_plan.steps[0].status = "completed"

        errors = PlanIntegrityChecker.validate_after_replan(new_plan, original)
        assert errors == []

    def test_validate_rejects_removed_completed_task(self):
        """Removing a completed task is rejected."""
        original = _make_plan([
            _make_step("a"),
            _make_step("b", deps=["a"]),
        ])
        original.steps[0].status = "completed"

        # New plan removes step_a
        new_plan = _make_plan([
            _make_step("b"),  # step_a removed, deps cleared
        ])

        errors = PlanIntegrityChecker.validate_after_replan(new_plan, original)
        assert any("removed" in e.lower() for e in errors)

    def test_validate_rejects_changed_completed_status(self):
        """Changing a completed task's status is rejected."""
        original = _make_plan([
            _make_step("a"),
            _make_step("b", deps=["a"]),
        ])
        original.steps[0].status = "completed"

        # New plan changes step_a status
        new_plan = _make_plan([
            _make_step("a"),
            _make_step("b", deps=["a"]),
        ])
        new_plan.steps[0].status = "failed"  # Changed from completed

        errors = PlanIntegrityChecker.validate_after_replan(new_plan, original)
        assert any("status changed" in e.lower() for e in errors)

    def test_validate_rejects_duplicate_ids(self):
        """Duplicate task IDs are rejected."""
        plan = ExecutionPlan(
            objective="test",
            steps=[_make_step("a"), _make_step("a")],
        )
        errors = PlanIntegrityChecker.validate_after_replan(plan)
        assert any("duplicate" in e.lower() for e in errors)

    def test_validate_rejects_circular_dependencies(self):
        """Circular dependencies are rejected."""
        plan = ExecutionPlan(
            objective="test",
            steps=[
                _make_step("a", deps=["c"]),
                _make_step("b", deps=["a"]),
                _make_step("c", deps=["b"]),
            ],
        )
        errors = PlanIntegrityChecker.validate_after_replan(plan)
        assert any("circular" in e.lower() for e in errors)

    def test_validate_rejects_empty_plan(self):
        """Empty plan is rejected."""
        plan = ExecutionPlan(objective="test", steps=[])
        errors = PlanIntegrityChecker.validate_after_replan(plan)
        assert len(errors) == 1

    def test_validate_rejects_self_dependency(self):
        """Self-dependency is rejected."""
        plan = ExecutionPlan(
            objective="test",
            steps=[_make_step("a", deps=["a"])],
        )
        errors = PlanIntegrityChecker.validate_after_replan(plan)
        assert any("itself" in e.lower() for e in errors)


class TestReplanLoopDetection:
    """Tests for replan loop detection."""

    def test_no_loop_initial(self):
        """No loop at start."""
        plan = _make_plan([_make_step("a")])
        is_loop, reason = PlanIntegrityChecker.check_replan_loop(plan, max_replans=3)
        assert is_loop is False

    def test_loop_at_max_replans(self):
        """Loop detected at max replan count."""
        plan = _make_plan([_make_step("a")])
        plan.replan_count = 3
        is_loop, reason = PlanIntegrityChecker.check_replan_loop(plan, max_replans=3)
        assert is_loop is True
        assert "exceeded" in reason.lower()

    def test_loop_repeated_changes(self):
        """Loop detected from repeated changes to same tasks."""
        plan = _make_plan([_make_step("a")])
        # Simulate 3 repeated changes to same task
        changes = [
            {"modified": ["a"], "added": [], "removed": []},
            {"modified": ["a"], "added": [], "removed": []},
            {"modified": ["a"], "added": [], "removed": []},
        ]
        is_loop, reason = PlanIntegrityChecker.check_replan_loop(plan, max_replans=5, recent_changes=changes)
        assert is_loop is True
        assert "repeated" in reason.lower()

    def test_no_loop_different_changes(self):
        """No loop with different changes."""
        plan = _make_plan([_make_step("a")])
        changes = [
            {"modified": ["a"], "added": [], "removed": []},
            {"modified": ["b"], "added": [], "removed": []},
            {"modified": ["c"], "added": [], "removed": []},
        ]
        is_loop, reason = PlanIntegrityChecker.check_replan_loop(plan, max_replans=5, recent_changes=changes)
        assert is_loop is False


# ══════════════════════════════════════════════════════════════════════════════
# AdaptiveTaskExecutionObserver tests
# ══════════════════════════════════════════════════════════════════════════════

from orcha.nodes.task_executor import AdaptiveTaskExecutionObserver


class TestAdaptiveTaskExecutionObserver:
    """Tests for the enhanced adaptive observer."""

    @pytest.mark.asyncio
    async def test_observer_success_continues(self):
        """Successful task continues to next task."""
        observer = AdaptiveTaskExecutionObserver()
        step1 = _make_step("step_1")
        step1.status = "in_progress"
        step2 = _make_step("step_2", deps=["step_1"])
        plan = _make_plan([step1, step2])
        result = TaskExecutionResult(
            step_id="step_1",
            success=True,
            output="Task completed successfully",
        )
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["task_execution_result"] = result.model_dump()
        ctx = _make_ctx()
        response = await observer.run(packet, ctx)
        assert response.payload["task_action"] == "next_task"

    @pytest.mark.asyncio
    async def test_observer_final_task_verifies(self):
        """Final task triggers verification."""
        observer = AdaptiveTaskExecutionObserver()
        step1 = _make_step("step_1")
        step1.status = "in_progress"
        plan = _make_plan([step1])
        result = TaskExecutionResult(
            step_id="step_1",
            success=True,
            output="Task completed",
        )
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["task_execution_result"] = result.model_dump()
        ctx = _make_ctx()
        response = await observer.run(packet, ctx)
        assert response.payload["task_action"] == "verify"

    @pytest.mark.asyncio
    async def test_observer_loop_detection_stops(self):
        """Replan loop detection stops execution."""
        observer = AdaptiveTaskExecutionObserver(max_replans=2)
        step1 = _make_step("step_1")
        step1.status = "in_progress"
        plan = _make_plan([step1])
        plan.replan_count = 2  # At max

        result = TaskExecutionResult(
            step_id="step_1",
            success=False,
            output="",
            exhaustion_reason="repeated_tool_calls",
        )
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["task_execution_result"] = result.model_dump()
        ctx = _make_ctx()
        response = await observer.run(packet, ctx)
        assert response.payload["replan_loop_detected"] is True

    @pytest.mark.asyncio
    async def test_observer_with_llm_downstream_validation(self):
        """Observer uses LLM for downstream validation."""
        async def mock_completion(messages, system, tools):
            return {"content": "DECISION: VALID\nREASON: Downstream tasks are unaffected"}

        observer = AdaptiveTaskExecutionObserver(completion_fn=mock_completion)
        step1 = _make_step("step_1")
        step1.status = "in_progress"
        step2 = _make_step("step_2", deps=["step_1"])
        plan = _make_plan([step1, step2])
        result = TaskExecutionResult(
            step_id="step_1",
            success=True,
            output="Found 5 files",
        )
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["task_execution_result"] = result.model_dump()
        ctx = _make_ctx()
        response = await observer.run(packet, ctx)
        # Should continue to next task
        assert response.payload["task_action"] == "next_task"

    @pytest.mark.asyncio
    async def test_observer_emits_events(self):
        """Observer emits observation events."""
        observer = AdaptiveTaskExecutionObserver()
        step1 = _make_step("step_1")
        step1.status = "in_progress"
        plan = _make_plan([step1])
        result = TaskExecutionResult(
            step_id="step_1",
            success=True,
            output="Done",
        )
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["task_execution_result"] = result.model_dump()
        ctx = _make_ctx()
        await observer.run(packet, ctx)

        # Check that events were emitted
        calls = ctx.emit.emit.call_args_list
        event_types = [call[0][0] for call in calls]
        assert "task_complete" in event_types


# ══════════════════════════════════════════════════════════════════════════════
# Downstream validation tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDownstreamValidation:
    """Tests for downstream task validation."""

    def test_rule_based_valid_on_success(self):
        """Rule-based validation returns VALID on success."""
        observer = AdaptiveTaskExecutionObserver()
        step1 = _make_step("step_1")
        step2 = _make_step("step_2", deps=["step_1"])
        result = TaskExecutionResult(
            step_id="step_1",
            success=True,
            output="Done",
        )
        decision = observer._rule_based_downstream_validation(
            step1, result, [step2]
        )
        assert decision == "VALID"

    def test_rule_based_modify_on_not_found(self):
        """Rule-based validation returns MODIFY on 'not found' error."""
        observer = AdaptiveTaskExecutionObserver()
        step1 = _make_step("step_1")
        step2 = _make_step("step_2", deps=["step_1"])
        result = TaskExecutionResult(
            step_id="step_1",
            success=False,
            output="",
            error="File not found",
        )
        decision = observer._rule_based_downstream_validation(
            step1, result, [step2]
        )
        assert decision == "MODIFY_DOWNSTREAM"

    def test_parse_downstream_decision_valid(self):
        """Parse VALID decision."""
        observer = AdaptiveTaskExecutionObserver()
        content = "DECISION: VALID\nREASON: All good"
        assert observer._parse_downstream_decision(content) == "VALID"

    def test_parse_downstream_decision_modify(self):
        """Parse MODIFY_DOWNSTREAM decision."""
        observer = AdaptiveTaskExecutionObserver()
        content = "DECISION: MODIFY_DOWNSTREAM\nREASON: Need to change approach"
        assert observer._parse_downstream_decision(content) == "MODIFY_DOWNSTREAM"

    def test_parse_downstream_decision_skip(self):
        """Parse SKIP_DOWNSTREAM decision."""
        observer = AdaptiveTaskExecutionObserver()
        content = "DECISION: SKIP_DOWNSTREAM\nREASON: No longer needed"
        assert observer._parse_downstream_decision(content) == "SKIP_DOWNSTREAM"

    def test_parse_downstream_decision_blocked(self):
        """Parse BLOCKED decision."""
        observer = AdaptiveTaskExecutionObserver()
        content = "DECISION: BLOCKED\nREASON: Missing dependency"
        assert observer._parse_downstream_decision(content) == "BLOCKED"


# ══════════════════════════════════════════════════════════════════════════════
# Adaptive replanning integration tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveReplanningIntegration:
    """Integration tests for adaptive observation and replanning."""

    @pytest.mark.asyncio
    async def test_full_observation_replan_cycle(self):
        """Test complete observation -> handler processing cycle."""
        # Set up step as failed with a retryable exhaustion reason
        step1 = _make_step("step_1")
        step1.status = "failed"
        step2 = _make_step("step_2", deps=["step_1"])
        plan = _make_plan([step1, step2])

        observer = AdaptiveTaskExecutionObserver()
        handler = TaskRetryHandler()

        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["task_execution_result"] = TaskExecutionResult(
            step_id="step_1",
            success=False,
            output="CANNOT COMPLETE: Missing permissions",
            error="Missing permissions",
            exhaustion_reason="empty_model_outputs",  # Triggers RETRY in rule-based
        ).model_dump()
        ctx = _make_ctx()

        # Observer decides
        obs_response = await observer.run(packet, ctx)
        assert obs_response.payload["task_action"] == "retry"

        # Handler processes
        packet.payload["task_action"] = obs_response.payload["task_action"]
        handler_response = await handler.run(packet, ctx)
        assert handler_response.payload["task_action"] == "retry_same"

    @pytest.mark.asyncio
    async def test_completed_tasks_preserved_after_replan(self):
        """Completed tasks are preserved after replanning."""
        from orcha.nodes.task_executor import AdaptiveReplanner

        async def mock_replan_completion(messages, system, tools):
            return {"content": json.dumps({
                "modified_steps": [],
                "added_steps": [{
                    "id": "step_3_new",
                    "title": "New step",
                    "objective": "Do something new",
                    "dependencies": ["step_2"],
                }],
                "removed_step_ids": [],
                "kept_step_ids": ["step_1", "step_2"],
                "reason": "Added new step based on results",
            })}

        step1 = _make_step("step_1")
        step1.status = "completed"
        step1.result = TaskResult(step_id="step_1", output="Done", success=True)
        step2 = _make_step("step_2", deps=["step_1"])
        plan = _make_plan([step1, step2])
        plan.replan_count = 0

        replanner = AdaptiveReplanner(completion_fn=mock_replan_completion)
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["task_observation"] = TaskObservation(
            step_id="step_1",
            step_result=TaskResult(step_id="step_1", output="Done", success=True),
            decision=TaskObservationDecision.REPLAN,
        ).model_dump()
        ctx = _make_ctx()

        result = await replanner.run(packet, ctx)
        new_plan = ExecutionPlan(**result.payload["execution_plan"])

        # step_1 should still be completed
        assert new_plan.step_by_id("step_1").status == "completed"
        # New step should be added
        assert new_plan.step_by_id("step_3_new") is not None

    @pytest.mark.asyncio
    async def test_replan_loop_prevention(self):
        """Replan loop prevention stops infinite replanning."""
        from orcha.nodes.task_executor import AdaptiveReplanner

        async def mock_completion(messages, system, tools):
            return {"content": json.dumps({
                "modified_steps": [],
                "added_steps": [],
                "removed_step_ids": [],
                "kept_step_ids": ["step_1"],
                "reason": "Minor adjustment",
            })}

        step1 = _make_step("step_1")
        plan = _make_plan([step1])
        plan.replan_count = 3  # At max

        replanner = AdaptiveReplanner(completion_fn=mock_completion)
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["task_observation"] = TaskObservation(
            step_id="step_1",
            step_result=TaskResult(step_id="step_1", output="", success=False),
            decision=TaskObservationDecision.REPLAN,
        ).model_dump()
        ctx = _make_ctx()

        result = await replanner.run(packet, ctx)
        assert result.payload["replan_exhausted"] is True


# ══════════════════════════════════════════════════════════════════════════════
# Verification Layer Tests
# ══════════════════════════════════════════════════════════════════════════════

from orcha.core.packets import (
    VerificationSignal,
    VerificationResult,
    ObjectiveVerificationResult,
    RecoveryTask,
    FinalResponse,
    TaskArtifact,
)
from orcha.nodes.task_executor import (
    DeterministicVerifier,
    TaskStepVerifier,
    ObjectiveVerifier,
    RecoveryPlanner,
    FinalResponseGenerator,
    TASK_STEP_VERIFICATION_PROMPT,
    OBJECTIVE_VERIFICATION_PROMPT,
)


class TestVerificationModels:
    """Tests for verification data models."""

    def test_verification_signal_defaults(self):
        sig = VerificationSignal(signal_type="test")
        assert sig.signal_type == "test"
        assert sig.passed is False
        assert sig.expected is None
        assert sig.actual is None

    def test_verification_result_defaults(self):
        vr = VerificationResult(step_id="s1")
        assert vr.step_id == "s1"
        assert vr.verified is False
        assert vr.signals == []
        assert vr.issues == []

    def test_objective_verification_result_defaults(self):
        ovr = ObjectiveVerificationResult()
        assert ovr.status == "unverified"
        assert ovr.objective_satisfied is False
        assert ovr.verified is False

    def test_recovery_task_defaults(self):
        rt = RecoveryTask(title="Fix something")
        assert rt.title == "Fix something"
        assert rt.id  # auto-generated UUID
        assert rt.priority == 0

    def test_final_response_defaults(self):
        fr = FinalResponse()
        assert fr.status == "unverified"
        assert fr.completed_items == []
        assert fr.failed_items == []


class TestDeterministicVerifier:
    """Tests for DeterministicVerifier."""

    def test_extract_signals_success(self):
        """Successful task produces passing signals."""
        step = _make_step("s1", likely_tools=["read_file"])
        step.expected_outcome = "File contents displayed"
        result = TaskResult(
            step_id="s1",
            output="TASK COMPLETE: File contents displayed successfully",
            success=True,
            tool_calls=[{"tool": "read_file", "args_preview": "{}"}],
        )
        signals = DeterministicVerifier.extract_signals(step, result)
        assert len(signals) > 0
        # Success flag should pass
        success_signals = [s for s in signals if s.signal_type == "task_success_flag"]
        assert len(success_signals) == 1
        assert success_signals[0].passed is True

    def test_extract_signals_failure(self):
        """Failed task produces failing signals."""
        step = _make_step("s1")
        result = TaskResult(
            step_id="s1",
            output="",
            success=False,
            error="Permission denied",
        )
        signals = DeterministicVerifier.extract_signals(step, result)
        # Should have blocking failure
        assert DeterministicVerifier.has_blocking_failure(signals)

    def test_extract_signals_no_result(self):
        """Missing result produces no_result signal."""
        step = _make_step("s1")
        signals = DeterministicVerifier.extract_signals(step, None)
        assert len(signals) == 1
        assert signals[0].signal_type == "no_result"
        assert signals[0].passed is False

    def test_compute_pass_rate(self):
        signals = [
            VerificationSignal(signal_type="a", passed=True),
            VerificationSignal(signal_type="b", passed=True),
            VerificationSignal(signal_type="c", passed=False),
        ]
        assert DeterministicVerifier.compute_pass_rate(signals) == pytest.approx(2/3)

    def test_compute_pass_rate_empty(self):
        assert DeterministicVerifier.compute_pass_rate([]) == 0.0

    def test_has_blocking_failure_success_flag(self):
        signals = [VerificationSignal(signal_type="task_success_flag", passed=False)]
        assert DeterministicVerifier.has_blocking_failure(signals) is True

    def test_has_blocking_failure_error(self):
        signals = [VerificationSignal(signal_type="no_error", passed=False)]
        assert DeterministicVerifier.has_blocking_failure(signals) is True

    def test_has_blocking_failure_none(self):
        signals = [
            VerificationSignal(signal_type="output_nonempty", passed=True),
            VerificationSignal(signal_type="tools_used", passed=False),
        ]
        assert DeterministicVerifier.has_blocking_failure(signals) is False

    def test_extract_signals_file_artifacts(self):
        step = _make_step("s1")
        result = TaskResult(step_id="s1", output="Done", success=True)
        artifacts = [
            TaskArtifact(step_id="s1", kind="file", location="/tmp/test.py"),
        ]
        signals = DeterministicVerifier.extract_signals(step, result, artifacts)
        file_signals = [s for s in signals if s.signal_type == "file_artifact_recorded"]
        assert len(file_signals) == 1
        assert file_signals[0].passed is True

    def test_extract_signals_complete_marker(self):
        step = _make_step("s1")
        result = TaskResult(step_id="s1", output="TASK COMPLETE: Done", success=True)
        signals = DeterministicVerifier.extract_signals(step, result)
        marker_signals = [s for s in signals if s.signal_type == "complete_marker"]
        assert len(marker_signals) == 1
        assert marker_signals[0].passed is True

    def test_extract_signals_expected_outcome_match(self):
        step = _make_step("s1")
        step.expected_outcome = "File created with proper content"
        result = TaskResult(
            step_id="s1",
            output="Created file with proper content and structure",
            success=True,
        )
        signals = DeterministicVerifier.extract_signals(step, result)
        outcome_signals = [s for s in signals if s.signal_type == "outcome_content_match"]
        assert len(outcome_signals) == 1
        assert outcome_signals[0].passed is True

    def test_extract_signals_tools_used(self):
        step = _make_step("s1", likely_tools=["read_file", "write_file"])
        result = TaskResult(
            step_id="s1",
            output="Done",
            success=True,
            tool_calls=[
                {"tool": "read_file", "args_preview": "{}"},
                {"tool": "write_file", "args_preview": "{}"},
            ],
        )
        signals = DeterministicVerifier.extract_signals(step, result)
        tools_signals = [s for s in signals if s.signal_type == "tools_used"]
        assert len(tools_signals) == 1
        assert tools_signals[0].passed is True

    def test_extract_signals_file_exists_on_disk(self):
        """File artifact that exists on disk produces passing file_exists signal."""
        import tempfile
        import os
        step = _make_step("s1")
        result = TaskResult(step_id="s1", output="Done", success=True)
        # Create a temp file that actually exists
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        try:
            artifacts = [TaskArtifact(step_id="s1", kind="file", location=temp_path)]
            signals = DeterministicVerifier.extract_signals(step, result, artifacts)
            file_exists_signals = [s for s in signals if s.signal_type == "file_exists"]
            assert len(file_exists_signals) == 1
            assert file_exists_signals[0].passed is True
        finally:
            os.unlink(temp_path)

    def test_extract_signals_file_not_exists_on_disk(self):
        """File artifact that does not exist on disk produces failing file_exists signal."""
        step = _make_step("s1")
        result = TaskResult(step_id="s1", output="Done", success=True)
        artifacts = [TaskArtifact(step_id="s1", kind="file", location="/nonexistent/path/file.txt")]
        signals = DeterministicVerifier.extract_signals(step, result, artifacts)
        file_exists_signals = [s for s in signals if s.signal_type == "file_exists"]
        assert len(file_exists_signals) == 1
        assert file_exists_signals[0].passed is False

    def test_extract_signals_exit_code_success(self):
        """Command with exit code 0 produces passing command_exit_code signal."""
        step = _make_step("s1")
        result = TaskResult(
            step_id="s1",
            output="Command completed successfully. exit_code: 0",
            success=True,
            tool_calls=[{"tool": "run_command", "args_preview": "ls"}],
        )
        signals = DeterministicVerifier.extract_signals(step, result)
        exit_code_signals = [s for s in signals if s.signal_type == "command_exit_code"]
        assert len(exit_code_signals) == 1
        assert exit_code_signals[0].passed is True
        assert exit_code_signals[0].metadata["exit_code"] == 0

    def test_extract_signals_exit_code_failure(self):
        """Command with non-zero exit code produces failing command_exit_code signal."""
        step = _make_step("s1")
        result = TaskResult(
            step_id="s1",
            output="Error: exit_code: 1",
            success=False,
            tool_calls=[{"tool": "run_command", "args_preview": "failing_cmd"}],
        )
        signals = DeterministicVerifier.extract_signals(step, result)
        exit_code_signals = [s for s in signals if s.signal_type == "command_exit_code"]
        assert len(exit_code_signals) == 1
        assert exit_code_signals[0].passed is False
        assert exit_code_signals[0].metadata["exit_code"] == 1

    def test_extract_signals_build_success(self):
        """Successful build produces passing build_result signal."""
        step = _make_step("s1")
        result = TaskResult(
            step_id="s1",
            output="Build successful. 5 files compiled.",
            success=True,
            tool_calls=[{"tool": "run_command", "args_preview": "npm run build"}],
        )
        signals = DeterministicVerifier.extract_signals(step, result)
        build_signals = [s for s in signals if s.signal_type == "build_result"]
        assert len(build_signals) == 1
        assert build_signals[0].passed is True

    def test_extract_signals_build_failure(self):
        """Failed build produces failing build_result signal."""
        step = _make_step("s1")
        result = TaskResult(
            step_id="s1",
            output="Build failed. Compilation error in main.ts",
            success=False,
            tool_calls=[{"tool": "run_command", "args_preview": "cargo build"}],
        )
        signals = DeterministicVerifier.extract_signals(step, result)
        build_signals = [s for s in signals if s.signal_type == "build_result"]
        assert len(build_signals) == 1
        assert build_signals[0].passed is False

    def test_extract_signals_test_success(self):
        """All tests passing produces passing test_result signal."""
        step = _make_step("s1")
        result = TaskResult(
            step_id="s1",
            output="10 passed, 0 failed, 0 errors",
            success=True,
            tool_calls=[{"tool": "run_command", "args_preview": "pytest"}],
        )
        signals = DeterministicVerifier.extract_signals(step, result)
        test_signals = [s for s in signals if s.signal_type == "test_result"]
        assert len(test_signals) == 1
        assert test_signals[0].passed is True
        assert test_signals[0].metadata["passed"] == 10
        assert test_signals[0].metadata["failed"] == 0

    def test_extract_signals_test_failure(self):
        """Test failures produce failing test_result signal."""
        step = _make_step("s1")
        result = TaskResult(
            step_id="s1",
            output="8 passed, 2 failed, 0 errors",
            success=False,
            tool_calls=[{"tool": "run_command", "args_preview": "pytest"}],
        )
        signals = DeterministicVerifier.extract_signals(step, result)
        test_signals = [s for s in signals if s.signal_type == "test_result"]
        assert len(test_signals) == 1
        assert test_signals[0].passed is False
        assert test_signals[0].metadata["failed"] == 2

    def test_extract_signals_schema_valid_json(self):
        """Valid JSON output produces passing schema_valid signal."""
        step = _make_step("s1")
        result = TaskResult(
            step_id="s1",
            output='{"key": "value", "count": 42}',
            success=True,
        )
        signals = DeterministicVerifier.extract_signals(step, result)
        schema_signals = [s for s in signals if s.signal_type == "schema_valid"]
        assert len(schema_signals) == 1
        assert schema_signals[0].passed is True

    def test_extract_signals_schema_valid_json_array(self):
        """Valid JSON array output produces passing schema_valid signal."""
        step = _make_step("s1")
        result = TaskResult(
            step_id="s1",
            output='[1, 2, 3]',
            success=True,
        )
        signals = DeterministicVerifier.extract_signals(step, result)
        schema_signals = [s for s in signals if s.signal_type == "schema_valid"]
        assert len(schema_signals) == 1
        assert schema_signals[0].passed is True

    def test_extract_signals_schema_invalid_json(self):
        """Invalid JSON output produces failing schema_valid signal."""
        step = _make_step("s1")
        result = TaskResult(
            step_id="s1",
            output='{"key": "value", invalid json',
            success=True,
        )
        signals = DeterministicVerifier.extract_signals(step, result)
        schema_signals = [s for s in signals if s.signal_type == "schema_valid"]
        assert len(schema_signals) == 1
        assert schema_signals[0].passed is False

    def test_extract_signals_non_json_no_schema_signal(self):
        """Non-JSON output does not produce schema_valid signal."""
        step = _make_step("s1")
        result = TaskResult(
            step_id="s1",
            output="This is plain text output",
            success=True,
        )
        signals = DeterministicVerifier.extract_signals(step, result)
        schema_signals = [s for s in signals if s.signal_type == "schema_valid"]
        assert len(schema_signals) == 0

    def test_extract_signals_no_build_tool_no_build_signal(self):
        """Non-build tool does not produce build_result signal."""
        step = _make_step("s1")
        result = TaskResult(
            step_id="s1",
            output="File read successfully",
            success=True,
            tool_calls=[{"tool": "read_file", "args_preview": "{}"}],
        )
        signals = DeterministicVerifier.extract_signals(step, result)
        build_signals = [s for s in signals if s.signal_type == "build_result"]
        assert len(build_signals) == 0

    def test_extract_signals_no_test_tool_no_test_signal(self):
        """Non-test tool does not produce test_result signal."""
        step = _make_step("s1")
        result = TaskResult(
            step_id="s1",
            output="File written successfully",
            success=True,
            tool_calls=[{"tool": "write_file", "args_preview": "{}"}],
        )
        signals = DeterministicVerifier.extract_signals(step, result)
        test_signals = [s for s in signals if s.signal_type == "test_result"]
        assert len(test_signals) == 0


class TestTaskStepVerifier:
    """Tests for TaskStepVerifier."""

    @pytest.mark.asyncio
    async def test_verify_success(self):
        """Successful task passes verification."""
        verifier = TaskStepVerifier()
        step = _make_step("s1")
        step.result = TaskResult(
            step_id="s1",
            output="TASK COMPLETE: Done",
            success=True,
            tool_calls=[{"tool": "read_file", "args_preview": "{}"}],
        )
        plan = _make_plan([step])
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["step_id"] = "s1"
        ctx = _make_ctx()

        response = await verifier.run(packet, ctx)
        vr = VerificationResult(**response.payload["verification_result"])
        assert vr.verified is True
        assert vr.deterministic_pass_rate > 0.5

    @pytest.mark.asyncio
    async def test_verify_failure(self):
        """Failed task does not pass verification."""
        verifier = TaskStepVerifier()
        step = _make_step("s1")
        step.result = TaskResult(
            step_id="s1",
            output="",
            success=False,
            error="Permission denied",
        )
        plan = _make_plan([step])
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["step_id"] = "s1"
        ctx = _make_ctx()

        response = await verifier.run(packet, ctx)
        vr = VerificationResult(**response.payload["verification_result"])
        assert vr.verified is False
        assert len(vr.issues) > 0

    @pytest.mark.asyncio
    async def test_verify_no_plan(self):
        verifier = TaskStepVerifier()
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        ctx = _make_ctx()
        response = await verifier.run(packet, ctx)
        vr = VerificationResult(**response.payload["verification_result"])
        assert vr.verified is False

    @pytest.mark.asyncio
    async def test_verify_with_llm_fallback(self):
        """LLM fallback is used when deterministic signals are ambiguous."""
        async def mock_completion(messages, system, tools):
            return {"content": "DECISION: VERIFIED\nCONFIDENCE: 0.85\nREASON: Output looks correct"}

        verifier = TaskStepVerifier(completion_fn=mock_completion)
        step = _make_step("s1")
        step.verification_criteria = "Code should compile"
        step.result = TaskResult(
            step_id="s1",
            output="Done",
            success=True,
            tool_calls=[],
        )
        plan = _make_plan([step])
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["step_id"] = "s1"
        ctx = _make_ctx()

        response = await verifier.run(packet, ctx)
        vr = VerificationResult(**response.payload["verification_result"])
        assert vr.used_llm_fallback is True
        assert vr.llm_confidence == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_verify_emits_events(self):
        verifier = TaskStepVerifier()
        step = _make_step("s1")
        step.result = TaskResult(step_id="s1", output="Done", success=True)
        plan = _make_plan([step])
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["step_id"] = "s1"
        ctx = _make_ctx()

        await verifier.run(packet, ctx)
        calls = ctx.emit.emit.call_args_list
        event_types = [call[0][0] for call in calls]
        assert "verification_start" in event_types
        assert "verification_complete" in event_types


class TestObjectiveVerifier:
    """Tests for ObjectiveVerifier."""

    @pytest.mark.asyncio
    async def test_verify_all_complete(self):
        """All tasks complete -> objective satisfied."""
        verifier = ObjectiveVerifier()
        step1 = _make_step("s1")
        step1.status = "completed"
        step1.result = TaskResult(step_id="s1", output="Done", success=True)
        plan = _make_plan([step1])
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["verification_results"] = [
            VerificationResult(step_id="s1", verified=True).model_dump()
        ]
        ctx = _make_ctx()

        response = await verifier.run(packet, ctx)
        ovr = ObjectiveVerificationResult(**response.payload["objective_verification"])
        assert ovr.objective_satisfied is True
        assert ovr.status == "verified"

    @pytest.mark.asyncio
    async def test_verify_with_failures(self):
        """Failed tasks -> objective not satisfied."""
        verifier = ObjectiveVerifier()
        step1 = _make_step("s1")
        step1.status = "completed"
        step1.result = TaskResult(step_id="s1", output="Done", success=True)
        step2 = _make_step("s2")
        step2.status = "failed"
        step2.result = TaskResult(step_id="s2", output="", success=False, error="Failed")
        plan = _make_plan([step1, step2])
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["verification_results"] = [
            VerificationResult(step_id="s1", verified=True).model_dump(),
            VerificationResult(step_id="s2", verified=False).model_dump(),
        ]
        ctx = _make_ctx()

        response = await verifier.run(packet, ctx)
        ovr = ObjectiveVerificationResult(**response.payload["objective_verification"])
        assert ovr.objective_satisfied is False
        assert ovr.status == "failed"

    @pytest.mark.asyncio
    async def test_verify_with_llm(self):
        """LLM evaluation is used for qualitative assessment."""
        async def mock_completion(messages, system, tools):
            return {
                "content": (
                    "DECISION: OBJECTIVE_SATISFIED\n"
                    "CONFIDENCE: 0.9\n"
                    "REASON: All tasks completed successfully\n"
                    "COMPLETED_ITEMS: task s1 done, task s2 done\n"
                    "INCOMPLETE_ITEMS: none\n"
                    "LIMITATIONS: none"
                )
            }

        verifier = ObjectiveVerifier(completion_fn=mock_completion)
        step1 = _make_step("s1")
        step1.status = "completed"
        step1.result = TaskResult(step_id="s1", output="Done", success=True)
        plan = _make_plan([step1])
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["verification_results"] = [
            VerificationResult(step_id="s1", verified=True).model_dump()
        ]
        ctx = _make_ctx()

        response = await verifier.run(packet, ctx)
        ovr = ObjectiveVerificationResult(**response.payload["objective_verification"])
        assert ovr.objective_satisfied is True
        assert len(ovr.incomplete_items) == 0

    @pytest.mark.asyncio
    async def test_verify_no_plan(self):
        verifier = ObjectiveVerifier()
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        ctx = _make_ctx()
        response = await verifier.run(packet, ctx)
        ovr = ObjectiveVerificationResult(**response.payload["objective_verification"])
        assert ovr.status == "failed"

    @pytest.mark.asyncio
    async def test_verify_artifacts_missing(self):
        """Artifacts that don't exist on disk are reported in artifacts_missing."""
        import tempfile
        import os
        verifier = ObjectiveVerifier()
        step1 = _make_step("s1")
        step1.status = "completed"
        step1.result = TaskResult(step_id="s1", output="Done", success=True)

        # Create a temp file that exists
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        try:
            # Add observation with artifacts
            from orcha.core.packets import TaskObservation, TaskArtifact
            obs = TaskObservation(
                step_id="s1",
                step_result=step1.result,
                artifacts=[
                    TaskArtifact(step_id="s1", kind="file", location=temp_path),
                    TaskArtifact(step_id="s1", kind="file", location="/nonexistent/file.txt"),
                ],
            )
            plan = _make_plan([step1])
            plan.observations = [obs]

            packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
            packet.payload["execution_plan"] = plan.model_dump()
            packet.payload["verification_results"] = [
                VerificationResult(step_id="s1", verified=True).model_dump()
            ]
            ctx = _make_ctx()

            response = await verifier.run(packet, ctx)
            ovr = ObjectiveVerificationResult(**response.payload["objective_verification"])
            assert temp_path in ovr.artifacts_verified
            assert "/nonexistent/file.txt" in ovr.artifacts_missing
        finally:
            os.unlink(temp_path)


class TestRecoveryPlanner:
    """Tests for RecoveryPlanner."""

    @pytest.mark.asyncio
    async def test_no_recovery_needed(self):
        """No recovery when objective is satisfied."""
        planner = RecoveryPlanner()
        step1 = _make_step("s1")
        step1.status = "completed"
        plan = _make_plan([step1])
        obj_ver = ObjectiveVerificationResult(
            objective_satisfied=True,
            incomplete_items=[],
            unresolved_errors=[],
        )
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["objective_verification"] = obj_ver.model_dump()
        ctx = _make_ctx()

        response = await planner.run(packet, ctx)
        assert response.payload["recovery_needed"] is False
        assert response.payload["recovery_tasks"] == []

    @pytest.mark.asyncio
    async def test_recovery_for_incomplete(self):
        """Recovery tasks created for incomplete items."""
        planner = RecoveryPlanner()
        step1 = _make_step("s1")
        step1.status = "completed"
        plan = _make_plan([step1])
        obj_ver = ObjectiveVerificationResult(
            objective_satisfied=False,
            incomplete_items=["s1: Task not verified"],
            unresolved_errors=[],
        )
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["objective_verification"] = obj_ver.model_dump()
        ctx = _make_ctx()

        response = await planner.run(packet, ctx)
        assert response.payload["recovery_needed"] is True
        tasks = response.payload["recovery_tasks"]
        assert len(tasks) == 1
        assert "s1" in tasks[0]["title"]

    @pytest.mark.asyncio
    async def test_recovery_for_errors(self):
        """Recovery tasks created for unresolved errors."""
        planner = RecoveryPlanner()
        step1 = _make_step("s1")
        step1.status = "completed"
        plan = _make_plan([step1])
        obj_ver = ObjectiveVerificationResult(
            objective_satisfied=False,
            incomplete_items=[],
            unresolved_errors=["[s1] Permission denied"],
        )
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["objective_verification"] = obj_ver.model_dump()
        ctx = _make_ctx()

        response = await planner.run(packet, ctx)
        assert response.payload["recovery_needed"] is True
        tasks = response.payload["recovery_tasks"]
        assert len(tasks) == 1
        assert "Error recovery" in tasks[0]["title"]

    @pytest.mark.asyncio
    async def test_recovery_emits_events(self):
        planner = RecoveryPlanner()
        step1 = _make_step("s1")
        step1.status = "completed"
        plan = _make_plan([step1])
        obj_ver = ObjectiveVerificationResult(
            objective_satisfied=False,
            incomplete_items=["Missing file"],
            unresolved_errors=[],
        )
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["objective_verification"] = obj_ver.model_dump()
        ctx = _make_ctx()

        await planner.run(packet, ctx)
        calls = ctx.emit.emit.call_args_list
        event_types = [call[0][0] for call in calls]
        assert "recovery_task_created" in event_types


class TestFinalResponseGenerator:
    """Tests for FinalResponseGenerator."""

    @pytest.mark.asyncio
    async def test_success_response(self):
        """Successful completion generates success response."""
        generator = FinalResponseGenerator()
        step1 = _make_step("s1")
        step1.status = "completed"
        step1.result = TaskResult(step_id="s1", output="Done", success=True)
        plan = _make_plan([step1])
        obj_ver = ObjectiveVerificationResult(
            status="verified",
            objective_satisfied=True,
            completed_items=["s1: Test step"],
            incomplete_items=[],
            unresolved_errors=[],
        )
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["objective_verification"] = obj_ver.model_dump()
        packet.payload["recovery_tasks"] = []
        ctx = _make_ctx()

        response = await generator.run(packet, ctx)
        fr = FinalResponse(**response.payload["final_response"])
        assert fr.status == "success"
        assert len(fr.completed_items) > 0
        assert "Objective achieved" in fr.summary

    @pytest.mark.asyncio
    async def test_failure_response(self):
        """Failed completion generates failure response."""
        generator = FinalResponseGenerator()
        step1 = _make_step("s1")
        step1.status = "failed"
        step1.result = TaskResult(step_id="s1", output="", success=False, error="Failed")
        plan = _make_plan([step1])
        obj_ver = ObjectiveVerificationResult(
            status="failed",
            objective_satisfied=False,
            completed_items=[],
            incomplete_items=["s1: Step failed"],
            unresolved_errors=["[s1] Failed"],
        )
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["objective_verification"] = obj_ver.model_dump()
        packet.payload["recovery_tasks"] = []
        ctx = _make_ctx()

        response = await generator.run(packet, ctx)
        fr = FinalResponse(**response.payload["final_response"])
        assert fr.status == "failure"
        assert len(fr.failed_items) > 0

    @pytest.mark.asyncio
    async def test_partial_response(self):
        """Mixed results generate partial response."""
        generator = FinalResponseGenerator()
        step1 = _make_step("s1")
        step1.status = "completed"
        step1.result = TaskResult(step_id="s1", output="Done", success=True)
        step2 = _make_step("s2")
        step2.status = "failed"
        step2.result = TaskResult(step_id="s2", output="", success=False, error="Failed")
        plan = _make_plan([step1, step2])
        obj_ver = ObjectiveVerificationResult(
            status="partially_verified",
            objective_satisfied=False,
            completed_items=["s1: Step 1"],
            incomplete_items=["s2: Step 2 not done"],
            unresolved_errors=[],
        )
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["objective_verification"] = obj_ver.model_dump()
        packet.payload["recovery_tasks"] = []
        ctx = _make_ctx()

        response = await generator.run(packet, ctx)
        fr = FinalResponse(**response.payload["final_response"])
        assert fr.status == "partial"

    @pytest.mark.asyncio
    async def test_response_tracks_file_changes(self):
        """File changes are tracked in response."""
        generator = FinalResponseGenerator()
        step1 = _make_step("s1")
        step1.status = "completed"
        step1.result = TaskResult(
            step_id="s1",
            output="Done",
            success=True,
            tool_calls=[{"tool": "write_file", "args_preview": '{"path": "/tmp/test.py"}'}],
        )
        plan = _make_plan([step1])
        obj_ver = ObjectiveVerificationResult(
            status="verified",
            objective_satisfied=True,
        )
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["objective_verification"] = obj_ver.model_dump()
        packet.payload["recovery_tasks"] = []
        ctx = _make_ctx()

        response = await generator.run(packet, ctx)
        fr = FinalResponse(**response.payload["final_response"])
        assert any("/tmp/test.py" in c for c in fr.changed_items)

    @pytest.mark.asyncio
    async def test_response_with_recovery_tasks(self):
        """Recovery tasks appear as limitations."""
        generator = FinalResponseGenerator()
        step1 = _make_step("s1")
        step1.status = "completed"
        plan = _make_plan([step1])
        obj_ver = ObjectiveVerificationResult(
            status="failed",
            objective_satisfied=False,
        )
        recovery = RecoveryTask(title="Fix issue", objective="Fix")
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["objective_verification"] = obj_ver.model_dump()
        packet.payload["recovery_tasks"] = [recovery.model_dump()]
        ctx = _make_ctx()

        response = await generator.run(packet, ctx)
        fr = FinalResponse(**response.payload["final_response"])
        assert any("recovery task" in lim.lower() for lim in fr.limitations)

    @pytest.mark.asyncio
    async def test_response_emits_events(self):
        generator = FinalResponseGenerator()
        step1 = _make_step("s1")
        step1.status = "completed"
        plan = _make_plan([step1])
        obj_ver = ObjectiveVerificationResult(objective_satisfied=True, status="verified")
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        packet.payload["objective_verification"] = obj_ver.model_dump()
        packet.payload["recovery_tasks"] = []
        ctx = _make_ctx()

        await generator.run(packet, ctx)
        calls = ctx.emit.emit.call_args_list
        event_types = [call[0][0] for call in calls]
        assert "final_response" in event_types


class TestEnhancedTaskFinalVerifier:
    """Tests for the enhanced TaskFinalVerifier."""

    @pytest.mark.asyncio
    async def test_verifier_all_complete(self):
        """All tasks complete -> verified."""
        verifier = TaskFinalVerifier()
        step1 = _make_step("s1")
        step1.status = "completed"
        step1.result = TaskResult(
            step_id="s1", output="TASK COMPLETE: Done", success=True,
            tool_calls=[{"tool": "read_file", "args_preview": "{}"}],
        )
        plan = _make_plan([step1])
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        ctx = _make_ctx()

        response = await verifier.run(packet, ctx)
        assert response.payload["objective_met"] is True
        assert "VERIFIED" in response.payload["verification_result"]

    @pytest.mark.asyncio
    async def test_verifier_with_failures(self):
        """Failed tasks -> not verified."""
        verifier = TaskFinalVerifier()
        step1 = _make_step("s1")
        step1.status = "failed"
        step1.result = TaskResult(
            step_id="s1", output="", success=False, error="Failed",
        )
        plan = _make_plan([step1])
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        ctx = _make_ctx()

        response = await verifier.run(packet, ctx)
        assert response.payload["objective_met"] is False
        assert "NOT VERIFIED" in response.payload["verification_result"]

    @pytest.mark.asyncio
    async def test_verifier_no_plan(self):
        verifier = TaskFinalVerifier()
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        ctx = _make_ctx()
        response = await verifier.run(packet, ctx)
        assert response.payload["objective_met"] is False

    @pytest.mark.asyncio
    async def test_verifier_produces_verification_results(self):
        """Verifier produces per-step verification results."""
        verifier = TaskFinalVerifier()
        step1 = _make_step("s1")
        step1.status = "completed"
        step1.result = TaskResult(
            step_id="s1", output="TASK COMPLETE: Done", success=True,
        )
        plan = _make_plan([step1])
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        ctx = _make_ctx()

        response = await verifier.run(packet, ctx)
        vr_list = response.payload.get("verification_results", [])
        assert len(vr_list) == 1
        vr = VerificationResult(**vr_list[0])
        assert vr.verified is True

    @pytest.mark.asyncio
    async def test_verifier_produces_final_response(self):
        """Verifier produces a final user-facing response."""
        verifier = TaskFinalVerifier()
        step1 = _make_step("s1")
        step1.status = "completed"
        step1.result = TaskResult(
            step_id="s1", output="TASK COMPLETE: Done", success=True,
        )
        plan = _make_plan([step1])
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        ctx = _make_ctx()

        response = await verifier.run(packet, ctx)
        fr_raw = response.payload.get("final_response")
        assert fr_raw is not None
        fr = FinalResponse(**fr_raw)
        assert fr.status == "success"
        assert len(fr.completed_items) > 0

    @pytest.mark.asyncio
    async def test_verifier_creates_recovery_tasks(self):
        """Recovery tasks are created when verification fails."""
        async def mock_completion(messages, system, tools):
            return {
                "content": (
                    "DECISION: OBJECTIVE_NOT_SATISFIED\n"
                    "CONFIDENCE: 0.9\n"
                    "REASON: Task failed\n"
                    "COMPLETED_ITEMS: none\n"
                    "INCOMPLETE_ITEMS: s1 step not done\n"
                    "LIMITATIONS: permission issue"
                )
            }

        verifier = TaskFinalVerifier(completion_fn=mock_completion)
        step1 = _make_step("s1")
        step1.status = "failed"
        step1.result = TaskResult(
            step_id="s1", output="", success=False, error="Permission denied",
        )
        plan = _make_plan([step1])
        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        packet.payload["execution_plan"] = plan.model_dump()
        ctx = _make_ctx()

        response = await verifier.run(packet, ctx)
        assert response.payload["objective_met"] is False
        recovery = response.payload.get("recovery_tasks", [])
        assert len(recovery) > 0


class TestVerificationPrompts:
    """Tests for verification prompt constants."""

    def test_step_verification_prompt_format(self):
        prompt = TASK_STEP_VERIFICATION_PROMPT.format(
            step_id="s1",
            step_title="Test Step",
            step_objective="Do something",
            expected_outcome="Something done",
            verification_criteria="File exists",
            task_output="Output here",
            tool_calls="[]",
            deterministic_signals="  [PASS] test: ok",
        )
        assert "s1" in prompt
        assert "Test Step" in prompt
        assert "Do something" in prompt

    def test_objective_verification_prompt_format(self):
        prompt = OBJECTIVE_VERIFICATION_PROMPT.format(
            objective="Build feature X",
            task_graph_summary="  [DONE] s1: Step 1",
            task_results="  s1: success=True",
            artifacts="  - /tmp/file.py",
            unresolved_errors="  None",
        )
        assert "Build feature X" in prompt
        assert "[DONE]" in prompt


# ══════════════════════════════════════════════════════════════════════════════
# Event Stream Tests
# ══════════════════════════════════════════════════════════════════════════════

from orcha.core.packets import ExecutionEvent, ExecutionEventKind, RunStateSnapshot
from orcha.graph.context import (
    RunStateTracker,
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
)
from orcha.nodes.task_executor import EventStreamAdapter


class TestExecutionEventModels:
    """Tests for ExecutionEvent and RunStateSnapshot models."""

    def test_execution_event_defaults(self):
        event = ExecutionEvent(
            run_id="r1",
            kind=ExecutionEventKind.RUN_CREATED,
        )
        assert event.run_id == "r1"
        assert event.seq == 0
        assert event.kind == ExecutionEventKind.RUN_CREATED
        assert event.ts > 0

    def test_execution_event_with_task_context(self):
        event = ExecutionEvent(
            run_id="r1",
            kind=ExecutionEventKind.TASK_STARTED,
            task_id="t1",
            task_title="Build feature",
            task_status="in_progress",
            plan_progress=0.5,
        )
        assert event.task_id == "t1"
        assert event.task_title == "Build feature"
        assert event.plan_progress == 0.5

    def test_execution_event_kind_values(self):
        """All event kinds have correct string values."""
        assert ExecutionEventKind.RUN_CREATED.value == "run_created"
        assert ExecutionEventKind.TASK_COMPLETED.value == "task_completed"
        assert ExecutionEventKind.VERIFICATION_COMPLETED.value == "verification_completed"

    def test_run_state_snapshot_defaults(self):
        snapshot = RunStateSnapshot(run_id="r1")
        assert snapshot.run_id == "r1"
        assert snapshot.status == "created"
        assert snapshot.events == []
        assert snapshot.completed_task_ids == []

    def test_run_state_snapshot_with_events(self):
        event = ExecutionEvent(
            run_id="r1",
            kind=ExecutionEventKind.RUN_CREATED,
            seq=1,
        )
        snapshot = RunStateSnapshot(
            run_id="r1",
            status="running",
            events=[event],
            completed_task_ids=["t1", "t2"],
        )
        assert len(snapshot.events) == 1
        assert snapshot.completed_task_ids == ["t1", "t2"]


class TestRunStateTracker:
    """Tests for RunStateTracker."""

    def test_tracker_initial_state(self):
        tracker = RunStateTracker(run_id="r1", objective="Test")
        assert tracker.run_id == "r1"
        assert tracker.status == "created"
        assert tracker.seq == 0
        assert len(tracker.events) == 0

    def test_tracker_emit_event_increments_seq(self):
        tracker = RunStateTracker(run_id="r1")
        e1 = tracker.emit_event("run_created", summary="Started")
        e2 = tracker.emit_event("task_started", summary="Task 1")
        assert e1.seq == 1
        assert e2.seq == 2
        assert tracker.seq == 2

    def test_tracker_events_ordered_by_seq(self):
        tracker = RunStateTracker(run_id="r1")
        tracker.emit_event("run_created", summary="A")
        tracker.emit_event("task_started", summary="B")
        tracker.emit_event("task_completed", summary="C")
        events = tracker.events
        assert [e.seq for e in events] == [1, 2, 3]

    def test_tracker_snapshot(self):
        tracker = RunStateTracker(run_id="r1", objective="Build X")
        tracker.emit_event("run_created", summary="Started")
        tracker.set_status("running")
        snapshot = tracker.snapshot()
        assert snapshot.run_id == "r1"
        assert snapshot.status == "running"
        assert snapshot.objective == "Build X"
        assert len(snapshot.events) == 1

    @pytest.mark.asyncio
    async def test_tracker_subscribe_to_emitter(self):
        """Tracker auto-translates internal events to structured events."""
        from orcha.graph.context import EventEmitter

        tracker = RunStateTracker(run_id="r1", objective="Test")
        emitter = EventEmitter(run_id="r1")
        tracker.subscribe_to(emitter)

        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        await emitter.emit("task_start", "executor", packet,
                           task_id="t1", task_title="Task 1", plan_progress=0.3)

        events = tracker.events
        assert len(events) == 1
        assert events[0].kind == EVT_EXEC_TASK_STARTED
        assert events[0].task_id == "t1"
        assert events[0].task_title == "Task 1"

    @pytest.mark.asyncio
    async def test_tracker_task_complete_event(self):
        from orcha.graph.context import EventEmitter

        tracker = RunStateTracker(run_id="r1")
        emitter = EventEmitter(run_id="r1")
        tracker.subscribe_to(emitter)

        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        await emitter.emit("task_complete", "executor", packet,
                           task_id="t1", task_title="Task 1", success=True,
                           plan_progress=0.5, duration_s=1.5)

        events = tracker.events
        assert len(events) == 1
        assert events[0].kind == EVT_EXEC_TASK_COMPLETED
        assert events[0].task_status == "completed"

    @pytest.mark.asyncio
    async def test_tracker_task_failure_event(self):
        from orcha.graph.context import EventEmitter

        tracker = RunStateTracker(run_id="r1")
        emitter = EventEmitter(run_id="r1")
        tracker.subscribe_to(emitter)

        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        await emitter.emit("task_complete", "executor", packet,
                           task_id="t1", task_title="Task 1", success=False,
                           exhaustion_reason="timeout")

        events = tracker.events
        assert len(events) == 1
        assert events[0].kind == EVT_EXEC_TASK_FAILED
        assert events[0].error == "timeout"

    @pytest.mark.asyncio
    async def test_tracker_run_complete_event(self):
        from orcha.graph.context import EventEmitter

        tracker = RunStateTracker(run_id="r1")
        emitter = EventEmitter(run_id="r1")
        tracker.subscribe_to(emitter)

        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        await emitter.emit("run_complete", "", packet)

        assert tracker.status == "completed"
        assert len(tracker.events) == 1
        assert tracker.events[0].kind == EVT_EXEC_RUN_COMPLETED

    @pytest.mark.asyncio
    async def test_tracker_cancel_event(self):
        from orcha.graph.context import EventEmitter

        tracker = RunStateTracker(run_id="r1")
        emitter = EventEmitter(run_id="r1")
        tracker.subscribe_to(emitter)

        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        await emitter.emit("cancel", "", packet, reason="user cancelled")

        assert tracker.status == "cancelled"
        assert tracker.events[0].error == "user cancelled"

    @pytest.mark.asyncio
    async def test_tracker_error_event(self):
        from orcha.graph.context import EventEmitter

        tracker = RunStateTracker(run_id="r1")
        emitter = EventEmitter(run_id="r1")
        tracker.subscribe_to(emitter)

        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        await emitter.emit("error", "", packet, error="model timeout")

        assert tracker.status == "failed"
        assert tracker.events[0].error == "model timeout"

    @pytest.mark.asyncio
    async def test_tracker_unsubscribe(self):
        from orcha.graph.context import EventEmitter

        tracker = RunStateTracker(run_id="r1")
        emitter = EventEmitter(run_id="r1")
        tracker.subscribe_to(emitter)
        tracker.unsubscribe()

        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        await emitter.emit("task_start", "", packet, task_id="t1")

        # Should not have received the event after unsubscribe
        assert len(tracker.events) == 0

    def test_tracker_deterministic_ordering(self):
        """Events are always ordered by sequence number."""
        tracker = RunStateTracker(run_id="r1")
        for i in range(10):
            tracker.emit_event("run_created", summary=f"Event {i}")
        events = tracker.events
        seqs = [e.seq for e in events]
        assert seqs == list(range(1, 11))


class TestEventStreamAdapter:
    """Tests for EventStreamAdapter."""

    def test_adapter_initial_state(self):
        adapter = EventStreamAdapter(run_id="r1", objective="Test")
        assert adapter.get_status() == "created"
        assert len(adapter.get_events()) == 0

    def test_adapter_emit_run_created(self):
        adapter = EventStreamAdapter(run_id="r1")
        event = adapter.emit_run_created(objective="Build feature X")
        assert event.kind == EVT_EXEC_RUN_CREATED
        assert adapter.get_status() == "running"
        assert "Build feature X" in event.summary

    def test_adapter_emit_task_lifecycle(self):
        adapter = EventStreamAdapter(run_id="r1")
        adapter.emit_run_created()

        e1 = adapter.emit_task_started(task_id="t1", task_title="Step 1")
        assert e1.kind == EVT_EXEC_TASK_STARTED
        assert e1.task_id == "t1"

        e2 = adapter.emit_task_completed(task_id="t1", task_title="Step 1")
        assert e2.kind == EVT_EXEC_TASK_COMPLETED
        assert e2.task_status == "completed"

    def test_adapter_emit_task_failed(self):
        adapter = EventStreamAdapter(run_id="r1")
        event = adapter.emit_task_failed(
            task_id="t1", task_title="Step 1", error="timeout"
        )
        assert event.kind == EVT_EXEC_TASK_FAILED
        assert event.error == "timeout"

    def test_adapter_emit_tool_lifecycle(self):
        adapter = EventStreamAdapter(run_id="r1")
        e1 = adapter.emit_tool_started(task_id="t1", tool_name="read_file")
        assert e1.kind == EVT_EXEC_TOOL_STARTED
        assert e1.metadata["tool"] == "read_file"

        e2 = adapter.emit_tool_completed(task_id="t1", tool_name="read_file", success=True)
        assert e2.kind == EVT_EXEC_TOOL_COMPLETED
        assert e2.metadata["success"] is True

    def test_adapter_emit_retry(self):
        adapter = EventStreamAdapter(run_id="r1")
        event = adapter.emit_retry_started(
            task_id="t1", task_title="Step 1",
            retry_reason="timeout", retry_count=1,
        )
        assert event.kind == EVT_EXEC_RETRY_STARTED
        assert event.metadata["retry_count"] == 1

    def test_adapter_emit_replanning(self):
        adapter = EventStreamAdapter(run_id="r1")
        event = adapter.emit_replanning_started(
            reason="structural change needed", replan_count=2,
        )
        assert event.kind == EVT_EXEC_REPLANNING_STARTED
        assert event.metadata["replan_count"] == 2

    def test_adapter_emit_verification(self):
        adapter = EventStreamAdapter(run_id="r1")
        e1 = adapter.emit_verification_started(task_id="t1", task_title="Step 1")
        assert e1.kind == EVT_EXEC_VERIFICATION_STARTED

        e2 = adapter.emit_verification_completed(
            task_id="t1", verified=True, pass_rate=0.9
        )
        assert e2.kind == EVT_EXEC_VERIFICATION_COMPLETED
        assert e2.verification_status == "verified"

    def test_adapter_emit_run_completed(self):
        adapter = EventStreamAdapter(run_id="r1")
        adapter.emit_run_created()
        event = adapter.emit_run_completed(summary_text="All done")
        assert event.kind == EVT_EXEC_RUN_COMPLETED
        assert adapter.get_status() == "completed"

    def test_adapter_emit_run_failed(self):
        adapter = EventStreamAdapter(run_id="r1")
        adapter.emit_run_created()
        event = adapter.emit_run_failed(error="model timeout")
        assert event.kind == EVT_EXEC_RUN_FAILED
        assert adapter.get_status() == "failed"

    def test_adapter_emit_run_cancelled(self):
        adapter = EventStreamAdapter(run_id="r1")
        adapter.emit_run_created()
        event = adapter.emit_run_cancelled(reason="user cancelled")
        assert event.kind == EVT_EXEC_RUN_CANCELLED
        assert adapter.get_status() == "cancelled"

    def test_adapter_snapshot(self):
        adapter = EventStreamAdapter(run_id="r1", objective="Build X")
        adapter.emit_run_created()
        adapter.emit_task_started(task_id="t1", task_title="Step 1")
        adapter.emit_task_completed(task_id="t1", task_title="Step 1")

        snapshot = adapter.get_snapshot()
        assert snapshot.run_id == "r1"
        assert snapshot.objective == "Build X"
        assert snapshot.status == "running"
        assert len(snapshot.events) == 3
        assert snapshot.completed_task_ids == ["t1"]

    def test_adapter_events_ordered(self):
        adapter = EventStreamAdapter(run_id="r1")
        adapter.emit_run_created()
        adapter.emit_task_started(task_id="t1")
        adapter.emit_task_completed(task_id="t1")
        adapter.emit_run_completed()

        events = adapter.get_events()
        assert [e.seq for e in events] == [1, 2, 3, 4]

    def test_adapter_intent_analyzed(self):
        adapter = EventStreamAdapter(run_id="r1")
        event = adapter.emit_intent_analyzed(
            intent="user wants to build a feature", complexity="complex"
        )
        assert event.kind == EVT_EXEC_INTENT_ANALYZED
        assert "complex" in event.metadata["complexity"]

    def test_adapter_complexity_determined(self):
        adapter = EventStreamAdapter(run_id="r1")
        event = adapter.emit_complexity_determined(
            complexity="complex", needs_planning=True
        )
        assert event.kind == EVT_EXEC_COMPLEXITY_DETERMINED
        assert event.metadata["needs_planning"] is True

    def test_adapter_planning_started(self):
        adapter = EventStreamAdapter(run_id="r1")
        event = adapter.emit_planning_started()
        assert event.kind == EVT_EXEC_PLANNING_STARTED

    def test_adapter_plan_created(self):
        adapter = EventStreamAdapter(run_id="r1")
        event = adapter.emit_plan_created(step_count=5)
        assert event.kind == EVT_EXEC_PLAN_CREATED
        assert event.plan_step_count == 5

    def test_adapter_plan_updated(self):
        adapter = EventStreamAdapter(run_id="r1")
        event = adapter.emit_plan_updated(
            step_count=5, completed_count=2, plan_progress=0.4
        )
        assert event.kind == EVT_EXEC_PLAN_UPDATED
        assert event.plan_progress == 0.4

    def test_adapter_model_started(self):
        adapter = EventStreamAdapter(run_id="r1")
        event = adapter.emit_model_started(task_id="t1", model_name="gpt-4")
        assert event.kind == EVT_EXEC_MODEL_STARTED
        assert event.metadata["model"] == "gpt-4"

    def test_adapter_task_observed(self):
        adapter = EventStreamAdapter(run_id="r1")
        event = adapter.emit_task_observed(
            task_id="t1", task_title="Step 1", decision="continue"
        )
        assert event.kind == EVT_EXEC_TASK_OBSERVED
        assert event.metadata["decision"] == "continue"

    def test_adapter_task_ready(self):
        adapter = EventStreamAdapter(run_id="r1")
        event = adapter.emit_task_ready(task_id="t1", task_title="Step 1")
        assert event.kind == EVT_EXEC_TASK_READY
        assert event.task_status == "pending"

    @pytest.mark.asyncio
    async def test_adapter_attach_detach(self):
        from orcha.graph.context import EventEmitter

        adapter = EventStreamAdapter(run_id="r1")
        emitter = EventEmitter(run_id="r1")
        adapter.attach(emitter)

        packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
        await emitter.emit("task_start", "executor", packet,
                           task_id="t1", task_title="Task 1")
        assert len(adapter.get_events()) == 1

        adapter.detach()

        await emitter.emit("task_start", "executor", packet,
                           task_id="t2", task_title="Task 2")
        # Should not receive events after detach
        assert len(adapter.get_events()) == 1

    def test_adapter_update_plan(self):
        adapter = EventStreamAdapter(run_id="r1")
        plan = _make_plan([_make_step("s1")])
        adapter.update_plan(plan)
        snapshot = adapter.get_snapshot()
        assert snapshot.plan is not None

    def test_adapter_set_final_response(self):
        from orcha.core.packets import FinalResponse

        adapter = EventStreamAdapter(run_id="r1")
        response = FinalResponse(status="success", summary="All done")
        adapter.set_final_response(response)
        snapshot = adapter.get_snapshot()
        assert snapshot.final_response.status == "success"

    def test_adapter_set_verification_result(self):
        from orcha.core.packets import ObjectiveVerificationResult

        adapter = EventStreamAdapter(run_id="r1")
        result = ObjectiveVerificationResult(
            status="verified", objective_satisfied=True, verified=True
        )
        adapter.set_verification_result(result)
        snapshot = adapter.get_snapshot()
        assert snapshot.verification_result.verified is True

    def test_adapter_metadata_preserved(self):
        adapter = EventStreamAdapter(
            run_id="r1", metadata={"source": "test", "version": "1.0"}
        )
        snapshot = adapter.get_snapshot()
        assert snapshot.metadata["source"] == "test"

    def test_adapter_tool_completed_failure(self):
        adapter = EventStreamAdapter(run_id="r1")
        event = adapter.emit_tool_completed(
            task_id="t1", tool_name="write_file", success=False
        )
        assert event.error is not None
        assert "failed" in event.error.lower()


# ── Integration: EventStreamAdapter wired into TaskExecutionLoop ─────────────────


@pytest.mark.asyncio
async def test_execution_loop_creates_event_adapter():
    """TaskExecutionLoop.run() creates and persists an EventStreamAdapter."""
    from orcha.nodes.task_executor import EventStreamAdapter

    async def mock_completion(messages, system, tools):
        return {"content": "TASK COMPLETE: Done"}

    plan = _make_plan([_make_step("step_1")])
    loop = TaskExecutionLoop(completion_fn=mock_completion)
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = _make_ctx()
    result = await loop.run(packet, ctx)

    adapter = result.payload.get("_event_adapter")
    assert adapter is not None
    assert isinstance(adapter, EventStreamAdapter)
    snapshot = result.payload.get("_event_stream_snapshot")
    assert snapshot is not None


@pytest.mark.asyncio
async def test_execution_loop_adapter_persists_across_calls():
    """The EventStreamAdapter is persisted in packet and survives across invocations."""
    from orcha.nodes.task_executor import EventStreamAdapter

    async def mock_completion(messages, system, tools):
        return {"content": "TASK COMPLETE: Done"}

    plan = _make_plan([_make_step("step_1"), _make_step("step_2")])
    loop = TaskExecutionLoop(completion_fn=mock_completion)
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = _make_ctx()

    # First call
    result1 = await loop.run(packet, ctx)
    adapter1 = result1.payload.get("_event_adapter")
    assert adapter1 is not None

    # Update plan with first step done
    plan2 = ExecutionPlan(**result1.payload.get("execution_plan"))
    packet2 = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet2.payload["execution_plan"] = plan2.model_dump()
    packet2.payload["_event_adapter"] = adapter1

    # Second call with same adapter
    result2 = await loop.run(packet2, ctx)
    adapter2 = result2.payload.get("_event_adapter")

    assert adapter2 is adapter1
    assert len(adapter2.get_events()) >= 2


@pytest.mark.asyncio
async def test_execution_loop_emits_structured_task_events():
    """TaskExecutionLoop emits structured ExecutionEvent for task lifecycle."""
    async def mock_completion(messages, system, tools):
        return {"content": "TASK COMPLETE: Done"}

    plan = _make_plan([_make_step("step_1", title="Test Task")])
    loop = TaskExecutionLoop(completion_fn=mock_completion)
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = _make_ctx()
    result = await loop.run(packet, ctx)

    adapter = result.payload.get("_event_adapter")
    events = adapter.get_events()
    event_kinds = [e.kind for e in events]

    from orcha.graph.context import (
        EVT_EXEC_TASK_STARTED,
        EVT_EXEC_TASK_COMPLETED,
    )
    assert EVT_EXEC_TASK_STARTED in event_kinds
    assert EVT_EXEC_TASK_COMPLETED in event_kinds


@pytest.mark.asyncio
async def test_execution_loop_snapshot_contains_events():
    """The RunStateSnapshot in the result contains all emitted ExecutionEvents."""
    async def mock_completion(messages, system, tools):
        return {"content": "TASK COMPLETE: Done"}

    plan = _make_plan([_make_step("step_1")])
    loop = TaskExecutionLoop(completion_fn=mock_completion)
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = _make_ctx()
    result = await loop.run(packet, ctx)

    snapshot = result.payload.get("_event_stream_snapshot")
    assert snapshot is not None
    assert len(snapshot.events) >= 2
    assert snapshot.status in ("created", "running", "completed")


@pytest.mark.asyncio
async def test_execution_loop_with_tool_emits_tool_events():
    """Tool calls inside the loop emit structured tool events."""
    call_count = 0

    async def mock_completion(messages, system, tools):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "content": "Let me read the file",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "read_file", "arguments": '{"path":"foo.txt"}'},
                    }
                ],
            }
        return {"content": "TASK COMPLETE: Done reading"}

    plan = _make_plan([_make_step("step_1", likely_tools=["read_file"])])

    tool_result = MagicMock()
    tool_result.to_message.return_value = "file contents..."
    tool_result.success = True

    def mock_invoke(name, **kwargs):
        return tool_result

    mock_executor = MagicMock()
    mock_executor.schemas.return_value = [
        {"type": "function", "function": {"name": "read_file", "description": "Read file"}}
    ]
    mock_executor.invoke = mock_invoke

    loop = TaskExecutionLoop(completion_fn=mock_completion, executor=mock_executor)
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = _make_ctx()
    result = await loop.run(packet, ctx)

    adapter = result.payload.get("_event_adapter")
    events = adapter.get_events()
    event_kinds = [str(e.kind.value) for e in events]

    from orcha.graph.context import EVT_EXEC_TOOL_STARTED, EVT_EXEC_TOOL_COMPLETED
    assert EVT_EXEC_TOOL_STARTED in event_kinds
    assert EVT_EXEC_TOOL_COMPLETED in event_kinds
