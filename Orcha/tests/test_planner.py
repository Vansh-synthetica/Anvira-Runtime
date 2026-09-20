"""Tests for the Orcha Task Planner."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from orcha.core.packets import (
    ExecutionPlan,
    OrchaPacket,
    PacketKind,
    PlannerRequest,
    TaskObservation,
    TaskResult,
    TaskStep,
)
from orcha.nodes.planner import (
    ComplexityGateConfig,
    ComplexityGateNode,
    TaskPlannerNode,
    TaskContextBuilderNode,
    TaskSummaryMemory,
    ObserverNode,
    ReplannerNode,
    VerifierNode,
    OrchaTaskPlanner,
    TaskPlanBuilder,
    TaskType,
    _heuristic_complexity_score,
    _parse_tasks_from_model,
    _split_clauses,
    _classify_task_type,
    _tools_for_task_type,
    _action_for_task_type,
    validate_plan,
    build_planner_prompt,
    PlanValidationError,
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


# ── PlannerRequest ───────────────────────────────────────────────────────────

def test_planner_request_defaults():
    req = PlannerRequest(
        original_request="do something",
        normalized_objective="do something useful",
    )
    assert req.constraints == []
    assert req.context == ""
    assert req.tools == []
    assert req.workspace_state == ""


def test_planner_request_full():
    req = PlannerRequest(
        original_request="inspect repo",
        normalized_objective="understand the project",
        constraints=["Python 3.11", "no external deps"],
        context="Project has src/ and tests/ directories",
        tools=["list_directory", "read_file"],
        workspace_state="Git repo with 50 files",
    )
    assert len(req.constraints) == 2
    assert len(req.tools) == 2


# ── TaskStep ─────────────────────────────────────────────────────────────────

def test_task_step_required_fields():
    step = TaskStep(
        id="step_1",
        title="Inspect repository",
        objective="Understand project structure",
        execution_instructions="List all files and read key configs",
    )
    assert step.id == "step_1"
    assert step.title == "Inspect repository"
    assert step.dependencies == []
    assert step.likely_tools == []
    assert step.action == "tool_use"


def test_task_step_with_dependencies():
    step = TaskStep(
        id="step_2",
        title="Identify framework",
        objective="Determine framework and patterns",
        execution_instructions="Analyze package.json and config files",
        dependencies=["step_1"],
        required_context=["step_1"],
        likely_tools=["read_file", "search_code"],
    )
    assert step.dependencies == ["step_1"]
    assert step.required_context == ["step_1"]
    assert len(step.likely_tools) == 2


# ── Validation ───────────────────────────────────────────────────────────────

def test_validate_plan_valid():
    plan = ExecutionPlan(
        objective="test",
        steps=[
            _make_step("a", deps=[]),
            _make_step("b", deps=["a"]),
            _make_step("c", deps=["a", "b"]),
        ],
    )
    errors = validate_plan(plan)
    assert errors == []


def test_validate_plan_empty():
    plan = ExecutionPlan(objective="test", steps=[])
    errors = validate_plan(plan)
    assert len(errors) == 1
    assert "no tasks" in errors[0].lower()


def test_validate_plan_duplicate_ids():
    plan = ExecutionPlan(
        objective="test",
        steps=[
            _make_step("a"),
            _make_step("a"),
        ],
    )
    errors = validate_plan(plan)
    assert any("duplicate" in e.lower() for e in errors)


def test_validate_plan_self_dependency():
    plan = ExecutionPlan(
        objective="test",
        steps=[_make_step("a", deps=["a"])],
    )
    errors = validate_plan(plan)
    assert any("itself" in e.lower() for e in errors)


def test_validate_plan_impossible_dependency():
    plan = ExecutionPlan(
        objective="test",
        steps=[_make_step("a", deps=["nonexistent"])],
    )
    errors = validate_plan(plan)
    assert any("non-existent" in e.lower() for e in errors)


def test_validate_plan_circular_dependency():
    plan = ExecutionPlan(
        objective="test",
        steps=[
            _make_step("a", deps=["c"]),
            _make_step("b", deps=["a"]),
            _make_step("c", deps=["b"]),
        ],
    )
    errors = validate_plan(plan)
    assert any("circular" in e.lower() for e in errors)


def test_validate_plan_empty_objective():
    plan = ExecutionPlan(
        objective="test",
        steps=[_make_step("a", objective="")],
    )
    errors = validate_plan(plan)
    assert any("empty" in e.lower() and "objective" in e.lower() for e in errors)


def test_validate_plan_short_objective():
    plan = ExecutionPlan(
        objective="test",
        steps=[_make_step("a", objective="abc")],  # < 5 chars
    )
    errors = validate_plan(plan)
    assert any("short" in e.lower() and "objective" in e.lower() for e in errors)


def test_validate_plan_empty_title():
    plan = ExecutionPlan(
        objective="test",
        steps=[_make_step("a", title="")],
    )
    errors = validate_plan(plan)
    assert any("empty" in e.lower() and "title" in e.lower() for e in errors)


# ── Cycle detection ──────────────────────────────────────────────────────────

def test_detect_cycle_linear():
    steps = [_make_step("a"), _make_step("b", deps=["a"])]
    plan = ExecutionPlan(objective="test", steps=steps)
    errors = validate_plan(plan)
    assert errors == []


def test_detect_cycle_diamond():
    steps = [
        _make_step("a"),
        _make_step("b", deps=["a"]),
        _make_step("c", deps=["a"]),
        _make_step("d", deps=["b", "c"]),
    ]
    plan = ExecutionPlan(objective="test", steps=steps)
    errors = validate_plan(plan)
    assert errors == []


def test_detect_cycle_self_loop():
    steps = [_make_step("a", deps=["a"])]
    plan = ExecutionPlan(objective="test", steps=steps)
    errors = validate_plan(plan)
    assert any("itself" in e.lower() for e in errors)


# ── Task planner node ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_task_planner_no_model():
    planner = TaskPlannerNode(completion_fn=None)
    packet = OrchaPacket(kind=PacketKind.QUERY, query="complex task")
    ctx = MagicMock(spec=RunContext)
    result = await planner.run(packet, ctx)
    plan = result.payload["execution_plan"]
    assert plan["status"] == "in_progress"
    assert len(plan["steps"]) == 1
    assert plan["steps"][0]["id"] == "execute_directly"


@pytest.mark.asyncio
async def test_task_planner_with_valid_plan():
    valid_plan = json.dumps([
        {
            "id": "inspect_repo",
            "title": "Inspect repository structure",
            "objective": "Understand the project layout and existing code",
            "execution_instructions": "List all files, read key configuration files",
            "dependencies": [],
            "expected_outcome": "Clear understanding of project structure",
            "required_context": [],
            "likely_tools": ["list_directory", "read_file"],
            "verification_criteria": "Can describe the project structure",
            "action": "search",
        },
        {
            "id": "identify_framework",
            "title": "Identify framework and patterns",
            "objective": "Determine the framework, patterns, and conventions used",
            "execution_instructions": "Analyze package.json, read config files",
            "dependencies": ["inspect_repo"],
            "expected_outcome": "Documented framework choice and patterns",
            "required_context": ["inspect_repo"],
            "likely_tools": ["read_file"],
            "verification_criteria": "Can list the framework and conventions",
            "action": "tool_use",
        },
    ])

    async def mock_completion(messages, system, tools):
        return {"content": valid_plan}

    planner = TaskPlannerNode(completion_fn=mock_completion)
    packet = OrchaPacket(kind=PacketKind.QUERY, query="inspect and identify framework")
    ctx = MagicMock(spec=RunContext)
    result = await planner.run(packet, ctx)
    plan = result.payload["execution_plan"]
    assert plan["status"] == "in_progress"
    assert len(plan["steps"]) == 2
    assert plan["steps"][0]["id"] == "inspect_repo"
    assert plan["steps"][1]["dependencies"] == ["inspect_repo"]


@pytest.mark.asyncio
async def test_task_planner_invalid_plan_fallback():
    # Plan with circular dependency
    invalid_plan = json.dumps([
        {"id": "a", "title": "Task A", "objective": "Do A", "dependencies": ["b"]},
        {"id": "b", "title": "Task B", "objective": "Do B", "dependencies": ["a"]},
    ])

    async def mock_completion(messages, system, tools):
        return {"content": invalid_plan}

    planner = TaskPlannerNode(completion_fn=mock_completion, retries=1)
    packet = OrchaPacket(kind=PacketKind.QUERY, query="circular task")
    ctx = MagicMock(spec=RunContext)
    result = await planner.run(packet, ctx)
    plan = result.payload["execution_plan"]
    # Should fallback to single-step plan
    assert len(plan["steps"]) == 1
    assert plan["steps"][0]["id"] == "execute_directly"


@pytest.mark.asyncio
async def test_task_planner_malformed_output():
    async def mock_completion(messages, system, tools):
        return {"content": "I don't understand what you want"}

    planner = TaskPlannerNode(completion_fn=mock_completion, retries=1)
    packet = OrchaPacket(kind=PacketKind.QUERY, query="complex task")
    ctx = MagicMock(spec=RunContext)
    result = await planner.run(packet, ctx)
    plan = result.payload["execution_plan"]
    # Should fallback
    assert len(plan["steps"]) == 1


@pytest.mark.asyncio
async def test_task_planner_with_planner_request():
    request = PlannerRequest(
        original_request="inspect repo, add feature, run tests",
        normalized_objective="Understand project, implement feature, verify",
        constraints=["Python 3.11"],
        tools=["list_directory", "read_file", "execute_command"],
    )
    valid_plan = json.dumps([
        {
            "id": "inspect",
            "title": "Inspect repository",
            "objective": "Understand the project structure and conventions",
            "execution_instructions": "List files, read configs, identify patterns",
            "dependencies": [],
            "expected_outcome": "Clear understanding of the project",
            "required_context": [],
            "likely_tools": ["list_directory", "read_file"],
            "verification_criteria": "Can describe project structure",
            "action": "search",
        },
    ])

    async def mock_completion(messages, system, tools):
        return {"content": valid_plan}

    planner = TaskPlannerNode(completion_fn=mock_completion)
    packet = OrchaPacket(kind=PacketKind.QUERY, query="inspect repo")
    packet.payload["planner_request"] = request.model_dump()
    ctx = MagicMock(spec=RunContext)
    result = await planner.run(packet, ctx)
    plan = result.payload["execution_plan"]
    assert plan["steps"][0]["id"] == "inspect"
    assert plan["steps"][0]["likely_tools"] == ["list_directory", "read_file"]


# ── Parse tasks from model ──────────────────────────────────────────────────

def test_parse_tasks_valid_json():
    content = json.dumps([
        {"id": "step_1", "title": "Step 1", "objective": "Do step 1", "dependencies": []},
    ])
    steps = _parse_tasks_from_model(content)
    assert len(steps) == 1
    assert steps[0].id == "step_1"


def test_parse_tasks_markdown_fenced():
    content = "```json\n" + json.dumps([
        {"id": "step_1", "title": "Step 1", "objective": "Do step 1"},
    ]) + "\n```"
    steps = _parse_tasks_from_model(content)
    assert len(steps) == 1


def test_parse_tasks_embedded_json():
    content = "Here's the plan:\n" + json.dumps([
        {"id": "step_1", "title": "Step 1", "objective": "Do step 1"},
    ]) + "\nThat's it."
    steps = _parse_tasks_from_model(content)
    assert len(steps) == 1


def test_parse_tasks_missing_required_fields():
    content = json.dumps([
        {"id": "step_1", "title": "", "objective": ""},
        {"id": "step_2", "title": "Valid", "objective": "Valid objective"},
    ])
    steps = _parse_tasks_from_model(content)
    # Only step_2 should be parsed (step_1 has empty title/objective)
    assert len(steps) == 1
    assert steps[0].id == "step_2"


def test_parse_tasks_regex_fallback():
    content = '"id": "step_1", "title": "First step", "objective": "Do something", "action": "search"'
    steps = _parse_tasks_from_model(content)
    assert len(steps) == 1
    assert steps[0].id == "step_1"
    assert steps[0].action == "search"


def test_parse_tasks_garbage():
    steps = _parse_tasks_from_model("not json at all")
    assert len(steps) == 0


# ── Build planner prompt ────────────────────────────────────────────────────

def test_build_planner_prompt():
    request = PlannerRequest(
        original_request="inspect repo",
        normalized_objective="understand project",
        constraints=["no external deps"],
        tools=["read_file"],
    )
    prompt = build_planner_prompt(request)
    assert "inspect repo" in prompt
    assert "understand project" in prompt
    assert "no external deps" in prompt
    assert "read_file" in prompt


def test_build_planner_prompt_with_context():
    request = PlannerRequest(
        original_request="fix bug",
        normalized_objective="fix the failing test",
        context="Tests fail with ImportError",
    )
    prompt = build_planner_prompt(request)
    assert "Tests fail with ImportError" in prompt


# ── TaskContextBuilderNode ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_task_context_builder():
    builder = TaskContextBuilderNode()
    plan = ExecutionPlan(
        objective="test objective",
        steps=[
            _make_step("step_1", title="Find files", objective="Find all Python files"),
            _make_step("step_2", title="Read files", objective="Read the found files", deps=["step_1"]),
        ],
    )
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = MagicMock(spec=RunContext)
    result = await builder.run(packet, ctx)
    assert result.payload["task_step_id"] == "step_1"
    assert "Find files" in result.payload["task_step_context"]
    assert "Find all Python files" in result.payload["task_step_context"]


@pytest.mark.asyncio
async def test_task_context_builder_includes_original_request():
    """Regression: the step context must always carry the plan's own
    objective (the original user request), not just the per-step
    objective/execution_instructions the planner wrote. Confirmed live —
    a real run asking for a detailed Snake game (canvas, arrow keys,
    collision, score) decomposed into a single thin step titled "Create
    Snake game HTML file" with none of that detail, and the executing
    model correctly refused with "CANNOT COMPLETE: No specific
    instructions or code provided" because it genuinely never saw the
    original request — only the planner's compressed summary of it."""
    builder = TaskContextBuilderNode()
    detailed_request = (
        "Create a fully playable Snake game as a single self-contained HTML "
        "file. Use HTML5 canvas and vanilla JavaScript. The snake must move "
        "with arrow keys, grow when eating food, and the game must end on "
        "collision with a visible game-over message and score."
    )
    plan = ExecutionPlan(
        objective=detailed_request,
        steps=[_make_step(
            "create_snake_game",
            title="Create Snake game HTML file",
            objective="Create the game",  # thin, as a real planner run produced
        )],
    )
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = MagicMock(spec=RunContext)
    result = await builder.run(packet, ctx)
    assert "arrow keys" in result.payload["task_step_context"]
    assert "collision" in result.payload["task_step_context"]
    assert "game-over" in result.payload["task_step_context"]


@pytest.mark.asyncio
async def test_task_context_builder_no_plan():
    builder = TaskContextBuilderNode()
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    ctx = MagicMock(spec=RunContext)
    result = await builder.run(packet, ctx)
    assert "task_step_id" not in result.payload


# ── TaskSummaryMemory ────────────────────────────────────────────────────────

def test_task_summary_memory():
    memory = TaskSummaryMemory()
    step = _make_step("step_1", title="Find files", objective="Find Python files")
    result = TaskResult(step_id="step_1", output="Found 5 files", success=True)
    memory.add(step, result)
    summary = memory.build_summary()
    assert "step_1" in summary
    assert "Find files" in summary
    assert "Find Python files" in summary
    assert "✓" in summary


def test_task_summary_memory_failed():
    memory = TaskSummaryMemory()
    step = _make_step("step_1", title="Failed task", objective="Do something")
    result = TaskResult(step_id="step_1", output="", success=False, error="timeout")
    memory.add(step, result)
    summary = memory.build_summary()
    assert "✗" in summary
    assert "timeout" in summary


# ── ExecutionPlan ────────────────────────────────────────────────────────────

def test_execution_plan_progress():
    plan = ExecutionPlan(
        objective="test",
        steps=[
            _make_step("step_1", deps=[]),
            _make_step("step_2", deps=[]),
        ],
    )
    plan.steps[0].status = "completed"
    assert plan.progress == 0.5
    assert not plan.is_complete
    assert len(plan.completed_steps) == 1


def test_execution_plan_next_step():
    plan = ExecutionPlan(
        objective="test",
        steps=[
            _make_step("step_1", deps=[]),
            _make_step("step_2", deps=["step_1"]),
        ],
    )
    plan.steps[0].status = "completed"
    step = plan.next_pending_step()
    assert step is not None
    assert step.id == "step_2"


def test_execution_plan_next_step_deps_not_met():
    plan = ExecutionPlan(
        objective="test",
        steps=[
            _make_step("step_1", deps=[]),
            _make_step("step_2", deps=["step_1"]),
            _make_step("step_3", deps=["step_2"]),
        ],
    )
    # step_1 is pending, step_2 needs step_1, step_3 needs step_2
    step = plan.next_pending_step()
    assert step is not None
    assert step.id == "step_1"  # This one can run


def test_execution_plan_step_by_id():
    plan = ExecutionPlan(
        objective="test",
        steps=[_make_step("a"), _make_step("b")],
    )
    assert plan.step_by_id("a") is not None
    assert plan.step_by_id("a").id == "a"
    assert plan.step_by_id("nonexistent") is None


def test_execution_plan_dependency_graph():
    plan = ExecutionPlan(
        objective="test",
        steps=[
            _make_step("a", deps=[]),
            _make_step("b", deps=["a"]),
            _make_step("c", deps=["a", "b"]),
        ],
    )
    graph = plan.get_dependency_graph()
    assert graph["a"] == []
    assert graph["b"] == ["a"]
    assert graph["c"] == ["a", "b"]


# ── ObserverNode ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_observer_success():
    observer = ObserverNode()
    step = _make_step("step_1")
    step.status = "in_progress"
    step.result = TaskResult(step_id="step_1", output="Found", success=True)
    plan = ExecutionPlan(
        objective="test",
        steps=[step],
    )
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = MagicMock(spec=RunContext)
    result = await observer.run(packet, ctx)
    assert result.payload["objective_met"] is True


@pytest.mark.asyncio
async def test_observer_failure():
    observer = ObserverNode()
    step = _make_step("step_1")
    step.status = "in_progress"
    step.result = TaskResult(step_id="step_1", output="", success=False, error="failed")
    plan = ExecutionPlan(
        objective="test",
        steps=[step, _make_step("step_2")],
    )
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = MagicMock(spec=RunContext)
    result = await observer.run(packet, ctx)
    assert result.payload["objective_met"] is False


# ── ReplannerNode ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_replanner_needed():
    replanner = ReplannerNode()
    step = _make_step("step_1")
    step.status = "failed"
    plan = ExecutionPlan(
        objective="test",
        steps=[step, _make_step("step_2")],
    )
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = MagicMock(spec=RunContext)
    result = await replanner.run(packet, ctx)
    assert result.payload["replan_needed"] is True


@pytest.mark.asyncio
async def test_replanner_max_replans():
    replanner = ReplannerNode()
    step = _make_step("step_1")
    step.status = "failed"
    plan = ExecutionPlan(
        objective="test",
        replan_count=3,
        steps=[step],
    )
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = MagicMock(spec=RunContext)
    result = await replanner.run(packet, ctx)
    assert result.payload["replan_needed"] is False


# ── VerifierNode ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_verifier_success():
    verifier = VerifierNode()
    plan = ExecutionPlan(
        objective="test",
        steps=[
            _make_step("step_1"),
            _make_step("step_2"),
        ],
    )
    plan.steps[0].status = "completed"
    plan.steps[1].status = "completed"
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = MagicMock(spec=RunContext)
    result = await verifier.run(packet, ctx)
    assert result.payload["objective_met"] is True


@pytest.mark.asyncio
async def test_verifier_incomplete():
    verifier = VerifierNode()
    plan = ExecutionPlan(
        objective="test",
        steps=[
            _make_step("step_1"),
            _make_step("step_2"),
        ],
    )
    plan.steps[0].status = "completed"
    plan.steps[1].status = "pending"
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    packet.payload["execution_plan"] = plan.model_dump()
    ctx = MagicMock(spec=RunContext)
    result = await verifier.run(packet, ctx)
    assert result.payload["objective_met"] is False


# ── Packet integration ──────────────────────────────────────────────────────

def test_packet_execution_plan_helpers():
    packet = OrchaPacket(kind=PacketKind.QUERY, query="test")
    plan = ExecutionPlan(
        objective="test",
        steps=[_make_step("step_1")],
    )
    packet.set_execution_plan(plan)
    retrieved = packet.get_execution_plan()
    assert retrieved is not None
    assert retrieved.objective == "test"
    assert len(retrieved.steps) == 1


# ── Realistic complex request tests ──────────────────────────────────────────

def test_validate_plan_realistic_feature_implementation():
    """Validate a realistic plan for implementing a feature."""
    plan = ExecutionPlan(
        objective="Inspect this project, understand the existing architecture, "
                  "implement the requested feature, run the tests, fix failures, "
                  "and verify the final result.",
        steps=[
            _make_step(
                "inspect_repo",
                title="Inspect repository structure",
                objective="Understand the project layout, tech stack, and architecture",
                execution_instructions="List all files, read key configuration files, "
                    "identify the main entry points and patterns used",
                deps=[],
                expected_outcome="Clear understanding of project structure and conventions",
                likely_tools=["list_directory", "read_file"],
                verification_criteria="Can describe the project structure, tech stack, "
                    "and main entry points accurately",
                action="search",
            ),
            _make_step(
                "identify_framework",
                title="Identify framework and patterns",
                objective="Determine the framework, testing approach, and coding conventions",
                execution_instructions="Analyze package.json or requirements.txt, "
                    "read config files, examine existing code patterns",
                deps=["inspect_repo"],
                expected_outcome="Documented framework choice, patterns, and conventions",
                required_context=["inspect_repo"],
                likely_tools=["read_file", "search_code"],
                verification_criteria="Can list the framework, testing approach, "
                    "and coding conventions",
                action="tool_use",
            ),
            _make_step(
                "understand_feature",
                title="Understand the requested feature",
                objective="Clarify what the feature should do and where it fits",
                execution_instructions="Read the feature request, identify affected areas, "
                    "determine integration points",
                deps=["inspect_repo", "identify_framework"],
                expected_outcome="Clear feature specification with affected files identified",
                required_context=["inspect_repo", "identify_framework"],
                likely_tools=["read_file", "search_code"],
                verification_criteria="Can describe the feature requirements and affected files",
                action="tool_use",
            ),
            _make_step(
                "implement_feature",
                title="Implement the feature",
                objective="Write the code for the new feature",
                execution_instructions="Create or modify files according to the feature spec, "
                    "following the project's coding conventions",
                deps=["understand_feature"],
                expected_outcome="Feature code written and integrated",
                required_context=["understand_feature", "identify_framework"],
                likely_tools=["create_file", "edit_file", "read_file"],
                verification_criteria="Code compiles and follows project conventions",
                action="create",
            ),
            _make_step(
                "run_tests",
                title="Run the test suite",
                objective="Execute all tests to verify nothing is broken",
                execution_instructions="Run the project's test command, capture output",
                deps=["implement_feature"],
                expected_outcome="Test results showing pass/fail status",
                required_context=["implement_feature"],
                likely_tools=["execute_command"],
                verification_criteria="All tests pass or failures are documented",
                action="execute",
            ),
            _make_step(
                "fix_failures",
                title="Fix test failures",
                objective="Resolve any test failures from the previous step",
                execution_instructions="Analyze failing tests, identify root causes, "
                    "fix the code or tests as needed",
                deps=["run_tests"],
                expected_outcome="All tests passing",
                required_context=["run_tests", "implement_feature"],
                likely_tools=["read_file", "edit_file", "execute_command"],
                verification_criteria="All tests pass after fixes",
                action="modify",
            ),
            _make_step(
                "verify_result",
                title="Final verification",
                objective="Verify the complete feature works as expected",
                execution_instructions="Run full test suite, check code quality, "
                    "verify integration",
                deps=["fix_failures"],
                expected_outcome="Feature fully implemented and verified",
                required_context=["fix_failures", "implement_feature"],
                likely_tools=["execute_command", "read_file"],
                verification_criteria="All tests pass, code follows conventions, "
                    "feature requirements are met",
                action="execute",
            ),
        ],
    )
    errors = validate_plan(plan)
    assert errors == []
    # Verify dependency chain
    assert plan.get_dependency_graph()["inspect_repo"] == []
    assert plan.get_dependency_graph()["identify_framework"] == ["inspect_repo"]
    assert plan.get_dependency_graph()["understand_feature"] == ["inspect_repo", "identify_framework"]
    assert plan.get_dependency_graph()["implement_feature"] == ["understand_feature"]
    assert plan.get_dependency_graph()["run_tests"] == ["implement_feature"]
    assert plan.get_dependency_graph()["fix_failures"] == ["run_tests"]
    assert plan.get_dependency_graph()["verify_result"] == ["fix_failures"]
    # Verify execution order
    assert plan.next_pending_step().id == "inspect_repo"
    plan.steps[0].status = "completed"
    assert plan.next_pending_step().id == "identify_framework"


def test_validate_plan_parallel_tasks():
    """Validate a plan with independent parallel tasks."""
    plan = ExecutionPlan(
        objective="Run lint and tests in parallel, then combine results",
        steps=[
            _make_step("lint", title="Run linter", objective="Check code style",
                       execution_instructions="Run linter on all source files",
                       likely_tools=["execute_command"]),
            _make_step("test", title="Run tests", objective="Execute test suite",
                       execution_instructions="Run pytest on all test files",
                       likely_tools=["execute_command"]),
            _make_step("combine", title="Combine results", objective="Merge lint and test results",
                       execution_instructions="Combine lint and test output into a report",
                       deps=["lint", "test"],
                       required_context=["lint", "test"],
                       likely_tools=["create_file"]),
        ],
    )
    errors = validate_plan(plan)
    assert errors == []
    # Both lint and test can run in parallel (no deps)
    step = plan.next_pending_step()
    assert step.id in ("lint", "test")


def test_validate_plan_recovery_tasks():
    """Validate a plan with recovery/retry tasks."""
    plan = ExecutionPlan(
        objective="Run tests and fix failures",
        steps=[
            _make_step("run_tests", title="Run tests", objective="Execute test suite",
                       likely_tools=["execute_command"]),
            _make_step("diagnose", title="Diagnose failures", objective="Analyze test failures",
                       deps=["run_tests"], required_context=["run_tests"],
                       likely_tools=["read_file"]),
            _make_step("fix", title="Fix failures", objective="Resolve test failures",
                       deps=["diagnose"], required_context=["diagnose", "run_tests"],
                       likely_tools=["edit_file"]),
            _make_step("verify", title="Verify fixes", objective="Re-run tests to verify fixes",
                       deps=["fix"], required_context=["fix"],
                       likely_tools=["execute_command"]),
        ],
    )
    errors = validate_plan(plan)
    assert errors == []


# ── Heuristic complexity scoring ────────────────────────────────────────────

def test_heuristic_simple_queries():
    assert _heuristic_complexity_score("what is Python?") == 0.0
    assert _heuristic_complexity_score("explain inheritance") == 0.0


def test_heuristic_complex_queries():
    # High complexity: multi-step with "and then"
    score = _heuristic_complexity_score("find all Python files and then read them")
    assert score >= 0.4
    # Medium complexity: "find and edit"
    score = _heuristic_complexity_score("find and edit config.json")
    assert score >= 0.2


# ── ComplexityGateNode ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_complexity_gate_simple():
    gate = ComplexityGateNode(config=ComplexityGateConfig(heuristic_threshold=0.5))
    packet = OrchaPacket(kind=PacketKind.QUERY, query="what is Python?")
    ctx = MagicMock(spec=RunContext)
    result = await gate.run(packet, ctx)
    assert result.payload["complexity_gate"] == "simple"


@pytest.mark.asyncio
async def test_complexity_gate_complex_heuristic():
    gate = ComplexityGateNode(config=ComplexityGateConfig(heuristic_threshold=0.5))
    packet = OrchaPacket(kind=PacketKind.QUERY,
                         query="find all Python files and then read them")
    ctx = MagicMock(spec=RunContext)
    result = await gate.run(packet, ctx)
    assert result.payload["complexity_gate"] == "complex"


# ══════════════════════════════════════════════════════════════════════════════
# OrchaTaskPlanner tests
# ══════════════════════════════════════════════════════════════════════════════

class TestOrchaTaskPlanner:
    """Tests for the OrchaTaskPlanner core planner."""

    def test_simple_request(self):
        """A simple request produces a single-step plan."""
        planner = OrchaTaskPlanner()
        request = PlannerRequest(
            original_request="what is Python?",
            normalized_objective="Explain what Python is",
        )
        plan = planner.plan(request)
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) >= 1
        errors = validate_plan(plan)
        assert errors == []

    def test_multi_clause_request(self):
        """A multi-clause request produces multiple steps."""
        planner = OrchaTaskPlanner()
        request = PlannerRequest(
            original_request="inspect the repo, identify the framework, and then implement the feature",
            normalized_objective="Understand project and add feature",
        )
        plan = planner.plan(request)
        assert len(plan.steps) >= 2
        errors = validate_plan(plan)
        assert errors == []

    def test_sequential_dependencies(self):
        """Steps in a multi-clause request have proper sequential dependencies."""
        planner = OrchaTaskPlanner()
        request = PlannerRequest(
            original_request="inspect repo, add feature, run tests, fix failures",
            normalized_objective="Full feature implementation cycle",
        )
        plan = planner.plan(request)
        # Each step (except first) depends on the previous
        for i, step in enumerate(plan.steps):
            if i == 0:
                assert step.dependencies == []
            else:
                assert len(step.dependencies) > 0

    def test_plan_has_all_required_fields(self):
        """Every step has all required fields."""
        planner = OrchaTaskPlanner()
        request = PlannerRequest(
            original_request="inspect repo and add feature",
            normalized_objective="Understand and implement",
        )
        plan = planner.plan(request)
        for step in plan.steps:
            assert step.id, "Step must have an ID"
            assert step.title, "Step must have a title"
            assert step.objective, "Step must have an objective"
            assert step.execution_instructions, "Step must have execution instructions"
            assert step.expected_outcome, "Step must have expected outcome"
            assert step.verification_criteria, "Step must have verification criteria"
            assert step.action in ("tool_use", "search", "create", "modify", "execute", "think")

    def test_no_circular_dependencies(self):
        """Plan has no circular dependencies."""
        planner = OrchaTaskPlanner()
        request = PlannerRequest(
            original_request="inspect repo, add feature, run tests, fix failures, verify result",
            normalized_objective="Complete implementation cycle",
        )
        plan = planner.plan(request)
        errors = validate_plan(plan)
        assert errors == []

    def test_no_duplicate_task_ids(self):
        """Plan has no duplicate task IDs."""
        planner = OrchaTaskPlanner()
        request = PlannerRequest(
            original_request="inspect repo and add feature and run tests",
            normalized_objective="Multi-step task",
        )
        plan = planner.plan(request)
        ids = [s.id for s in plan.steps]
        assert len(ids) == len(set(ids))

    def test_plan_status_is_pending(self):
        """New plan starts with pending status."""
        planner = OrchaTaskPlanner()
        request = PlannerRequest(
            original_request="do something",
            normalized_objective="Do something useful",
        )
        plan = planner.plan(request)
        assert plan.status == "pending"

    def test_plan_has_request(self):
        """Plan retains the original request."""
        planner = OrchaTaskPlanner()
        request = PlannerRequest(
            original_request="inspect repo",
            normalized_objective="Understand project",
        )
        plan = planner.plan(request)
        assert plan.request is not None
        assert plan.request.original_request == "inspect repo"

    def test_plan_with_constraints(self):
        """Plan works with constrained requests."""
        planner = OrchaTaskPlanner()
        request = PlannerRequest(
            original_request="add a feature",
            normalized_objective="Implement the feature",
            constraints=["Python 3.11", "no external deps"],
            tools=["read_file", "edit_file"],
        )
        plan = planner.plan(request)
        errors = validate_plan(plan)
        assert errors == []

    def test_plan_with_context(self):
        """Plan works with existing context."""
        planner = OrchaTaskPlanner()
        request = PlannerRequest(
            original_request="fix the bug",
            normalized_objective="Resolve the failing test",
            context="Tests fail with ImportError in module foo",
        )
        plan = planner.plan(request)
        errors = validate_plan(plan)
        assert errors == []

    def test_empty_request_fallback(self):
        """Empty request produces a single fallback step."""
        planner = OrchaTaskPlanner()
        request = PlannerRequest(
            original_request="",
            normalized_objective="",
        )
        plan = planner.plan(request)
        assert len(plan.steps) >= 1
        assert plan.steps[0].id == "execute_directly"

    @pytest.mark.asyncio
    async def test_plan_async_with_model(self):
        """Test async planning with a model callback."""
        valid_plan = json.dumps([
            {
                "id": "step_1",
                "title": "Inspect repository",
                "objective": "Understand the project structure",
                "execution_instructions": "List files and read configs",
                "dependencies": [],
                "expected_outcome": "Clear understanding",
                "required_context": [],
                "likely_tools": ["list_directory", "read_file"],
                "verification_criteria": "Can describe project",
                "action": "search",
            },
        ])

        async def mock_completion(messages, system, tools):
            return {"content": valid_plan}

        planner = OrchaTaskPlanner(completion_fn=mock_completion)
        request = PlannerRequest(
            original_request="inspect the repo",
            normalized_objective="Understand project",
        )
        plan = await planner.plan_async(request)
        assert len(plan.steps) == 1
        assert plan.steps[0].id == "step_1"

    @pytest.mark.asyncio
    async def test_plan_async_model_fallback(self):
        """Test async planning falls back to heuristic on model failure."""
        async def failing_completion(messages, system, tools):
            raise RuntimeError("Model unavailable")

        planner = OrchaTaskPlanner(completion_fn=failing_completion)
        request = PlannerRequest(
            original_request="inspect and add feature",
            normalized_objective="Understand and implement",
        )
        plan = await planner.plan_async(request)
        # Should fall back to heuristic plan
        assert len(plan.steps) >= 1
        errors = validate_plan(plan)
        assert errors == []

    def test_tool_classification(self):
        """Tools are correctly classified for task types."""
        assert "list_directory" in _tools_for_task_type(TaskType.INSPECT)
        assert "execute_command" in _tools_for_task_type(TaskType.EXECUTE)
        assert "edit_file" in _tools_for_task_type(TaskType.MODIFY)
        assert "create_file" in _tools_for_task_type(TaskType.CREATE)

    def test_action_classification(self):
        """Actions are correctly classified for task types."""
        assert _action_for_task_type(TaskType.INSPECT) == "search"
        assert _action_for_task_type(TaskType.EXECUTE) == "execute"
        assert _action_for_task_type(TaskType.CREATE) == "create"
        assert _action_for_task_type(TaskType.MODIFY) == "modify"
        assert _action_for_task_type(TaskType.THINK) == "think"


# ══════════════════════════════════════════════════════════════════════════════
# TaskPlanBuilder tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTaskPlanBuilder:
    """Tests for the TaskPlanBuilder helper."""

    def test_build_simple_plan(self):
        """Build a simple two-step plan."""
        plan = (
            TaskPlanBuilder("Inspect and implement feature")
            .add_step("inspect", "Inspect repo", "Understand project structure",
                      likely_tools=["list_directory", "read_file"])
            .add_step("implement", "Implement feature", "Write the code",
                      dependencies=["inspect"],
                      required_context=["inspect"],
                      likely_tools=["edit_file"])
            .build()
        )
        assert len(plan.steps) == 2
        assert plan.steps[0].dependencies == []
        assert plan.steps[1].dependencies == ["inspect"]
        errors = validate_plan(plan)
        assert errors == []

    def test_build_diamond_plan(self):
        """Build a diamond dependency plan (A -> B, A -> C, B+C -> D)."""
        plan = (
            TaskPlanBuilder("Diamond workflow")
            .add_step("a", "Step A", "Analyze the initial requirements thoroughly")
            .add_step("b", "Step B", "Implement component B based on analysis",
                      dependencies=["a"])
            .add_step("c", "Step C", "Implement component C based on analysis",
                      dependencies=["a"])
            .add_step("d", "Step D", "Integrate components B and C together",
                      dependencies=["b", "c"],
                      required_context=["b", "c"])
            .build()
        )
        graph = plan.get_dependency_graph()
        assert graph["a"] == []
        assert graph["b"] == ["a"]
        assert graph["c"] == ["a"]
        assert graph["d"] == ["b", "c"]
        errors = validate_plan(plan)
        assert errors == []

    def test_build_parallel_plan(self):
        """Build a plan with independent parallel tasks."""
        plan = (
            TaskPlanBuilder("Run lint and tests in parallel")
            .add_step("lint", "Run linter", "Check code style",
                      likely_tools=["execute_command"])
            .add_step("test", "Run tests", "Execute test suite",
                      likely_tools=["execute_command"])
            .add_step("report", "Generate report", "Combine results",
                      dependencies=["lint", "test"],
                      required_context=["lint", "test"])
            .build()
        )
        # lint and test have no deps, so they can run in parallel
        assert plan.steps[0].dependencies == []
        assert plan.steps[1].dependencies == []
        assert plan.steps[2].dependencies == ["lint", "test"]
        errors = validate_plan(plan)
        assert errors == []

    def test_build_recovery_plan(self):
        """Build a plan with recovery/retry tasks."""
        plan = (
            TaskPlanBuilder("Run tests and fix failures")
            .add_step("run_tests", "Run tests", "Execute test suite",
                      likely_tools=["execute_command"])
            .add_step("diagnose", "Diagnose failures", "Analyze test output",
                      dependencies=["run_tests"],
                      required_context=["run_tests"],
                      likely_tools=["read_file"])
            .add_step("fix", "Fix failures", "Resolve test failures",
                      dependencies=["diagnose"],
                      required_context=["diagnose", "run_tests"],
                      likely_tools=["edit_file"])
            .add_step("verify", "Verify fixes", "Re-run tests",
                      dependencies=["fix"],
                      required_context=["fix"],
                      likely_tools=["execute_command"])
            .build()
        )
        errors = validate_plan(plan)
        assert errors == []

    def test_duplicate_id_raises(self):
        """Adding a step with duplicate ID raises ValueError."""
        builder = TaskPlanBuilder("test")
        builder.add_step("a", "Step A", "Do A")
        with pytest.raises(ValueError, match="Duplicate"):
            builder.add_step("a", "Step A again", "Do A again")

    def test_invalid_plan_raises(self):
        """Building a plan with validation errors raises PlanValidationError."""
        builder = TaskPlanBuilder("test")
        # Empty plan should fail validation
        with pytest.raises(PlanValidationError):
            builder.build()

    def test_build_returns_execution_plan(self):
        """build() returns an ExecutionPlan."""
        plan = (
            TaskPlanBuilder("test objective")
            .add_step("step_1", "Step 1", "Do step 1")
            .build()
        )
        assert isinstance(plan, ExecutionPlan)
        assert plan.objective == "test objective"


# ══════════════════════════════════════════════════════════════════════════════
# TaskType classification tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTaskTypeClassification:
    """Tests for task type classification."""

    def test_inspect_classification(self):
        assert _classify_task_type("inspect the repository") == TaskType.INSPECT
        assert _classify_task_type("examine the codebase") == TaskType.INSPECT
        assert _classify_task_type("review the code") == TaskType.INSPECT

    def test_identify_classification(self):
        assert _classify_task_type("identify the framework") == TaskType.IDENTIFY
        assert _classify_task_type("determine the patterns") == TaskType.IDENTIFY

    def test_understand_classification(self):
        assert _classify_task_type("understand the architecture") == TaskType.UNDERSTAND
        assert _classify_task_type("analyze the requirements") == TaskType.UNDERSTAND

    def test_create_classification(self):
        assert _classify_task_type("implement the feature") == TaskType.CREATE
        assert _classify_task_type("add a new function") == TaskType.CREATE
        assert _classify_task_type("create a new module") == TaskType.CREATE

    def test_modify_classification(self):
        assert _classify_task_type("modify the config") == TaskType.MODIFY
        assert _classify_task_type("update the function") == TaskType.MODIFY
        assert _classify_task_type("fix the typo") == TaskType.RECOVER

    def test_execute_classification(self):
        assert _classify_task_type("run the tests") == TaskType.EXECUTE
        assert _classify_task_type("execute the test suite") == TaskType.EXECUTE

    def test_verify_classification(self):
        assert _classify_task_type("verify the result") == TaskType.VERIFY
        assert _classify_task_type("validate the output") == TaskType.VERIFY

    def test_think_fallback(self):
        assert _classify_task_type("something unrelated") == TaskType.THINK


# ══════════════════════════════════════════════════════════════════════════════
# Clause splitting tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSplitClauses:
    """Tests for clause splitting."""

    def test_comma_separated(self):
        clauses = _split_clauses("inspect repo, add feature, run tests")
        assert len(clauses) == 3

    def test_and_separated(self):
        clauses = _split_clauses("inspect repo and add feature")
        assert len(clauses) == 2

    def test_and_then_separated(self):
        clauses = _split_clauses("inspect repo and then add feature")
        assert len(clauses) == 2

    def test_followed_by(self):
        clauses = _split_clauses("inspect repo followed by add feature")
        assert len(clauses) == 2

    def test_single_clause(self):
        clauses = _split_clauses("just do this")
        assert len(clauses) == 1

    def test_empty(self):
        clauses = _split_clauses("")
        assert len(clauses) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Realistic complex request tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRealisticComplexRequests:
    """Tests with realistic complex requests."""

    def test_feature_implementation_cycle(self):
        """Full feature implementation: inspect, understand, implement, test, fix, verify."""
        planner = OrchaTaskPlanner()
        request = PlannerRequest(
            original_request="Inspect this project, understand the existing architecture, "
                             "implement the requested feature, run the tests, fix failures, "
                             "and verify the final result.",
            normalized_objective="Complete feature implementation with verification",
            constraints=["Python 3.11", "use pytest"],
            tools=["list_directory", "read_file", "edit_file", "execute_command", "search_code"],
            workspace_state="Git repo with 50 files, pytest configured",
        )
        plan = planner.plan(request)
        errors = validate_plan(plan)
        assert errors == []
        # Should have multiple steps
        assert len(plan.steps) >= 3
        # First step has no dependencies
        assert plan.steps[0].dependencies == []
        # Later steps depend on earlier ones
        for i, step in enumerate(plan.steps):
            if i > 0:
                assert len(step.dependencies) > 0

    def test_parallel_quality_checks(self):
        """Parallel lint and test, then combine results."""
        builder = TaskPlanBuilder("Run quality checks in parallel")
        plan = (
            builder
            .add_step("lint", "Run linter", "Check code style across all files",
                      likely_tools=["execute_command"])
            .add_step("typecheck", "Run type checker", "Check types across all files",
                      likely_tools=["execute_command"])
            .add_step("test", "Run tests", "Execute full test suite",
                      likely_tools=["execute_command"])
            .add_step("report", "Generate quality report", "Combine all results",
                      dependencies=["lint", "typecheck", "test"],
                      required_context=["lint", "typecheck", "test"],
                      likely_tools=["create_file"])
            .build()
        )
        errors = validate_plan(plan)
        assert errors == []
        # lint, typecheck, test are independent
        assert plan.steps[0].dependencies == []
        assert plan.steps[1].dependencies == []
        assert plan.steps[2].dependencies == []
        # report depends on all three
        assert set(plan.steps[3].dependencies) == {"lint", "typecheck", "test"}

    def test_debug_and_fix_cycle(self):
        """Diagnose a bug, fix it, verify the fix."""
        builder = TaskPlanBuilder("Debug and fix failing test")
        plan = (
            builder
            .add_step("reproduce", "Reproduce the failure", "Run the failing test",
                      likely_tools=["execute_command"])
            .add_step("diagnose", "Diagnose root cause", "Analyze test output and stack trace",
                      dependencies=["reproduce"],
                      required_context=["reproduce"],
                      likely_tools=["read_file", "search_code"])
            .add_step("fix", "Apply fix", "Modify code to resolve the issue",
                      dependencies=["diagnose"],
                      required_context=["diagnose", "reproduce"],
                      likely_tools=["edit_file"])
            .add_step("verify", "Verify fix", "Re-run the test to confirm resolution",
                      dependencies=["fix"],
                      required_context=["fix"],
                      likely_tools=["execute_command"])
            .build()
        )
        errors = validate_plan(plan)
        assert errors == []

    def test_multi_file_refactor(self):
        """Refactor across multiple files with dependency chain."""
        planner = OrchaTaskPlanner()
        request = PlannerRequest(
            original_request="Find all usages of the old API, "
                             "update each file to use the new API, "
                             "run the tests, and fix any breakage",
            normalized_objective="Migrate from old API to new API",
        )
        plan = planner.plan(request)
        errors = validate_plan(plan)
        assert errors == []
        assert len(plan.steps) >= 2

    def test_documentation_update(self):
        """Update documentation after code changes."""
        builder = TaskPlanBuilder("Update documentation")
        plan = (
            builder
            .add_step("read_docs", "Read existing docs", "Understand current documentation structure",
                      likely_tools=["list_directory", "read_file"])
            .add_step("identify_gaps", "Identify documentation gaps", "Find areas missing docs",
                      dependencies=["read_docs"],
                      required_context=["read_docs"],
                      likely_tools=["search_code"])
            .add_step("write_docs", "Write new documentation", "Create/update documentation files",
                      dependencies=["identify_gaps"],
                      required_context=["identify_gaps", "read_docs"],
                      likely_tools=["create_file", "edit_file"])
            .add_step("verify_docs", "Verify documentation", "Ensure docs are accurate and complete",
                      dependencies=["write_docs"],
                      required_context=["write_docs"],
                      likely_tools=["read_file"])
            .build()
        )
        errors = validate_plan(plan)
        assert errors == []


# ══════════════════════════════════════════════════════════════════════════════
# Validation edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestValidationEdgeCases:
    """Tests for validation edge cases."""

    def test_rejects_self_dependency(self):
        """Self-dependency is rejected."""
        plan = ExecutionPlan(
            objective="test",
            steps=[_make_step("a", deps=["a"])],
        )
        errors = validate_plan(plan)
        assert any("itself" in e.lower() for e in errors)

    def test_rejects_impossible_dependency(self):
        """Dependency on non-existent task is rejected."""
        plan = ExecutionPlan(
            objective="test",
            steps=[_make_step("a", deps=["nonexistent"])],
        )
        errors = validate_plan(plan)
        assert any("non-existent" in e.lower() for e in errors)

    def test_rejects_circular_dependency(self):
        """Circular dependency is rejected."""
        plan = ExecutionPlan(
            objective="test",
            steps=[
                _make_step("a", deps=["c"]),
                _make_step("b", deps=["a"]),
                _make_step("c", deps=["b"]),
            ],
        )
        errors = validate_plan(plan)
        assert any("circular" in e.lower() for e in errors)

    def test_rejects_empty_plan(self):
        """Empty plan is rejected."""
        plan = ExecutionPlan(objective="test", steps=[])
        errors = validate_plan(plan)
        assert len(errors) == 1

    def test_rejects_duplicate_ids(self):
        """Duplicate task IDs are rejected."""
        plan = ExecutionPlan(
            objective="test",
            steps=[_make_step("a"), _make_step("a")],
        )
        errors = validate_plan(plan)
        assert any("duplicate" in e.lower() for e in errors)

    def test_rejects_empty_objective(self):
        """Empty objective is rejected."""
        plan = ExecutionPlan(
            objective="test",
            steps=[_make_step("a", objective="")],
        )
        errors = validate_plan(plan)
        assert any("objective" in e.lower() for e in errors)

    def test_rejects_short_objective(self):
        """Too-short objective is rejected."""
        plan = ExecutionPlan(
            objective="test",
            steps=[_make_step("a", objective="abc")],
        )
        errors = validate_plan(plan)
        assert any("objective" in e.lower() for e in errors)

    def test_rejects_empty_title(self):
        """Empty title is rejected."""
        plan = ExecutionPlan(
            objective="test",
            steps=[_make_step("a", title="")],
        )
        errors = validate_plan(plan)
        assert any("title" in e.lower() for e in errors)


# ══════════════════════════════════════════════════════════════════════════════
# Plan graph structure tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPlanGraphStructure:
    """Tests for plan graph structure and traversal."""

    def test_dependency_graph(self):
        """Dependency graph is correctly computed."""
        plan = ExecutionPlan(
            objective="test",
            steps=[
                _make_step("a"),
                _make_step("b", deps=["a"]),
                _make_step("c", deps=["a", "b"]),
            ],
        )
        graph = plan.get_dependency_graph()
        assert graph["a"] == []
        assert graph["b"] == ["a"]
        assert graph["c"] == ["a", "b"]

    def test_next_pending_step(self):
        """next_pending_step respects dependencies."""
        plan = ExecutionPlan(
            objective="test",
            steps=[
                _make_step("a"),
                _make_step("b", deps=["a"]),
                _make_step("c", deps=["b"]),
            ],
        )
        # First step is ready
        assert plan.next_pending_step().id == "a"
        # Complete first step
        plan.steps[0].status = "completed"
        # Second step is now ready
        assert plan.next_pending_step().id == "b"
        # Complete second step
        plan.steps[1].status = "completed"
        # Third step is now ready
        assert plan.next_pending_step().id == "c"

    def test_next_pending_step_deps_not_met(self):
        """next_pending_step skips steps whose deps are not met."""
        plan = ExecutionPlan(
            objective="test",
            steps=[
                _make_step("a"),
                _make_step("b", deps=["a"]),
            ],
        )
        # b depends on a, so only a is ready
        step = plan.next_pending_step()
        assert step.id == "a"

    def test_progress_tracking(self):
        """Progress is correctly tracked."""
        plan = ExecutionPlan(
            objective="test",
            steps=[
                _make_step("a"),
                _make_step("b"),
                _make_step("c"),
            ],
        )
        assert plan.progress == 0.0
        plan.steps[0].status = "completed"
        assert abs(plan.progress - 1 / 3) < 0.01
        plan.steps[1].status = "completed"
        assert abs(plan.progress - 2 / 3) < 0.01
        plan.steps[2].status = "completed"
        assert plan.progress == 1.0
        assert plan.is_complete

    def test_step_by_id(self):
        """step_by_id finds steps correctly."""
        plan = ExecutionPlan(
            objective="test",
            steps=[_make_step("a"), _make_step("b")],
        )
        assert plan.step_by_id("a") is not None
        assert plan.step_by_id("a").id == "a"
        assert plan.step_by_id("nonexistent") is None

    def test_get_step_ids(self):
        """get_step_ids returns all IDs in order."""
        plan = ExecutionPlan(
            objective="test",
            steps=[_make_step("a"), _make_step("b"), _make_step("c")],
        )
        assert plan.get_step_ids() == ["a", "b", "c"]
