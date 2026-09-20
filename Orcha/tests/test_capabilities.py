"""
Tests for the capability/tool system:
registry routing, schemas, safety metadata, structured errors, approval/action
modes, executor fault-isolation and the ToolExecutor-backed agent native loop.
"""
import asyncio

import pytest

from orcha.capabilities.base import (
    WRITE,
    CapabilityContext,
    PermissionPolicy,
    ToolExecutor,
    ToolResult,
    spec,
)
from orcha.capabilities.registry import CapabilityRegistry
from orcha.capabilities.reasoning import LEVELS, reasoning_config

from test_nodes import make_ctx, make_packet


@pytest.fixture
def registry():
    return CapabilityRegistry().register_defaults()


@pytest.fixture
def ctx(tmp_path):
    return CapabilityContext(roots=[str(tmp_path)])


def _build(registry, ctx, names=None, policy=None):
    return registry.build(names, ctx=ctx, policy=policy)


# ── Registry routing ──────────────────────────────────────────────────────────

def test_registry_has_all_eight_capabilities(registry):
    assert set(registry.names()) == {
        "filesystem", "workspace", "search", "web", "terminal", "git",
        "diagnostics", "code_intelligence",
    }


def test_resolve_capabilities_validates_names(registry):
    assert registry.resolve_capabilities(["filesystem", "search"]) == ["filesystem", "search"]
    with pytest.raises(ValueError):
        registry.resolve_capabilities(["filesystem", "bogus"])


def test_build_subset_routes_only_declared_capabilities(registry, ctx):
    ex = _build(registry, ctx, ["filesystem", "search"])
    names = set(ex.names())
    assert "read_file" in names and "grep" in names
    assert "run_command" not in names  # terminal not declared


def test_build_all_by_default(registry, ctx):
    ex = _build(registry, ctx)
    assert "read_file" in ex.names()
    assert "run_command" in ex.names()
    assert "git_status" in ex.names()


# ── Schema + metadata ─────────────────────────────────────────────────────────

def test_every_tool_has_metadata(registry, ctx):
    ex = _build(registry, ctx)
    for tool in ex.describe():
        assert tool["name"]
        assert tool["description"]
        assert tool["capability"] in registry.names()
        assert set(tool["permissions"]) <= {"read", "write", "execute", "network", "dangerous"}
        assert tool["safety_level"] in {"safe", "cautious", "dangerous"}
        assert isinstance(tool["parameters"], dict)


def test_openai_schema_shape(registry, ctx):
    ex = _build(registry, ctx, ["filesystem"])
    schema = next(s for s in ex.schemas() if s["function"]["name"] == "read_file")
    assert schema["type"] == "function"
    assert "path" in schema["function"]["parameters"]["properties"]


def test_dangerous_tools_flagged(registry, ctx):
    ex = _build(registry, ctx, ["filesystem"])
    spec_by_name = {t.name: t for t in ex.tools()}
    assert spec_by_name["delete_file"].needs_approval() is True
    assert spec_by_name["read_file"].needs_approval() is False


# ── Path safety ───────────────────────────────────────────────────────────────

def test_traversal_rejected(registry, ctx):
    ex = _build(registry, ctx, ["filesystem"])
    r = ex.invoke("read_file", path="..\\..\\windows\\win.ini")
    assert r.ok is False
    assert r.error["code"] == "path_outside_workspace"


def test_absolute_outside_root_rejected(registry, ctx, tmp_path):
    outside = tmp_path.parent / "sneaky.txt"
    outside.write_text("secret")
    ex = _build(registry, ctx, ["filesystem"])
    r = ex.invoke("read_file", path=str(outside))
    assert r.error["code"] == "path_outside_workspace"


def test_missing_file_structured_error(registry, ctx):
    ex = _build(registry, ctx, ["filesystem"])
    r = ex.invoke("read_file", path="missing.txt")
    assert r.error["code"] == "file_not_found"


def test_filesystem_write_read_roundtrip(registry, ctx):
    ex = _build(registry, ctx, ["filesystem"])
    assert ex.invoke("write_file", path="a.txt", content="hello").ok
    r = ex.invoke("read_file", path="a.txt")
    assert r.ok and r.value == "hello"


def test_filesystem_write_emits_real_file_change(registry, ctx):
    ex = _build(registry, ctx, ["filesystem"])
    created = ex.invoke("write_file", path="a.txt", content="hello\n")
    assert created.ok
    first = created.value.file_changes[0]
    assert first["operation"] == "created"
    assert first["before"] == ""
    assert first["after"] == "hello\n"
    assert "+hello" in first["diff"]

    modified = ex.invoke("write_file", path="a.txt", content="hello from Anvira\n")
    assert modified.ok
    second = modified.value.file_changes[0]
    assert second["operation"] == "modified"
    assert second["before"] == "hello\n"
    assert second["after"] == "hello from Anvira\n"
    assert "-hello" in second["diff"] and "+hello from Anvira" in second["diff"]


# ── Structured errors + fault isolation ───────────────────────────────────────

def test_validation_error_missing_required(registry, ctx):
    ex = _build(registry, ctx, ["filesystem"])
    r = ex.invoke("read_file")  # missing required path
    assert r.error["code"] == "validation_error"


def test_unknown_tool_error(registry, ctx):
    ex = _build(registry, ctx)
    r = ex.invoke("does_not_exist")
    assert r.error["code"] == "unknown_tool"


def test_common_tool_name_alias_is_resolved(registry, ctx, tmp_path):
    """
    The same wrong-but-plausible-guess problem test_common_argument_name_
    alias_is_normalized fixes for argument names also happens for the tool
    NAME itself — a model reaching for "write_tool" when the real tool is
    "write_file". Previously this failed outright with "unknown_tool" and
    cost a whole extra round-trip (or, for a weaker model that never
    retries, silently gave up). See orcha/capabilities/tool_aliases.py.
    """
    ex = _build(registry, ctx, ["filesystem"])
    target = tmp_path / "aliased.txt"
    r = ex.invoke("write_tool", path=str(target), content="hi")
    assert r.ok is True, r.error
    assert target.read_text() == "hi"


def test_tool_name_typo_resolves_via_fuzzy_match(registry, ctx, tmp_path):
    """A plain typo of a real tool name (not in the curated alias list)
    should still resolve via the normalize/fuzzy fallback."""
    ex = _build(registry, ctx, ["filesystem"])
    target = tmp_path / "typo.txt"
    r = ex.invoke("write_fiel", path=str(target), content="hi")
    assert r.ok is True, r.error


def test_genuinely_unknown_tool_still_reports_unknown_tool(registry, ctx):
    """The alias/fuzzy resolver must not paper over an actually-bogus tool
    name — it should still report unknown_tool, not silently resolve to
    something unrelated. (Regression: an early version of the fuzzy
    fallback matched "does_not_exist" to "exists" on substring overlap.)"""
    ex = _build(registry, ctx)
    r = ex.invoke("does_not_exist")
    assert r.error["code"] == "unknown_tool"


def test_common_argument_name_alias_is_normalized(registry, ctx):
    """
    Local/edge models frequently guess a plausible-but-wrong argument name
    (e.g. "filename" instead of create_file's real "path") even when the
    schema documents the correct one, and don't reliably self-correct after
    a validation error. The executor should rename a known alias to the
    real parameter name so the call just succeeds instead of failing on a
    purely cosmetic naming mismatch.
    """
    ex = _build(registry, ctx, ["filesystem"])
    r = ex.invoke("create_file", filename="notes.txt", content="hello")
    assert r.ok is True, r.error

    # The real name always wins if the model (unusually) supplies both.
    r2 = ex.invoke("create_file", filename="wrong.txt", path="right.txt", content="hi")
    assert r2.ok is True, r2.error


def test_executor_never_crashes_on_generic_exception():
    def boom():
        raise RuntimeError("kaboom")
    executor = ToolExecutor([
        spec("boom_tool", "explodes", {"type": "object"}, boom),
    ])
    r = executor.invoke("boom_tool")
    assert r.ok is False
    assert r.error["code"] == "tool_error"
    assert "kaboom" in r.error["message"]


def test_tool_result_message_rendering():
    ok = ToolResult.success({"x": 1})
    assert '"x": 1' in ok.to_message()
    err = ToolResult.failure("code_x", "msg", {"k": "v"})
    assert "code_x" in err.to_message()


# ── Approval + action modes ───────────────────────────────────────────────────

def test_approval_mode_blocks_dangerous_tools(registry, ctx):
    policy = PermissionPolicy(access_mode="approval")
    ex = _build(registry, ctx, ["filesystem"], policy=policy)
    r = ex.invoke("delete_file", path="x.txt")
    assert r.ok is False
    assert r.error["code"] == "approval_required"
    # Safe read still runs in approval mode.
    assert ex.invoke("read_file", path="missing.txt").error["code"] == "file_not_found"


def test_approval_mode_with_trusted_tool(registry, ctx):
    policy = PermissionPolicy(access_mode="approval", trusted_tools=["delete_file"])
    ex = _build(registry, ctx, ["filesystem"], policy=policy)
    r = ex.invoke("delete_file", path="missing.txt")
    assert r.error["code"] == "file_not_found"  # ran (not blocked by approval)


def test_action_mode_runs_dangerous_tools(registry, ctx, tmp_path):
    policy = PermissionPolicy(access_mode="action")
    ex = _build(registry, ctx, ["filesystem"], policy=policy)
    assert ex.invoke("write_file", path="d.txt", content="x").ok
    r = ex.invoke("delete_file", path="d.txt")
    assert r.ok is True


# ── Reasoning levels ──────────────────────────────────────────────────────────

def test_reasoning_levels_and_knobs():
    assert LEVELS == ["fast", "light", "medium", "high", "max"]
    fast = reasoning_config("fast")
    mx = reasoning_config("max")
    assert fast["max_iterations"] < mx["max_iterations"]
    assert mx["verify"] is True and mx["multi_agent"] is True
    assert fast["multi_agent"] is False
    with pytest.raises(ValueError):
        reasoning_config("nope")


def test_pipeline_config_scales_effort():
    from orcha.capabilities.reasoning import pipeline_config

    fast = pipeline_config("fast")
    light = pipeline_config("light")
    med = pipeline_config("medium")
    high = pipeline_config("high")
    mx = pipeline_config("max")
    # Low levels: one-shot, no expert fan-out, permissive threshold.
    assert fast["max_iterations"] == 1
    assert fast["run_all_experts"] is False
    assert fast["width"] == 1
    assert fast["threshold"] < 0.60  # looser than the default floor
    assert light["width"] > fast["width"]
    # Medium leaves the pipeline at its defaults.
    assert med["max_iterations"] is None and med["run_all_experts"] is None
    # High/max spend more: more iterations, all experts, stricter bar.
    assert high["max_iterations"] == 4
    assert mx["max_iterations"] == 5
    assert mx["run_all_experts"] is True
    assert mx["width"] > high["width"] > fast["width"]
    assert mx["threshold"] > high["threshold"] > 0.60
    # Output budget scales with level.
    assert fast["output_scale"] < med["output_scale"] < mx["output_scale"]
    with pytest.raises(ValueError):
        pipeline_config("nope")


def test_effort_for_level_mapping():
    from orcha.capabilities.reasoning import effort_for_level

    assert effort_for_level("fast") == "low"
    assert effort_for_level("light") == "low"
    assert effort_for_level("medium") == "medium"
    assert effort_for_level("max") == "high"


def test_model_native_reasoning_detection():
    from orcha.capabilities.reasoning import model_supports_native_reasoning

    assert model_supports_native_reasoning("deepseek-r1:8b")
    assert model_supports_native_reasoning("QwQ-32B-Q4_K_M")
    assert model_supports_native_reasoning("gpt-oss-20b")
    assert not model_supports_native_reasoning("llama3.1:8b")
    assert not model_supports_native_reasoning(None)
    assert not model_supports_native_reasoning("")


# ── Agent native loop with ToolExecutor ───────────────────────────────────────

def _make_ctx():
    return make_ctx()


def _make_packet(**kwargs):
    return make_packet(**kwargs)


def test_agent_native_loop_runs_tool_through_executor(tmp_path):
    from orcha.nodes.agent import AgentConfig, AgentNode

    registry = CapabilityRegistry().register_defaults()
    ctx = CapabilityContext(roots=[str(tmp_path)])
    executor = registry.build(["filesystem"], ctx=ctx)

    async def fake_completion(messages, system, tool_schemas):
        if not any(m.get("role") == "tool" for m in messages):
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": {"path": "made.txt", "content": "hi"}},
                }],
            }
        return {"role": "assistant", "content": "done"}

    agent = AgentNode(
        name="cap_agent",
        config=AgentConfig(completion_fn=fake_completion, executor=executor, max_iterations=4),
    )
    result = asyncio.run(agent.run(_make_packet(task="t"), _make_ctx()))
    assert result.payload["agent_completed"] is True
    call = result.payload["agent_tool_calls"][0]
    assert call["name"] == "write_file" and call["result_type"] == "ok"
    assert (tmp_path / "made.txt").read_text() == "hi"


def test_agent_native_loop_relays_approval_required(tmp_path):
    from orcha.nodes.agent import AgentConfig, AgentNode

    registry = CapabilityRegistry().register_defaults()
    ctx = CapabilityContext(roots=[str(tmp_path)])
    executor = registry.build(["filesystem"], ctx=ctx, policy=PermissionPolicy(access_mode="approval"))

    async def fake_completion(messages, system, tool_schemas):
        if not any(m.get("role") == "tool" for m in messages):
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "delete_file", "arguments": {"path": "x.txt"}},
                }],
            }
        return {"role": "assistant", "content": "asked for approval"}

    agent = AgentNode(
        name="approval_agent",
        config=AgentConfig(completion_fn=fake_completion, executor=executor, max_iterations=4),
    )
    result = asyncio.run(agent.run(_make_packet(task="t"), _make_ctx()))
    call = result.payload["agent_tool_calls"][0]
    assert call["error_code"] == "approval_required"
    assert "Approval is required" in call["result"]


# ── Workspace / search smoke ──────────────────────────────────────────────────

def test_workspace_and_search_capabilities(tmp_path):
    (tmp_path / "readme.md").write_text("the needle is here\n", encoding="utf-8")
    registry = CapabilityRegistry().register_defaults()
    ctx = CapabilityContext(roots=[str(tmp_path)])
    ex = registry.build(["workspace", "search"], ctx=ctx)

    r = ex.invoke("current_workspace")
    assert r.ok and str(tmp_path) in r.value["roots"]

    g = ex.invoke("grep", query="needle", path=".")
    assert g.ok and "readme.md" in g.value


# ── Legacy facade delegation ──────────────────────────────────────────────────

def test_legacy_facade_delegates_metadata(tmp_path):
    from orcha.tools.workspace import build_workspace_tools

    tools = build_workspace_tools([str(tmp_path)])
    spec_by_name = {t.name: t for t in tools}
    assert set(spec_by_name) == {"read_file", "write_file", "edit_file", "list_dir", "search_text"}
    assert spec_by_name["write_file"].capability == "filesystem"
    assert spec_by_name["write_file"].permissions == [WRITE]
