"""
Tests for orcha.tools.workspace — root-guarded filesystem tools for agents.
"""
import pytest

from orcha.tools.workspace import (
    WorkspaceToolError,
    build_workspace_tools,
    tool_schemas,
)


@pytest.fixture
def roots(tmp_path):
    return [str(tmp_path)]


@pytest.fixture
def tools(roots):
    return build_workspace_tools(roots)


def _by_name(tools, name):
    return next(t for t in tools if t.name == name)


# ── Tool catalogue ────────────────────────────────────────────────────────────

def test_builds_all_tools_by_default(tools):
    names = {t.name for t in tools}
    assert names == {"read_file", "write_file", "edit_file", "list_dir", "search_text"}


def test_allow_whitelist_filters(roots):
    tools = build_workspace_tools(roots, allow=["read_file", "write_file"])
    assert {t.name for t in tools} == {"read_file", "write_file"}


def test_tool_schemas_shape():
    schemas = tool_schemas()
    assert set(schemas) >= {"read_file", "write_file", "edit_file"}
    assert schemas["read_file"]["type"] == "object"
    assert "path" in schemas["read_file"]["properties"]


def test_spec_has_openai_shape(tools):
    spec = _by_name(tools, "write_file")
    schema = spec.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "write_file"
    assert schema["function"]["description"]
    assert schema["function"]["parameters"]["required"] == ["path", "content"]


# ── Round trips ───────────────────────────────────────────────────────────────

def test_write_then_read_roundtrip(tools):
    write = _by_name(tools, "write_file")
    read = _by_name(tools, "read_file")

    out = write.invoke_kwargs(path="hello.txt", content="line one\nline two\n")
    assert "created" in out

    text = read.invoke_kwargs(path="hello.txt")
    assert text == "line one\nline two\n"


def test_write_updates_existing_file(tools):
    write = _by_name(tools, "write_file")
    assert "created" in write.invoke_kwargs(path="a.txt", content="v1")
    out = write.invoke_kwargs(path="a.txt", content="v2")
    assert "updated" in out

    read = _by_name(tools, "read_file")
    assert read.invoke_kwargs(path="a.txt") == "v2"


def test_write_create_only_refuses_existing(tools):
    write = _by_name(tools, "write_file")
    write.invoke_kwargs(path="x.txt", content="x")
    with pytest.raises(WorkspaceToolError):
        write.invoke_kwargs(path="x.txt", content="y", create_only=True)


def test_edit_file_targeted_replace(tools):
    write = _by_name(tools, "write_file")
    edit = _by_name(tools, "edit_file")
    write.invoke_kwargs(path="e.txt", content="foo bar foo")
    edit.invoke_kwargs(path="e.txt", old_string="foo", new_string="baz")
    read = _by_name(tools, "read_file")
    assert read.invoke_kwargs(path="e.txt") == "baz bar foo"


def test_edit_file_replace_all(tools):
    write = _by_name(tools, "write_file")
    edit = _by_name(tools, "edit_file")
    write.invoke_kwargs(path="e.txt", content="foo foo foo")
    edit.invoke_kwargs(path="e.txt", old_string="foo", new_string="bar", replace_all=True)
    read = _by_name(tools, "read_file")
    assert read.invoke_kwargs(path="e.txt") == "bar bar bar"


def test_edit_missing_old_string_raises(tools):
    write = _by_name(tools, "write_file")
    edit = _by_name(tools, "edit_file")
    write.invoke_kwargs(path="e.txt", content="hello")
    with pytest.raises(WorkspaceToolError):
        edit.invoke_kwargs(path="e.txt", old_string="nope", new_string="x")


# ── Guardrails ────────────────────────────────────────────────────────────────

def test_absolute_path_outside_root_rejected(tools, tmp_path):
    outside = tmp_path.parent / "sneaky.txt"
    outside.write_text("secret")
    with pytest.raises(WorkspaceToolError):
        _by_name(tools, "read_file").invoke_kwargs(path=str(outside))


def test_relative_traversal_rejected(tools):
    with pytest.raises(WorkspaceToolError):
        _by_name(tools, "read_file").invoke_kwargs(path="../outside.txt")


def test_no_roots_rejected():
    tools = build_workspace_tools([])
    with pytest.raises(WorkspaceToolError):
        tools[0].invoke_kwargs(path="anything")


def test_read_missing_file_raises(tools):
    with pytest.raises(WorkspaceToolError):
        _by_name(tools, "read_file").invoke_kwargs(path="missing.txt")


# ── Listing & search ──────────────────────────────────────────────────────────

def test_list_dir(tools, tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    listing = _by_name(tools, "list_dir").invoke_kwargs(path=".")
    assert "a.txt" in listing
    assert "sub" in listing


def test_list_dir_ignores_hidden(tools, tmp_path):
    (tmp_path / ".hidden").write_text("h")
    (tmp_path / "visible.txt").write_text("v")
    listing = _by_name(tools, "list_dir").invoke_kwargs(path=".")
    assert ".hidden" not in listing
    assert "visible.txt" in listing


def test_search_text(tools, tmp_path):
    (tmp_path / "code.ts").write_text("const needle = 42;\nconst other = 1;\n")
    (tmp_path / "readme.md").write_text("no match here")
    out = _by_name(tools, "search_text").invoke_kwargs(query="needle", path=".", glob="*.ts")
    assert "code.ts:1:" in out
    assert "no match" not in out


def test_search_no_matches(tools, tmp_path):
    (tmp_path / "f.txt").write_text("nothing")
    out = _by_name(tools, "search_text").invoke_kwargs(query="zzz", path=".")
    assert "No matches" in out


# ── Native agent tool loop ────────────────────────────────────────────────────

def test_agent_native_loop_executes_tool_call(tmp_path):
    """An agent completion that emits a tool_call then a final answer runs the
    tool and returns the answer."""
    import asyncio

    from orcha.nodes.agent import AgentConfig, AgentNode
    from test_nodes import make_ctx, make_packet

    tools = build_workspace_tools([str(tmp_path)])

    calls = []

    async def fake_completion(messages, system, tool_schemas):
        calls.append(len(messages))
        # First call → issue write_file tool call; second call → final answer.
        if not any(m.get("role") == "tool" for m in messages):
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": {"path": "made.txt", "content": "made by agent"},
                    },
                }],
            }
        return {"role": "assistant", "content": "file created"}

    agent = AgentNode(
        name="test_agent",
        config=AgentConfig(
            system_prompt="Task: {task} Context: {context}",
            completion_fn=fake_completion,
            tools=tools,
            max_iterations=5,
        ),
    )

    pkt = make_packet(task="create a file")
    result = asyncio.run(agent.run(pkt, make_ctx()))

    assert result.payload["agent_completed"] is True
    assert result.payload["agent_output"] == "file created"
    assert len(result.payload["agent_tool_calls"]) == 1
    assert result.payload["agent_tool_calls"][0]["name"] == "write_file"
    assert (tmp_path / "made.txt").read_text() == "made by agent"
    # First completion got the schema, then the tool result was fed back.
    assert calls == [1, 3]


def test_agent_native_loop_unknown_tool(tools, tmp_path):
    import asyncio

    from orcha.nodes.agent import AgentConfig, AgentNode
    from test_nodes import make_ctx, make_packet

    async def fake_completion(messages, system, tool_schemas):
        if not any(m.get("role") == "tool" for m in messages):
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_x",
                    "type": "function",
                    "function": {"name": "nope_tool", "arguments": {}},
                }],
            }
        return {"role": "assistant", "content": "FINAL ANSWER: done"}

    agent = AgentNode(
        name="test_agent",
        config=AgentConfig(
            completion_fn=fake_completion,
            tools=tools,
            max_iterations=3,
        ),
    )
    result = asyncio.run(agent.run(make_packet(task="x"), make_ctx()))
    calls = result.payload["agent_tool_calls"]
    assert len(calls) == 1
    assert calls[0]["name"] == "nope_tool"
    assert "[No tool" in calls[0]["result"]
