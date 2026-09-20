"""
Tests for the skills API manager: directory scanning, listing shape, and
injection of skill tools into agent run executors (via _build_agent_tools).
"""
import json

import pytest


@pytest.fixture
def skills_dir(tmp_path, monkeypatch):
    """Two skills: one prompt-mode markdown package, one script."""
    d = tmp_path / "skills"
    review = d / "review_pr"
    review.mkdir(parents=True)
    (review / "SKILL.md").write_text(
        "---\n"
        "name: review_pr\n"
        'description: "Review a diff for correctness"\n'
        'when_to_use: "when the user shares a diff"\n'
        "context: inline\n"
        "---\n"
        "Review $ARGUMENTS for edge cases.",
        encoding="utf-8",
    )
    greet = d / "greet"
    greet.mkdir(parents=True)
    (greet / "SKILL.md").write_text(
        "---\nname: greet\ndescription: says hello\n---\nunused",
        encoding="utf-8",
    )
    (greet / "entrypoint.py").write_text("def run():\n    return 'hi'\n", encoding="utf-8")
    monkeypatch.setenv("ORCHA_SKILLS_DIR", str(d))
    return d


def test_list_skills_shape(skills_dir):
    from orcha.api.skills_manager import SkillsManager

    manager = SkillsManager()
    items = manager.list_skills()
    by_name = {item["name"]: item for item in items}
    assert set(by_name) == {"review_pr", "greet"}
    assert by_name["review_pr"]["mode"] == "prompt"
    assert by_name["review_pr"]["when_to_use"] == "when the user shares a diff"
    assert by_name["greet"]["mode"] == "script"
    assert all(item["available"] for item in items)


def test_active_tool_specs_injectable(skills_dir):
    from orcha.api.skills_manager import SkillsManager
    from orcha.capabilities.base import ToolExecutor

    specs = SkillsManager().active_tool_specs()
    names = {s.name for s in specs}
    assert {"review_pr", "greet"} <= names
    # Prompt skill is read-only and executable through the executor.
    ex = ToolExecutor(specs)
    result = ex.invoke("review_pr", **{"diff": "-a\n+b"})
    assert result.ok and "edge cases" in str(result.value)


def test_missing_directory_is_silently_skipped(tmp_path, monkeypatch):
    from orcha.api.skills_manager import SkillsManager

    monkeypatch.setenv("ORCHA_SKILLS_DIR", str(tmp_path / "nope"))
    manager = SkillsManager()
    assert manager.list_skills() == []
    assert manager.active_tool_specs() == []


def test_run_request_carries_rules_end_to_end(skills_dir, monkeypatch, tmp_path):
    """
    Integration: a run request with allow_rules + skills on disk builds an
    executor whose policy auto-approves the allowed rule even in approval
    mode, and whose tool surface includes skill tools.
    """
    from orcha.api.server import _build_agent_tools

    monkeypatch.setenv("ORCHA_MCP_CONFIG", str(tmp_path / "mcp.json"))
    built = _build_agent_tools(
        [str(tmp_path)],
        ["filesystem"],
        None,
        "approval",
        True,
        allow_rules=["read_file(docs/*)"],
    )
    ex = built["executor"]
    names = set(ex.names())
    assert "review_pr" in names  # skill injected
    spec_delete = next(s for s in ex.tools() if s.name == "delete_file")
    # Dangerous tool: no matching rule → interactive ask in approval mode.
    assert ex.policy.evaluate(spec_delete, {"path": "docs/x.md"}) == "ask"
    # A matching allow rule overrides the default ask.
    spec_read = next(s for s in ex.tools() if s.name == "read_file")
    assert ex.policy.evaluate(spec_read, {"path": "src/app.py"}) == "allow"
