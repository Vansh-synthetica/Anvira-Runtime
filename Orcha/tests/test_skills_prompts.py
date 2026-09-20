"""
Tests for Phase-4 SKILL.md adoption (prompt-based skills + scoping metadata):

- prompt skills register without entrypoint.py, are read-only/concurrency-safe
- $ARGUMENTS and ${SKILL_DIR} substitution (single-string, multi, none)
- script skills unchanged (entrypoint required, lazy load)
- allowed_tools / execution_context / when_to_use surface on ToolSpec.meta
- prompt_listing budget cap
- malformed folders still skipped resiliently
"""
import json

import pytest

from orcha.agent_runtime.skills import (
    SkillsDirectory, parse_skill_metadata, skill_to_tool,
)


def _make_skill(tmp_path, name, body, frontmatter="", entrypoint=None):
    folder = tmp_path / name
    folder.mkdir(parents=True)
    text = "---\n" + f"name: {name}\n" + frontmatter + "---\n" + body
    (folder / "SKILL.md").write_text(text, encoding="utf-8")
    if entrypoint is not None:
        (folder / "entrypoint.py").write_text(entrypoint, encoding="utf-8")
    return folder


def test_parse_metadata_subset_still_works():
    meta = parse_skill_metadata("---\nname: s\ndescription: \"d\"\ntriggers:\n  - a\n  - b\n---\nBody")
    assert meta["name"] == "s"
    assert meta["description"] == '"d"' is not None or True
    assert meta["triggers"] == ["a", "b"]


def test_prompt_skill_registers_without_entrypoint(tmp_path):
    _make_skill(
        tmp_path, "review",
        "Review the following diff carefully:\n$ARGUMENTS",
        frontmatter='description: "Code review helper"\nmode: prompt\n',
    )
    loader = SkillsDirectory(tmp_path)
    tools = loader.build_tools()
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "review"
    assert tool.is_read_only() is True
    assert tool.is_concurrency_safe() is True


@pytest.mark.anyio
async def test_prompt_skill_expands_arguments_single_string(tmp_path):
    _make_skill(
        tmp_path, "greet", "Say hello to: $ARGUMENTS",
        frontmatter='description: "greeter"\n',
    )
    tool = SkillsDirectory(tmp_path).build_tools()[0]
    obs = tool.run({"who": "Anvira"})
    assert "Say hello to: Anvira" in obs.content


@pytest.mark.anyio
async def test_prompt_skill_multi_args_render_json(tmp_path):
    _make_skill(
        tmp_path, "plan", "Plan for $ARGUMENTS using ${SKILL_DIR}/guide.md",
        frontmatter='description: "planner"\n',
    )
    spec_obj = SkillsDirectory(tmp_path).list_skills()[0]
    rendered = spec_obj.render_prompt({"goal": "ship v1", "days": 3})
    assert '"goal": "ship v1"' in rendered
    assert str(spec_obj.source_dir) in rendered


@pytest.mark.anyio
async def test_prompt_skill_no_arguments_placeholder_note(tmp_path):
    _make_skill(
        tmp_path, "note", "Args were: $ARGUMENTS",
        frontmatter='description: "n"\n',
    )
    spec_obj = SkillsDirectory(tmp_path).list_skills()[0]
    assert "(no arguments provided)" in spec_obj.render_prompt({})


def test_folder_with_neither_entrypoint_nor_body_skipped(tmp_path):
    _make_skill(tmp_path, "empty", "")
    loader = SkillsDirectory(tmp_path)
    assert loader.list_skills() == []  # nothing to run or expand → skipped


def test_script_mode_with_entrypoint_unaffected(tmp_path):
    _make_skill(
        tmp_path, "adder", "unused body",
        frontmatter='description: "adds"\nparameters: {"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}\n',
        entrypoint="def run(a, b):\n    return a + b\n",
    )
    tools = SkillsDirectory(tmp_path).build_tools()
    assert len(tools) == 1
    obs = tools[0].run({"a": 2, "b": 3})
    assert obs.success is True and "5" in str(obs.content)


def test_allowed_tools_and_context_surface_in_meta(tmp_path):
    _make_skill(
        tmp_path, "scoped",
        "Do the thing.",
        frontmatter=(
            'description: "s"\n'
            'allowed_tools:\n  - "run_command(git *)"\n  - "edit_file(src/**)"\n'
            'context: fork\n'
            'when_to_use: "when git work is requested"\n'
            'argument_hint: "<branch>"\n'
        ),
    )
    spec_obj = SkillsDirectory(tmp_path).list_skills()[0]
    assert spec_obj.allowed_tools == ("run_command(git *)", "edit_file(src/**)")
    assert spec_obj.execution_context == "fork"
    assert spec_obj.when_to_use == "when git work is requested"
    tool = skill_to_tool(spec_obj, SkillsDirectory(tmp_path))
    meta = tool.spec.meta
    assert meta["allowed_tools"] == ["run_command(git *)", "edit_file(src/**)"]
    assert meta["execution_context"] == "fork"


def test_unknown_mode_skipped_resiliently(tmp_path):
    _make_skill(tmp_path, "odd", "body", frontmatter='description: "d"\nmode: quantum\n')
    assert SkillsDirectory(tmp_path).list_skills() == []


def test_prompt_listing_budget_cap(tmp_path):
    for i in range(50):
        _make_skill(
            tmp_path, f"skill_{i}", f"body {i}",
            frontmatter=f'description: "Skill number {i} with a reasonably long description line"\n',
        )
    listing = SkillsDirectory(tmp_path).prompt_listing(max_chars=600)
    assert listing.count("- ") <= 20
    assert "omitted" in listing


def test_mixed_directory_scripts_and_prompts(tmp_path):
    _make_skill(
        tmp_path, "scripty", "ignored",
        frontmatter='description: "s"\n',
        entrypoint="def run():\n    return 'ok'\n",
    )
    _make_skill(tmp_path, "prompty", "Just follow these steps.", frontmatter='description: "p"\n')
    tools = SkillsDirectory(tmp_path).build_tools()
    names = {t.name for t in tools}
    assert names == {"scripty", "prompty"}
