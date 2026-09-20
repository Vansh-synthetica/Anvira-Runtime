"""Small-model completion gates, driven through the real agent graph with a scripted fake model.

Each scenario is a failure a real Qwen2.5-Coder-3B showed on a real task (see docs/REAL_WORLD_RESULTS.md):
  * reads a file and "finishes" without changing anything;
  * renames a symbol in one file and stops;
  * calls a tool by a plausible-but-wrong name;
  * calls run_command with NO arguments;
and the checks that stop it without an extra model call.
"""
import asyncio

import pytest

from orcha.agent_runtime import completion_checks as cc
from orcha.builders.agent import AgentGraphConfig, build_agent_graph
from orcha.capabilities.base import CapabilityContext
from orcha.capabilities.registry import CapabilityRegistry
from orcha.graph.runtime import GraphRuntime
from orcha.graph.store import MemoryStore
from orcha.nodes.agent import AgentConfig


def _executor(root, caps=("filesystem", "terminal")):
    return CapabilityRegistry().register_defaults().build(list(caps), ctx=CapabilityContext(roots=[str(root)]))


def _call(name, arguments, call_id=None):
    return {"role": "assistant", "content": None, "finish_reason": "stop",
            "tool_calls": [{"id": call_id or f"call_{name}", "type": "function", "function": {"name": name, "arguments": arguments}}]}


def _say(text):
    return {"role": "assistant", "content": text, "finish_reason": "stop"}


def _run_agent(root, task, script, caps=("filesystem", "terminal"), max_iterations=14):
    """script: list of assistant messages (or callables(messages, system) -> message), consumed one per model call."""
    seen_systems, queue = [], list(script)

    async def fake(messages, system, tool_schemas):
        seen_systems.append(system or "")
        step = queue.pop(0) if queue else _say("done")
        return step(messages, system) if callable(step) else step
    executor = _executor(root, caps)
    graph = build_agent_graph(AgentGraphConfig(
        agent_config=AgentConfig(completion_fn=fake, executor=executor, max_iterations=max_iterations, workspace_roots=[str(root)]),
        executor=executor))
    result = asyncio.run(GraphRuntime(graph, store=MemoryStore()).run(task))
    return result.packet.payload, seen_systems


# ----------------------------------------------------------------------------- classifiers
@pytest.mark.parametrize("task,expected", [
    ("The tests are failing. Read stats.py, find the bug, and fix it.", True),
    ("Create a file slugify.py with a function", True),
    ("Rename calc_total to compute_total everywhere", True),
    ("Add a test for slugify", True),
    ("Explain how to fix a memory leak", False),
    ("What does stats.py do?", False),
    ("List the files in this folder", False),
    ("Read config.json and tell me the port", False),
    ("how do I rename a file", False),
    ("", False),
])
def test_task_requires_change(task, expected):
    assert cc.task_requires_change(task) is expected


def test_parse_rename_and_remaining_scan(tmp_path):
    assert cc.parse_rename("Rename the function calc_total to compute_total everywhere") == ("calc_total", "compute_total")
    assert cc.parse_rename("rename `oldName` as `newName`") == ("oldName", "newName")
    assert cc.parse_rename("Rename the file a.txt to b.txt") is None          # dotted paths are not identifiers
    assert cc.parse_rename("rename x to y") is None                           # too short to be safe
    assert cc.parse_rename("Fix the bug") is None
    (tmp_path / "a.py").write_text("def calc_total():\n    pass\n\nx = calc_total_extra\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("from a import calc_total\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "skip.js").write_text("calc_total", encoding="utf-8")
    (tmp_path / "data.bin").write_bytes(b"calc_total\x00\x01")
    hits = cc.find_remaining([str(tmp_path)], "calc_total")
    assert sorted(h.split(":")[0] for h in hits) == ["a.py", "sub/b.py"]        # whole word only; skips node_modules and binaries
    assert cc.find_remaining([str(tmp_path / "missing")], "calc_total") == []


def test_guess_run_command(tmp_path):
    assert cc.guess_run_command([str(tmp_path)], "") is None
    assert cc.guess_run_command([str(tmp_path)], "app.py") == "python app.py"
    assert cc.guess_run_command([str(tmp_path)], "index.js") == "node index.js"
    (tmp_path / "test_x.py").write_text("", encoding="utf-8")
    assert cc.guess_run_command([str(tmp_path)], "app.py") == "python -m unittest discover"


# ------------------------------------------------------------------------------- the gates
def test_reading_then_answering_is_not_finishing_a_fix(tmp_path):
    (tmp_path / "stats.py").write_text("def mean(v):\n    return sum(v) / (len(v) + 1)\n", encoding="utf-8")
    forced = []

    def edits_after_nudge(messages, system):
        forced.append("FORCE" in (system or "").upper() or True)          # the loop asked for a tool call this turn
        return _call("edit_file", {"path": "stats.py", "old_string": "(len(v) + 1)", "new_string": "len(v)"}, "c2")
    p, _ = _run_agent(tmp_path, "The tests fail. Read stats.py, find the bug and fix it in stats.py.",
                      [_call("read_file", {"path": "stats.py"}, "c1"),
                       _say("The bug is that mean divides by len+1."),            # premature: nothing changed
                       edits_after_nudge, _say("Fixed the divisor.")], caps=("filesystem",))
    assert "len(v)" in (tmp_path / "stats.py").read_text(encoding="utf-8") and "+ 1" not in (tmp_path / "stats.py").read_text(encoding="utf-8")
    notes = [s.get("note", "") for s in p["agent_steps"]]
    assert any("nothing was changed" in n for n in notes)
    assert p["agent_completed"] is True


def test_a_question_is_still_answered_without_edits(tmp_path):
    (tmp_path / "stats.py").write_text("def mean(v):\n    return sum(v) / len(v)\n", encoding="utf-8")
    p, _ = _run_agent(tmp_path, "What does stats.py do?", [_call("read_file", {"path": "stats.py"}), _say("It computes the mean.")],
                      caps=("filesystem",))
    assert p["agent_completed"] and not any("completion-gate" in s.get("note", "") for s in p["agent_steps"])


def test_rename_is_not_finished_until_every_file_is_changed(tmp_path):
    (tmp_path / "billing.py").write_text("def calc_total(i):\n    return len(i)\n", encoding="utf-8")
    (tmp_path / "report.py").write_text("from billing import calc_total\nprint(calc_total([1]))\n", encoding="utf-8")
    p, _ = _run_agent(tmp_path, "Rename the function calc_total to compute_total everywhere in this project.",
                      [_call("edit_file", {"path": "billing.py", "old_string": "calc_total", "new_string": "compute_total"}, "c1"),
                       _say("Renamed it."),                                                                  # stops after ONE file
                       _call("edit_file", {"path": "report.py", "old_string": "calc_total", "new_string": "compute_total",
                                           "replace_all": True}, "c2"),
                       _say("Renamed everywhere.")], caps=("filesystem",))
    assert "calc_total" not in (tmp_path / "billing.py").read_text(encoding="utf-8") + (tmp_path / "report.py").read_text(encoding="utf-8")
    assert any("rename unfinished" in s.get("note", "") for s in p["agent_steps"]) and p["agent_completed"]


def test_gates_are_bounded_a_stubborn_model_cannot_loop_forever(tmp_path):
    (tmp_path / "a.py").write_text("def calc_total():\n    pass\n", encoding="utf-8")
    p, _ = _run_agent(tmp_path, "Rename calc_total to compute_total.", [_say("Done.")] * 30, caps=("filesystem",), max_iterations=20)
    gate_notes = [s.get("note", "") for s in p["agent_steps"] if "completion-gate" in s.get("note", "")]
    assert 1 <= len(gate_notes) <= 5                                 # at most 2 (nothing changed) + 3 (rename), then it lets go
    assert p["agent_completed"] is True                              # and the run still ends with an answer


def test_a_wrong_tool_name_is_resolved_through_the_alias_table(tmp_path):
    (tmp_path / "n.txt").write_text("hello world", encoding="utf-8")
    p, _ = _run_agent(tmp_path, "Change hello to goodbye in n.txt",
                      [_call("modify_file", {"path": "n.txt", "old_string": "hello", "new_string": "goodbye"}), _say("done")],
                      caps=("filesystem",))
    assert (tmp_path / "n.txt").read_text(encoding="utf-8") == "goodbye world"
    call = p["agent_tool_calls"][0]
    assert call["name"] == "edit_file" and call["requested_name"] == "modify_file" and call["result_type"] == "ok"


def test_run_command_with_no_arguments_runs_the_project_tests_and_is_not_a_failed_command(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tmp_path / "test_calc.py").write_text("import unittest\nfrom calc import add\n\n\nclass T(unittest.TestCase):\n"
                                           "    def test(self):\n        self.assertEqual(add(1, 2), 3)\n", encoding="utf-8")
    p, _ = _run_agent(tmp_path, "Create calc.py with an add function",
                      [_call("write_file", {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"}, "c1"),
                       _call("run_command", {}, "c2"),                                # a small model's empty call
                       _say("Created calc.py and the tests pass.")], caps=("filesystem", "terminal"))
    cmd = next(c for c in p["agent_tool_calls"] if c["name"] == "run_command")
    assert cmd["arguments"]["command"].endswith("-m unittest discover") and cmd["result_type"] == "ok"
    assert "no command was given" in cmd["note"]
    assert not any("fix-gate" in s.get("note", "") for s in p["agent_steps"])
    assert p["agent_completed"] is True


def test_a_validation_error_is_not_mistaken_for_a_failing_command(tmp_path):
    """run_command with empty args and nothing to guess: the tool says what is missing; ORCHA must not order a code rewrite."""
    p, _ = _run_agent(tmp_path, "Create notes.txt containing hi",
                      [_call("write_file", {"path": "notes.txt", "content": "hi"}, "c1"), _call("run_command", {}, "c2"),
                       _say("done"), _say("done")], caps=("filesystem", "terminal"))
    assert not any("fix-gate" in s.get("note", "") for s in p["agent_steps"])
