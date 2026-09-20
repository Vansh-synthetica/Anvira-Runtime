"""
Regression tests for tool-call event enrichment.

Before this fix, `tool_started`/`tool_completed` events carried only a tool
name and a truncated string preview of its arguments — never a real
affected-file list or a structured args dict the UI could render as an
actual tool-call card. That's what let a failed/incomplete tool call (e.g.
"copy these files" where the copy silently failed) get reported to the user
as an unqualified success: there was no way to see what the tool was
actually asked to do versus what the model claimed afterwards.
"""
from orcha.nodes.task_executor import _extract_affected_files, _sanitize_tool_args_for_event


def test_extract_affected_files_finds_known_path_keys():
    assert _extract_affected_files({"path": "a.txt"}) == ["a.txt"]
    assert _extract_affected_files({"src": "a.pdf", "dst": "b/a.pdf"}) == ["a.pdf", "b/a.pdf"]


def test_extract_affected_files_ignores_unrelated_args():
    assert _extract_affected_files({"command": "npm test"}) == []
    assert _extract_affected_files({}) == []


def test_extract_affected_files_handles_non_dict_input():
    assert _extract_affected_files(None) == []  # type: ignore[arg-type]
    assert _extract_affected_files("not a dict") == []  # type: ignore[arg-type]


def test_sanitize_tool_args_passes_small_args_through_unchanged():
    args = {"path": "a.txt", "content": "hello"}
    assert _sanitize_tool_args_for_event(args) == args


def test_sanitize_tool_args_caps_large_content_fields():
    big = "x" * 3000
    sanitized = _sanitize_tool_args_for_event({"path": "a.txt", "content": big})
    assert len(sanitized["content"]) < len(big)
    assert sanitized["content"].startswith("x" * 2000)
    assert "more chars" in sanitized["content"]
    assert sanitized["path"] == "a.txt"
