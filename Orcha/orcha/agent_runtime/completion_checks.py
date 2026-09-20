"""Deterministic "is the work really finished?" checks for small-model agent loops.

A 1.5B-3B model routinely declares victory too early: it reads a file and answers without changing anything, or renames a
symbol in one file and stops. Asking the same small model "are you sure?" is unreliable, so these checks are mechanical -
plain regexes and a bounded text scan of the workspace, no extra model calls (cheap on edge hardware). Each returns a short,
concrete instruction that the agent loop feeds back as the next user turn, in the same style as the existing
verification/fix gates.
"""
from __future__ import annotations

import os
import re
from typing import Iterable, List, Optional, Tuple

# Tools that change the workspace. (`edit_file` and the file-management tools were missing from the original write set.)
MUTATING_TOOLS = frozenset({
    "write_file", "create_file", "append_file", "replace_text", "apply_patch", "edit_file", "delete_file",
    "move_file", "rename_file", "copy_file", "create_directory", "delete_directory", "move_directory",
    "rename_directory", "copy_directory",
})
EDIT_TOOLS_HINT = "edit_file (change part of an existing file: path, old_string, new_string) or write_file (a new file: path, complete content)"

_CHANGE_VERBS = re.compile(
    r"\b(fix|repair|correct|change|modify|update|edit|create|write|add|implement|build|make|generate|rename|replace|refactor|"
    r"remove|delete|rewrite|patch|convert|migrate|move)\b", re.I)
# A request that STARTS like a question is a question ("explain how to fix X"); anything else containing a change verb is work
# ("The tests fail. Read stats.py, find the bug and fix it" reads a file on the way but is still a change request).
_QUESTION_LEAD = re.compile(r"^\W*(explain|describe|summari[sz]e|what|how|why|who|where|when|which|list|show|tell|review|analy[sz]e|"
                            r"is|are|does|do|can|could|would|should|has|have|read|look|check|find out)\b", re.I)
# The request must point at something on disk: a file/extension, or files/code words. 'write a long answer' is text, not an edit.
_TARGET_RE = re.compile(r'(\b[\w.-]+\.(?:py|js|ts|tsx|jsx|java|c|h|cpp|cs|go|rs|rb|php|md|json|ya?ml|toml|ini|cfg|html?|css|sql|sh|txt|csv)\b)|(\b(files?|folders?|director(?:y|ies)|projects?|codebase|repo|repository|code|functions?|classes|class|methods?|modules?|scripts?|tests?|bugs?|readme|config|workspace|package|component)\b)', re.I)
_RENAME = re.compile(
    r"\brename\s+(?:the\s+)?(?:(?:function|method|class|variable|constant|symbol|identifier|field|attribute|parameter|file)\s+)?"
    r"[`'\"]?([A-Za-z_][\w.]*)[`'\"]?\s+(?:to|as|into)\s+[`'\"]?([A-Za-z_][\w.]*)[`'\"]?", re.I)
_SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build", ".idea", ".vscode",
              ".mypy_cache", ".pytest_cache", "target", ".tox"}
_TEXT_SUFFIXES = {".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs", ".java", ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb",
                  ".php", ".kt", ".swift", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".html", ".css", ".scss",
                  ".sql", ".sh", ".ps1", ".bat", ".cmd", ".vue", ".svelte"}
_MAX_FILE_BYTES = 1_000_000
_MAX_FILES = 4000


def task_requires_change(task: str) -> bool:
    """True when the request is about *doing* something to files (fix/create/rename/...), not just answering."""
    text = (task or "").strip()
    if _QUESTION_LEAD.match(text):
        return False
    if parse_rename(text):                       # 'rename calc_total to compute_total' names its own target
        return True
    return bool(_CHANGE_VERBS.search(text) and _TARGET_RE.search(text))


def parse_rename(task: str) -> Optional[Tuple[str, str]]:
    """``rename calc_total to compute_total`` -> ("calc_total", "compute_total"); None if the task is not a rename."""
    m = _RENAME.search(task or "")
    if not m:
        return None
    old, new = m.group(1), m.group(2)
    if old == new or len(old) < 3 or "." in old:      # too generic / dotted paths are not a single identifier
        return None
    return old, new


def find_remaining(roots: Iterable[str], token: str, limit: int = 8) -> List[str]:
    """``file:line: text`` for whole-word occurrences of ``token`` under ``roots`` (bounded scan, text files only)."""
    pat = re.compile(r"(?<![\w])" + re.escape(token) + r"(?![\w])")
    hits: List[str] = []
    scanned = 0
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
            for fn in sorted(filenames):
                if os.path.splitext(fn)[1].lower() not in _TEXT_SUFFIXES:
                    continue
                path = os.path.join(dirpath, fn)
                scanned += 1
                if scanned > _MAX_FILES:
                    return hits
                try:
                    if os.path.getsize(path) > _MAX_FILE_BYTES:
                        continue
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        for n, line in enumerate(fh, 1):
                            if pat.search(line):
                                rel = os.path.relpath(path, root).replace(os.sep, "/")
                                hits.append(f"{rel}:{n}: {line.strip()[:100]}")
                                if len(hits) >= limit:
                                    return hits
                except OSError:
                    continue
    return hits


def nothing_changed_message() -> str:
    return ("You have NOT changed any file yet, but the task asks for a change. Do not describe the change - make it. "
            f"Your next reply must be a tool call: {EDIT_TOOLS_HINT}. Read the file first only if you still need its exact text.")


def rename_unfinished_message(old: str, new: str, hits: List[str]) -> str:
    lines = "\n".join(f"  {h}" for h in hits)
    return (f"Not finished: `{old}` still appears in the project:\n{lines}\n"
            f"Change every one of them to `{new}` (one edit_file call per file: old_string `{old}`, new_string `{new}`, "
            "replace_all true if it appears several times in that file). Only answer when none are left.")


def guess_run_command(roots: Iterable[str], last_written: str = "", python: str = "python") -> Optional[str]:
    """A sensible ``run_command`` for a model that called it with NO arguments (small models often do, even when forced).

    Deterministic and conservative: run the project's Python tests if there are any, else run the file that was just written.
    Returns None when there is nothing obviously right to run (the model then gets the normal "missing argument" error).
    """
    py = f'"{python}"' if (" " in python and not python.startswith('"')) else python
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
            if any(f.startswith("test_") and f.endswith(".py") or f.endswith("_test.py") for f in filenames):
                rel = os.path.relpath(dirpath, root)
                return f"{py} -m unittest discover" + ("" if rel == "." else f' -s "{rel}"')
            if dirpath.count(os.sep) - root.count(os.sep) >= 2:
                dirnames[:] = []
    name = (last_written or "").strip()
    if name.endswith(".py"):
        return f"{py} {name}"
    if name.endswith((".js", ".mjs")):
        return f"node {name}"
    return None
