"""
orcha.capabilities.tool_aliases
================================
Tool-NAME resolution for the exact same failure mode
``_ARG_NAME_ALIASES``/``_normalize_arg_aliases`` in ``base.py`` already
fixes for argument names: a model calling a real capability by a
plausible-but-wrong name (``write_tool``, ``writ_tools``, ``save_file``
instead of the real ``write_file``) used to just fail with "unknown tool"
and burn a whole extra round-trip waiting for the model to notice and
retry — or, for a weaker model, never notice at all and give up.

Two layers, applied in order by ``resolve_tool_name``:

1. ``TOOL_NAME_ALIASES`` — a curated map of the specific wrong names a
   model plausibly reaches for per real tool (synonyms, missing/extra
   underscores, singular vs. plural, a stray "_tool"/"_tools" suffix).
2. A generic normalize-and-fuzzy-match fallback, so a call this list
   didn't anticipate — a typo, a new capability module added later — still
   has a real shot at resolving instead of only ever working for names
   someone thought to hand-enumerate.
"""
from __future__ import annotations

import difflib
import logging
import re
from typing import Dict, List, Optional, Tuple

# Real tool name -> known wrong-but-plausible names a model might call
# instead. Keys are the actual registered ToolSpec names (see
# orcha/capabilities/{filesystem,workspace,search,terminal,git,diagnostics,
# code_intelligence}.py) — this file is a maintained reference to those,
# not a second source of truth: resolve_tool_name always validates against
# the caller's live registered tool set before returning a match.
TOOL_NAME_ALIASES: Dict[str, Tuple[str, ...]] = {
    # ── filesystem ──────────────────────────────────────────────────────
    "read_file": ("read_tool", "get_file", "open_file", "load_file", "view_file", "cat_file", "file_read"),
    "write_file": ("write_tool", "writ_tools", "write_tools", "save_file", "create_file_content", "put_file", "file_write", "save_tool"),
    "apply_patch": ("patch", "apply_diff", "multi_edit", "multiedit", "patch_files"),
    "edit_file": ("edit_tool", "modify_file", "update_file", "patch_file", "change_file", "str_replace", "str_replace_editor"),
    "append_file": ("append_tool", "add_to_file", "file_append"),
    "create_file": ("create_tool", "new_file", "touch_file", "make_file"),
    "delete_file": ("delete_tool", "remove_file", "rm_file", "unlink_file", "file_delete"),
    "rename_file": ("rename_tool", "move_file_rename", "file_rename"),
    "copy_file": ("copy_tool", "cp_file", "duplicate_file", "file_copy"),
    "move_file": ("move_tool", "mv_file", "relocate_file", "file_move"),
    "create_directory": ("create_dir", "make_directory", "mkdir", "new_directory", "new_folder", "create_folder"),
    "delete_directory": ("delete_dir", "remove_directory", "rmdir", "delete_folder", "remove_folder"),
    "rename_directory": ("rename_dir", "rename_folder"),
    "copy_directory": ("copy_dir", "duplicate_directory", "copy_folder"),
    "move_directory": ("move_dir", "relocate_directory", "move_folder"),
    "list_directory": ("list_dir", "ls", "listdir", "list_files", "get_directory_listing", "dir_list", "list_folder"),
    "directory_tree": ("dir_tree", "tree", "get_tree", "folder_tree", "file_tree"),
    "exists": ("file_exists", "check_exists", "path_exists", "does_exist"),
    "file_info": ("get_file_info", "stat_file", "file_stat", "file_metadata"),
    "glob_search": ("glob", "find_files", "glob_files", "file_glob"),
    "search_text": ("text_search", "search_in_file", "find_text"),
    "replace_text": ("text_replace", "find_and_replace", "search_and_replace"),
    "read_multiple_files": ("read_files", "batch_read", "multi_file_read"),
    # ── workspace ────────────────────────────────────────────────────────
    "current_workspace": ("get_workspace", "workspace_info", "active_workspace"),
    "workspace_tree": ("get_workspace_tree", "project_tree"),
    "list_projects": ("get_projects", "projects_list"),
    "attached_files": ("get_attached_files", "list_attached_files"),
    "attached_folders": ("get_attached_folders", "list_attached_folders"),
    "active_file": ("get_active_file", "current_file"),
    "recent_files": ("get_recent_files", "recently_used_files"),
    "project_summary": ("get_project_summary", "summarize_project"),
    # ── search ───────────────────────────────────────────────────────────
    "grep": ("grep_search", "grep_files"),
    "regex_search": ("regexp_search", "pattern_search"),
    "filename_search": ("find_filename", "search_filename", "file_name_search"),
    "symbol_search": ("search_symbol", "find_symbol_search"),
    "workspace_search": ("search_workspace", "project_search"),
    # ── web ──────────────────────────────────────────────────────────────
    "web_search": ("search_web", "internet_search", "google_search", "search_internet",
                   "browse_web", "search_online", "online_search", "web_query"),
    "web_fetch": ("fetch_url", "fetch_web", "get_url", "read_url", "scrape_url",
                  "scrape_page", "fetch_page", "get_webpage", "browse_url", "load_url",
                  "open_url", "read_webpage", "url_fetch"),
    # ── terminal ─────────────────────────────────────────────────────────
    "run_command": ("execute_command", "run_terminal_command", "shell_exec", "exec_command", "run_shell", "terminal_run", "bash"),
    "stream_output": ("get_output", "read_output", "stream_command_output"),
    "stop_process": ("kill_process", "terminate_process", "cancel_process"),
    "running_processes": ("list_processes", "get_running_processes", "ps"),
    "command_history": ("get_command_history", "history"),
    # ── git ──────────────────────────────────────────────────────────────
    "git_status": ("git_stat",),
    "git_diff": ("git_diff_tool",),
    "git_add": ("git_stage",),
    "git_commit": ("git_commit_tool",),
    "git_branch": ("git_branches", "list_branches"),
    "git_checkout": ("git_switch",),
    "git_log": ("git_history",),
    "git_show": ("git_show_commit",),
    "git_restore": ("git_revert_file",),
    "git_pull": ("git_fetch_pull",),
    "git_push": ("git_push_tool",),
    # ── diagnostics ──────────────────────────────────────────────────────
    "run_build": ("build", "run_build_tool"),
    "run_tests": ("test", "run_test", "run_test_suite"),
    "run_linter": ("lint", "run_lint"),
    "type_check": ("typecheck", "run_type_check"),
    "dependency_check": ("check_dependencies", "deps_check"),
    "read_logs": ("get_logs", "logs"),
    # ── code_intelligence ────────────────────────────────────────────────
    "find_symbol": ("locate_symbol", "search_for_symbol"),
    "find_references": ("get_references", "find_usages"),
    "rename_symbol": ("rename_symbol_tool", "refactor_rename"),
    "document_symbol": ("get_document_symbols", "document_symbols"),
    "outline_file": ("get_file_outline", "file_outline"),
}

# Reverse index (alias -> canonical), built once at import time.
_ALIAS_TO_REAL: Dict[str, str] = {
    alias: real for real, aliases in TOOL_NAME_ALIASES.items() for alias in aliases
}

# Every resolution outcome is logged (not just failures): a name saved by
# the fuzzy/normalized fallback still succeeded, but a human reviewing
# logs should be able to spot "this keeps resolving via the low-confidence
# fuzzy layer" and promote it into TOOL_NAME_ALIASES for a guaranteed
# match next time, rather than relying on the fallback layers forever.
# This is the "see what the mistake was" half of growing the alias table
# from real observed model behavior instead of only hand-guessed entries.
logger = logging.getLogger("orcha.capabilities.tool_aliases")

# Words a model tacks on/leaves off around an otherwise-correct guess —
# stripped during normalization so e.g. "write_file_tool" and "writefile"
# both collapse to the same comparable key as "write_file". Widened beyond
# the original small set with more verbs/fillers a model plausibly wraps
# around a real tool name ("do_write_file", "invoke_read_file",
# "perform_git_commit") — but deliberately EXCLUDES words that are
# themselves real tool-name components in this codebase (run_command,
# run_build, run_tests, run_linter): stripping "run"/"command"/"execute"
# as noise emptied BOTH the guess's and the real tool's own normalized
# form down to nothing, so they matched by accident — or worse, collided
# with every OTHER tool that also emptied out the same way. Confirmed
# live: adding "run"/"command"/"exec" broke resolving "perform_run_command"
# to the real "run_command" tool entirely. Every word kept here was
# checked against the real tool-name list first.
_NOISE_WORDS = (
    "tool", "tools", "function", "fn", "action", "call", "do", "perform",
    "invoke", "handler", "handle", "method", "op", "operation", "util",
    "utility", "helper", "api",
)

# A boundary before an uppercase letter that follows a lowercase letter or
# digit — the camelCase/PascalCase equivalent of the underscore split
# below. Without this, "writeFile" or "WriteFile" (a model reaching for a
# JS-style naming convention instead of this codebase's snake_case) stays
# as one fused token "writefile" and never matches "write_file" via the
# normalized-token layers, falling through to the fuzzy-match last resort
# instead of resolving cleanly.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokenize(name: str) -> List[str]:
    """Split camelCase/PascalCase boundaries, then on any non-alphanumeric
    run, lowercase, drop noise words and trailing plural 's'."""
    spaced = _CAMEL_BOUNDARY.sub("_", name)
    tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", spaced.lower()) if t]
    tokens = [t[:-1] if t.endswith("s") and len(t) > 3 else t for t in tokens]
    return [t for t in tokens if t not in _NOISE_WORDS]


def _normalize(name: str) -> str:
    """Cleaned tokens rejoined in their ORIGINAL order — the same key two
    spellings of "the same tool" collapse to, e.g. "write_file_tool" and
    "writefile" both become "write_file"."""
    return "_".join(_tokenize(name))


def _normalize_sorted(name: str) -> str:
    """Same cleaned tokens, but SORTED — so word-ORDER stops mattering too:
    "file_write" and "write_file" both become "file_write". This is the
    layer that makes an alias curated as "write_file": (..., "file_write")
    unnecessary for every other tool too — any permutation of a real
    tool's own words resolves without having to be hand-enumerated."""
    return "_".join(sorted(_tokenize(name)))


def resolve_tool_name(name: str, real_names: "set[str]") -> Optional[str]:
    """
    Resolve a model-supplied tool name to a real, registered one.

    Tries, in order: exact match (already real — nothing to do here, but
    checked so callers can use the return value unconditionally), the
    curated alias table, a normalized-token match against every real name,
    and finally a fuzzy closest-match as a last resort for a plain typo.
    Returns None only when nothing clears a reasonable confidence bar —
    callers should still surface the original "unknown tool" error in that
    case rather than guess.
    """
    if name in real_names:
        return name

    aliased = _ALIAS_TO_REAL.get(name)
    if aliased in real_names:
        return aliased

    target = _normalize(name)
    if target:
        normalized_map = {_normalize(real): real for real in real_names}
        if target in normalized_map:
            resolved = normalized_map[target]
            logger.info("tool name '%s' resolved to '%s' via normalized-token match", name, resolved)
            return resolved

    # Word-order-independent match: a model's guess whose words are all
    # correct but reordered ("file_write" for "write_file", "directory_list"
    # for "list_directory") resolves here without needing to be curated as
    # an explicit alias for every real tool. Guarded against ambiguity: only
    # trusted when exactly one real tool's own words sort to the same key —
    # two real tools legitimately sharing a whole word set would otherwise
    # resolve to whichever happened to be inserted last.
    sorted_target = _normalize_sorted(name)
    if sorted_target:
        sorted_buckets: Dict[str, List[str]] = {}
        for real in real_names:
            sorted_buckets.setdefault(_normalize_sorted(real), []).append(real)
        bucket = sorted_buckets.get(sorted_target)
        if bucket and len(bucket) == 1:
            logger.info("tool name '%s' resolved to '%s' via word-order-independent match", name, bucket[0])
            return bucket[0]

    # Last resort: fuzzy match against real names AND their own aliases —
    # a genuine typo ("write_fiel") should still resolve, and matching
    # against the alias list too catches a typo'd alias, not just a
    # typo'd real name. difflib's ratio is generous with shared substrings
    # regardless of overall length ("does_not_exist" vs "exists" scores
    # high on overlap alone) — confirmed live as a false positive that
    # broke test_unknown_tool_error, so a length-ratio guard runs first to
    # rule out comparisons like that before trusting the score at all.
    candidates = [
        c for c in list(real_names) + list(_ALIAS_TO_REAL.keys())
        if min(len(c), len(name)) / max(len(c), len(name)) >= 0.6
    ]
    close = difflib.get_close_matches(name, candidates, n=1, cutoff=0.88)
    if close:
        match = close[0]
        resolved = _ALIAS_TO_REAL.get(match, match)
        if resolved in real_names:
            logger.info("tool name '%s' resolved to '%s' via fuzzy match (low confidence — consider curating this into TOOL_NAME_ALIASES)", name, resolved)
            return resolved

    # A genuine miss: nothing resolved it. Logged at WARNING (not silently
    # swallowed) so real model tool-naming mistakes are visible in server
    # logs and reviewable — the concrete "see what the mistake was" input
    # for growing TOOL_NAME_ALIASES from actually-observed behavior rather
    # than only hand-guessed entries.
    logger.warning("tool name '%s' could not be resolved against any of %d registered tools", name, len(real_names))
    return None
