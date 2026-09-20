"""
orcha.capabilities.search
=========================
Semantic and text search across the workspace: grep, regex, filename,
symbol and whole-workspace search.
"""
from __future__ import annotations

import fnmatch
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from ..nodes.tool import ToolSpec
from .base import READ, SAFE, CapabilityContext, ToolError, spec
from .pathing import (
    MAX_RESULTS, MAX_TEXT_CHARS, decode, ensure_dir, is_text, read_bytes,
    read_text_stream, walk_files,
)

CAPABILITY_NAME = "search"
CAPABILITY_LABEL = "Search"
CAPABILITY_DESCRIPTION = (
    "Text, regex, filename and symbol search across workspace files."
)

_READ_PERMS = [READ]

_SYMBOL_RE = re.compile(
    r"^\s*(?:(?:export\s+)?(?:async\s+)?(?:def|class|function|interface|type|enum)\s+"
    r"|(?:export\s+)?(?:const|let|var|fn|func)\s+"
    r"|(?:public|private|protected)\s+(?:static\s+)?(?:function|def|class)\s+)"
    r"([A-Za-z_$][\w$]*)",
)

_MAX_WORKERS = 8


def _grep(roots):
    def tool(query: str, path: str = ".", glob: Optional[str] = None) -> str:
        needle = str(query or "")
        if not needle.strip():
            raise ToolError("invalid_query", "query must not be empty")
        fp = ensure_dir(roots, path)
        return _run_search(roots, fp, glob, lambda text: needle in text)
    return tool


def _regex_search(roots):
    def tool(pattern: str, path: str = ".", glob: Optional[str] = None) -> str:
        if not (pattern or "").strip():
            raise ToolError("invalid_pattern", "pattern must not be empty")
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            raise ToolError("invalid_regex", f"Invalid regular expression: {exc}")
        fp = ensure_dir(roots, path)
        return _run_search(roots, fp, glob, lambda text: bool(rx.search(text)))
    return tool


def _run_search(roots, base_abs, glob, predicate) -> str:
    results: List[str] = []

    def _scan_file(rel_full):
        rel, full = rel_full
        matches = []
        for lineno, line in read_text_stream(full, max_chars=MAX_TEXT_CHARS):
            if predicate(line):
                matches.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                if len(matches) >= MAX_RESULTS:
                    break
        return matches

    files = list(walk_files(base_abs, glob=glob))
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_scan_file, rf): rf for rf in files}
        for future in as_completed(futures):
            try:
                results.extend(future.result())
                if len(results) >= MAX_RESULTS:
                    for f in futures:
                        f.cancel()
                    break
            except Exception:
                continue
    return "\n".join(results[:MAX_RESULTS]) if results else "No matches found."


def _filename_search(roots):
    def tool(pattern: str, path: str = ".") -> Dict[str, Any]:
        if not (pattern or "").strip():
            raise ToolError("invalid_pattern", "pattern must not be empty")
        fp = ensure_dir(roots, path)
        matches: List[str] = []
        for rel, _full in walk_files(fp):
            if fnmatch.fnmatch(os.path.basename(rel), pattern):
                matches.append(rel.replace("\\", "/"))
                if len(matches) >= MAX_RESULTS:
                    break
        return {"pattern": pattern, "matches": matches, "count": len(matches)}
    return tool


def _symbol_search(roots):
    def tool(name: str, path: str = ".", glob: Optional[str] = None) -> Dict[str, Any]:
        needle = str(name or "").strip()
        if not needle:
            raise ToolError("invalid_symbol", "name must not be empty")
        fp = ensure_dir(roots, path)
        found = []

        def _scan_file(rel_full):
            rel, full = rel_full
            matches = []
            for lineno, line in read_text_stream(full, max_chars=MAX_TEXT_CHARS):
                if re.search(r"\b" + re.escape(needle) + r"\b", line):
                    m = _SYMBOL_RE.match(line)
                    kind = "definition" if m and m.group(1) == needle else "symbol"
                    matches.append({"file": rel, "line": lineno, "kind": kind, "text": line.strip()[:160]})
                    if len(matches) >= MAX_RESULTS:
                        break
            return matches

        files = list(walk_files(fp, glob=glob))
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {pool.submit(_scan_file, rf): rf for rf in files}
            for future in as_completed(futures):
                try:
                    found.extend(future.result())
                    if len(found) >= MAX_RESULTS:
                        for f in futures:
                            f.cancel()
                        break
                except Exception:
                    continue
        return {"symbol": needle, "matches": found[:MAX_RESULTS], "count": len(found)}
    return tool


def _workspace_search(roots):
    def tool(query: str, glob: Optional[str] = None) -> str:
        needle = str(query or "")
        if not needle.strip():
            raise ToolError("invalid_query", "query must not be empty")
        if not roots:
            raise ToolError("no_workspace", "No workspace is attached.")
        results: List[str] = []
        for root in roots:
            if not os.path.isdir(root):
                continue
            chunk = _run_search(roots, root, glob, lambda text: needle in text)
            if chunk != "No matches found.":
                results.append(f"[{os.path.basename(root.rstrip('/\\'))}]\n{chunk}")
        return "\n\n".join(results) if results else "No matches found."
    return tool


def build_tools(ctx: CapabilityContext) -> List[ToolSpec]:
    roots = ctx.roots
    return [
        spec("grep",
             "Case-sensitive literal text search across workspace files. USE THIS WHEN: finding where a specific string (function name, variable, error message, comment) appears in the codebase. Returns matching lines with file:line: content format. Use search_text for case-insensitive search.",
             {"type": "object", "properties": {
                 "query": {"type": "string", "description": "Exact case-sensitive text to find. Examples: 'useState', 'def process_data', 'TODO: fix'."},
                 "path": {"type": "string", "description": "Directory to search in. Default: '.' (entire workspace). Use 'src' to narrow."},
                 "glob": {"type": "string", "description": "Filter by filename. Examples: '*.ts', '*.py', '*.{ts,tsx}'."}},
              "required": ["query"]},
             _grep(roots), permissions=_READ_PERMS, safety_level=SAFE, capability=CAPABILITY_NAME),
        spec("regex_search",
             "Search workspace files using a regular expression pattern. USE THIS WHEN: literal search is too rigid and you need pattern matching, e.g. finding all function definitions, all imports from a module, or matching variable naming patterns.",
             {"type": "object", "properties": {
                 "pattern": {"type": "string", "description": "Regex pattern. Examples: 'def \\w+\\(' (Python function defs), 'import.*from.*react' (React imports), '\\bclass\\s+\\w+' (class definitions)."},
                 "path": {"type": "string", "description": "Directory to search. Default: '.' (entire workspace)."},
                 "glob": {"type": "string", "description": "Filter by filename pattern. Examples: '*.py', '*.ts'."}},
              "required": ["pattern"]},
             _regex_search(roots), permissions=_READ_PERMS, safety_level=SAFE, capability=CAPABILITY_NAME),
        spec("filename_search",
             "Find files whose NAME matches a glob pattern. USE THIS WHEN: you know the filename or pattern but not where it is, e.g. find all test files, config files, or a specific file like 'jest.config.js'. Returns matching file paths.",
             {"type": "object", "properties": {
                 "pattern": {"type": "string", "description": "Filename glob pattern. Examples: '*.test.ts' (all TypeScript test files), 'jest.config.*' (Jest config), '*.py' (all Python files), 'setup.*'."},
                 "path": {"type": "string", "description": "Directory to search in. Default: '.' (entire workspace)."}},
              "required": ["pattern"]},
             _filename_search(roots), permissions=_READ_PERMS, safety_level=SAFE, capability=CAPABILITY_NAME,
             result_format={"type": "object", "properties": {"matches": {"type": "array"}, "count": {"type": "integer"}}}),
        spec("symbol_search",
             "Find a symbol (function, class, variable, etc.) across workspace files using word-boundary matching. USE THIS WHEN: you want to find where a specific identifier is defined or referenced. Better than grep for code symbols because it uses word boundaries to avoid partial matches.",
             {"type": "object", "properties": {
                 "name": {"type": "string", "description": "Symbol name to search for. Examples: 'processData', 'MyComponent', 'config'."},
                 "path": {"type": "string", "description": "Directory to search. Default: '.' (entire workspace)."},
                 "glob": {"type": "string", "description": "Filter by filename pattern."}},
              "required": ["name"]},
             _symbol_search(roots), permissions=_READ_PERMS, safety_level=SAFE, capability=CAPABILITY_NAME,
             result_format={"type": "object", "properties": {"matches": {"type": "array"}, "count": {"type": "integer"}}}),
        spec("workspace_search",
             "Search ALL attached workspace roots (all open projects) for a string. USE THIS WHEN: you don't know which project contains something, or you need to search across multiple projects at once. Broader than grep — searches everything.",
             {"type": "object", "properties": {
                 "query": {"type": "string", "description": "Text to search for (case-sensitive). Example: 'handleAuth'."},
                 "glob": {"type": "string", "description": "Filter by filename pattern. Example: '*.ts'."}},
              "required": ["query"]},
             _workspace_search(roots), permissions=_READ_PERMS, safety_level=SAFE, capability=CAPABILITY_NAME),
    ]
