"""
orcha.capabilities.code_intelligence
====================================
Lightweight code intelligence: symbol discovery, references, documentation,
renaming and file outlining. Uses a dependency-free regex symbol index for the
common manifest languages (Python, JS/TS, Rust/Go/C/C++/Java, JSON/YAML).
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..nodes.tool import ToolSpec
from .base import DANGEROUS_LEVEL, READ, SAFE, WRITE, CapabilityContext, ToolError, spec
from .pathing import MAX_TEXT_CHARS, IGNORED_DIRS, IGNORED_FILES, is_text, walk_files

CAPABILITY_NAME = "code_intelligence"
CAPABILITY_LABEL = "Code Intelligence"
CAPABILITY_DESCRIPTION = "Find symbols, references and definitions; rename symbols; outline files."

SYMBOL_PATTERNS = {
    ".py": re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)"),
    ".js": re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:class|function)\s+([A-Za-z_$]\w*)"),
    ".ts": re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:class|function|interface|type|enum)\s+([A-Za-z_$]\w*)"),
    ".jsx": re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:function|class)\s+([A-Za-z_$]\w*)"),
    ".tsx": re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:function|class)\s+([A-Za-z_$]\w*)"),
    ".rs": re.compile(r"^\s*(?:pub\s+)?(?:fn|struct|enum|trait|impl|mod|type)\s+([A-Za-z_]\w*)"),
    ".go": re.compile(r"^\s*(?:func|type|struct|interface|const|var)\s+([A-Za-z_]\w*)"),
    ".c": re.compile(r"^\s*(?:static\s+)?(?:int|void|char|float|double|long|short|unsigned|struct|enum|bool)\s+([A-Za-z_]\w*)\s*\("),
    ".h": re.compile(r"^\s*(?:static\s+)?(?:int|void|char|float|double|long|short|unsigned|struct|enum|bool)\s+([A-Za-z_]\w*)\s*\("),
    ".cpp": re.compile(r"^\s*(?:static\s+)?(?:int|void|char|float|double|long|short|unsigned|struct|enum|bool|auto|std::\w+)\s+([A-Za-z_]\w*)\s*\("),
    ".hpp": re.compile(r"^\s*(?:static\s+)?(?:int|void|char|float|double|long|short|unsigned|struct|enum|bool|auto|std::\w+)\s+([A-Za-z_]\w*)\s*\("),
    ".java": re.compile(r"^\s*(?:public|private|protected|static|final|abstract|synchronized|native)?\s*(?:class|interface|enum|record)\s+([A-Za-z_]\w*)"),
    ".cs": re.compile(r"^\s*(?:public|private|protected|internal|static|sealed|abstract|partial)?\s*(?:class|interface|enum|record|struct)\s+([A-Za-z_]\w*)"),
    ".json": re.compile(r'^\s*"([A-Za-z_]\w*)"\s*:'),
    ".yml": re.compile(r"^([A-Za-z_]\w*):"),
    ".yaml": re.compile(r"^([A-Za-z_]\w*):"),
}

_WORD_RE = re.compile(r"\b([A-Za-z_]\w*)\b")

# ── Symbol index cache ──────────────────────────────────────────────────────
# Avoids re-scanning the entire workspace on every symbol lookup.
# TTL = 60 seconds; invalidated on rename_symbol writes.

_SYMBOL_INDEX_CACHE: Dict[Tuple[str, ...], Tuple[float, Dict[str, List[Dict[str, Any]]]]] = {}
_SYMBOL_INDEX_TTL = 60.0


def _identify_ext(path: str) -> str:
    p = Path(path).name.lower()
    for ext, pat in SYMBOL_PATTERNS.items():
        if p.endswith(ext):
            return ext
    return ""


def _index_file(path: str) -> List[Dict[str, Any]]:
    ext = _identify_ext(path)
    if not ext:
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read(MAX_TEXT_CHARS)
    except OSError:
        return []
    pat = SYMBOL_PATTERNS[ext]
    symbols = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        m = pat.search(line)
        if m:
            symbols.append({"name": m.group(1), "kind": "definition", "line": lineno,
                            "file": path, "signature": line.strip()[:200]})
    return symbols


def _build_index(roots) -> Dict[str, List[Dict[str, Any]]]:
    """Build or retrieve a cached symbol index for the given roots.

    The cache key is the sorted tuple of roots. Entries expire after
    ``_SYMBOL_INDEX_TTL`` seconds so that file changes are picked up
    without manual invalidation.
    """
    cache_key = tuple(sorted(roots))
    now = time.monotonic()
    cached = _SYMBOL_INDEX_CACHE.get(cache_key)
    if cached is not None:
        ts, index = cached
        if now - ts < _SYMBOL_INDEX_TTL:
            return index
    index: Dict[str, List[Dict[str, Any]]] = {}
    for root in roots:
        for _rel, path in walk_files(root):
            for sym in _index_file(path):
                index.setdefault(sym["name"], []).append(sym)
    _SYMBOL_INDEX_CACHE[cache_key] = (now, index)
    return index


def _invalidate_index_cache(roots) -> None:
    """Invalidate the symbol index cache for the given roots (after writes)."""
    cache_key = tuple(sorted(roots))
    _SYMBOL_INDEX_CACHE.pop(cache_key, None)


def build_tools(ctx: CapabilityContext) -> List[ToolSpec]:
    roots = ctx.roots

    def find_symbol(name: str, path: str = "") -> Dict[str, Any]:
        if not (name or "").strip():
            raise ToolError("invalid_symbol", "name must not be empty")
        results = []
        for sym_path in ([path] if path else roots):
            if path and Path(sym_path).is_file():
                results.extend(s for s in _index_file(sym_path) if s["name"] == name)
            else:
                for _rel, sp in walk_files(sym_path):
                    results.extend(s for s in _index_file(sp) if s["name"] == name)
        return {"symbol": name, "matches": results[:50], "count": len(results)}

    def find_references(name: str, path: str = "") -> Dict[str, Any]:
        if not (name or "").strip():
            raise ToolError("invalid_symbol", "name must not be empty")
        refs = []
        for sym_path in ([path] if path else roots):
            files = [sym_path] if path and Path(sym_path).is_file() else [sp for _, sp in walk_files(sym_path)]
            for sp in files:
                try:
                    with open(sp, "r", encoding="utf-8", errors="replace") as fh:
                        text = fh.read(MAX_TEXT_CHARS)
                except OSError:
                    continue
                for lineno, line in enumerate(text.splitlines(), start=1):
                    if _WORD_RE.search(line) and re.search(rf"\b{re.escape(name)}\b", line):
                        refs.append({"file": sp, "line": lineno, "text": line.strip()[:200]})
        defs = {f"{r['file']}:{r['line']}" for r in refs}
        return {"symbol": name, "references": [r for r in refs if f"{r['file']}:{r['line']}" not in defs][:50]}

    def rename_symbol(name: str, new_name: str) -> Dict[str, Any]:
        if not (name or "").strip():
            raise ToolError("invalid_symbol", "name must not be empty")
        if not re.fullmatch(r"[A-Za-z_]\w*", new_name or ""):
            raise ToolError("invalid_symbol", f"'{new_name}' is not a valid identifier.")
        changed = 0
        files = 0
        for root in roots:
            for _rel, sp in walk_files(root):
                try:
                    with open(sp, "r", encoding="utf-8", errors="replace") as fh:
                        text = fh.read(MAX_TEXT_CHARS)
                except OSError:
                    continue
                new_text = re.sub(rf"\b{re.escape(name)}\b", new_name, text)
                if new_text != text:
                    with open(sp, "w", encoding="utf-8", newline="") as fh:
                        fh.write(new_text)
                    changed += len(re.findall(rf"\b{re.escape(name)}\b", text))
                    files += 1
        if files > 0:
            _invalidate_index_cache(roots)
        return {"symbol": name, "renamed_to": new_name, "occurrences_changed": changed, "files_changed": files}

    def document_symbol(name: str) -> Dict[str, Any]:
        index = _build_index(roots)
        syms = index.get(name, [])
        if not syms:
            raise ToolError("symbol_not_found", f"Symbol '{name}' not found in the workspace.")
        docs = []
        for s in syms[:10]:
            try:
                with open(s["file"], "r", encoding="utf-8", errors="replace") as fh:
                    lines = fh.read(MAX_TEXT_CHARS).splitlines()
            except OSError:
                lines = []
            start = max(0, s["line"] - 3)
            docs.append({"file": s["file"], "line": s["line"],
                         "context": "\n".join(lines[start:s["line"] + 3])})
        return {"symbol": name, "definitions": docs}

    def outline_file(path: str) -> Dict[str, Any]:
        abs_path = Path(path).resolve() if Path(path).is_absolute() else None
        if not path:
            raise ToolError("missing_path", "path is required")
        found = None
        for root in roots:
            for _rel, sp in walk_files(root):
                if sp == str(abs_path) or (path and Path(path).resolve() == Path(sp).resolve()):
                    found = sp
                    break
            if found:
                break
        if not found:
            raise ToolError("file_not_found", f"No file matching '{path}' in the workspace.")
        return {"path": found, "symbols": _index_file(found)}

    return [
        spec("find_symbol",
             "Find where a symbol (function, class, variable, interface, type) is DEFINED in the codebase. USE THIS WHEN: you need to locate the definition of a function/class/type before reading or modifying it. Returns file paths, line numbers, and the definition line.",
             {"type": "object", "properties": {
                 "name": {"type": "string", "description": "Symbol name to find. Example: 'processAuth', 'UserModel', 'Config'."},
                 "path": {"type": "string", "description": "Specific file or directory to search in. Default: entire workspace."}},
              "required": ["name"]},
             find_symbol, permissions=[READ], safety_level=SAFE, capability=CAPABILITY_NAME),
        spec("find_references",
             "Find all usages/references of a symbol across the codebase (excluding its definition). USE THIS WHEN: you want to know everywhere a function/class/variable is used — essential before renaming or modifying it. Returns file, line, and context for each reference.",
             {"type": "object", "properties": {
                 "name": {"type": "string", "description": "Symbol name to find references for. Example: 'processAuth'."},
                 "path": {"type": "string", "description": "Specific file or directory to search in. Default: entire workspace."}},
              "required": ["name"]},
             find_references, permissions=[READ], safety_level=SAFE, capability=CAPABILITY_NAME),
        spec("rename_symbol",
             "Rename a symbol across ALL files in the workspace. USE THIS WHEN: you want to rename a function, class, variable, or type everywhere it appears. WARNING: This modifies multiple files and requires approval. Always run find_references first to see what will change.",
             {"type": "object", "properties": {
                 "name": {"type": "string", "description": "Current symbol name. Example: 'oldFunctionName'."},
                 "new_name": {"type": "string", "description": "New symbol name. Must be a valid identifier. Example: 'newFunctionName'."}},
              "required": ["name", "new_name"]},
             rename_symbol, permissions=[WRITE, "dangerous"], safety_level=DANGEROUS_LEVEL,
             capability=CAPABILITY_NAME, requires_approval=True),
        spec("document_symbol",
             "Show a symbol's definition and surrounding context (the code around it). USE THIS WHEN: understanding what a function/class does by reading its implementation and nearby comments. Returns the definition with 3 lines of context above and below.",
             {"type": "object", "properties": {
                 "name": {"type": "string", "description": "Symbol name to document. Example: 'processAuth'."}},
              "required": ["name"]},
             document_symbol, permissions=[READ], safety_level=SAFE, capability=CAPABILITY_NAME),
        spec("outline_file",
             "Show the outline (all definitions with line numbers) of a file. USE THIS WHEN: quickly understanding what's in a file — all functions, classes, interfaces, types, and their locations. Great for getting an overview before reading the full file.",
             {"type": "object", "properties": {
                 "path": {"type": "string", "description": "File path. Example: 'src/app.ts'."}},
              "required": ["path"]},
             outline_file, permissions=[READ], safety_level=SAFE, capability=CAPABILITY_NAME),
    ]
