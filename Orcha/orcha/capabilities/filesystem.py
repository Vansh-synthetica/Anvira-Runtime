"""
orcha.capabilities.filesystem
=============================
Production-grade filesystem tools for the workspace.

Every tool validates its paths against the workspace roots (see
:mod:`orcha.capabilities.pathing`), supports UTF-8, preserves line endings,
and returns structured errors via :class:`orcha.capabilities.base.ToolError`.
"""
from __future__ import annotations

import fnmatch
import difflib
import os
import shutil
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..nodes.tool import ToolSpec
from .base import (
    CAUTIOUS, DANGEROUS_LEVEL, READ, SAFE, WRITE,
    CapabilityContext, ToolError, spec,
)
from . import editing
from .pathing import (
    IGNORED_DIRS, MAX_RESULTS, MAX_TEXT_CHARS, decode, ensure_dir, ensure_file,
    is_pdf, is_text, match_ignored, normalize_path, read_bytes, read_pdf_text,
    resolve_pair, resolve_target, walk_files,
)
from .readstate import ReadStateCache

CAPABILITY_NAME = "filesystem"
CAPABILITY_LABEL = "Filesystem"
CAPABILITY_DESCRIPTION = (
    "Create, read, edit, delete, move, copy and inspect files and "
    "directories inside the workspace."
)

_READ_PERMS = [READ]
_WRITE_PERMS = [WRITE]
_MUTATE_PERMS = [WRITE, "dangerous"]
_MAX_DIFF_TEXT_CHARS = 128_000


class _FileChangeResult(str):
    """Existing string tool result plus a structured, execution-only change record."""

    def __new__(cls, message: str, file_changes: List[Dict[str, Any]]):
        value = super().__new__(cls, message)
        value.file_changes = file_changes
        return value


def _text_before(path: str) -> Optional[str]:
    """Return a bounded text snapshot for an on-disk change record."""
    if not os.path.isfile(path):
        return ""
    data = read_bytes(path)
    if not is_text(data):
        return None
    text = decode(data)
    return text if len(text) <= _MAX_DIFF_TEXT_CHARS else None


def _file_change(path: str, operation: str, before: Optional[str], after: Optional[str]) -> Dict[str, Any]:
    """Build an honest change record from actual pre/post file contents."""
    change: Dict[str, Any] = {
        "path": path,
        "operation": operation,
        "before": before,
        "after": after,
        "diff": None,
    }
    if before is not None and after is not None:
        change["diff"] = "".join(difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="\n",
        ))
    return change


# ── Implementations (closures over the workspace roots) ──────────────────────
#
# When a ReadStateCache is attached to the CapabilityContext, read tools
# record what the agent saw and mutating tools refuse blind/stale writes
# (see orcha.capabilities.readstate). With no cache attached the tools
# behave exactly as before.

def _read_file(roots, rs: Optional[ReadStateCache] = None):
    def tool(path: str, offset: int = 0, limit: Optional[int] = None) -> str:
        fp = ensure_file(roots, path)
        partial = bool(int(offset or 0)) or limit is not None
        # PDF extraction path
        if is_pdf(fp):
            text = read_pdf_text(fp)
            if rs is not None:
                # PDF text is an extraction, not file content — track as
                # seen-but-unverifiable so later edits still require intent.
                rs.record_read(fp, full_text=None, partial=False)
            lines = text.split("\n")
            start = max(0, int(offset or 0))
            end = len(lines) if limit is None else min(len(lines), start + int(limit))
            out = "\n".join(lines[start:end])
            return out if out.strip() else "(PDF contains no extractable text)"
        data = read_bytes(fp)
        if not is_text(data):
            if rs is not None:
                rs.record_read(fp, full_text=None, partial=False)
            return "[Binary file — contents not readable]"
        full_text = decode(data)
        if rs is not None:
            rs.record_read(
                fp,
                full_text=None if partial else full_text,
                partial=partial,
            )
        lines = full_text.split("\n")
        start = max(0, int(offset or 0))
        end = len(lines) if limit is None else min(len(lines), start + int(limit))
        out = "\n".join(lines[start:end])
        return out if out.strip() else "(empty file)"
    return tool


def _write_file(roots, rs: Optional[ReadStateCache] = None):
    def tool(path: str, content: str, create_only: bool = False) -> str:
        fp = resolve_target(roots, path)
        existed = os.path.exists(fp)
        if create_only and existed:
            raise ToolError("file_exists", f"File already exists: {path}")
        if existed and os.path.isdir(fp):
            raise ToolError("invalid_path", f"Cannot write to a directory path: {path}")
        if rs is not None and existed:
            refusal = rs.check_overwrite(fp)
            if refusal:
                raise ToolError("stale_write", refusal)
        before = _text_before(fp) if existed else ""
        after = str(content or "")
        os.makedirs(os.path.dirname(fp) or ".", exist_ok=True)
        # Atomic write: write to temp file then rename for crash safety
        import tempfile
        dir_name = os.path.dirname(fp) or "."
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(after)
            os.replace(tmp_path, fp)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        if rs is not None:
            rs.record_write(fp, full_text=after)
        verb = "updated" if existed else "created"
        return _FileChangeResult(
            f"Wrote {len(after)} chars to {path} ({verb}).",
            [_file_change(path, "modified" if existed else "created", before, after)],
        )
    return tool


def _edit_file(roots, rs: Optional[ReadStateCache] = None):
    def tool(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        fp = ensure_file(roots, path)
        if rs is not None:
            refusal = rs.check_edit(fp)
            if refusal:
                raise ToolError("stale_write", refusal)
        text = decode(read_bytes(fp))
        try:
            result = editing.replace(text, old_string, new_string, bool(replace_all), label=path)
        except editing.EditError as exc:
            raise ToolError(exc.code, exc.message)
        updated = result.text
        _atomic_write(fp, updated)
        if rs is not None:
            rs.record_write(fp, full_text=updated)
        notes = []
        if result.strategy != "exact":
            notes.append(f"matched after {result.strategy} normalisation")
        if result.remaining:
            notes.append(f"{result.remaining} other occurrence(s) were left unchanged - pass replace_all=true to change all")
        suffix = f" ({'; '.join(notes)})" if notes else ""
        return _FileChangeResult(
            f"Edited {path}: replaced {result.count} occurrence(s) of '{old_string[:40]}…' with '{new_string[:40]}…'{suffix}.",
            [_file_change(path, "modified", text, updated)],
        )
    return tool


def _atomic_write(fp: str, content: str) -> None:
    """Write via temp file + rename (crash safe); ``newline=""`` preserves the file's own line endings."""
    import tempfile
    dir_name = os.path.dirname(fp) or "."
    os.makedirs(dir_name, exist_ok=True)
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        os.replace(tmp_path, fp)
    except BaseException:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def _apply_patch(roots, rs: Optional[ReadStateCache] = None):
    """opencode-style multi-file patch: validate everything first, then write, rolling back on any failure."""
    def tool(patch_text: str) -> str:
        try:
            hunks = editing.parse_patch(patch_text)
        except editing.EditError as exc:
            raise ToolError(exc.code, f"patch verification failed: {exc.message}")
        # ---- phase 1: compute every result in memory (nothing is written if any hunk is bad)
        plan = []          # (kind, display_path, fp, before, after, move_fp, move_display)
        seen = set()
        for h in hunks:
            try:
                if h.kind == "add":
                    fp = resolve_target(roots, h.path)
                    if os.path.exists(fp):
                        raise ToolError("file_exists", f"Add File '{h.path}': the file already exists. Use Update File instead.")
                    plan.append(("add", h.path, fp, None, h.contents, None, None))
                elif h.kind == "delete":
                    fp = ensure_file(roots, h.path)
                    if rs is not None and (refusal := rs.check_edit(fp)):
                        raise ToolError("stale_write", refusal)
                    plan.append(("delete", h.path, fp, decode(read_bytes(fp)), None, None, None))
                else:
                    fp = ensure_file(roots, h.path)
                    if rs is not None and (refusal := rs.check_edit(fp)):
                        raise ToolError("stale_write", refusal)
                    before = decode(read_bytes(fp))
                    after = editing.derive_update(h.path, h.chunks, before)
                    move_fp = resolve_target(roots, h.move_to) if h.move_to else None
                    if move_fp and move_fp != fp and os.path.exists(move_fp):
                        raise ToolError("file_exists", f"Move to '{h.move_to}': destination already exists.")
                    plan.append(("update", h.path, fp, before, after, move_fp, h.move_to))
            except editing.EditError as exc:
                raise ToolError(exc.code, f"{exc.message} (patch not applied; no files were changed)")
            key = os.path.normcase(plan[-1][2])
            if key in seen and h.kind != "update":
                raise ToolError("invalid_patch", f"'{h.path}' appears in more than one hunk; combine them.")
            seen.add(key)
        # ---- phase 2: apply, remembering originals so a mid-way failure restores everything
        done, changes, summary = [], [], []
        try:
            for kind, disp, fp, before, after, move_fp, move_disp in plan:
                if kind == "add":
                    _atomic_write(fp, after)
                    done.append((fp, None))
                    changes.append(_file_change(disp, "created", "", after)); summary.append(f"A {disp}")
                elif kind == "delete":
                    os.unlink(fp)
                    done.append((fp, before))
                    changes.append(_file_change(disp, "deleted", before, "")); summary.append(f"D {disp}")
                else:
                    target = move_fp or fp
                    _atomic_write(target, after)
                    done.append((target, None if move_fp else before))
                    if move_fp and move_fp != fp:
                        os.unlink(fp)
                        done.append((fp, before))
                    changes.append(_file_change(move_disp or disp, "modified", before, after))
                    summary.append(f"M {disp}" + (f" -> {move_disp}" if move_disp else ""))
                if rs is not None and kind != "delete":
                    rs.record_write(move_fp or fp, full_text=after)
        except BaseException as exc:
            for fp, original in reversed(done):
                try:
                    if original is None:
                        os.unlink(fp)
                    else:
                        _atomic_write(fp, original)
                except OSError:
                    pass
            raise ToolError("patch_failed", f"Patch failed while applying and was rolled back: {exc}")
        return _FileChangeResult("Success. Patch applied:\n" + "\n".join(summary), changes)
    return tool


def _append_file(roots, rs: Optional[ReadStateCache] = None):
    def tool(path: str, content: str) -> str:
        fp = resolve_target(roots, path)
        existed = os.path.isfile(fp)
        if rs is not None and existed:
            refusal = rs.check_overwrite(fp)
            if refusal:
                raise ToolError("stale_write", refusal)
        before = _text_before(fp) if existed else ""
        os.makedirs(os.path.dirname(fp) or ".", exist_ok=True)
        after_append = str(content or "")
        with open(fp, "a", encoding="utf-8", newline="") as f:
            f.write(after_append)
        full_after = decode(read_bytes(fp)) if os.path.isfile(fp) else after_append
        if rs is not None:
            rs.record_write(fp, full_text=full_after)
        return _FileChangeResult(
            f"Appended {len(after_append)} chars to {path}.",
            [_file_change(path, "modified" if existed else "created", before, full_after)],
        )
    return tool


def _create_file(roots, rs: Optional[ReadStateCache] = None):
    def tool(path: str, content: str) -> str:
        fp = resolve_target(roots, path)
        if os.path.exists(fp):
            raise ToolError("file_exists", f"File already exists: {path}")
        os.makedirs(os.path.dirname(fp) or ".", exist_ok=True)
        payload = str(content or "")
        with open(fp, "w", encoding="utf-8", newline="") as f:
            f.write(payload)
        if rs is not None:
            rs.record_write(fp, full_text=payload)
        return f"Created {path} ({len(payload)} chars)."
    return tool


def _delete_file(roots, rs: Optional[ReadStateCache] = None):
    def tool(path: str) -> str:
        fp = ensure_file(roots, path)
        if rs is not None:
            refusal = rs.check_overwrite(fp)
            if refusal:
                raise ToolError("stale_write", refusal)
        before = _text_before(fp)
        os.remove(fp)
        if rs is not None:
            rs.forget(fp)
        return _FileChangeResult(
            f"Deleted file {path}.",
            [_file_change(path, "deleted", before, "")],
        )
    return tool


def _rename_file(roots):
    def tool(old_path: str, new_path: str) -> str:
        src, dst = resolve_pair(roots, old_path, new_path)
        if not os.path.isfile(src):
            raise ToolError("file_not_found", f"File not found: {old_path}")
        if os.path.exists(dst):
            raise ToolError("file_exists", f"Target already exists: {new_path}")
        os.rename(src, dst)
        return f"Renamed {old_path} → {new_path}."
    return tool


def _copy_file(roots):
    def tool(source: str, target: str) -> str:
        src, dst = resolve_pair(roots, source, target)
        if not os.path.isfile(src):
            raise ToolError("file_not_found", f"File not found: {source}")
        if os.path.isdir(dst):
            raise ToolError("invalid_path", f"Target is a directory: {target}")
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copy2(src, dst)
        return f"Copied {source} → {target}."
    return tool


def _move_file(roots):
    def tool(source: str, target: str) -> str:
        src, dst = resolve_pair(roots, source, target)
        if not os.path.isfile(src):
            raise ToolError("file_not_found", f"File not found: {source}")
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.move(src, dst)
        return f"Moved {source} → {target}."
    return tool


def _create_directory(roots):
    def tool(path: str) -> str:
        fp = resolve_target(roots, path)
        os.makedirs(fp, exist_ok=True)
        return f"Created directory {path}."
    return tool


def _delete_directory(roots):
    def tool(path: str, recursive: bool = False) -> str:
        fp = ensure_dir(roots, path)
        if os.listdir(fp) and not recursive:
            raise ToolError(
                "directory_not_empty",
                f"Directory {path} is not empty. Pass recursive=True to delete it anyway.",
            )
        shutil.rmtree(fp) if recursive else os.rmdir(fp)
        return f"Deleted directory {path}."
    return tool


def _rename_directory(roots):
    def tool(old_path: str, new_path: str) -> str:
        src, dst = resolve_pair(roots, old_path, new_path)
        if not os.path.isdir(src):
            raise ToolError("directory_not_found", f"Directory not found: {old_path}")
        if os.path.exists(dst):
            raise ToolError("file_exists", f"Target already exists: {new_path}")
        os.rename(src, dst)
        return f"Renamed directory {old_path} → {new_path}."
    return tool


def _copy_directory(roots):
    def tool(source: str, target: str) -> str:
        src, dst = resolve_pair(roots, source, target)
        if not os.path.isdir(src):
            raise ToolError("directory_not_found", f"Directory not found: {source}")
        if os.path.exists(dst):
            raise ToolError("file_exists", f"Target already exists: {target}")
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*_IGNORE_PATTERNS))
        return f"Copied directory {source} → {target}."
    return tool


_IGNORE_PATTERNS = sorted(IGNORED_DIRS)


def _move_directory(roots):
    def tool(source: str, target: str) -> str:
        src, dst = resolve_pair(roots, source, target)
        if not os.path.isdir(src):
            raise ToolError("directory_not_found", f"Directory not found: {source}")
        if os.path.exists(dst):
            raise ToolError("file_exists", f"Target already exists: {target}")
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.move(src, dst)
        return f"Moved directory {source} → {target}."
    return tool


def _list_directory(roots):
    def tool(path: str = ".", show_hidden: bool = False) -> str:
        fp = ensure_dir(roots, path)
        entries = []
        for name in sorted(os.listdir(fp)):
            if match_ignored(name, os.path.isdir(os.path.join(fp, name))):
                continue
            if not show_hidden and name.startswith("."):
                continue
            full = os.path.join(fp, name)
            kind = "dir" if os.path.isdir(full) else "file"
            size = "" if kind == "dir" else str(os.path.getsize(full))
            entries.append(f"{name}\t{kind}\t{size}")
        return "\n".join(entries[:200]) if entries else "(empty directory)"
    return tool


def _directory_tree(roots):
    def _tree(root_abs: str, base: str, depth: int, show_hidden: bool, cur: int) -> List[str]:
        lines: List[str] = []
        if cur > depth:
            return lines
        try:
            names = sorted(os.listdir(base))
        except OSError:
            return lines
        for name in names:
            full = os.path.join(base, name)
            is_dir = os.path.isdir(full)
            if match_ignored(name, is_dir):
                continue
            if not show_hidden and name.startswith("."):
                continue
            prefix = "  " * (cur - 1) + ("├─ " if cur > 1 else "")
            lines.append(prefix + name + ("/" if is_dir else ""))
            if is_dir:
                lines.extend(_tree(root_abs, full, depth, show_hidden, cur + 1))
        return lines

    def tool(path: str = ".", depth: int = 3, show_hidden: bool = False) -> str:
        fp = ensure_dir(roots, path)
        lines = [os.path.basename(fp) or fp]
        lines.extend(_tree(fp, fp, int(depth), show_hidden, 1))
        return "\n".join(lines)

    return tool


def _exists(roots):
    def tool(path: str) -> Dict[str, Any]:
        fp = resolve_target(roots, path)
        exists = os.path.exists(fp)
        return {
            "path": path,
            "exists": exists,
            "type": "dir" if (exists and os.path.isdir(fp)) else ("file" if exists else None),
        }
    return tool


def _file_info(roots):
    def tool(path: str) -> Dict[str, Any]:
        fp = ensure_file(roots, path)
        st = os.stat(fp)
        try:
            data = read_bytes(fp)
            text = is_text(data)
        except ToolError:
            text = False
        return {
            "path": path,
            "type": "file",
            "size_bytes": st.st_size,
            "lines": len(decode(data).split("\n")) if text else None,
            "text": text,
            "extension": os.path.splitext(fp)[1].lstrip(".") or None,
            "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        }
    return tool


def _glob_search(roots):
    def tool(pattern: str, path: str = ".") -> Dict[str, Any]:
        if not (pattern or "").strip():
            raise ToolError("invalid_pattern", "pattern must not be empty")
        fp = ensure_dir(roots, path)
        matches: List[str] = []
        pat = pattern.replace("\\", "/")
        for rel, full in walk_files(fp):
            rel_norm = rel.replace("\\", "/")
            if fnmatch.fnmatch(rel_norm, pat) or fnmatch.fnmatch(rel_norm, f"**/{pat}") or fnmatch.fnmatch(os.path.basename(full), pat):
                matches.append(rel_norm)
                if len(matches) >= MAX_RESULTS:
                    break
        return {"pattern": pattern, "matches": matches, "count": len(matches)}
    return tool


def _search_text(roots):
    def tool(query: str, path: str = ".", glob: Optional[str] = None) -> str:
        needle = str(query or "").lower()
        if not needle:
            raise ToolError("invalid_query", "query must not be empty")
        fp = ensure_dir(roots, path)
        results: List[str] = []
        for rel, full in walk_files(fp, glob=glob):
            # One unreadable file (a Windows MAX_PATH overflow on a deeply
            # nested path, a permission error, a broken symlink) must not
            # abort matches already found in every other file — confirmed
            # live: read_bytes raising here took down an entire search_text
            # call over one bad file deep in a vendored dependency tree.
            # search.py's grep/regex_search already skip a file like this
            # (via read_text_stream's own internal OSError handling); this
            # brings _search_text to the same standard.
            try:
                data = read_bytes(full)[:MAX_TEXT_CHARS]
            except ToolError:
                continue
            if not is_text(data):
                continue
            for i, line in enumerate(decode(data).split("\n")):
                if needle in line.lower():
                    results.append(f"{rel}:{i + 1}: {line.strip()[:200]}")
                    if len(results) >= MAX_RESULTS:
                        break
            if len(results) >= MAX_RESULTS:
                break
        return "\n".join(results) if results else "No matches found."
    return tool


def _replace_text(roots, rs: Optional[ReadStateCache] = None):
    _inner = _edit_file(roots, rs)
    def tool(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        return _inner(path, old_string, new_string, replace_all)
    return tool


def _read_multiple(roots, rs: Optional[ReadStateCache] = None):
    def tool(paths: List[str]) -> Dict[str, Any]:
        if not paths:
            raise ToolError("invalid_paths", "paths must be a non-empty list")
        out: Dict[str, Any] = {}
        for p in paths:
            try:
                fp = ensure_file(roots, p)
                if is_pdf(fp):
                    out[p] = read_pdf_text(fp)
                    if rs is not None:
                        rs.record_read(fp, full_text=None, partial=False)
                else:
                    data = read_bytes(fp)
                    if is_text(data):
                        text = decode(data)
                        out[p] = text
                        if rs is not None:
                            rs.record_read(fp, full_text=text, partial=False)
                    else:
                        out[p] = "[Binary file — contents not readable]"
                        if rs is not None:
                            rs.record_read(fp, full_text=None, partial=False)
            except ToolError as exc:
                out[p] = {"error": exc.to_dict()}
        return {"files": out, "count": len(out)}
    return tool


# ── Schemas ───────────────────────────────────────────────────────────────────

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "read_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file, relative to workspace root or absolute. Example: 'src/app.py' or 'README.md'."},
            "offset": {"type": "integer", "description": "Line number to start reading from (0-indexed). Use this for large files. Default: 0 (start from beginning). Example: 50 to skip the first 50 lines."},
            "limit": {"type": "integer", "description": "Maximum number of lines to return. Use for large files to avoid huge outputs. Example: 100 to read lines 0-99."},
        },
        "required": ["path"],
    },
    "write_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to create or overwrite. Example: 'src/components/Button.tsx'. Parent directories are created automatically."},
            "content": {"type": "string", "description": "The COMPLETE file contents to write. This REPLACES the entire file — nothing is preserved from any previous version. Include everything."},
            "create_only": {"type": "boolean", "description": "If true, refuses to overwrite an existing file (errors if file exists). Default: false (overwrites)."},
        },
        "required": ["path", "content"],
    },
    "edit_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path of the existing file to edit. You MUST read the file first with read_file to get the exact text."},
            "old_string": {"type": "string", "description": "The EXACT text to find and replace — must match the file content character-for-character including whitespace and indentation. Use read_file first to get this right."},
            "new_string": {"type": "string", "description": "The replacement text that will replace old_string."},
            "replace_all": {"type": "boolean", "description": "If true, replaces ALL occurrences of old_string. If false (default), replaces only the first occurrence. Do NOT include line-number prefixes (like '12: ') from read_file output in old_string/new_string."},
        },
        "required": ["path", "old_string", "new_string"],
    },
    "apply_patch": {
        "type": "object",
        "properties": {
            "patch_text": {"type": "string", "description": "The whole patch. Format: '*** Begin Patch', then one or more file sections, then '*** End Patch'. Sections: '*** Add File: <path>' (every following line starts with +), '*** Delete File: <path>', or '*** Update File: <path>' (optional '*** Move to: <new path>') followed by hunks that start with '@@ <optional nearby line>' and use ' ' for context, '-' for removed and '+' for added lines."},
        },
        "required": ["patch_text"],
    },
    "append_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to append to. Creates the file if it doesn't exist. Example: 'logs/build.log'."},
            "content": {"type": "string", "description": "Text to add at the end of the file."},
        },
        "required": ["path", "content"],
    },
    "create_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Full path for the NEW file including filename. Example: 'src/utils/helpers.ts'. This tool refuses to overwrite existing files."},
            "content": {"type": "string", "description": "The COMPLETE file content for the new file. Do NOT create empty files — include all content in this one call."},
        },
        "required": ["path", "content"],
    },
    "delete_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path of the file to permanently delete. This cannot be undone. Example: 'temp/debug.log'."},
        },
        "required": ["path"],
    },
    "rename_file": {
        "type": "object",
        "properties": {
            "old_path": {"type": "string", "description": "Current path of the file. Example: 'src/old_name.ts'."},
            "new_path": {"type": "string", "description": "New path/name for the file. Example: 'src/new_name.ts'. Target must not already exist."},
        },
        "required": ["old_path", "new_path"],
    },
    "copy_file": {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Path of the file to copy. Example: 'src/config.ts'."},
            "target": {"type": "string", "description": "Destination path for the copy. Example: 'src/config.backup.ts'. Parent dirs are created automatically."},
        },
        "required": ["source", "target"],
    },
    "move_file": {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Path of the file to move. Example: 'src/utils.ts'."},
            "target": {"type": "string", "description": "Destination path. Example: 'lib/utils.ts'. The file is removed from the source location."},
        },
        "required": ["source", "target"],
    },
    "create_directory": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path to create (including any missing parents). Example: 'src/components/ui'."},
        },
        "required": ["path"],
    },
    "delete_directory": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to delete. Example: 'dist' or 'temp'. Must be empty unless recursive=true."},
            "recursive": {"type": "boolean", "description": "If true, deletes the directory and ALL its contents recursively. Default: false."},
        },
        "required": ["path"],
    },
    "rename_directory": {
        "type": "object",
        "properties": {
            "old_path": {"type": "string", "description": "Current directory path. Example: 'src/components_old'."},
            "new_path": {"type": "string", "description": "New directory path. Example: 'src/components'. Target must not exist."},
        },
        "required": ["old_path", "new_path"],
    },
    "copy_directory": {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Directory to copy recursively. Example: 'src/utils'."},
            "target": {"type": "string", "description": "Destination path. Example: 'src/utils_backup'. Target must not exist."},
        },
        "required": ["source", "target"],
    },
    "move_directory": {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Directory to move. Example: 'src/helpers'."},
            "target": {"type": "string", "description": "Destination path. Example: 'lib/helpers'. Source is removed after the move."},
        },
        "required": ["source", "target"],
    },
    "list_directory": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list. Use '.' for project root, 'src' for src dir, etc. Default: '.' (project root)."},
            "show_hidden": {"type": "boolean", "description": "Include hidden files/folders (starting with '.'). Default: false."},
        },
    },
    "directory_tree": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Root directory for the tree. Default: '.' (project root)."},
            "depth": {"type": "integer", "description": "How deep to go (1-20). Use 2 for a quick overview, 5 for deep exploration. Default: 3."},
            "show_hidden": {"type": "boolean", "description": "Include hidden files/folders. Default: false."},
        },
    },
    "exists": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to check. Example: 'src/app.py' or 'package.json'. Returns whether it exists and if it's a file or directory."},
        },
        "required": ["path"],
    },
    "file_info": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File to inspect. Returns size, line count, extension, and last modified time. Example: 'src/app.py'."},
        },
        "required": ["path"],
    },
    "glob_search": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern to match files. Examples: '**/*.ts' (all TypeScript files), '*.json' (JSON in root), 'src/**/*.test.*' (all test files under src)."},
            "path": {"type": "string", "description": "Directory to search under. Default: '.' (entire workspace). Use 'src' to narrow the search."},
        },
        "required": ["pattern"],
    },
    "search_text": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Case-insensitive text to search for across all text files. Example: 'useState' or 'TODO: fix'. Returns matching lines with file paths and line numbers."},
            "path": {"type": "string", "description": "Directory to search in. Default: '.' (entire workspace). Use 'src/components' to narrow."},
            "glob": {"type": "string", "description": "Filter by filename pattern. Examples: '*.ts' (only TypeScript), '*.py' (only Python), '*.{ts,tsx}' (TypeScript files)."},
        },
        "required": ["query"],
    },
    "replace_text": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File to modify. Must exist."},
            "old_string": {"type": "string", "description": "Exact text to find and replace (must match exactly)."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences (true) or just the first (false, default)."},
        },
        "required": ["path", "old_string", "new_string"],
    },
    "read_multiple_files": {
        "type": "object",
        "properties": {
            "paths": {"type": "array", "items": {"type": "string"}, "description": "List of file paths to read at once. Example: ['src/app.ts', 'src/index.ts', 'package.json']. Returns a dict keyed by path."},
        },
        "required": ["paths"],
    },
}


# ── Tool collection ───────────────────────────────────────────────────────────

def _validate_depth(kwargs: Dict[str, Any]) -> Optional[str]:
    depth = kwargs.get("depth")
    if depth is not None and not (1 <= int(depth) <= 20):
        return "depth must be between 1 and 20"


def build_tools(ctx: CapabilityContext) -> List[ToolSpec]:
    roots = ctx.roots
    rs = ctx.read_state
    return [
        spec("read_file",
             "Read a text file's full contents and return the text. USE THIS WHEN: you need to see what's in a file before editing it, you want to answer questions about file content, or you want to verify changes after writing. ALWAYS call this BEFORE edit_file — edit_file requires the exact old_string to match the file. Returns the file text, or '(empty file)' for empty files, or '[Binary file]' for non-text files. Supports offset/limit for large files.",
             TOOL_SCHEMAS["read_file"], _read_file(roots, rs), permissions=_READ_PERMS, safety_level=SAFE, capability=CAPABILITY_NAME,
             result_format={"type": "string", "description": "The file text, or '(empty file)' / a binary notice."}),
        spec("write_file",
             "Create a new file OR completely overwrite an existing file with the provided content. USE THIS WHEN: you need to create a brand new file, or replace the entire contents of a file. The content you provide becomes the ENTIRE file — nothing from the previous version is preserved. Parent directories are created automatically. Prefer edit_file for small targeted changes to existing files.",
             TOOL_SCHEMAS["write_file"], _write_file(roots, rs), permissions=_WRITE_PERMS, safety_level=CAUTIOUS, capability=CAPABILITY_NAME),
        spec("edit_file",
             "Make a targeted text replacement in an existing file. USE THIS WHEN: you want to change a specific part of a file without rewriting the whole thing. You MUST call read_file first to get the exact old_string — it must match character-for-character including all whitespace and indentation. If old_string is not found, the tool will error. Use replace_all=true to change every occurrence.",
             TOOL_SCHEMAS["edit_file"], _edit_file(roots, rs), permissions=_WRITE_PERMS, safety_level=CAUTIOUS, capability=CAPABILITY_NAME),
        spec("apply_patch",
             "Apply a multi-file patch in one atomic step: add, update (with context lines), delete or rename several files at once. USE THIS WHEN a change touches more than one file or more than one place in a file. All hunks are validated first; if any fails nothing is written. Format: *** Begin Patch / *** Update File: src/app.py / @@ def greet(): / -print(\"Hi\") / +print(\"Hello\") / *** Add File: notes.txt / +text / *** Delete File: old.txt / *** End Patch. Context lines are matched leniently (trailing spaces, curly quotes).",
             TOOL_SCHEMAS["apply_patch"], _apply_patch(roots, rs), permissions=_WRITE_PERMS, safety_level=CAUTIOUS, capability=CAPABILITY_NAME),
        spec("append_file",
             "Add text to the end of a file. Creates the file if it doesn't exist. USE THIS WHEN: you want to add new content to the end of an existing file, like adding a new function, a new log entry, or extending configuration.",
             TOOL_SCHEMAS["append_file"], _append_file(roots, rs), permissions=_WRITE_PERMS, safety_level=CAUTIOUS, capability=CAPABILITY_NAME),
        spec("create_file",
             "Create a brand new file with COMPLETE content in one call. USE THIS WHEN: creating a file that doesn't exist yet. This tool REFUSES to overwrite existing files and REFUSES empty content. Include the FULL file content in this single call — do not create an empty file planning to fill it in later.",
             TOOL_SCHEMAS["create_file"], _create_file(roots, rs), permissions=_WRITE_PERMS, safety_level=CAUTIOUS, capability=CAPABILITY_NAME),
        spec("delete_file",
             "Permanently delete a file. USE THIS WHEN: you need to remove a file that is no longer needed. WARNING: This cannot be undone. Prefer git_restore if you want to undo changes to tracked files.",
             TOOL_SCHEMAS["delete_file"], _delete_file(roots, rs), permissions=_MUTATE_PERMS, safety_level=DANGEROUS_LEVEL, capability=CAPABILITY_NAME),
        spec("rename_file",
             "Rename or move a file to a new path. USE THIS WHEN: you want to give a file a new name or move it within the workspace. The old path must exist and the new path must not already exist.",
             TOOL_SCHEMAS["rename_file"], _rename_file(roots), permissions=_MUTATE_PERMS, safety_level=DANGEROUS_LEVEL, capability=CAPABILITY_NAME),
        spec("copy_file",
             "Copy a file to a new location (original stays). USE THIS WHEN: you want a duplicate of a file at a different path, or want to create a backup before editing. The source must exist.",
             TOOL_SCHEMAS["copy_file"], _copy_file(roots), permissions=_MUTATE_PERMS, safety_level=CAUTIOUS, capability=CAPABILITY_NAME),
        spec("move_file",
             "Move a file from one location to another (source is removed). USE THIS WHEN: you want to relocate a file. The source must exist. Prefer rename_file for simple renames within the same directory.",
             TOOL_SCHEMAS["move_file"], _move_file(roots), permissions=_MUTATE_PERMS, safety_level=DANGEROUS_LEVEL, capability=CAPABILITY_NAME),
        spec("create_directory",
             "Create a new directory (and any missing parent directories). USE THIS WHEN: you need a new folder structure, e.g. before creating files in a new subdirectory.",
             TOOL_SCHEMAS["create_directory"], _create_directory(roots), permissions=_WRITE_PERMS, safety_level=CAUTIOUS, capability=CAPABILITY_NAME),
        spec("delete_directory",
             "Delete a directory. Must be empty unless recursive=true. USE THIS WHEN: removing a directory that is no longer needed. WARNING: recursive deletion cannot be undone.",
             TOOL_SCHEMAS["delete_directory"], _delete_directory(roots), permissions=_MUTATE_PERMS, safety_level=DANGEROUS_LEVEL, capability=CAPABILITY_NAME),
        spec("rename_directory",
             "Rename a directory. USE THIS WHEN: giving a directory a new name. The old path must exist and the new path must not exist.",
             TOOL_SCHEMAS["rename_directory"], _rename_directory(roots), permissions=_MUTATE_PERMS, safety_level=DANGEROUS_LEVEL, capability=CAPABILITY_NAME),
        spec("copy_directory",
             "Recursively copy an entire directory tree to a new location. USE THIS WHEN: you want to duplicate a folder and all its contents. Target must not exist.",
             TOOL_SCHEMAS["copy_directory"], _copy_directory(roots), permissions=_MUTATE_PERMS, safety_level=CAUTIOUS, capability=CAPABILITY_NAME),
        spec("move_directory",
             "Move an entire directory to a new location (source is removed). USE THIS WHEN: relocating a folder and all its contents.",
             TOOL_SCHEMAS["move_directory"], _move_directory(roots), permissions=_MUTATE_PERMS, safety_level=DANGEROUS_LEVEL, capability=CAPABILITY_NAME),
        spec("list_directory",
             "List all files and folders in a directory. USE THIS WHEN: exploring what's in a folder, finding files to work with. Shows name, type (file/dir), and size. Use directory_tree for a nested view.",
             TOOL_SCHEMAS["list_directory"], _list_directory(roots), permissions=_READ_PERMS, safety_level=SAFE, capability=CAPABILITY_NAME),
        spec("directory_tree",
             "Show a nested tree view of a directory. USE THIS WHEN: you need to understand the project structure at a glance. Shows folders and files in a tree format. Use depth=2 for overview, depth=5 for deep exploration.",
             TOOL_SCHEMAS["directory_tree"], _directory_tree(roots), permissions=_READ_PERMS, safety_level=SAFE, capability=CAPABILITY_NAME, validate=_validate_depth),
        spec("exists",
             "Check if a path exists and whether it is a file or directory. USE THIS WHEN: verifying a file/folder exists before reading, editing, or deleting it. Returns {path, exists, type}.",
             TOOL_SCHEMAS["exists"], _exists(roots), permissions=_READ_PERMS, safety_level=SAFE, capability=CAPABILITY_NAME,
             result_format={"type": "object", "properties": {"path": {"type": "string"}, "exists": {"type": "boolean"}, "type": {"type": ["string", "null"]}}}),
        spec("file_info",
             "Get file metadata: size in bytes, line count, extension, and last modified time. USE THIS WHEN: you need to know how big a file is, how many lines it has, or when it was last changed.",
             TOOL_SCHEMAS["file_info"], _file_info(roots), permissions=_READ_PERMS, safety_level=SAFE, capability=CAPABILITY_NAME,
             result_format={"type": "object"}),
        spec("glob_search",
             "Find files matching a glob pattern (like **/*.ts for all TypeScript files). USE THIS WHEN: you want to find all files of a certain type or matching a pattern. Returns a list of matching file paths. Broader than search_text — this searches file NAMES not content.",
             TOOL_SCHEMAS["glob_search"], _glob_search(roots), permissions=_READ_PERMS, safety_level=SAFE, capability=CAPABILITY_NAME),
        spec("search_text",
             "Search all text files ON DISK, in the LOCAL workspace/codebase, for a case-insensitive string. USE THIS WHEN: you want to find where a function, variable, or text appears in the local project's own files. Returns 'file:line: content' format. Use glob param to filter by file type. Does NOT search the internet — this only ever looks at files already in the workspace. If the request mentions 'the web', 'online', 'the internet', a URL, or anything not already part of this project, use web_search instead, not this tool.",
             TOOL_SCHEMAS["search_text"], _search_text(roots), permissions=_READ_PERMS, safety_level=SAFE, capability=CAPABILITY_NAME),
        spec("replace_text",
             "Find and replace text occurrences in a file. USE THIS WHEN: same as edit_file — this is an alias for the same operation. Requires exact text match. Use read_file first.",
             TOOL_SCHEMAS["replace_text"], _replace_text(roots, rs), permissions=_WRITE_PERMS, safety_level=CAUTIOUS, capability=CAPABILITY_NAME),
        spec("read_multiple_files",
             "Read multiple files in a single call. USE THIS WHEN: you need to examine several files at once, e.g. to understand a module, compare files, or gather context before making changes. Returns a dict keyed by file path.",
             TOOL_SCHEMAS["read_multiple_files"], _read_multiple(roots, rs), permissions=_READ_PERMS, safety_level=SAFE, capability=CAPABILITY_NAME,
             result_format={"type": "object", "properties": {"files": {"type": "object"}, "count": {"type": "integer"}}}),
    ]
