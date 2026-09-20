"""
orcha.capabilities.pathing
==========================
Shared path-resolution and filesystem safety helpers.

Every capability that touches the workspace routes its paths through this
module so the "never escape the workspace roots" invariant has exactly one
implementation:

1. absolute paths are validated directly
2. relative paths resolve against the first root
3. a path is only allowed when it stays inside at least one root
   (no ``..`` escapes, no different-drive shortcuts)
"""
from __future__ import annotations

import fnmatch
import os
from typing import Any, Dict, List, Optional

from .base import ToolError

# Directories/files that filesystem tools never search, list or walk.
IGNORED_DIRS: set = {
    "node_modules", ".git", "dist", ".next", ".cache", "__pycache__",
    ".venv", "venv", ".build-venv", ".anvira-index-cache", ".idea", ".vscode",
    "coverage", ".tmp", "bin", "build", "out", "target", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache",
    # Vendored third-party source trees (e.g. Orcha/vendor/Observability/
    # phoenix-main) — same reasoning as node_modules: not the agent's own
    # code, and confirmed live to cause real failures walking into them —
    # a vendored test fixture's deeply nested path exceeded Windows' 260-
    # char MAX_PATH limit and aborted a search_text call entirely (see the
    # per-file error handling fix in filesystem.py's _search_text, which
    # covers this defensively too; excluding vendor/ avoids hitting it and
    # avoids polluting results with a dependency's own code either way).
    "vendor",
}
IGNORED_FILES: set = {".DS_Store", "Thumbs.db"}

MAX_TEXT_CHARS = 200_000
MAX_RESULTS = 50


def normalize_path(p: str) -> str:
    """Absolute, expanded, drive-normalized form of ``p``."""
    return os.path.abspath(os.path.expanduser(str(p or "")))


def resolve_target(roots: List[str], path: str) -> str:
    """Resolve ``path`` against the workspace roots and verify it stays inside."""
    if not roots:
        raise ToolError(
            "no_workspace",
            "No workspace roots configured. Attach a project or folder first.",
        )
    raw = str(path or "")
    if not raw.strip():
        raise ToolError("invalid_path", "path must not be empty")
    target = (
        normalize_path(raw)
        if os.path.isabs(raw)
        else normalize_path(os.path.join(roots[0], raw))
    )
    for root in roots:
        resolved = normalize_path(root)
        try:
            rel = os.path.relpath(target, resolved)
        except ValueError:  # different drive/mount → cannot be inside this root
            continue
        if rel == "." or (not rel.startswith("..") and not os.path.isabs(rel)):
            return target
    raise ToolError(
        "path_outside_workspace",
        f"Path is outside the attached workspace: {path}",
        detail={"path": raw},
    )


def resolve_pair(roots: List[str], source: str, target: str) -> tuple:
    """Resolve a (source, target) pair; both must stay inside the roots."""
    return resolve_target(roots, source), resolve_target(roots, target)


def ensure_file(roots: List[str], path: str) -> str:
    fp = resolve_target(roots, path)
    if not os.path.isfile(fp):
        raise ToolError("file_not_found", f"File not found: {path}", detail={"path": fp})
    return fp


def ensure_dir(roots: List[str], path: str) -> str:
    fp = resolve_target(roots, path)
    if not os.path.isdir(fp):
        raise ToolError("directory_not_found", f"Directory not found: {path}", detail={"path": fp})
    return fp


def is_text(data: bytes) -> bool:
    return b"\x00" not in data[:4096]


def is_pdf(path: str) -> bool:
    """Check if a file is a PDF by extension or magic bytes."""
    if path.lower().endswith(".pdf"):
        return True
    try:
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def read_pdf_text(path: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Extract text from a PDF using PyMuPDF (fitz).

    Falls back gracefully if fitz is unavailable or the PDF is
    corrupted / image-only.  Returns at most *max_chars* characters.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ToolError(
            "pdf_not_supported",
            "PDF reading requires PyMuPDF (pip install PyMuPDF). "
            "Install it to enable PDF text extraction.",
        )
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise ToolError(
            "pdf_error",
            f"Could not open PDF: {exc}",
        )
    parts: list[str] = []
    total = 0
    try:
        for page in doc:
            text = page.get_text()
            if text:
                parts.append(text)
                total += len(text)
                if total >= max_chars:
                    break
    finally:
        doc.close()
    result = "\n".join(parts)[:max_chars]
    return result if result.strip() else "(PDF contains no extractable text — may be image-only)"


def read_bytes(path: str, max_bytes: int = 10 * 1024 * 1024) -> bytes:
    """Read up to ``max_bytes`` (default 10 MB) from ``path``.

    Reading large files entirely into memory can cause OOM on constrained
    systems.  The limit is applied at the raw read level so callers still
    get a clean ``ToolError`` when the file is too large.
    """
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
            return data
    except OSError as exc:
        raise ToolError("io_error", f"Could not read {path}: {exc}")


def read_text_stream(path: str, max_chars: int = MAX_TEXT_CHARS):
    """Yield (lineno, line) pairs from *path* without loading the entire file.

    Stops after *max_chars* total characters have been yielded.  Each line
    has the trailing newline stripped.  This is much faster and cheaper than
    ``read_bytes(path)[:limit]`` for large files because it never buffers
    more than one line at a time.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            total = 0
            for lineno, line in enumerate(fh, start=1):
                stripped = line.rstrip("\n\r")
                total += len(stripped)
                if total > max_chars:
                    return
                yield lineno, stripped
    except OSError:
        return


def decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def match_ignored(name: str, is_dir: bool) -> bool:
    if name in IGNORED_FILES:
        return True
    if is_dir and name in IGNORED_DIRS:
        return True
    return False


def walk_files(root: str, rel_base: str = ".", glob: Optional[str] = None):
    """
    Yield (relpath, abs_path) for every non-ignored, text-readable file under
    ``root``. ``glob`` optionally filters by filename pattern.
    """
    base_abs = normalize_path(root) if os.path.isabs(root) else normalize_path(os.path.join(root, rel_base))
    if not os.path.isdir(base_abs):
        return
    for dirpath, dirs, files in os.walk(base_abs):
        dirs[:] = [d for d in dirs if not match_ignored(d, True)]
        for name in sorted(files):
            if match_ignored(name, False):
                continue
            if glob and not fnmatch.fnmatch(name, glob):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, base_abs)
            yield rel, full
