"""
orcha.capabilities.outputstore
==============================
Oversized tool-output persistence.

When a tool produces more text than the model should ever see in one tool
result (a giant grep, a huge directory dump, chatty command output), the
full result is persisted to disk under a per-run directory and the model
receives a bounded preview plus the path of the saved file — which it can
then inspect with ``read_file`` (offset/limit) like any other file.

Design notes:

- The store is attached to a :class:`ToolExecutor`, so EVERY consumer of
  the executor (AgentRuntime loop, graph AgentNode, API surfaces) gets the
  same protection without touching individual tools.
- Persistence only ever applies to SUCCESSFUL results; errors stay small
  and structured.
- Files land in ``<directory>/<run_id>/<seq>_<tool>.txt`` with an atomic
  write; a global byte budget prunes the oldest files when exceeded, so a
  long-lived desktop process cannot fill the disk.
- ``attach=None`` (the default everywhere) keeps today's behavior exactly.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from typing import Any, Optional

_DEFAULT_THRESHOLD_CHARS = 16_000
_PREVIEW_CHARS = 12_000
_MAX_FILES = 500
_MAX_TOTAL_BYTES = 64 * 1024 * 1024


class OutputStore:
    """
    Persist oversized tool results to disk.

    Parameters
    ----------
    directory   Base directory for saved outputs (created lazily).
    threshold   Results rendering longer than this many chars are persisted.
    preview     Chars of the full text kept in the model-facing message.
    """

    def __init__(
        self,
        directory: str,
        *,
        threshold_chars: int = _DEFAULT_THRESHOLD_CHARS,
        preview_chars: int = _PREVIEW_CHARS,
    ) -> None:
        self.directory = directory
        self.threshold_chars = max(100, int(threshold_chars))
        self.preview_chars = max(50, min(int(preview_chars), self.threshold_chars))
        self._lock = threading.Lock()
        self._counter = 0

    # ── Core API ────────────────────────────────────────────────────────

    def wrap_if_oversized(self, tool_name: str, value: Any) -> "tuple[Any, Optional[str]]":
        """
        Return ``(value_or_preview, saved_path_or_None)``. When the rendered
        ``value`` fits the threshold it passes through untouched; otherwise
        the full render is written to disk and a preview string returned.
        """
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, default=str, ensure_ascii=False, indent=2)
            except Exception:
                text = str(value)
        if len(text) <= self.threshold_chars:
            return value, None
        path = self.persist(tool_name, text)
        preview = text[: self.preview_chars]
        omitted = len(text) - self.preview_chars
        wrapped = (
            f"{preview}\n\n[Output truncated for context: showing {self.preview_chars} "
            f"of {len(text)} chars ({omitted} omitted). Full output saved to: {path} — "
            "use read_file with offset/limit to inspect specific sections.]"
        )
        return wrapped, path

    def persist(self, tool_name: str, text: str) -> str:
        """Write ``text`` to a fresh file and return its absolute path."""
        safe_tool = "".join(c if c.isalnum() or c in "-_" else "_" for c in tool_name)[:40]
        run_dir = self._run_dir()
        os.makedirs(run_dir, exist_ok=True)
        with self._lock:
            self._counter += 1
            seq = self._counter
        filename = f"{int(time.time())}_{seq:06d}_{safe_tool}.txt"
        target = os.path.join(run_dir, filename)
        # Atomic write, mirroring the filesystem capability's convention.
        fd, tmp_path = tempfile.mkstemp(dir=run_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
            os.replace(tmp_path, target)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        self._prune()
        return target

    # ── Internals ───────────────────────────────────────────────────────

    def _run_dir(self) -> str:
        return self.directory

    def _prune(self) -> None:
        """
        Keep the store bounded: drop oldest files beyond ``_MAX_FILES`` or a
        total-byte budget. Failures are silently ignored — pruning must
        never break a tool result.
        """
        try:
            entries = []
            total = 0
            with os.scandir(self.directory) as it:
                for entry in it:
                    if not entry.is_file():
                        continue
                    try:
                        entries.append((entry.stat().st_mtime, entry.path, entry.stat().st_size))
                        total += entries[-1][2]
                    except OSError:
                        continue
            entries.sort()
            excess_files = max(0, len(entries) - _MAX_FILES)
            for idx, (_, path, size) in enumerate(entries):
                if excess_files > 0 or (total > _MAX_TOTAL_BYTES and idx < len(entries) - 1):
                    try:
                        os.unlink(path)
                        total -= size
                        if excess_files > 0:
                            excess_files -= 1
                    except OSError:
                        continue
                else:
                    break
        except OSError:
            pass


__all__ = ["OutputStore"]
