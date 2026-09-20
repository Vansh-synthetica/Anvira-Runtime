"""
orcha.capabilities.readstate
============================
Stale-write protection for the filesystem capability.

The runtime keeps a bounded cache of what the AGENT has recently read from
the workspace (mtime + content hash + a partial-view flag). Before a tool
mutates an existing file the cache answers one question:

    "Is the on-disk file still what the agent believes it is?"

Three failure modes are caught:

1. **Never read** — an existing file is edited/appended/deleted without any
   prior ``read_file``. The model would be writing blind; refused with a
   pointer to read it first.
2. **Partial view** — only a slice (offset/limit) was read, so an edit
   anchored to "exact text currently in the file" may match somewhere the
   agent never saw. Refused for in-place edits until the whole file is read.
3. **Changed since read** — the mtime advanced after the last read. Because
   Windows mtimes can lie, the check falls back to comparing actual content
   against what was recorded before declaring staleness.

The cache is deliberately *advisory infrastructure with teeth*: when no
cache is attached to a :class:`~orcha.capabilities.base.CapabilityContext`
(the default), tools behave exactly as they always have. Attaching a cache
turns every check into a structured :class:`ToolError` the model can act
on ("re-read the file"), never a crash.

Every successful write/edit/append flows through :meth:`record_write`, so a
chain of edits against fresh state never trips the guard, and the cache
tracks at most ``max_entries`` paths (LRU eviction).
"""
from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from typing import Optional

logger_name = "orcha.capabilities.readstate"


class _Entry:
    __slots__ = ("path", "mtime_ns", "size", "sha256", "full_text", "partial")

    def __init__(
        self,
        path: str,
        mtime_ns: int,
        size: int,
        sha256: str,
        full_text: Optional[str],
        partial: bool,
    ) -> None:
        self.path = path
        self.mtime_ns = mtime_ns
        self.size = size
        self.sha256 = sha256
        # ``full_text`` is None for binary/unreadable files and for entries
        # evicted down to metadata-only; staleness then relies on mtime+size.
        self.full_text = full_text
        self.partial = partial


class ReadStateCache:
    """
    Bounded, thread-safe record of what the agent last saw on disk.

    Keys are normalized absolute paths. ``max_entries`` bounds memory;
    eviction is LRU. All methods are best-effort: I/O errors during
    verification degrade to "treat as changed" (fail closed), never raise.
    """

    def __init__(self, max_entries: int = 512) -> None:
        self._entries: "OrderedDict[str, _Entry]" = OrderedDict()
        self._max_entries = max(16, int(max_entries))
        self._lock = threading.Lock()

    # ── Recording ──────────────────────────────────────────────────────

    def record_read(
        self,
        path: str,
        *,
        full_text: Optional[str],
        partial: bool,
    ) -> None:
        """Record that the agent read ``path``. ``partial=True`` when only a
        slice was returned; ``full_text=None`` for binary/unhashable reads."""
        try:
            st = os.stat(path)
        except OSError:
            return
        entry = _Entry(
            path=path,
            mtime_ns=st.st_mtime_ns,
            size=st.st_size,
            sha256=_hash_text(full_text) if full_text is not None else "",
            full_text=full_text,
            partial=bool(partial),
        )
        with self._lock:
            self._entries[path] = entry
            self._entries.move_to_end(path)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def record_write(self, path: str, *, full_text: Optional[str]) -> None:
        """Record the post-write state so consecutive edits see fresh data."""
        self.record_read(path, full_text=full_text, partial=False)

    def forget(self, path: str) -> None:
        """Drop the record (e.g. the file was deleted)."""
        with self._lock:
            self._entries.pop(path, None)

    def has_seen(self, path: str) -> bool:
        with self._lock:
            return path in self._entries

    # ── The gate ────────────────────────────────────────────────────────

    def check_overwrite(self, path: str) -> Optional[str]:
        """
        Guard for tools that replace/append/delete the CONTENT of an existing
        file (write_file, edit_file, append_file, delete_file).

        Returns ``None`` when the write may proceed, or a human/model-readable
        refusal explaining exactly how to proceed.
        """
        if not os.path.exists(path):
            # Creating a brand-new file needs no read-first ceremony.
            return None
        if os.path.isdir(path):
            return None  # directory tools have their own guards
        with self._lock:
            entry = self._entries.get(path)
            if entry is not None:
                self._entries.move_to_end(path)
        if entry is None:
            return (
                f"Refusing to modify '{_display(path)}': the file has not been "
                "read in this session yet. Call read_file on it first so the "
                "edit is grounded in its current contents."
            )
        if entry.partial:
            return (
                f"Refusing to modify '{_display(path)}': only a partial view "
                "(offset/limit) was read. Re-read the whole file first, then "
                "apply the edit."
            )
        return self._check_stale(path, entry)

    def check_edit(self, path: str) -> Optional[str]:
        """Alias of :meth:`check_overwrite` kept for call-site clarity."""
        return self.check_overwrite(path)

    # ── Staleness verification ────────────────────────────────────────

    def _check_stale(self, path: str, entry: _Entry) -> Optional[str]:
        try:
            st = os.stat(path)
        except OSError:
            return None  # vanished → downstream tool reports the real error
        unchanged_meta = (
            st.st_mtime_ns == entry.mtime_ns
            and st.st_size == entry.size
        )
        if unchanged_meta:
            return None
        # Metadata moved — verify against actual content before refusing
        # (Windows mtime granularity can report false changes).
        if entry.full_text is not None:
            current = _read_text(path)
            if current is not None and _hash_text(current) == entry.sha256:
                # Content identical: refresh the cached stat and allow.
                entry.mtime_ns = st.st_mtime_ns
                entry.size = st.st_size
                return None
        return (
            f"Refusing to modify '{_display(path)}': the file changed on disk "
            "since it was last read (modified by another process or editor). "
            "Re-read it to see the current contents, then re-apply your change."
        )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _read_text(path: str) -> Optional[str]:
    """Best-effort UTF-8 text read for content comparison; None on failure."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def _display(path: str) -> str:
    """Show workspace-relative-looking paths when possible."""
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)
    return "/".join(parts[-3:]) if len(parts) > 3 else norm


__all__ = ["ReadStateCache"]
