"""Markdown memory store: the file-backed complement to the SQL memory layer.

Format (per user, under ``settings.memory_md_dir``):

    <user_id>/
      MEMORY.md          # capped one-line-per-memory index, injected into
                         # system prompts; deeper recall reads topic files.
      <slug>.md          # one topic file per memory: frontmatter (title,
                         # description, type) + free-form markdown body.
      .cursor.json       # background-extraction cursor (last processed
                         # message id) so extraction resumes exactly once.

Design rules (shared with the Orcha extractor that writes here):
- Index lines are pointers, never content: ``- [Title](file.md) — hook``.
- The index is hard-capped (MAX_INDEX_LINES); when the cap fires the store
  reports ``truncated=True`` so callers can warn the model.
- Topic files are small and typed (user / feedback / project / reference).
- Everything is plain UTF-8 markdown — greppable by both humans and agents.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

MAX_INDEX_LINES = 200
MAX_FILE_BYTES = 64 * 1024
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
MEMORY_TYPES = ("user", "feedback", "project", "reference")

MemoryType = Literal["user", "feedback", "project", "reference"]


class MemoryMdError(Exception):
    """Structured failure for the md layer; message is API-safe."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class IndexLine:
    title: str
    slug: str
    hook: str


def slugify(name: str) -> str:
    """Lowercase ASCII slug safe as a filename component."""
    slug = re.sub(r"[^a-z0-9._-]+", "-", name.strip().lower()).strip("-.")
    return slug[:64] or "memory"


def parse_frontmatter(text: str) -> tuple[Dict[str, str], str]:
    """Minimal frontmatter parser (key: value lines). Returns (meta, body)."""
    match = re.match(r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)", text, re.DOTALL)
    if not match:
        return {}, text
    meta: Dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    return meta, text[match.end():].lstrip("\n")


class MarkdownMemoryStore:
    """File-backed memory for ONE user. All paths are precomputed; every
    write is atomic (temp file + replace)."""

    def __init__(self, base_dir: Path | str, user_id: str) -> None:
        self.user_id = str(user_id)
        self.dir = Path(base_dir) / self.user_id
        self.index_path = self.dir / "MEMORY.md"
        self.cursor_path = self.dir / ".cursor.json"

    # ── Layout ──────────────────────────────────────────────────────────

    def ensure(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def _file(self, slug: str) -> Path:
        if not SLUG_RE.match(slug):
            raise MemoryMdError("invalid_name", f"Invalid memory file name: {slug!r}")
        return self.dir / f"{slug}.md"

    # ── Index ───────────────────────────────────────────────────────────

    def read_index(self) -> tuple[str, bool]:
        """The MEMORY.md content capped at MAX_INDEX_LINES, plus whether
        older lines were cut (the caller should warn the model)."""
        if not self.index_path.exists():
            return "", False
        lines = self.index_path.read_text(encoding="utf-8").splitlines()
        truncated = len(lines) > MAX_INDEX_LINES
        if truncated:
            # Keep the NEWEST entries — oldest pointers fall off first.
            lines = lines[-MAX_INDEX_LINES:]
        return "\n".join(lines).strip(), truncated

    def _upsert_index_line(self, title: str, slug: str, hook: str) -> None:
        """Insert/replace the pointer line for ``slug``, newest last.
        The file itself grows unbounded; the 200-line cap applies at READ
        time so the truncation flag can be reported honestly."""
        line = f"- [{title}]({slug}.md) — {hook}"
        lines: List[str] = []
        if self.index_path.exists():
            lines = [
                ln for ln in self.index_path.read_text(encoding="utf-8").splitlines()
                if f"]({slug}.md)" not in ln
            ]
        lines.append(line)
        self._atomic_write(self.index_path, "\n".join(lines) + "\n")

    def _remove_index_line(self, slug: str) -> None:
        if not self.index_path.exists():
            return
        lines = [
            ln for ln in self.index_path.read_text(encoding="utf-8").splitlines()
            if f"]({slug}.md)" not in ln
        ]
        self._atomic_write(self.index_path, "\n".join(lines) + ("\n" if lines else ""))

    # ── Topic files ─────────────────────────────────────────────────────

    def write_file(
        self,
        name: str,
        *,
        title: str,
        description: str,
        type_: str,
        body: str,
    ) -> Dict[str, Any]:
        if type_ not in MEMORY_TYPES:
            raise MemoryMdError(
                "invalid_type",
                f"type must be one of {', '.join(MEMORY_TYPES)} (got {type_!r})",
            )
        slug = slugify(name)
        self.ensure()
        path = self._file(slug)
        if len(body.encode("utf-8")) > MAX_FILE_BYTES:
            raise MemoryMdError("too_large", "memory body exceeds 64KB")
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        content = (
            "---\n"
            f"title: {title}\n"
            f"description: {description}\n"
            f"type: {type_}\n"
            f"updated: {stamp}\n"
            "---\n\n"
            + body.strip() + "\n"
        )
        self._atomic_write(path, content)
        hook = description or body.strip().splitlines()[0][:80] if body.strip() else title
        self._upsert_index_line(title, slug, hook)
        return {"name": slug, "title": title, "type": type_, "bytes": len(content)}

    def read_file(self, name: str) -> Dict[str, Any]:
        path = self._file(slugify(name))
        if not path.exists():
            raise MemoryMdError("not_found", f"No memory named {name!r}")
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        return {
            "name": slugify(name),
            "title": meta.get("title", ""),
            "description": meta.get("description", ""),
            "type": meta.get("type", ""),
            "updated": meta.get("updated", ""),
            "body": body,
        }

    def delete_file(self, name: str) -> bool:
        slug = slugify(name)
        path = self._file(slug)
        existed = path.exists()
        if existed:
            path.unlink()
            self._remove_index_line(slug)
        return existed

    def list_files(self) -> List[Dict[str, Any]]:
        if not self.dir.exists():
            return []
        out: List[Dict[str, Any]] = []
        for path in sorted(self.dir.glob("*.md")):
            if path.name == "MEMORY.md":
                continue
            meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
            out.append(
                {
                    "name": path.stem,
                    "title": meta.get("title", ""),
                    "description": meta.get("description", ""),
                    "type": meta.get("type", ""),
                    "updated": meta.get("updated", ""),
                    "size": path.stat().st_size,
                }
            )
        return out

    # ── Recall (deterministic pre-filter; LLM selection lives upstream) ──

    def recall_manifest(self, query: str, *, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Frontmatter-scan recall candidates ranked by cheap term overlap +
        recency. Returns at most ``limit`` entries WITHOUT bodies — the
        caller (Orcha selector / prompt builder) picks winners and reads
        those files explicitly. Deterministic: same store state + query →
        same ranking.
        """
        terms = [t.casefold() for t in re.findall(r"[a-z0-9]{3,}", query)]
        scored: List[tuple[float, Dict[str, Any]]] = []
        now = time.time()
        for item in self.list_files():
            hay = " ".join(
                (item["name"], item["title"], item["description"], item["type"])
            ).casefold()
            overlap = sum(1 for t in terms if t in hay)
            try:
                updated = time.mktime(time.strptime(item["updated"], "%Y-%m-%dT%H:%M:%SZ"))
                age_days = max(0.0, (now - updated) / 86400)
            except (TypeError, ValueError):
                age_days = 30.0
            score = overlap * 2.0 + max(0.0, 1.0 - age_days / 90.0)
            if overlap or not terms:
                scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], str(pair[1]["name"])))
        return [item for _, item in scored[:limit]]

    # ── Extraction cursor ───────────────────────────────────────────────

    def get_cursor(self) -> Dict[str, Any]:
        if not self.cursor_path.exists():
            return {"last_message_id": None}
        try:
            return json.loads(self.cursor_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"last_message_id": None}

    def set_cursor(self, last_message_id: Optional[str]) -> None:
        self.ensure()
        self._atomic_write(
            self.cursor_path,
            json.dumps({"last_message_id": last_message_id}),
        )

    # ── Internals ───────────────────────────────────────────────────────

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        import os
        import tempfile

        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(content)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


__all__ = [
    "MarkdownMemoryStore", "MemoryMdError", "IndexLine",
    "MAX_INDEX_LINES", "MAX_FILE_BYTES", "MEMORY_TYPES",
    "slugify", "parse_frontmatter",
]
