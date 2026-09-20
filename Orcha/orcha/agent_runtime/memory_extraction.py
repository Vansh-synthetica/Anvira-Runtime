"""
orcha.agent_runtime.memory_extraction
=====================================
Background memory extraction — the fire-and-forget pass that distills a
finished agent turn into durable markdown memories (the file-format
sibling of Nomi's ``/api/v1/memory/md`` layer; the same files work with
either).

Design (mirroring proven local-first agent memory systems):

- **Runs after a completed turn**, off the critical path: the caller
  decides scheduling (e.g. ``asyncio.create_task``); this module only
  provides the coroutine.
- **Cursor-based**: the store remembers the last event ``seq`` processed;
  extraction covers exactly the unprocessed suffix and advances the
  cursor ONLY on success, so a crash never loses or duplicates work.
- **Mutual exclusion**: if the MAIN agent already wrote to the memory
  directory during the turn, extraction skips — dual writers would race.
- **Restricted scope**: extraction is ONE structured backend call with a
  fixed prompt (taxonomy + "what NOT to save"), not an agentic loop —
  cheap, bounded, and deterministic to retry.

The output is plain data (validated dicts); persistence is delegated to a
writer callable so the same extractor feeds Nomi's HTTP API, a local
:class:`MarkdownMemoryWriter`, or tests.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .events import Event, EventKind, MessageAction, ToolCallAction, UserMessageObservation

logger = logging.getLogger("orcha.agent_runtime.memory_extraction")

MEMORY_TYPES = ("user", "feedback", "project", "reference")
_WRITE_TOOL_NAMES = frozenset({
    "write_file", "edit_file", "append_file", "create_file", "replace_text", "apply_patch",
})

_EXTRACT_SYSTEM_PROMPT = """You extract durable long-term memories from an agent work session.

Save ONLY knowledge worth keeping across sessions, typed as one of:
- user: who the person is (role, preferences, goals, constraints they stated)
- feedback: corrections or guidance the user gave about how to work
- project: durable facts about the project (stack, structure, decisions, conventions)
- reference: pointers to where knowledge lives (files, docs, commands)

What NOT to save:
- anything derivable from the code or git history (fix recipes, error traces)
- transient task state, intermediate results, or step-by-step narration
- secrets, tokens, or credentials of any kind

Respond with ONLY a JSON object:
{"memories": [{"type": "...", "name": "short_snake_case_slug",
               "title": "Short Title", "description": "one-line hook",
               "body": "markdown body, concise and factual"}]}
Return {"memories": []} when nothing qualifies."""


@dataclass
class MemoryExtractionConfig:
    """Bounded, conservative defaults."""
    max_transcript_chars: int = 24_000
    max_memories_per_run: int = 5
    max_body_chars: int = 4_000
    # Skip when fewer than this many user messages arrived since the cursor.
    min_new_user_messages: int = 1


class MemoryExtractor:
    """
    One-shot extraction over an event suffix. ``backend`` is any Orcha
    ModelBackend (the SAME instance the main agent uses, so cached prefix
    tokens make the extra call cheap).
    """

    def __init__(
        self,
        backend: Any,
        *,
        config: Optional[MemoryExtractionConfig] = None,
        writer: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
        memory_dir_hint: str = "memory",
    ) -> None:
        self.backend = backend
        self.config = config or MemoryExtractionConfig()
        self.writer = writer
        self.memory_dir_hint = memory_dir_hint.casefold()

    # ── Decision helpers (pure, unit-testable) ──────────────────────────

    @staticmethod
    def last_seq(events: Sequence[Event]) -> int:
        return events[-1].seq if events else 0

    def has_new_work(self, events: Sequence[Event], cursor_seq: int) -> bool:
        new_user = sum(
            1 for ev in events
            if ev.seq > cursor_seq and isinstance(ev.payload, UserMessageObservation)
        )
        return new_user >= self.config.min_new_user_messages

    def agent_wrote_memory(self, events: Sequence[Event], cursor_seq: int = 0) -> bool:
        """
        Mutual exclusion: did the main agent itself touch memory files this
        turn? Checks write-capable tool calls whose arguments mention the
        memory directory hint (path/command text).
        """
        for ev in events:
            if ev.seq <= cursor_seq or ev.kind != EventKind.ACTION:
                continue
            action = ev.payload
            if not isinstance(action, ToolCallAction) or action.name not in _WRITE_TOOL_NAMES:
                continue
            haystack = json.dumps(action.arguments, default=str).casefold()
            if self.memory_dir_hint in haystack:
                return True
        return False

    # ── Rendering ───────────────────────────────────────────────────────

    def render_transcript(self, events: Sequence[Event], cursor_seq: int) -> str:
        lines: List[str] = []
        budget = self.config.max_transcript_chars
        for ev in events:
            if ev.seq <= cursor_seq:
                continue
            payload = ev.payload
            if isinstance(payload, UserMessageObservation):
                lines.append(f"USER: {payload.content}")
            elif isinstance(payload, MessageAction):
                lines.append(f"ASSISTANT: {payload.content}")
            elif isinstance(payload, ToolCallAction):
                args = json.dumps(payload.arguments, ensure_ascii=False, default=str)
                if len(args) > 200:
                    args = args[:200] + "…"
                lines.append(f"AGENT tool_call {payload.name}({args})")
            if sum(len(ln) for ln in lines) > budget:
                break
        text = "\n".join(lines)
        return text[:budget]

    # ── The pass ────────────────────────────────────────────────────────

    async def run(
        self, events: Sequence[Event], cursor_seq: int,
    ) -> Dict[str, Any]:
        """
        Extract from everything after ``cursor_seq``. Returns::

            {"skipped": reason?, "memories": [...], "cursor": new_seq}

        Never raises into the caller's loop: every failure degrades to a
        skipped run with the cursor unchanged.
        """
        result: Dict[str, Any] = {"memories": [], "cursor": cursor_seq}
        if not self.has_new_work(events, cursor_seq):
            result["skipped"] = "no_new_user_messages"
            return result
        if self.agent_wrote_memory(events, cursor_seq):
            result["skipped"] = "main_agent_already_wrote_memory"
            result["cursor"] = self.last_seq(events)  # advance past safely
            return result

        transcript = self.render_transcript(events, cursor_seq)
        if not transcript.strip():
            result["skipped"] = "empty_transcript"
            return result

        try:
            from .agent import AgentConfig
            response = await self.backend.complete(
                [
                    {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
                    {"role": "user", "content": transcript},
                ],
                tools=None,
                config=AgentConfig(
                    model=getattr(self.backend, "model_name", "") or "local",
                ),
            )
            raw = getattr(response, "text", "") or ""
            memories = parse_memories_payload(
                raw,
                max_memories=self.config.max_memories_per_run,
                max_body_chars=self.config.max_body_chars,
            )
        except Exception as exc:
            logger.warning("memory extraction failed (cursor unchanged): %s", exc)
            result["skipped"] = f"extraction_error: {exc}"
            return result

        result["memories"] = memories
        result["cursor"] = self.last_seq(events)  # advance ONLY on success
        if memories and self.writer is not None:
            try:
                self.writer(memories)
            except Exception as exc:
                # Memories were extracted but persisting failed: keep the
                # returned data so the caller can retry persistence.
                logger.warning("memory writer failed: %s", exc)
                result["write_failed"] = str(exc)
        return result


# ── Response parsing (pure) ──────────────────────────────────────────────────

def parse_memories_payload(
    raw: str,
    *,
    max_memories: int = 5,
    max_body_chars: int = 4_000,
) -> List[Dict[str, Any]]:
    """
    Robustly parse the model's JSON answer into validated memory dicts.
    Tolerates fenced blocks and surrounding prose; drops invalid entries;
    enforces type vocabulary and size caps. Deterministic.
    """
    payload = _loads_loose(raw)
    if not isinstance(payload, dict):
        return []
    items = payload.get("memories")
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    seen_names: set = set()
    for item in items[: max_memories * 3]:
        if len(out) >= max_memories:
            break
        if not isinstance(item, dict):
            continue
        type_ = str(item.get("type") or "")
        if type_ not in MEMORY_TYPES:
            continue
        name = re.sub(r"[^a-z0-9_-]+", "-", str(item.get("name") or "").lower()).strip("-")
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        title = str(item.get("title") or name)[:120]
        description = str(item.get("description") or "")[:300]
        body = str(item.get("body") or "").strip()[:max_body_chars]
        if not body:
            continue
        out.append(
            {
                "type": type_,
                "name": name,
                "title": title,
                "description": description,
                "body": body,
            }
        )
    return out


def _loads_loose(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


# ── Local markdown writer (same format as Nomi's md layer) ──────────────────

class MarkdownMemoryWriter:
    """
    Direct-to-disk writer implementing the shared format: per-user dir with
    topic files + capped MEMORY.md index. Used when Anvira/Orcha persists
    locally without going through Nomi's HTTP API.
    """

    MAX_INDEX_LINES = 200

    def __init__(self, base_dir: Path | str, user_id: str = "local") -> None:
        self.dir = Path(base_dir) / user_id
        self.index_path = self.dir / "MEMORY.md"

    def __call__(self, memories: List[Dict[str, Any]]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for mem in memories:
            slug = re.sub(r"[^a-z0-9._-]+", "-", str(mem.get("name", ""))).strip("-.") or "memory"
            content = (
                "---\n"
                f"title: {mem.get('title', slug)}\n"
                f"description: {mem.get('description', '')}\n"
                f"type: {mem.get('type', 'project')}\n"
                f"updated: {stamp}\n"
                "---\n\n" + str(mem.get("body", "")).strip() + "\n"
            )
            target = self.dir / f"{slug}.md"
            tmp = target.with_suffix(".tmp")
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(target)
            line = (
                f"- [{mem.get('title', slug)}]({slug}.md) — "
                f"{mem.get('description', '') or mem.get('title', '')}"
            )
            existing: List[str] = []
            if self.index_path.exists():
                existing = [
                    ln for ln in self.index_path.read_text(encoding="utf-8").splitlines()
                    if f"]({slug}.md)" not in ln
                ]
            existing.append(line)
            self.index_path.write_text("\n".join(existing) + "\n", encoding="utf-8")


__all__ = [
    "MemoryExtractor", "MemoryExtractionConfig", "MarkdownMemoryWriter",
    "parse_memories_payload", "MEMORY_TYPES",
]
