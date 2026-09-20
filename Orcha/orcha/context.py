"""
orcha.context
=============
Structured context assembly for model calls.

Instead of one giant concatenated string that gets hard-sliced at the end
(which is how long answers and instructions silently disappeared before),
this module gives the pipeline two small, testable helpers:

- :func:`assemble_system_prompt` — joins named prompt parts in *priority*
  order and, when the combined text exceeds the budget, drops or truncates
  the lowest-priority parts first. High-priority instructions always survive.
- :func:`trim_history` — caps a conversation (role/content message list) to
  a character budget, keeping the most recent turns and always keeping the
  oldest turn as anchor context.

Both are pure functions: they never touch the network or the file system.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Default system-prompt budget. Mirrors the frontend's MAX_SYSTEM_PROMPT_LENGTH
# so the agent path and the chat path agree on how much fits in one prompt.
DEFAULT_SYSTEM_PROMPT_MAX = 6800

# The last N characters of a part are reserved for the truncation marker so
# models can tell the context was cut rather than ending mid-sentence.
_TRUNCATION_NOTE = "[context truncated to fit prompt budget]"


def assemble_system_prompt(
    parts: Optional[List[Dict[str, Any]]] = None,
    max_len: int = DEFAULT_SYSTEM_PROMPT_MAX,
) -> str:
    """
    Compose a system prompt from named parts, respecting priority.

    ``parts`` is a list of dicts::

        {"name": "persona", "text": "...", "priority": 100}
        {"name": "instructions", "text": "...", "priority": 80}
        {"name": "workspace", "text": "...", "priority": 20}

    Parts are joined in descending ``priority`` order (higher first). The
    combined text never exceeds ``max_len``:

    - whole low-priority parts are dropped first,
    - the first part that does not fully fit is truncated at its tail,
    - a short ``[context truncated ...]`` marker is appended so models can
      tell the boundary is deliberate.

    A missing/empty ``text`` is skipped. If every part is empty, returns "".
    """
    if not parts:
        return ""

    ordered = sorted(
        (p for p in parts if (p.get("text") or "").strip()),
        key=lambda p: int(p.get("priority", 0)),
        reverse=True,
    )
    if not ordered:
        return ""

    chunks: List[str] = []
    used = 0
    for part in ordered:
        text = part["text"].strip()
        sep = "\n\n" if chunks else ""
        if used + len(sep) + len(text) <= max_len:
            chunks.append(text)
            used += len(sep) + len(text)
            continue
        # This part (or lower-priority parts) no longer fit.
        room = max_len - used - len(sep)
        if room <= 0:
            break
        note = f"\n{_TRUNCATION_NOTE}"
        if room <= len(note) + 20:
            break
        chunks.append(text[: room - len(note)].rstrip() + note)
        used += len(sep) + room
        break

    return "\n\n".join(chunks)


def trim_history(
    messages: Optional[List[Dict[str, Any]]] = None,
    max_len: int = 16000,
    max_turns: int = 40,
) -> List[Dict[str, Any]]:
    """
    Cap a message history to ``max_len`` characters and ``max_turns`` entries.

    Keeps the most recent messages (what the model needs to answer) plus the
    oldest message as a stable conversation anchor, so the model still knows
    what the session is about even after a long chat. Messages are returned
    in original order.
    """
    if not messages:
        return []
    trimmed = list(messages)

    if len(trimmed) > max_turns:
        trimmed = trimmed[:1] + trimmed[-(max_turns - 1):]

    total = sum(len(_msg_text(m) or "") for m in trimmed)
    if total <= max_len:
        return trimmed

    # Drop from the second-oldest towards the newest until we fit, keeping
    # the anchor (oldest) and as much of the recent tail as possible.
    anchor = trimmed[0]
    budget = max_len - len(_msg_text(anchor) or "")
    recent: List[Dict[str, Any]] = []
    for message in reversed(trimmed[1:]):
        text = _msg_text(message)
        if not text:
            recent.append(message)
            continue
        if budget - len(text) < 0:
            continue
        budget -= len(text)
        recent.append(message)
    recent.reverse()
    return [anchor, *recent]


def _msg_text(message: Dict[str, Any]) -> Optional[str]:
    if not isinstance(message, dict):
        return None
    return message.get("content")


__all__ = [
    "DEFAULT_SYSTEM_PROMPT_MAX",
    "assemble_system_prompt",
    "trim_history",
]
