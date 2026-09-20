"""
orcha.agent_runtime.compaction
==============================
Layered context management for long agent runs (the "compaction ladder"),
implemented as PURE PROJECTIONS over the EventLog — the authoritative log
is never mutated, so deterministic replay is unaffected.

The ladder, cheapest step first (checked per model turn):

1. **Do nothing** — while the turn's projected context fits comfortably
   inside the configured window, nothing happens at all.
2. **Microcompact** — strip the CONTENT of old successful tool results
   (read-only tools only), replacing each with ``[older tool result
   cleared]``. Action/Observation pairing stays valid because payloads are
   replaced, not removed. Newest steps are protected.
3. **Autocompact** — when the projection STILL exceeds the hard threshold,
   summarize the whole turn into a structured brief (9 sections) via one
   extra backend call and carry it as ``ContextMemory.summary`` instead of
   the raw slice; the full pre-compaction transcript is dumped to disk as
   JSONL first, so nothing is ever lost ("the full transcript is at …").
4. **Circuit breaker** — after ``max_consecutive_failures`` failed summary
   attempts the compactor stands down for the rest of the session and the
   legacy char-truncation in the agent takes over (graceful degradation).

Everything here mirrors the runtime's existing conventions: token counts
use the same charged accounting (``tokens + reasoning_tokens``), estimates
use :func:`~orcha.agent_runtime.events.estimate_tokens`, configs are
frozen pydantic models, and every failure degrades — never raises into
the agent loop.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from .events import (
    Event, EventKind, FinishAction, MessageAction,
    ToolCallAction, ToolResultObservation, UserMessageObservation,
    estimate_tokens,
)
from .memory import ContextMemory

logger = logging.getLogger("orcha.agent_runtime.compaction")

# Tools whose SUCCESSFUL results are safe to microcompact: re-readable,
# read-only lookups. Anything that wrote, executed or errored stays.
COMPACTABLE_TOOLS = frozenset({
    "read_file", "read_multiple_files", "list_directory", "directory_tree",
    "glob_search", "search_text", "file_info", "exists",
})

_CLEARED_MARK = "[older tool result cleared to save context]"


class CompactionConfig(BaseModel):
    """Immutable thresholds for the ladder. All values are TOKENS using the
    runtime's charged accounting."""
    model_config = ConfigDict(frozen=True)

    # Soft target: microcompact once the projected turn exceeds this.
    soft_limit_tokens: int = Field(default=18_000, ge=10)
    # Hard trigger: autocompact (summarize) once still above this after
    # microcompaction. Defaults keep ~4k headroom under a 24k window.
    hard_limit_tokens: int = Field(default=20_000, ge=10)
    # Newest ACTION/OBSERVATION pairs never touched by microcompaction.
    protected_tail_pairs: int = Field(default=3, ge=0)
    # Circuit breaker: give up for the session after N failed summaries.
    max_consecutive_failures: int = Field(default=3, ge=1)
    # Minimum chars a transcript must reach before summarization is worth a
    # call (guards tiny slices from burning tokens on summaries).
    min_summary_chars: int = Field(default=2_000, ge=0)


def estimate_slice_tokens(events: Sequence[Event]) -> int:
    """
    Charged-token estimate of an event sequence: the SAME accounting the
    budget cutoffs use (tokens + reasoning), falling back to a character
    estimate for zero-charged events so un-metered backends still trip the
    ladder.
    """
    total = 0
    for ev in events:
        charged = ev.tokens + ev.reasoning_tokens
        if charged > 0:
            total += charged
            continue
        payload = ev.payload
        text = getattr(payload, "content", None) or getattr(payload, "message", None)
        if text is None:
            if isinstance(payload, ToolCallAction):
                text = payload.name + json.dumps(payload.arguments, default=str)
            else:
                text = payload.model_dump_json() if hasattr(payload, "model_dump_json") else str(payload)
        total += estimate_tokens(text)
    return total


# ── Rung 2: microcompact ─────────────────────────────────────────────────────

def _pairs(events: Sequence[Event]) -> List[Tuple[Optional[int], int]]:
    """Index action→observation pairs: list of (action_idx_or_None, obs_idx)."""
    pairs: List[Tuple[Optional[int], int]] = []
    last_action: Optional[int] = None
    for i, ev in enumerate(events):
        if ev.kind == EventKind.ACTION:
            last_action = i
        elif isinstance(ev.payload, ToolResultObservation):
            pairs.append((last_action, i))
            last_action = None
    return pairs


def microcompact(
    events: Sequence[Event],
    *,
    protected_tail_pairs: int = 3,
    compactable_tools: Optional[frozenset] = None,
) -> Tuple[Tuple[Event, ...], int, int]:
    """
    Return ``(projected_events, cleared_count, freed_tokens)`` where old
    COMPACTABLE tool results have their content replaced by a placeholder.
    Pure: input events are never mutated; output copies are fresh objects.

    An observation is compactable when its producing action names a
    read-only lookup tool and the result succeeded. Errors, writes,
    execution results and the newest ``protected_tail_pairs`` results are
    always kept verbatim.
    """
    tools = compactable_tools or COMPACTABLE_TOOLS
    pairs = _pairs(events)
    if not pairs:
        return tuple(events), 0, 0

    # Mark all but the newest N pairs for clearing.
    clear_obs_indices: set = set()
    for action_idx, obs_idx in pairs[:-protected_tail_pairs] if protected_tail_pairs else pairs:
        obs = events[obs_idx].payload
        assert isinstance(obs, ToolResultObservation)
        if not obs.success:
            continue
        if action_idx is not None:
            actor = events[action_idx].payload
            if isinstance(actor, ToolCallAction) and actor.name in tools:
                clear_obs_indices.add(obs_idx)

    if not clear_obs_indices:
        return tuple(events), 0, 0

    out: List[Event] = []
    cleared = 0
    freed = 0
    for i, ev in enumerate(events):
        if i in clear_obs_indices:
            obs = ev.payload
            assert isinstance(obs, ToolResultObservation)
            freed += max(ev.tokens, estimate_tokens(obs.content))
            replacement = obs.model_copy(update={"content": _CLEARED_MARK})
            out.append(ev.model_copy(update={"payload": replacement}))
            cleared += 1
        else:
            out.append(ev.model_copy(deep=True))
    return tuple(out), cleared, freed


# ── Transcript escape hatch ──────────────────────────────────────────────────

class TranscriptStore:
    """
    Dump full event sequences to JSONL before any lossy summarization, so
    the pre-compaction record always survives on disk. Files live in one
    directory and are named by wall-clock + monotonic counter.
    """

    def __init__(self, directory: str) -> None:
        self.directory = directory
        self._counter = 0

    def dump(self, events: Sequence[Event]) -> str:
        """Write ``events`` as JSONL; returns the absolute path."""
        os.makedirs(self.directory, exist_ok=True)
        self._counter += 1
        path = os.path.join(
            self.directory,
            f"transcript_{int(time.time())}_{self._counter:06d}.jsonl",
        )
        fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
                for ev in events:
                    fh.write(json.dumps(_event_to_dict(ev), ensure_ascii=False, default=str))
                    fh.write("\n")
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path


def _event_to_dict(ev: Event) -> Dict[str, Any]:
    return {
        "seq": ev.seq,
        "kind": ev.kind.value if hasattr(ev.kind, "value") else str(ev.kind),
        "ts": ev.ts,
        "tokens": ev.tokens,
        "reasoning_tokens": ev.reasoning_tokens,
        "payload": ev.payload.model_dump(),
    }


# ── Rung 3: structured summarization ────────────────────────────────────────

_SUMMARY_PROMPT = """You are compressing an agent work session into a handoff brief so work can continue in a much smaller context. Write the brief directly — no preamble.

<analysis>
First think privately about what matters: the user's actual goal, what was tried, what worked, what failed and why, and exactly what remains. Draft your notes here.
</analysis>

<summary>
1. Primary request: what the user asked for, in one or two sentences, quoting their wording for key constraints.
2. Key technical concepts: technologies, files, APIs and domain terms the work depends on.
3. Files and code sections: every file that was read or modified, with VERBATIM snippets of the critical lines (paths exact).
4. Errors and fixes: each error, its cause, and the fix that worked — including anything the user corrected.
5. Problem solving: the approaches attempted and their outcomes.
6. All user messages: every user message, condensed but complete in intent, in order.
7. Pending tasks: explicitly unfinished items (say "none" if truly none).
8. Current work: precisely what was happening immediately before this point.
9. Next step: the single concrete next action, quoting any relevant instruction verbatim.
</summary>"""


class Compactor:
    """
    Owns the ladder's decisions and the (optional) LLM-backed rung.

    ``backend`` may be None for a projections-only compactor: microcompact
    and transcript dumps still work, autocompact then degrades to a
    mechanical first-events digest instead of an LLM call.
    """

    def __init__(
        self,
        backend: Any = None,
        *,
        config: Optional[CompactionConfig] = None,
        transcript_store: Optional[TranscriptStore] = None,
        max_summary_chars: int = 6_000,
    ) -> None:
        self.backend = backend
        self.config = config or CompactionConfig()
        self.transcripts = transcript_store
        self.max_summary_chars = max(500, int(max_summary_chars))
        self._consecutive_failures = 0
        self.disabled = False  # circuit breaker state (session-scoped)

    # ── Introspection ───────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return not self.disabled

    def should_microcompact(self, events: Sequence[Event]) -> bool:
        return (
            self.available
            and estimate_slice_tokens(events) > self.config.soft_limit_tokens
        )

    def should_autocompact(self, projected: Sequence[Event]) -> bool:
        return (
            self.available
            and self.backend is not None
            and estimate_slice_tokens(projected) > self.config.hard_limit_tokens
        )

    # ── The ladder ──────────────────────────────────────────────────────

    def project(self, events: Sequence[Event]) -> Tuple[Tuple[Event, ...], Dict[str, Any]]:
        """
        Microcompact rung (pure, synchronous). Returns the projected events
        plus stats. Safe to call every turn: below the soft limit it is a
        no-op returning the input unchanged.
        """
        stats: Dict[str, Any] = {}
        if not self.should_microcompact(events):
            return tuple(events), stats
        projected, cleared, freed = microcompact(
            events, protected_tail_pairs=self.config.protected_tail_pairs,
        )
        stats.update(microcompacted=cleared, tokens_freed=freed)
        return projected, stats

    async def autocompact(
        self, projected: Sequence[Event],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Summarize a (usually microcompacted) slice. Returns
        ``(summary_text_or_None, transcript_path_or_None)``. On failure the
        circuit breaker ticks and None is returned — callers keep the raw
        projection and the agent's own truncation handles the rest.
        """
        if self.disabled or self.backend is None:
            return None, None
        rendered = _render_for_summary(projected)
        if len(rendered) < self.config.min_summary_chars:
            return None, None
        transcript_path: Optional[str] = None
        if self.transcripts is not None:
            try:
                transcript_path = self.transcripts.dump(projected)
            except OSError as exc:
                logger.warning("transcript dump failed: %s", exc)
        try:
            from .agent import AgentConfig
            response = await self.backend.complete(
                [
                    {"role": "system", "content": _SUMMARY_PROMPT},
                    {"role": "user", "content": rendered},
                ],
                tools=None,
                config=AgentConfig(model=getattr(self.backend, "model_name", "") or "local"),
            )
            summary = (getattr(response, "text", "") or "").strip()
            if not summary:
                raise ValueError("empty summary response")
            if "<summary>" in summary:
                start = summary.index("<summary>")
                end = summary.rindex("</summary>") + len("</summary>")
                inner = summary[start:end]
                summary = inner.replace("<summary>", "").replace("</summary>", "").strip()
            if len(summary) > self.max_summary_chars:
                summary = summary[: self.max_summary_chars] + "\n… (summary truncated)"
            self._consecutive_failures = 0
            suffix = (
                f"\n\n[Full pre-compaction transcript saved to: {transcript_path}]"
                if transcript_path else ""
            )
            return summary + suffix, transcript_path
        except Exception as exc:
            self._consecutive_failures += 1
            logger.warning(
                "autocompact summary failed (%s/%s): %s",
                self._consecutive_failures, self.config.max_consecutive_failures, exc,
            )
            if self._consecutive_failures >= self.config.max_consecutive_failures:
                self.disabled = True
                logger.warning("compactor disabled for this session (circuit breaker open)")
            return None, transcript_path


def _render_for_summary(events: Sequence[Event]) -> str:
    """Deterministic text render of a slice for the summarizer."""
    lines: List[str] = []
    for ev in events:
        p = ev.payload
        if isinstance(p, UserMessageObservation):
            lines.append(f"USER: {p.content}")
        elif isinstance(p, ToolCallAction):
            args = json.dumps(p.arguments, ensure_ascii=False, default=str)
            if len(args) > 400:
                args = args[:400] + "…"
            lines.append(f"AGENT tool_call {p.name}({args})")
        elif isinstance(p, ToolResultObservation):
            content = p.content if p.success else f"ERROR: {p.content}"
            if len(content) > 1_200:
                content = content[:1_200] + f"… (+{len(content) - 1200} chars)"
            lines.append(f"TOOL{' OK' if p.success else ' ERR'}: {content}")
        elif isinstance(p, MessageAction):
            body = p.content if len(p.content) <= 2_000 else p.content[:2_000] + "…"
            lines.append(f"ASSISTANT: {body}")
        elif isinstance(p, FinishAction):
            lines.append(f"FINISH ({p.reason}): {p.content}")
        else:
            lines.append(str(getattr(p, "message", "") or p.kind))
    return "\n".join(lines)


__all__ = [
    "CompactionConfig", "Compactor", "TranscriptStore",
    "COMPACTABLE_TOOLS", "estimate_slice_tokens", "microcompact",
]
