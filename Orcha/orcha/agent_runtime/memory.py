"""
orcha.agent_runtime.memory
==========================
Working memory for the AgentRuntime loop (clean-room design, grounded in
the externally observable shape of local-first agent memory subsystems —
smolagents' step memory is the reference for the short-term side; the
durable-facts store convention used across local agent frameworks is the
reference for the long-term side. No implementation detail is shared with
any specific project).

The split (short-term vs long-term)
-----------------------------------
- SHORT-TERM step memory: one compact, structured record per agent step —
  the action taken, the observation received, wall-clock timing, token
  usage in/out and the optional reasoning-token share. Cheap to
  re-serialize into the next prompt (the "succinct steps" pattern). This
  is the agent's working memory across steps of the CURRENT session; it
  is a pure projection of the EventLog, so it is deterministic and
  replayable — no new mutable state.
- LONG-TERM durable memory: explicit, longer-lived facts and preferences
  (user-stated or extracted), stored separately from steps and re-injected
  into every step's context as a compact section. In Orcha these facts are
  EVENTS (FactObservation) — the EventLog stays the only authoritative
  state, and replay reconstructs the same durable memory.

Bounded context
---------------
On every step the model sees a WINDOWED SUMMARY of recent steps, not the
raw EventLog: ``render_window`` walks the step records newest-first,
charging each step ``tokens_in + reasoning_tokens + tokens_out`` against
``max_context_tokens`` (the SAME accounting as the budget: reasoning
tokens are charged to the window whether or not the backend reports them
— absence simply means zero). Steps that do not fit fold into one digest
line with totals, so the context size is bounded no matter how long the
session runs.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from .events import (
    Event, EventKind, ToolResultObservation, ErrorObservation,
    UserMessageObservation, FactObservation,
)


# ── Short-term: one compact step record (smolagents ActionStep shape) ────────

class StepRecord(BaseModel):
    """
    One agent step, serialized compactly for re-injection into the next
    prompt: action taken, observation received, wall-clock timing, token
    usage in/out, plus the optional reasoning-token share.
    """
    model_config = ConfigDict(frozen=True)

    step_number: int = 0                # 1-based index of the agent step
    action: str = ""                    # compact action text
    action_kind: str = ""               # tool_call | message | finish | error
    observation: str = ""               # tool result / error text (short)
    observation_kind: str = ""          # tool_result | error | ""
    success: bool = True                # did the step's work succeed
    error: str = ""                     # error message when the step failed
    duration_ms: float = 0.0            # wall-clock timing (event-to-event)
    tokens_in: int = 0                  # action token cost (prompt+completion)
    reasoning_tokens: int = 0           # optional reasoning-token share
    tokens_out: int = 0                 # observation token cost

    @property
    def charge(self) -> int:
        """Token cost this step contributes to the context window — the
        SAME accounting as the budget (reasoning folds in, absence = 0)."""
        return self.tokens_in + self.reasoning_tokens + self.tokens_out

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable compact dict (get_succinct_steps analog)."""
        return {
            "step": self.step_number,
            "action": self.action,
            "observation": self.observation,
            "success": self.success,
            "error": self.error or None,
            "duration_ms": round(self.duration_ms, 1),
            "tokens_in": self.tokens_in,
            "reasoning_tokens": self.reasoning_tokens,
            "tokens_out": self.tokens_out,
        }

    def to_line(self) -> str:
        """One-line render for the prompt window."""
        line = f"step {self.step_number}: {self.action}"
        if self.observation:
            line += f" -> {self.observation}"
        if self.error:
            line += f" [error: {self.error}]"
        return line


class StepMemory:
    """
    Short-term working memory: a pure projection of the EventLog onto
    compact step records. Same events in → same records out (deterministic,
    replayable, zero side effects).
    """

    def __init__(self, records: Sequence[StepRecord] = ()) -> None:
        self._records: Tuple[StepRecord, ...] = tuple(records)

    @property
    def records(self) -> Tuple[StepRecord, ...]:
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    @classmethod
    def from_events(cls, events: Sequence[Event]) -> "StepMemory":
        """
        Fold events into step records. An ACTION event opens a record; the
        next OBSERVATION event (if any) fills in the result. Wall-clock
        timing comes from the event timestamps (event-to-event duration).
        """
        records: List[StepRecord] = []
        step_number = 0
        pending: Optional[Dict[str, Any]] = None

        for i, ev in enumerate(events):
            if ev.kind == EventKind.ACTION:
                pending = {
                    "step_number": step_number + 1,
                    "action": _action_text(ev.payload),
                    "action_kind": ev.payload.kind,
                    "tokens_in": ev.tokens,
                    "reasoning_tokens": ev.reasoning_tokens,
                    "ts": ev.ts,
                }
                step_number += 1
                continue

            if pending is not None and ev.kind == EventKind.OBSERVATION:
                rec = _observe(pending, ev)
                next_ts = events[i + 1].ts if i + 1 < len(events) else None
                if next_ts is not None:
                    rec["duration_ms"] = max(0.0, (next_ts - ev.ts) * 1000.0)
                records.append(StepRecord(**rec))
                pending = None

        if pending is not None:
            # The action has no observation yet (e.g. the current in-flight
            # step): record it as-is.
            records.append(StepRecord(**pending))

        return cls(records)

    def extend(self, new_events: Sequence[Event]) -> "StepMemory":
        """
        Incrementally extend existing step records with new events.
        Only processes events not already seen.
        """
        if not new_events:
            return self
        records = list(self._records)
        step_number = len(records)
        pending: Optional[Dict[str, Any]] = None
        # Check if the last record is an in-flight step (no observation yet).
        if records and not records[-1].observation:
            pending = {
                "step_number": records[-1].step_number,
                "action": records[-1].action,
                "action_kind": records[-1].action_kind,
                "tokens_in": records[-1].tokens_in,
                "reasoning_tokens": records[-1].reasoning_tokens,
                "ts": 0.0,  # will be filled from event
            }
            records = records[:-1]
            step_number -= 1

        for i, ev in enumerate(new_events):
            if ev.kind == EventKind.ACTION:
                pending = {
                    "step_number": step_number + 1,
                    "action": _action_text(ev.payload),
                    "action_kind": ev.payload.kind,
                    "tokens_in": ev.tokens,
                    "reasoning_tokens": ev.reasoning_tokens,
                    "ts": ev.ts,
                }
                step_number += 1
                continue

            if pending is not None and ev.kind == EventKind.OBSERVATION:
                rec = _observe(pending, ev)
                next_ts = new_events[i + 1].ts if i + 1 < len(new_events) else None
                if next_ts is not None:
                    rec["duration_ms"] = max(0.0, (next_ts - ev.ts) * 1000.0)
                records.append(StepRecord(**rec))
                pending = None

        if pending is not None:
            records.append(StepRecord(**pending))

        return StepMemory(records)

    # ── Windowed summarization ────────────────────────────────────────

    def render_window(
        self,
        max_context_tokens: int,
        *,
        render_step=StepRecord.to_line,
    ) -> str:
        """
        Render a WINDOWED SUMMARY of the recent steps: newest first, each
        step charged ``tokens_in + reasoning_tokens + tokens_out`` against
        ``max_context_tokens``. Steps that do not fit fold into one digest
        line with totals — the output is bounded no matter how long the
        session is.

        Always includes at least the newest step (graceful degradation on
        tiny budgets) and is deterministic: same records, same render.
        """
        if not self._records:
            return ""
        lines: List[str] = []
        used = 0
        omitted = 0
        omitted_in = omitted_reasoning = omitted_out = 0

        for rec in reversed(self._records):
            charge = rec.charge
            if lines and used + charge > max_context_tokens:
                omitted += 1
                omitted_in += rec.tokens_in
                omitted_reasoning += rec.reasoning_tokens
                omitted_out += rec.tokens_out
                continue
            lines.append(render_step(rec))
            used += charge

        head = "\n".join(reversed(lines))
        if omitted:
            head += (
                f"\n… {omitted} earlier step(s) omitted (tokens_in={omitted_in}, "
                f"reasoning={omitted_reasoning}, tokens_out={omitted_out})"
            )
        return head


def _action_text(action: Any) -> str:
    kind = action.kind
    if kind == "tool_call":
        return f"tool call: {action.name}({action.arguments})"
    if kind in ("message", "finish"):
        content = getattr(action, "content", "") or ""
        return f"assistant: {content}"
    return f"[{kind}]: {getattr(action, 'message', '')}"


def _observe(pending: Dict[str, Any], ev: Event) -> Dict[str, Any]:
    obs = ev.payload
    kind = obs.kind
    rec = dict(pending)
    rec["tokens_out"] = ev.tokens
    if isinstance(obs, ToolResultObservation):
        rec["observation"] = obs.content
        rec["observation_kind"] = "tool_result"
        rec["success"] = bool(obs.success)
    elif isinstance(obs, ErrorObservation):
        rec["observation"] = obs.message
        rec["observation_kind"] = "error"
        rec["success"] = False
        rec["error"] = obs.message
    else:
        rec["observation"] = getattr(obs, "content", "") or ""
        rec["observation_kind"] = kind
    return rec


# ── Long-term: durable facts and preferences ─────────────────────────────────

class LongTermMemory:
    """
    Durable, longer-lived facts/preferences, kept SEPARATE from the step
    memory: a compact list of short declarative facts (user-stated or
    developer-provided) that is re-injected into every step's context.

    In Orcha the facts are EVENTS (FactObservation) on the EventLog — the
    only authoritative state — so this object is a pure projection:
    ``from_events`` rebuilds identical memory from the same events
    (replay-safe), and ``to_dict``/``from_dict`` allow cross-session
    persistence as plain JSON (no new file format).
    """

    def __init__(
        self, facts: Sequence[Dict[str, Any]] = (), *, max_facts: int = 50,
    ) -> None:
        self._facts: List[Dict[str, Any]] = []
        self.max_facts = max_facts
        seen: set = set()
        for fact in facts:
            text = str(fact.get("text", "")).strip()
            if not text or text in seen:
                continue
            if len(self._facts) >= self.max_facts:
                break
            seen.add(text)
            self._facts.append(dict(fact))

    @property
    def facts(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(self._facts)

    @property
    def texts(self) -> Tuple[str, ...]:
        return tuple(str(f["text"]) for f in self._facts)

    def __len__(self) -> int:
        return len(self._facts)

    @classmethod
    def from_events(cls, events: Sequence[Event], *, max_facts: int = 50) -> "LongTermMemory":
        """Project the log's FactObservation events onto durable memory,
        deduplicated (exact repeats collapse) and capped at ``max_facts``."""
        facts = [
            {"text": str(ev.payload.content), "source_seq": ev.seq, "ts": ev.ts}
            for ev in events
            if ev.kind == EventKind.OBSERVATION
            and isinstance(ev.payload, FactObservation)
        ]
        return cls(facts, max_facts=max_facts)

    def extend(self, new_events: Sequence[Event]) -> "LongTermMemory":
        """Incrementally extend with new FactObservation events."""
        new_facts = [
            {"text": str(ev.payload.content), "source_seq": ev.seq, "ts": ev.ts}
            for ev in new_events
            if ev.kind == EventKind.OBSERVATION
            and isinstance(ev.payload, FactObservation)
        ]
        if not new_facts:
            return self
        all_facts = list(self._facts) + new_facts
        return LongTermMemory(all_facts, max_facts=self.max_facts)

    def render(self, max_chars: int = 2_000) -> str:
        """A compact context section; empty when there is nothing durable."""
        if not self._facts:
            return ""
        lines = [f"- {f['text']}" for f in self._facts]
        used = 0
        out: List[str] = []
        for line in lines:
            if used + len(line) > max_chars:
                break
            out.append(line)
            used += len(line)
        return "\n".join(out)

    def to_dict(self) -> Dict[str, Any]:
        return {"facts": list(self._facts), "max_facts": self.max_facts}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "LongTermMemory":
        if not data:
            return cls()
        return cls(data.get("facts") or [], max_facts=int(data.get("max_facts") or 50))


# ── The bundle handed to the agent each step ─────────────────────────────────

@dataclass(frozen=True)
class ContextMemory:
    """
    Immutable snapshot of working memory for ONE agent step: the short-term
    step records plus the durable facts. The agent is stateless — it
    renders whatever context it is handed; the Conversation rebuilds this
    snapshot from the EventLog every step (pure projections, no new
    mutable state).

    ``summary`` optionally carries a structured brief of OLDER conversation
    that was compacted out of the raw slice (see
    ``orcha.agent_runtime.compaction``); it renders as its own section and
    never touches the authoritative log.
    """
    steps: Tuple[StepRecord, ...] = ()
    facts: Tuple[str, ...] = ()
    summary: str = ""

    @classmethod
    def from_events(
        cls, events: Sequence[Event], *, max_facts: int = 50,
    ) -> "ContextMemory":
        return cls(
            steps=StepMemory.from_events(events).records,
            facts=LongTermMemory.from_events(events, max_facts=max_facts).texts,
        )

    def with_summary(self, summary: str) -> "ContextMemory":
        """Same projection carrying a compaction brief (pure replace)."""
        if not summary:
            return self
        return ContextMemory(steps=self.steps, facts=self.facts, summary=summary)


class IncrementalContextMemory:
    """
    Incremental wrapper around ContextMemory that avoids O(n) rebuilds
    from all events every step. Maintains running StepMemory and
    LongTermMemory, only processing new events since the last build.

    Usage in the conversation loop:
        memory = self._memory_builder.build(self._log.events)
    """

    def __init__(self, max_facts: int = 50) -> None:
        self._max_facts = max_facts
        self._step_memory = StepMemory()
        self._long_term = LongTermMemory(max_facts=max_facts)
        self._last_event_count = 0
        self._cached: Optional[ContextMemory] = None

    def build(self, events: Sequence[Event]) -> ContextMemory:
        """
        Build ContextMemory incrementally. Only processes events that
        weren't seen in the previous build.
        """
        n = len(events)
        if n == self._last_event_count and self._cached is not None:
            return self._cached

        if n < self._last_event_count:
            # Events were replaced (e.g. after compaction) — full rebuild.
            self._step_memory = StepMemory.from_events(events)
            self._long_term = LongTermMemory.from_events(events, max_facts=self._max_facts)
        else:
            # Incremental: only process new events.
            new_events = events[self._last_event_count:]
            if new_events:
                self._step_memory = self._step_memory.extend(new_events)
                self._long_term = self._long_term.extend(new_events)

        self._last_event_count = n
        self._cached = ContextMemory(
            steps=self._step_memory.records,
            facts=self._long_term.texts,
        )
        return self._cached


def render_memory_section(
    memory: Optional[ContextMemory],
    *,
    max_context_tokens: int,
    facts_max_chars: int = 2_000,
) -> str:
    """
    Assemble the working-memory block for a prompt: any compaction summary
    first (it stands in for older history), then durable facts (if any),
    then the windowed summary of recent steps (if any). Returns "" when
    there is nothing to inject.
    """
    if memory is None:
        return ""
    parts: List[str] = []
    if memory.summary:
        parts.append("[Earlier conversation]\n" + memory.summary)
    facts = LongTermMemory(
        [{"text": t} for t in memory.facts],
    ).render(max_chars=facts_max_chars)
    steps = StepMemory(memory.steps).render_window(max_context_tokens)
    if facts:
        parts.append("[Durable facts]\n" + facts)
    if steps:
        parts.append("[Recent steps]\n" + steps)
    return "\n\n".join(parts)


__all__ = [
    "StepRecord", "StepMemory", "LongTermMemory", "ContextMemory",
    "render_memory_section",
]
