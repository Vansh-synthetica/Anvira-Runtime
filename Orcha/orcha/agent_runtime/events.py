"""
orcha.agent_runtime.events
==========================
The event-sourced data model of the AgentRuntime (OpenHands V1 style):

- ``Action`` — typed union of everything the AGENT emits:
  ``ToolCallAction``, ``MessageAction``, ``FinishAction``, ``ErrorAction``.
- ``Observation`` — typed union of everything the SYSTEM observes:
  ``ToolResultObservation``, ``ErrorObservation``, ``UserMessageObservation``.
- ``Event`` — one immutable, append-only, ordered log entry. An Event wraps
  exactly one Action or Observation plus the token cost it incurred.
- ``EventLog`` — the append-only, replayable store. **This is the only
  authoritative state in the system.** Nothing else may hold mutable state
  that matters: every state change — a user message, an agent step, a tool
  result, a cutoff — is an Event appended here.
- ``fold_log`` — the pure fold that projects an EventLog onto a
  ``LogState``. Deterministic replay is exactly this fold: given the same
  events it reconstructs identical state with zero side effects (it never
  touches the agent, the workspace, the filesystem, or the clock).

Contract
--------
- Events are immutable once appended and their ``seq`` is monotonic.
- State is always READ via ``fold`` / ``fold_log``; never by poking the log.
- ``tokens`` is decided at append time (see ``estimate_tokens``); the fold
  only accumulates it.
"""
from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Annotated, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Literal

logger = logging.getLogger("orcha.agent_runtime.events")


# ── Actions (agent → system) ────────────────────────────────────────────────

class Action(BaseModel):
    """
    Base class of the typed Action union. Never instantiate directly.

    ``tokens`` / ``reasoning_tokens`` are the token cost the agent
    attributes to this action (from the backend's usage report; zero when
    the backend reports none). The Conversation copies them onto the
    wrapped Event when appending.
    """
    kind: str
    tokens: int = 0
    reasoning_tokens: int = 0


class ToolCallAction(Action):
    """The agent wants the workspace to execute a tool."""
    kind: Literal["tool_call"] = "tool_call"
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    tool_call_id: Optional[str] = None


class MessageAction(Action):
    """The agent speaks to the user but keeps working."""
    kind: Literal["message"] = "message"
    content: str


class FinishAction(Action):
    """The agent is done. ``content`` is the final answer text; ``reason``
    records why the conversation ended (``done`` when the agent decided to
    stop, otherwise the Conversation's cutoff name)."""
    kind: Literal["finish"] = "finish"
    content: str = ""
    reason: str = "done"


class ErrorAction(Action):
    """The agent failed and aborts the turn."""
    kind: Literal["error_action"] = "error_action"
    message: str


# ── Observations (system → agent) ──────────────────────────────────────────

class Observation(BaseModel):
    """Base class of the typed Observation union. Never instantiate directly."""
    kind: str


class ToolResultObservation(Observation):
    """The result of a ToolCallAction the workspace executed."""
    kind: Literal["tool_result"] = "tool_result"
    tool_call_id: Optional[str] = None
    content: str
    success: bool = True
    # Filesystem tools attach real before/after snapshots here.  This is part
    # of the ordinary append-only execution event, not a separate edit path.
    file_changes: List[Dict[str, Any]] = Field(default_factory=list)


class ErrorObservation(Observation):
    """Something went wrong while executing an action."""
    kind: Literal["error"] = "error"
    message: str


class UserMessageObservation(Observation):
    """A message the human sent into the conversation."""
    kind: Literal["user_message"] = "user_message"
    content: str


class FactObservation(Observation):
    """
    A durable fact or preference recorded for long-term memory
    (``Conversation.remember``). Kept distinct from step-level tool
    observations so the memory projections can separate short-term step
    memory from longer-lived facts; never resets the turn boundary.
    """
    kind: Literal["fact"] = "fact"
    content: str


# ── Union aliases (pydantic discriminated unions on `kind`) ────────────────

ActionPayload = Annotated[
    Union[ToolCallAction, MessageAction, FinishAction, ErrorAction],
    Field(discriminator="kind"),
]

ObservationPayload = Annotated[
    Union[
        ToolResultObservation, ErrorObservation, UserMessageObservation,
        FactObservation,
    ],
    Field(discriminator="kind"),
]

EventPayload = Annotated[
    Union[
        ToolCallAction, MessageAction, FinishAction, ErrorAction,
        ToolResultObservation, ErrorObservation, UserMessageObservation,
        FactObservation,
    ],
    Field(discriminator="kind"),
]

# ── Events ──────────────────────────────────────────────────────────────────

class EventKind(str, Enum):
    USER_MESSAGE = "user_message"  # the human spoke
    ACTION       = "action"        # the agent acted (one agent step)
    OBSERVATION  = "observation"   # the system observed the result


class Event(BaseModel):
    """
    One immutable entry in the EventLog.

    ``payload`` is exactly one Action or Observation; ``tokens`` is the
    token cost this entry contributed to the conversation's budget;
    ``reasoning_tokens`` is the optional reasoning-token cost the backend
    reported for this step (zero when the backend does not report it —
    absence is never treated specially, it simply adds nothing).
    """
    model_config = ConfigDict(frozen=True)

    seq: int = Field(ge=1)
    kind: EventKind
    payload: EventPayload
    tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    ts: float = Field(default_factory=time.time)

    @property
    def action(self) -> Optional[Action]:
        """The wrapped Action, if this event is an agent step."""
        return self.payload if self.kind == EventKind.ACTION else None

    @property
    def observation(self) -> Optional[Observation]:
        """The wrapped Observation, if this event is an observation."""
        return self.payload if self.kind != EventKind.ACTION else None

    def __repr__(self) -> str:
        return (
            f"Event(seq={self.seq}, kind={self.kind.value}, "
            f"tokens={self.tokens}, payload={type(self.payload).__name__})"
        )


def kind_of(payload: Any) -> EventKind:
    """Infer the EventKind for a payload object (an Action or an
    Observation). UserMessageObservation is kept distinct from plain
    tool/error observations so the fold can track turn boundaries."""
    if isinstance(payload, UserMessageObservation):
        return EventKind.USER_MESSAGE
    if isinstance(payload, Action):
        return EventKind.ACTION
    return EventKind.OBSERVATION


# ── The fold: deterministic projection ─────────────────────────────────────

class LogState(BaseModel):
    """
    The deterministic projection of an EventLog.

    This is the ONLY way to read conversation state. It is produced by a
    pure fold over events, so two logs with identical events yield
    identical LogStates — that property is what makes replay safe.
    """
    seq: int = 0                  # highest event seq in the log
    events: int = 0               # total event count
    actions: int = 0              # count of ACTION events
    observations: int = 0         # count of OBSERVATION events
    steps: int = 0                # agent iterations (ACTION events, excluding
                                  # the terminal FinishAction emitted on cutoff)
    tokens_used: int = 0          # accumulated token cost of all events
    reasoning_tokens_used: int = 0  # accumulated reasoning-token cost (folded
                                  # into the SAME budget accounting as
                                  # tokens_used — see Conversation's cutoff)
    last_user_seq: int = 0        # seq of the most recent user message (turn
                                  # boundary for the agent's EventLog slice)
    last_action: Optional[ActionPayload] = None
    last_observation: Optional[ObservationPayload] = None
    finished: bool = False        # a FinishAction has been emitted
    finish_reason: Optional[str] = None
    answer: Optional[str] = None  # final answer text (FinishAction.content)


def fold_log(events: Sequence[Event], start: Optional[LogState] = None) -> LogState:
    """
    Pure fold over ``events`` onto a LogState. No side effects: the events
    are never mutated, nothing is read from the environment, and calling it
    repeatedly with the same events returns identical state. Deterministic
    replay *is* this function.

    ``start`` allows incremental folding (e.g. a partial replay); when
    given, it is deep-copied so the caller's object is never mutated.
    Optimized: skip the deep copy when start is None (the common path).
    """
    if start is not None:
        state = start.model_copy(deep=True)
    else:
        state = LogState()

    for ev in events:
        state.seq = ev.seq
        state.events += 1
        state.tokens_used += ev.tokens
        state.reasoning_tokens_used += ev.reasoning_tokens

        kind = ev.kind
        if kind == EventKind.USER_MESSAGE:
            state.last_user_seq = ev.seq
            state.observations += 1
            state.last_observation = ev.payload

        elif kind == EventKind.ACTION:
            state.actions += 1
            state.last_action = ev.payload
            if ev.payload.kind == "finish":
                state.finished = True
                state.finish_reason = ev.payload.reason
                state.answer = ev.payload.content
            else:
                state.steps += 1

        else:  # EventKind.OBSERVATION
            state.observations += 1
            state.last_observation = ev.payload

    return state


def estimate_tokens(text: str) -> int:
    """
    Rough deterministic token estimate (≈1 token per 4 chars) used by the
    skeleton until real model metering arrives with the Ollama integration.
    """
    return max(1, len(text) // 4)


# ── The log ────────────────────────────────────────────────────────────────

class EventLog:
    """
    Append-only, ordered, replayable event store.

    - ``append`` is the ONLY mutation; events cannot be updated or removed.
    - ``events`` returns an immutable snapshot (tuple).
    - ``fold`` / ``fold_log`` read state; ``replay`` rebuilds state from any
      event sequence without touching this log.
    """

    def __init__(self) -> None:
        self._events: List[Event] = []
        self._next_seq: int = 1  # 1-based seqs, like OpenHands
        self._subscribers: List[Callable[[Event], None]] = []
        self._cached_state: Optional[LogState] = None
        self._cached_seq: int = 0
        self._cached_index: int = 0  # index into _events for incremental fold

    # ── Subscription (optional live feed) ──────────────────────────────

    def subscribe(self, callback: Callable[[Event], None]) -> Callable[[Event], None]:
        """
        Register ``callback`` to be invoked (synchronously, in append order)
        for every event appended after this call. Returns the same callback
        for convenient ``unsubscribe(subscribe(fn))`` chaining.
        """
        if callback not in self._subscribers:
            self._subscribers.append(callback)
        return callback

    def unsubscribe(self, callback: Callable[[Event], None]) -> None:
        """Stop delivering appended events to ``callback``. No-op if absent."""
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass

    # ── Mutation (append-only) ────────────────────────────────────────

    def append(self, payload: Any, tokens: int = 0, reasoning_tokens: int = 0) -> Event:
        """
        Append one event wrapping ``payload`` (an Action or an Observation).
        The event's kind is inferred from the payload type. Returns the
        appended Event.
        """
        event = Event(
            seq=self._next_seq,
            kind=kind_of(payload),
            payload=payload,
            tokens=max(0, int(tokens)),
            reasoning_tokens=max(0, int(reasoning_tokens)),
        )
        self._events.append(event)
        self._next_seq += 1
        self._cached_state = None  # invalidate incremental fold cache
        self._cached_index = 0
        for callback in tuple(self._subscribers):
            try:
                callback(event)
            except Exception:
                logger.exception("event subscriber failed on seq=%s", event.seq)
        return event

    # ── Reads (immutable snapshots) ───────────────────────────────────

    @property
    def events(self) -> Tuple[Event, ...]:
        """Immutable snapshot of every event, in append order."""
        return tuple(self._events)

    def slice(self, after_seq: int = 0, limit: Optional[int] = None) -> Tuple[Event, ...]:
        """Events whose seq is strictly greater than ``after_seq``
        (optionally capped at ``limit`` entries), in order.

        Uses binary search for O(log n) start position when the log is large.
        """
        if not self._events or self._events[-1].seq <= after_seq:
            return ()
        # Binary search for the first event with seq > after_seq.
        lo, hi = 0, len(self._events)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._events[mid].seq <= after_seq:
                lo = mid + 1
            else:
                hi = mid
        out = self._events[lo:]
        return tuple(out[:limit] if limit is not None else out)

    def fold(self) -> LogState:
        """Project the whole log onto a LogState (pure fold).

        Uses incremental caching: if called repeatedly, only new events
        since the last fold are processed, avoiding O(n) re-scans.
        Returns a copy of cached state to prevent mutation of the cache.
        """
        if not self._events:
            return LogState()
        if self._cached_state is not None:
            # Check if cache is still valid (same last seq)
            if self._cached_seq == self._events[-1].seq:
                return self._cached_state.model_copy(deep=False)
            # Incremental: only fold new events since last cache.
            # Use index-based slice instead of O(n) filter.
            cached_idx = self._cached_index
            new_events = self._events[cached_idx:]
            if new_events:
                self._cached_state = fold_log(new_events, start=self._cached_state)
            self._cached_seq = self._events[-1].seq
            self._cached_index = len(self._events)
            return self._cached_state.model_copy(deep=False)
        # First call: full fold
        self._cached_state = fold_log(self._events)
        self._cached_seq = self._events[-1].seq
        self._cached_index = len(self._events)
        return self._cached_state.model_copy(deep=False)

    @classmethod
    def replay(cls, events: Sequence[Event]) -> LogState:
        """
        Reconstruct state from an arbitrary event sequence. Used to replay
        a captured conversation with zero side effects: the caller's events
        are only read, never mutated.
        """
        return fold_log(events)

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self):
        return iter(self._events)

    def __repr__(self) -> str:
        return f"EventLog(events={len(self._events)}, next_seq={self._next_seq})"


__all__ = [
    "Action", "ToolCallAction", "MessageAction", "FinishAction", "ErrorAction",
    "Observation", "ToolResultObservation", "ErrorObservation",
    "UserMessageObservation", "FactObservation",
    "ActionPayload", "ObservationPayload", "EventPayload",
    "EventKind", "Event", "kind_of",
    "LogState", "fold_log", "estimate_tokens",
    "EventLog",
]
