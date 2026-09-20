from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .conversation import Conversation
from .diagnostics import diag_log
from .events import Event, EventKind, EventLog
from .memory import StepMemory

if TYPE_CHECKING:
    from .diagnostics import DiagnosticSession

logger = logging.getLogger("orcha.agent_runtime.stream")

__all__ = [
    "event_to_wire",
    "AgentEventStream",
    "AgentSession",
    "AgentSessionRegistry",
]


def event_to_wire(event: Event, step_records: Optional[Dict[int, Any]] = None) -> Dict[str, Any]:
    """
    Thin serialization of an internal Event for the SSE wire, preserving the
    Prompt-1 event model (seq, kind, ts, tokens, reasoning_tokens, payload)
    and, when available, the Prompt-4 step-memory metadata for the seq.

    ``step_records`` maps event seq -> StepRecord dict (from ``StepRecord.to_dict``).
    """
    wire: Dict[str, Any] = {
        "seq": event.seq,
        "kind": event.kind.value,
        "event_type": event.payload.kind,
        "ts": event.ts,
        "tokens": event.tokens,
        "reasoning_tokens": event.reasoning_tokens,
        "payload": event.payload.model_dump(mode="json"),
    }
    if step_records is not None:
        wire["step"] = step_records.get(event.seq)
    return wire


class AgentEventStream:
    """
    A single live consumer of an EventLog with cursor-based catch-up.

    - ``start()`` replays every event with seq > cursor (strictly), then
      transitions to live delivery via an EventLog subscription.
    - ``next()`` yields events strictly in seq order; ``None`` once the
      stream is finished/closed and drained.
    - At-least-once semantics across reconnects: a client that reconnects
      with its last received seq as the cursor re-receives nothing already
      delivered (catch-up starts strictly after the cursor). Within one
      connection every event is delivered exactly once.
    """

    def __init__(self, log: EventLog, cursor: int = 0) -> None:
        self._log = log
        self._cursor = max(0, int(cursor))
        self._buffer: collections.deque = collections.deque()
        self._wake = asyncio.Event()
        self._subscription = log.subscribe(self._on_event)
        self._closed = False
        self._delivered = self._cursor

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Replay history strictly after the cursor, then go live."""
        last = self._cursor
        replay: collections.deque = collections.deque()
        for ev in self._log.slice(after_seq=self._cursor):
            replay.append(ev)
            last = ev.seq
        # Nothing has been handed out yet; the next() guard skips anything
        # at or below the cursor.
        self._delivered = self._cursor
        # Buffer entries with seq <= last were already replayed (queued live
        # before start()); entries with seq > last were appended after the
        # replay snapshot — deliver them live, after the replayed history.
        live = collections.deque(ev for ev in self._buffer if ev is None or ev.seq > last)
        self._buffer = replay + live
        self._wake.set()

    def close(self) -> None:
        """Stop live delivery; already-buffered events are still drained."""
        if not self._closed:
            self._closed = True
            self._log.unsubscribe(self._subscription)
            self._wake.set()

    def finish(self) -> None:
        """Close and signal end-of-stream to ``next()`` consumers."""
        self.close()
        self._buffer.append(None)
        self._wake.set()

    # ── consumption ───────────────────────────────────────────────────

    async def next(self) -> Optional[Event]:
        """Next event in seq order, or None once finished/closed and drained."""
        while True:
            if self._buffer:
                ev = self._buffer.popleft()  # O(1) instead of O(n) pop(0)
                if ev is None:
                    return None
                if ev.seq <= self._delivered:
                    continue  # defensive: never deliver out of order
                self._delivered = ev.seq
                return ev
            if self._closed:
                return None
            self._wake.clear()
            await self._wake.wait()

    # ── internals ─────────────────────────────────────────────────────

    def _on_event(self, event: Event) -> None:
        if not self._closed:
            self._buffer.append(event)
            self._wake.set()


def _step_records_by_seq(events) -> Dict[int, Any]:
    """
    Correlate step records with the event seqs that produced them.

    ``StepMemory.from_events`` opens one record per ACTION event (in log
    order) and closes it on the next OBSERVATION; so the i-th ACTION maps to
    the i-th record, and the observation that closes it maps to the same
    (final) record. This mirrors that folding exactly, with zero changes to
    memory.py.
    """
    records = StepMemory.from_events(events).records
    mapping: Dict[int, Any] = {}
    record_idx = -1
    pending = False
    for ev in events:
        if ev.kind == EventKind.ACTION:
            record_idx += 1
            pending = True
            if 0 <= record_idx < len(records):
                mapping[ev.seq] = records[record_idx].to_dict()
        elif pending and ev.kind == EventKind.OBSERVATION:
            pending = False
            if 0 <= record_idx < len(records):
                mapping[ev.seq] = records[record_idx].to_dict()
    return mapping


class AgentSession:
    """
    One agent run: the conversation/log the agent appends to, the
    step-memory projection used to enrich the wire events, and the set of
    attached live streams.
    """

    def __init__(self, session_id: str, conversation: Conversation, query: str = "") -> None:
        self.session_id = session_id
        self.conversation = conversation
        self.query = query
        self.log = conversation.log
        self.started = False
        self.finished = False
        self.error: Optional[str] = None
        # Read-only observability (Prompt 7): never consulted by the loop,
        # the log, replay, memory or budgeting.
        self.diagnostics: Optional["DiagnosticSession"] = None
        self._streams: List[AgentEventStream] = []

    @property
    def last_seq(self) -> int:
        """Seq of the most recently appended event (0 when empty)."""
        return self.log._next_seq - 1 if len(self.log) else 0

    def attach(self, stream: AgentEventStream) -> None:
        replay_n = len(list(self.log.slice(after_seq=stream._cursor)))
        diag_log(
            logger, "sse", "attach",
            session=self.session_id, cursor=stream._cursor,
            replay_events=replay_n, finished=self.finished,
        )
        self._streams.append(stream)
        stream.start()
        if self.finished:
            # A reconnect onto an already-finished session must terminate
            # (replay the catch-up, then signal end-of-stream).
            stream.finish()

    def detach(self, stream: AgentEventStream) -> None:
        delivered_n = max(0, stream._delivered - stream._cursor)
        diag_log(
            logger, "sse", "disconnect",
            session=self.session_id, delivered_events=delivered_n,
        )
        stream.close()
        with contextlib.suppress(ValueError):
            self._streams.remove(stream)

    def notify_finished(self) -> None:
        self.finished = True
        diag_log(logger, "sse", "end", session=self.session_id, error=None)
        for stream in tuple(self._streams):
            stream.finish()

    def notify_error(self, message: str) -> None:
        self.finished = True
        self.error = message
        diag_log(
            logger, "sse", "end", level=logging.ERROR,
            session=self.session_id, error=message,
        )
        for stream in tuple(self._streams):
            stream.finish()

    def step_records(self) -> Dict[int, Any]:
        """Map of event seq -> step-record dict (for wire enrichment)."""
        return _step_records_by_seq(self.log.events)


class AgentSessionRegistry:
    """
    Owns every live agent session. Sessions are keyed by session_id; streams
    attach to a session's log and are served by the router with cursor
    catch-up across reconnects.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, AgentSession] = {}

    # ── lifecycle ─────────────────────────────────────────────────────

    def create(self, conversation: Conversation, query: str = "") -> AgentSession:
        session = AgentSession(self._new_id(), conversation, query=query)
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Optional[AgentSession]:
        return self._sessions.get(session_id)

    def remove(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.notify_finished()

    # ── helpers ───────────────────────────────────────────────────────

    def _new_id(self) -> str:
        import uuid

        return uuid.uuid4().hex
