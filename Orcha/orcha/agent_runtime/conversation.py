"""
orcha.agent_runtime.conversation
===============================
The Conversation owns the agent loop (OpenHands V1 model):

    pull latest EventLog slice ──► Agent.step() ──► Action
        ──► (bus) Workspace.execute() ──► Observation ──► append to EventLog
        ──► repeat until FinishAction or a cutoff.

Budget discipline
-----------------
Cutoffs are enforced as pure folds over the EventLog (the authoritative
state) — never stored as mutable counters:

- ``max_steps``  — maximum agent iterations. When a ``BudgetState`` rides
  on the incoming packet, its ``max_iterations`` also caps the loop
  (whichever is tighter), honoring Orcha's existing budget machinery.
- ``max_tokens`` — total token cost of all events in the log. Reasoning
  tokens reported by the backend fold into the SAME accounting: the cutoff
  compares ``tokens_used + reasoning_tokens_used`` against the cap, so a
  reasoning-heavy backend is charged exactly like any other — absence
  simply means zero, no special-casing.
- Wall-clock and the stop decision defer to Orcha's existing budget
  machinery: the agent loop tracks elapsed time ON the shared
  ``BudgetState`` (``latency_used_s``) and asks ``BudgetPlanner.plan`` —
  the same planner the orchestration pipeline uses — whether to stop. No
  parallel budget path: one budget object, one planner, same exhaustion
  semantics.

Working memory
--------------
Each step the model receives a WINDOWED SUMMARY of recent steps plus any
durable facts (``ContextMemory``), never the raw EventLog — so context
stays bounded across long sessions (see ``orcha.agent_runtime.memory``).
``remember()`` records durable facts as ordinary events.

Every transition — including the cutoffs themselves — is an Event appended
to the log, so replay sees the exact same sequence and reconstructs
identical state.

Since Prompt 2 the loop is async (backends are I/O-bound), and the agent
may return several actions per model call (``step_batch``) — the loop
executes them one at a time, so multiple tool calls per turn cost exactly
ONE model call.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from ..core.packets import BudgetState, OrchaPacket, PacketKind
from ..orchestration.planner import BudgetPlanner
from .agent import Agent, AgentConfig
from .bus import (
    action_packet, observation_from_packet,
)
from .compaction import Compactor
from .diagnostics import diag_log
from .errors import EmptyLogError
from .events import (
    Action, ErrorAction, Event, EventLog, EventKind, FinishAction, LogState,
    MessageAction, ToolCallAction, ToolResultObservation,
    UserMessageObservation, FactObservation,
    estimate_tokens, fold_log,
)
from .memory import ContextMemory, IncrementalContextMemory
from .workspace import Workspace

logger = logging.getLogger("orcha.agent_runtime.conversation")


def sanitize_final_answer(text: Optional[str]) -> Optional[str]:
    """
    Guarantee the final answer is natural language — never an empty string
    and never a bare tool-call JSON object (the model occasionally emits a
    tool-call-shaped object as its final text; that must remain an internal
    execution record, never the user-facing answer).

    Returns None when the text is unusable (empty / whitespace / JSON tool
    call), so callers fall back to a graceful message.
    """
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except Exception:
        # Not JSON — plain natural language (even "[stub] canned reply").
        return stripped
    if isinstance(parsed, (dict, list)):
        # JSON tool-call objects/arrays are execution records, never answers.
        return None
    return stripped


def _graceful_limits_answer(
    reason: str, steps: int, tool_calls: int,
) -> str:
    """Natural-language fallback when the run ends at a cutoff."""
    tool_note = f" and {tool_calls} tool call(s)" if tool_calls else ""
    reason_label = {
        "text_repetition": "the model entered a repetition loop",
        "stalled": "the model stopped making progress",
        "all_tools_unavailable": "no tools were available",
        "max_steps": "the step limit was reached",
        "max_tokens": "the token budget was exhausted",
        "budget": "the run budget was exhausted",
        "error": "an error occurred",
    }.get(reason, reason)
    return (
        f"I wasn't able to finish: {reason_label}. "
        f"I performed {steps} step(s){tool_note} before stopping. "
        "Try rephrasing the request or activating an expert model."
    )


def _texts_are_repetitive(texts: List[str]) -> bool:
    """
    Detect degenerate text repetition in consecutive MessageActions.

    Returns True when the last N messages are "highly similar" — either by
    exact-match (after stripping whitespace), substring containment (shorter
    text is mostly contained in the longer one), or high character overlap.
    This catches the classic small-model loop where the model repeats the
    same phrase or sentence over and over.
    """
    if len(texts) < 2:
        return False
    # Compare each text against the previous one.
    for i in range(1, len(texts)):
        prev = texts[i - 1].strip()
        curr = texts[i].strip()
        if not prev or not curr:
            continue
        # Exact match after stripping.
        if prev == curr:
            return True
        # One text contains the other (substring).
        shorter, longer = (prev, curr) if len(prev) <= len(curr) else (curr, prev)
        if len(shorter) >= 20 and shorter in longer:
            return True
        # High character overlap (Jaccard on character bigrams).
        if len(prev) > 10 and len(curr) > 10:
            bigrams_prev = set(zip(prev, prev[1:]))
            bigrams_curr = set(zip(curr, curr[1:]))
            if bigrams_prev and bigrams_curr:
                overlap = len(bigrams_prev & bigrams_curr)
                union = len(bigrams_prev | bigrams_curr)
                if union > 0 and overlap / union > 0.7:
                    return True
    return False


def _has_intra_message_repetition(text: str, min_phrase_len: int = 2, threshold: int = 4) -> bool:
    """
    Detect degenerate token-level repetition WITHIN a single message.

    Catches multiple pathological patterns common in small models:
    1. Exact consecutive phrase repeats: "and my and my and my ..."
    2. Vocabulary collapse: output shrinks to <15 unique words over 50+ words
    3. Single-word dominance: one word appears >30% of the time
    4. Alternating short-phrase loops: "this project, this folder, this file ..."
    """
    import re
    from collections import Counter

    words = re.split(r'\s+', text.strip())
    if len(words) < 8:
        return False

    # ── Pattern 1: Exact consecutive phrase repeats ──────────────────
    for phrase_len in range(min_phrase_len, min(7, len(words) // threshold + 1)):
        i = 0
        while i <= len(words) - phrase_len * threshold:
            phrase = tuple(words[i:i + phrase_len])
            count = 0
            j = i
            while j <= len(words) - phrase_len:
                if tuple(words[j:j + phrase_len]) == phrase:
                    count += 1
                    j += phrase_len
                else:
                    break
            if count >= threshold:
                return True
            i += 1

    # ── Pattern 2: Vocabulary collapse ───────────────────────────────
    if len(words) > 50:
        unique_words = set(w.lower().strip('",.:;!?') for w in words)
        ratio = len(unique_words) / len(words)
        if len(unique_words) < 15 or ratio < 0.08:
            return True

    # ── Pattern 3: Single-word dominance ─────────────────────────────
    if len(words) > 30:
        clean = [w.lower().strip('",.:;!?') for w in words]
        counts = Counter(clean)
        most_common_word, most_common_count = counts.most_common(1)[0]
        if most_common_count / len(words) > 0.30 and most_common_count >= 10:
            return True

    # ── Pattern 4: Alternating short-phrase loops ────────────────────
    if len(words) >= 18:
        for cycle_len in (2, 3, 4):
            if len(words) < cycle_len * 6:
                continue
            phrases_at_pos = []
            for pos in range(cycle_len):
                phrases = []
                for start in range(pos, len(words) - 1, cycle_len):
                    phrases.append(tuple(words[start:start + 2]))
                phrases_at_pos.append(phrases)
            cycle_phrases = []
            for pos_phrases in phrases_at_pos:
                if not pos_phrases:
                    continue
                phrase_counts = Counter(pos_phrases)
                most_common, count = phrase_counts.most_common(1)[0]
                if count / len(pos_phrases) >= 0.6:
                    cycle_phrases.append(most_common)
                else:
                    cycle_phrases = []
                    break
            if len(cycle_phrases) == cycle_len:
                matched = 0
                for i in range(len(words) - 1):
                    w = tuple(words[i:i + 2])
                    if w in cycle_phrases:
                        matched += 1
                if matched / max(len(words) - 1, 1) > 0.50:
                    return True

    return False


class ConversationConfig(BaseModel):
    """Immutable loop limits for a Conversation."""
    model_config = ConfigDict(frozen=True)

    max_steps: int = Field(default=30, ge=1)     # max agent iterations
    max_tokens: int = Field(default=0, ge=0)     # 0 = unlimited token budget
    # Loop guards (Problem 1): identical tool+args retried more than
    # ``max_tool_repeats`` times is marked unavailable for the rest of the
    # run; ``max_stalled_turns`` tool-only turns without any message or
    # finish force a graceful stop. These bound pathological loops WITHOUT
    # inflating max_steps.
    max_tool_repeats: int = Field(default=2, ge=1)
    max_stalled_turns: int = Field(default=4, ge=1)
    # Loop guard: when the model produces N consecutive assistant messages
    # with highly similar content (text repetition / degenerate loop), stop.
    max_message_repeats: int = Field(default=3, ge=1)


class ConversationResult(BaseModel):
    """
    Terminal projection of a Conversation run. ``events`` is the full
    append-only log; folding it again (``EventLog.replay(events)``)
    reconstructs exactly ``log_state`` — deterministic replay.
    """
    query: str
    answer: Optional[str] = None        # FinishAction content, if any
    terminated_by: str = "finish"       # finish | max_steps | max_tokens | budget | error
    steps: int = 0
    tokens_used: int = 0                # total charged: tokens + reasoning
    reasoning_tokens_used: int = 0      # the reasoning-token share
    events: tuple = Field(default_factory=tuple)  # tuple[Event, ...] — the full log
    log_state: LogState = Field(default_factory=LogState)

    @property
    def completed(self) -> bool:
        """True when the agent itself chose to finish (no cutoff)."""
        return self.terminated_by == "finish"

    @property
    def log(self) -> EventLog:
        """A fresh EventLog rebuilt from this run's events — replaying it
        reproduces the run exactly (same seqs, same tokens, same state)."""
        rebuilt = EventLog()
        for ev in self.events:
            rebuilt.append(
                ev.payload,
                tokens=ev.tokens,
                reasoning_tokens=ev.reasoning_tokens,
            )
        return rebuilt

    def __repr__(self) -> str:
        return (
            f"ConversationResult(terminated_by={self.terminated_by!r}, "
            f"steps={self.steps}, tokens={self.tokens_used}, "
            f"events={len(self.events)})"
        )


class Conversation:
    """
    The loop owner. Holds the EventLog (the only authoritative state), the
    stateless Agent, and the Workspace; drives Actions and Observations
    across the OrchaPacket bus.

    Usage
    -----
        convo = Conversation(agent, workspace, query="...")
        convo.submit_user_message("read the attached folder")
        result = await convo.run()
        # replay: EventLog.replay(result.events) == result.log_state
    """

    def __init__(
        self,
        agent: Agent,
        workspace: Workspace,
        *,
        config: Optional[ConversationConfig] = None,
        agent_config: Optional[AgentConfig] = None,
        query: str = "",
        budget: Optional[BudgetState] = None,
        budget_planner: Optional[BudgetPlanner] = None,
        compactor: Optional["Compactor"] = None,
    ) -> None:
        self._agent = agent
        self._workspace = workspace
        self._config = config or ConversationConfig()
        self._agent_config = agent_config or AgentConfig()
        self._query = query
        self._budget = budget
        self._planner = budget_planner or BudgetPlanner()
        self._log = EventLog()
        # Optional compaction ladder (pure projections; the log is never
        # mutated, so replay stays byte-identical with or without one).
        self._compactor: Optional[Compactor] = compactor
        self.last_result: Optional[ConversationResult] = None
        # Loop-guard state (per run): tools marked unavailable this run and
        # the count of identical (tool, args) calls, so repeated calls to a
        # tool that cannot help are stopped at the source.
        self._unavailable: set = set()
        self._tool_call_counts: Dict[Tuple[str, str], int] = {}
        self._stalled_turns: int = 0
        # Track recent MessageAction content for text-repetition detection.
        self._recent_message_texts: List[str] = []
        # Incremental memory builder — avoids O(n) rebuilds every step.
        self._memory_builder = IncrementalContextMemory()

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def log(self) -> EventLog:
        """The append-only event store (authoritative state)."""
        return self._log

    @property
    def state(self) -> LogState:
        """Current deterministic projection of the log."""
        return self._log.fold()

    @property
    def answer(self) -> Optional[str]:
        """The run's final answer (None before the run finishes)."""
        if self.last_result is not None:
            return self.last_result.answer
        return self._log.fold().answer

    def submit_user_message(self, content: str, tokens: int = 0) -> LogState:
        """
        Append a UserMessageObservation event (the human's input). The
        conversation must have a user message before ``run()``.
        """
        self._log.append(UserMessageObservation(content=content), tokens=tokens)
        return self._log.fold()

    def remember(self, content: str) -> LogState:
        """
        Record a durable fact or preference as a FactObservation event.

        Facts live in long-term memory: they are excluded from the turn
        transcript (they never reset the turn boundary) and are instead
        projected into the ``[Durable facts]`` working-memory section of
        every step's context (see ``orcha.agent_runtime.memory``).
        """
        self._log.append(FactObservation(content=content))
        return self._log.fold()

    async def run(self, budget: Optional[BudgetState] = None) -> ConversationResult:
        """
        Run the loop to a FinishAction or a cutoff.

        ``budget`` (an Orcha BudgetState, mirroring the packet bus) is
        honored on top of the config:

        - the effective step limit is ``min(max_steps, max_iterations)``;
        - wall-clock and iteration accounting ride ON the same budget
          object (``latency_used_s`` / ``iterations``), and the loop asks
          Orcha's ``BudgetPlanner`` — the same planner the orchestration
          pipeline uses — for the stop decision each iteration, so a
          budget-limit hit ends in a clean FinishAction, never a loop
          without end.

        Each agent step receives a working-memory snapshot (windowed step
        records + durable facts), so the model's context stays bounded
        across arbitrarily long sessions.
        """
        budget = budget or self._budget
        if self._log.fold().events == 0:
            raise EmptyLogError(
                "submit_user_message() must be called before run() — the "
                "EventLog is the only source of truth and it is empty"
            )

        limit, limit_source = self._effective_step_limit(budget)
        t0 = time.perf_counter()

        while True:
            state = self._log.fold()
            if state.finished:
                break

            # Cutoffs, checked before every agent step.
            if state.steps >= limit:
                self._append_finish(reason=limit_source)
                break
            if self._config.max_tokens and self._total_tokens(state) >= self._config.max_tokens:
                self._append_finish(reason="max_tokens")
                break
            if budget is not None:
                # Budget integration on the SAME BudgetState object: fold
                # wall-clock consumed so far into the shared budget and let
                # Orcha's planner decide whether to stop (completed
                # iterations are counted after execution, matching the
                # orchestrator's semantics). No parallel budget path.
                budget.latency_used_s = time.perf_counter() - t0
                plan = self._planner.plan(self._budget_packet(budget))
                if plan.payload.get("stop"):
                    self._append_finish(reason="budget")
                    break

            # One model turn: the agent may return several actions (e.g.
            # multiple tool calls) — execute them one at a time. The model
            # sees a windowed summary of the log (ContextMemory), never
            # the raw EventLog. When a compactor is attached, the slice is
            # first passed through the ladder (microcompact, then — only
            # above the hard limit — LLM summarization carried on the
            # memory projection). The authoritative log is untouched.
            log_slice = self._log.slice(after_seq=state.last_user_seq)
            memory = self._memory_builder.build(self._log.events)
            if self._compactor is not None and len(log_slice) > 0:
                log_slice, memory, compact_stats = await self._compact_view(
                    log_slice, memory,
                )
                if compact_stats:
                    diag_log(
                        logger, "conversation", "compaction",
                        seq=state.seq, **compact_stats,
                    )
            actions = await self._agent.step_batch(
                self._agent_config, log_slice, memory=memory,
            )

            # Guard: if the agent returns no actions, treat as a finish to
            # avoid spinning until the step cap.
            if not actions:
                self._append(FinishAction(
                    content=_graceful_limits_answer(
                        "empty_actions", self.state.steps,
                        sum(self._tool_call_counts.values()),
                    ),
                    reason="no_actions",
                ))
                break

            # Observability only: the step's decision + memory projection.
            diag_log(
                logger, "conversation", "step",
                seq=state.seq,
                actions=",".join(a.kind for a in actions),
                memory_steps=len(memory.steps),
                memory_facts=len(memory.facts),
                context_chars=_context_chars(log_slice),
                budget_iterations=getattr(budget, "iterations", None),
            )

            terminated = False
            made_progress = False
            for action in actions:
                if isinstance(action, FinishAction):
                    # Final-answer semantics: a tool-call JSON object (or
                    # empty text) can never be the user-facing answer.
                    clean = sanitize_final_answer(action.content)
                    if clean is None:
                        action = FinishAction(
                            content=_graceful_limits_answer(
                                "empty_answer", self.state.steps,
                                sum(self._tool_call_counts.values()),
                            ),
                            reason=action.reason,
                            tokens=action.tokens,
                            reasoning_tokens=action.reasoning_tokens,
                        )
                    elif _has_intra_message_repetition(clean):
                        action = FinishAction(
                            content=_graceful_limits_answer(
                                "text_repetition", self.state.steps,
                                sum(self._tool_call_counts.values()),
                            ),
                            reason=action.reason,
                            tokens=action.tokens,
                            reasoning_tokens=action.reasoning_tokens,
                        )
                    self._append(action)
                    terminated = True
                    made_progress = True
                    break

                if isinstance(action, MessageAction):
                    # User-facing chatter: log it, keep looping. No
                    # workspace round-trip, no observation.
                    self._append(action)
                    made_progress = True
                    # Intra-message repetition guard: detect degenerate
                    # word/phrase loops within a single message (e.g.
                    # "and my and my and my ..."). This catches the classic
                    # small-model degenerate loop before it even accumulates
                    # across multiple messages.
                    if _has_intra_message_repetition(action.content):
                        diag_log(
                            logger, "conversation", "intra_message_repetition_detected",
                            sample=action.content[:200],
                        )
                        # Replace the message content with a graceful stop.
                        self._log.events[-1] = self._log.events[-1].model_copy(
                            update={"payload": MessageAction(
                                content=_graceful_limits_answer(
                                    "text_repetition", self.state.steps,
                                    sum(self._tool_call_counts.values()),
                                ),
                                tokens=action.tokens,
                                reasoning_tokens=action.reasoning_tokens,
                            )}
                        )
                        self._append_finish(reason="text_repetition")
                        terminated = True
                        break
                    # Text-repetition guard: track recent message content
                    # and stop when the model is clearly looping.
                    self._recent_message_texts.append(action.content)
                    if len(self._recent_message_texts) > self._config.max_message_repeats:
                        self._recent_message_texts = self._recent_message_texts[-self._config.max_message_repeats:]
                    if (
                        len(self._recent_message_texts) >= self._config.max_message_repeats
                        and _texts_are_repetitive(self._recent_message_texts)
                    ):
                        diag_log(
                            logger, "conversation", "text_repetition_detected",
                            repeats=self._config.max_message_repeats,
                            sample=self._recent_message_texts[-1][:120],
                        )
                        self._append_finish(reason="text_repetition")
                        terminated = True
                        break
                    continue

                if isinstance(action, ErrorAction):
                    # Agent aborts: report the error back through the
                    # workspace so the log records an ErrorObservation,
                    # then stop.
                    self._execute_and_append(action, budget)
                    self._append_finish(reason="error")
                    terminated = True
                    break

                if isinstance(action, ToolCallAction):
                    # Loop guard: identical (tool, args) calls are counted;
                    # a tool that keeps returning the same dead-end result
                    # is marked unavailable so the model must pick another
                    # tool (or answer directly).
                    self._recent_message_texts.clear()  # tool calls break the repetition streak
                    if self._execute_or_mark(action, budget):
                        made_progress = True
                    continue

                # Unknown action type: treat as a system error and stop.
                self._append(action)
                self._append_finish(reason="error")
                terminated = True
                break

            if terminated:
                break

            # Loop guard: the model only called tools that are gone (or the
            # registry was empty) — it cannot make progress, so finish with
            # a graceful answer instead of spinning against an empty tool
            # surface. Checked AFTER the action loop so a model error or a
            # real finish is honored first.
            if not made_progress and self._available_tools() == 0:
                self._append_finish(reason="all_tools_unavailable")
                break

            # Loop guard: N consecutive turns that produced only tool calls
            # (no message, no finish) mean the model is spinning — stop
            # with a graceful answer instead of running to the step cap.
            if not made_progress:
                self._stalled_turns += 1
                if self._stalled_turns >= self._config.max_stalled_turns:
                    self._append_finish(reason="stalled")
                    break
            else:
                self._stalled_turns = 0

            # One iteration completed: count it on the shared budget so the
            # planner's progress/exhaustion view stays exact.
            if budget is not None:
                budget.iterations += 1

        state = self._log.fold()
        diag_log(
            logger, "conversation", "finish",
            terminated_by=state.finish_reason or "finish",
            steps=state.steps,
            tokens=state.tokens_used + state.reasoning_tokens_used,
            events=len(self._log.events),
            unavailable=sorted(self._unavailable),
            tool_calls=sum(self._tool_call_counts.values()),
        )
        self.last_result = ConversationResult(
            query=self._query,
            answer=state.answer,
            terminated_by="finish" if state.finish_reason in (None, "done")
            else (state.finish_reason or "finish"),
            steps=state.steps,
            tokens_used=state.tokens_used + state.reasoning_tokens_used,
            reasoning_tokens_used=state.reasoning_tokens_used,
            events=self._log.events,
            log_state=state,
        )
        return self.last_result

    # ── Internals ─────────────────────────────────────────────────────

    async def _compact_view(
        self,
        log_slice: Tuple[Event, ...],
        memory: ContextMemory,
    ) -> "Tuple[Tuple[Event, ...], ContextMemory, Dict[str, Any]]":
        """
        The compaction ladder for one model turn (pure — never mutates the
        log). Rung 1 is a no-op below the soft limit; rung 2 microcompacts
        old compactable tool results; rung 3 summarizes into the memory
        projection once still above the hard limit.
        """
        assert self._compactor is not None
        compactor = self._compactor
        stats: Dict[str, Any] = {}
        projected, micro_stats = compactor.project(log_slice)
        if micro_stats:
            stats.update(micro_stats)
            projected_events: Tuple[Event, ...] = projected
        else:
            projected_events = log_slice
        if compactor.should_autocompact(projected_events):
            summary, transcript_path = await compactor.autocompact(projected_events)
            if summary:
                memory = memory.with_summary(summary)
                stats.update(
                    autocompacted=True,
                    slice_events_dropped=len(projected_events),
                    transcript=transcript_path or "",
                )
                # Summary now carries the history; hand the agent only the
                # newest protected tail so the raw text stays bounded.
                tail = compactor.config.protected_tail_pairs * 2 + 4
                projected_events = projected_events[-tail:]
                stats["tail_events"] = len(projected_events)
            else:
                stats.update(autocompact_failed=True)
        return projected_events, memory, stats

    def _effective_step_limit(
        self, budget: Optional[BudgetState],
    ) -> Tuple[int, str]:
        """(limit, source) — source is "max_steps" or "budget"."""
        limit = self._config.max_steps
        source = "max_steps"
        if budget is not None and 0 < budget.max_iterations < limit:
            limit = budget.max_iterations
            source = "budget"
        return limit, source

    def _budget_packet(self, budget: BudgetState) -> OrchaPacket:
        """Assemble the packet the agent loop presents to
        ``BudgetPlanner.plan()`` — the same bus packet shape the
        orchestration pipeline uses, carrying the SAME BudgetState
        object, so the stop decision is exactly the planner's (shared
        exhaustion semantics, no parallel budget path)."""
        return action_packet(
            FinishAction(content="", reason="budget"),
            query=self._query,
            seq=self._log.fold().seq + 1,
            tokens=0,
            budget=budget,
        ).stamp("agent.loop", 0.0)

    @staticmethod
    def _total_tokens(state: LogState) -> int:
        """The budget charge of a log state: tokens + reasoning tokens,
        folded into one accounting."""
        return state.tokens_used + state.reasoning_tokens_used

    def _append(self, action: Action) -> None:
        """Append an action event, honoring the action's own token cost
        (from the backend) with a deterministic estimate as fallback."""
        self._log.append(
            action,
            tokens=action.tokens or estimate_tokens(self._action_text(action)),
            reasoning_tokens=action.reasoning_tokens,
        )

    def _append_finish(self, reason: str) -> None:
        """System-side termination (cutoff / loop guard): record it as an
        event so the log stays the complete, replayable record. The finish
        ALWAYS carries a real natural-language answer — the most recent
        assistant message content when the agent already wrote one, else a
        graceful explanation. Never empty, never a tool-call JSON object."""
        content = self._last_assistant_message()
        if content is None:
            content = _graceful_limits_answer(
                reason, self.state.steps, sum(self._tool_call_counts.values()),
            )
        else:
            content = sanitize_final_answer(content) or _graceful_limits_answer(
                reason, self.state.steps, sum(self._tool_call_counts.values()),
            )
        self._log.append(FinishAction(content=content, reason=reason))

    def _last_assistant_message(self) -> Optional[str]:
        """The most recent assistant MessageAction content, if any."""
        for ev in reversed(self._log.events):
            payload = ev.payload
            if getattr(payload, "kind", None) == "message" and payload.content:
                return payload.content
        return None

    def _available_tools(self) -> int:
        """Count of tools still available to the model this run."""
        registry = getattr(self._agent, "tools", None)
        if registry is not None:
            return len([n for n in registry.names() if n not in self._unavailable])
        return 0  # pragma: no cover — fallback

    def _execute_or_mark(
        self, action: ToolCallAction, budget: Optional[BudgetState],
    ) -> bool:
        """
        Execute one tool call — unless the loop guard has marked the tool
        unavailable, in which case a synthetic observation is appended and
        the call is NOT executed. Returns True when real progress was made.
        """
        name = action.name
        # Use repr(sorted items) for a stable key — faster than json.dumps
        # with sort_keys=True on every tool call.
        key = (name, repr(sorted((action.arguments or {}).items())))
        self._tool_call_counts[key] = self._tool_call_counts.get(key, 0) + 1

        if name in self._unavailable:
            self._append(action)
            self._log.append(
                ToolResultObservation(
                    tool_call_id=action.tool_call_id,
                    content=(
                        f"[tool unavailable] {name} is unavailable in this "
                        "runtime — choose another tool or answer directly."
                    ),
                    success=False,
                )
            )
            return False

        repeats = self._tool_call_counts[key]
        if repeats > self._config.max_tool_repeats:
            self._mark_unavailable(
                name,
                f"identical call repeated {repeats} times",
            )
            self._append(action)
            self._log.append(
                ToolResultObservation(
                    tool_call_id=action.tool_call_id,
                    content=(
                        f"[tool unavailable] {name} was called {repeats} times "
                        "with the same arguments and did not help — it is "
                        "removed for the rest of this run. Choose another "
                        "tool or answer directly."
                    ),
                    success=False,
                )
            )
            return False

        self._execute_and_append(action, budget)
        return True

    def _mark_unavailable(self, name: str, reason: str) -> None:
        """Mark a tool unavailable: drop it from the registry so every
        future schema excludes it, and log the decision."""
        if name in self._unavailable:
            return
        self._unavailable.add(name)
        removed = False
        registry = getattr(self._agent, "tools", None)
        if registry is not None:
            removed = registry.remove(name)
        # Invalidate the agent's schema cache so the removed tool's
        # schema is no longer sent to the model on subsequent steps.
        if removed and hasattr(self._agent, "_cached_schemas"):
            self._agent._cached_schemas = None
        diag_log(
            logger, "conversation", "tool_unavailable",
            tool=name, reason=reason, removed=removed,
        )

    def _execute_and_append(
        self, action: Any, budget: Optional[BudgetState],
    ) -> None:
        """Route an action across the bus to the workspace, then append the
        action event and its observation event to the log."""
        state = self._log.fold()
        packet = action_packet(
            action,
            query=self._query,
            seq=state.seq + 1,
            tokens=action.tokens or estimate_tokens(self._action_text(action)),
            budget=budget,
        ).stamp("agent.step", 0.0, action=action.kind)

        result_packet = self._workspace.execute(packet)
        observation = observation_from_packet(result_packet)

        self._append(action)
        self._log.append(
            observation, tokens=estimate_tokens(self._observation_text(observation)),
        )

    @staticmethod
    def _action_text(action: Any) -> str:
        if isinstance(action, ToolCallAction):
            return action.name + str(action.arguments)
        return str(action)

    @staticmethod
    def _observation_text(observation: Any) -> str:
        return getattr(observation, "content", None) or getattr(
            observation, "message", ""
        ) or str(observation)


def _context_chars(events: Sequence[Event]) -> int:
    """Observability-only: rough context size (chars) of a turn slice."""
    total = 0
    for ev in events:
        try:
            total += len(ev.payload.model_dump_json())
        except Exception:
            total += len(str(ev))
    return total


__all__ = [
    "ConversationConfig", "ConversationResult", "Conversation",
]
