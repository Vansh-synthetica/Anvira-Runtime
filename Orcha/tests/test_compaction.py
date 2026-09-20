"""
Tests for Phase-2 context compaction (the ladder):

- estimate_slice_tokens charged accounting + fallback estimation
- microcompact: compactable-only, protected tail, pairing preserved,
  purity (input events untouched)
- TranscriptStore JSONL dumps
- Compactor circuit breaker + degradation
- Conversation integration: projections only — log replay stays identical
"""
import json
import os

import pytest

from orcha.agent_runtime.compaction import (
    COMPACTABLE_TOOLS, CompactionConfig, Compactor, TranscriptStore,
    estimate_slice_tokens, microcompact,
)
from orcha.agent_runtime.conversation import Conversation
from orcha.agent_runtime.events import (
    ErrorObservation, EventLog, ToolCallAction, ToolResultObservation,
    UserMessageObservation, fold_log,
)
from orcha.agent_runtime.memory import ContextMemory, render_memory_section
from orcha.agent_runtime.workspace import Workspace


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tool_pair(log: EventLog, name="read_file", content="x" * 400, call_id=None, success=True):
    action = log.append(ToolCallAction(name=name, arguments={"path": "a.txt"}, tool_call_id=call_id))
    log.append(
        ToolResultObservation(tool_call_id=call_id, content=content, success=success),
        tokens=max(1, len(content) // 4),
    )
    return action


def _slice(events):
    return tuple(events)


class _NullWorkspace(Workspace):
    def execute(self, packet):
        raise AssertionError("no tools should execute in these tests")

    async def execute_async(self, packet):  # pragma: no cover
        raise AssertionError("no tools should execute in these tests")


# ── Token estimation ──────────────────────────────────────────────────────────

def test_estimate_uses_charged_tokens_first():
    log = EventLog()
    ev = log.append(UserMessageObservation(content="hello world"), tokens=50)
    assert estimate_slice_tokens([ev]) == 50


def test_estimate_falls_back_to_chars_when_uncharged():
    log = EventLog()
    ev = log.append(UserMessageObservation(content="a" * 400))  # no charge
    assert estimate_slice_tokens([ev]) == 100  # len//4


# ── Microcompact ──────────────────────────────────────────────────────────────

def _build_log(pairs=6):
    log = EventLog()
    for i in range(pairs):
        _tool_pair(log, call_id=f"c{i}")
    return log


def test_microcompact_clears_old_compactable_results():
    log = _build_log(6)
    projected, cleared, freed = microcompact(_slice(log.events), protected_tail_pairs=3)
    assert cleared == 3 and freed > 0
    contents = [ev.payload.content for ev in projected
                if isinstance(ev.payload, ToolResultObservation)]
    assert contents[:3] == ["[older tool result cleared to save context]"] * 3
    assert contents[3:] != ["[older tool result cleared to save context]"] * 3


def test_microcompact_respects_protected_tail():
    log = _build_log(2)
    projected, cleared, freed = microcompact(_slice(log.events), protected_tail_pairs=3)
    assert cleared == 0 and freed == 0
    assert projected == tuple(log.events)


def test_microcompact_skips_errors_and_writes():
    log = EventLog()
    _tool_pair(log, name="write_file", call_id="w1")
    _tool_pair(log, name="run_command", content="boom", call_id="r1", success=False)
    projected, cleared, _ = microcompact(_slice(log.events), protected_tail_pairs=0)
    assert cleared == 0
    contents = [ev.payload.content for ev in projected
                if isinstance(ev.payload, ToolResultObservation)]
    # write result and error both survive verbatim
    assert "boom" in contents
    assert "[older tool result cleared" not in "".join(contents)


def test_microcompact_is_pure():
    log = _build_log(5)
    before = [ev.payload.model_dump() for ev in log.events]
    microcompact(_slice(log.events), protected_tail_pairs=1)
    after = [ev.payload.model_dump() for ev in log.events]
    assert before == after


def test_microcompact_preserves_event_kinds_and_seq():
    log = _build_log(4)
    projected, _, _ = microcompact(_slice(log.events), protected_tail_pairs=1)
    assert [(ev.seq, ev.kind) for ev in projected] == [(ev.seq, ev.kind) for ev in log.events]


# ── TranscriptStore ───────────────────────────────────────────────────────────

def test_transcript_dump_roundtrip(tmp_path):
    store = TranscriptStore(str(tmp_path / "t"))
    log = _build_log(3)
    path = store.dump(tuple(log.events))
    assert os.path.isfile(path)
    lines = [json.loads(l) for l in open(path, encoding="utf-8").read().splitlines() if l]
    assert len(lines) == len(log.events)
    assert lines[0]["payload"]["kind"] == "user_message" or "kind" in lines[0]["payload"]


# ── Compactor decisions ───────────────────────────────────────────────────────

def test_project_noop_below_soft_limit():
    compactor = Compactor(config=CompactionConfig(soft_limit_tokens=999_999))
    log = _build_log(3)
    projected, stats = compactor.project(_slice(log.events))
    assert projected == tuple(log.events) and stats == {}


class _ExplodingBackend:
    model_name = "fake"

    async def complete(self, messages, tools=None, config=None):
        raise RuntimeError("backend down")


class _EchoBackend:
    model_name = "fake"

    def __init__(self):
        self.calls = 0

    async def complete(self, messages, tools=None, config=None):
        self.calls += 1

        class R:
            text = "<summary>1. Primary request: demo</summary>"

        return R()


@pytest.mark.anyio
async def test_autocompact_failure_ticks_breaker_then_disables():
    cfg = CompactionConfig(max_consecutive_failures=2, min_summary_chars=10)
    compactor = Compactor(_ExplodingBackend(), config=cfg)
    log = _build_log(4)
    s1, _ = await compactor.autocompact(_slice(log.events))
    assert s1 is None and not compactor.disabled
    s2, _ = await compactor.autocompact(_slice(log.events))
    assert s2 is None and compactor.disabled  # breaker open
    s3, _ = await compactor.autocompact(_slice(log.events))
    assert s3 is None  # stands down without calling backend again


@pytest.mark.anyio
async def test_autocompact_success_returns_summary_and_resets_breaker(tmp_path):
    backend = _EchoBackend()
    compactor = Compactor(
        backend,
        config=CompactionConfig(min_summary_chars=10),
        transcript_store=TranscriptStore(str(tmp_path / "t")),
    )
    summary, transcript = await compactor.autocompact(_slice(_build_log(2).events))
    assert summary and "Primary request" in summary
    assert transcript and os.path.isfile(transcript)
    assert "Full pre-compaction transcript saved to" in summary
    assert not compactor.disabled


def test_render_memory_section_includes_summary():
    memory = ContextMemory(summary="Earlier work summary.")
    rendered = render_memory_section(memory, max_context_tokens=8000)
    assert "[Earlier conversation]" in rendered
    assert "Earlier work summary." in rendered


def test_with_summary_replaces_purely():
    base = ContextMemory(facts=("f",))
    upgraded = base.with_summary("s")
    assert upgraded.summary == "s" and upgraded.facts == ("f",)
    assert base.summary == ""  # original untouched
    assert base.with_summary("") is base  # no-op returns same frozen object


# ── Conversation integration ──────────────────────────────────────────────────

class _StubAgent:
    """Emits one FinishAction immediately; records what it was shown."""

    def __init__(self):
        self.seen_slices = []
        self.seen_memories = []

    async def step_batch(self, config, log_slice, memory=None):
        self.seen_slices.append(tuple(log_slice))
        self.seen_memories.append(memory)
        from orcha.agent_runtime.events import FinishAction
        return [FinishAction(content="done")]


@pytest.mark.anyio
async def test_conversation_without_compactor_is_unchanged():
    agent = _StubAgent()
    convo = Conversation(agent, _NullWorkspace())
    convo.submit_user_message("hi")
    result = await convo.run()
    assert result.completed
    assert agent.seen_memories[0].summary == ""


@pytest.mark.anyio
async def test_conversation_compactor_projection_never_mutates_log():
    agent = _StubAgent()
    compactor = Compactor(
        _EchoBackend(),
        config=CompactionConfig(
            soft_limit_tokens=10, hard_limit_tokens=20, min_summary_chars=5,
        ),
        transcript_store=TranscriptStore(os.path.join(
            os.environ.get("TEMP", "/tmp"), "orcha_test_transcripts",
        )),
    )
    convo = Conversation(agent, _NullWorkspace(), compactor=compactor)
    convo.submit_user_message("please do the long thing")
    for i in range(6):
        _tool_pair(convo.log, call_id=f"c{i}")
    before_events = tuple(convo.log.events)
    before_fold = fold_log(before_events)
    result = await convo.run()
    after_events = tuple(convo.log.events)

    # Log untouched by compaction; replay identical.
    assert len(after_events) >= len(before_events)
    for b, a in zip(before_events, after_events):
        assert b.seq == a.seq and b.payload == a.payload
    assert fold_log(after_events).seq == before_fold.seq + 1  # + finish event

    # The agent saw either the raw slice or a compacted projection with a
    # summary attached.
    seen = agent.seen_memories[-1]
    assert seen is not None
