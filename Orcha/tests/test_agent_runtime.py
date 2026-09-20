"""
Tests for orcha.agent_runtime — the event-sourced AgentRuntime skeleton.

Focus areas (per the founding contract):
- Deterministic replay: folding an EventLog reconstructs identical state
  with zero side effects (pure fold over events).
- The OrchaPacket bus is the transport for Actions/Observations (no
  parallel bus).
- Conversation honors FinishAction, max-steps, and token-budget cutoffs,
  and rides Orcha's existing BudgetState when supplied.
- The Agent is stateless: step(config, log_slice) is a pure function
  (async since Prompt 2; Conversation.run() is awaited).
"""
from typing import Sequence

import pytest
from pydantic import ValidationError

from orcha import PacketKind, BudgetState, OrchaPacket
from orcha.agent_runtime import (
    Agent, AgentConfig, Conversation, ConversationConfig, ConversationResult,
    EmptyLogError, ErrorAction, ErrorObservation, Event, EventKind, EventLog,
    FinishAction, LogState, MessageAction, Observation, StubAgent,
    StubWorkspace, ToolCallAction, ToolResultObservation,
    UserMessageObservation, action_from_packet, action_packet,
    estimate_tokens, event_from_packet, event_to_packet, fold_log,
    observation_from_packet, observation_packet,
)


# ── Test doubles (stateless by contract) ────────────────────────────────────

class ScriptedAgent(Agent):
    """
    Stateless scripted agent: picks the next action from the number of
    agent steps already taken inside the current turn's slice. Deriving
    the decision purely from the slice keeps the double deterministic and
    honest about the stateless contract.
    """

    def __init__(self, actions):
        self._actions = tuple(actions)

    async def step(self, config: AgentConfig, log_slice: Sequence[Event]) -> object:
        state = fold_log(log_slice)
        if state.actions < len(self._actions):
            return self._actions[state.actions]
        return FinishAction(content="scripted done", reason="done")


class AlwaysToolAgent(Agent):
    """Never finishes: exercises the step/token/budget cutoffs."""

    async def step(self, config: AgentConfig, log_slice: Sequence[Event]) -> object:
        return ToolCallAction(name="do", arguments={})


class RecordingWorkspace(StubWorkspace):
    def __init__(self):
        self.calls = 0

    def execute(self, packet):
        self.calls += 1
        return super().execute(packet)


def _sized_payload_log():
    """A fabricated EventLog with known token costs and structure."""
    log = EventLog()
    log.append(UserMessageObservation(content="read the folder"), tokens=10)
    log.append(ToolCallAction(name="ls", arguments={"path": "/tmp"}), tokens=20)
    log.append(ToolResultObservation(content="a.txt  b.txt", success=True), tokens=5)
    log.append(FinishAction(content="all done", reason="done"), tokens=0)
    return log


# ── The fold: deterministic replay ──────────────────────────────────────────

def test_fold_reconstructs_exact_state():
    state = _sized_payload_log().fold()

    assert state.seq == 4
    assert state.events == 4
    assert state.actions == 2              # ls + finish
    assert state.steps == 1                # only the non-finish action counts
    assert state.observations == 2         # user message + tool result
    assert state.tokens_used == 35
    assert state.last_user_seq == 1
    assert state.finished is True
    assert state.finish_reason == "done"
    assert state.answer == "all done"
    assert isinstance(state.last_action, FinishAction)
    assert isinstance(state.last_observation, ToolResultObservation)


def test_fold_is_deterministic():
    log = _sized_payload_log()
    events = log.events

    s1 = fold_log(events)
    s2 = fold_log(events)            # same input, twice
    s3 = fold_log(tuple(events))     # same events via a different container
    s4 = EventLog.replay(events)     # replay entry point

    assert s1 == s2 == s3 == s4
    assert log.fold() == s1          # the live log folds identically


def test_fold_has_zero_side_effects():
    log = _sized_payload_log()
    events = log.events
    snapshot_before = [ev.model_dump() for ev in events]

    fold_log(events)
    fold_log(events)

    assert [ev.model_dump() for ev in events] == snapshot_before
    assert log.fold().events == 4    # the live log was not touched either


def test_fold_with_start_copies_dont_share_state():
    base = LogState(tokens_used=7, steps=2)
    first = fold_log(_sized_payload_log().events, start=base)
    assert first.tokens_used == 7 + 35
    assert first.steps == 2 + 1
    assert base.tokens_used == 7      # caller's start object untouched


def test_replay_reconstructs_identical_state_from_captured_log():
    log = _sized_payload_log()
    captured = tuple(log.events)      # e.g. what a client would persist

    # Replaying the captured events must rebuild the exact same state
    # without the original log being present at all.
    replayed = EventLog.replay(captured)
    assert replayed == log.fold()
    assert replayed == fold_log(captured)

    # And the captured sequence is untouched by the replay.
    assert tuple(log.events) == captured


def test_eventlog_is_append_only():
    log = EventLog()
    log.append(UserMessageObservation(content="hi"))
    first_snapshot = log.events

    log.append(ToolCallAction(name="ls", arguments={}))

    assert len(log) == 2
    assert len(first_snapshot) == 1   # old snapshots never change
    assert log.events[0].seq == 1 and log.events[1].seq == 2
    assert not hasattr(log, "remove")
    assert not hasattr(log, "clear")
    with pytest.raises(TypeError):
        log.events[0] = log.events[1]  # tuple snapshot is immutable


def test_eventlog_slice_after_seq_and_limit():
    log = _sized_payload_log()
    assert [ev.seq for ev in log.slice(after_seq=2)] == [3, 4]
    assert [ev.seq for ev in log.slice(after_seq=1, limit=2)] == [2, 3]
    assert log.slice(after_seq=0) == log.events


def test_estimate_tokens_is_deterministic():
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 100) == 25


# ── The OrchaPacket bus ──────────────────────────────────────────────────────

def test_action_round_trips_through_packet_bus():
    act = ToolCallAction(name="read", arguments={"path": "x.txt"}, tool_call_id="t1")
    pkt = action_packet(act, query="q", seq=7, tokens=9)

    assert pkt.kind == PacketKind.ACTION
    assert pkt.query == "q"
    assert pkt.payload["event_seq"] == 7
    assert pkt.metadata["tokens"] == 9
    assert action_from_packet(pkt) == act


def test_observation_round_trips_through_packet_bus():
    obs = ToolResultObservation(content="ok", success=True)
    pkt = observation_packet(obs, query="q", seq=8)

    assert pkt.kind == PacketKind.OBSERVATION
    assert observation_from_packet(pkt) == obs


def test_bus_packets_are_json_serialisable():
    pkt = action_packet(
        ToolCallAction(name="read", arguments={"path": "x.txt"}),
        query="q", seq=3, tokens=7,
    )
    pkt2 = OrchaPacket.from_json(pkt.to_json())
    assert action_from_packet(pkt2) == ToolCallAction(
        name="read", arguments={"path": "x.txt"},
    )
    assert pkt2.kind == PacketKind.ACTION


def test_event_round_trips_through_packet_bus():
    for payload in (
        MessageAction(content="hi"),
        ToolResultObservation(content="result", success=False),
    ):
        ev = Event(seq=5, kind=EventKind.ACTION, payload=payload, tokens=4) \
            if isinstance(payload, MessageAction) else \
            Event(seq=5, kind=EventKind.OBSERVATION, payload=payload, tokens=4)
        pkt = event_to_packet(ev, query="q")
        back = event_from_packet(pkt)
        assert back.model_dump(exclude={"ts"}) == ev.model_dump(exclude={"ts"})


def test_packet_helpers_reject_mismatched_payloads():
    obs_pkt = observation_packet(UserMessageObservation(content="x"))
    with pytest.raises(Exception):
        action_from_packet(obs_pkt)


# ── The stateless Agent ──────────────────────────────────────────────────────

async def test_agent_is_stateless():
    cfg = AgentConfig(model="stub-model")
    sl = (Event(seq=1, kind=EventKind.USER_MESSAGE,
                payload=UserMessageObservation(content="hi")),)

    a1, a2 = StubAgent(), StubAgent()
    r1 = await a1.step(cfg, sl)
    r2 = await a2.step(cfg, sl)
    r3 = await a1.step(cfg, sl)  # same agent, again — identical output

    assert isinstance(r1, FinishAction)
    assert r1 == r2 == r3


def test_agent_config_is_immutable():
    cfg = AgentConfig()
    with pytest.raises(ValidationError):
        cfg.model = "other"


# ── The Conversation loop ────────────────────────────────────────────────────

async def test_conversation_appends_action_and_observation_events_per_step():
    agent = ScriptedAgent([ToolCallAction(name="ls", arguments={"path": "/"})])
    convo = Conversation(agent, StubWorkspace(), query="q")
    convo.submit_user_message("list the root")

    result: ConversationResult = await convo.run()

    kinds = [ev.kind for ev in result.events]
    assert kinds == [
        EventKind.USER_MESSAGE,
        EventKind.ACTION, EventKind.OBSERVATION,   # ls → tool result
        EventKind.ACTION,                          # finish
    ]
    assert isinstance(result.events[1].payload, ToolCallAction)
    obs = result.events[2].payload
    assert isinstance(obs, ToolResultObservation)
    assert obs.success and obs.content.startswith("[stub:ls]")
    assert isinstance(result.events[3].payload, FinishAction)
    assert result.steps == 1
    assert result.terminated_by == "finish"
    assert result.completed is True
    assert result.answer == "scripted done"


async def test_conversation_message_actions_skip_the_workspace():
    ws = RecordingWorkspace()
    agent = ScriptedAgent([MessageAction(content="thinking out loud")])
    convo = Conversation(agent, ws, query="q")
    convo.submit_user_message("hi")

    result = await convo.run()

    assert ws.calls == 0                       # never reached the bus
    assert result.steps == 1
    assert isinstance(result.events[1].payload, MessageAction)
    assert result.terminated_by == "finish"


async def test_conversation_runs_multi_step_script():
    actions = [
        ToolCallAction(name="ls", arguments={"path": "/"}),
        MessageAction(content="found the files"),
        ToolCallAction(name="read", arguments={"path": "/a.txt"}),
    ]
    convo = Conversation(ScriptedAgent(actions), StubWorkspace(), query="q")
    convo.submit_user_message("go through this folder")

    result = await convo.run()

    assert result.steps == 3
    assert [type(ev.payload).__name__ for ev in result.events] == [
        "UserMessageObservation",
        "ToolCallAction", "ToolResultObservation",
        "MessageAction",
        "ToolCallAction", "ToolResultObservation",
        "FinishAction",
    ]
    assert result.terminated_by == "finish"
    assert result.events[-1].payload.content == "scripted done"


async def test_error_action_terminates_with_error_observation():
    class FailingAgent(Agent):
        async def step(self, config, log_slice):
            return ErrorAction(message="cannot reach the model")

    convo = Conversation(FailingAgent(), StubWorkspace(), query="q")
    convo.submit_user_message("go")

    result = await convo.run()

    assert isinstance(result.events[-2].payload, ErrorObservation)
    assert isinstance(result.events[-1].payload, FinishAction)
    assert result.terminated_by == "error"
    assert result.completed is False


async def test_run_requires_a_user_message_first():
    convo = Conversation(StubAgent(), StubWorkspace(), query="q")
    with pytest.raises(EmptyLogError):
        await convo.run()


# ── Cutoffs: max-steps, token budget, packet BudgetState ────────────────────

async def test_max_steps_cutoff():
    convo = Conversation(
        AlwaysToolAgent(), StubWorkspace(), query="q",
        config=ConversationConfig(max_steps=3, max_tool_repeats=100),
    )
    convo.submit_user_message("go")

    result = await convo.run()

    assert result.steps == 3
    assert result.terminated_by == "max_steps"
    assert result.completed is False
    assert isinstance(result.events[-1].payload, FinishAction)
    assert result.events[-1].payload.reason == "max_steps"
    # every step left an action + observation pair
    assert len(result.events) == 1 + 3 * 2 + 1


async def test_token_budget_cutoff():
    class FatToolAgent(Agent):
        async def step(self, config, log_slice):
            return ToolCallAction(
                name="big", arguments={"blob": "x" * 300},
            )

    convo = Conversation(
        FatToolAgent(), StubWorkspace(), query="q",
        config=ConversationConfig(max_tokens=80),
    )
    convo.submit_user_message("go")

    result = await convo.run()

    assert result.terminated_by == "max_tokens"
    assert result.tokens_used >= 80
    assert result.events[-1].payload.reason == "max_tokens"


async def test_packet_budget_max_iterations_is_honored():
    convo = Conversation(AlwaysToolAgent(), StubWorkspace(), query="q")
    convo.submit_user_message("go")

    result = await convo.run(budget=BudgetState(max_iterations=2))

    assert result.steps == 2
    assert result.terminated_by == "budget"
    assert result.events[-1].payload.reason == "budget"


async def test_pre_exhausted_packet_budget_stops_immediately():
    convo = Conversation(AlwaysToolAgent(), StubWorkspace(), query="q")
    convo.submit_user_message("go")

    result = await convo.run(budget=BudgetState(max_iterations=0))

    assert result.steps == 0
    assert result.terminated_by == "budget"


async def test_budget_and_config_take_the_tighter_step_limit():
    convo = Conversation(
        AlwaysToolAgent(), StubWorkspace(), query="q",
        config=ConversationConfig(max_steps=5, max_tool_repeats=100),
    )
    convo.submit_user_message("go")

    result = await convo.run(budget=BudgetState(max_iterations=3))

    assert result.steps == 3
    assert result.terminated_by == "budget"

    # budget looser than config → max_steps wins
    convo2 = Conversation(
        AlwaysToolAgent(), StubWorkspace(), query="q",
        config=ConversationConfig(max_steps=2, max_tool_repeats=100),
    )
    convo2.submit_user_message("go")
    result2 = await convo2.run(budget=BudgetState(max_iterations=10))
    assert result2.steps == 2
    assert result2.terminated_by == "max_steps"


# ── Deterministic end-to-end replay ──────────────────────────────────────────

async def test_two_fresh_conversations_fold_to_identical_state():
    actions = [
        ToolCallAction(name="ls", arguments={"path": "/"}),
        ToolCallAction(name="read", arguments={"path": "/a.txt"}),
    ]

    async def make_run():
        convo = Conversation(
            ScriptedAgent(actions), StubWorkspace(), query="q",
            config=ConversationConfig(max_steps=10),
        )
        convo.submit_user_message("go through this folder")
        return await convo.run()

    r1, r2 = await make_run(), await make_run()

    assert r1.log_state == r2.log_state
    assert r1.steps == r2.steps == 2
    assert r1.tokens_used == r2.tokens_used


async def test_replay_of_result_events_reconstructs_result_state():
    convo = Conversation(
        ScriptedAgent([ToolCallAction(name="ls", arguments={"path": "/"})]),
        StubWorkspace(), query="q",
    )
    convo.submit_user_message("go")
    result = await convo.run()

    # Replay the persisted events in a completely fresh runtime: the pure
    # fold must rebuild the exact same authoritative state.
    replayed = EventLog.replay(result.events)
    assert replayed == result.log_state
    assert replayed.tokens_used == result.tokens_used
    assert replayed.steps == result.steps

    # The result object's own log helper folds to the same state too.
    assert result.log.fold() == result.log_state


def test_workspace_stub_echoes_tool_calls():
    ws = StubWorkspace()
    out = ws.execute(
        action_packet(ToolCallAction(name="grep", arguments={"pattern": "x"}))
    )
    assert out.kind == PacketKind.OBSERVATION
    obs = observation_from_packet(out)
    assert isinstance(obs, ToolResultObservation)
    assert obs.success and "[stub:grep]" in obs.content
