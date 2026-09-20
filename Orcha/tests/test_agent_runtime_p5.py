"""
Tests for the Prompt 5 layer of orcha.agent_runtime: the SSE event stream
for Anvira.

- Wire format: event_to_wire is a THIN serialization of the Prompt-1
  internal event model (seq, kind, ts, tokens, reasoning_tokens, payload)
  enriched with the Prompt-4 step-memory metadata (action + observation
  events of a step carry the same final StepRecord dict).
- AgentEventStream: cursor-based catch-up (replay strictly after the
  cursor), live delivery, strict seq order, race purge when events are
  appended between subscribe and start, finish/close semantics.
- AgentSessionRegistry: session lifecycle, attach/detach, finished/error
  signaling terminating every attached stream.
- HTTP: /v1/agent-runs (create/start/status/events), shared error envelope,
  cursor replay on reconnect, end-to-end run over the stub backend.
- Smoke test: a fabricated session, stream, mid-stream disconnect, more
  events, reconnect with cursor — no lost, no duplicated, contiguous seqs.
"""
import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from orcha.agent_runtime import (
    AgentConfig, Conversation, ErrorObservation, FactObservation,
    FinishAction, MessageAction, StubAgent, StubWorkspace, ToolCallAction,
    ToolResultObservation, UserMessageObservation,
)
from orcha.agent_runtime.events import Event, EventKind
from orcha.agent_runtime.stream import (
    AgentEventStream, AgentSession, AgentSessionRegistry, event_to_wire,
)
from orcha.api.agent_stream import get_agent_session_registry
from orcha.api.server import app


# ── Fixtures / helpers ───────────────────────────────────────────────────────

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _make_convo(query: str = "fabricated") -> Conversation:
    return Conversation(StubAgent(), StubWorkspace(), query=query)


def _append(log, payload, tokens: int = 0, reasoning_tokens: int = 0) -> Event:
    return log.append(payload, tokens=tokens, reasoning_tokens=reasoning_tokens)


def _parse_sse_lines(resp) -> List[Dict[str, Any]]:
    return [
        json.loads(line[6:])
        for line in resp.iter_lines()
        if line.startswith("data: ")
    ]


def _wire_events(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fabricate a log from bare payload dicts, return wire events."""
    log = _make_convo().log
    for p in payloads:
        _append(log, p["payload"], tokens=p.get("tokens", 0),
                reasoning_tokens=p.get("reasoning_tokens", 0))
    return [event_to_wire(ev) for ev in log.events]


# ── Wire format ──────────────────────────────────────────────────────────────

def test_wire_tool_call_action():
    log = _make_convo().log
    ev = _append(log, ToolCallAction(name="read_file", arguments={"path": "a.txt"}), tokens=12, reasoning_tokens=4)
    wire = event_to_wire(ev)
    assert wire["seq"] == 1
    assert wire["kind"] == "action"
    assert wire["event_type"] == "tool_call"
    assert wire["tokens"] == 12
    assert wire["reasoning_tokens"] == 4
    assert isinstance(wire["ts"], float)
    assert wire["payload"]["name"] == "read_file"
    assert wire["payload"]["arguments"] == {"path": "a.txt"}
    assert "step" not in wire  # metadata only when supplied


def test_wire_tool_result_and_error_observations():
    log = _make_convo().log
    ok = _append(log, ToolResultObservation(tool_call_id="c1", content="the file", success=True), tokens=9)
    err = _append(log, ErrorObservation(message="boom"))
    ok_w, err_w = event_to_wire(ok), event_to_wire(err)
    assert ok_w["kind"] == "observation"
    assert ok_w["event_type"] == "tool_result"
    assert ok_w["payload"]["success"] is True
    assert ok_w["payload"]["content"] == "the file"
    assert ok_w["payload"]["tool_call_id"] == "c1"
    assert ok_w["tokens"] == 9
    assert err_w["event_type"] == "error"
    assert err_w["payload"]["message"] == "boom"


def test_wire_message_and_finish_actions():
    log = _make_convo().log
    msg = _append(log, MessageAction(content="still working"))
    fin = _append(log, FinishAction(content="final answer", reason="done"))
    msg_w, fin_w = event_to_wire(msg), event_to_wire(fin)
    assert msg_w["event_type"] == "message"
    assert msg_w["payload"]["content"] == "still working"
    assert fin_w["event_type"] == "finish"
    assert fin_w["payload"]["reason"] == "done"
    assert fin_w["payload"]["content"] == "final answer"


def test_wire_user_message_and_fact_observations():
    log = _make_convo().log
    user = _append(log, UserMessageObservation(content="hello"))
    fact = _append(log, FactObservation(content="user prefers python"))
    user_w, fact_w = event_to_wire(user), event_to_wire(fact)
    assert user_w["event_type"] == "user_message"
    assert user_w["payload"]["content"] == "hello"
    assert fact_w["event_type"] == "fact"
    assert fact_w["payload"]["content"] == "user prefers python"


def test_wire_json_serializable():
    log = _make_convo().log
    _append(log, ToolCallAction(name="run_command", arguments={"command": "ls"}))
    _append(log, ToolResultObservation(content="a  b  c", success=True))
    _append(log, FinishAction(content="done"))
    for ev in log.events:
        json.dumps(event_to_wire(ev))  # must not raise


def test_wire_step_metadata_maps_action_and_observation_to_same_record():
    convo = _make_convo()
    log = convo.log
    _append(log, UserMessageObservation(content="hello"))
    _append(log, ToolCallAction(name="echo", arguments={"text": "x"}), tokens=10, reasoning_tokens=5)
    _append(log, ToolResultObservation(content="x", success=True), tokens=20)
    session = AgentSession("s1", convo)
    records = session.step_records()
    action_seq = [ev.seq for ev in log.events if ev.kind == EventKind.ACTION][0]
    obs_seq = [ev.seq for ev in log.events
               if ev.kind == EventKind.OBSERVATION and ev.payload.kind == "tool_result"][0]
    assert records[action_seq]["step"] == 1
    assert records[action_seq]["tokens_in"] == 10
    assert records[action_seq]["reasoning_tokens"] == 5
    # the observation that closes the step carries the SAME final record
    assert records[obs_seq] == records[action_seq]
    assert records[obs_seq]["tokens_out"] == 20
    assert records[obs_seq]["action"] == "tool call: echo({'text': 'x'})"
    # user_message is not part of any step
    user_seq = [ev.seq for ev in log.events if ev.payload.kind == "user_message"][0]
    assert user_seq not in records


# ── AgentEventStream ─────────────────────────────────────────────────────────

def test_stream_replays_history_after_cursor():
    log = _make_convo().log
    for i in range(5):
        _append(log, UserMessageObservation(content=f"m{i}"))
    stream = AgentEventStream(log, cursor=2)
    stream.start()
    stream.finish()  # replay-only scenario: signal end after the catch-up

    async def drain():
        out = []
        while True:
            ev = await stream.next()
            if ev is None:
                return out
            out.append(ev.seq)

    assert asyncio.run(drain()) == [3, 4, 5]


def test_stream_delivers_live_events_in_seq_order():
    log = _make_convo().log
    stream = AgentEventStream(log)
    stream.start()

    async def consume():
        _append(log, UserMessageObservation(content="live 1"))
        _append(log, ToolCallAction(name="echo", arguments={"text": "a"}))
        _append(log, ToolResultObservation(content="a", success=True))
        seqs = []
        while len(seqs) < 3:
            ev = await stream.next()
            if ev is None:
                break
            seqs.append(ev.seq)
        return seqs

    # consume() runs on the loop, so the appends land on the same loop as
    # the stream's asyncio.Event — no cross-thread wakeup involved.
    assert asyncio.run(consume()) == [1, 2, 3]


def test_stream_no_duplicates_when_appended_before_start():
    """Events appended between subscribe (init) and start() must not be
    delivered twice (once queued live, once replayed)."""
    log = _make_convo().log
    stream = AgentEventStream(log, cursor=0)
    _append(log, UserMessageObservation(content="a"))
    _append(log, UserMessageObservation(content="b"))
    stream.start()
    stream.finish()

    async def drain():
        out = []
        while True:
            ev = await stream.next()
            if ev is None:
                return out
            out.append(ev.seq)

    assert asyncio.run(drain()) == [1, 2]


def test_stream_finish_ends_with_none():
    log = _make_convo().log
    stream = AgentEventStream(log)
    stream.start()
    _append(log, UserMessageObservation(content="a"))
    stream.finish()

    async def drain():
        out = []
        while True:
            ev = await stream.next()
            if ev is None:
                return out
            out.append(ev.seq)

    assert asyncio.run(drain()) == [1]


def test_stream_close_stops_live_delivery():
    log = _make_convo().log
    stream = AgentEventStream(log)
    stream.start()
    _append(log, UserMessageObservation(content="before"))
    stream.close()
    _append(log, UserMessageObservation(content="after close"))

    async def drain():
        out = []
        while True:
            ev = await stream.next()
            if ev is None:
                return out
            out.append(ev.seq)

    assert asyncio.run(drain()) == [1]  # buffered drained; post-close dropped


# ── Registry / session ───────────────────────────────────────────────────────

def test_registry_lifecycle():
    reg = AgentSessionRegistry()
    session = reg.create(_make_convo(), query="q")
    assert reg.get(session.session_id) is session
    assert session.query == "q"
    assert session.last_seq == 0
    reg.remove(session.session_id)
    assert reg.get(session.session_id) is None
    # removing signals end-of-stream to attached consumers
    stream = AgentEventStream(session.log)
    session.attach(stream)

    async def after_remove():
        # a removed session's log stops delivering: buffer drains to None
        ev = await stream.next()
        assert ev is None

    asyncio.run(after_remove())


def test_session_attach_detach_stops_stream():
    convo = _make_convo()
    session = AgentSession("s2", convo)
    _append(convo.log, UserMessageObservation(content="a"))
    stream = AgentEventStream(convo.log)
    session.attach(stream)
    session.detach(stream)
    _append(convo.log, UserMessageObservation(content="b"))

    async def drain():
        out = []
        while True:
            ev = await stream.next()
            if ev is None:
                return out
            out.append(ev.seq)

    assert asyncio.run(drain()) == [1]  # detached: live event 2 not delivered


def test_session_notify_finished_terminates_attached_streams():
    convo = _make_convo()
    session = AgentSession("s3", convo)
    _append(convo.log, UserMessageObservation(content="a"))
    stream = AgentEventStream(convo.log)
    session.attach(stream)
    session.notify_finished()

    async def drain():
        out = []
        while True:
            ev = await stream.next()
            if ev is None:
                return out
            out.append(ev.seq)

    assert asyncio.run(drain()) == [1]
    assert session.finished


def test_session_notify_error_records_message_and_finishes():
    convo = _make_convo()
    session = AgentSession("s4", convo)
    session.notify_error("backend down")
    assert session.finished
    assert session.error == "backend down"
    # reconnecting onto a failed session terminates immediately
    stream = AgentEventStream(convo.log)
    session.attach(stream)

    async def first():
        return await stream.next()

    assert asyncio.run(first()) is None


def test_step_records_deterministic_across_rebuilds():
    convo = _make_convo()
    log = convo.log
    _append(log, UserMessageObservation(content="q"))
    _append(log, ToolCallAction(name="read_file", arguments={"path": "x"}), tokens=8)
    _append(log, ToolResultObservation(content="contents", success=True), tokens=4)
    _append(log, FinishAction(content="ok"))
    session = AgentSession("s5", convo)
    assert session.step_records() == session.step_records()
    # pending action with no observation is still recorded
    _append(log, ToolCallAction(name="echo", arguments={"text": "z"}), tokens=3)
    records = session.step_records()
    assert len({r["step"] for r in records.values()}) == 2


# ── HTTP surface ─────────────────────────────────────────────────────────────

def test_api_create_run_stub(client):
    r = client.post("/v1/agent-runs",
                    json={"query": "hello", "backend": {"type": "stub"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "created"
    assert body["run_id"]
    assert body["last_seq"] == 0
    assert get_agent_session_registry().get(body["run_id"]) is not None


def test_api_create_run_with_workspace_tools(client):
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        r = client.post("/v1/agent-runs", json={
            "query": "read the files",
            "workspace_roots": [root],
            "backend": {"type": "stub"},
        })
        assert r.status_code == 200, r.text
        session = get_agent_session_registry().get(r.json()["run_id"])
        assert session is not None
        names = {t.name for t in session.conversation._agent.tools.tools()}
        assert {"read_file", "write_file", "run_command"} <= names


def test_api_unknown_run_404(client):
    r = client.get("/v1/agent-runs/nope")
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "run_not_found"
    r = client.get("/v1/agent-runs/nope/events")
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "run_not_found"
    r = client.post("/v1/agent-runs/nope/start")
    assert r.status_code == 404


def test_api_start_twice_conflict(client):
    run_id = client.post("/v1/agent-runs",
                         json={"query": "hi", "backend": {"type": "stub"}}).json()["run_id"]
    assert client.post(f"/v1/agent-runs/{run_id}/start").status_code == 200
    r = client.post(f"/v1/agent-runs/{run_id}/start")
    assert r.status_code == 409
    assert r.json()["error"]["type"] == "already_started"


def test_api_start_runs_to_finish(client):
    run_id = client.post("/v1/agent-runs",
                         json={"query": "run me", "backend": {"type": "stub"}}).json()["run_id"]
    client.post(f"/v1/agent-runs/{run_id}/start")
    for _ in range(200):
        status = client.get(f"/v1/agent-runs/{run_id}").json()["status"]
        if status == "finished":
            break
    assert status == "finished"
    body = client.get(f"/v1/agent-runs/{run_id}").json()
    assert body["last_seq"] >= 3
    assert body["error"] is None


def test_api_events_stream_end_to_end(client):
    run_id = client.post("/v1/agent-runs",
                         json={"query": "hello", "backend": {"type": "stub"}}).json()["run_id"]
    client.post(f"/v1/agent-runs/{run_id}/start")
    with client.stream("GET", f"/v1/agent-runs/{run_id}/events") as resp:
        assert resp.status_code == 200
        events = _parse_sse_lines(resp)
    seqs = [e["seq"] for e in events if e["kind"] != "end"]
    assert seqs == list(range(1, len(seqs) + 1))  # contiguous from 1
    assert events[0]["event_type"] == "user_message"
    assert events[-1]["kind"] == "end"
    assert events[-1]["seq"] == len(seqs)
    assert events[-2]["event_type"] == "finish"
    for e in events[:-1]:
        assert {"seq", "kind", "event_type", "ts", "tokens", "reasoning_tokens", "payload"} <= set(e)


def test_api_events_cursor_replay_no_duplicates(client):
    registry = get_agent_session_registry()
    session = registry.create(_make_convo(), query="replay")
    log = session.log
    _append(log, UserMessageObservation(content="q"))
    _append(log, ToolCallAction(name="echo", arguments={"text": "a"}), tokens=6)
    _append(log, ToolResultObservation(content="a", success=True), tokens=3)
    _append(log, FinishAction(content="done"))
    session.notify_finished()
    run_id = session.session_id

    with client.stream("GET", f"/v1/agent-runs/{run_id}/events") as resp:
        first = _parse_sse_lines(resp)
    first_seqs = [e["seq"] for e in first if e["kind"] != "end"]
    assert first_seqs == [1, 2, 3, 4]
    assert first[-1]["kind"] == "end"
    assert first[1]["step"] is not None  # action carries step metadata

    # Reconnect with cursor = last seq: nothing re-delivered, immediate end.
    # The end frame carries the final answer (Prompt 9) on top of error/seq.
    with client.stream("GET", f"/v1/agent-runs/{run_id}/events?cursor=4") as resp:
        second = _parse_sse_lines(resp)
    assert second == [{"kind": "end", "seq": 4, "error": None, "answer": "done"}]


# ── Smoke: mid-stream disconnect + reconnect, no loss / no duplication ───────

def test_smoke_disconnect_reconnect_no_loss_no_dup():
    """
    Fabricated session → live stream → disconnect mid-stream → more events
    → reconnect with cursor → seqs contiguous, nothing lost, nothing
    duplicated. Runs entirely on one event loop (the production classes are
    loop-bound), exercising replay, live delivery, and the race purge.
    """

    async def scenario():
        registry = AgentSessionRegistry()
        session = registry.create(_make_convo(), query="smoke")
        log = session.log
        _append(log, UserMessageObservation(content="hello"))  # seq 1

        stream = AgentEventStream(log, cursor=0)
        session.attach(stream)
        seen: List[int] = []

        async def consume_until(n: int):
            while len(seen) < n:
                ev = await stream.next()
                if ev is None:
                    break
                seen.append(ev.seq)

        await consume_until(1)  # seq 1 delivered
        _append(log, ToolCallAction(name="echo", arguments={"text": "a"}), tokens=5)  # seq 2
        await consume_until(2)

        # disconnect mid-stream; the client remembers its last seq (2)
        session.detach(stream)

        # events arrive while disconnected
        _append(log, ToolResultObservation(content="a", success=True), tokens=3)  # seq 3
        _append(log, MessageAction(content="almost there"))                     # seq 4
        _append(log, FinishAction(content="done"))                               # seq 5
        session.notify_finished()

        # reconnect with cursor = last received seq (2)
        stream2 = AgentEventStream(log, cursor=seen[-1])
        session.attach(stream2)
        while True:
            ev = await stream2.next()
            if ev is None:
                break
            seen.append(ev.seq)

        return seen

    seen = asyncio.run(scenario())
    assert seen == [1, 2, 3, 4, 5]  # contiguous — no loss, no duplication
