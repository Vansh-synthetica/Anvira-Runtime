"""
Tests for the Prompt 7 layer of orcha.agent_runtime: the developer-grade,
READ-ONLY diagnostics & log system.

- One run → one DiagnosticSession (meta + structured entries captured from
  the EXISTING orcha logging tree plus direct stage records).
- Never influences the run: identical ConversationResult with and without
  capture; replay folds identically; EventLog untouched.
- Full failure visibility: model exceptions, backend timeouts, tool
  exceptions, terminal timeouts, filesystem errors, MCP discovery
  failures, attachment failures — all captured without collapsing.
- Privacy: secrets (API keys, Authorization, cookies, tokens, passwords)
  are redacted from payloads and the plain-text report.
- HTTP: diagnostics endpoints (json/text/list/clear); clear removes ONLY
  stored diagnostics.
"""
import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import httpx
import pytest
from fastapi.testclient import TestClient

from orcha.agent_runtime import (
    AgentRuntimeConfig, Conversation, DiagnosticSession, DiagnosticsStore,
    EventLog, FinishAction, ToolCallAction, ToolResultObservation,
    UserMessageObservation, build_agent_runtime, diagnose_run,
    discover_tools, get_diagnostics_store, redact_payload, render_report,
)
from orcha.agent_runtime.backend import (
    ModelBackend, ModelResponse, TokenUsage, ToolCall,
)
from orcha.agent_runtime.events import EventKind
from orcha.api.agent_stream import get_agent_session_registry
from orcha.api.server import app


class FakeBackend(ModelBackend):
    """Scripted backend; answers "done" when the script is exhausted."""

    def __init__(
        self, responses: Optional[Sequence[ModelResponse]] = None,
        model: str = "fake-model",
    ) -> None:
        self._responses = list(responses or [])
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_streaming_tool_calls(self) -> bool:
        return False

    @property
    def reports_reasoning_tokens(self) -> bool:
        return False

    def complete(self, messages, tools=None, config=None) -> ModelResponse:
        if not self._responses:
            return ModelResponse(
                content="done",
                usage=TokenUsage(prompt_tokens=4, completion_tokens=7),
                finish_reason="stop",
            )
        return self._responses.pop(0)


def tool_call_response(name: str, arguments: Dict[str, Any]) -> ModelResponse:
    return ModelResponse(
        content="",
        tool_calls=[ToolCall(id=f"call-{name}", name=name, arguments=arguments)],
        usage=TokenUsage(prompt_tokens=5, completion_tokens=7),
        finish_reason="tool_calls",
    )


def _conversation(script=None, tmp_path=None) -> Conversation:
    config = AgentRuntimeConfig(workspace_roots=[str(tmp_path)] if tmp_path else [])
    conversation, _, _ = build_agent_runtime(config, backend=FakeBackend(script or []))
    return conversation


def _stages(session: DiagnosticSession) -> List[str]:
    return [e.stage + "." + e.event for e in session.entries]


def _find(session: DiagnosticSession, stage_event: str) -> List[Any]:
    return [e for e in session.entries if f"{e.stage}.{e.event}" == stage_event]


# ── Successful & failed runs generate diagnostics ────────────────────────────

async def test_successful_run_generates_diagnostics(tmp_path):
    store = DiagnosticsStore()
    diag = store.start(
        "run-ok",
        meta={"model": "fake-model", "backend": "stub",
              "workspace_roots": [str(tmp_path)], "runtime_version": "0.4.0"},
    )
    convo = _conversation(
        script=[tool_call_response("run_command", {"command": "echo hi"})],
        tmp_path=tmp_path,
    )
    convo.submit_user_message("run it")
    result = await diagnose_run(diag, convo)

    assert result.terminated_by == "finish"
    assert diag.outcome == "success"
    assert diag.duration_ms is not None
    d = diag.to_dict()
    assert d["meta"]["model"] == "fake-model"

    stages = _stages(diag)
    assert "session.start" in stages
    assert "session.finish" in stages
    assert "conversation.step" in stages
    assert "conversation.finish" in stages
    assert "agent.decision" in stages
    assert "tool.result" in stages
    assert not diag.errors()


async def test_failed_run_generates_diagnostics(tmp_path):
    class BoomBackend(FakeBackend):
        def complete(self, messages, tools=None, config=None) -> ModelResponse:
            raise RuntimeError("model exploded")

    from orcha.agent_runtime.agent import ToolCallingAgent
    from orcha.agent_runtime.conversation import Conversation
    from orcha.agent_runtime.workspace import StubWorkspace

    store = DiagnosticsStore()
    diag = store.start("run-fail", meta={"model": "boom", "backend": "stub"})
    convo = Conversation(ToolCallingAgent(BoomBackend()), StubWorkspace(), query="q")
    convo.submit_user_message("go")
    result = await diagnose_run(diag, convo)

    assert result.terminated_by == "error"  # loop survives: ErrorAction, no raise
    assert diag.outcome == "failed"
    assert diag.error_summary is not None
    errors = diag.errors()
    assert errors
    failed = [e for e in diag.entries if e.event == "model_call_failed"]
    assert failed
    assert failed[-1].fields["exception_type"] == "RuntimeError"
    assert failed[-1].traceback


# ── Exception capture never collapses ─────────────────────────────────────────

async def test_model_exception_captured_with_traceback(tmp_path):
    class BoomBackend(FakeBackend):
        def complete(self, messages, tools=None, config=None) -> ModelResponse:
            raise ValueError("bad payload from model")

    from orcha.agent_runtime.agent import ToolCallingAgent
    from orcha.agent_runtime.conversation import Conversation
    from orcha.agent_runtime.workspace import StubWorkspace

    store = DiagnosticsStore()
    diag = store.start("run-model-exc")
    convo = Conversation(ToolCallingAgent(BoomBackend()), StubWorkspace(), query="q")
    convo.submit_user_message("go")
    result = await diagnose_run(diag, convo)

    assert result.terminated_by == "error"
    exc_entries = [e for e in diag.entries if e.event == "model_call_failed"]
    assert exc_entries
    entry = exc_entries[-1]
    assert entry.fields["exception_type"] == "ValueError"
    assert "bad payload from model" in entry.fields["exception_message"]
    assert entry.traceback
    assert "ValueError" in entry.traceback
    assert "bad payload from model" in entry.traceback
    assert diag.outcome == "failed"


def test_backend_timeout_captured():
    """httpx timeout through the real OpenAI-compatible backend keeps the
    traceback and surfaces as a clean run failure."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connect timed out", request=request)

    from orcha.agent_runtime.agent import ToolCallingAgent
    from orcha.agent_runtime.backends.openai_compat import (
        OpenAICompatBackend, OpenAICompatBackendConfig,
    )
    from orcha.agent_runtime.conversation import Conversation
    from orcha.agent_runtime.workspace import StubWorkspace

    backend = OpenAICompatBackend(
        OpenAICompatBackendConfig(base_url="http://localhost:1/v1", model="m"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    store = DiagnosticsStore()
    diag = store.start("run-timeout")
    convo = Conversation(ToolCallingAgent(backend), StubWorkspace(), query="q")
    convo.submit_user_message("go")
    result = asyncio.run(diagnose_run(diag, convo))

    assert result.terminated_by == "error"  # httpx timeout surfaced, not swallowed
    entries = _find(diag, "backend.request_failed")
    assert entries
    entry = entries[-1]
    assert entry.fields["exception_type"] == "ConnectTimeout"
    assert entry.fields["exception_message"] == "connect timed out"
    assert entry.traceback and "ConnectTimeout" in entry.traceback
    assert diag.outcome == "failed"


def test_tool_exception_captured(tmp_path):
    from orcha.agent_runtime.tools import ToolRegistry, Tool
    from orcha.agent_runtime.workspace import ToolWorkspace
    from orcha.agent_runtime.agent import ToolCallingAgent
    from orcha.agent_runtime.conversation import Conversation
    from orcha.capabilities.base import READ, SAFE, spec

    def exploding_impl() -> Dict[str, Any]:
        raise ValueError("tool internals broke")

    tool = Tool(spec(
        "exploder", "Always breaks.",
        {"type": "object", "properties": {}},
        exploding_impl, permissions=[READ], safety_level=SAFE,
    ))
    registry = ToolRegistry([tool])
    store = DiagnosticsStore()
    diag = store.start("run-tool-exc")

    class ToolBackend(FakeBackend):
        def __init__(self):
            super().__init__([tool_call_response("exploder", {})])

    convo = Conversation(
        ToolCallingAgent(ToolBackend()), ToolWorkspace(registry), query="q",
    )
    convo.submit_user_message("explode")
    result = asyncio.run(diagnose_run(diag, convo))

    assert result.terminated_by == "finish"  # loop survived the tool failure
    tool_entries = [e for e in diag.entries if e.stage == "tool"]
    failures = [e for e in tool_entries if e.event == "result" and not e.fields["ok"]]
    assert failures
    entry = failures[-1]
    assert entry.fields["error_code"] == "tool_error"
    assert "tool internals broke" in entry.fields["error_message"]
    detail = json.dumps(entry.fields.get("error_detail") or {})
    assert "ValueError" in detail  # original exception type survives


async def test_terminal_exception_captured(tmp_path):
    sleep_cmd = "Start-Sleep -Seconds 5" if os.name == "nt" else "sleep 5"
    script = [
        tool_call_response(
            "run_command",
            {"command": sleep_cmd, "timeout_s": 1},
        ),
    ]
    store = DiagnosticsStore()
    diag = store.start("run-terminal-timeout")
    convo = _conversation(script=script, tmp_path=tmp_path)
    convo.submit_user_message("sleep")
    result = await diagnose_run(diag, convo)

    assert result.terminated_by == "finish"
    failures = [
        e for e in diag.entries
        if e.stage == "tool" and e.event == "result" and not e.fields["ok"]
    ]
    assert failures
    assert failures[-1].fields["error_code"] == "command_timeout"


async def test_filesystem_exception_captured(tmp_path):
    script = [
        tool_call_response("read_file", {"path": "does-not-exist.txt"}),
    ]
    store = DiagnosticsStore()
    diag = store.start("run-fs-error")
    convo = _conversation(script=script, tmp_path=tmp_path)
    convo.submit_user_message("read it")
    result = await diagnose_run(diag, convo)

    assert result.terminated_by == "finish"
    failures = [
        e for e in diag.entries
        if e.stage == "tool" and e.event == "result" and not e.fields["ok"]
    ]
    assert failures
    assert failures[-1].fields["error_code"] == "file_not_found"


def test_mcp_exception_captured():
    store = DiagnosticsStore()
    diag = store.start("run-mcp")
    config = AgentRuntimeConfig(mcp_servers=[
        {
            "name": "broken-server",
            "transport": "stdio",
            "command": "orcha-nonexistent-mcp-binary-xyz",
        }
    ])

    async def scenario():
        async with diag.scope():
            registry = discover_tools(config)
        assert not registry.names()

    asyncio.run(scenario())

    mcp_entries = [
        e for e in diag.entries
        if "mcp" in e.stage.lower() or "mcp" in e.event.lower()
    ]
    assert mcp_entries
    assert any(e.level == "warning" for e in mcp_entries)


def test_attachment_failure_captured():
    store = DiagnosticsStore()
    diag = store.start("run-attach")
    diag.record_attachment("C:/definitely/not/a/real/path-xyz")
    ok = [e for e in diag.entries if e.stage == "attachment" and e.event == "failed"]
    assert ok
    assert ok[-1].fields["reason"] == "path does not exist"
    assert diag.errors() == []  # attachment failure is a warning, not a crash


# ── The report: complete + redacted ──────────────────────────────────────────

def test_copy_returns_complete_log(tmp_path):
    async def scenario():
        store = DiagnosticsStore()
        diag = store.start(
            "run-copy",
            meta={"model": "m", "backend": "stub",
                  "workspace_roots": [str(tmp_path)],
                  "environment": {"python": "3.13", "platform": "win32"},
                  "query": "the original question"},
        )
        convo = _conversation(
            script=[tool_call_response("run_command", {"command": "echo marker-xyz"})],
            tmp_path=tmp_path,
        )
        convo.submit_user_message("the original question")
        await diagnose_run(diag, convo)
        return diag, list(convo.log.events)

    diag, events = asyncio.run(scenario())

    report = render_report(diag, events)
    assert "ORCHA AGENT DIAGNOSTIC LOG" in report
    assert "run-copy" in report
    assert "the original question" in report
    assert "EVENT LOG" in report
    assert "seq=1" in report and "tool_call" in report
    assert "STAGES" in report
    assert "conversation.step" in report
    assert "tool.result" in report
    assert "ERRORS" in report
    assert "END OF DIAGNOSTIC LOG" in report
    assert "marker-xyz" in report  # real execution output is preserved


def test_secrets_are_redacted():
    payload = {
        "api_key": "sk-super-secret-123456",
        "Authorization": "Bearer hunter2-secret",
        "cookie": "session=abc123",
        "password": "p@ssw0rd",
        "token": "ghp_0123456789012345678901",
        "env": {"ANVIRA_API_KEY": "sk-env-secret", "PATH": "/usr/bin"},
        "useful": {"command": "echo hi", "cwd": "/workspace"},
    }
    redacted = redact_payload(payload)
    text = json.dumps(redacted)
    for secret in (
        "sk-super-secret-123456", "hunter2-secret", "abc123",
        "p@ssw0rd", "ghp_0123456789012345678901", "sk-env-secret",
    ):
        assert secret not in text, secret
    assert "***" in text
    # useful context survives
    assert "echo hi" in text and "/workspace" in text and "/usr/bin" in text


def test_report_redacts_meta_secrets():
    store = DiagnosticsStore()
    diag = store.start(
        "run-secret",
        meta={"model": "m", "api_key": "sk-leak-123456789",
              "workspace_roots": ["C:/ws"], "query": "q"},
    )
    diag.record("backend", "request", api_key="sk-leak-123456789",
                base_url="http://x/api?api_key=sk-leak-123456789")
    diag.record("tool", "result", command="echo ok")
    report = render_report(diag, [])
    assert "sk-leak-123456789" not in report
    assert "***" in report
    assert "echo ok" in report  # useful context preserved


# ── Storage: bounded, clear is surgical ───────────────────────────────────────

def test_clear_removes_stored_diagnostics_only(tmp_path):
    async def run_one():
        store = DiagnosticsStore()
        diag = store.start("run-clear", meta={"model": "m"})
        convo = _conversation(tmp_path=tmp_path)
        convo.submit_user_message("go")
        await diagnose_run(diag, convo)
        return store, diag, convo

    store, diag, convo = asyncio.run(run_one())
    events_before = list(convo.log.events)
    assert store.list()

    cleared = store.clear()
    assert cleared == 1
    assert store.list() == []
    assert store.get("run-clear") is None

    # EventLog / conversation state untouched
    assert list(convo.log.events) == events_before
    assert convo.state.events == len(events_before)
    replay = EventLog()
    for ev in events_before:
        replay.append(ev.payload, tokens=ev.tokens, reasoning_tokens=ev.reasoning_tokens)
    assert replay.fold().events == convo.state.events


def test_store_is_bounded():
    store = DiagnosticsStore(max_sessions=3)
    for i in range(6):
        store.start(f"run-{i}", meta={"model": "m"})
    ids = [s["session_id"] for s in store.list()]
    assert ids == ["run-5", "run-4", "run-3"]  # newest first, ring-bounded


# ── Runtime behavior and replay are unchanged ─────────────────────────────────

def test_runtime_behavior_unchanged(tmp_path):
    async def run_with(capture: bool):
        store = DiagnosticsStore()
        script = [
            tool_call_response("run_command", {"command": "echo same-answer"}),
        ]
        convo = _conversation(script=script, tmp_path=tmp_path)
        convo.submit_user_message("same input")
        if capture:
            diag = store.start("run-a")
            result = await diagnose_run(diag, convo)
        else:
            result = await convo.run()
        return result

    with_capture = asyncio.run(run_with(True))
    without = asyncio.run(run_with(False))

    assert with_capture.terminated_by == without.terminated_by
    assert with_capture.steps == without.steps
    assert with_capture.tokens_used == without.tokens_used
    assert [ev.kind for ev in with_capture.events] == [ev.kind for ev in without.events]
    assert [ev.payload.kind for ev in with_capture.events] == [
        ev.payload.kind for ev in without.events
    ]
    assert with_capture.events[-1].payload.content == without.events[-1].payload.content


def test_event_replay_unaffected(tmp_path):
    async def scenario():
        store = DiagnosticsStore()
        diag = store.start("run-replay", meta={"model": "m"})
        convo = _conversation(
            script=[tool_call_response("run_command", {"command": "echo replay-me"})],
            tmp_path=tmp_path,
        )
        convo.submit_user_message("go")
        await diagnose_run(diag, convo)
        return list(convo.log.events), convo.state

    events, state = asyncio.run(scenario())
    rebuilt = EventLog()
    for ev in events:
        rebuilt.append(ev.payload, tokens=ev.tokens, reasoning_tokens=ev.reasoning_tokens)
    replayed = rebuilt.fold()
    assert replayed.seq == state.seq
    assert replayed.events == state.events
    assert replayed.answer == state.answer
    assert replayed.finish_reason == state.finish_reason


# ── HTTP surface ──────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _wait_finished(client, run_id, timeout_s=10.0) -> Dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        body = client.get(f"/v1/agent-runs/{run_id}").json()
        if body["status"] == "finished":
            return body
        time.sleep(0.05)
    raise AssertionError("run did not finish in time")


def test_http_diagnostics_lifecycle(client):
    resp = client.post("/v1/agent-runs", json={
        "query": "diagnose me", "backend": {"type": "stub"},
    })
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    resp = client.post(f"/v1/agent-runs/{run_id}/start")
    assert resp.status_code == 200
    _wait_finished(client, run_id)

    # structured diagnostics
    body = client.get(f"/v1/agent-runs/{run_id}/diagnostics").json()
    assert body["session_id"] == run_id
    assert body["outcome"] in ("success", "failed")
    assert body["meta"]["backend"] == "stub"
    assert body["events"], "diagnostics include the EventLog"
    stages = " ".join(f"{e['stage']}.{e['event']}" for e in body["entries"])
    assert "conversation.step" in stages
    assert "session.finish" in stages

    # plain-text report for copy/paste
    text = client.get(f"/v1/agent-runs/{run_id}/diagnostics?format=text")
    assert text.status_code == 200
    assert "ORCHA AGENT DIAGNOSTIC LOG" in text.text
    assert "END OF DIAGNOSTIC LOG" in text.text

    # list + clear
    listed = client.get("/v1/diagnostics").json()
    assert any(r["session_id"] == run_id for r in listed["runs"])
    cleared = client.delete("/v1/diagnostics").json()
    assert cleared["runs_cleared"] >= 1
    assert client.get("/v1/diagnostics").json()["runs"] == []

    # the run itself and its EventLog are untouched by clear
    status = client.get(f"/v1/agent-runs/{run_id}").json()
    assert status["status"] == "finished"
    events = client.get(f"/v1/agent-runs/{run_id}/events?cursor=0")
    assert events.status_code == 200


def test_http_diagnostics_unknown_run_404(client):
    resp = client.get("/v1/agent-runs/nope/diagnostics")
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "run_not_found"


# ── SSE lifecycle entries ─────────────────────────────────────────────────────

def test_sse_lifecycle_recorded():
    from orcha.agent_runtime.agent import StubAgent
    from orcha.agent_runtime.workspace import StubWorkspace
    from orcha.agent_runtime.stream import AgentEventStream, AgentSessionRegistry

    async def scenario():
        store = DiagnosticsStore()
        registry = AgentSessionRegistry()
        convo = Conversation(StubAgent(), StubWorkspace(), query="q")
        convo.submit_user_message("hello")
        session = registry.create(convo, query="q")
        diag = store.start(session.session_id, meta={"model": "stub"})
        session.diagnostics = diag

        stream = AgentEventStream(session.log, cursor=0)
        async with diag.scope():
            session.attach(stream)
            await stream.next()
            session.detach(stream)
            session.notify_finished()
        return diag

    diag = asyncio.run(scenario())
    stages = _stages(diag)
    assert "sse.attach" in stages
    assert "sse.disconnect" in stages
    assert "sse.end" in stages
