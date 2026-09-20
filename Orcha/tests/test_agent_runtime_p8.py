"""
Tests for the Prompt 8 fix: backend routing can never silently fall through
to the stub test double for a live request.

Regression contract
-------------------
(a) A live-request-shaped call with NO explicit ``backend`` cannot resolve
    to ``_StubBackend`` — it either uses the real configured default (the
    primary local model in ``orcha.settings.settings``, the same singleton
    the ``/v1/run`` orchestrator is built from) or fails with a typed
    OrchaError (``no_model_configured``) carrying type/message/status/
    trace_id.
(b) A real-backend failure (server unreachable) surfaces as a typed,
    visible failure — never a silent fallback to "[stub] canned reply".
(c) Diagnostics meta ``backend``/``model`` are ``"stub"/"stub"`` ONLY for
    requests that explicitly requested the test double.
(d) Both entry paths (Chat ``/v1/run`` orchestrator and Agent Activity
    ``/v1/agent-runs``) derive their real backend from the SAME settings
    singleton, so they cannot drift apart.
"""
import time

import pytest
from fastapi.testclient import TestClient

from orcha.api.agent_stream import _default_backend
from orcha.api.server import app
from orcha.settings import settings


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _wait_finished(client, run_id, timeout_s=15.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        body = client.get(f"/v1/agent-runs/{run_id}").json()
        if body["status"] == "finished":
            return body
        time.sleep(0.05)
    raise AssertionError("run did not finish in time")


@pytest.fixture
def configured_model():
    """Point settings at a fake 'activated' model, restore afterwards."""
    prev_model = settings.local_server_model
    prev_base = settings.local_server_base_url
    prev_extras = list(settings.local_server_extra_models)
    settings.local_server_model = "Qwen2.5-Coder"
    settings.local_server_base_url = "http://localhost:9999/v1"
    settings.local_server_extra_models = []
    try:
        yield
    finally:
        settings.local_server_model = prev_model
        settings.local_server_base_url = prev_base
        settings.local_server_extra_models = prev_extras


@pytest.fixture
def no_configured_model():
    """Force the 'nothing configured' state, restore afterwards."""
    prev_model = settings.local_server_model
    prev_base = settings.local_server_base_url
    prev_extras = list(settings.local_server_extra_models)
    settings.local_server_model = ""
    settings.local_server_base_url = "http://localhost:8080/v1"
    settings.local_server_extra_models = []
    try:
        yield
    finally:
        settings.local_server_model = prev_model
        settings.local_server_base_url = prev_base
        settings.local_server_extra_models = prev_extras


def _sse_events(client, run_id):
    frames = []
    with client.stream("GET", f"/v1/agent-runs/{run_id}/events") as resp:
        for line in resp.iter_lines():
            if line.startswith("data:"):
                frames.append(line[len("data:"):].strip())
    return [__import__("json").loads(f) for f in frames if f]


# ── (a) No explicit backend → real default or typed error ────────────────────

def test_no_backend_without_configured_model_is_typed_error(client, no_configured_model):
    r = client.post("/v1/agent-runs", json={"query": "make a file"})
    assert r.status_code == 400
    envelope = r.json()["error"]
    assert envelope["type"] == "no_model_configured"
    assert envelope["status"] == 400
    assert isinstance(envelope["message"], str) and envelope["message"]
    assert envelope["trace_id"], "envelope must carry a real trace id"


def test_no_backend_uses_configured_default_model(client, configured_model):
    r = client.post("/v1/agent-runs", json={"query": "make a file"})
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    # Diagnostics meta records the RESOLVED backend — never "stub" here.
    body = client.get(f"/v1/agent-runs/{run_id}/diagnostics").json()
    assert body["meta"]["backend"] == "openai_compat"
    assert body["meta"]["model"] == "Qwen2.5-Coder"


def test_explicit_stub_is_the_only_way_to_get_stub(client, no_configured_model):
    """(c) The test double is reachable ONLY by explicit request."""
    r = client.post("/v1/agent-runs", json={
        "query": "hi", "backend": {"type": "stub"},
    })
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    client.post(f"/v1/agent-runs/{run_id}/start")
    _wait_finished(client, run_id)
    body = client.get(f"/v1/agent-runs/{run_id}/diagnostics").json()
    assert body["meta"]["backend"] == "stub"
    assert body["meta"]["model"] == "stub"
    frames = _sse_events(client, run_id)
    assert any("[stub] canned reply" in str(f.get("payload", {})) for f in frames)


# ── (b) Real-backend failure: typed, visible, never stub ─────────────────────

def test_real_backend_failure_is_typed_and_never_stub(client, no_configured_model):
    """Unreachable OpenAI-compatible server → typed failure, no canned reply."""
    r = client.post("/v1/agent-runs", json={
        "query": "do something",
        "backend": {"type": "openai_compat",
                    "base_url": "http://localhost:1/v1", "model": "m"},
    })
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    client.post(f"/v1/agent-runs/{run_id}/start")
    body = _wait_finished(client, run_id)

    # The failure is visible, not a clean end: the run reports an error.
    assert body["error"], "run failure must be surfaced on the run status"
    assert "stub" not in body["error"]

    # EventLog contains the typed error action — and NEVER a canned reply.
    frames = _sse_events(client, run_id)
    assert not any("[stub] canned reply" in str(f.get("payload", {})) for f in frames)
    assert any(f.get("event_type") == "error_action" for f in frames)

    # Diagnostics carry the original exception type + traceback.
    diag = client.get(f"/v1/agent-runs/{run_id}/diagnostics").json()
    assert diag["outcome"] == "failed"
    assert diag["meta"]["backend"] == "openai_compat"
    entries = diag["entries"]
    failed = [e for e in entries if e["event"] == "request_failed"]
    assert failed
    assert failed[-1]["fields"]["exception_type"] in ("ConnectTimeout", "ConnectError")
    assert failed[-1]["traceback"]


def test_terminated_runs_end_cleanly_with_synthesized_answer(client, no_configured_model):
    """A loop that hits a cutoff ends with a real natural-language answer,
    not an error and not an empty end frame — the Prompt 9 final-answer
    contract. Only genuine failures (terminated_by == 'error') or runs
    without any answer are surfaced as errors."""
    import asyncio

    from orcha.agent_runtime.conversation import ConversationResult
    from orcha.agent_runtime.events import EventLog
    from orcha.api.agent_stream import _drive_session, get_agent_session_registry

    class LoopConvo:
        def __init__(self, terminated_by, answer):
            self.log = EventLog()
            self._terminated_by = terminated_by
            self._answer = answer

        @property
        def answer(self):
            return self._answer

        async def run(self):
            return ConversationResult(query="q", terminated_by=self._terminated_by,
                                      answer=self._answer,
                                      steps=0, tokens_used=0, events=[])

    async def scenario(terminated_by, answer):
        session = get_agent_session_registry().create(
            LoopConvo(terminated_by, answer), query="q",
        )
        await _drive_session(session)
        return session

    # Cutoff WITH a synthesized answer → clean finish, no error.
    session = asyncio.run(scenario("max_steps", "I hit the run limit but here is a graceful note."))
    assert session.finished
    assert session.error is None, "a cutoff with an answer must not error"

    # Genuine error termination → typed error.
    session = asyncio.run(scenario("error", "something broke"))
    assert session.finished
    assert session.error and "error" in session.error

    # No answer at all → typed error, never a clean end.
    session = asyncio.run(scenario("max_steps", None))
    assert session.finished
    assert session.error and "final answer" in session.error


# ── (d) One resolution source for both entry paths ───────────────────────────

def test_default_backend_matches_orchestrator_config(configured_model):
    backend = _default_backend()
    assert isinstance(backend.model_name, str)
    assert backend.model_name == "Qwen2.5-Coder"
    assert backend.config.base_url == "http://localhost:9999/v1"
    # Parity: build_orchestrator() consumes the same singleton.
    assert settings.local_server_model == backend.model_name
    assert settings.local_server_base_url == backend.config.base_url


def test_default_backend_falls_back_to_first_extra_model(configured_model):
    settings.local_server_model = ""
    settings.local_server_extra_models = [
        {"model": "extra-1", "base_url": "http://localhost:8001/v1"},
        {"model": "extra-2", "base_url": "http://localhost:8002/v1"},
    ]
    backend = _default_backend()
    assert backend.model_name == "extra-1"
    assert backend.config.base_url == "http://localhost:8001/v1"


def test_health_and_run_paths_share_the_same_model_source(configured_model):
    """/v1/run's orchestrator and /v1/agent-runs' default both come from
    settings.local_server_model — the 'currently loaded model'. The client
    is built AFTER configuring the model so the orchestrator sees it."""
    with TestClient(app) as client:
        health = client.get("/v1/health").json()
    assert health["source"] == "local", "orchestrator must build from the local model"
    assert any("Qwen2_5" in e for e in health["experts"]), (
        "the orchestrator serving /v1/run should know the configured model"
    )
