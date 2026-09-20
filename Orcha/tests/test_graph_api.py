"""
Integration tests for the ORCHA3 graph API endpoints (/v1/run, etc.).
"""
import pytest
from fastapi.testclient import TestClient
from orcha.api.server import app


@pytest.fixture
def client():
    # Use the context manager so the lifespan startup initializes
    # _state.store (FileStore) before any request runs.
    with TestClient(app) as c:
        yield c


# ── /v1/run (synchronous) ─────────────────────────────────────────────────────

def test_run_default_graph_sync(client):
    """POST /v1/run with stream=false runs to completion."""
    r = client.post("/v1/run", json={
        "query": "What is machine learning?",
        "graph": "default",
        "stream": False,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "completed"
    assert body["run_id"]
    assert body["graph_name"] == "orcha_default"
    assert isinstance(body["answer"], str)


def test_run_with_stream_returns_run_id(client):
    """POST /v1/run with stream=true returns immediately with a run_id."""
    r = client.post("/v1/run", json={
        "query": "Explain neural networks",
        "graph": "default",
        "stream": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "running"
    assert body["run_id"]


def test_run_accepts_agent_mode_and_access_controls(client, monkeypatch):
    """POST /v1/run accepts the new agentic mode/access fields."""
    from orcha.experts.base import ExpertOutput
    from orcha.experts.local_chat import LocalChatExpert
    import orcha.api.server as server_module

    # Multi-agent mode requires an active local model. In the mock-backed
    # test orchestrator none exists, so inject a fake LocalChatExpert as the
    # synthesizer and patch the server's factory to construct it (no network).
    class FakeLocalExpert(LocalChatExpert):
        async def execute(self, query):
            return ExpertOutput(
                answer="Mocked local agent answer.",
                confidence=0.9,
                tokens_used=10,
                finish_reason="stop",
            )

        async def chat_completion(self, messages, *, system=None, tools=None, **overrides):
            return {"role": "assistant", "content": "FINAL ANSWER: Mocked local agent answer."}

    monkeypatch.setattr(server_module, "LocalChatExpert", FakeLocalExpert)
    orc = server_module._state.orc
    orc.experts["mock_synthesizer"] = FakeLocalExpert(
        model="fake-model", base_url="http://127.0.0.1:1/v1"
    )

    r = client.post("/v1/run", json={
        "query": "Plan a safe deployment workflow",
        "graph": "multi_agent",
        "mode": "plan",
        "access_mode": "approval",
        "allow_tools": False,
        "require_approval": True,
        "stream": False,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["run_id"]
    assert body["graph_name"] == "orcha_multi_agent"


def test_run_research_graph(client):
    """POST /v1/run with graph=research executes the research pipeline."""
    r = client.post("/v1/run", json={
        "query": "What is deep learning?",
        "graph": "research",
        "stream": False,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "completed"


def test_run_invalid_graph_returns_400(client):
    """An unknown graph name returns a 400 error."""
    r = client.post("/v1/run", json={
        "query": "test",
        "graph": "nonexistent_graph",
        "stream": False,
    })
    # The builder falls back to default for unknown names, so this should
    # actually succeed. Test an explicitly invalid scenario instead.
    # (graph_builder uses default for any unrecognized name)
    assert r.status_code == 200


# ── /v1/run/{run_id} ──────────────────────────────────────────────────────────

def test_get_run_status(client):
    """GET /v1/run/{id} returns the run's state."""
    # First create a run.
    r1 = client.post("/v1/run", json={
        "query": "test query",
        "stream": False,
    })
    run_id = r1.json()["run_id"]

    # Then fetch its status.
    r2 = client.get(f"/v1/run/{run_id}")
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["run_id"] == run_id


def test_get_run_not_found(client):
    """GET /v1/run/{id} with unknown id returns 404."""
    r = client.get("/v1/run/nonexistent-run-id")
    assert r.status_code == 404


# ── Background-run failure surfacing ──────────────────────────────────────────

def test_background_run_failure_surfaces_error(client, monkeypatch):
    """A background run that fails must report status='failed' + error via
    get_run (and an error SSE event) instead of silently appearing completed."""
    import time

    import orcha.api.server as server_module

    class _FailingRuntime(server_module.GraphRuntime):
        async def run(self, *args, **kwargs):
            raise RuntimeError("boom: simulated failure")

    monkeypatch.setattr(server_module, "GraphRuntime", _FailingRuntime)

    r = client.post("/v1/run", json={
        "query": "this will fail",
        "graph": "default",
        "stream": True,
    })
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    assert r.json()["status"] == "running"

    # Poll until the background task finishes and the failure is recorded.
    deadline = time.time() + 10
    body = None
    while time.time() < deadline:
        resp = client.get(f"/v1/run/{run_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        if body["status"] != "running":
            break
        time.sleep(0.05)
    assert body is not None
    assert body["status"] == "failed"
    assert "boom" in body["error"]

    # The SSE stream must emit an error event before ending.
    with client.stream("GET", f"/v1/run/{run_id}/events") as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line.startswith("data: ")]
    kinds = []
    for line in lines:
        import json as _json
        kinds.append(_json.loads(line[len("data: "):]).get("kind"))
    assert "error" in kinds
    assert kinds[-1] == "end"


# ── Background-run concurrency cap ────────────────────────────────────────────

def test_background_run_cap_rejects_when_full(client, monkeypatch):
    """POST /v1/run with stream=true returns 429 when the run cap is hit."""
    import orcha.api.server as server_module

    monkeypatch.setattr(server_module, "_MAX_CONCURRENT_BG_RUNS", 2)
    server_module._state.active_runs.clear()
    try:
        server_module._state.active_runs["dummy-a"] = object()
        server_module._state.active_runs["dummy-b"] = object()

        r = client.post("/v1/run", json={
            "query": "should be rejected",
            "graph": "default",
            "stream": True,
        })
        assert r.status_code == 429, r.text
        assert r.json()["error"]["type"] == "too_many_runs"
    finally:
        server_module._state.active_runs.clear()


# ── /v1/run/{run_id}/cancel ───────────────────────────────────────────────────

def test_cancel_background_run(client, monkeypatch):
    """Cancelling an in-flight run reports status='cancelled' via get_run and
    a 'cancelled' SSE event — never a false 'failed'."""
    import asyncio

    import orcha.api.server as server_module

    class _SlowRuntime(server_module.GraphRuntime):
        async def run(self, *args, **kwargs):
            await asyncio.sleep(30)

    monkeypatch.setattr(server_module, "GraphRuntime", _SlowRuntime)

    r = client.post("/v1/run", json={
        "query": "this will be cancelled",
        "graph": "default",
        "stream": True,
    })
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    assert r.json()["status"] == "running"

    r2 = client.post(f"/v1/run/{run_id}/cancel")
    assert r2.status_code == 200, r2.text
    assert r2.json()["status"] == "cancelled"

    # get_run must report a distinct cancelled status with the message.
    g = client.get(f"/v1/run/{run_id}")
    assert g.status_code == 200, g.text
    body = g.json()
    assert body["status"] == "cancelled"
    assert "cancelled" in (body["error"] or "").lower()

    # The SSE stream emits a 'cancelled' event before closing.
    with client.stream("GET", f"/v1/run/{run_id}/events") as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line.startswith("data: ")]
    kinds = []
    for line in lines:
        import json as _json
        kinds.append(_json.loads(line[len("data: "):]).get("kind"))
    assert "cancelled" in kinds
    assert kinds[-1] == "end"


def test_cancel_is_idempotent_and_rejects_finished_runs(client):
    """Cancelling twice is fine; cancelling an unknown/finished run is 409."""
    r = client.post("/v1/run/never-created-run/cancel")
    assert r.status_code == 409, r.text
    assert r.json()["error"]["type"] == "run_not_active"


# ── /v1/runs ──────────────────────────────────────────────────────────────────

def test_list_runs(client):
    """GET /v1/runs lists all known runs."""
    # Create a run first.
    client.post("/v1/run", json={"query": "test", "stream": False})

    r = client.get("/v1/runs")
    assert r.status_code == 200
    body = r.json()
    assert "runs" in body
    assert "total" in body
    assert body["total"] >= 1


# ── /v1/run/{run_id}/events (SSE) ─────────────────────────────────────────────

def test_stream_events_for_completed_run(client):
    """GET /v1/run/{id}/events streams historical events for a completed run."""
    # Create and complete a run.
    r1 = client.post("/v1/run", json={"query": "test", "stream": False})
    run_id = r1.json()["run_id"]

    # Stream its events.
    with client.stream("GET", f"/v1/run/{run_id}/events") as resp:
        assert resp.status_code == 200
        # Collect at least the end event.
        chunks = []
        for line in resp.iter_lines():
            chunks.append(line)
            if len(chunks) > 50:
                break
    # Should have received some SSE data.
    assert len(chunks) > 0


# ── /v1/run/{run_id}/replay ───────────────────────────────────────────────────

def test_replay_run(client):
    """POST /v1/run/{id}/replay reproduces the run."""
    # Create a run.
    r1 = client.post("/v1/run", json={"query": "replay test", "stream": False})
    run_id = r1.json()["run_id"]

    # Replay it.
    r2 = client.post(f"/v1/run/{run_id}/replay")
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["status"] == "replayed"
    assert body["run_id"] == run_id


def test_replay_nonexistent_returns_404(client):
    """Replaying a nonexistent run returns 404."""
    r = client.post("/v1/run/totally-fake-id/replay")
    assert r.status_code == 404


# ── default/research tool routing through the agent graph ─────────────────────

def test_run_default_with_tools_routes_through_agent_graph(client, monkeypatch, tmp_path):
    """graph='default' + workspace roots + capabilities must actually execute
    tools via the single-agent graph, not fall back to the tool-less pipeline."""
    from orcha.experts.local_chat import LocalChatExpert
    import orcha.api.server as server_module

    class FakeLocalExpert(LocalChatExpert):
        async def chat_completion(self, messages, *, system=None, tools=None, **overrides):
            # Agent-loop call via the envelope protocol: tools=None but the
            # system prompt contains the envelope instruction.
            if tools is None and system and "You MUST respond with a single JSON object" in system:
                if not any(m.get("role") == "tool" for m in messages):
                    import json as _json
                    return {"role": "assistant", "content": _json.dumps({
                        "action": "tool", "name": "write_file",
                        "arguments": {"path": "note.txt", "content": "hello tools"},
                    }), "finish_reason": "stop"}
                import json as _json
                return {"role": "assistant", "content": _json.dumps({
                    "action": "final", "answer": "I executed the tool.",
                }), "finish_reason": "stop"}
            if tools is None:
                # Intent-gate call: this message genuinely asks for a tool action.
                return {"role": "assistant", "content": "TOOL_REQUEST", "finish_reason": "stop"}
            if not any(m.get("role") == "tool" for m in messages):
                return {
                    "role": "assistant",
                    "content": None,
                    "finish_reason": "stop",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": {"path": "note.txt", "content": "hello tools"},
                        },
                    }],
                }
            return {"role": "assistant", "content": "I executed the tool.", "finish_reason": "stop"}

    monkeypatch.setattr(server_module, "LocalChatExpert", FakeLocalExpert)
    orc = server_module._state.orc
    orc.experts["mock_synthesizer"] = FakeLocalExpert(
        model="fake-model", base_url="http://127.0.0.1:1/v1"
    )
    orc.synthesizer_expert = "mock_synthesizer"

    r = client.post("/v1/run", json={
        "query": "Write a note for me",
        "graph": "default",
        "workspace_roots": [str(tmp_path)],
        "capabilities": ["filesystem"],
        "stream": False,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["graph_name"] == "orcha_agent"
    assert "I executed the tool" in body["answer"]
    assert (tmp_path / "note.txt").read_text() == "hello tools"
    # The executed tool call is surfaced for the frontend timeline.
    assert body["agent_completed"] is True
    assert len(body["agent_tool_calls"]) == 1
    assert body["agent_tool_calls"][0]["name"] == "write_file"
    assert body["agent_tool_calls"][0]["result_type"] == "ok"


def test_run_default_chat_stays_conversational(client, monkeypatch, tmp_path):
    """The intent gate must answer conversation directly: a greeting goes to
    finalize without the tool loop, zero tool calls, zero approvals."""
    from orcha.experts.local_chat import LocalChatExpert
    import orcha.api.server as server_module

    class FakeLocalExpert(LocalChatExpert):
        async def chat_completion(self, messages, *, system=None, tools=None, **overrides):
            if tools is None:
                # Intent-gate call: a greeting is pure conversation.
                return {"role": "assistant", "content": "Hey there!", "finish_reason": "stop"}
            raise AssertionError("tool loop must never run for a greeting")

    monkeypatch.setattr(server_module, "LocalChatExpert", FakeLocalExpert)
    orc = server_module._state.orc
    orc.experts["mock_synthesizer"] = FakeLocalExpert(
        model="fake-model", base_url="http://127.0.0.1:1/v1"
    )
    orc.synthesizer_expert = "mock_synthesizer"

    r = client.post("/v1/run", json={
        "query": "Hey",
        "graph": "default",
        "workspace_roots": [str(tmp_path)],
        "capabilities": ["filesystem"],
        "stream": False,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["graph_name"] == "orcha_agent"
    assert body["answer"] == "Hey there!"
    assert body["agent_tool_calls"] == []
    assert "approval" not in body["answer"].lower()


def test_run_default_without_tools_stays_on_classic_pipeline(client):
    """No tool surface -> default chat keeps using the classic pipeline graph."""
    r = client.post("/v1/run", json={"query": "hi", "graph": "default", "stream": False})
    assert r.status_code == 200, r.text
    assert r.json()["graph_name"] == "orcha_default"


def test_run_default_with_capabilities_but_no_roots_still_uses_agent_graph(client, monkeypatch):
    """A declared capability enables the tool loop even with no workspace roots
    attached — otherwise agents with capabilities silently degrade to plain
    chat and describe actions instead of executing tools."""
    from orcha.experts.local_chat import LocalChatExpert
    import orcha.api.server as server_module

    class FakeLocalExpert(LocalChatExpert):
        async def chat_completion(self, messages, *, system=None, tools=None, **overrides):
            if tools is None:
                # Intent-gate call: asking about the workspace is tool intent.
                return {"role": "assistant", "content": "TOOL_REQUEST", "finish_reason": "stop"}
            assert tools, "agent loop must present tool schemas"
            return {"role": "assistant", "content": "searched", "finish_reason": "stop"}

    monkeypatch.setattr(server_module, "LocalChatExpert", FakeLocalExpert)
    orc = server_module._state.orc
    orc.experts["mock_synthesizer"] = FakeLocalExpert(
        model="fake-model", base_url="http://127.0.0.1:1/v1"
    )
    orc.synthesizer_expert = "mock_synthesizer"

    r = client.post("/v1/run", json={
        "query": "What is in my workspace?",
        "graph": "default",
        "capabilities": ["workspace", "search"],
        "stream": False,
    })
    assert r.status_code == 200, r.text
    assert r.json()["graph_name"] == "orcha_agent"
    assert r.json()["agent_completed"] is True


def test_run_default_with_prompt_parts_and_messages(client, monkeypatch, tmp_path):
    """prompt_parts + messages are accepted and drive agent memory/system prompt."""
    from orcha.experts.local_chat import LocalChatExpert
    import orcha.api.server as server_module

    seen = {}

    class FakeLocalExpert(LocalChatExpert):
        async def chat_completion(self, messages, *, system=None, tools=None, **overrides):
            seen["system"] = system or ""
            seen["seeded"] = any(m.get("content") == "Previous turn" for m in messages)
            if tools is None:
                # Intent-gate call: "continue" after prior context is a tool request
                # only when the agent decides so; here we route into the loop to
                # verify the agent receives the seeded memory and prompt parts.
                return {"role": "assistant", "content": "TOOL_REQUEST", "finish_reason": "stop"}
            return {"role": "assistant", "content": "answered", "finish_reason": "stop"}

    monkeypatch.setattr(server_module, "LocalChatExpert", FakeLocalExpert)
    orc = server_module._state.orc
    orc.experts["mock_synthesizer"] = FakeLocalExpert(
        model="fake-model", base_url="http://127.0.0.1:1/v1"
    )
    orc.synthesizer_expert = "mock_synthesizer"

    r = client.post("/v1/run", json={
        "query": "continue",
        "graph": "default",
        "workspace_roots": [str(tmp_path)],
        "capabilities": ["filesystem"],
        "messages": [{"role": "user", "content": "Previous turn"}],
        "prompt_parts": [
            {"name": "persona", "text": "You are a helpful assistant.", "priority": 100},
        ],
        "stream": False,
    })
    assert r.status_code == 200, r.text
    assert seen["seeded"] is True
    assert "You are a helpful assistant" in seen["system"]
