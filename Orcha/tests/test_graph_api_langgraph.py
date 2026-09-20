"""
Integration tests for the LangGraph agent engine over the graph API.

With ``ORCHA_AGENT_ENGINE=langgraph``, agent runs (default/research with a
tool surface) execute on a genuine LangGraph StateGraph instead of the
native ``GraphRuntime``. These tests verify the HTTP surface end-to-end:
tool execution, the interrupt-based final-answer approval gate
(``pending_approval`` + ``POST /v1/run/{id}/approval``), and that the
native engine is untouched.
"""
import time

import pytest
from fastapi.testclient import TestClient

from orcha.api.server import app
from orcha.experts.local_chat import LocalChatExpert


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _clean_pending_state():
    import orcha.api.server as server_module

    yield
    server_module._state.pending_approvals.clear()
    server_module._state.run_runtimes.clear()
    server_module._state.run_errors.clear()


def _patch_fake_expert(monkeypatch, completion):
    """Install a fake LocalChatExpert with the given chat_completion as the
    agent's completion_fn (and the intent gate's router)."""
    import orcha.api.server as server_module

    class FakeLocalExpert(LocalChatExpert):
        async def chat_completion(self, messages, *, system=None, tools=None, **overrides):
            return await completion(messages, system, tools)

    monkeypatch.setattr(server_module, "LocalChatExpert", FakeLocalExpert)
    orc = server_module._state.orc
    orc.experts["mock_synth"] = FakeLocalExpert(
        model="fake-model", base_url="http://127.0.0.1:1/v1"
    )
    orc.synthesizer_expert = "mock_synth"


async def _tool_route_completion(messages, system, tools):
    # Agent-loop call via the envelope protocol: tools=None but the
    # system prompt contains the envelope instruction.
    if tools is None and system and "You MUST respond with a single JSON object" in system:
        import json as _json
        if not any(m.get("role") == "tool" for m in messages):
            return {"role": "assistant", "content": _json.dumps({
                "action": "tool", "name": "write_file",
                "arguments": {"path": "note.txt", "content": "hello tools"},
            }), "finish_reason": "stop"}
        return {"role": "assistant", "content": _json.dumps({
            "action": "final", "answer": "I executed the tool.",
        }), "finish_reason": "stop"}
    if tools is None:
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


# ── LangGraph engine end-to-end ───────────────────────────────────────────────

def test_langgraph_engine_routes_through_agent_graph(client, monkeypatch, tmp_path):
    """graph='default' + tool surface under ORCHA_AGENT_ENGINE=langgraph runs
    the LangGraph StateGraph and executes real tools."""
    monkeypatch.setenv("ORCHA_AGENT_ENGINE", "langgraph")
    _patch_fake_expert(monkeypatch, _tool_route_completion)

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
    assert body["status"] == "completed"
    assert "I executed the tool" in body["answer"]
    assert (tmp_path / "note.txt").read_text() == "hello tools"
    assert body["agent_completed"] is True
    assert len(body["agent_tool_calls"]) == 1
    assert body["agent_tool_calls"][0]["name"] == "write_file"
    assert body["agent_tool_calls"][0]["result_type"] == "ok"
    assert body["approval"] is None


def test_langgraph_engine_chat_stays_conversational(client, monkeypatch, tmp_path):
    """The intent gate answers chat directly under the LangGraph engine."""
    monkeypatch.setenv("ORCHA_AGENT_ENGINE", "langgraph")

    async def completion(messages, system, tools):
        assert tools is None  # gate call only
        return {"role": "assistant", "content": "Hey there!", "finish_reason": "stop"}

    _patch_fake_expert(monkeypatch, completion)

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


def test_default_engine_runs_without_env(client, monkeypatch, tmp_path):
    """Without ORCHA_AGENT_ENGINE the default (langgraph) agent graph
    runs — the engine flip made langgraph the no-env default."""
    _patch_fake_expert(monkeypatch, _tool_route_completion)

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
    assert body["status"] == "completed"
    assert (tmp_path / "note.txt").read_text() == "hello tools"


# ── Final-answer approval gate ────────────────────────────────────────────────

def test_langgraph_require_approval_pauses(client, monkeypatch, tmp_path):
    """require_approval + LangGraph engine: the run pauses with a
    pending_approval response carrying the proposed answer."""
    monkeypatch.setenv("ORCHA_AGENT_ENGINE", "langgraph")
    _patch_fake_expert(monkeypatch, _tool_route_completion)

    r = client.post("/v1/run", json={
        "query": "Write a note for me",
        "graph": "default",
        "workspace_roots": [str(tmp_path)],
        "capabilities": ["filesystem"],
        "require_approval": True,
        "stream": False,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "pending_approval"
    assert body["graph_name"] == "orcha_agent"
    approval = body["approval"]
    assert approval is not None
    assert approval["kind"] == "agent_approval"
    assert approval["output"] == "I executed the tool."
    assert [c["name"] for c in approval["tool_calls"]] == ["write_file"]
    assert approval["iterations"] == 2


def test_langgraph_approval_approve_resumes(client, monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHA_AGENT_ENGINE", "langgraph")
    _patch_fake_expert(monkeypatch, _tool_route_completion)

    r1 = client.post("/v1/run", json={
        "query": "Write a note for me",
        "graph": "default",
        "workspace_roots": [str(tmp_path)],
        "capabilities": ["filesystem"],
        "require_approval": True,
        "stream": False,
    })
    run_id = r1.json()["run_id"]
    assert r1.json()["status"] == "pending_approval"

    r2 = client.post(f"/v1/run/{run_id}/approval", json={"approved": True})
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["status"] == "approved"
    assert "I executed the tool" in body["answer"]
    assert (tmp_path / "note.txt").read_text() == "hello tools"

    # The decision is single-use.
    r3 = client.post(f"/v1/run/{run_id}/approval", json={"approved": True})
    assert r3.status_code == 404


def test_langgraph_approval_reject_surfaces_refusal(client, monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHA_AGENT_ENGINE", "langgraph")
    _patch_fake_expert(monkeypatch, _tool_route_completion)

    r1 = client.post("/v1/run", json={
        "query": "Write a note for me",
        "graph": "default",
        "workspace_roots": [str(tmp_path)],
        "capabilities": ["filesystem"],
        "require_approval": True,
        "stream": False,
    })
    run_id = r1.json()["run_id"]

    r2 = client.post(
        f"/v1/run/{run_id}/approval",
        json={"approved": False, "note": "not needed"},
    )
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["status"] == "rejected"
    assert body["answer"] == "The proposed action was not approved. Reason: not needed"


def test_langgraph_approval_unknown_run_returns_404(client, monkeypatch):
    monkeypatch.setenv("ORCHA_AGENT_ENGINE", "langgraph")
    r = client.post("/v1/run/does-not-exist/approval", json={"approved": True})
    assert r.status_code == 404


def test_langgraph_require_approval_without_gate_engine_ignores_pause(client, monkeypatch, tmp_path):
    """require_approval on the native engine keeps the broker-based flow:
    the run completes (no pause) because the fake model never requests
    per-tool approval."""
    monkeypatch.setenv("ORCHA_AGENT_ENGINE", "native")
    _patch_fake_expert(monkeypatch, _tool_route_completion)

    r = client.post("/v1/run", json={
        "query": "Write a note for me",
        "graph": "default",
        "workspace_roots": [str(tmp_path)],
        "capabilities": ["filesystem"],
        "require_approval": True,
        "stream": False,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "completed"
    assert body["approval"] is None


# ── Background runs ───────────────────────────────────────────────────────────

def test_background_langgraph_run_with_approval_keeps_runner(client, monkeypatch, tmp_path):
    """A paused background run must keep its runner alive so the decision
    endpoint can resume it."""
    monkeypatch.setenv("ORCHA_AGENT_ENGINE", "langgraph")
    _patch_fake_expert(monkeypatch, _tool_route_completion)

    r1 = client.post("/v1/run", json={
        "query": "Write a note for me",
        "graph": "default",
        "workspace_roots": [str(tmp_path)],
        "capabilities": ["filesystem"],
        "require_approval": True,
        "stream": True,
    })
    assert r1.status_code == 200, r1.text
    run_id = r1.json()["run_id"]
    assert r1.json()["status"] == "running"

    import orcha.api.server as server_module

    deadline = time.time() + 10
    while time.time() < deadline and run_id not in server_module._state.pending_approvals:
        time.sleep(0.05)
    assert run_id in server_module._state.pending_approvals
    assert run_id in server_module._state.run_runtimes

    r2 = client.post(f"/v1/run/{run_id}/approval", json={"approved": True})
    assert r2.status_code == 200, r2.text
    assert r2.json()["status"] == "approved"
    assert run_id not in server_module._state.pending_approvals
    assert run_id not in server_module._state.run_runtimes


# ── Multi-agent graph engine switch ───────────────────────────────────────────

def test_langgraph_engine_multi_agent_routes_through_langgraph(client, monkeypatch, tmp_path):
    """graph='multi_agent' under ORCHA_AGENT_ENGINE=langgraph runs the
    LangGraph StateGraph (Send-based scatter/gather) over HTTP."""
    monkeypatch.setenv("ORCHA_AGENT_ENGINE", "langgraph")

    async def completion(messages, system, tools):
        return {"role": "assistant", "content": "subtask done", "finish_reason": "stop"}

    _patch_fake_expert(monkeypatch, completion)

    r = client.post("/v1/run", json={
        "query": "Analyze the project",
        "graph": "multi_agent",
        "workspace_roots": [str(tmp_path)],
        "capabilities": ["filesystem"],
        "stream": False,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["graph_name"] == "orcha_multi_agent"
    assert body["status"] == "completed"
    assert body["answer"]


def test_default_multi_agent_runs_without_env(client, monkeypatch, tmp_path):
    """Without ORCHA_AGENT_ENGINE the default (langgraph) multi-agent
    graph runs — the engine flip made langgraph the no-env default."""
    async def completion(messages, system, tools):
        return {"role": "assistant", "content": "subtask done", "finish_reason": "stop"}

    _patch_fake_expert(monkeypatch, completion)

    r = client.post("/v1/run", json={
        "query": "Analyze the project",
        "graph": "multi_agent",
        "workspace_roots": [str(tmp_path)],
        "capabilities": ["filesystem"],
        "stream": False,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["graph_name"] == "orcha_multi_agent"
    assert body["status"] == "completed"
    assert body["answer"]


def test_langgraph_engine_multi_agent_background_run(client, monkeypatch, tmp_path):
    """Background (streamed) multi-agent run completes on the LangGraph
    engine and leaves no stale runner state."""
    monkeypatch.setenv("ORCHA_AGENT_ENGINE", "langgraph")

    async def completion(messages, system, tools):
        return {"role": "assistant", "content": "subtask done", "finish_reason": "stop"}

    _patch_fake_expert(monkeypatch, completion)

    r = client.post("/v1/run", json={
        "query": "Analyze the project",
        "graph": "multi_agent",
        "workspace_roots": [str(tmp_path)],
        "capabilities": ["filesystem"],
        "stream": True,
    })
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    assert r.json()["status"] == "running"

    import orcha.api.server as server_module

    deadline = time.time() + 10
    while time.time() < deadline and run_id in server_module._state.active_runs:
        time.sleep(0.05)
    assert run_id not in server_module._state.active_runs
    assert run_id not in server_module._state.run_runtimes
    assert run_id not in server_module._state.pending_approvals
