"""
tests.test_langgraph_checkpoint
===============================
Stage 8: the durable LangGraph checkpointer (OrchaSqliteCheckpointer).

Verifies that threads survive process restarts — a paused approval
thread resumed through a brand-new compiled graph + engine on the same
database — plus the server-side integration: restart-surviving approval
decisions (runner rebuilt from the persisted request) and get_run
surfacing langgraph runs whose runner has been dropped.
"""
import asyncio
import os
import time

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "false")

from orcha.api.server import app
from orcha.builders import build_agent_runner
from orcha.capabilities.base import CapabilityContext
from orcha.capabilities.registry import CapabilityRegistry
from orcha.experts.local_chat import LocalChatExpert
from orcha.integrations.langgraph import (
    OrchaSqliteCheckpointer,
    build_sqlite_checkpointer,
)
from orcha.nodes.agent import AgentConfig

pytestmark = pytest.mark.usefixtures("_no_env_engine")


def _filesystem_executor(root):
    registry = CapabilityRegistry().register_defaults()
    ctx = CapabilityContext(roots=[str(root)])
    return registry.build(["filesystem"], ctx=ctx)


@pytest.fixture
def _no_env_engine(monkeypatch):
    monkeypatch.delenv("ORCHA_AGENT_ENGINE", raising=False)


async def _tool_approval_completion(messages, system, tools):
    # Agent-loop call via the envelope protocol: tools=None but the
    # system prompt contains the envelope instruction.
    if tools is None and system and "You MUST respond with a single JSON object" in system:
        import json as _json
        if not any(m.get("role") == "tool" for m in messages):
            return {"role": "assistant", "content": _json.dumps({
                "action": "tool", "name": "write_file",
                "arguments": {"path": "note.txt", "content": "hello durable"},
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
                    "arguments": {"path": "note.txt", "content": "hello durable"},
                },
            }],
        }
    return {"role": "assistant", "content": "I executed the tool.", "finish_reason": "stop"}


# ── Checkpointer round-trip ───────────────────────────────────────────


def test_sqlite_thread_survives_engine_recreation(tmp_path):
    """A paused approval thread must resume through a brand-new compiled
    graph + engine sharing the same database file."""
    db = str(tmp_path / "threads.db")

    async def main():
        from orcha.builders import ApprovalPending

        cp = build_sqlite_checkpointer(db)
        runner = build_agent_runner(
            engine="langgraph",
            agent_config=AgentConfig(
                completion_fn=_tool_approval_completion,
                max_iterations=3,
                executor=_filesystem_executor(tmp_path),
            ),
            require_approval=True,
            checkpointer=cp,
        )
        run_id = "durable-1"
        try:
            await runner.run("write the file", run_id=run_id)
            paused = False
        except ApprovalPending:
            paused = True
        assert paused
        # Thread is on disk, not in memory.
        thread = cp.read_thread(run_id)
        assert thread is not None
        assert thread["paused"]  # paused at the approval gate

        # "Restart": a fresh saver + fresh compiled graph + fresh engine.
        cp2 = build_sqlite_checkpointer(db)
        runner2 = build_agent_runner(
            engine="langgraph",
            agent_config=AgentConfig(
                completion_fn=_tool_approval_completion,
                max_iterations=3,
                executor=_filesystem_executor(tmp_path),
            ),
            require_approval=True,
            checkpointer=cp2,
        )
        assert await runner2.engine.pending(run_id)
        result = await runner2.run(
            "write the file", run_id=run_id,
            resume_from=object(), resume_value=True,
        )
        assert result.packet.payload["answer"] == "I executed the tool."

    asyncio.run(main())


def test_sqlite_checkpointer_reads_and_lists(tmp_path):
    db = str(tmp_path / "threads.db")
    cp = build_sqlite_checkpointer(db)
    assert isinstance(cp, OrchaSqliteCheckpointer)

    async def main():
        runner = build_agent_runner(
            engine="langgraph",
            agent_config=AgentConfig(
                completion_fn=_tool_approval_completion,
                max_iterations=3,
                executor=_filesystem_executor(tmp_path),
            ),
            checkpointer=cp,
        )
        await runner.run("write the file", run_id="history-1")
        thread = cp.read_thread("history-1")
        assert thread is not None
        packet = thread["values"].get("packet")
        assert packet is not None
        assert packet.payload["answer"] == "I executed the tool."
        assert thread["paused"] is False

        snap = await runner.engine.snapshot("history-1")
        assert snap is not None
        assert snap.get("packet") is not None
        hist = await runner.engine.replay("history-1")
        assert len(hist) >= 4  # START + one checkpoint per node transition

    asyncio.run(main())


def test_sqlite_checkpointer_default_path_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHA_LANGGRAPH_DB", str(tmp_path / "env.db"))
    cp = build_sqlite_checkpointer()
    assert os.path.exists(tmp_path / "env.db")


# ── Server: restart-surviving approval ────────────────────────────────


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Point the server's durable state (checkpointer DB + pending records)
    at a temp dir so tests never touch ~/.orcha, and clear the in-memory
    state between tests."""
    import orcha.api.server as server_module

    monkeypatch.setenv("ORCHA_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("ORCHA_LANGGRAPH_DB", str(tmp_path / "threads.db"))
    server_module._state.langgraph_checkpointer = None
    yield
    server_module._state.pending_approvals.clear()
    server_module._state.run_runtimes.clear()
    server_module._state.run_errors.clear()
    server_module._state.langgraph_checkpointer = None


def _patch_fake_expert(monkeypatch, completion):
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


def test_server_approval_gate_survives_restart(client, monkeypatch, tmp_path):
    """A paused approval run decided AFTER a simulated restart (in-memory
    runner + record wiped, disk state kept) must rebuild the runner from
    the persisted request and resume the durable thread."""
    import orcha.api.server as server_module

    monkeypatch.setenv("ORCHA_AGENT_ENGINE", "langgraph")
    _patch_fake_expert(monkeypatch, _tool_approval_completion)

    r1 = client.post("/v1/run", json={
        "query": "Write a note for me",
        "graph": "default",
        "workspace_roots": [str(tmp_path)],
        "capabilities": ["filesystem"],
        "require_approval": True,
        "stream": False,
    })
    assert r1.status_code == 200, r1.text
    run_id = r1.json()["run_id"]
    assert r1.json()["status"] == "pending_approval"

    # The paused record is on disk.
    from orcha.api.server import _read_pending_records
    assert run_id in _read_pending_records()

    # ── Simulated restart: wipe the in-memory state, keep the disk. ──
    server_module._state.pending_approvals.clear()
    server_module._state.run_runtimes.clear()

    # get_run still surfaces the paused thread from the durable DB.
    g = client.get(f"/v1/run/{run_id}")
    assert g.status_code == 200, g.text
    assert g.json()["status"] == "pending_approval"

    # The decision rebuilds the runner from the persisted request.
    r2 = client.post(f"/v1/run/{run_id}/approval", json={"approved": True})
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["status"] == "approved"
    assert "I executed the tool" in body["answer"]
    assert (tmp_path / "note.txt").read_text() == "hello durable"

    # Record cleaned up.
    assert run_id not in _read_pending_records()
    assert run_id not in server_module._state.pending_approvals


def test_server_get_run_surfaces_completed_langgraph_run(client, monkeypatch, tmp_path):
    """get_run must return the final result of a completed langgraph run
    even after its runner has been dropped from memory."""
    import orcha.api.server as server_module

    monkeypatch.setenv("ORCHA_AGENT_ENGINE", "langgraph")
    _patch_fake_expert(monkeypatch, _tool_approval_completion)

    r1 = client.post("/v1/run", json={
        "query": "Write a note for me",
        "graph": "default",
        "workspace_roots": [str(tmp_path)],
        "capabilities": ["filesystem"],
        "stream": False,
    })
    assert r1.status_code == 200, r1.text
    run_id = r1.json()["run_id"]
    assert r1.json()["status"] == "completed"
    assert run_id not in server_module._state.run_runtimes

    g = client.get(f"/v1/run/{run_id}")
    assert g.status_code == 200, g.text
    body = g.json()
    assert body["status"] == "completed"
    assert "I executed the tool" in body["answer"]
    assert (tmp_path / "note.txt").read_text() == "hello durable"
