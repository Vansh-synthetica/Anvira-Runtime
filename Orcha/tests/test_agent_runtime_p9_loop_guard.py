"""
Regression tests: agent-loop guards and final-answer semantics.

Covers the reported runtime integration bug:

1. ``query_experts`` must NOT be registered (or callable) when no expert
   selector is configured — the model can then never loop on it.
2. Identical tool calls (same tool + same arguments) repeated within a run
   are stopped by a loop guard: the tool is marked unavailable and removed
   from the registry, and the run ends with a graceful natural-language
   answer instead of spinning to the step cap.
3. Cutoff endings (max_steps) always produce a real NL answer, never an
   empty string and never a tool-call JSON object.
4. The final answer reaches the API surface: ``GET /v1/agent-runs/{id}``
   and the SSE ``end`` frame both carry it.
5. Project/file tasks execute real filesystem tools (read_file etc.).
6. A model-produced final text that looks like a tool-call JSON object is
   never surfaced as the answer (conversation-level and finalize-level).
7. The LangGraph native loop (used by /v1/run) applies the same repeated-
   call guard.
"""
import asyncio
import json
from typing import Any, Dict, List, Optional, Sequence

import pytest
from fastapi.testclient import TestClient

from orcha.agent_runtime.agent import ToolCallingAgent
from orcha.agent_runtime.backend import (
    ModelBackend, ModelResponse, TokenUsage, ToolCall,
)
from orcha.agent_runtime.conversation import (
    Conversation, ConversationConfig, sanitize_final_answer,
)
from orcha.agent_runtime.tools import ToolRegistry, build_default_tools
from orcha.agent_runtime.workspace import ToolWorkspace
from orcha.api.server import app


class FakeBackend(ModelBackend):
    """Scripted backend; answers "done" when the script is exhausted."""

    def __init__(self, responses: Optional[Sequence[ModelResponse]] = None,
                 model: str = "fake-model") -> None:
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


def _build_conversation(script, roots, config=None):
    registry = ToolRegistry(build_default_tools(roots=roots))
    agent = ToolCallingAgent(FakeBackend(script), tools=registry)
    workspace = ToolWorkspace(registry)
    convo = Conversation(
        agent, workspace, config=config or ConversationConfig(), query="q",
    )
    return convo, registry


# ── 1. query_experts is only registered when a selector exists ───────────────

def test_query_experts_not_registered_without_selector(tmp_path):
    tools = build_default_tools(roots=[str(tmp_path)])
    names = [t.name for t in tools]
    assert "query_experts" not in names
    assert "read_file" in names
    assert "list_directory" in names
    assert "search_text" in names
    assert "directory_tree" in names
    assert "run_command" in names


def test_query_experts_registered_with_selector():
    class DummySelector:
        def select(self, packet):  # pragma: no cover - never invoked
            raise AssertionError("select should not be called")

    tools = build_default_tools(roots=[], selector=DummySelector())
    assert "query_experts" in [t.name for t in tools]


# ── 2. Loop guard: identical repeated calls are stopped ──────────────────────

async def test_repeated_identical_tool_call_is_marked_unavailable(tmp_path):
    note = tmp_path / "notes.txt"
    note.write_text("hello workspace", encoding="utf-8")
    script = [
        tool_call_response("read_file", {"path": "notes.txt"}),
        tool_call_response("read_file", {"path": "notes.txt"}),
        tool_call_response("read_file", {"path": "notes.txt"}),
        tool_call_response("read_file", {"path": "notes.txt"}),
    ]
    convo, registry = _build_conversation(script, roots=[str(tmp_path)])
    convo.submit_user_message("read notes")
    result = await convo.run()

    assert result.terminated_by == "finish"
    assert result.answer == "done"
    # The 3rd identical call (count 3 > max_tool_repeats=2) was NOT executed.
    assert not registry.has("read_file"), "loop guard must remove the tool"
    texts = [
        ev.payload.content
        for ev in convo.log.events
        if ev.kind.value == "observation" and hasattr(ev.payload, "content")
    ]
    unavailable = [t for t in texts if "[tool unavailable]" in t]
    assert unavailable, "the loop guard must record the unavailable decision"
    executed = [t for t in texts if "hello workspace" in t]
    assert len(executed) == 2, "only the first two identical calls may execute"


async def test_always_repeating_backend_ends_gracefully(tmp_path):
    # The backend never finishes and always issues the same call — the loop
    # guard must stop it long before max_steps.
    script = [tool_call_response("read_file", {"path": "nope.txt"})] * 50
    convo, _ = _build_conversation(
        script,
        roots=[str(tmp_path)],
        config=ConversationConfig(max_steps=30),
    )
    convo.submit_user_message("read nope")
    result = await convo.run()

    assert result.steps < 30, "loop guard must stop before the step cap"
    assert result.answer and result.answer.strip()
    assert not result.answer.startswith("{"), "answer must be natural language"
    assert not result.answer.startswith("[")


# ── 3. Cutoffs end with a real NL answer ─────────────────────────────────────

async def test_cutoff_ends_with_natural_language_answer(tmp_path):
    script = [tool_call_response("read_file", {"path": "a.txt"})] * 10
    convo, _ = _build_conversation(
        script,
        roots=[str(tmp_path)],
        config=ConversationConfig(max_steps=2),
    )
    convo.submit_user_message("read a")
    result = await convo.run()

    assert result.terminated_by == "max_steps"
    assert result.answer and result.answer.strip()
    assert not result.answer.startswith("{")
    assert "limit" in result.answer


async def test_finish_with_tool_json_content_is_sanitized(tmp_path):
    json_content = json.dumps({"name": "query_experts", "arguments": {"query": "decide"}})
    script = [ModelResponse(
        content=json_content,
        usage=TokenUsage(prompt_tokens=5, completion_tokens=20),
        finish_reason="stop",
    )]
    convo, _ = _build_conversation(script, roots=[str(tmp_path)])
    convo.submit_user_message("hi")
    result = await convo.run()

    assert result.answer is not None
    assert not result.answer.startswith("{"), "tool JSON must never be the answer"
    assert "query_experts" not in result.answer


# ── sanitize_final_answer unit cases ─────────────────────────────────────────

def test_sanitize_final_answer_cases():
    assert sanitize_final_answer(None) is None
    assert sanitize_final_answer("   ") is None
    assert sanitize_final_answer('{"name": "query_experts", "arguments": {}}') is None
    assert sanitize_final_answer('{"kind": "tool_call"}') is None
    assert sanitize_final_answer('{"tool_call": 1}') is None
    assert sanitize_final_answer('{"a": "real json answer"}') is None
    assert sanitize_final_answer('[{"name": "read_file", "arguments": {}}]') is None
    assert sanitize_final_answer("[stub] canned reply") == "[stub] canned reply"
    assert sanitize_final_answer("a real answer") == "a real answer"


# ── 4. The answer reaches the API surface ────────────────────────────────────

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _wait_finished(client, run_id, timeout_s=15.0):
    import time
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        body = client.get(f"/v1/agent-runs/{run_id}").json()
        if body["status"] == "finished":
            return body
        time.sleep(0.05)
    raise AssertionError("run did not finish in time")


def _sse_frames(client, run_id):
    frames = []
    with client.stream("GET", f"/v1/agent-runs/{run_id}/events") as resp:
        for line in resp.iter_lines():
            if line.startswith("data:"):
                frames.append(line[len("data:"):].strip())
    return [json.loads(f) for f in frames if f]


def test_get_run_and_end_frame_carry_answer(client, tmp_path):
    r = client.post("/v1/agent-runs", json={
        "query": "hi",
        "workspace_roots": [str(tmp_path)],
        "backend": {"type": "stub"},
    })
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    client.post(f"/v1/agent-runs/{run_id}/start")
    body = _wait_finished(client, run_id)

    assert body["error"] is None
    assert body["answer"] and body["answer"].strip()
    assert not body["answer"].startswith("{")

    frames = _sse_frames(client, run_id)
    end = [f for f in frames if f.get("kind") == "end"]
    assert end, "the stream must end with an end frame"
    assert end[-1].get("answer"), "the end frame must carry the final answer"


# ── 5. Project/file tasks execute real filesystem tools ──────────────────────

async def test_file_task_executes_real_filesystem_tools(tmp_path):
    note = tmp_path / "notes.txt"
    note.write_text("the secret payload is XYZ", encoding="utf-8")
    script = [
        tool_call_response("list_directory", {"path": "."}),
        tool_call_response("read_file", {"path": "notes.txt"}),
    ]
    convo, _ = _build_conversation(script, roots=[str(tmp_path)])
    convo.submit_user_message("go through this file and tell me what we have")
    result = await convo.run()

    assert result.terminated_by == "finish"
    texts = [
        ev.payload.content
        for ev in convo.log.events
        if ev.kind.value == "observation" and hasattr(ev.payload, "content")
    ]
    assert any("the secret payload is XYZ" in t for t in texts), (
        "the read_file tool must have actually read the attached file"
    )
    assert result.answer == "done"


# ── 6. Finalize (the /v1/run pipeline) never returns empty / JSON ────────────

async def test_finalize_agent_chat_empty_answer_falls_back():
    from orcha.builders.agent import _FinalizeAgent
    from orcha.core.packets import OrchaPacket, PacketKind

    node = _FinalizeAgent()
    pkt = OrchaPacket(
        kind=PacketKind.QUERY, query="hi",
        payload={"intent_gate_intent": "chat", "intent_gate_response": ""},
    )
    out = await node.run(pkt, None)
    answer = out.payload.get("answer", "")
    assert answer and answer.strip()
    assert not answer.startswith("{")


async def test_finalize_agent_json_output_is_replaced():
    from orcha.builders.agent import _FinalizeAgent
    from orcha.core.packets import OrchaPacket, PacketKind

    node = _FinalizeAgent()
    pkt = OrchaPacket(
        kind=PacketKind.QUERY, query="go through the file",
        payload={
            "agent_output": '{"name": "query_experts", "arguments": {"query": "decide your next action", "limit": 3}}',
            "agent_steps": [{"iteration": 1}],
            "agent_tool_calls": [{"name": "query_experts"}],
            "agent_iterations": 1,
            "agent_completed": True,
        },
    )
    out = await node.run(pkt, None)
    answer = out.payload.get("answer", "")
    assert answer and answer.strip()
    assert not answer.startswith("{")
    assert "query_experts" not in answer


# ── 7. The LangGraph native loop applies the same guard ──────────────────────

async def test_graph_text_loop_repeated_call_guard():
    """The legacy TOOL:name:arg text protocol (used when no completion_fn is
    configured — the path a model without native tool-calling support falls
    back to) must apply the same repeated-call guard as the native loop.
    Confirmed as a real gap directly: a live run against a free OpenRouter
    model with no native tool-calling issued `TOOL:current_workspace:{}`
    twice in a row with nothing stopping a third, each repeat costing a full
    (slow, free-tier) model round-trip for zero progress."""
    from orcha.nodes.agent import AgentConfig, AgentNode
    from orcha.capabilities.base import READ, SAFE, ToolExecutor, spec

    async def model_fn(prompt, system):
        return 'TOOL:read_file:{"path": "a.txt"}'

    tool_spec = spec(
        "read_file", "Read a file.",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        lambda path: "file contents",
        permissions=[READ], safety_level=SAFE,
    )
    executor = ToolExecutor([tool_spec])

    node = AgentNode(
        name="agent",
        config=AgentConfig(
            model_fn=model_fn,
            executor=executor,
            max_iterations=5,
        ),
    )
    from orcha.core.packets import OrchaPacket, PacketKind
    ctx = type("Ctx", (), {"cancelled": False, "run_id": "test-run"})()
    out = await node.run(
        OrchaPacket(kind=PacketKind.QUERY, query="read a"),
        ctx,
    )
    tool_calls = out.payload.get("agent_tool_calls", [])
    unavailable = [t for t in tool_calls if "[tool unavailable]" in (t.get("result") or "")]
    assert unavailable, "the 3rd+ identical text-protocol call must be blocked by the guard"
    executed = [t for t in tool_calls if t.get("result") == "file contents"]
    assert len(executed) == 2, "only the first two identical calls may actually execute"


async def test_graph_native_loop_repeated_call_guard():
    from orcha.nodes.agent import AgentConfig, AgentNode
    from orcha.capabilities.base import READ, SAFE, ToolExecutor, spec

    async def completion_fn(messages, system_prompt, tool_schemas):
        return {
            "content": "",
            "tool_calls": [{
                "id": f"call-{len(messages)}",
                "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
            }],
        }

    tool_spec = spec(
        "read_file", "Read a file.",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        lambda path: "file contents",
        permissions=[READ], safety_level=SAFE,
    )
    executor = ToolExecutor([tool_spec])

    node = AgentNode(
        name="agent",
        config=AgentConfig(
            completion_fn=completion_fn,
            executor=executor,
            max_iterations=4,
        ),
    )
    from orcha.core.packets import OrchaPacket, PacketKind
    ctx = type("Ctx", (), {"cancelled": False, "run_id": "test-run"})()
    out = await node.run(
        OrchaPacket(kind=PacketKind.QUERY, query="read a"),
        ctx,
    )
    tool_calls = out.payload.get("agent_tool_calls", [])
    blocked = [t for t in tool_calls if t.get("error_code") == "tool_loop_guard"]
    assert blocked, "the repeated identical call must be blocked by the guard"
    assert len(blocked) >= 1
    output = out.payload.get("agent_output") or ""
    assert output and output.strip(), "graceful NL output must replace the empty placeholder"
    assert not output.startswith("{"), "loop output must be natural language"
    assert "4 iterations" in output
