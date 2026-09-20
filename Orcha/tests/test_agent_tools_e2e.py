"""
End-to-end integration test for the agent tools pipeline.
"""
import asyncio
import json
import os
import tempfile

import pytest

from orcha.core.packets import OrchaPacket, PacketKind
from orcha.graph.context import RunContext, CancelToken, EventEmitter
from orcha.graph.store import MemoryStore
from orcha.observability import get_logger
from orcha.nodes.intent import (
    IntentGateConfig, IntentGateNode, INTENT_FILE_OPERATION,
    INTENT_SEARCH, INTENT_TOOL_REQUEST, _fast_path_intent,
)
from orcha.nodes.agent import AgentConfig, AgentNode
from orcha.capabilities.base import CapabilityContext
from orcha.capabilities.registry import CapabilityRegistry
from orcha.builders.agent import AgentGraphConfig, build_agent_graph


def make_packet(**payload):
    return OrchaPacket(
        kind=PacketKind.AGGREGATION,
        query="test query",
        payload=payload,
    )


def make_ctx(run_id="test-run"):
    return RunContext(
        run_id=run_id, store=MemoryStore(), cancel=CancelToken(),
        emit=EventEmitter(run_id, logger=get_logger("test")),
        logger=get_logger("test"), graph_name="test",
    )


def _gate_packet(task):
    """Create a packet the way the intent gate expects it."""
    return OrchaPacket(kind=PacketKind.QUERY, query=task)


# ── 1. Fast-path heuristic ─────────────────────────────────────────────────

class TestFastPathHeuristic:
    @pytest.mark.parametrize("query,expected", [
        ("write a file called main.py", INTENT_FILE_OPERATION),
        ("create a new component", INTENT_FILE_OPERATION),
        ("edit the config.yaml", INTENT_FILE_OPERATION),
        ("delete the old folder", INTENT_FILE_OPERATION),
        ("rename main.ts to app.tsx", INTENT_FILE_OPERATION),
        ("search for TODO comments", INTENT_SEARCH),
        ("find files containing import", INTENT_SEARCH),
        ("list all files in the project", INTENT_SEARCH),
        ("run the tests", INTENT_TOOL_REQUEST),
        ("build the app", INTENT_TOOL_REQUEST),
        ("compile this project", INTENT_TOOL_REQUEST),
    ])
    def test_tool_requests_detected(self, query, expected):
        assert _fast_path_intent(query) == expected

    @pytest.mark.parametrize("query", [
        "hello how are you", "what is the meaning of life",
        "tell me a joke", "write me a poem about nature",
        "who are you", "explain quantum computing",
    ])
    def test_conversations_not_misclassified(self, query):
        assert _fast_path_intent(query) is None


# ── 2. Intent gate routing ─────────────────────────────────────────────────

class TestIntentGateToAgentGraph:
    def test_file_operation_bypasses_model(self):
        async def bad_completion(m, s, t):
            raise AssertionError("Model should not be called")
        gate = IntentGateNode(config=IntentGateConfig(completion_fn=bad_completion))
        result = asyncio.run(gate.run(_gate_packet("write a file called main.py"), make_ctx()))
        assert result.payload["intent_gate_intent"] == INTENT_FILE_OPERATION

    def test_search_bypasses_model(self):
        async def bad_completion(m, s, t):
            raise AssertionError("Model should not be called")
        gate = IntentGateNode(config=IntentGateConfig(completion_fn=bad_completion))
        result = asyncio.run(gate.run(_gate_packet("search for all TODO comments"), make_ctx()))
        assert result.payload["intent_gate_intent"] == INTENT_SEARCH

    def test_chat_still_calls_model(self):
        model_called = {"v": False}
        async def completion(m, s, t):
            model_called["v"] = True
            return {"role": "assistant", "content": "Hello!"}
        gate = IntentGateNode(config=IntentGateConfig(completion_fn=completion))
        result = asyncio.run(gate.run(_gate_packet("hello how are you"), make_ctx()))
        assert model_called["v"]
        assert result.payload["intent_gate_intent"] == "CHAT"


# ── 3. Agent tool execution ────────────────────────────────────────────────

class TestAgentToolExecution:
    def test_agent_receives_tool_schemas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = CapabilityContext(roots=[tmpdir])
            executor = CapabilityRegistry().register_defaults().build(["workspace", "search"], ctx=ctx)
            tools = executor.tools()
            tool_names = [t.name for t in tools]
            assert len(tools) > 0
            assert any("workspace" in n or "search" in n or "grep" in n for n in tool_names)

    def test_native_loop_passes_tools_to_completion(self):
        calls_received = []
        async def completion(messages, system, tools):
            calls_received.append({
                "has_tools": tools is not None,
                "tool_count": len(tools) if tools else 0,
                "tool_names": [t.get("function", {}).get("name", "") for t in (tools or [])],
            })
            return {"role": "assistant", "content": "Done.", "finish_reason": "stop"}

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = CapabilityContext(roots=[tmpdir])
            executor = CapabilityRegistry().register_defaults().build(["workspace", "search"], ctx=ctx)
            config = AgentConfig(
                completion_fn=completion, tools=executor.tools(),
                executor=executor, capabilities=["workspace", "search"], max_iterations=1,
            )
            agent = AgentNode(name="test_agent", config=config)
            pkt = OrchaPacket(kind=PacketKind.QUERY, query="list all files")
            pkt = pkt.fork(pkt.kind, task="list all files", agent_context="")
            asyncio.run(agent.run(pkt, make_ctx()))

            assert len(calls_received) > 0
            first = calls_received[0]
            assert first["has_tools"], "Model should receive tool schemas"
            assert first["tool_count"] > 0
            # Tools from workspace + search capabilities
            assert len(first["tool_names"]) >= 5

    def test_native_loop_executes_tool_call(self):
        call_count = {"n": 0}
        async def completion(messages, system, tools):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {
                    "role": "assistant", "content": "",
                    "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {"name": "grep", "arguments": json.dumps({"pattern": "TODO"})},
                    }],
                    "finish_reason": "tool_calls",
                }
            return {"role": "assistant", "content": "Found TODOs.", "finish_reason": "stop"}

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = CapabilityContext(roots=[tmpdir])
            executor = CapabilityRegistry().register_defaults().build(["workspace", "search"], ctx=ctx)
            config = AgentConfig(
                completion_fn=completion, tools=executor.tools(),
                executor=executor, capabilities=["workspace", "search"], max_iterations=5,
            )
            agent = AgentNode(name="test_agent", config=config)
            pkt = OrchaPacket(kind=PacketKind.QUERY, query="find TODOs")
            pkt = pkt.fork(pkt.kind, task="find TODOs", agent_context="")
            result = asyncio.run(agent.run(pkt, make_ctx()))

            assert call_count["n"] >= 2
            assert result.payload.get("agent_output")
            tool_calls = result.payload.get("agent_tool_calls", [])
            assert len(tool_calls) > 0
            assert tool_calls[0]["name"] == "grep"


# ── 4. Full graph build ────────────────────────────────────────────────────

class TestFullGraphIntegration:
    def test_graph_builds_with_capabilities(self):
        async def completion(m, s, t):
            return {"role": "assistant", "content": "ok", "finish_reason": "stop"}

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = CapabilityContext(roots=[tmpdir])
            executor = CapabilityRegistry().register_defaults().build(["workspace", "search"], ctx=ctx)
            graph = build_agent_graph(config=AgentGraphConfig(
                agent_config=AgentConfig(
                    completion_fn=completion, tools=executor.tools(),
                    executor=executor, capabilities=["workspace", "search"], max_iterations=3,
                ),
                executor=executor,
                gate_config=IntentGateConfig(completion_fn=completion, system_prompt="Route."),
                capabilities=["workspace", "search"],
            ))
            assert graph is not None
            # graph.nodes is a dict of name -> node
            assert "intent_gate" in graph.nodes
            assert "agent" in graph.nodes
            assert "finalize_agent" in graph.nodes
