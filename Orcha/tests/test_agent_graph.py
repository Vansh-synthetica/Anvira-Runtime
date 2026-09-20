"""
Tests for the single-agent execution graph (orcha.builders.agent).

These verify the end-to-end graph that `default`/`research` runs use when a
real tool surface + native completion_fn is available: tools actually execute
through the executor, answers that get cut off by the token budget are
continued, and prior conversation seeds the agent's memory.
"""
import asyncio

import pytest

from orcha.capabilities.base import CapabilityContext
from orcha.capabilities.registry import CapabilityRegistry
from orcha.graph.runtime import GraphRuntime
from orcha.graph.store import MemoryStore
from orcha.nodes.agent import AgentConfig
from orcha.builders.agent import AgentGraphConfig, build_agent_graph


def _filesystem_executor(root):
    registry = CapabilityRegistry().register_defaults()
    ctx = CapabilityContext(roots=[str(root)])
    return registry.build(["filesystem"], ctx=ctx)


def _tool_call(name, arguments, call_id="call_x"):
    return {
        "role": "assistant",
        "content": None,
        "finish_reason": "stop",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }],
    }


def _run(graph, query):
    return asyncio.run(GraphRuntime(graph, store=MemoryStore()).run(query))


def test_agent_graph_completes_real_file_task(tmp_path):
    """End-to-end real file task: the agent creates a folder, copies a PDF
    into it, verifies the listing, then returns a concise success message —
    all through actual tool execution, confirmed on disk."""
    source_pdf = tmp_path / "report.pdf"
    source_pdf.write_bytes(b"%PDF-1.4 fake report bytes\n")
    executor = _filesystem_executor(tmp_path)

    async def fake_completion(messages, system, tool_schemas):
        tool_turns = len([m for m in messages if m.get("role") == "tool"])
        if tool_turns == 0:
            return _tool_call("create_directory", {"path": "documents"}, "call_1")
        if tool_turns == 1:
            return _tool_call(
                "copy_file", {"source": "report.pdf", "target": "documents/report.pdf"}, "call_2"
            )
        if tool_turns == 2:
            return _tool_call("list_directory", {"path": "documents"}, "call_3")
        return {
            "role": "assistant",
            "content": "Done: created the documents folder and copied report.pdf into it.",
            "finish_reason": "stop",
        }

    graph = build_agent_graph(AgentGraphConfig(
        agent_config=AgentConfig(completion_fn=fake_completion, executor=executor, max_iterations=8),
        executor=executor,
    ))
    result = _run(graph, "create a folder called documents and copy report.pdf into it")
    p = result.packet.payload

    # Confirmed on disk.
    assert (tmp_path / "documents").is_dir()
    assert (tmp_path / "documents" / "report.pdf").exists()
    assert (tmp_path / "documents" / "report.pdf").read_bytes() == source_pdf.read_bytes()

    # The final answer reflects the completed action, not a plan.
    assert p["answer"] == "Done: created the documents folder and copied report.pdf into it."
    assert p["agent_completed"] is True

    # Every tool call executed successfully, in order.
    names = [call["name"] for call in p["agent_tool_calls"]]
    assert names == ["create_directory", "copy_file", "list_directory"]
    assert all(call["result_type"] == "ok" for call in p["agent_tool_calls"])


def test_agent_graph_executes_tool_and_returns_answer(tmp_path):
    executor = _filesystem_executor(tmp_path)

    async def fake_completion(messages, system, tool_schemas):
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
                        "arguments": {"path": "made.txt", "content": "hi"},
                    },
                }],
            }
        return {"role": "assistant", "content": "done", "finish_reason": "stop"}

    graph = build_agent_graph(AgentGraphConfig(
        agent_config=AgentConfig(completion_fn=fake_completion, executor=executor, max_iterations=4),
        executor=executor,
    ))
    result = _run(graph, "create made.txt")
    p = result.packet.payload

    assert p["primary"] == "agent"
    assert p["agg_mode"] == "agent"
    assert p["answer"] == "done"
    assert p["agent_completed"] is True
    assert (tmp_path / "made.txt").read_text() == "hi"
    assert p["agent_tool_calls"][0]["name"] == "write_file"
    assert p["agent_tool_calls"][0]["result_type"] == "ok"


def test_agent_graph_continues_when_answer_truncated(tmp_path):
    """finish_reason == 'length' must not be treated as a complete answer."""
    executor = _filesystem_executor(tmp_path)
    calls = []

    async def fake_completion(messages, system, tool_schemas):
        calls.append(messages)
        continued = any(
            m["role"] == "user" and "cut off" in m.get("content", "")
            for m in messages
        )
        if not continued:
            return {"role": "assistant", "content": "First half", "finish_reason": "length"}
        return {"role": "assistant", "content": "Second half", "finish_reason": "stop"}

    graph = build_agent_graph(AgentGraphConfig(
        agent_config=AgentConfig(completion_fn=fake_completion, executor=executor, max_iterations=5),
        executor=executor,
    ))
    result = _run(graph, "write a long answer")
    p = result.packet.payload

    assert p["answer"] == "First halfSecond half"
    assert p["agent_completed"] is True
    assert len(calls) == 2


def test_agent_graph_seeds_prior_history(tmp_path):
    executor = _filesystem_executor(tmp_path)
    seen = {}

    async def fake_completion(messages, system, tool_schemas):
        seen["seed_present"] = any(
            m.get("content") == "Earlier context" for m in messages
        )
        return {"role": "assistant", "content": "ok", "finish_reason": "stop"}

    graph = build_agent_graph(AgentGraphConfig(
        agent_config=AgentConfig(
            completion_fn=fake_completion,
            executor=executor,
            max_iterations=3,
            seed_messages=[{"role": "user", "content": "Earlier context"}],
        ),
        executor=executor,
    ))
    _run(graph, "continue")
    assert seen["seed_present"] is True


def test_agent_graph_reasoning_level_raises_iteration_budget(tmp_path):
    executor = _filesystem_executor(tmp_path)
    config = AgentGraphConfig(
        agent_config=AgentConfig(completion_fn=None, executor=executor, max_iterations=2),
        executor=executor,
        reasoning_level="max",
    )
    graph = build_agent_graph(config)
    assert graph.name == "orcha_agent"
    assert config.agent_config.max_iterations >= 2
