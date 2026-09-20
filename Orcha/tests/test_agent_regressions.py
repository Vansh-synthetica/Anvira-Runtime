"""
Regression tests for the agent bug-fix pass.

Covers the backend fixes:
  BUG 1  — multi-agent workers now propagate seed_messages
  BUG 2  — tool_call_id is resolved/written back so tool messages always match
  BUG 3  — the synthesis node receives the model_fn (real synthesis, not dry-run)
  BUG 4  — interactive tool-approval channel (ApprovalBroker + agent loop wait)
  BUG 6  — agent iteration counts surface in RunResult.iterations
  BUG 7  — truncated answers are marked 'continued', no empty assistant msg
  BUG 9  — transient model errors retry; repeated errors fail the node
  S1     — malformed tool-call arguments are not silently executed as {}
  S5     — failed verification triggers one bounded re-synthesis
"""
import asyncio

import pytest

from orcha.capabilities.base import (
    ApprovalBroker,
    CapabilityContext,
    PermissionPolicy,
)
from orcha.capabilities.registry import CapabilityRegistry
from orcha.graph.runtime import GraphRuntime
from orcha.graph.store import MemoryStore
from orcha.nodes.agent import AgentConfig, AgentNode
from orcha.builders.agent import AgentGraphConfig, build_agent_graph
from orcha.builders.multi_agent import MultiAgentGraphConfig, build_multi_agent_graph

from test_nodes import make_ctx, make_packet


def _filesystem_executor(root, policy=None):
    registry = CapabilityRegistry().register_defaults()
    ctx = CapabilityContext(roots=[str(root)])
    return registry.build(["filesystem"], ctx=ctx, policy=policy)


def _run(graph, query):
    return asyncio.run(GraphRuntime(graph, store=MemoryStore()).run(query))


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


# ── BUG 2: tool_call_id sync ──────────────────────────────────────────────────

def test_agent_loop_syncs_missing_tool_call_ids(tmp_path):
    """A model that omits tool-call ids must still produce matching tool messages."""
    executor = _filesystem_executor(tmp_path)
    captured = {}

    async def fake_completion(messages, system, tool_schemas):
        if not any(m.get("role") == "tool" for m in messages):
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": None,
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": {"path": "made.txt", "content": "hi"},
                    },
                }],
            }
        captured["second_messages"] = [dict(m) for m in messages]
        return {"role": "assistant", "content": "done", "finish_reason": "stop"}

    agent = AgentNode(
        name="id_sync_agent",
        config=AgentConfig(completion_fn=fake_completion, executor=executor, max_iterations=4),
    )
    result = asyncio.run(agent.run(make_packet(task="write made.txt"), make_ctx()))
    calls = result.payload["agent_tool_calls"]

    assert (tmp_path / "made.txt").read_text() == "hi"
    assert calls[0]["result_type"] == "ok"
    assert calls[0]["tool_call_id"] == "call_1_0"

    assistant = next(m for m in captured["second_messages"] if m.get("role") == "assistant")
    tool = next(m for m in captured["second_messages"] if m.get("role") == "tool")
    assistant_id = assistant["tool_calls"][0]["id"]
    assert assistant_id == "call_1_0"
    assert tool["tool_call_id"] == assistant_id


# ── S1: malformed tool-call arguments ─────────────────────────────────────────

def test_agent_malformed_tool_args_not_executed(tmp_path):
    executor = _filesystem_executor(tmp_path)

    async def fake_completion(messages, system, tool_schemas):
        if not any(m.get("role") == "tool" for m in messages):
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": "{this is not json",
                    },
                }],
            }
        return {"role": "assistant", "content": "done", "finish_reason": "stop"}

    agent = AgentNode(
        name="bad_args_agent",
        config=AgentConfig(completion_fn=fake_completion, executor=executor, max_iterations=4),
    )
    result = asyncio.run(agent.run(make_packet(task="write made.txt"), make_ctx()))
    call = result.payload["agent_tool_calls"][0]

    assert call["error_code"] == "malformed_arguments"
    assert "could not be parsed" in call["result"]
    assert not (tmp_path / "made.txt").exists()
    assert result.payload["agent_completed"] is True


# ── BUG 9: model-error handling ───────────────────────────────────────────────

def test_agent_model_error_recovers_with_retry():
    attempts = {"n": 0}

    async def fake_completion(messages, system, tool_schemas):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("server went away")
        return {"role": "assistant", "content": "recovered", "finish_reason": "stop"}

    agent = AgentNode(
        name="retry_agent",
        config=AgentConfig(completion_fn=fake_completion, max_iterations=4),
    )
    result = asyncio.run(agent.run(make_packet(task="t"), make_ctx()))

    assert result.payload["agent_output"] == "recovered"
    assert result.payload["agent_completed"] is True
    assert result.payload["agent_steps"][0]["status"] == "error"


def test_agent_repeated_model_errors_fail_run():
    async def fake_completion(messages, system, tool_schemas):
        raise ConnectionError("server went away")

    agent = AgentNode(
        name="failing_agent",
        config=AgentConfig(completion_fn=fake_completion, max_iterations=4),
    )
    with pytest.raises(RuntimeError, match="Model call failed 2 times in a row"):
        asyncio.run(agent.run(make_packet(task="t"), make_ctx()))


# ── BUG 7: truncation step status + no empty assistant message ────────────────

def test_agent_truncation_marks_step_continued(tmp_path):
    executor = _filesystem_executor(tmp_path)
    captured = {}
    calls = {"n": 0}

    async def fake_completion(messages, system, tool_schemas):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"role": "assistant", "content": "First half", "finish_reason": "length"}
        captured["second_messages"] = [dict(m) for m in messages]
        return {"role": "assistant", "content": "Second half", "finish_reason": "stop"}

    agent = AgentNode(
        name="trunc_agent",
        config=AgentConfig(completion_fn=fake_completion, executor=executor, max_iterations=5),
    )
    result = asyncio.run(agent.run(make_packet(task="long answer"), make_ctx()))
    steps = result.payload["agent_steps"]

    assert steps[0]["status"] == "continued"
    assert steps[0]["note"] == "truncated; continuing"
    assert steps[1]["status"] == "completed"
    # The continuation must not inject an empty assistant message.
    for m in captured["second_messages"]:
        if m.get("role") == "assistant":
            assert m.get("content")


# ── BUG 6: agent iterations surface in RunResult ──────────────────────────────

def test_agent_run_iterations_surface_in_result(tmp_path):
    executor = _filesystem_executor(tmp_path)
    turns = {"n": 0}

    async def fake_completion(messages, system, tool_schemas):
        turns["n"] += 1
        if turns["n"] == 1:
            return _tool_call("write_file", {"path": "made.txt", "content": "hi"}, "call_1")
        return {"role": "assistant", "content": "done", "finish_reason": "stop"}

    graph = build_agent_graph(AgentGraphConfig(
        agent_config=AgentConfig(completion_fn=fake_completion, executor=executor, max_iterations=4),
        executor=executor,
    ))
    result = _run(graph, "write made.txt")

    assert result.iterations == 2
    assert result.packet.payload["agent_iterations"] == 2
    assert result.agent_completed is True


# ── BUG 4: interactive approval channel ───────────────────────────────────────

def test_approval_broker_lifecycle():
    broker = ApprovalBroker()

    entry = broker.request("run-1", "call_x", "delete_file", {"path": "x.txt"})
    assert entry["status"] == "pending"
    assert entry["tool_call_id"] == "call_x"
    assert len(broker.list_pending("run-1")) == 1
    assert broker.list_pending("run-2") == []

    # Timeout yields None (no decision in time).
    async def _timeout():
        return await broker.wait(entry["id"], timeout=0.01)
    assert asyncio.run(_timeout()) is None

    # A decision for a different run is rejected.
    assert broker.decide(entry["id"], approved=True, run_id="other-run") is False
    # The real decision lands.
    assert broker.decide(entry["id"], approved=True, run_id="run-1") is True
    assert broker.decide(entry["id"], approved=True, run_id="run-1") is False  # already decided

    async def _wait_approved():
        return await broker.wait(entry["id"], timeout=1)
    assert asyncio.run(_wait_approved()) is True

    # One-shot grant: consumed by the policy check — and strictly scoped to
    # the run that obtained it.
    assert broker.has_approval("run-1", "delete_file", {"path": "x.txt"}) is True
    assert broker.has_approval("run-1", "delete_file", {"path": "x.txt"}) is False
    # A different run's identical call is NOT covered by run-1's grant.
    assert broker.has_approval("run-2", "delete_file", {"path": "x.txt"}) is False
    # Different arguments are not covered by the grant.
    assert broker.has_approval("run-1", "delete_file", {"path": "other.txt"}) is False

    # Rejection path.
    e2 = broker.request("run-1", "call_y", "delete_file", {"path": "y.txt"})
    assert broker.decide(e2["id"], approved=False) is True

    async def _wait_rejected():
        return await broker.wait(e2["id"], timeout=1)
    assert asyncio.run(_wait_rejected()) is False
    assert broker.has_approval("run-1", "delete_file", {"path": "y.txt"}) is False


def test_approval_broker_grant_is_run_scoped():
    """A grant recorded for one run can never satisfy another run's call."""
    broker = ApprovalBroker()
    e_a = broker.request("run-a", "call_1", "delete_file", {"path": "x.txt"})
    assert broker.decide(e_a["id"], approved=True, run_id="run-a") is True

    assert broker.has_approval("run-a", "delete_file", {"path": "x.txt"}) is True
    # Same broker, same tool, same args — different run: no grant.
    assert broker.has_approval("run-b", "delete_file", {"path": "x.txt"}) is False


def test_approval_broker_clear_run_removes_state():
    """Terminating a run must purge its pending requests and grants."""
    broker = ApprovalBroker()
    e_a = broker.request("run-a", "call_1", "delete_file", {"path": "a.txt"})
    e_b = broker.request("run-b", "call_1", "delete_file", {"path": "b.txt"})
    assert broker.decide(e_a["id"], approved=True, run_id="run-a") is True

    broker.clear_run("run-a")
    assert broker.list_pending("run-a") == []
    assert [e["id"] for e in broker.list_pending("run-b")] == [e_b["id"]]
    # The decided entry is gone — it can no longer be decided late.
    assert broker.decide(e_a["id"], approved=True, run_id="run-a") is False
    # Its grant is gone too — run-b (or any run) can't pick it up.
    assert broker.has_approval("run-a", "delete_file", {"path": "a.txt"}) is False
    assert broker.has_approval("run-b", "delete_file", {"path": "a.txt"}) is False


def test_agent_loop_waits_for_approval_then_executes(tmp_path):
    target = tmp_path / "x.txt"
    target.write_text("boom")
    broker = ApprovalBroker()
    executor = _filesystem_executor(
        tmp_path, policy=PermissionPolicy(access_mode="approval", broker=broker)
    )

    async def fake_completion(messages, system, tool_schemas):
        if not any(m.get("role") == "tool" for m in messages):
            return _tool_call("delete_file", {"path": "x.txt"}, "call_9")
        return {"role": "assistant", "content": "done", "finish_reason": "stop"}

    agent = AgentNode(
        name="approval_agent",
        config=AgentConfig(
            completion_fn=fake_completion,
            executor=executor,
            max_iterations=4,
            approval_broker=broker,
            approval_timeout_s=5,
        ),
    )

    async def approver():
        await asyncio.sleep(0.05)
        pending = broker.list_pending()
        assert len(pending) == 1
        assert pending[0]["tool"] == "delete_file"
        assert broker.decide(pending[0]["id"], approved=True) is True

    async def main():
        task = asyncio.create_task(approver())
        try:
            return await agent.run(make_packet(task="delete x.txt"), make_ctx())
        finally:
            await task

    result = asyncio.run(main())
    call = result.payload["agent_tool_calls"][0]

    assert call["name"] == "delete_file"
    assert call["result_type"] == "ok"
    assert not target.exists()
    assert result.payload["agent_completed"] is True


def test_agent_approval_not_leaked_across_runs(tmp_path):
    """An approval granted to one run must never auto-authorize another
    run's identical tool call — same broker, same workspace, same args."""
    target = tmp_path / "x.txt"
    broker = ApprovalBroker()

    def make_agent():
        executor = _filesystem_executor(
            tmp_path, policy=PermissionPolicy(access_mode="approval", broker=broker)
        )

        async def fake_completion(messages, system, tool_schemas):
            if not any(m.get("role") == "tool" for m in messages):
                return _tool_call("delete_file", {"path": "x.txt"}, "call_9")
            return {"role": "assistant", "content": "done", "finish_reason": "stop"}

        return AgentNode(
            name="approval_agent",
            config=AgentConfig(
                completion_fn=fake_completion,
                executor=executor,
                max_iterations=4,
                approval_broker=broker,
                approval_timeout_s=0.05,
            ),
        )

    async def main():
        target.write_text("boom")
        agent_a = make_agent()

        async def approver():
            await asyncio.sleep(0.02)
            pending = broker.list_pending()
            assert len(pending) == 1
            return broker.decide(pending[0]["id"], approved=True)

        task = asyncio.create_task(approver())
        try:
            result_a = await agent_a.run(
                make_packet(task="delete x.txt"), make_ctx(run_id="run-a")
            )
        finally:
            await task
        assert result_a.payload["agent_tool_calls"][0]["result_type"] == "ok"
        assert not target.exists()

        # Second run, fresh executor but the SAME shared broker, identical
        # call. It must NOT inherit run-a's grant — no decision was made
        # for this run, so the call is denied again.
        target.write_text("boom")
        agent_b = make_agent()
        result_b = await agent_b.run(
            make_packet(task="delete x.txt"), make_ctx(run_id="run-b")
        )
        call_b = result_b.payload["agent_tool_calls"][0]
        assert call_b["error_code"] == "approval_required"
        assert target.exists()

    asyncio.run(main())


# ── BUG 1: multi-agent workers propagate seed_messages ────────────────────────

def test_multi_agent_worker_seeds_prior_history():
    captured = {"seeded": None}

    async def fake_completion(messages, system, tool_schemas):
        if captured["seeded"] is None:
            captured["seeded"] = any(
                m.get("content") == "Earlier context" for m in messages
            )
        return {"role": "assistant", "content": "branch done", "finish_reason": "stop"}

    graph = build_multi_agent_graph(
        MultiAgentGraphConfig(
            agent_config=AgentConfig(
                completion_fn=fake_completion,
                max_iterations=2,
                seed_messages=[{"role": "user", "content": "Earlier context"}],
            ),
            verify=False,
        )
    )
    result = _run(graph, "Explain AI")

    assert result.run_id
    assert captured["seeded"] is True


# ── BUG 3: synthesis receives the model_fn ────────────────────────────────────

def test_multi_agent_synthesis_uses_model_fn():
    async def fake_worker(messages, system, tool_schemas):
        return {"role": "assistant", "content": "branch answer", "finish_reason": "stop"}

    async def fake_synth(prompt, system):
        return "ANSWER: synthesized answer"

    graph = build_multi_agent_graph(
        MultiAgentGraphConfig(
            agent_config=AgentConfig(completion_fn=fake_worker, max_iterations=2),
            model_fn=fake_synth,
            verify=False,
        )
    )
    result = _run(graph, "Explain AI")
    p = result.packet.payload

    assert p["synthesized"] is True
    assert p["answer"] == "synthesized answer"
    assert p["confidence"] == 0.7


# ── S5: verification failure triggers one bounded retry ───────────────────────

def test_multi_agent_verify_retry_is_bounded():
    graph = build_multi_agent_graph(verify=True)
    result = _run(graph, "Explain AI")
    p = result.packet.payload

    # Dry-run synthesis lands below the verify confidence bar (0.4 < 0.5),
    # so exactly one bounded re-synthesis happens, then the run ends.
    assert result.run_id
    assert p.get("verify_retries") == 1
    assert p.get("verify_failures")
    verify_passes = [t for t in result.trace if t.get("stage") == "verify"]
    assert len(verify_passes) == 2
