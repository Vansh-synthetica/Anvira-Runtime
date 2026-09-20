"""
Tests for the Prompt 6 layer of orcha.agent_runtime: the FINAL execution
gap — a booted runtime must actually execute tool calls in the real
terminal, never just describe them.

- Bootstrap: ``build_agent_runtime`` registers the shipped default toolset
  (run_command/read_file/write_file/query_experts) when ``workspace_roots``
  are configured — before this fix the one-call boot exposed ZERO tools to
  the model, so it could only reply in prose ("I would run ...").
- End-to-end real execution: a scripted model requests ``run_command`` →
  ToolCallAction → ToolWorkspace → terminal capability → real subprocess →
  ToolResultObservation carrying REAL stdout (never stub text), appended to
  the log in order (tool_call → tool_result → finish).
- Safety: denied commands still become ``[denied_command]``
  ErrorObservations; unknown tools error cleanly.
- Wire: ``event_to_wire`` renders the real execution output, so the SSE
  stream Anvira renders reflects the actual command result.
"""
import uuid

import json
from typing import Any, Dict, List, Optional, Sequence

from orcha.agent_runtime import (
    AgentRuntimeConfig, Conversation, build_agent_runtime,
)
from orcha.agent_runtime.backend import (
    ModelBackend, ModelResponse, TokenUsage, ToolCall,
)
from orcha.agent_runtime.events import EventKind
from orcha.agent_runtime.stream import event_to_wire


class FakeBackend(ModelBackend):
    """Scripted backend; records every call so tests can inspect exactly
    what the agent offered the model. Answers "done" when the script is
    exhausted."""

    def __init__(
        self, responses: Optional[Sequence[ModelResponse]] = None, *,
        supports_streaming_tool_calls: bool = False,
        reports_reasoning_tokens: bool = False,
        model: str = "fake-model",
    ) -> None:
        self._responses = list(responses or [])
        self.calls: List[Dict[str, Any]] = []
        self._streams_tools = supports_streaming_tool_calls
        self._reports_reasoning = reports_reasoning_tokens
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_streaming_tool_calls(self) -> bool:
        return self._streams_tools

    @supports_streaming_tool_calls.setter
    def supports_streaming_tool_calls(self, value: bool) -> None:
        self._streams_tools = value

    @property
    def reports_reasoning_tokens(self) -> bool:
        return self._reports_reasoning

    @reports_reasoning_tokens.setter
    def reports_reasoning_tokens(self, value: bool) -> None:
        self._reports_reasoning = value

    def complete(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Any] = None,
    ) -> ModelResponse:
        self.calls.append({
            "messages": list(messages),
            "tools": tools,
            "config": config,
        })
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


def _tool_names(tools: Optional[List[Dict[str, Any]]]) -> List[str]:
    return [t.get("function", {}).get("name", "") for t in tools or []]


def _boot(roots, script=None, **kw) -> Conversation:
    """One-call boot with an injected scripted backend."""
    backend = FakeBackend(script or [])
    conversation, _, _ = build_agent_runtime(
        AgentRuntimeConfig(workspace_roots=roots),
        backend=backend,
        **kw,
    )
    return conversation


# ── Bootstrap: the registration gap ───────────────────────────────────────────

async def test_build_agent_runtime_without_roots_exposes_no_tools():
    """Regression: before the fix, a one-call boot offered the model ZERO
    tools — it could only describe execution in prose."""
    backend = FakeBackend()
    conversation, registry, booted = build_agent_runtime(
        AgentRuntimeConfig(),
        backend=backend,
    )
    assert booted is backend
    assert registry.names() == []
    conversation.submit_user_message("list the folder")
    result = await conversation.run()
    # No tools → the model's text reply is the answer. Before the Prompt-8
    # fix, a text-only turn was ALWAYS "intermediate", so this run dead-
    # looped on identical canned replies until max_steps (the exact stub
    # symptom); now the second consecutive text reply finishes cleanly.
    assert result.terminated_by == "finish"
    assert result.answer and result.answer.strip()
    assert backend.calls[0]["tools"] in (None, [])


async def test_build_agent_runtime_registers_default_tools_with_roots(tmp_path):
    conversation, registry, backend = build_agent_runtime(
        AgentRuntimeConfig(workspace_roots=[str(tmp_path)]),
        backend=FakeBackend(),
    )
    names = registry.names()
    # Loop-guard contract (Prompt 9): query_experts is only registered with
    # an expert selector; the filesystem toolset is always available.
    assert "query_experts" not in names, names
    for expected in ("run_command", "read_file", "write_file", "search_text"):
        assert expected in names, names

    conversation.submit_user_message("do something")
    await conversation.run()
    offered = _tool_names(backend.calls[0]["tools"])
    assert "run_command" in offered


# ── The final execution gap: real subprocess, real output ────────────────────

async def test_run_command_executes_real_process_end_to_end(tmp_path):
    """A scripted model requests run_command; the log records the REAL
    subprocess output (marker echoed by the shell), never stub text."""
    marker = f"P6_MARKER_{uuid.uuid4().hex[:8]}"
    script = [
        tool_call_response("run_command", {"command": f"echo {marker}"}),
    ]
    backend = FakeBackend(script)
    conversation, registry, _ = build_agent_runtime(
        AgentRuntimeConfig(workspace_roots=[str(tmp_path)]),
        backend=backend,
    )
    assert registry.has("run_command")

    conversation.submit_user_message(f"echo {marker} and report")
    result = await conversation.run()

    kinds = [ev.kind for ev in result.events]
    assert kinds == [
        EventKind.USER_MESSAGE, EventKind.ACTION, EventKind.OBSERVATION,
        EventKind.ACTION,
    ]

    action_ev = result.events[1]
    assert action_ev.payload.kind == "tool_call"
    assert action_ev.payload.name == "run_command"
    assert action_ev.payload.arguments["command"] == f"echo {marker}"

    obs_ev = result.events[2]
    assert obs_ev.payload.kind == "tool_result"
    content = obs_ev.payload.content
    assert marker in content          # real stdout, not "[stub: ...]"
    assert "[stub:" not in content
    assert '"exit_code": 0' in content
    assert '"ok": true' in content

    finish_ev = result.events[3]
    assert finish_ev.payload.kind == "finish"
    assert result.terminated_by == "finish"


async def test_run_command_sees_workspace_cwd(tmp_path):
    """The command's cwd is resolved inside the workspace root: the result
    reports the workspace path as the process cwd."""
    marker = f"P6_CWD_{uuid.uuid4().hex[:8]}"
    script = [
        tool_call_response("run_command", {"command": f"echo {marker}"}),
    ]
    conversation = _boot([str(tmp_path)], script=script)

    conversation.submit_user_message("run in the workspace")
    result = await conversation.run()

    content = result.events[2].payload.content
    assert marker in content
    assert f'"cwd": {json.dumps(str(tmp_path))}' in content
    assert result.terminated_by == "finish"


async def test_denied_command_becomes_error_observation(tmp_path):
    """Safety: a deny-listed command never executes; the loop records a
    clean [denied_command] ErrorObservation and keeps going."""
    script = [
        tool_call_response("run_command", {"command": "shutdown -s -t 0"}),
    ]
    conversation = _boot([str(tmp_path)], script=script)

    conversation.submit_user_message("shut the machine down")
    result = await conversation.run()

    obs_ev = result.events[2]
    assert obs_ev.payload.kind == "error"
    assert "denied_command" in obs_ev.payload.message
    assert result.terminated_by == "finish"


async def test_unknown_tool_still_errors_cleanly(tmp_path):
    script = [
        tool_call_response("not_a_real_tool", {"x": 1}),
    ]
    conversation = _boot([str(tmp_path)], script=script)

    conversation.submit_user_message("use the mystery tool")
    result = await conversation.run()

    obs_ev = result.events[2]
    assert obs_ev.payload.kind == "error"
    assert "unknown tool 'not_a_real_tool'" in obs_ev.payload.message
    assert result.terminated_by == "finish"


# ── Wire: the SSE stream reflects real execution results ─────────────────────

async def test_tool_result_wire_carries_real_output(tmp_path):
    marker = f"P6_WIRE_{uuid.uuid4().hex[:8]}"
    script = [
        tool_call_response("run_command", {"command": f"echo {marker}"}),
    ]
    conversation = _boot([str(tmp_path)], script=script)

    conversation.submit_user_message("run it")
    result = await conversation.run()

    wire = event_to_wire(result.events[2], step_records=None)
    assert wire["kind"] == "observation"
    assert wire["event_type"] == "tool_result"
    assert marker in wire["payload"]["content"]
    assert '"exit_code": 0' in wire["payload"]["content"]
