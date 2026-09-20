"""
Tests for the forced-tool envelope (orcha/api/server.py's
_AGENT_TOOL_ENVELOPE_SCHEMA_FORCED / _AGENT_ENVELOPE_INSTRUCTION_FORCED,
triggered by orcha/nodes/task_executor.py's FORCE_TOOL_CALL_MARKER).

Context: live testing confirmed that the normal envelope's "action":
"final" option lets a small/quantized local model choose to describe an
answer in prose instead of calling a tool, even under schema-constrained
decoding (so the JSON itself was always valid) and even after an explicit
corrective retry telling it to call the tool. The forced variant removes
"final" from the schema's enum entirely, so a task-execution turn that
hasn't made a real tool call yet gets a schema the model is structurally
unable to satisfy with anything other than a real tool call.

These tests exercise the schema shape and the envelope decoder directly —
not a live model — since that's the part this fix actually changes:
_apply_agent_envelope already round-trips a `{"action": "tool", ...}`
message correctly (it never depended on "answer" being present), the
forced schema just removes the other option a model could pick.
"""
from orcha.api.server import (
    _AGENT_TOOL_ENVELOPE_SCHEMA,
    _AGENT_TOOL_ENVELOPE_SCHEMA_FORCED,
    _AGENT_ENVELOPE_INSTRUCTION_FORCED,
    _apply_agent_envelope,
)
from orcha.nodes.task_executor import FORCE_TOOL_CALL_MARKER


def test_forced_schema_has_no_final_option():
    """The whole point of the forced variant: "final" must not be a
    reachable value under strict json_schema decoding."""
    action_enum = _AGENT_TOOL_ENVELOPE_SCHEMA_FORCED["properties"]["action"]["enum"]
    assert action_enum == ["tool"]
    assert "final" not in action_enum


def test_forced_schema_drops_answer_field():
    """No "answer" property at all — a model constrained to this schema
    cannot smuggle a text explanation into a field the normal envelope
    would have accepted, even alongside a nominal tool call."""
    assert "answer" not in _AGENT_TOOL_ENVELOPE_SCHEMA_FORCED["properties"]
    assert "answer" not in _AGENT_TOOL_ENVELOPE_SCHEMA_FORCED["required"]


def test_normal_schema_unchanged_for_backward_compatibility():
    """Every other call site (single-turn AgentNode chat, etc.) must keep
    getting the original schema — only task_executor.py turns that append
    FORCE_TOOL_CALL_MARKER should ever see the forced variant."""
    action_enum = _AGENT_TOOL_ENVELOPE_SCHEMA["properties"]["action"]["enum"]
    assert set(action_enum) == {"tool", "final"}


def test_forced_instruction_explicitly_forbids_describing():
    instruction = _AGENT_ENVELOPE_INSTRUCTION_FORCED
    assert "MUST call a tool" in instruction
    assert "Do not explain, describe" in instruction


def test_apply_agent_envelope_decodes_forced_shape_response():
    """A model correctly following the forced schema returns
    {"action":"tool","name":...,"arguments":...} with no "answer" key at
    all — _apply_agent_envelope must still decode that into a proper
    tool_calls message (it was written for the 4-key shape, but only
    ever reads "action"/"name"/"arguments", never "answer")."""
    raw = '{"action": "tool", "name": "write_file", "arguments": "{\\"path\\": \\"snake.html\\", \\"content\\": \\"<html></html>\\"}"}'
    result = _apply_agent_envelope({"content": raw, "finish_reason": "stop"}, "test-model")
    assert result["tool_calls"][0]["function"]["name"] == "write_file"
    import json
    args = json.loads(result["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == "snake.html"


def test_marker_is_a_distinct_recognizable_suffix():
    """Sanity check on the contract server.py's completion_fn relies on:
    the marker must be a suffix so `system.endswith(...)` detection and
    `system[:-len(marker)]` stripping (see _build_agent_completion_fn)
    round-trip cleanly without mangling a real system prompt that
    happens to end in similar text."""
    system_prompt = "You are executing a task." + FORCE_TOOL_CALL_MARKER
    assert system_prompt.endswith(FORCE_TOOL_CALL_MARKER)
    stripped = system_prompt[: -len(FORCE_TOOL_CALL_MARKER)]
    assert stripped == "You are executing a task."
    assert FORCE_TOOL_CALL_MARKER not in stripped
