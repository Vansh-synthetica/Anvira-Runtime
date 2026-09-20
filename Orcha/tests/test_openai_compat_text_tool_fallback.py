"""
Regression test for the local/edge-model tool-call fallback.

Small models served over an OpenAI-compatible endpoint (llama.cpp, older
Ollama builds, ...) frequently ignore the `tools` field entirely and just
narrate a tool call in plain text instead of returning structured
`message.tool_calls`. Before this fix that narration was treated as a
final/plain message and nothing ever executed, while the UI still showed
it as if the agent were taking real action. `_response_from_delta` must
recover a real ToolCall from that text when it matches the shape the
agent's text-mode directive asks the model to use.
"""
from orcha.agent_runtime.backends.openai_compat import _response_from_delta

TOOLS = [
    {"type": "function", "function": {"name": "write_file", "parameters": {}}},
    {"type": "function", "function": {"name": "run_command", "parameters": {}}},
]


def _choice(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}


def test_native_tool_calls_pass_through_unchanged():
    data = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "write_file", "arguments": '{"path": "a.py", "content": "x"}'},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }
    response = _response_from_delta(data, TOOLS)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "write_file"
    assert response.content == ""


def test_bare_json_text_tool_call_is_recovered():
    content = '{"name": "run_command", "arguments": {"command": "npm test"}}'
    response = _response_from_delta(_choice(content), TOOLS)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "run_command"
    assert response.tool_calls[0].arguments == {"command": "npm test"}
    assert response.content == ""


def test_fenced_json_text_tool_call_is_recovered():
    content = (
        "Sure, I'll run the tests now.\n\n"
        "```json\n"
        '{"name": "run_command", "arguments": {"command": "pytest"}}\n'
        "```\n"
    )
    response = _response_from_delta(_choice(content), TOOLS)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "run_command"
    assert response.tool_calls[0].arguments == {"command": "pytest"}


def test_plain_prose_is_left_as_a_message_not_misfired():
    content = "I looked at the file and it seems fine, no changes needed."
    response = _response_from_delta(_choice(content), TOOLS)
    assert response.tool_calls == []
    assert response.content == content


def test_unknown_tool_name_is_not_treated_as_a_call():
    content = '{"name": "delete_everything", "arguments": {}}'
    response = _response_from_delta(_choice(content), TOOLS)
    assert response.tool_calls == []
    assert response.content == content


def test_no_tools_offered_never_triggers_fallback():
    content = '{"name": "run_command", "arguments": {"command": "pytest"}}'
    response = _response_from_delta(_choice(content), None)
    assert response.tool_calls == []
    assert response.content == content
