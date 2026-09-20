"""
Stage 2 tests: LangChain adapter layer (orcha.integrations.langchain).

Verifies the component boundary with langchain-core's FakeMessagesListChatModel
— no network, no API keys. Execution stays in Orcha's permission path in
both tool directions; the backend contract stays Orcha's ModelBackend.
"""
import pytest

from orcha.agent_runtime.backend import GenerationConfig, ModelResponse, TokenUsage
from orcha.agent_runtime.tools import Tool, ToolRegistry
from orcha.capabilities.base import READ, SAFE, spec
from orcha.integrations import LangChainBoundary
from orcha.integrations.langchain import (
    LangChainModelBackend,
    langchain_message_to_orcha, langchain_tool_to_orcha,
    orcha_message_to_langchain, orcha_tool_to_langchain,
)

pytestmark = pytest.mark.skipif(
    not LangChainBoundary().status().available,
    reason="orcha[lang] extra not installed",
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_weather_tool() -> Tool:
    def get_weather(city: str) -> str:
        return f"Weather in {city}: 22C, clear"
    return Tool(spec(
        "get_weather", "Get the current weather for a city.",
        {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
        }, "required": ["city"]},
        get_weather,
        permissions=[READ], safety_level=SAFE, capability="demo",
        result_format={"type": "object"},
    ))


# ── Message conversion ───────────────────────────────────────────────────────

def test_text_message_round_trip():
    msg = {"role": "user", "content": "hello"}
    lc = orcha_message_to_langchain(msg)
    assert lc.type == "human"
    assert lc.content == "hello"
    assert langchain_message_to_orcha(lc) == msg

def test_system_message_round_trip():
    msg = {"role": "system", "content": "be concise"}
    lc = orcha_message_to_langchain(msg)
    assert lc.type == "system"
    assert langchain_message_to_orcha(lc) == msg

def test_assistant_tool_call_round_trip():
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_1", "name": "get_weather", "arguments": {"city": "Paris"}},
        ],
    }
    lc = orcha_message_to_langchain(msg)
    assert lc.type == "ai"
    assert lc.tool_calls[0]["args"] == {"city": "Paris"}
    back = langchain_message_to_orcha(lc)
    assert back["tool_calls"][0]["arguments"] == {"city": "Paris"}
    assert back["tool_calls"][0]["id"] == "call_1"

def test_tool_message_round_trip():
    msg = {"role": "tool", "content": "22C clear", "tool_call_id": "call_1"}
    lc = orcha_message_to_langchain(msg)
    assert lc.type == "tool"
    assert lc.tool_call_id == "call_1"
    assert langchain_message_to_orcha(lc) == msg

def test_content_blocks_pass_through():
    blocks = [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]
    msg = {"role": "user", "content": blocks}
    lc = orcha_message_to_langchain(msg)
    assert lc.content == blocks


def _json_schema(lc_tool) -> dict:
    """args_schema is a plain dict in langchain v1, a pydantic model in 0.3.x."""
    schema = lc_tool.args_schema
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    return schema


# ── Tool conversion ──────────────────────────────────────────────────────────

def test_orcha_tool_to_langchain_invokes_through_orcha():
    lc_tool = orcha_tool_to_langchain(_get_weather_tool())
    assert lc_tool.name == "get_weather"
    assert "city" in _json_schema(lc_tool)["properties"]
    result = lc_tool.invoke({"city": "Paris"})
    assert result == "Weather in Paris: 22C, clear"

def test_orcha_tool_schema_parity():
    orcha_tool = _get_weather_tool()
    lc_tool = orcha_tool_to_langchain(orcha_tool)
    schema = _json_schema(lc_tool)
    assert schema["required"] == ["city"]
    assert lc_tool.description == orcha_tool.description

def test_langchain_tool_to_orcha_registers_and_runs():
    from langchain_core.tools import StructuredTool
    lc_tool = StructuredTool.from_function(
        func=lambda a, b: a + b, name="add", description="Add two numbers",
        args_schema={"type": "object", "properties": {
            "a": {"type": "integer"}, "b": {"type": "integer"},
        }, "required": ["a", "b"]},
    )
    tool_spec = langchain_tool_to_orcha(lc_tool)
    registry = ToolRegistry()
    registry.register_spec(tool_spec)
    tool = registry.get("add")
    assert tool is not None
    assert tool.parameters["required"] == ["a", "b"]
    observation = tool.run({"a": 2, "b": 3})
    assert observation.success is True
    assert "5" in observation.content


# ── LangChainModelBackend ───────────────────────────────────────────────────

def _fake_backend(responses):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    model = FakeMessagesListChatModel(responses=responses)
    return LangChainModelBackend(model, name="fake-model")


def test_backend_text_response():
    from langchain_core.messages import AIMessage
    backend = _fake_backend([AIMessage(content="Hello from the model")])
    resp = backend.complete([{"role": "user", "content": "hi"}])
    assert isinstance(resp, ModelResponse)
    assert resp.content == "Hello from the model"
    assert resp.finish_reason == "stop"
    assert resp.tool_calls == []
    assert backend.model_name == "fake-model"

def test_backend_tool_call_response():
    from langchain_core.messages import AIMessage
    backend = _fake_backend([AIMessage(
        content="",
        tool_calls=[{
            "name": "get_weather", "args": {"city": "Paris"},
            "id": "call_9", "type": "tool_call",
        }],
        usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    )])
    resp = backend.complete(
        [{"role": "user", "content": "weather?"}],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )
    assert resp.has_tool_calls
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].arguments == {"city": "Paris"}
    assert resp.tool_calls[0].id == "call_9"
    assert resp.finish_reason == "tool_calls"
    assert resp.usage == TokenUsage(prompt_tokens=5, completion_tokens=3)

def test_backend_honors_generation_config():
    from langchain_core.messages import AIMessage
    backend = _fake_backend([AIMessage(content="x")])
    resp = backend.complete(
        [{"role": "user", "content": "hi"}],
        config=GenerationConfig(temperature=0.7, max_tokens=64, stop=["END"]),
    )
    assert resp.content == "x"

def test_backend_accepts_full_tool_cycle_messages():
    from langchain_core.messages import AIMessage
    backend = _fake_backend([AIMessage(content="done")])
    messages = [
        {"role": "system", "content": "you are a weather bot"},
        {"role": "user", "content": "weather in Paris?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_1", "name": "get_weather",
                         "arguments": {"city": "Paris"}}]},
        {"role": "tool", "content": "22C clear", "tool_call_id": "call_1"},
    ]
    resp = backend.complete(messages)
    assert resp.content == "done"


# ── Boundary status ──────────────────────────────────────────────────────────

def test_boundary_now_available():
    assert LangChainBoundary().status().available is True
    assert LangChainBoundary().status().version
