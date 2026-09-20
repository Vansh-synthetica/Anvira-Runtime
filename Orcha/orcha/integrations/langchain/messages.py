"""
orcha.integrations.langchain.messages
=====================================
Conversion between Orcha's message dicts and LangChain ``BaseMessage``.

Orcha's message shape is already OpenAI-compatible (``{"role": "user",
"content": "..."}`` with content as a string OR a list of content
blocks, exactly the shape ``orcha.agent_runtime.backend.render_content``
understands). LangChain's own converters also target the OpenAI shape,
so this adapter is intentionally minimal and version-stable: it is
hand-rolled rather than relying on ``langchain_core.adapters``.

Round-trip contract
-------------------
- ``assistant`` tool calls: Orcha ``arguments`` (parsed dict) ↔ LangChain
  ``args``. Tool-call IDs are preserved.
- ``tool`` messages carry ``tool_call_id`` on both sides.
- Content blocks (list of ``{"type": "text", ...}`` dicts) pass through
  unchanged on both sides.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..base import IntegrationUnavailable

__all__ = [
    "orcha_message_to_langchain", "langchain_message_to_orcha",
    "orcha_messages_to_langchain",
]


def _lc():
    try:
        from langchain_core.messages import (
            AIMessage, HumanMessage, SystemMessage, ToolMessage,
        )
        return AIMessage, HumanMessage, SystemMessage, ToolMessage
    except ImportError:
        raise IntegrationUnavailable(
            "LangChain is not installed. Install with "
            "`pip install \"orcha[lang]\"`."
        ) from None


def orcha_message_to_langchain(message: Dict[str, Any]):
    """
    Convert one Orcha message dict to a LangChain BaseMessage.
    """
    AIMessage, HumanMessage, SystemMessage, ToolMessage = _lc()
    role = message.get("role", "user")
    content = message.get("content", "")

    if role == "system":
        return SystemMessage(content=content)
    if role == "assistant":
        tool_calls = message.get("tool_calls")
        if tool_calls:
            lc_calls = [
                {
                    "type": "tool_call",
                    "id": str(tc.get("id") or ""),
                    "name": str(tc.get("name") or ""),
                    "args": tc.get("arguments", {}),
                }
                for tc in tool_calls
            ]
            return AIMessage(content=content, tool_calls=lc_calls)
        return AIMessage(content=content)
    if role == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=str(message.get("tool_call_id") or ""),
        )
    return HumanMessage(content=content)


def langchain_message_to_orcha(message) -> Dict[str, Any]:
    """
    Convert one LangChain BaseMessage to an Orcha message dict.
    """
    role_map = {
        "system": "system", "human": "user", "ai": "assistant",
        "tool": "tool",
    }
    out: Dict[str, Any] = {
        "role": role_map.get(getattr(message, "type", ""), "user"),
        "content": message.content,
    }
    tool_calls = getattr(message, "tool_calls", None)
    if out["role"] == "assistant" and tool_calls:
        out["tool_calls"] = [
            {
                "id": str(tc.get("id") or ""),
                "name": str(tc.get("name") or ""),
                "arguments": tc.get("args", {}),
            }
            for tc in tool_calls
        ]
    if out["role"] == "tool":
        out["tool_call_id"] = str(getattr(message, "tool_call_id", "") or "")
    return out


def orcha_messages_to_langchain(messages: Sequence[Dict[str, Any]]) -> List:
    """Convert a conversation (system first) to LangChain messages."""
    return [orcha_message_to_langchain(m) for m in messages]
