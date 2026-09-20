"""
orcha.integrations.langchain.backend
====================================
A LangChain-powered ``ModelBackend`` for the Orcha AgentRuntime.

``LangChainModelBackend`` wraps ANY LangChain chat model — OpenAI-compatible
localhost endpoints (llama.cpp, LM Studio, vLLM), Ollama, and optional cloud
providers — behind Orcha's own ``ModelBackend.complete`` contract. The
AgentRuntime, Conversation, and event system are untouched: they keep
speaking Orcha's message dicts and receive a normalized ``ModelResponse``.

Orcha remains the controller: the backend carries no conversation state,
per-call ``GenerationConfig`` still controls temperature/top_p/max_tokens/
stop (bound per call), and tool schemas are Orcha's OpenAI-compatible
dicts (``bind_tools`` accepts them verbatim).

Limitations (current)
---------------------
- ``complete`` is a single non-streamed call. LangChain supports token
  streaming, but the Orcha backend contract is one-shot; streaming stays
  at the AgentRuntime event layer. A future streaming extension can be
  added without changing the contract.
- Tool-call arguments arrive already parsed by LangChain; a malformed
  call raises ``ToolCallParseError`` so the AgentRuntime's single repair
  round fires exactly as with native backends.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from ...agent_runtime.backend import (
    GenerationConfig, ModelBackend, ModelMessage, ModelResponse, TokenUsage,
    ToolCall, ToolCallParseError,
)
from ..base import IntegrationUnavailable
from .messages import orcha_messages_to_langchain

__all__ = ["LangChainModelBackend"]


class LangChainModelBackend(ModelBackend):
    """
    Adapt any LangChain chat model to the Orcha ``ModelBackend`` contract.

    Parameters
    ----------
    model                      The LangChain chat model instance.
    name                       Optional human-readable model identifier
                               (defaults to the model's ``model_name``).
    reports_reasoning_tokens   Whether the model reports reasoning usage.
    """

    def __init__(
        self,
        model: Any,
        name: Optional[str] = None,
        reports_reasoning_tokens: bool = False,
    ) -> None:
        try:
            from langchain_core.language_models import BaseChatModel
        except ImportError:
            raise IntegrationUnavailable(
                "LangChain is not installed. Install with "
                "`pip install \"orcha[lang]\"`."
            ) from None
        if not isinstance(model, BaseChatModel):
            raise TypeError(
                f"expected a langchain_core BaseChatModel, got {type(model).__name__}"
            )
        self._model = model
        self._name = name or getattr(model, "model_name", None) or type(model).__name__
        self._reports_reasoning_tokens = reports_reasoning_tokens

    # ── Capability flags ───────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def supports_streaming_tool_calls(self) -> bool:
        return False

    @property
    def reports_reasoning_tokens(self) -> bool:
        return self._reports_reasoning_tokens

    # ── The one call ───────────────────────────────────────────────────

    def complete(
        self,
        messages: Sequence[ModelMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[GenerationConfig] = None,
    ) -> ModelResponse:
        config = config or GenerationConfig()
        lc_messages = orcha_messages_to_langchain(messages)

        bound = self._model
        if tools and self._can_bind_tools(self._model):
            bound = bound.bind_tools(tools)
        api_kwargs = config.to_api()
        if api_kwargs:
            bound = bound.bind(**api_kwargs)

        response = bound.invoke(lc_messages)

        content = self.render_content(getattr(response, "content", ""))
        tool_calls: List[ToolCall] = []
        for tc in response.tool_calls or []:
            args = tc.get("args")
            if not isinstance(args, dict):
                raise ToolCallParseError(
                    "malformed tool-call arguments", raw=json.dumps(tc, default=str)
                )
            tool_calls.append(ToolCall(
                id=str(tc.get("id") or ""),
                name=str(tc.get("name") or ""),
                arguments=args,
            ))

        usage = self._usage_from(response)
        meta = getattr(response, "response_metadata", {}) or {}
        finish = meta.get("finish_reason") or ("tool_calls" if tool_calls else "stop")

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=str(finish) if finish is not None else None,
        )

    @staticmethod
    def _usage_from(response) -> TokenUsage:
        meta = getattr(response, "usage_metadata", None) or {}
        if not isinstance(meta, dict):
            return TokenUsage()
        return TokenUsage(
            prompt_tokens=int(meta.get("input_tokens") or meta.get("prompt_tokens") or 0),
            completion_tokens=int(
                meta.get("output_tokens") or meta.get("completion_tokens") or 0
            ),
            reasoning_tokens=int(meta.get("reasoning_tokens") or 0),
        )

    @staticmethod
    def _can_bind_tools(model: Any) -> bool:
        """
        Only bind tool schemas when the concrete model implements
        ``bind_tools``. The langchain base class raises NotImplementedError;
        models without tool support simply get no schemas.
        """
        try:
            from langchain_core.language_models import BaseChatModel
        except ImportError:  # pragma: no cover — guarded at construction
            return False
        method = getattr(type(model), "bind_tools", None)
        return method is not None and method is not BaseChatModel.bind_tools
