"""
orcha.agent_runtime.backend
===========================
The model-access layer of the AgentRuntime — backend-agnostic by design.

A ``ModelBackend`` turns an ordered list of messages plus tool schemas into
a normalized ``ModelResponse``: free-text content, zero or more tool calls
(already parsed into structured args), and token usage split into prompt /
completion / reasoning. Backends differ in transport (OpenAI-compatible
HTTP, native SDK, TGI, …) but all speak this one protocol, so the agent,
workspace and conversation never know which backend produced an answer.

Rules:
- ``complete`` receives an explicit generation config every call; backends
  may ignore or reject optional fields (temperature, top_p, …).
- Token usage is *optional*: a backend that reports nothing returns a
  ``TokenUsage`` with zeros — the runtime treats missing usage exactly like
  zero, never as an error.
- Tool calls are handed to the agent *already parsed*; the agent validates
  them against its ToolRegistry and triggers a single repair round when
  parsing failed.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

# ── Protocol types ────────────────────────────────────────────────────────────

# A single chat message. System prompts are carried as ordinary messages.
ModelMessage = Dict[str, Any]


@dataclass(frozen=True)
class TokenUsage:
    """Token accounting for one model response. Zero = backend reported none."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def step_tokens(self) -> int:
        """Tokens charged against the run budget for this step."""
        return self.total_tokens + self.reasoning_tokens

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TokenUsage":
        if not data:
            return cls()
        return cls(
            prompt_tokens=int(data.get("prompt_tokens") or 0),
            completion_tokens=int(data.get("completion_tokens") or 0),
            reasoning_tokens=int(data.get("reasoning_tokens") or 0),
        )


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation requested by the model, arguments already parsed."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class GenerationConfig:
    """
    Per-call generation options. Everything is optional; a backend is free
    to ignore (or reject) any field it does not support.

    Streaming policy (conservative by default — tool-call JSON must stay
    intact, so tool rounds are never streamed unless the backend declares
    it can stream them):

    - ``stream``              — stream free-text rounds (no tools offered).
    - ``stream_tool_calls``   — stream rounds where tools are offered.
      Only honored when the backend reports ``supports_streaming_tool_calls``;
      otherwise tool rounds silently run non-streaming.
    - ``should_stream``       — resolves the effective decision for a round
      given the backend's capability flag.
    """
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Sequence[str]] = None
    stream: bool = False
    stream_tool_calls: bool = False
    # Anti-repetition penalties (applied when non-None; ignored by
    # backends that don't support them).  frequency_penalty penalises
    # tokens proportional to how often they've appeared; presence_penalty
    # penalises any token that has appeared at least once.
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def should_stream(self, round_requests_tools: bool, backend_streams_tools: bool) -> bool:
        """
        Effective streaming decision for a round.

        ``round_requests_tools`` is True when the model is being offered
        tool schemas this call. Tool rounds stream only when the backend
        declares it can stream tool calls; otherwise they run
        non-streaming regardless of the config.
        """
        if round_requests_tools:
            if not backend_streams_tools:
                return False
            return bool(self.stream_tool_calls)
        return bool(self.stream)

    def to_api(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.temperature is not None:
            out["temperature"] = self.temperature
        if self.top_p is not None:
            out["top_p"] = self.top_p
        if self.max_tokens is not None:
            out["max_tokens"] = self.max_tokens
        if self.stop:
            out["stop"] = list(self.stop)
        if self.frequency_penalty is not None:
            out["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            out["presence_penalty"] = self.presence_penalty
        return out


@dataclass(frozen=True)
class ModelResponse:
    """
    Normalized answer from a backend.

    ``content``   — free text (may be empty when only tool calls were made).
    ``tool_calls``— parsed tool calls (may be empty).
    ``usage``     — token accounting (zeros when the backend reports none).
    ``finish_reason`` — "stop" | "tool_calls" | "length" | "error" | None.
    """
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: Optional[str] = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class ToolCallParseError(Exception):
    """
    Raised by a backend when the model emitted a tool call it could not
    parse (broken JSON arguments, missing fields). The agent catches this,
    re-prompts the model once with the parse error, and only then fails
    the step cleanly.
    """

    def __init__(self, message: str, raw: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.raw = raw


# ── The backend contract ──────────────────────────────────────────────────────

class ModelBackend(ABC):
    """
    The single seam between the AgentRuntime and a model.

    Implementations must be *stateless with respect to the conversation*:
    ``complete`` is given everything it needs (messages, tool schemas,
    config) and returns everything the runtime needs. Per-backend
    differences (base URL, model name, capability flags) live on the
    backend's own config object — never in the agent.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier (logging, metadata)."""

    @property
    @abstractmethod
    def supports_streaming_tool_calls(self) -> bool:
        """Whether tool-call request turns may be streamed (default: False
        — the runtime's conservative policy streams only final text turns
        unless this is True)."""

    @property
    @abstractmethod
    def reports_reasoning_tokens(self) -> bool:
        """Whether the backend returns reasoning-token usage (e.g. DeepSeek
        reasoner). The runtime folds whatever it reports into the same
        budget; absence simply means zero."""

    @abstractmethod
    def complete(
        self,
        messages: Sequence[ModelMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[GenerationConfig] = None,
    ) -> ModelResponse:
        """
        Run one model call.

        Parameters
        ----------
        messages  Full conversation so far (system first), OpenAI-compatible
                  message dicts ({"role": ..., "content": ...}). Tool results
                  are appended as ordinary assistant/tool messages by the
                  agent, so backends need no special tool-message handling.
        tools     OpenAI-compatible tool schemas, or None for free-text only.
        config    Per-call generation options (temperature, streaming…).

        Returns
        -------
        A normalized ModelResponse. May raise ToolCallParseError (see above)
        or a backend-specific transport error; the agent loop isolates both.
        """
        raise NotImplementedError

    def render_content(self, content: Any) -> str:
        """Render a content value to text (list-of-blocks, string, …)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif "text" in block:
                        parts.append(str(block["text"]))
                else:
                    parts.append(str(block))
            return "".join(parts)
        return str(content) if content is not None else ""

    def render_tool_calls_for_repair(
        self, error: ToolCallParseError, raw: Optional[str] = None,
    ) -> str:
        """Message appended to the next model call during a repair round."""
        return (
            "One of your tool calls was malformed and could not be executed. "
            f"Parse error: {error.message}. "
            "Fix the JSON arguments and re-issue the tool call(s) — reply "
            "with either corrected tool_calls or a plain text answer. "
            "Do not repeat the broken call verbatim."
        )


__all__ = [
    "ModelMessage", "TokenUsage", "ToolCall", "GenerationConfig",
    "ModelResponse", "ToolCallParseError", "ModelBackend",
]
