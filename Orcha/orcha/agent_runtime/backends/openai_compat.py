"""
orcha.agent_runtime.backends.openai_compat
==========================================
The shipped ModelBackend for the OpenAI-compatible chat surface — the one
shape shared by Ollama, llama.cpp server and LM Studio. All three speak
POST /v1/chat/completions with nearly identical JSON; what differs per
vendor is the base URL and model-name convention, which is exactly what
``OpenAICompatBackendConfig`` carries (no subclassing per vendor):

- Ollama    → base_url http://localhost:11434/v1, model like "llama3.2:3b"
- llama.cpp → base_url http://localhost:8080/v1, model like "qwen3:8b"
- LM Studio → base_url http://localhost:1234/v1, model like "llama-3.2-3b"

The backend is deliberately thin: it builds the request, streams SSE when
the runtime's conservative policy allows it, and parses the response into
a normalized ``ModelResponse`` (including OpenAI-style tool_calls with
``id``/``function.name``/``function.arguments`` — a bare string arguments
payload is the OpenAI wire shape and is parsed as JSON here).
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import httpx

from ..backend import (
    GenerationConfig, ModelBackend, ModelMessage, ModelResponse,
    TokenUsage, ToolCall, ToolCallParseError,
)
from ..diagnostics import diag_log, diag_log_exc

logger = logging.getLogger("orcha.agent_runtime.backends.openai_compat")

DEFAULT_OLLAMA_V1      = "http://localhost:11434/v1"
DEFAULT_LLAMACPP_V1    = "http://localhost:8080/v1"
DEFAULT_LMSTUDIO_V1    = "http://localhost:1234/v1"

_TIMEOUT_S = 600.0

# SSE lines that carry a whole JSON chunk: "data: {...}" — everything else
# ("event:", "[DONE]", blank) is skipped.
_SSE_DATA = re.compile(r"^data:\s*(.+)$")


@dataclass
class OpenAICompatBackendConfig:
    """
    Everything that differs between vendors/instances. ``base_url`` should
    point at the /v1 root of the server (e.g. ``http://localhost:11434/v1``
    for Ollama); ``model`` is the model name the server expects
    (``llama3.2:3b`` for Ollama, plain tags for llama.cpp/LM Studio).
    """
    base_url: str = DEFAULT_OLLAMA_V1
    model: str = "llama3.2:3b"
    api_key: Optional[str] = None
    # Capability flags — conservative defaults for tool-call streaming.
    supports_streaming_tool_calls: bool = False
    reports_reasoning_tokens: bool = False
    # Transport tuning.
    timeout_s: float = _TIMEOUT_S
    headers: Dict[str, str] = field(default_factory=dict)
    # ── Inference acceleration (server-side hints) ────────────────────
    speculative_model: Optional[str] = None
    use_prefix_cache: bool = False
    kv_cache_quant: Optional[str] = None


class OpenAICompatBackend(ModelBackend):
    """
    ``ModelBackend`` over POST /v1/chat/completions via httpx.

    Stateless: every call is self-contained. Streaming follows the
    runtime's conservative policy resolved by the agent
    (``GenerationConfig.should_stream``); the final chunk's
    ``usage``/``finish_reason`` is used when present.
    """

    def __init__(
        self, config: Optional[OpenAICompatBackendConfig] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.config = config or OpenAICompatBackendConfig()
        self._injected_client = client  # optional injected transport (tests)
        # Lazily-created pooled client — avoids per-call TCP/TLS handshake.
        self._client: Optional[httpx.Client] = client
        self._owns_client = client is None

    # ── Backend contract ───────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self.config.model

    @property
    def supports_streaming_tool_calls(self) -> bool:
        return self.config.supports_streaming_tool_calls

    @property
    def reports_reasoning_tokens(self) -> bool:
        return self.config.reports_reasoning_tokens

    # ── Lifecycle ──────────────────────────────────────────────────────

    def close(self) -> None:
        """Release the pooled HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    # ── Implementation ─────────────────────────────────────────────────

    def complete(
        self,
        messages: Sequence[ModelMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[GenerationConfig] = None,
    ) -> ModelResponse:
        gen = config or GenerationConfig()
        stream = gen.should_stream(
            round_requests_tools=bool(tools),
            backend_streams_tools=self.supports_streaming_tool_calls,
        )
        t0 = time.perf_counter()
        diag_log(
            logger, "backend", "request",
            model=self.config.model,
            tools=",".join(
                (t.get("function") or {}).get("name", "?") for t in tools or []
            ) or None,
            stream=stream,
            temperature=gen.to_api().get("temperature"),
            messages=len(list(messages)),
        )

        body: Dict[str, Any] = {"model": self.config.model}
        body.update(gen.to_api())
        body["stream"] = stream
        body["messages"] = list(messages)
        if tools:
            body["tools"] = list(tools)
        # ── Inference acceleration hints (ignored by unsupported servers)
        if self.config.speculative_model:
            body["speculative_model"] = self.config.speculative_model
        if self.config.use_prefix_cache:
            body["prefix_cache"] = True
        if self.config.kv_cache_quant:
            body["kv_cache_quant"] = self.config.kv_cache_quant

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        headers.update(self.config.headers)

        client = self._client
        if client is None:
            # Connection pool tuning: keep-alive for persistent connections,
            # limit concurrent connections to avoid exhausting local resources.
            transport = httpx.HTTPTransport(
                retries=1,
                limits=httpx.Limits(
                    max_connections=8,
                    max_keepalive_connections=4,
                    keepalive_expiry=30,
                ),
            )
            client = httpx.Client(
                timeout=self.config.timeout_s,
                transport=transport,
            )
            self._client = client
            self._owns_client = False  # we manage lifecycle via close()
        try:
            if stream:
                response = self._complete_streaming(client, body, headers, tools)
            else:
                resp = client.post(
                    f"{self.config.base_url.rstrip('/')}/chat/completions",
                    json=body, headers=headers,
                )
                response = self._parse_response(resp.status_code, resp.text, tools)
                # ── Post-hoc repetition check for non-streaming ──────
                if response.content:
                    from ..conversation import _has_intra_message_repetition
                    if _has_intra_message_repetition(response.content):
                        diag_log(
                            logger, "backend", "repetition_detected",
                            level=logging.WARNING,
                            content_len=len(response.content),
                        )
                        # Keep the content so the conversation loop's
                        # repetition detector replaces it with a graceful
                        # message; just set the finish reason.
                        response = ModelResponse(
                            content=response.content,
                            tool_calls=[],
                            usage=response.usage,
                            finish_reason="repetition",
                        )
        except httpx.HTTPError as exc:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            diag_log_exc(
                logger, "backend", "request_failed", exc,
                latency_ms=latency_ms,
                status=getattr(getattr(exc, "response", None), "status_code", None),
            )
            raise RuntimeError(f"model request failed: {exc}") from exc
        except Exception as exc:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            diag_log_exc(logger, "backend", "parse_failed", exc, latency_ms=latency_ms)
            raise
        finally:
            # Pooled client is kept alive across calls — only injected
            # clients are closed per-call to preserve test semantics.
            if self._injected_client is not None and client is not None:
                client.close()

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        diag_log(
            logger, "backend", "response",
            latency_ms=latency_ms,
            content_len=len(response.content),
            tool_calls=json.dumps(
                [{"name": tc.name, "arguments": tc.arguments}
                 for tc in response.tool_calls],
                default=str, ensure_ascii=False,
            ) or None,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            reasoning_tokens=response.usage.reasoning_tokens,
            finish_reason=response.finish_reason,
        )
        return response

    # ── Streaming ──────────────────────────────────────────────────────

    def _complete_streaming(
        self,
        client: httpx.Client,
        body: Dict[str, Any],
        headers: Dict[str, str],
        tools: Optional[List[Dict[str, Any]]],
    ) -> ModelResponse:
        """Accumulate the SSE stream and parse the final delta as one
        response. Tool-call deltas concatenate their JSON fragments."""
        with client.stream(
            "POST",
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            json=body, headers=headers,
        ) as resp:
            if resp.status_code >= 400:
                return self._parse_response(resp.status_code, resp.read().decode("utf-8", "replace"))

            chunks: List[Dict[str, Any]] = []
            dropped = 0
            # ── Early stream termination for repetition loops ─────────
            # Small models can get stuck generating thousands of repeated
            # tokens.  We track the accumulated content and abort the
            # stream as soon as a degenerate repetition pattern appears.
            _acc_content = ""
            _check_interval = 20  # check every N chunks
            for line in resp.iter_lines():
                if not line:
                    continue
                match = _SSE_DATA.match(line.strip())
                if not match:
                    continue
                payload = match.group(1).strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    chunks.append(chunk)
                except json.JSONDecodeError:
                    dropped += 1
                    continue
                # Accumulate content for repetition checking.
                delta_obj = (chunk.get("choices") or [{}])[0].get("delta") or {}
                delta_text = delta_obj.get("content") or ""
                if delta_text:
                    _acc_content += delta_text
                # Check periodically (not every chunk for performance).
                if len(chunks) % _check_interval == 0 and len(_acc_content) > 100:
                    from ..conversation import _has_intra_message_repetition
                    if _has_intra_message_repetition(_acc_content):
                        diag_log(
                            logger, "backend", "stream_aborted_repetition",
                            level=logging.WARNING,
                            content_len=len(_acc_content),
                        )
                        break

            if dropped:
                diag_log(
                    logger, "backend", "stream_interruptions",
                    level=logging.WARNING, dropped_chunks=dropped,
                )

            delta = _merge_stream_chunks(chunks)
            return _response_from_delta(delta, tools)

    def _parse_response(
        self, status: int, text: str, tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ModelResponse:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"model returned non-JSON (status {status})"
            ) from exc

        if status >= 400:
            err = data.get("error", {})
            if isinstance(err, dict):
                raise RuntimeError(f"model API error {status}: {err.get('message', err)}")
            raise RuntimeError(f"model API error {status}: {data}")

        return _response_from_delta(data, tools)


# ── Response parsing (shared by streaming and non-streaming) ─────────────────

def _merge_stream_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fold a stream of chunk dicts into a single API-shaped response
    dict (choices → message), so the same parser handles streamed and
    non-streamed bodies: concatenated content, appended tool-call deltas,
    last usage/finish."""
    merged: Dict[str, Any] = {
        "choices": [{
            "message": {"role": "assistant", "content": "", "tool_calls": []},
            "finish_reason": None,
        }],
        "usage": None,
    }
    msg = merged["choices"][0]["message"]
    for chunk in chunks:
        if not chunk.get("choices"):
            if chunk.get("usage"):
                merged["usage"] = chunk["usage"]
            continue
        choice = chunk["choices"][0]
        delta = choice.get("delta") or {}
        if isinstance(delta.get("content"), str):
            msg["content"] += delta["content"]
        for tc in delta.get("tool_calls") or []:
            index = tc.get("index", 0)
            while len(msg["tool_calls"]) <= index:
                msg["tool_calls"].append({
                    "id": "", "function": {"name": "", "arguments": ""},
                })
            slot = msg["tool_calls"][index]
            fn = tc.get("function") or {}
            if tc.get("id"):
                slot["id"] = tc["id"]
            if fn.get("name"):
                slot["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                slot["function"]["arguments"] += fn["arguments"]
        if choice.get("finish_reason"):
            merged["choices"][0]["finish_reason"] = choice["finish_reason"]
    return merged


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _known_tool_names(tools: Optional[List[Dict[str, Any]]]) -> "set[str]":
    names = set()
    for t in tools or []:
        fn = (t or {}).get("function") or {}
        name = fn.get("name")
        if name:
            names.add(name)
    return names


def _extract_text_tool_call(
    content: str, tools: Optional[List[Dict[str, Any]]],
) -> Optional[ToolCall]:
    """
    Fallback for models that don't emit native ``tool_calls`` even when
    tool schemas were offered. Looks for the JSON shape the agent's
    text-mode directive asks for — ``{"name": "<tool>", "arguments": {}}``
    — bare or fenced in the reply. Returns None unless the parsed object
    names one of the tools actually on offer this round, so ordinary
    prose that happens to contain a JSON object is never misfired as a
    tool call.
    """
    known = _known_tool_names(tools)
    if not known or not content or not content.strip():
        return None

    candidates: List[str] = [m.group(1) for m in _FENCED_JSON_RE.finditer(content)]
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    if not candidates:
        match = _JSON_OBJECT_RE.search(content)
        if match:
            candidates.append(match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        name = parsed.get("name")
        arguments = parsed.get("arguments", {})
        if not isinstance(name, str) or name not in known:
            continue
        if not isinstance(arguments, dict):
            continue
        return ToolCall(id="", name=name, arguments=arguments)
    return None


def _response_from_delta(
    data: Dict[str, Any], tools: Optional[List[Dict[str, Any]]],
) -> ModelResponse:
    """Build a ModelResponse from a chat-completion body (streamed or not).
    ``tools`` is the schema list the request offered, used to fill
    ``finish_reason="tool_calls"`` when the server omits it."""
    usage = TokenUsage.from_dict(data.get("usage"))

    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}

    content = message.get("content") or ""
    if isinstance(content, list):  # multi-part content blocks
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        content = "".join(parts)

    finish_reason = choice.get("finish_reason")

    tool_calls: List[ToolCall] = []
    for raw in message.get("tool_calls") or []:
        tc_id = raw.get("id") or ""
        fn = raw.get("function") or {}
        name = fn.get("name") or ""
        arguments_raw = fn.get("arguments") or ""
        if isinstance(arguments_raw, dict):
            arguments: Any = arguments_raw
        elif isinstance(arguments_raw, str):
            if not arguments_raw.strip():
                arguments = {}
            else:
                try:
                    arguments = json.loads(arguments_raw, strict=False)
                except json.JSONDecodeError as exc:
                    raise ToolCallParseError(
                        f"tool call {name!r} has malformed JSON arguments",
                        raw=arguments_raw,
                    ) from exc
        else:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ToolCallParseError(
                f"tool call {name!r} arguments are not a JSON object",
                raw=repr(arguments_raw),
            )
        tool_calls.append(ToolCall(id=tc_id, name=name, arguments=arguments))

    if not tool_calls and tools and content:
        # Fallback for local/edge models that ignore native `tools` and
        # just narrate a tool-call attempt in plain text (see
        # `_TOOL_DIRECTIVE` in agent_runtime/agent.py, which asks for this
        # exact shape). Without this, such models are indistinguishable
        # from ones giving a genuine prose answer — the agent would treat
        # "I'll write foo.py now: {...}" as a final message, never
        # executing anything, while the UI still narrates it as a step.
        fallback = _extract_text_tool_call(content, tools)
        if fallback is not None:
            tool_calls = [fallback]
            content = ""

    if tool_calls and not finish_reason:
        finish_reason = "tool_calls"
    if content and not finish_reason:
        finish_reason = "stop"

    return ModelResponse(
        content=content,
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=finish_reason,
    )


__all__ = [
    "OpenAICompatBackendConfig", "OpenAICompatBackend",
    "DEFAULT_OLLAMA_V1", "DEFAULT_LLAMACPP_V1", "DEFAULT_LMSTUDIO_V1",
]
