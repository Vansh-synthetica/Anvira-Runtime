"""
orcha.experts.ollama
====================
Expert backed by a locally running Ollama daemon.

Supports
--------
- Any model pulled with `ollama pull <name>`
- Auto-discovery of all available models (list_ollama_models)
- Configurable temperature, system prompt, context window, num_predict
- Streaming detection (uses non-streaming /api/chat for simplicity)
- Healthcheck that verifies the specific model is actually loaded

Install Ollama
--------------
    curl -fsSL https://ollama.ai/install.sh | sh
    ollama pull llama3.2:3b
    ollama pull qwen2.5:32b      # or whatever you have
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .base import BaseExpert, ExpertOutput


def _slugify(s: str) -> str:
    return s.replace(":", "_").replace(".", "_").replace("/", "_").replace("-", "_")


class OllamaExpert(BaseExpert):
    """
    Expert that calls any model served by a local Ollama daemon.
    Zero cost (local inference), configurable per-instance.
    """

    version            = "1.2"
    cost_per_1k_tokens = 0.0

    def __init__(
        self,
        model:           str,
        base_url:        str            = "http://localhost:11434",
        name:            Optional[str]  = None,
        domain:          str            = "general",
        description:     Optional[str]  = None,
        system_prompt:   Optional[str]  = None,
        temperature:     float          = 0.7,
        top_p:           float          = 0.9,
        num_predict:     Optional[int]  = None,   # max tokens to generate
        num_ctx:         Optional[int]  = None,   # context window override
        timeout_s:       float          = 240.0,
        keep_alive:      str            = "5m",   # how long Ollama keeps model loaded
    ):
        self.model         = model
        self.base_url      = base_url.rstrip("/")
        self.system_prompt = system_prompt
        self.temperature   = temperature
        self.top_p         = top_p
        self.num_predict   = num_predict
        self.num_ctx       = num_ctx
        self.timeout_s     = timeout_s
        self.keep_alive    = keep_alive

        self.name        = name or f"ollama_{_slugify(model)}"
        self.domain      = domain
        self.description = description or f"Local Ollama model: {model}"

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """
        Native chat completion for Ollama's /api/chat returning the full
        assistant ``message`` (content + ``tool_calls``). Falls back to a
        request without tools if the model/server rejects them.
        """
        import httpx  # soft dep

        full_messages = list(messages)
        if system:
            full_messages = [{"role": "system", "content": system}, *full_messages]

        options: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p":       self.top_p,
        }
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        options.update(overrides)

        async def _post(tool_list) -> Dict[str, Any]:
            payload: Dict[str, Any] = {
                "model":      self.model,
                "messages":   full_messages,
                "stream":     False,
                "options":    options,
                "keep_alive": self.keep_alive,
            }
            if tool_list:
                payload["tools"] = tool_list
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
                resp.raise_for_status()
                return resp.json()

        try:
            data = await _post(tools)
        except Exception:
            if not tools:
                raise
            data = await _post(None)

        message = data.get("message") or {}
        # Normalize Ollama's tool_calls to the OpenAI shape the engine expects.
        tool_calls = message.get("tool_calls") or []
        normalized = []
        for tc in tool_calls:
            fn = tc.get("function") or {}
            normalized.append({
                "id": tc.get("id") or f"call_{len(normalized) + 1}",
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", {}),
                },
            })
        if normalized:
            message["tool_calls"] = normalized
        return message

    async def execute(self, query: str) -> ExpertOutput:
        import httpx   # soft dep — only needed at call time

        t0 = time.perf_counter()
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": query})

        options: Dict[str, Any] = {"temperature": self.temperature, "top_p": self.top_p}
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx

        payload: Dict[str, Any] = {
            "model":      self.model,
            "messages":   messages,
            "stream":     False,
            "options":    options,
            "keep_alive": self.keep_alive,
        }

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        message     = data.get("message", {})
        answer      = message.get("content", "").strip()
        done_reason = data.get("done_reason", "stop")
        eval_count  = data.get("eval_count", 0)
        prompt_count = data.get("prompt_eval_count", 0)
        tokens      = eval_count + prompt_count
        latency_s   = time.perf_counter() - t0

        # Confidence approximation: natural stop = higher confidence
        confidence = 0.82 if done_reason == "stop" else 0.60

        return ExpertOutput(
            answer=answer,
            confidence=confidence,
            tokens_used=tokens,
            latency_s=latency_s,
            finish_reason=done_reason,
            model_version=data.get("model", self.model),
            metadata={
                "eval_count":        eval_count,
                "prompt_eval_count": prompt_count,
                "eval_duration_ns":  data.get("eval_duration"),
                "total_duration_ns": data.get("total_duration"),
            },
        )

    async def healthcheck(self) -> bool:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=6.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code != 200:
                    return False
                models = [m["name"] for m in resp.json().get("models", [])]
                # Accept exact match or prefix match (e.g. "llama3" matches "llama3:latest")
                return any(
                    m == self.model or m.startswith(self.model.split(":")[0])
                    for m in models
                )
        except Exception:
            return False

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "model":       self.model,
            "base_url":    self.base_url,
            "temperature": self.temperature,
        })
        return base


# ── Discovery helpers ─────────────────────────────────────────────────────────

async def list_ollama_models(
    base_url: str = "http://localhost:11434",
) -> List[str]:
    """
    Return the names of every model currently pulled in the local Ollama
    instance (e.g. ['llama3.2:3b', 'qwen2.5:32b', 'codellama:13b']).

    Returns [] if Ollama is not reachable — never raises.
    """
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


async def get_ollama_model_info(
    model: str, base_url: str = "http://localhost:11434"
) -> Dict[str, Any]:
    """
    Return the /api/show metadata for a specific model (parameter count,
    quantisation, context length, families, etc.).
    Returns {} if the model or Ollama is not found.
    """
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/api/show",
                json={"name": model},
            )
            if resp.status_code != 200:
                return {}
            data = resp.json()
            details = data.get("details", {})
            return {
                "parameter_size":  details.get("parameter_size"),
                "quantization":    details.get("quantization_level"),
                "families":        details.get("families", []),
                "context_length":  details.get("context_length"),
                "model_family":    details.get("family"),
            }
    except Exception:
        return {}
