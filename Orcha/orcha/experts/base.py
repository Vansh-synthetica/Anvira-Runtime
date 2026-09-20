"""
orcha.experts.base
==================
The BaseExpert contract. Every model backend — Ollama, llama.cpp,
vLLM, a custom fine-tune, or a mock — implements this interface.

Design principles
-----------------
- Async-native: execute() is always a coroutine so the executor can run
  many experts truly concurrently with asyncio.gather().
- Self-describing: name, domain, description, and capabilities let the
  selector route queries intelligently without hard-coded rules.
- Cost-aware: cost_per_1k_tokens lets the budget system track spend.
- Observable: every ExpertOutput carries latency, token counts, and a
  finish_reason so the aggregator can make informed weighting decisions.
- Fault-tolerant: healthcheck() lets the server warm up gracefully and
  the executor detect dead experts before wasting a pipeline slot.
"""
from __future__ import annotations

import abc
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Output model ──────────────────────────────────────────────────────────────

class ExpertOutput(BaseModel):
    """
    Structured output from a single expert.execute() call.

    Every field beyond `answer` is optional but callers should populate
    as many as the backend exposes — the aggregator and evaluator use
    confidence, tokens_used, and latency_s to make weighting decisions.
    """
    answer:        str
    confidence:    float = Field(default=0.75, ge=0.0, le=1.0)
    tokens_used:   int   = 0
    latency_s:     float = 0.0
    finish_reason: Optional[str] = None  # "stop", "length", "timeout", …
    model_version: Optional[str] = None  # e.g. "llama3:8b-instruct-q4_K_M"
    metadata:      Dict[str, Any] = {}

    model_config = {"extra": "allow"}


# ── Capability descriptor ─────────────────────────────────────────────────────

class ExpertCapability(BaseModel):
    """
    Fine-grained capability descriptor. When populated, the selector can
    make much smarter routing decisions than pure domain-matching.
    """
    domain:           str         = "general"
    strength:         float       = Field(default=0.5, ge=0.0, le=1.0)
    sub_domains:      List[str]   = []        # e.g. ["portfolio", "derivatives"]
    context_window:   Optional[int] = None    # tokens
    languages:        List[str]   = ["en"]
    reasoning_depth:  str         = "medium"  # "shallow" | "medium" | "deep"
    structured_output: bool       = False     # can reliably produce JSON


# ── Base class ────────────────────────────────────────────────────────────────

class BaseExpert(abc.ABC):
    """
    Plugin contract for Orcha experts.

    Minimal implementation
    ----------------------
    class MyExpert(BaseExpert):
        name = "my_model"
        domain = "reasoning"
        description = "Custom fine-tune for logical reasoning"

        async def execute(self, query: str) -> ExpertOutput:
            text = await call_my_backend(query)
            return ExpertOutput(answer=text, confidence=0.85, tokens_used=200)

    The orchestrator handles retrying, budgeting, and synthesis — the expert
    only needs to worry about calling its backend and returning an output.
    """

    # ── Class-level metadata (override in subclasses) ──────────────────
    name:               str   = "unnamed_expert"
    domain:             str   = "general"
    description:        str   = "A general-purpose AI expert"
    version:            str   = "1.0"
    cost_per_1k_tokens: float = 0.0           # USD; 0 for local models
    max_retries:        int   = 1             # retries inside execute()
    default_timeout_s:  float = 180.0

    # ── Abstract interface ─────────────────────────────────────────────

    @abc.abstractmethod
    async def execute(self, query: str) -> ExpertOutput:
        """
        Run inference and return a structured output.
        Must be implemented by every expert.
        Raise any exception on unrecoverable failure — the executor
        will catch it and record it as a failed ExpertResult.
        """
        ...

    # ── Optional streaming interface ───────────────────────────────────

    supports_streaming: bool = False

    async def execute_stream(self, query: str):
        """
        Optional: yield the answer incrementally as it's generated,
        instead of only returning it once complete via execute().

        Must be an async generator yielding plain str chunks (each chunk
        is the next piece of text to append — NOT the full answer so
        far). The final chunk should be followed by a StopAsyncIteration
        as usual; callers are responsible for assembling the full answer
        from the yielded chunks if they need it (e.g. for logging or
        Nomi memory) once the generator is exhausted.

        Default implementation: falls back to a single-chunk "stream"
        containing the full execute() result. This means every expert
        works with streaming API consumers without needing to implement
        anything — experts that support real incremental generation
        (e.g. LocalChatExpert) override this for a genuinely better
        experience; everything else (mock experts, non-streaming remote
        APIs) still functions correctly, just without the incremental
        feel.
        """
        output = await self.execute(query)
        yield output.answer

    # ── Optional overrides ─────────────────────────────────────────────

    async def healthcheck(self) -> bool:
        """
        Return True if the backend is reachable and ready to serve requests.
        Called by the server on startup and after /reload. Default: True.
        """
        return True

    def capabilities(self) -> ExpertCapability:
        """
        Return a detailed capability descriptor for sophisticated routing.
        Default implementation derives from class-level attributes.
        """
        return ExpertCapability(domain=self.domain)

    def warm(self) -> None:
        """
        Optional: perform any initialisation that should happen once
        before the first execute() call (e.g. loading a tokenizer).
        Called by the registry after registration.
        """

    # ── Utility ───────────────────────────────────────────────────────

    def estimate_cost(self, tokens: int) -> float:
        """Estimate USD cost for a given token count."""
        return (tokens / 1000.0) * self.cost_per_1k_tokens

    async def execute_with_timing(self, query: str) -> ExpertOutput:
        """
        Wrapper that fills in latency_s even if the subclass forgets to.
        The executor always calls this instead of execute() directly.
        """
        t0 = time.perf_counter()
        output = await self.execute(query)
        if output.latency_s == 0.0:
            output.latency_s = time.perf_counter() - t0
        return output

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "description": self.description,
            "version": self.version,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, domain={self.domain!r}, "
            f"version={self.version!r})"
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BaseExpert) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)
