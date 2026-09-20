"""
orcha.orchestrator
====================
The top-level runtime. Wires all pipeline stages into the iterative loop
and exposes a clean public API.

Pipeline (one iteration)
-------------------------
    decompose
        ↓
    plan  ──► stop?  ──► return result
        ↓
    select  (scores experts, respects exclusions)
        ↓
    execute  (parallel asyncio.gather, timeout, per-call retry)
        ↓
    aggregate  (synthesis → vote → confidence-weighted)
        ↓
    evaluate  (5 quality dimensions)
        ↓
    retry_decision  ──► retry?  ──► back to plan
        ↓
    return result

Everything flows through a single OrchaPacket. Each stage reads from
packet.payload, writes results back, stamps a TraceStep, and returns
the packet. The loop is guaranteed to terminate because the executor
always increments budget.iterations, and the planner stops when the
budget is exhausted.

Usage patterns
--------------
# 1. Simple sync call (best for scripts)
orc = Orchestrator(experts=load_mock_experts())
result = orc.run("How should I diversify my portfolio?")
print(result.answer)

# 2. Async call (best for FastAPI / Jupyter)
result = await orc.run_async("...")

# 3. Full local-model setup
registry = LocalModelRegistry()
await registry.discover_ollama()
orc = Orchestrator(
    experts=registry.build(),
    synthesizer_expert=registry.pick_synthesizer(),
    run_all_experts=True,
    max_iterations=3,
)

# 4. With event hooks for observability
orc = Orchestrator(experts=..., on_stage=my_callback)
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

from .core.packets import OrchaPacket, PacketKind, BudgetState
from .experts.base import BaseExpert
from .observability import get_logger, log_stage
from .orchestration.decomposer import get_decomposer
from .orchestration.planner    import get_planner
from .orchestration.selector   import get_selector, ExpertSelector
from .orchestration.executor   import get_executor
from .orchestration.aggregator import get_aggregator
from .orchestration.evaluator  import get_evaluator
from .orchestration.retry      import get_retry_controller


# ── Result object ─────────────────────────────────────────────────────────────

class OrchaResult:
    """
    Immutable result of a completed orchestration run.

    Attributes
    ----------
    answer          Final synthesized (or best-of) answer text.
    confidence      0–1 quality score of the final answer.
    quality_score   Multi-dimensional score from the evaluator.
    synthesized     True if a local model synthesized all expert answers.
    primary         Name of the expert / synthesizer that produced the answer.
    contributors    All experts whose output fed into the final answer.
    domains         Detected query domains (finance, code, …).
    iterations      Number of pipeline iterations that ran.
    cost            Estimated total cost (USD) across all expert calls.
    latency_s       Max per-iteration latency summed across iterations.
    trace           Full list of {stage, duration_ms, data} dicts.
    packet          The underlying OrchaPacket (for advanced inspection).
    """

    def __init__(self, packet: OrchaPacket):
        self.packet       = packet
        p                 = packet.payload
        self.answer       = p.get("answer", "")
        self.confidence   = p.get("confidence", 0.0)
        self.quality_score = p.get("quality_score", self.confidence)
        self.synthesized  = p.get("synthesized", False)
        self.primary      = p.get("primary")
        self.contributors = p.get("contributors", [])
        self.domains      = p.get("domains", [])
        self.agg_mode     = p.get("agg_mode", "unknown")
        self.iterations   = packet.budget.iterations
        self.cost         = packet.budget.cost_used
        self.latency_s    = packet.budget.latency_used_s
        self.trace        = [t.model_dump() for t in packet.trace]

    # Alias kept for backwards compatibility
    @property
    def final_answer(self) -> str:
        return self.answer

    def explain(self) -> str:
        """Human-readable, aligned trace of every pipeline decision."""
        lines = [
            f"Query    : {self.packet.query[:120]}",
            f"Packet   : {self.packet.id[:8]}",
            "",
        ]
        for step in self.trace:
            extras = "  ".join(f"{k}={v}" for k, v in step["data"].items())
            lines.append(f"[{step['stage']:>16}] {step['duration_ms']:7.2f}ms  {extras}")
        lines += [
            "",
            f"Answer      : {self.answer[:120]}{'…' if len(self.answer) > 120 else ''}",
            f"Confidence  : {self.confidence:.3f}",
            f"Quality     : {self.quality_score:.3f}",
            f"Mode        : {self.agg_mode}  (synthesized={self.synthesized})",
            f"Iterations  : {self.iterations}",
            f"Cost        : ${self.cost:.4f}",
            f"Latency     : {self.latency_s:.2f}s",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer":        self.answer,
            "confidence":    self.confidence,
            "quality_score": self.quality_score,
            "synthesized":   self.synthesized,
            "primary":       self.primary,
            "contributors":  self.contributors,
            "domains":       self.domains,
            "agg_mode":      self.agg_mode,
            "iterations":    self.iterations,
            "cost":          self.cost,
            "latency_s":     self.latency_s,
            "trace":         self.trace,
        }

    def __repr__(self) -> str:
        return (
            f"OrchaResult(confidence={self.confidence:.2f}, "
            f"synthesized={self.synthesized}, "
            f"iterations={self.iterations}, cost=${self.cost:.4f})"
        )


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Main Orcha runtime. Thread-safe for concurrent run_async() calls
    (each call creates its own packet; shared state is read-only after init).
    """

    def __init__(
        self,
        experts:            Optional[Dict[str, BaseExpert]] = None,
        synthesizer_expert: Optional[str]                   = None,
        run_all_experts:    bool                            = False,
        max_cost:           float                           = 1.0,
        max_latency_s:      float                           = 120.0,
        max_iterations:     int                             = 3,
        use_embeddings:     bool                            = False,
        reasoning:          Optional[str]                   = None,
        on_stage:           Optional[Callable[[str, OrchaPacket], None]] = None,
    ):
        """
        Parameters
        ----------
        experts             Dict of {name: BaseExpert}. Add any mix of
                            OllamaExpert, LocalChatExpert, or custom experts.
        synthesizer_expert  Name of the expert that writes the final synthesis.
                            Should be your largest / most capable local model.
        run_all_experts     If True, ALL experts run every iteration (ignores
                            parallel_width from the planner). Best for small
                            pools of trusted local models.
        max_cost            Budget cap in USD. Local models cost $0 so this
                            only matters if you add cloud experts.
        max_latency_s       Maximum wall-clock latency budget across all
                            iterations combined.
        max_iterations      Hard cap on pipeline loop count.
        use_embeddings      Use sentence-transformers for semantic decomposition
                            and evaluation (requires: pip install sentence-transformers).
        reasoning           Reasoning level (fast/light/medium/high/max). Stamped
                            into every packet so the planner can tune width and
                            quality thresholds per level.
        on_stage            Optional callback invoked after every stage with
                            (stage_name, packet). Useful for streaming UIs.
        """
        self.experts:            Dict[str, BaseExpert] = dict(experts or {})
        self.synthesizer_expert: Optional[str]         = synthesizer_expert
        self.run_all_experts:    bool                  = run_all_experts
        self.max_cost:           float                 = max_cost
        self.max_latency_s:      float                 = max_latency_s
        self.max_iterations:     int                   = max_iterations
        self.reasoning:          Optional[str]         = reasoning
        self.on_stage:           Optional[Callable]    = on_stage

        self.decomposer = get_decomposer(use_embeddings=use_embeddings)
        self.planner    = get_planner()
        self.evaluator  = get_evaluator(use_embeddings=use_embeddings)

        self._selector: Optional[ExpertSelector] = None
        self._logger   = get_logger("orcha.orchestrator")
        self._refresh_routing()

    # ── Expert management ─────────────────────────────────────────────

    def _refresh_routing(self) -> None:
        """Rebuild selector/executor/aggregator/retry after expert pool changes."""
        self._selector  = get_selector(self.experts)
        self._executor  = get_executor(self.experts, selector=self._selector)
        self._aggregator = get_aggregator(
            experts=self.experts,
            synthesizer_expert=self.synthesizer_expert,
        )
        self._retry_ctrl = get_retry_controller(
            synthesizer_expert=self.synthesizer_expert,
            total_experts=len(self.experts),
        )

    def register_expert(self, name: str, expert: BaseExpert) -> "Orchestrator":
        """Add or replace an expert. Returns self for chaining."""
        expert.name      = name
        self.experts[name] = expert
        self._refresh_routing()
        return self

    def unregister_expert(self, name: str) -> "Orchestrator":
        self.experts.pop(name, None)
        self._refresh_routing()
        return self

    def set_synthesizer(self, name: Optional[str]) -> "Orchestrator":
        """Change the synthesizer expert. Pass None to disable synthesis."""
        self.synthesizer_expert = name
        self._refresh_routing()
        return self

    def list_experts(self) -> List[Dict[str, Any]]:
        return [
            {
                "name":           name,   # use registry key as canonical name
                "domain":         e.domain,
                "description":    e.description,
                "version":        e.version,
                "is_synthesizer": name == self.synthesizer_expert,
            }
            for name, e in self.experts.items()
        ]

    def selector_performance(self) -> Dict[str, Any]:
        """Per-expert performance history tracked by the selector."""
        if self._selector:
            return self._selector.performance_summary()
        return {}

    # ── Execution ─────────────────────────────────────────────────────

    async def run_async(self, query: str) -> OrchaResult:
        """Full async orchestration run. Safe to call concurrently."""
        start = time.perf_counter()
        log = self._logger

        payload: Dict[str, Any] = {"force_all_experts": self.run_all_experts}
        if self.reasoning:
            payload["reasoning"] = self.reasoning
        packet = OrchaPacket(
            kind=PacketKind.QUERY,
            query=query,
            payload=payload,
            budget=BudgetState(
                max_cost=self.max_cost,
                max_latency_s=self.max_latency_s,
                max_iterations=self.max_iterations,
            ),
        ).stamp("input", 0.0, query=query[:120])
        tid = packet.id
        log_stage(log, tid, "orchestration.start",
                  query=query[:80], experts=len(self.experts),
                  max_iterations=self.max_iterations)

        # ── Decompose (once, outside the loop) ────────────────────────
        packet = self.decomposer.decompose(packet)
        self._emit("decompose", packet)
        log_stage(log, tid, "stage", stage="decompose",
                  domains=packet.payload.get("domains"),
                  complexity=packet.payload.get("complexity"))

        # ── Iterative refinement loop ──────────────────────────────────
        while True:
            # Plan
            packet = self.planner.plan(packet)
            self._emit("plan", packet)
            if packet.payload.get("stop"):
                log_stage(log, tid, "orchestration.stop",
                          reason=packet.payload.get("reason"))
                break

            # Select
            packet = self._selector.select(packet)
            self._emit("select", packet)
            log_stage(log, tid, "stage", stage="select",
                      selected=packet.trace[-1].data.get("selected"),
                      mode=packet.trace[-1].data.get("mode"))

            # Execute (parallel)
            packet = await self._executor.execute(packet)
            self._emit("execute", packet)
            log_stage(log, tid, "stage", stage="execute",
                      ran=packet.trace[-1].data.get("ran"),
                      succeeded=packet.trace[-1].data.get("succeeded"),
                      failed=packet.trace[-1].data.get("failed"))

            # Aggregate (may call synthesizer model)
            packet = await self._aggregator.aggregate(packet)
            self._emit("aggregate", packet)

            # Evaluate
            packet = self.evaluator.evaluate(packet)
            self._emit("evaluate", packet)
            log_stage(log, tid, "stage", stage="evaluate",
                      passed=packet.payload.get("passed"),
                      score=packet.payload.get("quality_score"))

            # Retry decision
            packet = self._retry_ctrl.decide(packet)
            self._emit("retry", packet)
            if not packet.payload.get("retry"):
                break
            log_stage(log, tid, "orchestration.retry",
                      reason=packet.payload.get("retry_reason"),
                      excluded=packet.payload.get("excluded_experts"))

        # Finalise
        total_ms = (time.perf_counter() - start) * 1000
        packet = packet.fork(PacketKind.RESPONSE).stamp(
            "response", total_ms,
            total_ms=round(total_ms, 1),
            iterations=packet.budget.iterations,
        )
        log_stage(log, tid, "orchestration.finish",
                  iterations=packet.budget.iterations,
                  confidence=packet.payload.get("confidence"),
                  agg_mode=packet.payload.get("agg_mode"),
                  total_ms=round(total_ms, 1))
        return OrchaResult(packet)

    def run(self, query: str) -> OrchaResult:
        """
        Synchronous wrapper. Works in scripts and tests.
        Raises RuntimeError if called from inside a running event loop
        (e.g. inside a FastAPI route or Jupyter cell) — use run_async() there.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async(query))
        raise RuntimeError(
            "Orchestrator.run() called inside a running event loop. "
            "Use 'await orchestrator.run_async(query)' instead."
        )

    # ── Internals ─────────────────────────────────────────────────────

    def _emit(self, stage: str, packet: OrchaPacket) -> None:
        if self.on_stage is not None:
            try:
                self.on_stage(stage, packet)
            except Exception:
                pass   # hooks must never crash the pipeline
