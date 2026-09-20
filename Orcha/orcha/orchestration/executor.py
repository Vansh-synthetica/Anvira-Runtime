"""
orcha.orchestration.executor
=============================
Runs selected experts concurrently and collects structured results.

Key capabilities
----------------
- True parallelism: asyncio.gather() runs all selected experts at the
  same time, so total wall-clock time ≈ slowest expert, not sum of all.
- Per-call timeout: each expert call is cancelled after
  min(budget.remaining_latency_s, expert.default_timeout_s) seconds.
  The run continues with whoever responded in time.
- Fault isolation: an exception in one expert never crashes the others.
  The failure is recorded as a failed ExpertResult with the error string.
- Budget tracking: cost and latency consumed by this iteration are
  folded back into packet.budget so the planner can see remaining headroom.
- Selector feedback: on completion, the executor updates the selector's
  performance history so routing improves on retries.
- Per-expert retry: a configurable per-call retry count lets flaky
  backends succeed on a second attempt without triggering a full pipeline
  retry.
"""
from __future__ import annotations

import asyncio
import traceback
import time
from typing import Dict, List, Optional

from ..core.packets import OrchaPacket, PacketKind, ExpertResult, ExpertSlot
from ..experts.base import BaseExpert
from .selector import ExpertSelector


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_PER_CALL_TIMEOUT = 180.0   # seconds
DEFAULT_PER_CALL_RETRIES = 1       # retry once before marking as failed


# ── Executor ──────────────────────────────────────────────────────────────────

class ParallelExecutor:
    """
    Runs all selected experts concurrently with timeout, per-call retry,
    and budget tracking.
    """

    def __init__(
        self,
        experts:  Dict[str, BaseExpert],
        selector: Optional[ExpertSelector] = None,
        per_call_retries: int = DEFAULT_PER_CALL_RETRIES,
    ):
        self.experts          = experts
        self.selector         = selector   # optional, for feedback
        self.per_call_retries = per_call_retries

    # ── Public ────────────────────────────────────────────────────────

    async def execute(self, packet: OrchaPacket) -> OrchaPacket:
        t0       = time.perf_counter()
        selected = packet.get_selected_experts()

        if not selected:
            new_pkt = packet.fork(PacketKind.EXECUTION, results=[])
            new_pkt.budget.iterations += 1
            return new_pkt.stamp("execute", 0.0, ran=0, warning="no_experts_selected")

        # Determine per-call timeout from remaining budget
        budget_remaining = packet.budget.remaining_latency_s
        call_timeout     = min(
            budget_remaining if budget_remaining > 0 else DEFAULT_PER_CALL_TIMEOUT,
            DEFAULT_PER_CALL_TIMEOUT,
        )

        # Run all selected experts concurrently
        tasks   = [self._run_with_retry(slot, packet.query, call_timeout) for slot in selected]
        results: List[ExpertResult] = await asyncio.gather(*tasks, return_exceptions=False)

        # Feed results back into selector performance history
        if self.selector is not None:
            for r in results:
                self.selector.record_result(r.name, r.success, r.confidence)

        # Update budget
        total_cost     = sum(r.cost     for r in results)
        max_latency    = max((r.latency_s for r in results), default=0.0)
        wall_clock     = time.perf_counter() - t0
        duration_ms    = wall_clock * 1000

        new_pkt = packet.fork(
            PacketKind.EXECUTION,
            results=[r.model_dump() for r in results],
        )
        new_pkt.budget.cost_used      += total_cost
        new_pkt.budget.latency_used_s += max_latency
        new_pkt.budget.iterations     += 1

        succeeded = sum(1 for r in results if r.success)
        failed    = sum(1 for r in results if not r.success)

        return new_pkt.stamp(
            "execute", duration_ms,
            ran=len(results),
            succeeded=succeeded,
            failed=failed,
            cost=round(total_cost, 6),
            max_latency_s=round(max_latency, 3),
            wall_clock_s=round(wall_clock, 3),
            scores={r.name: round(r.confidence, 3) for r in results if r.success},
        )

    # ── Internal ──────────────────────────────────────────────────────

    async def _run_with_retry(
        self, slot: ExpertSlot, query: str, timeout: float
    ) -> ExpertResult:
        """
        Call a single expert, retrying up to self.per_call_retries times on
        transient failure. Returns a failed ExpertResult if all attempts fail.
        """
        expert = self.experts.get(slot.name)
        if expert is None:
            return ExpertResult(
                name=slot.name,
                output="",
                confidence=0.0,
                success=False,
                error=f"Expert '{slot.name}' not found in registry.",
                model_name=slot.name,
            )

        last_error: str = ""
        for attempt in range(1 + self.per_call_retries):
            result = await self._call_once(expert, query, timeout)
            if result.success:
                return result
            last_error = result.error or "unknown error"
            # Don't retry on timeout — the budget is the limit
            if "timeout" in last_error.lower():
                break

        # All attempts exhausted — return last failure
        return ExpertResult(
            name=slot.name,
            output="",
            confidence=0.0,
            success=False,
            error=f"All {1 + self.per_call_retries} attempt(s) failed. Last: {last_error}",
            model_name=getattr(expert, "model", expert.name),
        )

    async def _call_once(
        self, expert: BaseExpert, query: str, timeout: float
    ) -> ExpertResult:
        t0 = time.perf_counter()
        try:
            output = await asyncio.wait_for(
                expert.execute_with_timing(query),
                timeout=timeout,
            )
            latency_s = time.perf_counter() - t0
            return ExpertResult(
                name=expert.name,
                output=output.answer,
                confidence=output.confidence,
                tokens=output.tokens_used,
                latency_s=output.latency_s or latency_s,
                cost=expert.estimate_cost(output.tokens_used),
                success=True,
                finish_reason=output.finish_reason,
                model_name=output.model_version or getattr(expert, "model", expert.name),
                metadata=output.metadata,
            )
        except asyncio.TimeoutError:
            return ExpertResult(
                name=expert.name,
                output="",
                confidence=0.0,
                latency_s=time.perf_counter() - t0,
                success=False,
                error=f"Timed out after {timeout:.1f}s",
                model_name=getattr(expert, "model", expert.name),
            )
        except Exception:
            return ExpertResult(
                name=expert.name,
                output="",
                confidence=0.0,
                latency_s=time.perf_counter() - t0,
                success=False,
                error=traceback.format_exc(limit=3),
                model_name=getattr(expert, "model", expert.name),
            )


def get_executor(
    experts:  Dict[str, BaseExpert],
    selector: Optional[ExpertSelector] = None,
) -> ParallelExecutor:
    return ParallelExecutor(experts=experts, selector=selector)
