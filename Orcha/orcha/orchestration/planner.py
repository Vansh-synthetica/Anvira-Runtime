"""
orcha.orchestration.planner
=============================
The global control tower. Runs at the start of each pipeline iteration
to decide:

  - Should we stop now?  (quality already excellent, or budget gone)
  - How many experts to run in parallel this iteration?
  - What quality threshold must the answer meet to avoid a retry?
  - Which experts (if any) should be excluded this iteration?

Budget-aware effort scaling
----------------------------
The planner operates on a progress metric:

    progress = iterations_used / max_iterations   (0.0 → 1.0)

At progress=0 (first try): run conservatively — moderate width, moderate
threshold. This is usually enough for straightforward queries.

At progress>0 (retries): ramp up — more experts, stricter quality bar.
The logic: if an earlier attempt wasn't good enough, throw more compute
at it. The budget system ensures we can't over-spend.

Complexity-aware width
-----------------------
The decomposer embeds a `complexity` score (0.0–1.0) in the packet.
The planner uses this to add extra parallel width for hard queries even
on the first iteration.

Expert exclusion forwarding
-----------------------------
The retry controller marks experts for exclusion when they fail. The
planner reads this list and forwards it to the selector so it can avoid
reusing experts that are known to be failing.
"""
from __future__ import annotations

import time
from typing import List

from ..core.packets import OrchaPacket, PacketKind


# ── Scaling parameters ────────────────────────────────────────────────────────

MIN_WIDTH               = 2     # always run at least this many experts
MAX_WIDTH               = 10    # even with budget, don't fan out more than this
COMPLEXITY_WIDTH_BONUS  = 3     # extra experts added at complexity=1.0
MIN_QUALITY_THRESHOLD   = 0.60
MAX_QUALITY_THRESHOLD   = 0.90
EXCELLENT_CONFIDENCE    = 0.92  # above this, stop immediately


# Reasoning-level planner overrides. `width` pins the exact parallel width
# (None = default planner logic); `threshold` replaces the computed quality
# threshold (None = default scaling). Low levels relax both for a fast
# one-shot pass; high levels tighten them so weak answers are retried.
_REASONING_OVERRIDES = {
    "fast":   {"width": 1,  "threshold": 0.50},
    "light":  {"width": 2,  "threshold": 0.55},
    "medium": {"width": None, "threshold": None},
    "high":   {"width": 4,  "threshold": 0.70},
    "max":    {"width": 5,  "threshold": 0.80},
}


class BudgetPlanner:
    """
    Decides per-iteration strategy: stop, or set width + quality threshold.
    """

    def plan(self, packet: OrchaPacket) -> OrchaPacket:
        t0         = time.perf_counter()
        budget     = packet.budget
        iteration  = budget.iterations
        max_iters  = budget.max_iterations
        last_conf  = packet.payload.get("confidence")
        complexity  = float(packet.payload.get("complexity", 0.5))
        excluded    = list(packet.payload.get("excluded_experts", []))

        # ── Stop conditions ───────────────────────────────────────────
        if budget.exhausted:
            return packet.fork(
                PacketKind.PLAN, stop=True, reason="budget_exhausted",
                parallel_width=0, quality_threshold=0, excluded_experts=excluded,
            ).stamp("plan", 0.0, stop=True, reason="budget_exhausted",
                    iterations_used=iteration, budget=budget.utilisation())

        if last_conf is not None and last_conf >= EXCELLENT_CONFIDENCE:
            return packet.fork(
                PacketKind.PLAN, stop=True, reason="quality_excellent",
                parallel_width=0, quality_threshold=0, excluded_experts=excluded,
            ).stamp("plan", 0.0, stop=True, reason="quality_excellent",
                    confidence=last_conf)

        # ── Effort scaling ────────────────────────────────────────────
        progress = iteration / max(1, max_iters - 1) if max_iters > 1 else 0.0

        # Base width grows with iteration progress
        base_width      = MIN_WIDTH + round(progress * (MAX_WIDTH - MIN_WIDTH - COMPLEXITY_WIDTH_BONUS))
        complexity_bonus = round(complexity * COMPLEXITY_WIDTH_BONUS)
        parallel_width  = min(base_width + complexity_bonus, MAX_WIDTH)

        # Quality threshold rises with progress (we get stricter on retries)
        quality_threshold = round(
            MIN_QUALITY_THRESHOLD + progress * (MAX_QUALITY_THRESHOLD - MIN_QUALITY_THRESHOLD),
            3,
        )

        # On the first iteration of a complex query, also increase width
        if iteration == 0 and complexity > 0.6:
            parallel_width = min(parallel_width + 1, MAX_WIDTH)

        # ── Reasoning-level overrides ─────────────────────────────────
        # A requested reasoning level tunes how deep each iteration goes.
        # Low levels pin a narrow single/dual-expert pass and a permissive
        # threshold (fast one-shot replies); high levels fan out wider and
        # demand a stricter bar so weak answers get retried. The packet is
        # stamped by the Orchestrator from the request's `reasoning` field.
        reasoning = packet.payload.get("reasoning")
        if reasoning:
            ov = _REASONING_OVERRIDES.get(reasoning)
            if ov:
                if ov["width"] is not None:
                    parallel_width = min(ov["width"], MAX_WIDTH)
                if ov["threshold"] is not None:
                    quality_threshold = ov["threshold"]

        duration_ms = (time.perf_counter() - t0) * 1000

        return packet.fork(
            PacketKind.PLAN,
            stop=False,
            parallel_width=parallel_width,
            quality_threshold=quality_threshold,
            reason="continue",
            excluded_experts=excluded,
        ).stamp(
            "plan", duration_ms,
            width=parallel_width,
            threshold=quality_threshold,
            iteration=iteration,
            progress=round(progress, 2),
            complexity=round(complexity, 2),
            budget_left=budget.utilisation(),
        )

    def suggest_retry_width(
        self,
        current_width:   int,
        failed_count:    int,
        budget_remaining: float,
    ) -> int:
        """
        How many additional experts should a retry add?
        Scales with how many failed and how much budget is left.
        """
        if budget_remaining <= 0:
            return 0
        boost = min(failed_count, 3)
        return min(current_width + boost, MAX_WIDTH)


def get_planner() -> BudgetPlanner:
    return BudgetPlanner()
