"""
orcha.orchestration.selector
=============================
Routes each query to the most appropriate subset of registered experts.

Selection strategies
--------------------
RANKED   — score every expert against detected domains + query keywords,
           take the top-N by score. Default.
ALL      — send to every registered expert (force_all_experts=True).
WEIGHTED_RANDOM — stochastic: probability of selection proportional to score.
           Useful for exploration during evals.

Scoring model
-------------
Each expert starts with a base score of 0.25 (so even off-domain experts
can be selected if nothing better exists). Then:

  domain_match   +0.45  if expert.domain in detected_domains
  general_boost  +0.10  if expert.domain == "general" (catch-all)
  keyword_match  +0.00 – +0.20  based on query–description token overlap
  history_bonus  +0.00 – +0.10  if expert has good recent performance

The resulting 0.0–1.0 score is deterministic given the same inputs, making
runs reproducible unless WEIGHTED_RANDOM is selected.

Expert exclusion
----------------
The retry controller can pass a set of excluded expert names in the packet
payload ("excluded_experts"). Those experts are always skipped, allowing the
next iteration to try fresh models when a previous attempt had failures.

Circuit breaker
---------------
After CIRCUIT_OPEN_THRESHOLD consecutive failures from the same expert, the
selector marks it as excluded for the remainder of the run (stored on the
selector instance, per-query run). The circuit resets when the orchestrator
is re-instantiated.
"""
from __future__ import annotations

import enum
import random
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Set

import time

from ..core.packets import OrchaPacket, PacketKind, ExpertSlot
from ..experts.base import BaseExpert


# ── Constants ─────────────────────────────────────────────────────────────────

CIRCUIT_OPEN_THRESHOLD = 3   # consecutive failures before blacklisting
BASE_SCORE             = 0.25
DOMAIN_MATCH_BONUS     = 0.45
GENERAL_BOOST          = 0.10
MAX_KEYWORD_BONUS      = 0.20
MAX_HISTORY_BONUS      = 0.10


class SelectionStrategy(str, enum.Enum):
    """
    How the selector narrows scored experts down to the execution set.

    RANKED           Top-N by deterministic score (default, reproducible).
    WEIGHTED_RANDOM  Stochastic: selection probability proportional to score.
                     Useful for exploration during evaluation/benchmarking —
                     non-deterministic by design.
    """
    RANKED           = "ranked"
    WEIGHTED_RANDOM  = "weighted_random"


# ── Selector ──────────────────────────────────────────────────────────────────

class ExpertSelector:
    """
    Scores, filters, and selects experts for each pipeline iteration.

    Maintains per-expert performance history so selection improves
    as the orchestration loop runs more iterations.
    """

    def __init__(
        self,
        experts:  Dict[str, BaseExpert],
        strategy: SelectionStrategy = SelectionStrategy.RANKED,
        seed:     Optional[int] = None,
    ):
        self.experts: Dict[str, BaseExpert] = dict(experts)
        self.strategy: SelectionStrategy = strategy
        # A fixed seed makes WEIGHTED_RANDOM reproducible (e.g. for tests).
        self._rng: random.Random = random.Random(seed) if seed is not None else random

        # Pre-compute expert description tokens to avoid re-splitting on every
        # scoring call — descriptions are immutable, so compute once.
        self._desc_tokens: Dict[str, frozenset] = {
            name: frozenset(expert.description.lower().split())
            for name, expert in self.experts.items()
        }

        # Performance tracking (updated by record_result)
        self._success_count:    DefaultDict[str, int]   = defaultdict(int)
        self._failure_count:    DefaultDict[str, int]   = defaultdict(int)
        self._confidence_sum:   DefaultDict[str, float] = defaultdict(float)
        self._consecutive_fail: DefaultDict[str, int]   = defaultdict(int)
        self._circuit_open:     Set[str]                = set()

    # ── Public API ────────────────────────────────────────────────────

    def select(self, packet: OrchaPacket) -> OrchaPacket:
        t0            = time.perf_counter()
        domains       = packet.payload.get("domains", ["general"])
        width         = max(1, packet.payload.get("parallel_width", 3))
        force_all     = bool(packet.payload.get("force_all_experts", False))
        excluded      = set(packet.payload.get("excluded_experts", []))
        excluded     |= self._circuit_open   # add circuit-broken experts

        if not self.experts:
            return packet.fork(
                PacketKind.SELECTION, selected_experts=[]
            ).stamp("select", 0.0, selected=[], warning="no_experts_registered")

        # Score all eligible experts
        eligible = {
            name: expert
            for name, expert in self.experts.items()
            if name not in excluded
        }
        if not eligible:
            # All excluded — reset circuit breakers and try again
            self._circuit_open.clear()
            eligible = dict(self.experts)

        scored = self._score_all(eligible, domains, packet.query)

        if force_all:
            selected = scored                        # every eligible expert
        elif self.strategy is SelectionStrategy.WEIGHTED_RANDOM:
            selected = self._pick_weighted_random(scored, width)
        else:
            selected = self._pick(scored, width)     # top-N (default, deterministic)

        duration_ms = (time.perf_counter() - t0) * 1000
        slot_dicts  = [s.model_dump() for s in selected]

        return packet.fork(
            PacketKind.SELECTION,
            selected_experts=slot_dicts,
            excluded_experts=list(excluded),
        ).stamp(
            "select", duration_ms,
            selected=[s.name for s in selected],
            scores={s.name: round(s.score, 3) for s in selected},
            mode="all" if force_all else self.strategy.value,
            excluded=list(excluded),
        )

    def record_result(self, name: str, success: bool, confidence: float) -> None:
        """
        Feed back the result of an expert call so the selector can improve
        future routing. Called by the executor after each parallel run.
        """
        if success and confidence > 0:
            self._success_count[name] += 1
            self._confidence_sum[name] += confidence
            self._consecutive_fail[name] = 0
        else:
            self._failure_count[name] += 1
            self._consecutive_fail[name] += 1
            if self._consecutive_fail[name] >= CIRCUIT_OPEN_THRESHOLD:
                self._circuit_open.add(name)

    def reset_circuit_breakers(self) -> None:
        self._circuit_open.clear()
        self._consecutive_fail.clear()

    def performance_summary(self) -> Dict[str, Dict]:
        out = {}
        for name in self.experts:
            total = self._success_count[name] + self._failure_count[name]
            avg_conf = (
                self._confidence_sum[name] / self._success_count[name]
                if self._success_count[name] else 0.0
            )
            out[name] = {
                "calls":       total,
                "successes":   self._success_count[name],
                "failures":    self._failure_count[name],
                "avg_confidence": round(avg_conf, 3),
                "circuit_open":   name in self._circuit_open,
            }
        return out

    # ── Internal scoring ──────────────────────────────────────────────

    def _score_all(
        self,
        eligible: Dict[str, BaseExpert],
        domains: List[str],
        query: str,
    ) -> List[ExpertSlot]:
        query_tokens = frozenset(query.lower().split())
        domain_set = frozenset(domains)  # O(1) lookup vs O(n) list scan
        slots = []
        for name, expert in eligible.items():
            score = self._score_expert(name, expert, domain_set, query_tokens)
            slots.append(ExpertSlot(
                name=expert.name,
                domain=expert.domain,
                description=expert.description,
                score=round(min(score, 1.0), 4),
            ))
        slots.sort(key=lambda s: s.score, reverse=True)
        return slots

    def _score_expert(
        self,
        name: str,
        expert: BaseExpert,
        domain_set: frozenset,
        query_tokens: frozenset,
    ) -> float:
        score = BASE_SCORE

        # Domain match — O(1) frozenset lookup
        if expert.domain in domain_set:
            score += DOMAIN_MATCH_BONUS
        elif expert.domain == "general":
            score += GENERAL_BOOST

        # Keyword overlap — use pre-computed frozensets
        desc_tokens = self._desc_tokens.get(name, frozenset())
        overlap     = len(query_tokens & desc_tokens)
        score      += min(overlap * 0.04, MAX_KEYWORD_BONUS)

        # Historical performance bonus
        total = self._success_count[name] + self._failure_count[name]
        if total > 0:
            success_rate = self._success_count[name] / total
            avg_conf = (
                self._confidence_sum[name] / self._success_count[name]
                if self._success_count[name] else 0.0
            )
            history_score = (success_rate * 0.5 + avg_conf * 0.5) * MAX_HISTORY_BONUS
            score += history_score

        return score

    def _pick(self, scored: List[ExpertSlot], width: int) -> List[ExpertSlot]:
        """Top-N selection (default)."""
        return scored[:width]

    def _pick_weighted_random(
        self, scored: List[ExpertSlot], width: int
    ) -> List[ExpertSlot]:
        """
        Stochastic selection weighted by score — probability of selection is
        proportional to each expert's score. Samples without replacement.

        Useful for exploration during evaluation/benchmarking. Non-deterministic
        unless the selector was constructed with an explicit seed.
        """
        if not scored:
            return []
        k = min(width, len(scored))
        # A zero score still gets a small epsilon so it can be picked when
        # it's the only option (avoids random.choices rejecting all-zero weights).
        weights = [max(s.score, 1e-6) for s in scored]
        # random.choices with k unique indices, then dedupe while preserving count.
        # We use a sequence of k independent draws then de-duplicate to the
        # distinct chosen experts — matching "width = parallelism budget".
        chosen_idx = self._rng.choices(range(len(scored)), weights=weights, k=k)
        # Keep first occurrence of each chosen index, in score order.
        seen: set = set()
        distinct = []
        for i in sorted(set(chosen_idx)):
            if i not in seen:
                seen.add(i)
                distinct.append(scored[i])
        return distinct


def get_selector(
    experts:  Dict[str, BaseExpert],
    strategy: SelectionStrategy = SelectionStrategy.RANKED,
    seed:     Optional[int] = None,
) -> ExpertSelector:
    return ExpertSelector(experts, strategy=strategy, seed=seed)
