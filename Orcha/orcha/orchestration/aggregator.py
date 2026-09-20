"""
orcha.orchestration.aggregator
================================
Combines multiple ExpertResults into one final answer.

Aggregation modes
-----------------
SYNTHESIS  (preferred)
    The synthesizer expert reads every candidate answer inside a structured
    prompt and produces ONE refined, reconciled response — not a
    concatenation, an actual rewrite that resolves conflicts and keeps the
    best of each contributor. Requires synthesizer_expert to be set.

CONFIDENCE_WEIGHTED  (fallback)
    The highest quality-score answer is used as the primary voice. Other
    successful answers are referenced in a cross-check note.

VOTE  (alternative)
    Used when ≥3 experts answered and synthesis is unavailable. Groups
    answers by similarity (using token-overlap Jaccard index), picks the
    largest cluster, and returns its highest-confidence representative.

SINGLE
    Only one expert answered — return it directly.

Quality scoring
---------------
ExpertResult.quality_score() returns a composite 0–1 signal that blends:
  - confidence (reported by the expert / approximated from finish_reason)
  - length bonus (longer answers up to a ceiling are slightly favoured)
  - finish-reason bonus (+0.05 for natural "stop" completions)

Conflict detection
------------------
Before synthesising, the aggregator checks whether expert answers
contradict each other using a simple token-divergence metric. High
divergence triggers a more explicit synthesis prompt that asks the
synthesiser to explicitly resolve the disagreement.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from ..core.packets import (
    OrchaPacket, PacketKind, ExpertResult, AggregationMode,
)
from ..experts.base import BaseExpert


# ── Synthesis prompt templates ────────────────────────────────────────────────

_SYNTHESIS_PROMPT = """\
You are a synthesis expert. Multiple specialist AI models independently \
answered the question below. Your task is to combine their insights into \
ONE refined, accurate, well-organised final answer.

Rules:
- Do NOT simply concatenate the answers.
- Resolve any factual disagreements by favouring the better-reasoned position.
- Preserve unique insights from each contributor that add value.
- Write in a clear, direct style appropriate for the question.
- Do NOT mention that you are synthesising multiple sources.

Question:
{query}

Candidate answers:
{candidates}

Final answer:"""

_CONFLICT_SYNTHESIS_PROMPT = """\
You are a synthesis expert. Multiple specialist AI models answered the \
question below, but their answers CONFLICT on key points. Your task is to \
carefully evaluate each answer, identify which position is better supported \
by logic and evidence, and write ONE authoritative final answer that resolves \
the conflict.

Rules:
- Explicitly resolve any contradictions — do not leave ambiguity.
- Favour the position with stronger reasoning or factual grounding.
- Write in a clear, direct style.
- Do NOT mention that you are synthesising multiple sources.

Question:
{query}

Conflicting candidate answers:
{candidates}

Final answer (resolving the conflict):"""


def _format_candidates(results: List[ExpertResult]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(f"--- Candidate {i} ({r.name}) ---\n{r.output.strip()}")
    return "\n\n".join(blocks)


def _jaccard(a: str, b: str) -> float:
    """Token-overlap Jaccard similarity between two strings.
    Uses frozenset for O(1) membership testing and fast intersection."""
    ta = frozenset(a.lower().split())
    tb = frozenset(b.lower().split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _jaccard_from_sets(ta: frozenset, tb: frozenset) -> float:
    """Jaccard from pre-computed frozensets — avoids re-splitting."""
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _divergence(results: List[ExpertResult]) -> float:
    """
    Average pairwise token divergence (1 - Jaccard) across all pairs of
    successful expert answers. Returns 0.0 for a single result.
    Pre-computes frozensets to avoid redundant splitting.
    """
    if len(results) < 2:
        return 0.0
    # Pre-compute frozensets once per divergence call
    token_sets = [frozenset(r.output.lower().split()) for r in results]
    pairs = 0
    total = 0.0
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            total += 1.0 - _jaccard_from_sets(token_sets[i], token_sets[j])
            pairs += 1
    return total / pairs if pairs else 0.0


HIGH_DIVERGENCE_THRESHOLD = 0.70   # above this, use conflict synthesis prompt


class Aggregator:
    """
    Multi-mode aggregator with local-model synthesis and intelligent fallback.
    """

    def __init__(
        self,
        experts:            Optional[Dict[str, BaseExpert]] = None,
        synthesizer_expert: Optional[str] = None,
    ):
        self.experts            = experts or {}
        self.synthesizer_expert = synthesizer_expert

    async def aggregate(self, packet: OrchaPacket) -> OrchaPacket:
        t0          = time.perf_counter()
        successful  = packet.successful_results()

        if not successful:
            failed = [r.name for r in packet.get_results() if not r.success]
            return packet.fork(
                PacketKind.AGGREGATION,
                answer="No expert produced a usable response for this query.",
                confidence=0.0,
                contributors=[],
                primary=None,
                synthesized=False,
                agg_mode=AggregationMode.EMPTY,
            ).stamp("aggregate", 0.0, mode="empty", failed=failed)

        if len(successful) == 1:
            return self._single(packet, successful[0], time.perf_counter() - t0)

        # Check divergence to pick the right synthesis prompt
        divergence = _divergence(successful)

        # Try synthesis first (needs synthesizer + ≥2 successful experts)
        can_synthesize = (
            self.synthesizer_expert is not None
            and self.synthesizer_expert in self.experts
        )
        if can_synthesize:
            synth_pkt = await self._try_synthesize(
                packet, successful, divergence, t0
            )
            if synth_pkt is not None:
                return synth_pkt

        # Fall back: vote if 3+ experts, otherwise confidence-weighted
        if len(successful) >= 3:
            return self._vote(packet, successful, time.perf_counter() - t0)
        return self._confidence_weighted(packet, successful, time.perf_counter() - t0)

    # ── Synthesis path ────────────────────────────────────────────────

    async def _try_synthesize(
        self,
        packet:     OrchaPacket,
        successful: List[ExpertResult],
        divergence: float,
        t0:         float,
    ) -> Optional[OrchaPacket]:
        expert = self.experts[self.synthesizer_expert]

        template = _CONFLICT_SYNTHESIS_PROMPT if divergence >= HIGH_DIVERGENCE_THRESHOLD \
                   else _SYNTHESIS_PROMPT

        prompt = template.format(
            query=packet.query,
            candidates=_format_candidates(successful),
        )

        try:
            output = await expert.execute_with_timing(prompt)
        except Exception as exc:
            packet.stamp("aggregate", 0.0, mode="synthesis_failed", error=str(exc))
            return None

        if not output.answer or not output.answer.strip():
            packet.stamp("aggregate", 0.0, mode="synthesis_empty")
            return None

        synth_cost = expert.estimate_cost(output.tokens_used)
        avg_conf   = sum(r.confidence for r in successful) / len(successful)
        final_conf = round(max(output.confidence, avg_conf * 0.9), 4)
        duration_ms = (time.perf_counter() - t0) * 1000

        new_pkt = packet.fork(
            PacketKind.AGGREGATION,
            answer=output.answer,
            confidence=final_conf,
            contributors=[r.name for r in successful],
            primary=self.synthesizer_expert,
            synthesized=True,
            agg_mode=AggregationMode.SYNTHESIS,
            divergence=round(divergence, 3),
            conflict_resolved=divergence >= HIGH_DIVERGENCE_THRESHOLD,
        )
        new_pkt.budget.cost_used      += synth_cost
        new_pkt.budget.latency_used_s += output.latency_s

        return new_pkt.stamp(
            "aggregate", duration_ms,
            mode="synthesis",
            synthesizer=self.synthesizer_expert,
            contributors=len(successful),
            confidence=final_conf,
            divergence=round(divergence, 3),
            conflict=divergence >= HIGH_DIVERGENCE_THRESHOLD,
        )

    # ── Vote path ─────────────────────────────────────────────────────

    def _vote(
        self, packet: OrchaPacket, successful: List[ExpertResult], elapsed: float
    ) -> OrchaPacket:
        """
        Cluster answers by token similarity and pick the largest cluster's
        highest-quality representative. Pre-computes frozensets for O(n*k)
        Jaccard comparisons instead of O(n*k*len(text)).
        """
        # Pre-compute frozensets once
        token_sets = [frozenset(r.output.lower().split()) for r in successful]
        clusters: List[List[int]] = []  # indices into successful
        SIMILARITY_THRESHOLD = 0.35

        for idx in range(len(successful)):
            placed = False
            for cluster in clusters:
                rep_idx = cluster[0]
                if _jaccard_from_sets(token_sets[idx], token_sets[rep_idx]) >= SIMILARITY_THRESHOLD:
                    cluster.append(idx)
                    placed = True
                    break
            if not placed:
                clusters.append([idx])

        # Pick largest cluster (ties broken by total confidence)
        best_cluster_indices = max(
            clusters,
            key=lambda c: (len(c), sum(successful[i].confidence for i in c))
        )
        best = max((successful[i] for i in best_cluster_indices), key=lambda r: r.quality_score())

        best_cluster_set = set(best_cluster_indices)
        minority = [r for idx, r in enumerate(successful) if idx not in best_cluster_set]
        note = ""
        if minority:
            note = f"\n\n[Minority view from: {', '.join(r.name for r in minority[:2])}]"

        avg_conf    = sum(r.confidence for r in successful) / len(successful)
        final_conf  = round((best.confidence * 0.6 + avg_conf * 0.4), 4)
        duration_ms = elapsed * 1000

        return packet.fork(
            PacketKind.AGGREGATION,
            answer=best.output + note,
            confidence=final_conf,
            contributors=[r.name for r in successful],
            primary=best.name,
            synthesized=False,
            agg_mode=AggregationMode.VOTE,
            cluster_size=len(best_cluster_indices),
        ).stamp(
            "aggregate", duration_ms,
            mode="vote",
            primary=best.name,
            clusters=len(clusters),
            cluster_size=len(best_cluster_indices),
            confidence=final_conf,
        )

    # ── Confidence-weighted fallback ──────────────────────────────────

    def _confidence_weighted(
        self, packet: OrchaPacket, successful: List[ExpertResult], elapsed: float
    ) -> OrchaPacket:
        best   = max(successful, key=lambda r: r.quality_score())
        others = [r for r in successful if r.name != best.name]

        avg_conf   = sum(r.confidence for r in successful) / len(successful)
        final_conf = round((best.confidence * 0.70 + avg_conf * 0.30), 4)

        answer = best.output
        if others:
            agree  = ", ".join(f"{r.name} ({r.confidence:.2f})" for r in others[:3])
            answer = f"{answer}\n\n[Cross-checked against: {agree}]"

        duration_ms = elapsed * 1000
        return packet.fork(
            PacketKind.AGGREGATION,
            answer=answer,
            confidence=final_conf,
            contributors=[r.name for r in successful],
            primary=best.name,
            synthesized=False,
            agg_mode=AggregationMode.CONFIDENCE_WEIGHT,
        ).stamp(
            "aggregate", duration_ms,
            mode="confidence_weighted",
            primary=best.name,
            contributors=len(successful),
            confidence=final_conf,
        )

    # ── Single expert path ────────────────────────────────────────────

    def _single(
        self, packet: OrchaPacket, result: ExpertResult, elapsed: float
    ) -> OrchaPacket:
        duration_ms = elapsed * 1000
        return packet.fork(
            PacketKind.AGGREGATION,
            answer=result.output,
            confidence=round(result.confidence, 4),
            contributors=[result.name],
            primary=result.name,
            synthesized=False,
            agg_mode=AggregationMode.SINGLE,
        ).stamp(
            "aggregate", duration_ms,
            mode="single",
            primary=result.name,
            confidence=round(result.confidence, 4),
        )


def get_aggregator(
    experts:            Optional[Dict[str, BaseExpert]] = None,
    synthesizer_expert: Optional[str] = None,
) -> Aggregator:
    return Aggregator(experts=experts, synthesizer_expert=synthesizer_expert)
