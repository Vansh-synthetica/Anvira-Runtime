"""
orcha.nodes.verify
==================
Verification and quality-assurance nodes for the ORCHA3 graph.

These nodes sit in the pipeline as post-aggregation quality gates:

- **VerifyNode**       General-purpose verification: inspects the answer
                        against configurable criteria (completeness, relevance,
                        safety) and produces a pass/fail verdict.
- **CriticNode**       A specialized verifier that scores the answer on a
                        0–1 quality axis and writes quality dimensions into
                        the packet for downstream consumers.
- **FactCheckNode**    Checks factual claims against a ground-truth source
                        (typically a retrieval result). Used when the graph
                        includes a retrieval step.

All verification nodes write their verdict to the packet payload so
conditional edges can route on the outcome (e.g., retry on failure).

Contract
--------
Every verifier reads from the packet's payload (typically ``answer``,
``confidence``, ``results``) and writes back verification-specific keys
(e.g., ``verified``, ``verify_score``, ``quality_dimensions``, ``failures``).
"""
from __future__ import annotations

import re
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.packets import OrchaPacket, PacketKind
from ..graph.context import RunContext
from ..graph.node import Node


# ── Configuration types ────────────────────────────────────────────────────────

@dataclass
class VerifyCriteria:
    """
    Configurable verification criteria.

    Attributes
    ----------
    min_confidence     Minimum confidence threshold (0–1). Default 0.5.
    min_answer_length  Minimum answer character length. Default 20.
    require_answer     If True, an empty/whitespace-only answer is a failure.
    max_danger_score   Safety check: if the answer contains known refusal
                       or harmful patterns above this score, fail. 0–1.
    custom_checks      Optional list of (name, fn) callables that receive
                       the answer string and return a bool (True = pass).
    """
    min_confidence: float = 0.5
    min_answer_length: int = 20
    require_answer: bool = True
    max_danger_score: float = 0.3
    custom_checks: List[tuple] = field(default_factory=list)


@dataclass
class VerifyResult:
    """
    Output of a verification node.

    Attributes
    ----------
    passed      Whether verification passed.
    score       Composite quality score (0–1).
    failures    List of failure reasons (empty if passed).
    dimensions  Per-criterion scores for observability.
    duration_ms Time spent verifying.
    """
    passed: bool
    score: float
    failures: List[str]
    dimensions: Dict[str, float]
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "failures": self.failures,
            "dimensions": self.dimensions,
            "duration_ms": self.duration_ms,
        }


# ── Safety patterns ──────────────────────────────────────────────────────────

# Patterns that indicate a model refusal or potential safety concern.
_REFUSAL_PATTERNS = re.compile(
    r"(?i)(i cannot|i can't|inappropriate|harmful|dangerous|illegal|"
    r"against my (programming|guidelines|policy)|i'm (just|unable|not able))"
)
# Patterns that look like hallucinated disclaimers.
_HALLUCINATION_MARKERS = re.compile(
    r"(?i)(as an AI|as a language model|i am (just|an? ai)|"
    r"i don't (have|know|possess) (personal|real|factual))"
)


# ── VerifyNode ────────────────────────────────────────────────────────────────

class VerifyNode(Node):
    """
    General-purpose verification node.

    Checks the packet's answer against configurable criteria: minimum
    confidence, minimum length, safety patterns, and custom check functions.

    Writes to the packet payload:
      - ``verified`` (bool): overall pass/fail
      - ``verify_score`` (float): composite 0–1 score
      - ``verify_failures`` (list[str]): reasons for failure
      - ``verify_dimensions`` (dict): per-criterion breakdown

    Parameters
    ----------
    name       Node name (default "verify").
    criteria   Verification criteria configuration.
    timeout_s  Per-node timeout.
    retries    Retry budget.
    """

    def __init__(
        self,
        name: str = "verify",
        criteria: Optional[VerifyCriteria] = None,
        timeout_s: Optional[float] = None,
        retries: int = 0,
    ) -> None:
        self.name = name
        self.timeout_s = timeout_s
        self.retries = retries
        self.criteria = criteria or VerifyCriteria()

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        t0 = time.perf_counter()
        answer: str = packet.payload.get("answer", "")
        confidence: float = packet.payload.get("confidence", 0.0)

        failures: List[str] = []
        dimensions: Dict[str, float] = {}

        # ── Confidence check ──────────────────────────────────────────
        conf_pass = confidence >= self.criteria.min_confidence
        dimensions["confidence"] = confidence
        if not conf_pass:
            failures.append(
                f"confidence {confidence:.3f} < {self.criteria.min_confidence}"
            )

        # ── Length check ───────────────────────────────────────────────
        length = len(answer.strip())
        len_pass = length >= self.criteria.min_answer_length
        dimensions["answer_length"] = min(1.0, length / max(1, self.criteria.min_answer_length))
        if self.criteria.require_answer and not len_pass:
            failures.append(
                f"answer length {length} < {self.criteria.min_answer_length}"
            )

        # ── Safety check ───────────────────────────────────────────────
        refusal_matches = len(_REFUSAL_PATTERNS.findall(answer))
        hallucination_matches = len(_HALLUCINATION_MARKERS.findall(answer))
        total_markers = refusal_matches + hallucination_matches
        # Normalize: >3 markers → score approaches 1.0 (bad).
        danger = min(1.0, total_markers / 3.0)
        dimensions["safety"] = 1.0 - danger
        safe = danger <= self.criteria.max_danger_score
        if not safe:
            failures.append(f"safety score {danger:.3f} > {self.criteria.max_danger_score}")

        # ── Custom checks ─────────────────────────────────────────────
        for check_name, check_fn in self.criteria.custom_checks:
            try:
                check_pass = check_fn(answer)
                dimensions[check_name] = 1.0 if check_pass else 0.0
                if not check_pass:
                    failures.append(f"custom check '{check_name}' failed")
            except Exception as exc:
                ctx.logger.warning("verify_custom_check_failed check=%s error=%s", check_name, exc)
                dimensions[check_name] = 0.0
                failures.append(f"custom check '{check_name}' raised: {exc}")

        # ── Composite score ────────────────────────────────────────────
        if dimensions:
            score = sum(dimensions.values()) / len(dimensions)
        else:
            score = 1.0

        passed = len(failures) == 0
        duration_ms = (time.perf_counter() - t0) * 1000

        verify_result = VerifyResult(
            passed=passed, score=score, failures=failures,
            dimensions=dimensions, duration_ms=duration_ms,
        )

        return packet.fork(
            packet.kind,
            verified=passed,
            verify_score=score,
            verify_failures=failures,
            verify_dimensions=dimensions,
        )


# ── CriticNode ────────────────────────────────────────────────────────────────

class CriticNode(Node):
    """
    A critic that scores the answer on multiple quality dimensions.

    This is a more granular version of VerifyNode — it always produces
    a numerical score per dimension regardless of pass/fail, making it
    useful for conditional routing on specific quality signals.

    Writes to the packet payload:
      - ``critic_score`` (float): overall quality score
      - ``critic_dimensions`` (dict): per-dimension breakdown
      - ``critic_passed`` (bool): whether score >= threshold

    Parameters
    ----------
    name         Node name (default "critic").
    threshold    Score threshold for pass (default 0.6).
    timeout_s    Per-node timeout.
    retries      Retry budget.
    """

    DEFAULT_WEIGHTS = {
        "completeness": 0.30,
        "relevance": 0.25,
        "coherence": 0.20,
        "specificity": 0.15,
        "safety": 0.10,
    }

    def __init__(
        self,
        name: str = "critic",
        threshold: float = 0.6,
        weights: Optional[Dict[str, float]] = None,
        timeout_s: Optional[float] = None,
        retries: int = 0,
    ) -> None:
        self.name = name
        self.timeout_s = timeout_s
        self.retries = retries
        self.threshold = threshold
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        t0 = time.perf_counter()
        answer: str = packet.payload.get("answer", "")
        query: str = packet.query
        confidence: float = packet.payload.get("confidence", 0.0)

        dimensions: Dict[str, float] = {}

        # Completeness: token overlap between query and answer.
        query_tokens = set(query.lower().split())
        answer_tokens = set(answer.lower().split())
        if query_tokens:
            dimensions["completeness"] = len(query_tokens & answer_tokens) / len(query_tokens)
        else:
            dimensions["completeness"] = 0.0

        # Relevance: proxy via confidence from aggregation.
        dimensions["relevance"] = min(1.0, confidence)

        # Coherence: penalize excessive self-contradiction markers.
        contradictions = answer.lower().count("however") + answer.lower().count("on the other hand")
        word_count = max(1, len(answer.split()))
        dimensions["coherence"] = max(0.0, 1.0 - contradictions / word_count * 10)

        # Specificity: density of numbers and proper nouns.
        numbers = len(re.findall(r"\b\d+\.?\d*\b", answer))
        proper = len(re.findall(r"\b[A-Z][a-z]{2,}\b", answer))
        dimensions["specificity"] = min(1.0, (numbers + proper) / max(1, word_count) * 5)

        # Safety: inverse of danger markers.
        markers = len(_REFUSAL_PATTERNS.findall(answer)) + len(_HALLUCINATION_MARKERS.findall(answer))
        dimensions["safety"] = max(0.0, 1.0 - markers / 3.0)

        # Weighted composite.
        total_weight = sum(self.weights.get(k, 0.0) for k in dimensions)
        if total_weight > 0:
            score = sum(
                dimensions[k] * self.weights.get(k, 0.0)
                for k in dimensions
                if k in self.weights
            ) / total_weight
        else:
            score = 0.0

        passed = score >= self.threshold
        duration_ms = (time.perf_counter() - t0) * 1000

        return packet.fork(
            packet.kind,
            critic_score=score,
            critic_dimensions=dimensions,
            critic_passed=passed,
            quality_score=score,
        )


# ── FactCheckNode ────────────────────────────────────────────────────────────

class FactCheckNode(Node):
    """
    Checks factual claims in the answer against retrieval results.

    This node expects the packet to carry retrieval results (from a
    ``RetrievalNode`` upstream) under ``payload["retrieval_results"]`` as
    a list of strings. It extracts factual claims from the answer and checks
    whether the retrieval corpus supports them.

    Writes to the packet payload:
      - ``fact_check_passed`` (bool)
      - ``fact_check_score`` (float): fraction of claims supported
      - ``fact_check_details`` (list[dict]): per-claim breakdown
      - ``unsupported_claims`` (list[str]): claims with no support

    Parameters
    ----------
    name       Node name (default "fact_check").
    threshold  Fraction of claims that must be supported (default 0.5).
    timeout_s  Per-node timeout.
    retries    Retry budget.
    """

    # Simple claim extraction: split on sentence boundaries, filter noise.
    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|(?<=\n)")

    def __init__(
        self,
        name: str = "fact_check",
        threshold: float = 0.5,
        timeout_s: Optional[float] = None,
        retries: int = 0,
    ) -> None:
        self.name = name
        self.timeout_s = timeout_s
        self.retries = retries
        self.threshold = threshold

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        answer: str = packet.payload.get("answer", "")
        retrieval_results: List[str] = packet.payload.get("retrieval_results", [])

        if not retrieval_results:
            # No retrieval data: skip fact-checking (pass by default).
            return packet.fork(
                packet.kind,
                fact_check_passed=True,
                fact_check_score=1.0,
                fact_check_details=[],
                unsupported_claims=[],
            )

        # Build a flat corpus from retrieval results.
        corpus = " ".join(retrieval_results).lower()

        # Extract claims (sentences > 10 chars).
        sentences = [s.strip() for s in self._SENTENCE_RE.split(answer) if len(s.strip()) > 10]

        details: List[Dict[str, Any]] = []
        unsupported: List[str] = []
        supported_count = 0

        for claim in sentences:
            # Simple keyword overlap check.
            claim_words = set(claim.lower().split())
            claim_words -= {"the", "a", "an", "is", "are", "was", "were",
                            "it", "this", "that", "and", "or", "but", "in",
                            "on", "at", "to", "for", "of", "with", "as"}
            if not claim_words:
                continue
            overlap = len(claim_words & set(corpus.split()))
            support_ratio = overlap / len(claim_words)
            supported = support_ratio >= 0.3  # 30% keyword overlap threshold
            detail = {
                "claim": claim[:100],
                "supported": supported,
                "overlap_ratio": round(support_ratio, 3),
            }
            details.append(detail)
            if supported:
                supported_count += 1
            else:
                unsupported.append(claim[:100])

        total = len(details)
        score = supported_count / total if total > 0 else 1.0
        passed = score >= self.threshold

        return packet.fork(
            packet.kind,
            fact_check_passed=passed,
            fact_check_score=score,
            fact_check_details=details,
            unsupported_claims=unsupported,
        )


__all__ = [
    "VerifyNode", "CriticNode", "FactCheckNode",
    "VerifyCriteria", "VerifyResult",
]
