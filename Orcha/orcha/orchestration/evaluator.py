"""
orcha.orchestration.evaluator
==============================
Multi-dimensional quality assessment of the aggregated answer.

Dimensions scored
-----------------
COMPLETENESS  — Does the answer actually address the question?
                Heuristic: token overlap between query and answer,
                penalised for very short answers.

COHERENCE     — Is the answer internally consistent?
                Heuristic: checks for explicit self-contradiction markers
                ("however", "but actually", "on the contrary") weighted
                against answer length. A long answer with one "however"
                is fine; a short one with three is a red flag.

CONFIDENCE    — How confident were the contributing experts?
                Direct passthrough of packet.payload["confidence"].

SPECIFICITY   — Does the answer go beyond vague platitudes?
                Heuristic: density of numbers, proper nouns, domain
                keywords relative to total word count.

SAFETY        — Does the answer contain refusal markers that indicate
                the model declined to answer?
                Any refusal marker drops this dimension to 0.0.

Final score
-----------
    score = w_completeness * completeness
          + w_coherence    * coherence
          + w_confidence   * confidence
          + w_specificity  * specificity
          + w_safety       * safety

Default weights sum to 1.0. The score is compared against the
quality_threshold set by the planner. If it falls short, passed=False
and the retry controller will loop again.

Semantic evaluation (optional)
-------------------------------
If sentence-transformers is installed, COMPLETENESS is replaced with a
proper embedding cosine-similarity between the query and the answer,
which is considerably more accurate than the keyword-overlap heuristic.
"""
from __future__ import annotations

import re
import time
from typing import Dict, List, Optional

from ..core.packets import OrchaPacket, PacketKind


# ── Dimension weights ─────────────────────────────────────────────────────────

WEIGHTS: Dict[str, float] = {
    "completeness": 0.30,
    "coherence":    0.15,
    "confidence":   0.30,
    "specificity":  0.15,
    "safety":       0.10,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# ── Safety / refusal signals ──────────────────────────────────────────────────

REFUSAL_PATTERNS: List[str] = [
    r"\bi('m| am) (unable|not able) to\b",
    r"\bi can'?t (help|assist|answer|provide)\b",
    r"\bi (won'?t|will not|cannot) (help|answer|respond)\b",
    r"\bas an ai (language model|assistant|model)\b",
    r"\bthis (request|question|topic) (is|falls) (outside|beyond)\b",
    r"\bplease (consult|speak to|ask) a (professional|doctor|lawyer|expert)\b",
    r"\bi don'?t have (access|information|knowledge) (to|about|of)\b",
]

_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)

# ── Contradiction signals ─────────────────────────────────────────────────────

CONTRADICTION_MARKERS: List[str] = [
    "however,", "but actually,", "on the contrary,", "that said,",
    "conversely,", "in contrast,", "on the other hand,",
    "wait,", "actually,", "to be clear,", "correction:",
]

# ── Specificity signals ───────────────────────────────────────────────────────

_NUMBER_RE   = re.compile(r"\b\d[\d,\.%]*\b")
_PROPER_RE   = re.compile(r"\b[A-Z][a-z]{2,}\b")   # crude proper-noun detector
_WORD_RE     = re.compile(r"\b\w{4,}\b")            # 4+ char words for completeness

# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Multi-dimensional quality gate. Scores the aggregated answer and
    decides whether it is good enough to return or needs another iteration.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        use_embeddings: bool = False,
    ):
        """
        Parameters
        ----------
        weights        Optional override of the per-dimension weights.
        use_embeddings  Only attempt to load a sentence-transformers model
                        when True. The zero-dependency fast path
                        (use_embeddings=False) is the default and must NEVER
                        trigger a network download or model load, mirroring
                        the SmartDecomposer contract.
        """
        self.weights = weights or WEIGHTS
        self._embedder = None
        self._np = None
        if not use_embeddings:
            return
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self._np = np
        except ImportError:
            # Optional dependency not installed — fall back to the
            # keyword-overlap heuristic silently.
            pass

    def evaluate(self, packet: OrchaPacket) -> OrchaPacket:
        t0                = time.perf_counter()
        query             = packet.query
        answer            = packet.payload.get("answer", "")
        confidence        = float(packet.payload.get("confidence", 0.0))
        quality_threshold = float(packet.payload.get("quality_threshold", 0.70))

        if not answer or not answer.strip():
            return packet.fork(
                PacketKind.EVALUATION,
                passed=False,
                confidence=0.0,
                quality_score=0.0,
                quality_dimensions={"completeness": 0, "coherence": 0,
                                    "confidence": 0, "specificity": 0, "safety": 0},
                fail_reason="empty_answer",
                threshold=quality_threshold,
            ).stamp("evaluate", 0.0, passed=False, score=0.0, reason="empty_answer")

        dims = self._score_dimensions(query, answer, confidence)
        score = sum(self.weights[k] * dims[k] for k in self.weights)
        score = round(score, 4)

        passed     = score >= quality_threshold
        fail_reason = None if passed else self._primary_fail_reason(dims, quality_threshold)

        duration_ms = (time.perf_counter() - t0) * 1000
        return packet.fork(
            PacketKind.EVALUATION,
            passed=passed,
            confidence=confidence,
            quality_score=score,
            quality_dimensions={k: round(v, 3) for k, v in dims.items()},
            fail_reason=fail_reason,
            threshold=quality_threshold,
        ).stamp(
            "evaluate", duration_ms,
            passed=passed,
            score=score,
            threshold=quality_threshold,
            dims={k: round(v, 2) for k, v in dims.items()},
            reason=fail_reason,
        )

    # ── Dimension scorers ─────────────────────────────────────────────

    def _score_dimensions(
        self, query: str, answer: str, confidence: float
    ) -> Dict[str, float]:
        return {
            "completeness": self._completeness(query, answer),
            "coherence":    self._coherence(answer),
            "confidence":   min(1.0, max(0.0, confidence)),
            "specificity":  self._specificity(answer),
            "safety":       self._safety(answer),
        }

    def _completeness(self, query: str, answer: str) -> float:
        """
        How well does the answer address the query?
        Uses semantic similarity when embeddings available, token overlap otherwise.
        """
        if self._embedder is not None:
            try:
                vecs = self._embedder.encode([query, answer], normalize_embeddings=True)
                sim  = float(self._np.dot(vecs[0], vecs[1]))
                return max(0.0, min(1.0, sim))
            except Exception:
                pass

        # Keyword overlap fallback — use pre-compiled regex
        q_tokens = set(_WORD_RE.findall(query.lower()))
        a_tokens = set(_WORD_RE.findall(answer.lower()))
        if not q_tokens:
            return 0.5
        overlap = len(q_tokens & a_tokens) / len(q_tokens)

        # Penalise very short answers
        word_count   = len(answer.split())
        length_bonus = min(word_count / 80, 0.2)
        return min(1.0, overlap * 0.8 + length_bonus)

    def _coherence(self, answer: str) -> float:
        """
        Lower score for answers with many self-contradiction markers relative
        to their length.
        """
        text       = answer.lower()
        word_count = max(len(answer.split()), 1)
        hits       = sum(1 for m in CONTRADICTION_MARKERS if m in text)

        # One contradiction per 100 words is expected; more than that hurts
        expected = word_count / 100
        penalty  = max(0, hits - expected) * 0.15
        return max(0.0, 1.0 - penalty)

    def _specificity(self, answer: str) -> float:
        """
        Answers full of concrete numbers and named entities are more specific
        than those composed entirely of vague generalities.
        """
        words     = answer.split()
        n_words   = max(len(words), 1)
        n_numbers = len(_NUMBER_RE.findall(answer))
        n_proper  = len(_PROPER_RE.findall(answer))
        density   = (n_numbers * 2 + n_proper) / n_words
        return min(1.0, density * 5)    # scale so 20% specific tokens = 1.0

    def _safety(self, answer: str) -> float:
        """
        Returns 0.0 if the answer contains a refusal pattern, 1.0 otherwise.
        """
        return 0.0 if _REFUSAL_RE.search(answer) else 1.0

    def _primary_fail_reason(
        self, dims: Dict[str, float], threshold: float
    ) -> str:
        """Return the dimension that contributed most to the failure."""
        weighted = {k: self.weights[k] * (1.0 - dims[k]) for k in self.weights}
        return min(weighted, key=lambda k: -weighted[k])


def get_evaluator(
    weights: Optional[Dict[str, float]] = None,
    use_embeddings: bool = False,
) -> Evaluator:
    return Evaluator(weights=weights, use_embeddings=use_embeddings)
