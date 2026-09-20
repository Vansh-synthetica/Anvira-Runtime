"""
orcha.evaluation
================
Batch / offline evaluation harness for answers produced by Orcha runs.

This is a regression harness, distinct from the runtime quality gates
(``orcha.orchestration.evaluator`` and the nodes in ``orcha.nodes.verify``).
Those score a single answer *during* a run; this module scores a whole
*corpus of runs* against reference answers and retrieval contexts, so model
and prompt changes can be measured and compared.

Usage
-----
Zero-dependency scoring::

    from orcha.evaluation import EvalSample, LocalJudge, evaluate

    samples = [
        EvalSample(
            query="What is the capital of France?",
            answer="Paris is the capital of France.",
            contexts=["Paris is the capital city of France."],
            expected="Paris",
        ),
    ]
    report = evaluate(samples, LocalJudge())
    print(report.to_dict())

LLM-graded scoring via ragas (optional ``orcha[eval]`` extra)::

    from orcha.evaluation import RagasJudge
    report = evaluate(samples, RagasJudge())

Metric names deliberately mirror the ragas vocabulary so that the
deterministic judge and the LLM judge can be compared on the same axes:
``answer_relevancy``, ``faithfulness``, ``context_precision``,
``context_recall``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

__all__ = [
    "EvalSample", "EvalScore", "Judge", "LocalJudge", "SemanticJudge",
    "RagasJudge", "EvalReport", "evaluate",
]


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """
    One evaluation case.

    Attributes
    ----------
    query      The question asked.
    answer     The answer produced by the system under test.
    contexts   Retrieved context chunks the answer should be grounded in.
    expected   Optional reference / ground-truth answer.
    meta       Optional extra fields carried through to the report.
    """
    query: str
    answer: str
    contexts: List[str] = field(default_factory=list)
    expected: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


EvalScore = Dict[str, float]


class Judge(Protocol):
    """A scorer that produces one float metric per sample."""

    def score(self, sample: EvalSample) -> EvalScore:
        ...


# ── Tokenisation helpers ───────────────────────────────────────────────────────

_STOPWORDS = frozenset(
    "the a an is are was were it this that and or but in on at to for of with as"
    " be been being by from about into over after before between under again "
    "further then once here there when where why how all any both each few more "
    "most other some such no nor not only own same so than too very can will just"
    " should now".split()
)


def _tokens(text: str) -> set:
    return set(re.findall(r"\b\w{2,}\b", text.lower())) - _STOPWORDS


def _overlap(a: set, b: set) -> float:
    if not a:
        return 0.0
    return len(a & b) / len(a)


# ── LocalJudge: deterministic, zero-dependency ────────────────────────────────

class LocalJudge:
    """
    Deterministic heuristic scorer, mirroring the ragas metric vocabulary.

    No network access and no model downloads; safe to run in CI on any
    machine. Useful as a fast smoke gate and as the default judge when the
    LLM-graded ragas backend is not installed.

    Metrics
    -------
    answer_relevancy   Token overlap between the query and the answer,
                       penalised for very short answers.
    faithfulness       Fraction of answer token groups supported by the
                       contexts (when contexts are provided).
    context_precision  Fraction of contexts whose key tokens appear in the
                       answer (when contexts are provided).
    context_recall     Fraction of the expected answer's key tokens found
                       across the contexts (when expected and contexts are
                       provided).
    completeness       Fraction of the expected answer's key tokens found
                       in the answer (when expected is provided).
    """

    def score(self, sample: EvalSample) -> EvalScore:
        query_tokens = _tokens(sample.query)
        answer_tokens = _tokens(sample.answer)
        words = len(sample.answer.split())

        score: EvalScore = {}
        score["answer_relevancy"] = round(self._relevancy(
            query_tokens, answer_tokens, words), 4)

        if sample.contexts:
            score["faithfulness"] = round(self._faithfulness(
                answer_tokens, sample.contexts), 4)
            score["context_precision"] = round(self._context_precision(
                answer_tokens, sample.contexts), 4)
            if sample.expected:
                score["context_recall"] = round(self._context_recall(
                    sample.expected, sample.contexts), 4)

        if sample.expected:
            score["completeness"] = round(_overlap(
                _tokens(sample.expected), answer_tokens), 4)

        return score

    @staticmethod
    def _relevancy(query_tokens: set, answer_tokens: set, words: int) -> float:
        overlap = _overlap(query_tokens, answer_tokens)
        length_bonus = min(words / 40, 0.2)
        return min(1.0, overlap * 0.8 + length_bonus)

    @staticmethod
    def _faithfulness(answer_tokens: set, contexts: List[str]) -> float:
        """Share of the answer's tokens that also occur in the contexts."""
        if not answer_tokens:
            return 0.0
        corpus = set()
        for ctx in contexts:
            corpus |= _tokens(ctx)
        return len(answer_tokens & corpus) / len(answer_tokens)

    @staticmethod
    def _context_precision(answer_tokens: set, contexts: List[str]) -> float:
        """Share of contexts whose key tokens the answer actually used."""
        hits = 0
        for ctx in contexts:
            ctx_tokens = _tokens(ctx)
            if ctx_tokens and (ctx_tokens & answer_tokens):
                hits += 1
        return hits / len(contexts)

    @staticmethod
    def _context_recall(expected: str, contexts: List[str]) -> float:
        """Share of the expected answer's tokens covered by the contexts."""
        expected_tokens = _tokens(expected)
        corpus = set()
        for ctx in contexts:
            corpus |= _tokens(ctx)
        return _overlap(expected_tokens, corpus)


# ── SemanticJudge: optional embedding-based scoring ───────────────────────────

class SemanticJudge:
    """
    Embedding-based judge using sentence-transformers (optional dependency).

    If sentence-transformers is not installed, ``score`` raises ImportError
    with a clear message — use LocalJudge as the portable fallback.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "SemanticJudge requires 'sentence-transformers'. "
                "Install it separately or use LocalJudge."
            ) from None
        self._embedder = SentenceTransformer(model_name)

    def score(self, sample: EvalSample) -> EvalScore:
        import numpy as np
        score: EvalScore = {}
        pairs = [sample.query, sample.answer]
        names = ["answer_relevancy"]
        if sample.contexts:
            pairs.append(" ".join(sample.contexts))
            names.append("context_precision")
        if sample.expected:
            pairs.append(sample.expected)
            names.append("completeness")

        vecs = self._embedder.encode(pairs, normalize_embeddings=True)
        for i, name in enumerate(names):
            sim = float(np.dot(vecs[0], vecs[i + 1]))
            score[name] = round(max(0.0, min(1.0, sim)), 4)
        return score


# ── RagasJudge: LLM-graded scoring via the ragas boundary ─────────────────────
# The judge implementation lives in orcha.integrations.ragas so that
# external frameworks are reachable only through integration boundaries.
# This re-export keeps the public evaluation API stable.

from ..integrations.ragas import RagasJudge

__all__ = [
    "EvalSample", "EvalScore", "Judge", "LocalJudge", "SemanticJudge",
    "RagasJudge", "EvalReport", "evaluate",
]


# ── Batch evaluation ───────────────────────────────────────────────────────────

@dataclass
class EvalReport:
    """
    Aggregate outcome of scoring a corpus of samples.

    Attributes
    ----------
    samples   Per-sample scores, keyed by sample index.
    means     Mean score per metric across all samples.
    """

    samples: List[Dict[str, Any]]
    means: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {"samples": self.samples, "means": self.means}


def evaluate(samples: List[EvalSample], judge: Judge) -> EvalReport:
    """
    Score every sample with the given judge and aggregate the results.

    Parameters
    ----------
    samples  The evaluation corpus.
    judge    Any Judge (LocalJudge, SemanticJudge, RagasJudge, or custom).

    Returns
    -------
    EvalReport with per-sample scores and per-metric means.
    """
    per_sample: List[Dict[str, Any]] = []
    for i, sample in enumerate(samples):
        entry: Dict[str, Any] = {
            "index": i,
            "query": sample.query,
            "scores": judge.score(sample),
        }
        if sample.meta:
            entry["meta"] = sample.meta
        per_sample.append(entry)

    metrics: Dict[str, List[float]] = {}
    for entry in per_sample:
        for name, value in entry["scores"].items():
            metrics.setdefault(name, []).append(float(value))

    means = {name: round(sum(vals) / len(vals), 4)
             for name, vals in metrics.items()}
    return EvalReport(samples=per_sample, means=means)
