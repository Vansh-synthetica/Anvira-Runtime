"""
orcha.integrations.ragas
========================
Boundary around ragas — the LLM-graded batch / regression evaluation
backend (Apache-2.0), loaded lazily.

``RagasJudge`` lives HERE, behind the boundary, and is re-exported by
``orcha.evaluation`` so the public evaluation API stays stable. The
judge layer in ``orcha.evaluation`` (LocalJudge, SemanticJudge) is
deterministic and dependency-free; ragas is the optional LLM-graded
backend installed via the ``orcha[eval]`` extra.

Local-first: by default ragas calls the OpenAI API. To run fully
local, point the underlying LLM at any OpenAI-compatible endpoint
(e.g. Ollama / LM Studio) via an ``OPENAI_API_BASE`` env var or an
explicit ``llm`` kwarg. A local Anvira installation must be able to
evaluate local models — this boundary never requires cloud.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..base import Boundary, IntegrationUnavailable

if TYPE_CHECKING:
    from ...evaluation import EvalScore

__all__ = ["RagasBoundary", "RagasJudge"]

_RAGAS_DEFAULT_METRICS = ("answer_relevancy", "faithfulness",
                          "context_precision", "context_recall")


class RagasBoundary(Boundary):
    name = "ragas"
    package = "ragas"
    extra = "eval"

    def load(self) -> object:
        """Return the ``ragas`` module (raises if not installed)."""
        return self._import("ragas")


class RagasJudge:
    """
    LLM-graded judge backed by ragas, loaded lazily via the boundary.

    Requires the optional ``orcha[eval]`` extra::

        pip install "orcha[eval]"

    By default ragas calls the OpenAI API. To run fully local, point the
    underlying LLM at any OpenAI-compatible endpoint, e.g. with litellm
    or an env-based ``OPENAI_API_BASE`` pointing at Ollama / LM Studio.

    Parameters
    ----------
    metrics   Ragas metric instances. Defaults to the four metrics whose
              name mirrors LocalJudge (answer_relevancy, faithfulness,
              context_precision, context_recall).
    kwargs    Extra keyword arguments forwarded to ``ragas.evaluate``
              (e.g. ``llm``, ``embeddings``, ``raise_exceptions``).
    """

    def __init__(self, metrics: Optional[List[Any]] = None, **kwargs: Any) -> None:
        try:
            from ragas import metrics as _ragas_metrics  # noqa: F401
        except ImportError:
            raise ImportError(
                "RagasJudge requires ragas. Install with "
                "`pip install \"orcha[eval]\"` or use LocalJudge."
            ) from None
        self._metrics = list(metrics or [])
        self._kwargs = kwargs

    def score(self, sample: Any) -> Dict[str, float]:
        from ragas import evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import (
            answer_relevancy, context_precision, context_recall, faithfulness,
        )

        metric_map = {
            "answer_relevancy": answer_relevancy,
            "faithfulness": faithfulness,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        metrics = self._metrics or [metric_map[m] for m in _RAGAS_DEFAULT_METRICS]

        sample_kwargs: Dict[str, Any] = {
            "user_input": sample.query,
            "response": sample.answer,
        }
        if sample.contexts:
            sample_kwargs["retrieved_contexts"] = sample.contexts
        if sample.expected:
            sample_kwargs["reference"] = sample.expected

        result = evaluate(
            dataset=[SingleTurnSample(**sample_kwargs)],
            metrics=metrics,
            **self._kwargs,
        )
        return {name: float(result.scores[0][name])
                for name in result.scores[0] if name in _RAGAS_DEFAULT_METRICS}