"""Unit tests for orcha.evaluation — the batch evaluation harness."""
import pytest

from orcha.evaluation import (
    EvalSample, LocalJudge, RagasJudge, SemanticJudge, evaluate,
)


def make_sample(**kw):
    base = dict(
        query="What is the capital of France?",
        answer="Paris is the capital of France.",
        contexts=["Paris is the capital city of France."],
        expected="Paris",
    )
    base.update(kw)
    return EvalSample(**base)


# ── LocalJudge ────────────────────────────────────────────────────────────────

def test_local_judge_scores_grounded_answer():
    score = LocalJudge().score(make_sample())
    assert score["answer_relevancy"] > 0.5
    assert score["faithfulness"] > 0.5
    assert score["context_precision"] == 1.0
    assert score["context_recall"] == 1.0
    assert score["completeness"] > 0.0

def test_local_judge_punishes_off_topic_answer():
    good = LocalJudge().score(make_sample())
    bad = LocalJudge().score(make_sample(
        answer="The Eiffel Tower is a wrought-iron lattice tower.",
    ))
    assert bad["answer_relevancy"] < good["answer_relevancy"]

def test_local_judge_empty_answer_scores_zero():
    score = LocalJudge().score(make_sample(answer=""))
    assert score["answer_relevancy"] == 0.0
    assert score["faithfulness"] == 0.0

def test_local_judge_omits_context_metrics_without_contexts():
    score = LocalJudge().score(make_sample(contexts=[]))
    assert "faithfulness" not in score
    assert "context_precision" not in score
    assert "context_recall" not in score

def test_local_judge_no_expected_omits_recall_and_completeness():
    score = LocalJudge().score(make_sample(expected=None))
    assert "context_recall" not in score
    assert "completeness" not in score

def test_local_judge_faithfulness_drops_with_irrelevant_contexts():
    grounded = LocalJudge().score(make_sample())
    ungrounded = LocalJudge().score(make_sample(
        answer="Paris is the capital of France.",
        contexts=["Quantum computing uses qubits for computation."],
    ))
    assert ungrounded["faithfulness"] < grounded["faithfulness"]

def test_local_judge_never_exceeds_one():
    score = LocalJudge().score(make_sample())
    for value in score.values():
        assert 0.0 <= value <= 1.0


# ── evaluate ──────────────────────────────────────────────────────────────────

def test_evaluate_aggregates_means():
    report = evaluate(
        [
            make_sample(),
            make_sample(query="Where is the Sahara desert?",
                        answer="The Sahara is in North Africa.",
                        contexts=["The Sahara desert spans North Africa."],
                        expected="North Africa"),
        ],
        LocalJudge(),
    )
    assert len(report.samples) == 2
    assert "answer_relevancy" in report.means
    assert 0.0 <= report.means["answer_relevancy"] <= 1.0
    assert report.to_dict()["means"] == report.means

def test_evaluate_preserves_meta():
    report = evaluate(
        [make_sample(meta={"model": "stub-a"})],
        LocalJudge(),
    )
    assert report.samples[0]["meta"] == {"model": "stub-a"}

def test_evaluate_empty_corpus():
    report = evaluate([], LocalJudge())
    assert report.samples == []
    assert report.means == {}


# ── Optional judges ───────────────────────────────────────────────────────────

def test_ragas_judge_raises_without_ragas():
    try:
        import ragas  # noqa: F401
        pytest.skip("ragas installed; ImportError path untestable here")
    except ImportError:
        with pytest.raises(ImportError, match="orcha\\[eval\\]"):
            RagasJudge()

def test_semantic_judge_raises_without_sentence_transformers():
    try:
        import sentence_transformers  # noqa: F401
        pytest.skip("sentence-transformers installed; ImportError path untestable here")
    except ImportError:
        with pytest.raises(ImportError, match="sentence-transformers"):
            SemanticJudge()
