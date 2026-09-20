"""Unit tests for every orchestration stage in isolation."""
import pytest

from orcha.core.packets import OrchaPacket, PacketKind, BudgetState, ExpertResult
from orcha.experts.mock import load_mock_experts
from orcha.orchestration import (
    get_decomposer, get_planner, get_selector, SelectionStrategy,
    get_executor, get_aggregator, get_evaluator, get_retry_controller,
)


def make_packet(query="Analyse my investment portfolio risk", **payload):
    return OrchaPacket(kind=PacketKind.QUERY, query=query, payload=payload)


# ── Decomposer ──────────────────────────────────────────────────────────────

def test_decomposer_detects_finance_domain():
    d = get_decomposer(use_embeddings=False)
    out = d.decompose(make_packet("How should I manage portfolio risk and diversify investments?"))
    assert out.kind == PacketKind.SUBTASKS
    assert "finance" in out.payload["domains"]
    assert len(out.payload["subtasks"]) > 0

def test_decomposer_detects_code_domain():
    d = get_decomposer(use_embeddings=False)
    out = d.decompose(make_packet("Why is my async Python function not running in parallel?"))
    assert "code" in out.payload["domains"]

def test_decomposer_detects_reasoning_domain():
    d = get_decomposer(use_embeddings=False)
    out = d.decompose(make_packet("Why does this logical argument fail?"))
    assert "reasoning" in out.payload["domains"]

def test_decomposer_defaults_to_general():
    d = get_decomposer(use_embeddings=False)
    out = d.decompose(make_packet("hello there"))
    assert out.payload["domains"] == ["general"]

def test_decomposer_caps_subtasks():
    d = get_decomposer(use_embeddings=False)
    out = d.decompose(make_packet(
        "Analyse the market risk of this trading algorithm code, explain "
        "the scientific research methodology and creative writing needed"
    ))
    assert len(out.payload["subtasks"]) <= 10   # capped at max_tasks

def test_decomposer_complexity_estimate():
    d = get_decomposer(use_embeddings=False)
    simple  = d.decompose(make_packet("What is Python?"))
    complex_ = d.decompose(make_packet(
        "Compare and contrast the risk-adjusted returns of small-cap vs "
        "large-cap equity allocations in a rising-rate environment, "
        "considering correlation, drawdown, and Sharpe ratio implications. "
        "Why do some portfolio managers prefer factor-based approaches? "
        "What are the counter-arguments to this view?"
    ))
    assert complex_.payload["complexity"] >= simple.payload["complexity"]

def test_decomposer_query_type_classification():
    d = get_decomposer(use_embeddings=False)
    out = d.decompose(make_packet("How do I implement a binary search tree?"))
    assert out.payload["query_type"] == "how_to"

    out2 = d.decompose(make_packet("What is machine learning?"))
    assert out2.payload["query_type"] == "definition"


# ── Planner ─────────────────────────────────────────────────────────────────

def test_planner_stops_when_budget_exhausted():
    p = get_planner()
    pkt = make_packet(domains=["general"])
    pkt.budget = BudgetState(max_iterations=1, iterations=1)
    out = p.plan(pkt)
    assert out.payload["stop"] is True
    assert out.payload["reason"] == "budget_exhausted"

def test_planner_stops_on_excellent_confidence():
    p = get_planner()
    out = p.plan(make_packet(domains=["general"], confidence=0.95))
    assert out.payload["stop"] is True
    assert out.payload["reason"] == "quality_excellent"

def test_planner_first_iteration_moderate_width():
    p = get_planner()
    pkt = make_packet(domains=["general"], complexity=0.3)
    pkt.budget = BudgetState(max_iterations=3, iterations=0)
    out = p.plan(pkt)
    assert out.payload["stop"] is False
    assert 2 <= out.payload["parallel_width"] <= 10

def test_planner_scales_width_on_retry():
    p = get_planner()
    p0 = make_packet(domains=["general"], complexity=0.5)
    p0.budget = BudgetState(max_iterations=4, iterations=0)
    p1 = make_packet(domains=["general"], complexity=0.5)
    p1.budget = BudgetState(max_iterations=4, iterations=2)
    out0 = p.plan(p0)
    out1 = p.plan(p1)
    assert out1.payload["parallel_width"] >= out0.payload["parallel_width"]
    assert out1.payload["quality_threshold"] >= out0.payload["quality_threshold"]

def test_planner_high_complexity_gets_extra_width():
    p = get_planner()
    lo = make_packet(domains=["general"], complexity=0.1)
    hi = make_packet(domains=["general"], complexity=0.9)
    lo.budget = hi.budget = BudgetState(max_iterations=3, iterations=0)
    out_lo = p.plan(lo)
    out_hi = p.plan(hi)
    assert out_hi.payload["parallel_width"] >= out_lo.payload["parallel_width"]

def test_planner_reasoning_fast_is_single_expert_one_pass():
    p = get_planner()
    pkt = make_packet(domains=["general"], complexity=0.9, reasoning="fast")
    pkt.budget = BudgetState(max_iterations=3, iterations=0)
    out = p.plan(pkt)
    assert out.payload["parallel_width"] == 1
    assert out.payload["quality_threshold"] == 0.50

def test_planner_reasoning_max_fans_out_and_is_stricter():
    p = get_planner()
    pkt = make_packet(domains=["general"], complexity=0.1, reasoning="max")
    pkt.budget = BudgetState(max_iterations=5, iterations=0)
    out = p.plan(pkt)
    assert out.payload["parallel_width"] == 5
    assert out.payload["quality_threshold"] == 0.80

def test_planner_reasoning_medium_uses_default_logic():
    p = get_planner()
    default = p.plan(make_packet(domains=["general"], complexity=0.3))
    med = p.plan(make_packet(domains=["general"], complexity=0.3, reasoning="medium"))
    assert default.payload["parallel_width"] == med.payload["parallel_width"]
    assert default.payload["quality_threshold"] == med.payload["quality_threshold"]

def test_planner_ignores_unknown_reasoning_value():
    p = get_planner()
    out = p.plan(make_packet(domains=["general"], complexity=0.3, reasoning="bogus"))
    assert out.payload["stop"] is False
    assert 2 <= out.payload["parallel_width"] <= 10


# ── Selector ────────────────────────────────────────────────────────────────

def test_selector_prefers_domain_matching_experts():
    sel = get_selector(load_mock_experts())
    out = sel.select(make_packet("portfolio risk", domains=["finance"], parallel_width=2))
    names = [s["name"] for s in out.payload["selected_experts"]]
    assert "finance_specialist" in names
    assert len(names) <= 2

def test_selector_handles_empty_registry():
    sel = get_selector({})
    out = sel.select(make_packet(domains=["general"], parallel_width=3))
    assert out.payload["selected_experts"] == []
    assert "warning" in out.trace[-1].data

def test_selector_force_all_returns_everyone():
    experts = load_mock_experts()
    sel = get_selector(experts)
    out = sel.select(make_packet(domains=["general"], parallel_width=1, force_all_experts=True))
    assert len(out.payload["selected_experts"]) == len(experts)

def test_selector_respects_excluded_experts():
    experts = load_mock_experts()
    sel = get_selector(experts)
    out = sel.select(make_packet(
        domains=["general"], parallel_width=5,
        excluded_experts=["finance_specialist", "code_specialist"]
    ))
    names = [s["name"] for s in out.payload["selected_experts"]]
    assert "finance_specialist" not in names
    assert "code_specialist" not in names

def test_selector_records_performance():
    sel = get_selector(load_mock_experts())
    sel.record_result("fast_generalist", success=True,  confidence=0.8)
    sel.record_result("fast_generalist", success=False, confidence=0.0)
    perf = sel.performance_summary()
    assert perf["fast_generalist"]["successes"] == 1
    assert perf["fast_generalist"]["failures"]  == 1


# ── Executor ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_executor_runs_selected_experts_in_parallel():
    experts = load_mock_experts()
    exe = get_executor(experts)
    selected = [
        {"name": "fast_generalist", "domain": "general", "description": "x", "score": 0.5},
        {"name": "code_specialist",  "domain": "code",    "description": "x", "score": 0.5},
    ]
    out = await exe.execute(make_packet(selected_experts=selected))
    names = {r["name"] for r in out.payload["results"]}
    assert names == {"fast_generalist", "code_specialist"}
    assert out.budget.iterations == 1

@pytest.mark.asyncio
async def test_executor_handles_no_experts_without_infinite_loop():
    exe = get_executor({})
    out = await exe.execute(make_packet(selected_experts=[]))
    assert out.payload["results"] == []
    assert out.budget.iterations == 1

@pytest.mark.asyncio
async def test_executor_isolates_failing_expert():
    from orcha.experts.base import BaseExpert, ExpertOutput

    class BrokenExpert(BaseExpert):
        name = "broken"; domain = "general"; description = "always fails"
        async def execute(self, query):
            raise RuntimeError("intentional failure")

    class GoodExpert(BaseExpert):
        name = "good"; domain = "general"; description = "always works"
        async def execute(self, query):
            return ExpertOutput(answer="great answer", confidence=0.9, tokens_used=10)

    exe = get_executor({"broken": BrokenExpert(), "good": GoodExpert()})
    selected = [
        {"name": "broken", "domain": "general", "description": "x", "score": 0.5},
        {"name": "good",   "domain": "general", "description": "x", "score": 0.5},
    ]
    out = await exe.execute(make_packet(selected_experts=selected))
    results = {r["name"]: r for r in out.payload["results"]}
    assert results["broken"]["success"] is False
    assert "intentional failure" in results["broken"]["error"]
    assert results["good"]["success"] is True


# ── Aggregator ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_aggregator_picks_highest_confidence():
    agg = get_aggregator()
    results = [
        ExpertResult(name="a", output="answer A is complete", confidence=0.6, success=True),
        ExpertResult(name="b", output="answer B is thorough", confidence=0.9, success=True),
    ]
    out = await agg.aggregate(make_packet(results=[r.model_dump() for r in results]))
    assert out.payload["primary"] == "b"
    assert "answer B" in out.payload["answer"]
    assert 0.6 < out.payload["confidence"] <= 0.9

@pytest.mark.asyncio
async def test_aggregator_handles_all_failures():
    agg = get_aggregator()
    results = [ExpertResult(name="a", output="", confidence=0.0, success=False, error="timeout")]
    out = await agg.aggregate(make_packet(results=[r.model_dump() for r in results]))
    assert out.payload["confidence"] == 0.0
    assert out.payload["contributors"] == []

@pytest.mark.asyncio
async def test_aggregator_synthesis_with_mock_synthesizer():
    from orcha.experts.mock import MockSynthesizer
    synth = MockSynthesizer()
    agg = get_aggregator(
        experts={"mock_synthesizer": synth},
        synthesizer_expert="mock_synthesizer",
    )
    results = [
        ExpertResult(name="a", output="First expert says X is the answer", confidence=0.7, success=True),
        ExpertResult(name="b", output="Second expert says Y is the answer", confidence=0.8, success=True),
    ]
    out = await agg.aggregate(make_packet(results=[r.model_dump() for r in results]))
    assert out.payload["synthesized"] is True
    assert out.payload["primary"] == "mock_synthesizer"

@pytest.mark.asyncio
async def test_aggregator_vote_mode_with_three_experts():
    agg = get_aggregator()
    results = [
        ExpertResult(name="a", output="consensus answer about the topic", confidence=0.7, success=True),
        ExpertResult(name="b", output="consensus answer about the topic", confidence=0.75, success=True),
        ExpertResult(name="c", output="completely different minority view", confidence=0.6, success=True),
    ]
    out = await agg.aggregate(make_packet(results=[r.model_dump() for r in results]))
    assert "consensus" in out.payload["answer"]


# ── Evaluator ───────────────────────────────────────────────────────────────

def test_evaluator_passes_good_answer():
    ev = get_evaluator()
    # A realistic, specific answer that hits multiple quality dimensions
    good_answer = (
        "Portfolio diversification involves spreading investments across multiple "
        "asset classes — equities, bonds, real estate, and commodities — to reduce "
        "unsystematic risk. Academic research shows that holding 20-30 uncorrelated "
        "assets captures roughly 95% of diversification benefits. A Sharpe ratio "
        "above 1.0 indicates good risk-adjusted return. The key principle is that "
        "correlation between assets should be low or negative."
    )
    out = ev.evaluate(make_packet(answer=good_answer, confidence=0.85, quality_threshold=0.65))
    assert out.payload["passed"] is True
    assert "quality_dimensions" in out.payload

def test_evaluator_fails_empty_answer():
    ev = get_evaluator()
    out = ev.evaluate(make_packet(answer="", confidence=0.99, quality_threshold=0.70))
    assert out.payload["passed"] is False
    assert out.payload["fail_reason"] == "empty_answer"

def test_evaluator_fails_low_confidence():
    ev = get_evaluator()
    out = ev.evaluate(make_packet(answer="something vague", confidence=0.10, quality_threshold=0.70))
    assert out.payload["passed"] is False

def test_evaluator_detects_refusal():
    ev = get_evaluator()
    refusal = "I'm unable to help with that request as it falls outside my capabilities."
    out = ev.evaluate(make_packet(answer=refusal, confidence=0.9, quality_threshold=0.60))
    assert out.payload["quality_dimensions"]["safety"] == 0.0

def test_evaluator_quality_dimensions_all_present():
    ev = get_evaluator()
    out = ev.evaluate(make_packet(answer="some answer text here", confidence=0.7, quality_threshold=0.5))
    dims = out.payload["quality_dimensions"]
    for k in ("completeness", "coherence", "confidence", "specificity", "safety"):
        assert k in dims
        assert 0.0 <= dims[k] <= 1.0


# ── Retry controller ────────────────────────────────────────────────────────

def test_retry_on_fail_with_budget():
    rc = get_retry_controller()
    pkt = make_packet(passed=False)
    pkt.budget = BudgetState(max_iterations=3, iterations=1)
    assert rc.decide(pkt).payload["retry"] is True

def test_no_retry_when_passed():
    rc = get_retry_controller()
    pkt = make_packet(passed=True)
    pkt.budget = BudgetState(max_iterations=3, iterations=1)
    assert rc.decide(pkt).payload["retry"] is False

def test_no_retry_when_budget_exhausted():
    rc = get_retry_controller()
    pkt = make_packet(passed=False)
    pkt.budget = BudgetState(max_iterations=2, iterations=2)
    assert rc.decide(pkt).payload["retry"] is False

def test_retry_excludes_failed_experts():
    rc = get_retry_controller()
    results = [
        ExpertResult(name="bad_expert", output="", confidence=0.0, success=False, error="timeout"),
        ExpertResult(name="good_expert", output="ok", confidence=0.85, success=True),
    ]
    pkt = make_packet(passed=False, results=[r.model_dump() for r in results])
    pkt.budget = BudgetState(max_iterations=3, iterations=1)
    out = rc.decide(pkt)
    assert "bad_expert" in out.payload["excluded_experts"]
    assert "good_expert" not in out.payload["excluded_experts"]


# ── Regression: packet isolation via deep-copy ──────────────────────────────

def test_fork_deep_copies_payload_lists():
    """A forked packet must not share mutable payload list references with
    its parent. Mutating a list in the child must leave the parent intact."""
    parent = OrchaPacket(
        kind=PacketKind.QUERY,
        query="test",
        payload={"items": [1, 2, 3], "excluded_experts": ["a"]},
    )
    child = parent.fork(PacketKind.SUBTASKS)
    assert child.payload["items"] == [1, 2, 3]
    # Mutate the child's list
    child.payload["items"].append(4)
    child.payload["excluded_experts"].append("b")
    # Parent must be untouched
    assert parent.payload["items"] == [1, 2, 3]
    assert parent.payload["excluded_experts"] == ["a"]
    assert child.payload["items"] == [1, 2, 3, 4]


def test_fork_deep_copies_payload_dicts():
    """Nested dicts in payload are also deep-copied."""
    parent = OrchaPacket(
        kind=PacketKind.QUERY,
        query="test",
        payload={"nested": {"key": "original"}},
    )
    child = parent.fork(PacketKind.SUBTASKS)
    child.payload["nested"]["key"] = "mutated"
    assert parent.payload["nested"]["key"] == "original"
    assert child.payload["nested"]["key"] == "mutated"


# ── Regression: weighted-random selection strategy ──────────────────────────

def test_selector_weighted_random_returns_expected_count():
    sel = get_selector(
        load_mock_experts(),
        strategy=SelectionStrategy.WEIGHTED_RANDOM,
        seed=42,  # reproducible
    )
    out = sel.select(make_packet(
        domains=["general"], parallel_width=3,
    ))
    selected = out.payload["selected_experts"]
    assert len(selected) == 3
    # Must be actual experts from the pool
    names = {s["name"] for s in selected}
    assert names.issubset(set(load_mock_experts().keys()))


def test_selector_weighted_random_respects_width():
    sel = get_selector(
        load_mock_experts(),
        strategy=SelectionStrategy.WEIGHTED_RANDOM,
        seed=42,
    )
    out = sel.select(make_packet(
        domains=["general"], parallel_width=1,
    ))
    assert len(out.payload["selected_experts"]) == 1


def test_selector_weighted_random_mode_in_trace():
    sel = get_selector(
        load_mock_experts(),
        strategy=SelectionStrategy.WEIGHTED_RANDOM,
    )
    out = sel.select(make_packet(
        domains=["general"], parallel_width=2,
    ))
    assert out.trace[-1].data.get("mode") == "weighted_random"


# ── Regression: evaluator does not load embeddings by default ──────────────

def test_evaluator_no_embedding_load_without_flag():
    """Evaluator(use_embeddings=False) must never trigger a model download."""
    ev = get_evaluator(use_embeddings=False)
    assert ev._embedder is None
    assert ev._np is None


# ── Regression: decomposer mirrors the same contract ───────────────────────

def test_decomposer_no_embedding_load_without_flag():
    d = get_decomposer(use_embeddings=False)
    assert d._embedder is None
