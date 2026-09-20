"""
orcha.experts.mock
==================
Zero-dependency mock experts for demos, tests, and offline development.

Each expert has a distinct persona, domain, latency profile, confidence
distribution, and vocabulary — so the selector, aggregator, and evaluator
all get realistic variation to work with.

Included experts
----------------
FastGeneralistExpert     — general, fast, moderate confidence
DeepReasoningExpert      — reasoning, slow, high confidence
FinanceExpert            — finance, medium speed, domain-specific vocabulary
CodeExpert               — code, medium speed, technical vocabulary
ScienceExpert            — science, slow, methodical
SkepticalExpert          — general, always raises caveats and counter-arguments
CreativeExpert           — creative, fast, expressive
MockSynthesizer          — reads multi-expert synthesis prompts and combines them

All experts simulate realistic latency using asyncio.sleep() with
jitter so the executor's parallel timing is representative.
"""
from __future__ import annotations

import asyncio
import random
import re
from typing import Dict, List

from .base import BaseExpert, ExpertOutput


# ── Shared utilities ──────────────────────────────────────────────────────────

def _snip(query: str, n: int = 60) -> str:
    """Return first n chars of query, word-truncated."""
    return query[:n].rsplit(" ", 1)[0] + "…" if len(query) > n else query


async def _jitter(lo: float, hi: float) -> None:
    await asyncio.sleep(random.uniform(lo, hi))


# ── Expert implementations ────────────────────────────────────────────────────

class FastGeneralistExpert(BaseExpert):
    name               = "fast_generalist"
    domain             = "general"
    description        = "Fast general-purpose reasoning. Low latency, moderate accuracy."
    cost_per_1k_tokens = 0.005

    async def execute(self, query: str) -> ExpertOutput:
        await _jitter(0.05, 0.30)
        q = _snip(query)
        answer = (
            f"A quick assessment of '{q}': this question involves several "
            "interconnected factors. The most pragmatic starting point is to "
            "identify the core constraint, then work outward. For most cases, "
            "a direct, iterative approach outperforms over-engineering. "
            "Verify assumptions early and adjust based on feedback."
        )
        return ExpertOutput(
            answer=answer,
            confidence=random.uniform(0.52, 0.70),
            tokens_used=random.randint(55, 130),
            finish_reason="stop",
        )


class DeepReasoningExpert(BaseExpert):
    name               = "deep_reasoning"
    domain             = "reasoning"
    description        = "Slow, thorough analytical reasoning. High accuracy for complex questions."
    cost_per_1k_tokens = 0.025

    async def execute(self, query: str) -> ExpertOutput:
        await _jitter(0.70, 1.50)
        q = _snip(query)
        answer = (
            f"Examining '{q}' from first principles: the question presupposes "
            "certain conditions that are worth surfacing. First, we must distinguish "
            "between the empirical claim and the normative one embedded in the "
            "framing. The empirical evidence is mixed — studies on this class of "
            "problem show high variance depending on context. The logical structure "
            "suggests that if premise A holds and B is consistent with domain "
            "knowledge, then the most defensible conclusion is C — though this "
            "requires accepting that the initial framing is not underdetermined. "
            "A rigorous analysis would also consider the strongest counter-argument: "
            "that the conditions are rarely met in practice, which weakens the "
            "general claim considerably."
        )
        return ExpertOutput(
            answer=answer,
            confidence=random.uniform(0.78, 0.93),
            tokens_used=random.randint(200, 480),
            finish_reason="stop",
        )


class FinanceExpert(BaseExpert):
    name               = "finance_specialist"
    domain             = "finance"
    description        = "Financial analyst: markets, risk, valuation, portfolio construction."
    cost_per_1k_tokens = 0.035

    _MARKET_PHRASES = [
        "risk-adjusted returns", "Sharpe ratio", "drawdown exposure",
        "correlation with the broader index", "beta-adjusted position sizing",
        "liquidity constraints", "convexity of the payoff profile",
        "mean reversion tendencies", "macro headwinds",
    ]

    async def execute(self, query: str) -> ExpertOutput:
        await _jitter(0.35, 0.90)
        q      = _snip(query)
        phrase = random.choice(self._MARKET_PHRASES)
        answer = (
            f"From a financial perspective on '{q}': the critical variable here "
            f"is {phrase}. A well-structured approach requires first segmenting "
            "the universe by quality tier, then applying a factor overlay — "
            "specifically weighting towards value and low-volatility factors "
            "given the current rate environment. "
            "Historical analysis across three market cycles suggests that "
            "disciplined rebalancing at 5% drift thresholds outperforms "
            "calendar-based rebalancing by approximately 40–60bps annually "
            "after transaction costs. Position sizing should reflect both "
            "conviction and correlation to existing holdings to avoid "
            "inadvertent concentration risk."
        )
        return ExpertOutput(
            answer=answer,
            confidence=random.uniform(0.70, 0.88),
            tokens_used=random.randint(150, 340),
            finish_reason="stop",
        )


class CodeExpert(BaseExpert):
    name               = "code_specialist"
    domain             = "code"
    description        = "Software engineer: architecture, algorithms, debugging, design patterns."
    cost_per_1k_tokens = 0.018

    async def execute(self, query: str) -> ExpertOutput:
        await _jitter(0.20, 0.75)
        q = _snip(query)
        answer = (
            f"Engineering analysis of '{q}': the cleanest solution follows "
            "the Single Responsibility Principle with clear interface contracts. "
            "The key design decision is whether to use composition or inheritance "
            "here — composition is almost always the right call. "
            "For the async execution path, use asyncio.gather() with "
            "return_exceptions=True to prevent one failure from cancelling "
            "the entire batch. Add structured logging at the boundary between "
            "layers (not inside domain logic) and instrument with a context "
            "propagation ID so traces span across service calls. "
            "Test at the unit level with dependency injection, at the "
            "integration level with a real (but ephemeral) backend, and "
            "add at least one end-to-end smoke test per critical user journey."
        )
        return ExpertOutput(
            answer=answer,
            confidence=random.uniform(0.68, 0.87),
            tokens_used=random.randint(120, 300),
            finish_reason="stop",
        )


class ScienceExpert(BaseExpert):
    name               = "science_specialist"
    domain             = "science"
    description        = "Scientific reasoning: evidence evaluation, methodology, research synthesis."
    cost_per_1k_tokens = 0.022

    async def execute(self, query: str) -> ExpertOutput:
        await _jitter(0.50, 1.20)
        q = _snip(query)
        answer = (
            f"Scientific analysis of '{q}': the current empirical literature shows "
            "moderate to strong effect sizes (d ≈ 0.4–0.7) under controlled "
            "conditions, though effect sizes shrink substantially in naturalistic "
            "settings — a common finding in translational research. "
            "The most methodologically robust studies (n > 500, pre-registered, "
            "double-blind) converge on the following mechanism: the primary driver "
            "is not the factor most commonly cited in popular accounts, but rather "
            "a second-order interaction between baseline state and intervention "
            "intensity. Publication bias is a real concern in this literature — "
            "funnel plot asymmetry is detectable — so conclusions should be treated "
            "as directional rather than precise. Replication in diverse populations "
            "is still needed before strong causal claims are warranted."
        )
        return ExpertOutput(
            answer=answer,
            confidence=random.uniform(0.73, 0.91),
            tokens_used=random.randint(180, 400),
            finish_reason="stop",
        )


class SkepticalExpert(BaseExpert):
    name               = "skeptical_analyst"
    domain             = "reasoning"
    description        = "Devil's advocate. Surfaces hidden assumptions, risks, and counter-arguments."
    cost_per_1k_tokens = 0.015

    async def execute(self, query: str) -> ExpertOutput:
        await _jitter(0.25, 0.70)
        q = _snip(query)
        answer = (
            f"Challenging the framing of '{q}': several assumptions in this question "
            "are worth scrutinising. First, the premise that the most commonly "
            "proposed solution is effective is not well-supported — the evidence "
            "base is thinner than often presented, and most cited studies have "
            "serious methodological limitations. Second, there is a selection "
            "effect: the cases where this approach succeeds are far more visible "
            "than the ones where it fails. Third, the counterfactual is rarely "
            "examined — what would have happened without the intervention? "
            "A properly calibrated view should assign substantially more probability "
            "to the null hypothesis than is conventional. The honest answer is: "
            "we don't know with confidence, and anyone claiming otherwise is "
            "overstating the evidence."
        )
        return ExpertOutput(
            answer=answer,
            confidence=random.uniform(0.62, 0.79),
            tokens_used=random.randint(140, 310),
            finish_reason="stop",
        )


class CreativeExpert(BaseExpert):
    name               = "creative_writer"
    domain             = "creative"
    description        = "Creative thinker: narrative, metaphor, lateral thinking, generative ideation."
    cost_per_1k_tokens = 0.012

    async def execute(self, query: str) -> ExpertOutput:
        await _jitter(0.10, 0.45)
        q = _snip(query)
        answer = (
            f"A different lens on '{q}': imagine this as a design problem, not "
            "an information problem. The most elegant solutions often come from "
            "reframing constraints as creative possibilities. Think of it like "
            "a river finding its path — it doesn't fight the terrain, it works "
            "with it. The hidden opportunity here is the tension between the two "
            "apparently opposing needs: resolving that tension productively, rather "
            "than collapsing it in favour of one side, is where the most "
            "interesting answers live. Three unexpected angles worth exploring: "
            "(1) invert the problem entirely — what would the opposite look like? "
            "(2) shrink the scope radically and solve it perfectly at small scale "
            "before growing; (3) steal the solution from an adjacent domain that "
            "has already solved a structurally identical problem."
        )
        return ExpertOutput(
            answer=answer,
            confidence=random.uniform(0.58, 0.76),
            tokens_used=random.randint(100, 260),
            finish_reason="stop",
        )


class MockSynthesizer(BaseExpert):
    """
    Simulates the synthesis step: reads the structured multi-candidate prompt
    written by the aggregator and returns a merged, reconciled answer.
    Used in tests and offline demos where no real local model is available.
    """
    name               = "mock_synthesizer"
    domain             = "general"
    description        = "Combines all expert answers into one refined response (demo/test only)."
    cost_per_1k_tokens = 0.0

    _CANDIDATE_RE = re.compile(
        r"--- Candidate \d+ \(([^)]+)\) ---\s*\n(.+?)(?=\n\n---|\Z)", re.S
    )

    async def execute(self, query: str) -> ExpertOutput:
        await _jitter(0.12, 0.35)
        candidates: List[tuple] = self._CANDIDATE_RE.findall(query)

        if not candidates:
            return ExpertOutput(
                answer=f"Synthesized response: {query[:80]}",
                confidence=0.78,
                tokens_used=50,
                finish_reason="stop",
            )

        # Build a condensed merged answer from the first sentence of each candidate
        highlights = []
        for label, text in candidates:
            first = text.strip().split(". ")[0].strip().rstrip(".")
            highlights.append(f"{first} [{label}]")

        merged = (
            f"Drawing on {len(candidates)} independent expert perspectives, "
            "the evidence converges on the following: "
            + "; ".join(highlights)
            + ". Taken together, these viewpoints form a coherent, "
            "well-rounded answer that balances analytical depth, "
            "domain expertise, and practical applicability — while resolving "
            "the minor differences in emphasis between contributors in favour "
            "of the position best supported by the available evidence."
        )

        total_tokens = sum(len(t.split()) for _, t in candidates) + 80
        return ExpertOutput(
            answer=merged,
            confidence=0.91,
            tokens_used=total_tokens,
            finish_reason="stop",
        )


# ── Registry builder ──────────────────────────────────────────────────────────

def load_mock_experts() -> Dict[str, BaseExpert]:
    """
    Return the full mock expert pool including the synthesizer.
    Drop-in for any Orchestrator(experts=...) call.
    """
    pool = [
        FastGeneralistExpert(),
        DeepReasoningExpert(),
        FinanceExpert(),
        CodeExpert(),
        ScienceExpert(),
        SkepticalExpert(),
        CreativeExpert(),
        MockSynthesizer(),
    ]
    return {e.name: e for e in pool}


def load_domain_experts() -> Dict[str, BaseExpert]:
    """Subset without the synthesizer — useful when you bring your own."""
    return {
        k: v for k, v in load_mock_experts().items()
        if k != "mock_synthesizer"
    }
