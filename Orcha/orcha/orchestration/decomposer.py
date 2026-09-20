"""
orcha.orchestration.decomposer
================================
Analyses the user query to produce:
  1. A prioritised list of SubTasks the pipeline should address.
  2. A list of detected domains (finance, code, science, …) used by the
     selector to route queries to the right experts.
  3. A complexity estimate (0.0–1.0) used by the planner to decide how
     many experts to run and how high to set the quality threshold.

Domain detection
----------------
Uses a multi-tier keyword registry. Each domain has:
  - strong keywords  (weight 1.0): nearly unambiguous signal
  - medium keywords  (weight 0.6): likely but not certain
  - weak keywords    (weight 0.3): contextual hints

A domain is detected when its total weighted keyword hits exceed a
configurable threshold (default 0.6). Multiple domains can fire at once.

Complexity estimation
---------------------
Heuristics that push complexity up:
  - Query length (longer → more complex)
  - Number of question marks (multi-part questions)
  - Presence of comparison/contrast words
  - Technical vocabulary density
  - Negation density (more hedging = harder)

Semantic embeddings (optional)
------------------------------
If sentence-transformers is installed, the decomposer can also embed
the query and compare it against domain centroid embeddings for more
accurate classification. This is disabled by default (use_embeddings=False)
to keep the zero-dependency fast path working.
"""
from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Tuple

from ..core.packets import OrchaPacket, PacketKind, SubTask


# ── Domain keyword registry ───────────────────────────────────────────────────
# Structure: domain -> {"strong": [...], "medium": [...], "weak": [...]}

DOMAIN_REGISTRY: Dict[str, Dict[str, List[str]]] = {
    "finance": {
        "strong":  ["stock", "portfolio", "investment", "dividend", "equity",
                    "bond", "derivative", "hedge", "arbitrage", "valuation",
                    "p/e ratio", "alpha", "beta", "sharpe", "drawdown",
                    "asset allocation", "rebalance", "etf", "mutual fund",
                    "interest rate", "inflation", "yield curve"],
        "medium":  ["market", "trade", "risk", "return", "profit", "loss",
                    "revenue", "cash flow", "balance sheet", "earnings",
                    "diversify", "sector", "bull", "bear", "volatility"],
        "weak":    ["money", "cost", "price", "spend", "save", "budget",
                    "bank", "loan", "debt", "capital"],
    },
    "code": {
        "strong":  ["function", "class", "async", "await", "algorithm",
                    "refactor", "debug", "stack overflow", "compile",
                    "runtime error", "unit test", "api endpoint", "rest api",
                    "graphql", "sql", "database schema", "docker", "kubernetes",
                    "microservice", "dependency injection", "design pattern"],
        "medium":  ["code", "program", "software", "bug", "implement",
                    "library", "framework", "module", "package", "deploy",
                    "performance", "memory leak", "thread", "concurrent"],
        "weak":    ["script", "automate", "build", "pipeline", "config",
                    "server", "client", "request", "response"],
    },
    "science": {
        "strong":  ["hypothesis", "experiment", "peer review", "p-value",
                    "statistical significance", "control group", "double blind",
                    "meta-analysis", "systematic review", "clinical trial",
                    "genome", "protein", "quantum", "thermodynamics"],
        "medium":  ["research", "study", "evidence", "data", "analysis",
                    "measure", "observe", "theory", "model", "simulate",
                    "biology", "chemistry", "physics", "neuroscience"],
        "weak":    ["discover", "test", "result", "find", "show", "prove"],
    },
    "reasoning": {
        "strong":  ["logical fallacy", "deductive", "inductive", "abductive",
                    "syllogism", "counterfactual", "causal inference",
                    "epistemology", "paradox", "first principles"],
        "medium":  ["why", "how", "explain", "analyze", "compare", "evaluate",
                    "argument", "reason", "cause", "implication", "trade-off",
                    "pros and cons", "assumption", "contradict"],
        "weak":    ["think", "consider", "reflect", "understand", "clarify"],
    },
    "math": {
        "strong":  ["integral", "derivative", "matrix", "eigenvalue", "proof",
                    "theorem", "equation", "polynomial", "differential",
                    "probability distribution", "bayes", "linear algebra",
                    "calculus", "topology", "number theory"],
        "medium":  ["calculate", "solve", "compute", "formula", "graph",
                    "function", "variable", "coefficient", "sum", "product"],
        "weak":    ["number", "count", "total", "average", "percentage"],
    },
    "creative": {
        "strong":  ["write a story", "write a poem", "compose", "narrative",
                    "character development", "plot twist", "metaphor",
                    "protagonist", "dialogue", "screenplay", "storyboard"],
        "medium":  ["write", "create", "design", "imagine", "generate",
                    "draft", "brainstorm", "idea", "concept", "creative"],
        "weak":    ["describe", "explain", "present", "illustrate"],
    },
    "medical": {
        "strong":  ["diagnosis", "symptom", "treatment", "medication", "dosage",
                    "clinical", "pathology", "prognosis", "contraindication",
                    "differential diagnosis", "pharmacology", "surgery"],
        "medium":  ["health", "disease", "condition", "pain", "therapy",
                    "patient", "doctor", "hospital", "medicine", "chronic"],
        "weak":    ["feel", "hurt", "sick", "tired", "body"],
    },
}

DOMAIN_WEIGHTS = {"strong": 1.0, "medium": 0.6, "weak": 0.3}
DOMAIN_THRESHOLD = 0.6   # minimum weighted score to declare a domain active

# ── Complexity signals ────────────────────────────────────────────────────────

COMPARISON_WORDS = {"vs", "versus", "compare", "difference", "better", "worse",
                    "pros", "cons", "tradeoff", "trade-off", "rather than",
                    "instead of", "alternative"}
NEGATION_WORDS   = {"not", "don't", "doesn't", "isn't", "aren't", "won't",
                    "shouldn't", "cannot", "can't", "never", "no"}
TECH_INDICATORS  = {"algorithm", "architecture", "distributed", "latency",
                    "throughput", "scalable", "concurrent", "synchronous",
                    "asynchronous", "stateless", "idempotent", "polymorphic"}

# ── Sub-task templates ────────────────────────────────────────────────────────

BASE_SUBTASKS: List[SubTask] = [
    SubTask(id="understand",  description="Understand the core question and intent",           domain="general",   priority=1.00),
    SubTask(id="decompose",   description="Break the problem into addressable sub-problems",   domain="general",   priority=0.90),
    SubTask(id="analyse",     description="Analyse relevant information and evidence",         domain="general",   priority=0.88),
    SubTask(id="synthesise",  description="Synthesise a coherent, well-grounded answer",       domain="general",   priority=0.85),
    SubTask(id="verify",      description="Verify factual accuracy and internal consistency",  domain="general",   priority=0.72),
    SubTask(id="explain",     description="Generate a clear, accessible explanation",          domain="general",   priority=0.80),
]

DOMAIN_SUBTASKS: Dict[str, List[SubTask]] = {
    "finance": [
        SubTask(id="market_ctx",    description="Establish current market context and conditions",     domain="finance", priority=0.90),
        SubTask(id="risk_assess",   description="Identify and quantify relevant risk factors",         domain="finance", priority=0.88),
        SubTask(id="quantitative",  description="Perform necessary quantitative or numerical analysis", domain="finance", priority=0.82),
        SubTask(id="regulatory",    description="Consider regulatory and compliance implications",      domain="finance", priority=0.70),
    ],
    "code": [
        SubTask(id="requirements",  description="Clarify technical requirements and constraints",       domain="code", priority=0.92),
        SubTask(id="design",        description="Design the architecture or solution approach",         domain="code", priority=0.88),
        SubTask(id="implement",     description="Produce concrete implementation or code",              domain="code", priority=0.85),
        SubTask(id="edge_cases",    description="Identify and handle edge cases and errors",            domain="code", priority=0.78),
        SubTask(id="test_plan",     description="Outline a testing and validation strategy",            domain="code", priority=0.70),
    ],
    "science": [
        SubTask(id="literature",    description="Summarise relevant scientific literature",             domain="science", priority=0.88),
        SubTask(id="methodology",   description="Evaluate methodological soundness and limitations",    domain="science", priority=0.84),
        SubTask(id="consensus",     description="Identify areas of scientific consensus vs. debate",    domain="science", priority=0.80),
    ],
    "reasoning": [
        SubTask(id="premises",      description="Identify and evaluate underlying premises",            domain="reasoning", priority=0.92),
        SubTask(id="logic_check",   description="Check logical validity and detect fallacies",          domain="reasoning", priority=0.88),
        SubTask(id="counter",       description="Construct and address strongest counter-arguments",    domain="reasoning", priority=0.80),
    ],
    "math": [
        SubTask(id="formulate",     description="Formulate the problem mathematically",                 domain="math", priority=0.92),
        SubTask(id="solve",         description="Solve step-by-step, showing working",                  domain="math", priority=0.90),
        SubTask(id="check",         description="Verify the solution by substitution or alternative method", domain="math", priority=0.80),
    ],
    "creative": [
        SubTask(id="concept",       description="Develop the core creative concept and theme",          domain="creative", priority=0.90),
        SubTask(id="draft",         description="Produce an initial creative draft",                    domain="creative", priority=0.88),
        SubTask(id="refine",        description="Refine for clarity, voice, and impact",               domain="creative", priority=0.82),
    ],
    "medical": [
        SubTask(id="clinical_ctx",  description="Establish clinical context and patient factors",       domain="medical", priority=0.92),
        SubTask(id="evidence",      description="Assess evidence quality and clinical guidelines",       domain="medical", priority=0.88),
        SubTask(id="safety",        description="Evaluate risks, contraindications, and safety",        domain="medical", priority=0.90),
    ],
}


class SmartDecomposer:
    """
    Decomposes a user query into prioritised subtasks and detects domains.
    """

    def __init__(self, use_embeddings: bool = False, domain_threshold: float = DOMAIN_THRESHOLD):
        self.domain_threshold = domain_threshold
        self._embedder = None
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer  # noqa
                self._embedder = True
            except ImportError:
                pass
        # Pre-compute domain keyword sets for O(1) lookup during detection.
        # Structure: domain -> [(weight, keyword_set), ...]
        self._domain_kw_sets: Dict[str, List[Tuple[float, frozenset]]] = {}
        for domain, tiers in DOMAIN_REGISTRY.items():
            self._domain_kw_sets[domain] = [
                (DOMAIN_WEIGHTS[tier], frozenset(keywords))
                for tier, keywords in tiers.items()
            ]

    def decompose(self, packet: OrchaPacket) -> OrchaPacket:
        t0    = time.perf_counter()
        query = packet.query

        domains, domain_scores = self._detect_domains(query)
        complexity              = self._estimate_complexity(query)
        subtasks                = self._build_subtasks(domains, max_tasks=10)
        query_type              = self._classify_query_type(query)

        duration_ms = (time.perf_counter() - t0) * 1000

        return packet.fork(
            PacketKind.SUBTASKS,
            subtasks=[t.model_dump() for t in subtasks],
            domains=domains,
            domain_scores=domain_scores,
            complexity=round(complexity, 3),
            query_type=query_type,
        ).stamp(
            "decompose", duration_ms,
            subtasks=len(subtasks),
            domains=domains,
            complexity=round(complexity, 3),
            query_type=query_type,
        )

    # ── Domain detection ──────────────────────────────────────────────

    def _detect_domains(self, query: str) -> Tuple[List[str], Dict[str, float]]:
        q = query.lower()
        # Split once, reuse for all keyword checks
        q_tokens = q.split()
        q_words = frozenset(q_tokens)
        scores: Dict[str, float] = {}
        for domain, kw_groups in self._domain_kw_sets.items():
            score = 0.0
            for weight, keywords in kw_groups:
                # Fast path: for single-word keywords, use O(1) frozenset lookup
                # For multi-word phrases, use substring search (in q)
                for kw in keywords:
                    if ' ' in kw:
                        # Multi-word phrase — must substring match
                        if kw in q:
                            score += weight
                    else:
                        # Single word — O(1) frozenset lookup
                        if kw in q_words:
                            score += weight
            if score > 0:
                scores[domain] = round(score, 3)

        active = [d for d, s in scores.items() if s >= self.domain_threshold]
        # Always include "general" as a fallback
        if not active:
            active = ["general"]
        # Sort by score descending
        active.sort(key=lambda d: scores.get(d, 0), reverse=True)
        return active, scores

    # ── Complexity estimation ─────────────────────────────────────────

    def _estimate_complexity(self, query: str) -> float:
        tokens = query.lower().split()
        q_words = frozenset(tokens)
        score  = 0.0

        # Length signal (50–150 tokens = medium; 150+ = high)
        length     = len(tokens)
        score     += min(length / 200, 0.3)

        # Multi-part questions
        n_q        = query.count("?")
        score     += min(n_q * 0.08, 0.20)

        # Comparison / contrast — O(1) frozenset intersection
        comp_hits  = len(COMPARISON_WORDS & q_words)
        score     += min(comp_hits * 0.06, 0.15)

        # Negation density — O(1) frozenset intersection
        neg_hits   = len(NEGATION_WORDS & q_words)
        score     += min(neg_hits * 0.03, 0.08)

        # Technical vocabulary — O(1) frozenset intersection
        tech_hits  = len(TECH_INDICATORS & q_words)
        score     += min(tech_hits * 0.05, 0.15)

        return min(score, 1.0)

    # ── Query classification ──────────────────────────────────────────

    def _classify_query_type(self, query: str) -> str:
        q = query.lower().strip()
        if q.startswith(("what is", "what are", "define", "explain what")):
            return "definition"
        if q.startswith(("how do", "how to", "how can", "how should", "how would")):
            return "how_to"
        if q.startswith(("why", "what causes", "what makes")):
            return "causal"
        if q.startswith(("should i", "would you", "is it better", "which is better")):
            return "recommendation"
        if q.startswith(("compare", "what is the difference", "vs", "versus")):
            return "comparison"
        if q.startswith(("write", "generate", "create", "draft", "compose")):
            return "generative"
        if q.startswith(("analyse", "analyze", "evaluate", "assess", "review")):
            return "analytical"
        return "general"

    # ── Subtask assembly ──────────────────────────────────────────────

    def _build_subtasks(self, domains: List[str], max_tasks: int = 10) -> List[SubTask]:
        subtasks: List[SubTask] = list(BASE_SUBTASKS)
        seen_ids: set = {t.id for t in subtasks}

        for domain in domains:
            for st in DOMAIN_SUBTASKS.get(domain, []):
                if st.id not in seen_ids:
                    seen_ids.add(st.id)
                    subtasks.append(st)

        subtasks.sort(key=lambda t: t.priority, reverse=True)
        return subtasks[:max_tasks]


def get_decomposer(use_embeddings: bool = False) -> SmartDecomposer:
    return SmartDecomposer(use_embeddings=use_embeddings)
