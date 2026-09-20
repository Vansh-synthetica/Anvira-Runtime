"""
orcha.result
============
RunResult — the immutable terminal projection of a completed graph run.

It is the ORCHA3 analogue of ORCHA2's OrchaResult: a thin, friendly view
over the final packet's payload, budget, and trace, with the same
``.answer / .confidence / .explain() / .to_dict()`` surface so existing
callers migrate with a one-line rename.

The RunResult knows nothing about graph topology — it only reads the packet
the runtime handed it at termination. This keeps it decoupled from the
engine and trivially testable.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .core.packets import OrchaPacket


class RunResult:
    """
    Immutable result of a completed graph run.

    Attributes
    ----------
    answer         Final synthesized (or best-of) answer text.
    confidence     0–1 quality score of the final answer.
    quality_score  Multi-dimensional score (if an evaluator node ran).
    synthesized    True if a node synthesized the answer from contributors.
    primary        Name of the expert / synthesizer that produced the answer.
    contributors   All experts whose output fed into the final answer.
    domains        Detected query domains (finance, code, …).
    agg_mode       How aggregation combined the answers.
    iterations     Number of budget iterations consumed.
    cost           Estimated total cost (USD) across all expert calls.
    latency_s      Wall-clock latency budget consumed.
    run_id         The trace id of this run (== packet.id).
    graph_name     Name of the graph that produced this result.
    packet         The underlying OrchaPacket (for advanced inspection).
    trace          Full list of {node, duration_ms, data} trace steps.
    """

    def __init__(self, packet: OrchaPacket, graph_name: str = "", run_id: Optional[str] = None) -> None:
        self.packet: OrchaPacket = packet
        # run_id tracks the ORIGINAL run id (passed to run()), not the final
        # packet's id — because every fork() mints a fresh packet id, the
        # terminal packet's id is useless for correlating with checkpoints.
        # When not given, fall back to the packet id (legacy behavior).
        self.run_id: str = run_id or packet.id
        self.graph_name: str = graph_name
        p = packet.payload
        self.answer: str = p.get("answer", "")
        self.confidence: float = p.get("confidence", 0.0)
        self.quality_score: float = p.get("quality_score", self.confidence)
        self.synthesized: bool = p.get("synthesized", False)
        self.primary: Optional[str] = p.get("primary")
        self.contributors: List[str] = p.get("contributors", [])
        self.domains: List[str] = p.get("domains", [])
        self.agg_mode: str = p.get("agg_mode", "unknown")
        # Agent graphs record the real iteration count in the payload
        # (_FinalizeAgent), which the budget never increments — prefer it.
        self.iterations: int = p.get("iterations", packet.budget.iterations)
        self.cost: float = packet.budget.cost_used
        self.latency_s: float = packet.budget.latency_used_s
        self.trace: List[Dict[str, Any]] = [t.model_dump() for t in packet.trace]
        # Agent-execution detail (populated by the single-agent graph): every
        # tool call the agent actually ran, the per-iteration steps, and
        # whether the agent reached a natural completion.
        self.agent_steps: List[Dict[str, Any]] = p.get("agent_steps", [])
        self.agent_tool_calls: List[Dict[str, Any]] = p.get("agent_tool_calls", [])
        self.agent_completed: bool = p.get("agent_completed", False)

    # ── Convenience views ──────────────────────────────────────────────

    @property
    def final_answer(self) -> str:
        """Back-compat alias for .answer."""
        return self.answer

    def explain(self) -> str:
        """Human-readable, aligned trace of every node transition."""
        lines = [
            f"Graph     : {self.graph_name or '(unnamed)'}",
            f"Run       : {self.run_id[:8]}",
            f"Query     : {self.packet.query[:120]}",
            "",
        ]
        for step in self.trace:
            extras = "  ".join(f"{k}={v}" for k, v in step["data"].items())
            stage = step["stage"]
            lines.append(f"[{stage:>16}] {step['duration_ms']:7.2f}ms  {extras}")
        lines += [
            "",
            f"Answer      : {self.answer[:120]}{'…' if len(self.answer) > 120 else ''}",
            f"Confidence  : {self.confidence:.3f}",
            f"Quality     : {self.quality_score:.3f}",
            f"Mode        : {self.agg_mode}  (synthesized={self.synthesized})",
            f"Iterations  : {self.iterations}",
            f"Cost        : ${self.cost:.4f}",
            f"Latency     : {self.latency_s:.2f}s",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Flat JSON-serialisable projection of the result."""
        return {
            "run_id":       self.run_id,
            "graph_name":   self.graph_name,
            "answer":       self.answer,
            "confidence":   self.confidence,
            "quality_score": self.quality_score,
            "synthesized":  self.synthesized,
            "primary":      self.primary,
            "contributors": self.contributors,
            "domains":      self.domains,
            "agg_mode":     self.agg_mode,
            "iterations":   self.iterations,
            "cost":         self.cost,
            "latency_s":    self.latency_s,
            "trace":        self.trace,
            "agent_steps":  self.agent_steps,
            "agent_tool_calls": self.agent_tool_calls,
            "agent_completed": self.agent_completed,
        }

    def __repr__(self) -> str:
        return (
            f"RunResult(graph={self.graph_name!r}, run={self.run_id[:8]}, "
            f"confidence={self.confidence:.2f}, synthesized={self.synthesized}, "
            f"iterations={self.iterations}, cost=${self.cost:.4f})"
        )


__all__ = ["RunResult"]
