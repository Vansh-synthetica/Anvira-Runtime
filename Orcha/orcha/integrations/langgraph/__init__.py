"""
orcha.integrations.langgraph
============================
Boundary around LangGraph — the primary graph / state execution engine.

Orcha core never imports langgraph directly; it goes through this
boundary. The boundary is lazy: ``LangGraphBoundary()`` is cheap, and
``load()`` imports langgraph on demand, raising
``IntegrationUnavailable`` with an install hint when the optional
``orcha[lang]`` extra is missing.

Engine seam (Stage 3): ``LangGraphEngine`` executes any compiled
LangGraph whose state carries the OrchaPacket, behind Orcha's
``ExecutionEngine`` contract — ``execute(packet, ctx) -> packet``.
Thread id = Orcha run id (checkpoints/resume for free); run context is
injected per invoke and never checkpointed; interrupts surface as
``ExecutionInterrupt`` and resume via ``LangGraphEngine.resume``.

``orcha_node_to_langgraph`` / ``build_prototype_graph`` bridge Orcha
Nodes onto LangGraph and provide a linear prototype graph used to
prove the seam.
"""
from __future__ import annotations

from ..base import Boundary
from .engine import ExecutionInterrupt, LangGraphEngine
from .prototype import build_prototype_graph, orcha_node_to_langgraph, orcha_serde
from .checkpoint import OrchaSqliteCheckpointer, build_sqlite_checkpointer

__all__ = [
    "LangGraphBoundary", "LangGraphEngine", "ExecutionInterrupt",
    "orcha_node_to_langgraph", "build_prototype_graph", "orcha_serde",
    "OrchaSqliteCheckpointer", "build_sqlite_checkpointer",
]


class LangGraphBoundary(Boundary):
    name = "langgraph"
    package = "langgraph"
    extra = "lang"

    def load(self) -> object:
        """Return the ``langgraph`` module (raises if not installed)."""
        return self._import("langgraph")