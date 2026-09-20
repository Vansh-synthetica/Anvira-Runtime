"""
orcha.integrations
==================
Clean Orcha boundaries around external execution frameworks.

Architecture rule: LangGraph, LangChain, Deep Agents, ragas, and Phoenix
are accessed ONLY through these boundaries — never scattered through
Orcha core. External packages are imported lazily; Orcha core stays
zero-dependency and local-first.

Boundaries
----------
- langchain   component / model / tool abstraction layer
- langgraph   graph & state execution engine
- deepagents  high-level agent harness
- mcp         tool interoperability (native client; langchain adapters
              evaluation pending)
- ragas       batch / regression evaluation backend
- phoenix     OpenTelemetry / OpenInference developer observability
              (client packages only; the ELv2 Phoenix server is an
              external developer tool, never bundled)

External ecosystem source snapshots (for reference) live outside the
package, under ``Orcha/vendor/`` — they are not importable code.
"""
from __future__ import annotations

from .base import (
    Boundary, ExecutionEngine, IntegrationStatus, IntegrationUnavailable,
    availability_report,
)
from .deepagents import DeepAgentsBoundary
from .langchain import LangChainBoundary
from .langgraph import LangGraphBoundary
from .mcp import McpBoundary
from .phoenix import PhoenixBoundary
from .ragas import RagasBoundary

_BOUNDARIES = {
    "langchain": LangChainBoundary(),
    "langgraph": LangGraphBoundary(),
    "deepagents": DeepAgentsBoundary(),
    "mcp": McpBoundary(),
    "ragas": RagasBoundary(),
    "phoenix": PhoenixBoundary(),
}

__all__ = [
    "Boundary", "ExecutionEngine", "IntegrationStatus",
    "IntegrationUnavailable", "availability_report",
    "LangChainBoundary", "LangGraphBoundary", "DeepAgentsBoundary",
    "McpBoundary", "RagasBoundary", "PhoenixBoundary",
]