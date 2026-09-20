"""
orcha.builders
==============
Pre-built graph configurations for common orchestration patterns.

Builders produce fully-validated ``Graph`` objects that can be run directly
with ``GraphRuntime``. They are the primary way users construct ORCHA3
pipelines without manually wiring nodes and edges.

Available builders
------------------
- ``build_default_graph``  ORCHA2-compatible pipeline (regression bridge).
- ``build_default_runner``  Same pipeline, engine-switchable (native or LangGraph).
- ``build_agent_graph``  Single-agent graph with intent gate + tool loop.
- ``build_agent_runner``  Same graph, engine-switchable (native or LangGraph,
  with an interrupt-based approval gate when ``require_approval`` is set).
- ``ComplexityGateConfig``  Configuration for the complexity analysis gate.
- ``build_research_graph``  RAG-augmented research pipeline with retrieval + verification.
- ``build_multi_agent_graph``  Multi-agent collaboration pipeline.
- ``build_multi_agent_runner``  Same pipeline, engine-switchable (native or LangGraph).
"""
from .default import (
    build_default_graph,
    build_default_runner,
    DefaultGraphConfig,
)
from .langgraph_default import (
    build_default_graph_langgraph,
    LangGraphDefaultRunner,
)
from .agent import build_agent_graph, build_agent_runner, AgentGraphConfig, ComplexityGateConfig
from .langgraph_agent import (
    build_agent_graph_langgraph,
    AgentLangGraphRunner,
    ApprovalPending,
)
from .research import build_research_graph, ResearchGraphConfig
from .multi_agent import build_multi_agent_graph, build_multi_agent_runner, MultiAgentGraphConfig
from .langgraph_multi_agent import (
    build_multi_agent_graph_langgraph,
    MultiAgentLangGraphRunner,
)

__all__ = [
    "build_default_graph", "build_default_runner", "DefaultGraphConfig",
    "build_default_graph_langgraph", "LangGraphDefaultRunner",
    "build_agent_graph", "build_agent_runner", "AgentGraphConfig", "ComplexityGateConfig",
    "build_agent_graph_langgraph", "AgentLangGraphRunner", "ApprovalPending",
    "build_research_graph", "ResearchGraphConfig",
    "build_multi_agent_graph", "build_multi_agent_runner", "MultiAgentGraphConfig",
    "build_multi_agent_graph_langgraph", "MultiAgentLangGraphRunner",
]
