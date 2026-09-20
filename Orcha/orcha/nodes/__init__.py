"""
orcha.nodes
===========
Graph-node wrappers for ORCHA2 orchestration stages and new ORCHA3 node types.

Every ORCHA2 stage (decompose, plan, select, execute, aggregate, evaluate,
retry) has a corresponding ``*Node`` class that delegates to the legacy
stage object. This means existing ORCHA2 users can drop their stage objects
into a graph with zero code changes::

    from orcha.orchestration.decomposer import SmartDecomposer
    from orcha.nodes import DecomposeNode

    node = DecomposeNode(SmartDecomposer())
    graph.add_node(node, entry=True)

New ORCHA3 node types (VerifyNode, AgentNode, ToolNode, RetrievalNode) are
first-class graph citizens with full context and lifecycle support.
"""
from .stages import (
    DecomposeNode,
    PlanNode,
    SelectNode,
    ExecuteNode,
    AggregateNode,
    EvaluateNode,
    RetryNode,
)
from .verify import VerifyNode, CriticNode, FactCheckNode
from .agent import AgentNode, AgentConfig
from .tool import ToolNode, ToolSpec
from .retrieval import RetrievalNode, RetrievalConfig

__all__ = [
    # ORCHA2 stage wrappers
    "DecomposeNode", "PlanNode", "SelectNode",
    "ExecuteNode", "AggregateNode", "EvaluateNode", "RetryNode",
    # Verification nodes
    "VerifyNode", "CriticNode", "FactCheckNode",
    # Agent nodes
    "AgentNode", "AgentConfig",
    # Tool nodes
    "ToolNode", "ToolSpec",
    # Retrieval nodes
    "RetrievalNode", "RetrievalConfig",
]
