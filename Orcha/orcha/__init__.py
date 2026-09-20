"""
Orcha — Local-First Multi-Model Orchestration Engine
=====================================================

ORCHA3: Graph execution + durability + agents + retrieval-as-plugin.

Run as many local models as you want, in parallel, and get back ONE
refined answer synthesized from all of them. No cloud API keys required.

    import asyncio
    from orcha import Orchestrator
    from orcha.experts import LocalModelRegistry

    registry = LocalModelRegistry()
    asyncio.run(registry.discover_ollama())   # finds every `ollama pull`ed model

    orc = Orchestrator(
        experts=registry.build(),
        synthesizer_expert=registry.pick_synthesizer(),
        run_all_experts=True,
    )
    result = orc.run("Explain quantum entanglement simply")
    print(result.answer)

ORCHA3 graph API::

    from orcha.graph import Graph, GraphRuntime
    from orcha.graph.node import to_node

    async def my_node(pkt, ctx):
        return pkt.fork(pkt.kind, result="done")

    g = Graph(name="demo")
    g.add_node(to_node(my_node, name="worker"), entry=True)
    g.add_edge("worker", END)
    rt = GraphRuntime(g)
    result = await rt.run("hello world")

For a zero-setup demo with no local models installed, use
`orcha.experts.mock.load_mock_experts()` instead.
"""

# ── ORCHA2 surface (preserved for backward compatibility) ─────────────────────

from .orchestrator import Orchestrator, OrchaResult
from .core.packets import (
    OrchaPacket, PacketKind, BudgetState, ExpertResult, ExpertSlot, SubTask,
)
from .experts.base import BaseExpert, ExpertOutput
from .experts.registry import LocalModelRegistry

# ── ORCHA3 surface ────────────────────────────────────────────────────────────

from .result import RunResult
from .graph import Graph
from .graph.runtime import GraphRuntime
from .graph.node import Node, GatherNode, to_node
from .graph.edge import END
from .graph.store import Store, MemoryStore, FileStore, SqliteStore
from .graph.context import RunContext, CancelToken, EventEmitter, ScatterResult
from .graph.errors import (
    GraphError, GraphInvalid, NodeTimeout, NodeFailed,
    BudgetExceeded, Cancelled, GatherError, NoRoute,
)

# ── Builders ────────────────────────────────────────────────────────────────────

from .builders import (
    build_default_graph, build_default_runner, DefaultGraphConfig,
    build_agent_graph, build_agent_runner, AgentGraphConfig,
    build_agent_graph_langgraph, AgentLangGraphRunner, ApprovalPending,
    build_research_graph, ResearchGraphConfig,
    build_multi_agent_graph, MultiAgentGraphConfig,
)

# ── Nodes ─────────────────────────────────────────────────────────────────────

from .nodes import (
    DecomposeNode, PlanNode, SelectNode, ExecuteNode,
    AggregateNode, EvaluateNode, RetryNode,
    VerifyNode, CriticNode, FactCheckNode,
    AgentNode, AgentConfig,
    ToolNode, ToolSpec,
    RetrievalNode, RetrievalConfig,
)

# ── Agent runtime (event-sourced, OpenHands V1 style) ────────────────────────
# NOTE: AgentConfig intentionally NOT re-exported here — it would shadow the
# pre-existing `orcha.AgentConfig` (orcha.nodes). Import the runtime config
# as `from orcha.agent_runtime import AgentConfig`.

from .agent_runtime import (
    Agent, StubAgent, ToolCallingAgent,
    ModelBackend, GenerationConfig, ModelResponse,
    Workspace, StubWorkspace, ToolWorkspace,
    Tool, ToolRegistry, build_default_tools,
    McpServer, McpServerConfig, McpToolInfo, McpToolResult,
    McpTransport, StdioMcpTransport, SseMcpTransport,
    build_mcp_tool, register_mcp_server,
    SkillSpec, SkillsDirectory, skill_to_tool, register_skills,
    parse_skill_metadata,
    AgentRuntimeConfig, discover_tools, build_agent_runtime,
    Conversation, ConversationConfig, ConversationResult,
    EventLog, Event, EventKind, LogState, fold_log, estimate_tokens,
    Action, ToolCallAction, MessageAction, FinishAction, ErrorAction,
    Observation, ToolResultObservation, ErrorObservation, UserMessageObservation,
    FactObservation,
    StepMemory, StepRecord, LongTermMemory, ContextMemory, render_memory_section,
    AgentEventStream, AgentSession, AgentSessionRegistry, event_to_wire,
    action_packet, action_from_packet,
    observation_packet, observation_from_packet,
    event_to_packet, event_from_packet,
    AgentRuntimeError, EmptyLogError,
)

# ── Integration boundaries (external frameworks, lazy) ────────────────────────

from .integrations import (
    availability_report, IntegrationStatus, IntegrationUnavailable,
)

__version__ = "0.4.0"

__all__ = [
    # ORCHA2 (backward compat)
    "Orchestrator", "OrchaResult",
    "OrchaPacket", "PacketKind", "BudgetState",
    "ExpertResult", "ExpertSlot", "SubTask",
    "BaseExpert", "ExpertOutput", "LocalModelRegistry",
    # ORCHA3 core
    "RunResult",
    "Graph", "GraphRuntime", "Node", "GatherNode", "to_node",
    "END",
    "Store", "MemoryStore", "FileStore", "SqliteStore",
    "RunContext", "CancelToken", "EventEmitter", "ScatterResult",
    "GraphError", "GraphInvalid", "NodeTimeout", "NodeFailed",
    "BudgetExceeded", "Cancelled", "GatherError", "NoRoute",
    # Builders
    "build_default_graph", "build_default_runner", "DefaultGraphConfig",
    "build_agent_graph", "build_agent_runner", "AgentGraphConfig",
    "build_agent_graph_langgraph", "AgentLangGraphRunner", "ApprovalPending",
    "build_research_graph", "ResearchGraphConfig",
    "build_multi_agent_graph", "MultiAgentGraphConfig",
    # Nodes
    "DecomposeNode", "PlanNode", "SelectNode", "ExecuteNode",
    "AggregateNode", "EvaluateNode", "RetryNode",
    "VerifyNode", "CriticNode", "FactCheckNode",
    "AgentNode", "AgentConfig",
    "ToolNode", "ToolSpec",
    "RetrievalNode", "RetrievalConfig",
    # Agent runtime (event-sourced)
    "Agent", "StubAgent", "ToolCallingAgent",
    "ModelBackend", "GenerationConfig", "ModelResponse",
    "Workspace", "StubWorkspace", "ToolWorkspace",
    "Tool", "ToolRegistry", "build_default_tools",
    "McpServer", "McpServerConfig", "McpToolInfo", "McpToolResult",
    "McpTransport", "StdioMcpTransport", "SseMcpTransport",
    "build_mcp_tool", "register_mcp_server",
    "SkillSpec", "SkillsDirectory", "skill_to_tool", "register_skills",
    "parse_skill_metadata",
    "AgentRuntimeConfig", "discover_tools", "build_agent_runtime",
    "Conversation", "ConversationConfig", "ConversationResult",
    "EventLog", "Event", "EventKind", "LogState", "fold_log", "estimate_tokens",
    "Action", "ToolCallAction", "MessageAction", "FinishAction", "ErrorAction",
    "Observation", "ToolResultObservation", "ErrorObservation",
    "UserMessageObservation",
    "FactObservation",
    "StepMemory", "StepRecord", "LongTermMemory", "ContextMemory",
    "render_memory_section",
    "AgentEventStream", "AgentSession", "AgentSessionRegistry", "event_to_wire",
    "action_packet", "action_from_packet",
    "observation_packet", "observation_from_packet",
    "event_to_packet", "event_from_packet",
    "AgentRuntimeError", "EmptyLogError",
    # Integration boundaries
    "availability_report", "IntegrationStatus", "IntegrationUnavailable",
    "__version__",
]
