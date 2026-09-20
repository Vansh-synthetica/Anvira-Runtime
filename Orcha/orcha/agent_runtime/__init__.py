"""
orcha.agent_runtime
===================
The event-sourced AgentRuntime (OpenHands V1 architecture).

Components
----------
- ``Agent``            — STATELESS. ``step(config, log_slice) -> Action``
  (async since Prompt 2; ``step_batch`` returns several actions per model
  call). No agent-owned mutable fields, ever.
- ``Action``           — typed union the agent emits: ToolCallAction,
  MessageAction, FinishAction, ErrorAction.
- ``Observation``      — typed union the system observes: ToolResultObservation,
  ErrorObservation, UserMessageObservation.
- ``EventLog``         — append-only, ordered, replayable. The ONLY
  authoritative state in the system.
- ``Conversation``     — owns the loop: slice -> Agent.step() -> Action ->
  (OrchaPacket bus) Workspace -> Observation -> append to EventLog ->
  repeat, until FinishAction or a step/token/budget cutoff.
- ``Workspace``        — executes Actions, returns Observations.
- ``ModelBackend``     — the model seam (backend-agnostic): a backend is
  injected into the agent; only its config differs per vendor.
- ``Tool``/``ToolRegistry`` — flat typed tool system; the AgentRuntime's
  tools wrap Orcha's existing capability machinery.

Transport
---------
Actions and Observations cross boundaries wrapped in OrchaPackets
(``PacketKind.ACTION`` / ``PacketKind.OBSERVATION``) via bus.py — no
parallel bus exists.

NOTE: ``AgentConfig`` lives here (the runtime's immutable config). The
pre-existing ``orcha.AgentConfig`` from ``orcha.nodes`` is deliberately
left untouched at the package top level; import the runtime one as
``from orcha.agent_runtime import AgentConfig``.
"""
from .agent import (
    Agent, AgentConfig, StubAgent, ToolCallingAgent,
)
from .backend import (
    GenerationConfig, ModelBackend, ModelMessage, ModelResponse, ToolCall,
    TokenUsage, ToolCallParseError,
)
from .bootstrap import build_agent_runtime, discover_tools
from .bus import (
    action_from_packet, action_packet, event_from_packet, event_to_packet,
    observation_from_packet, observation_packet,
)
from .config import AgentRuntimeConfig, McpServerConfig
from .conversation import Conversation, ConversationConfig, ConversationResult
from .diagnostics import (
    DiagEntry, DiagnosticSession, DiagnosticsStore, diagnose_run,
    get_diagnostics_store, redact_payload, redact_text, render_report,
)
from .errors import AgentRuntimeError, EmptyLogError
from .events import (
    Action, ErrorAction, ErrorObservation, Event, EventKind, EventLog,
    FactObservation, FinishAction, LogState, MessageAction, Observation,
    ToolCallAction, ToolResultObservation, UserMessageObservation,
    estimate_tokens, fold_log,
)
from .memory import (
    ContextMemory, LongTermMemory, StepMemory, StepRecord,
    render_memory_section,
)
from .mcp import (
    McpServer, McpToolInfo, McpToolResult, McpTransport, SseMcpTransport,
    StdioMcpTransport, build_mcp_tool, build_transport, register_mcp_server,
)
from .skills import (
    SkillSpec, SkillsDirectory, parse_skill_metadata, register_skills,
    skill_to_tool,
)
from .stream import (
    AgentEventStream, AgentSession, AgentSessionRegistry, event_to_wire,
)
from .tools import Tool, ToolRegistry, build_default_tools
from .workspace import StubWorkspace, ToolWorkspace, Workspace

__all__ = [
    "Agent", "AgentConfig", "StubAgent", "ToolCallingAgent",
    "ModelBackend", "ModelMessage", "ModelResponse", "GenerationConfig",
    "TokenUsage", "ToolCall", "ToolCallParseError",
    "Workspace", "StubWorkspace", "ToolWorkspace",
    "Tool", "ToolRegistry", "build_default_tools",
    "McpServer", "McpServerConfig", "McpToolInfo", "McpToolResult",
    "McpTransport", "StdioMcpTransport", "SseMcpTransport",
    "build_mcp_tool", "build_transport", "register_mcp_server",
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
    "DiagEntry", "DiagnosticSession", "DiagnosticsStore",
    "get_diagnostics_store", "diagnose_run", "render_report",
    "redact_payload", "redact_text",
]
