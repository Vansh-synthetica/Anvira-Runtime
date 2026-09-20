"""
orcha.agent_runtime.bootstrap
=============================
Startup-time tool discovery for the AgentRuntime: pull external tools in
from MCP servers and skills directories into one ToolRegistry — the Agent
and ModelBackend never know where a tool came from.

Resilience contract
-------------------
An unreachable/misconfigured MCP server or a malformed skill folder logs
a warning and is skipped. Discovery NEVER raises: no matter what is
configured, AgentRuntime boot completes with a usable registry.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from ..core.packets import BudgetState
from ..capabilities.outputstore import OutputStore
from ..capabilities.readstate import ReadStateCache
from .agent import AgentConfig, ToolCallingAgent
from .config import AgentRuntimeConfig
from .mcp import register_mcp_server
from .skills import register_skills
from .tools import ToolRegistry, build_default_tools
from .workspace import ToolWorkspace

logger = logging.getLogger("orcha.agent_runtime.bootstrap")

_DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.expanduser("~"), ".orcha", "tool-outputs",
)

logger = logging.getLogger("orcha.agent_runtime.bootstrap")


def discover_tools(
    config: AgentRuntimeConfig,
    registry: Optional[ToolRegistry] = None,
    *,
    log: Optional[logging.Logger] = None,
) -> ToolRegistry:
    """
    Register every configured external tool source into ``registry``
    (fresh when not given):

    - each enabled MCP server (stdio/SSE) — its tools registered as
      ordinary Tools;
    - each skills directory — well-formed skill folders registered as
      lazily-loaded Tools.

    Resilient by contract: every per-source failure is logged and skipped.
    """
    log = log or logger
    registry = registry or ToolRegistry()

    for server_config in config.mcp_servers:
        register_mcp_server(registry, server_config, log=log)

    for directory in config.skills_dirs:
        register_skills(registry, directory, log=log)

    return registry


def build_agent_runtime(
    config: AgentRuntimeConfig,
    *,
    registry: Optional[ToolRegistry] = None,
    backend: Optional["ModelBackend"] = None,
    agent_config: Optional[AgentConfig] = None,
    query: str = "",
    budget: Optional[BudgetState] = None,
    compactor: Optional["Compactor"] = None,
) -> "tuple":
    """
    One-call boot: register the shipped toolset (when ``workspace_roots``
    are configured) + discovered external tools, wire the backend +
    ToolCallingAgent + ToolWorkspace + Conversation. Returns
    ``(conversation, registry, backend)`` where ``conversation`` is ready
    for ``submit_user_message`` + ``await run()``.

    Without ``workspace_roots`` the registry holds only external tools
    (MCP/skills): the model is offered no file/terminal tools and can
    only reply in prose — pass workspace roots to enable real execution.

    ``backend`` optionally injects a ModelBackend (default: the
    OpenAI-compatible backend from ``config.backend``); used by tests and
    embedding callers that already hold a client.

    Hardening switches (from ``config``):

    - ``stale_write_guard`` — attach a ReadStateCache so filesystem tools
      refuse blind/partial/stale writes (read-before-write discipline).
    - ``tool_output_dir`` — persist oversized tool results to disk and hand
      the model a preview + path instead of a truncated wall of text.

    ``compactor`` optionally attaches the context-compaction ladder to the
    Conversation (microcompact → structured summary → transcript-on-disk);
    without it long runs degrade to the legacy char truncation only.
    """
    from .backend import ModelBackend
    from .backends.openai_compat import OpenAICompatBackend
    from .compaction import Compactor, TranscriptStore
    from .conversation import Conversation, ConversationConfig

    registry = discover_tools(config, registry)
    if config.workspace_roots:
        read_state = ReadStateCache() if config.stale_write_guard else None
        for tool in build_default_tools(
            roots=config.workspace_roots, read_state=read_state,
        ):
            registry.register(tool)
    backend = backend or OpenAICompatBackend(config.backend)
    agent = ToolCallingAgent(backend, registry)
    conversation = Conversation(
        agent,
        ToolWorkspace(registry),
        agent_config=agent_config or AgentConfig(model=backend.model_name),
        query=query,
        budget=budget,
        compactor=compactor or (
            # Default ladder wired to the same backend: microcompact always
            # available; LLM summarization rides the backend; transcripts
            # land next to tool outputs.
            Compactor(
                backend,
                transcript_store=TranscriptStore(
                    os.environ.get(
                        "ORCHA_TRANSCRIPT_DIR",
                        os.path.join(os.path.expanduser("~"), ".orcha", "transcripts"),
                    ),
                ),
            )
            if config.compaction_enabled
            else None
        ),
    )
    _wire_output_store(registry, config)
    return conversation, registry, backend


def _wire_output_store(registry: ToolRegistry, config: AgentRuntimeConfig) -> None:
    """
    Attach an OutputStore to every Tool in the registry that owns a private
    executor (the default Tool wrapper). Oversized results then persist to
    disk with a model-facing preview + saved path. Best-effort: any failure
    logs a warning and leaves tools unwired.
    """
    directory = config.tool_output_dir or os.environ.get("ORCHA_TOOL_OUTPUT_DIR")
    if not directory:
        return
    try:
        store = OutputStore(directory)
        for tool in registry.tools():
            executor = getattr(tool, "_executor", None)
            if executor is not None:
                executor._output_store = store
    except Exception as exc:  # resilient-by-contract
        logger.warning("could not wire tool output store: %s", exc)


__all__ = ["discover_tools", "build_agent_runtime"]
