"""
orcha.agent_runtime.config
==========================
The "what's available in this local install" config surface, sitting
alongside the ModelBackend config from Prompt 2 (OpenAICompatBackendConfig
in ``backends/openai_compat.py``) — a plain pydantic model, same as every
other Orcha component config, no new file format.

``AgentRuntimeConfig`` bundles:

- ``backend``         — the model backend to run (Prompt 2's config).
- ``mcp_servers``     — MCP servers to connect at startup; their tools are
  registered into the ToolRegistry (stdio or SSE transport).
- ``skills_dirs``     — directories of local skill folders to scan; each
  well-formed folder registers one lazily-loaded tool.
- ``workspace_roots`` — the sandboxed workspace directories. When
  non-empty, the shipped default toolset (``run_command``,
  ``read_file``, ``write_file``, ``query_experts``) is registered at
  boot — without it the model is offered no tools and can only reply
  in prose, never execute.

Discovery itself (``orcha.agent_runtime.bootstrap.discover_tools``) is
resilient: an unreachable MCP server or a malformed skill folder logs a
warning and is skipped — AgentRuntime boot never crashes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .backends.openai_compat import OpenAICompatBackendConfig


class McpServerConfig(BaseModel):
    """
    One MCP server to connect at startup.

    stdio  — spawn a local process: ``command`` + ``args`` (+ optional
             ``env`` overrides), the MCP JSON-RPC stream on stdin/stdout.
             This is the shape used by npx/mcp-installed servers and local
             python servers (``command="npx", args=["-y", "pkg"]``).
    sse    — connect to a remote HTTP endpoint: ``url`` (+ optional
             ``headers``), the SSE transport per the MCP spec.
    """
    name: str = "mcp-server"
    transport: Literal["stdio", "sse"] = "stdio"
    # stdio
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    # sse
    url: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    # common
    timeout_s: float = 30.0
    enabled: bool = True

    def validate_fields(self) -> List[str]:
        """Descriptions of missing required fields, for a clean skip
        warning before we even try to connect."""
        missing: List[str] = []
        if self.transport == "stdio" and not self.command:
            missing.append("command")
        if self.transport == "sse" and not self.url:
            missing.append("url")
        return missing


class AgentRuntimeConfig(BaseModel):
    """Everything the AgentRuntime needs to boot in this local install."""

    backend: OpenAICompatBackendConfig = Field(
        default_factory=OpenAICompatBackendConfig,
    )
    mcp_servers: List[McpServerConfig] = Field(default_factory=list)
    skills_dirs: List[str] = Field(default_factory=list)
    workspace_roots: List[str] = Field(
        default_factory=list,
        description=(
            "Absolute workspace directories the agent's tools are sandboxed "
            "to. When non-empty, the default toolset (run_command, "
            "read_file, write_file, query_experts) is registered at boot."
        ),
    )
    stale_write_guard: bool = Field(
        default=False,
        description=(
            "Attach a ReadStateCache to the workspace toolset so mutating "
            "tools refuse blind/partial/stale writes (the agent must read "
            "a file before changing it). Opt-in because it tightens "
            "behavior; recommended for interactive desktop runs."
        ),
    )
    tool_output_dir: Optional[str] = Field(
        default=None,
        description=(
            "Directory for persisting oversized tool outputs. When set, a "
            "ToolExecutor-level OutputStore saves results beyond ~16k chars "
            "to disk and hands the model a bounded preview plus the path. "
            "None keeps in-message truncation only."
        ),
    )
    compaction_enabled: bool = Field(
        default=False,
        description=(
            "Attach the context-compaction ladder to conversations booted "
            "via build_agent_runtime (microcompact old tool results, then "
            "structured summarization above the hard limit, with full "
            "transcripts dumped to disk first). Opt-in; when off, long "
            "runs rely on legacy char truncation only."
        ),
    )

    @classmethod
    def from_json(cls, text: str) -> "AgentRuntimeConfig":
        return cls.model_validate_json(text)


__all__ = ["McpServerConfig", "AgentRuntimeConfig"]
