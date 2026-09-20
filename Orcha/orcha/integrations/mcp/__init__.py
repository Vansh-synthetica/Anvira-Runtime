"""
orcha.integrations.mcp
======================
Boundary for MCP — external tool interoperability.

Currently this boundary re-exports the native Orcha MCP client
(``orcha.agent_runtime.mcp``), which is kept because it is lightweight,
local-first, and already wired into the agent runtime and the Anvira
event contract.

Decision point (Stage 9): if LangChain's MCP adapters prove more mature
for the ecosystem we target, this boundary swaps its implementation
behind the same names — nothing above this module changes.

AICL and MCP remain separate by design:

- AICL  = LocalHouseLLM internal module communication.
- MCP   = external tool interoperability.
"""
from __future__ import annotations

from ...agent_runtime.mcp import (
    McpServer, McpServerConfig, McpToolInfo, McpToolResult,
    McpTransport, StdioMcpTransport, SseMcpTransport,
    build_mcp_tool, register_mcp_server,
)
from ..base import Boundary, IntegrationStatus

__all__ = [
    "McpBoundary",
    "McpServer", "McpServerConfig", "McpToolInfo", "McpToolResult",
    "McpTransport", "StdioMcpTransport", "SseMcpTransport",
    "build_mcp_tool", "register_mcp_server",
]


class McpBoundary(Boundary):
    """Native MCP client — always available, no extra dependency."""

    name = "mcp"
    package = "orcha"

    def status(self) -> IntegrationStatus:
        return IntegrationStatus(
            name=self.name, package=self.package, extra=None,
            available=True, version=None,
        )

    def load(self) -> object:
        return self