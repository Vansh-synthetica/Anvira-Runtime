"""
orcha.tools
===========
Real tools agents can invoke (native function-calling).

Currently ships the workspace filesystem toolset; new tool families
(terminal, git, build/test runners) should be added as sibling modules
and exposed through a registry here.
"""
from __future__ import annotations

from .workspace import (
    build_workspace_tools,
    tool_schemas,
    WorkspaceToolError,
)

__all__ = ["build_workspace_tools", "tool_schemas", "WorkspaceToolError"]
