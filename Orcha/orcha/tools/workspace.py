"""
orcha.tools.workspace
=====================
Backward-compatible facade over the capability-based filesystem tools.

This module keeps the legacy public API (``build_workspace_tools``,
``tool_schemas``, ``WorkspaceToolError``) but delegates every tool to the
capability system in :mod:`orcha.capabilities.filesystem`, so there is a
single implementation of root-guarding, UTF-8 handling, line-ending
preservation and structured errors.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..capabilities.base import CapabilityContext, ToolError
from ..capabilities.filesystem import CAPABILITY_NAME as _CAPABILITY_NAME
from ..capabilities.filesystem import build_tools as _build_capability_tools
from ..capabilities.pathing import normalize_path as _normalize_path
from ..nodes.tool import ToolSpec

# Legacy constants kept for compatibility with callers that imported them.
_IGNORED_DIRS = {
    "node_modules", ".git", "dist", ".next", ".cache", "__pycache__",
    ".venv", "venv", ".build-venv", ".anvira-index-cache", ".idea", ".vscode",
    "coverage", ".tmp", "bin", "build", "out", "target",
}
_IGNORED_FILES = {".DS_Store", "Thumbs.db"}

MAX_TEXT_CHARS = 200_000
MAX_RESULTS = 50


class WorkspaceToolError(ToolError):
    """
    Legacy structured error type. Subclasses :class:`ToolError` so existing
    ``except WorkspaceToolError`` code keeps working while structured codes
    are preserved.
    """

    def __init__(self, message: str, code: str = "workspace_tool_error", detail: Any = None) -> None:
        super().__init__(code, message, detail)


def normalize_path(p: str) -> str:
    """Absolute, expanded form of ``p`` (delegates to capability pathing)."""
    return _normalize_path(p)


# Legacy tool name → capability tool name.
_LEGACY_TO_CAPABILITY = {
    "read_file": "read_file",
    "write_file": "write_file",
    "edit_file": "edit_file",
    "list_dir": "list_directory",
    "search_text": "search_text",
}

_LEGACY_TOOL_NAMES = list(_LEGACY_TO_CAPABILITY)


def build_workspace_tools(
    roots: List[str],
    allow: Optional[List[str]] = None,
) -> List[ToolSpec]:
    """
    Build the workspace tool set bound to ``roots`` (legacy API).

    Delegates to the Filesystem capability; ``allow`` optionally restricts
    the returned tools to the named legacy subset. Every tool enforces that
    its ``path`` argument resolves inside one of ``roots``.
    """
    from ..capabilities.filesystem import CAPABILITY_NAME as capability_name

    allowed = [n for n in (allow or _LEGACY_TOOL_NAMES) if n in _LEGACY_TO_CAPABILITY]
    by_new_name = {
        t.name: t for t in _build_capability_tools(CapabilityContext(roots=roots))
    }

    tools: List[ToolSpec] = []
    for legacy_name in allowed:
        capability_name_of = _LEGACY_TO_CAPABILITY[legacy_name]
        source = by_new_name.get(capability_name_of)
        if source is None:
            continue
        impl = source.kwargs_fn

        def _wrap(_impl=impl, **kwargs: Any) -> Any:
            try:
                return _impl(**kwargs)
            except ToolError as exc:
                raise WorkspaceToolError(exc.message, code=exc.code, detail=exc.detail)

        tools.append(
            ToolSpec(
                name=legacy_name,
                description=source.description,
                kwargs_fn=_wrap,
                parameters=source.parameters,
                category="workspace",
                permissions=list(source.permissions),
                safety_level=source.safety_level,
                capability=capability_name,
                result_format=source.result_format,
                validate_fn=source.validate_fn,
                requires_approval=source.requires_approval,
            )
        )
    return tools


def tool_schemas() -> Dict[str, Dict[str, Any]]:
    """Return the JSON schemas for every workspace tool (legacy API)."""
    by_new_name = {
        t.name: t for t in _build_capability_tools(CapabilityContext(roots=[]))
    }
    schemas: Dict[str, Dict[str, Any]] = {}
    for legacy_name, capability_name_of in _LEGACY_TO_CAPABILITY.items():
        source = by_new_name.get(capability_name_of)
        if source is not None:
            schemas[legacy_name] = source.parameters or {}
    return schemas


__all__ = [
    "build_workspace_tools",
    "tool_schemas",
    "WorkspaceToolError",
    "normalize_path",
]
