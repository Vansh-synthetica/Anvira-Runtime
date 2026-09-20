"""
orcha.capabilities.registry
===========================
CapabilityRegistry — the central catalogue of capabilities and the factory
that builds a :class:`ToolExecutor` for a given agent's declared capabilities.

Every capability module exposes ``CAPABILITY_NAME``, ``CAPABILITY_LABEL``,
``CAPABILITY_DESCRIPTION`` and ``build_tools(ctx) -> List[ToolSpec]``.
The registry discovers them explicitly (no dynamic imports) and validates
declared capability names before building an executor.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from . import code_intelligence, diagnostics, filesystem, git, search, terminal, web, workspace
from .base import CapabilityContext, PermissionPolicy, ToolExecutor

Builder = Callable[[CapabilityContext], Any]

_DEFAULT_BUILDERS: Dict[str, Builder] = {
    "filesystem": filesystem.build_tools,
    "workspace": workspace.build_tools,
    "search": search.build_tools,
    "web": web.build_tools,
    "terminal": terminal.build_tools,
    "git": git.build_tools,
    "diagnostics": diagnostics.build_tools,
    "code_intelligence": code_intelligence.build_tools,
}

_LABELS: Dict[str, str] = {
    "filesystem": filesystem.CAPABILITY_LABEL,
    "workspace": workspace.CAPABILITY_LABEL,
    "search": search.CAPABILITY_LABEL,
    "web": web.CAPABILITY_LABEL,
    "terminal": terminal.CAPABILITY_LABEL,
    "git": git.CAPABILITY_LABEL,
    "diagnostics": diagnostics.CAPABILITY_LABEL,
    "code_intelligence": code_intelligence.CAPABILITY_LABEL,
}

_DESCRIPTIONS: Dict[str, str] = {
    "filesystem": filesystem.CAPABILITY_DESCRIPTION,
    "workspace": workspace.CAPABILITY_DESCRIPTION,
    "search": search.CAPABILITY_DESCRIPTION,
    "web": web.CAPABILITY_DESCRIPTION,
    "terminal": terminal.CAPABILITY_DESCRIPTION,
    "git": git.CAPABILITY_DESCRIPTION,
    "diagnostics": diagnostics.CAPABILITY_DESCRIPTION,
    "code_intelligence": code_intelligence.CAPABILITY_DESCRIPTION,
}


class CapabilityRegistry:
    """
    Mutable registry of capability builders.

    Example
    -------
    >>> registry = CapabilityRegistry().register_defaults()
    >>> executor = registry.build(["filesystem", "search"], ctx)
    """

    def __init__(self) -> None:
        self._builders: Dict[str, Builder] = {}
        self._labels: Dict[str, str] = {}
        self._descriptions: Dict[str, str] = {}

    # ── registration ─────────────────────────────────────────────────────────
    def add_capability(
        self,
        name: str,
        label: str,
        description: str,
        builder: Builder,
    ) -> "CapabilityRegistry":
        """Register a capability: name + label + description + tool builder."""
        self._builders[name] = builder
        self._labels[name] = label
        self._descriptions[name] = description
        return self

    def register_defaults(self) -> "CapabilityRegistry":
        """Register all seven built-in capabilities."""
        for name, builder in _DEFAULT_BUILDERS.items():
            self.add_capability(name, _LABELS[name], _DESCRIPTIONS[name], builder)
        return self

    # ── introspection ────────────────────────────────────────────────────────
    def names(self) -> List[str]:
        return list(self._builders)

    def has(self, name: str) -> bool:
        return name in self._builders

    def capabilities(self) -> List[Dict[str, Any]]:
        return [
            {"name": name, "label": self._labels.get(name, name),
             "description": self._descriptions.get(name, "")}
            for name in self._builders
        ]

    def tool_counts(self, ctx: CapabilityContext) -> Dict[str, int]:
        return {name: len(builder(ctx)) for name, builder in self._builders.items()}

    def resolve_capabilities(self, declared: Optional[List[str]]) -> List[str]:
        """Validate declared capability names and return the canonical list."""
        if not declared:
            return []
        resolved: List[str] = []
        for name in declared:
            if not self.has(name):
                raise ValueError(
                    f"Unknown capability '{name}'. Available: {', '.join(self.names()) or 'none'}"
                )
            if name not in resolved:
                resolved.append(name)
        return resolved

    # ── building ─────────────────────────────────────────────────────────────
    def build(
        self,
        capability_names: Optional[List[str]] = None,
        ctx: Optional[CapabilityContext] = None,
        policy: Optional[PermissionPolicy] = None,
    ) -> ToolExecutor:
        """
        Build a ToolExecutor for the given capabilities (default: all).
        ``capability_names`` are resolved (validated) first.
        """
        names = self.resolve_capabilities(capability_names)
        if not names:
            names = self.names()
        ctx = ctx or CapabilityContext()
        tools = []
        for name in names:
            tools.extend(self._builders[name](ctx))
        return ToolExecutor(tools, policy=policy)

    def schemas(
        self,
        capability_names: Optional[List[str]] = None,
        ctx: Optional[CapabilityContext] = None,
    ) -> List[Dict[str, Any]]:
        return self.build(capability_names, ctx=ctx).schemas()

    def describe_all(
        self,
        capability_names: Optional[List[str]] = None,
        ctx: Optional[CapabilityContext] = None,
    ) -> Dict[str, Any]:
        executor = self.build(capability_names, ctx=ctx)
        by_capability: Dict[str, List[Dict[str, Any]]] = {}
        for tool in executor.describe():
            by_capability.setdefault(tool.get("capability") or "general", []).append(tool)
        return {"capabilities": self.capabilities(), "tools_by_capability": by_capability}


__all__ = ["CapabilityRegistry"]
