"""
orcha.integrations.deepagents
=============================
Boundary around Deep Agents — the primary high-level agent harness for
subagents, supervisor delegation, the task tool, and isolated context.

Orcha core never imports deepagents directly; it goes through this
boundary. The boundary is lazy: ``DeepAgentsBoundary()`` is cheap, and
``load()`` imports deepagents on demand, raising
``IntegrationUnavailable`` with an install hint when the optional
``orcha[agents]`` extra is missing.

Orcha-owned controls that remain in place around the harness:
permissions, workspace boundaries, local model adapters, AICL, Nomi,
and the Anvira event contract.
"""
from __future__ import annotations

from ..base import Boundary

__all__ = ["DeepAgentsBoundary"]


class DeepAgentsBoundary(Boundary):
    name = "deepagents"
    package = "deepagents"
    extra = "agents"

    def load(self) -> object:
        """Return the ``deepagents`` module (raises if not installed)."""
        return self._import("deepagents")