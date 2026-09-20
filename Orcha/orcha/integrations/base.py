"""
orcha.integrations.base
=======================
Shared machinery for Orcha's external-framework integration boundaries.

Every external framework (LangChain, LangGraph, Deep Agents, ragas,
Phoenix) is reachable ONLY through a boundary in ``orcha.integrations``.
Boundaries are lazy: importing a boundary module never imports the
external framework. The framework is imported on demand via ``load()``,
and a clear ``IntegrationUnavailable`` error names the exact extra to
install when it is missing.

This keeps Orcha's core zero-dependency and local-first: a machine with
no Lang ecosystem packages installed still imports, runs, and tests
cleanly, and Anvira's settings screen can call ``availability_report()``
to show exactly which integrations are present.
"""
from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass
from typing import Optional

from ..core.packets import OrchaPacket
from ..graph.context import RunContext

__all__ = [
    "IntegrationUnavailable", "IntegrationStatus", "Boundary",
    "ExecutionEngine", "availability_report",
]


class IntegrationUnavailable(RuntimeError):
    """Raised when an external framework is not installed or not usable."""


@dataclass(frozen=True)
class IntegrationStatus:
    """
    Snapshot of one integration's availability.

    Attributes
    ----------
    name              Boundary name, e.g. ``"langgraph"``.
    package           Distribution to install, e.g. ``"langgraph"``.
    extra             The ``orcha[extra]`` that installs it, if any.
    available         Whether the distribution is importable right now.
    version           Installed version, or None when unavailable.
    """

    name: str
    package: str
    extra: Optional[str]
    available: bool
    version: Optional[str]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "package": self.package,
            "extra": self.extra,
            "available": self.available,
            "version": self.version,
        }


class Boundary:
    """
    Base class for an external-framework integration boundary.

    Subclasses set ``name`` and ``package`` (and optionally ``extra``)
    and provide a ``load()`` that returns the lazily imported module or
    raises ``IntegrationUnavailable``.
    """

    name: str = ""
    package: str = ""
    extra: Optional[str] = None

    def _import(self, module_name: Optional[str] = None) -> object:
        """Import the module, raising a helpful error when missing."""
        module_name = module_name or self.package
        try:
            return importlib.import_module(module_name)
        except ImportError:
            hint = f"pip install \"orcha[{self.extra}]\" or `pip install {self.package}`"
            raise IntegrationUnavailable(
                f"{self.name} is not installed ({module_name}). {hint}."
            ) from None

    def status(self) -> IntegrationStatus:
        """Report availability without importing anything heavy."""
        try:
            version = importlib.metadata.version(self.package)
            return IntegrationStatus(
                name=self.name, package=self.package, extra=self.extra,
                available=True, version=version,
            )
        except importlib.metadata.PackageNotFoundError:
            return IntegrationStatus(
                name=self.name, package=self.package, extra=self.extra,
                available=False, version=None,
            )


class ExecutionEngine:
    """
    Orcha-owned execution contract that any engine may implement.

    A native graph runtime, a LangGraph-backed engine, or a future
    Orcha-native replacement all execute the same way: they take an
    OrchaPacket and a RunContext and return an (updated) OrchaPacket.
    Anvira, AICL, and the public API only ever see this contract.
    """

    async def execute(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        raise NotImplementedError


def availability_report() -> dict:
    """Status of every registered integration boundary, by name."""
    from . import _BOUNDARIES
    return {name: boundary.status() for name, boundary in _BOUNDARIES.items()}
