"""
orcha.graph.errors
==================
Normalized error taxonomy for the graph runtime.

Every runtime failure is translated into one of these types so that callers
(API, CLI, embedding app) can switch on a stable, documented set of outcomes
rather than catching bare ``Exception`` and string-matching messages.

Hierarchy
---------
GraphError                  — base for everything the runtime raises
├── GraphInvalid            — topology problem (missing entry, dangling edge…)
├── NodeTimeout             — a node exceeded its timeout_s
├── NodeFailed              — a node raised; carries the node name + cause
├── BudgetExceeded          — cost / latency / iteration budget exhausted
├── Cancelled               — cooperative cancellation fired
├── GatherError             — a fan-in failed to merge its branches
└── NoRoute                 — a conditional edge predicate returned no label
"""
from __future__ import annotations

from typing import Optional


class GraphError(Exception):
    """Base for all graph-runtime errors."""

    def __init__(self, message: str, *, node: Optional[str] = None) -> None:
        super().__init__(message)
        self.node = node


class GraphInvalid(GraphError):
    """Raised at validate()/run() time when the topology is malformed."""


class NodeTimeout(GraphError):
    """A node exceeded its ``timeout_s`` limit."""


class NodeFailed(GraphError):
    """
    A node raised an unrecoverable exception.

    Carries the originating node name and the original exception so callers
    can render a precise diagnostic. Note this is distinct from a *leaf*
    expert producing a failed ExpertResult — that is normal fault isolation
    and does not raise; it is recorded in the packet and the run continues.
    """


class BudgetExceeded(GraphError):
    """The run exhausted its cost, latency, or iteration budget."""


class Cancelled(GraphError):
    """Cooperative cancellation was requested and honored."""


class GatherError(GraphError):
    """A fan-in (gather) node could not merge its incoming branches."""


class NoRoute(GraphError):
    """
    A conditional edge predicate returned a label with no registered
    destination, or returned None when one was required.
    """
