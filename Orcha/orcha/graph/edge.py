"""
orcha.graph.edge
================
Graph edges expressed as data, not control flow.

Edge kinds
----------
Edge              unconditional: from → to
ConditionalEdge   a predicate over the packet picks one of several dests
FanOutEdge        scatter: the node returns a ScatterResult, the runtime
                  spawns one branch per entry and routes them to gather_to
FanInEdge         gather: collects the child packets from a prior fan-out
                  and passes the merged packet downstream

The special destination ``END`` marks a terminal transition: the packet
reaching END becomes the run's final result.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.packets import OrchaPacket

# Sentinel for the terminal destination. Using a unique object (not a str)
# means a graph can never accidentally collide with a node named "END".
END = "__END__"

# A conditional predicate maps (packet) → label. The label must be a key
# in the ConditionalEdge.routes dict (or END).
Predicate = Callable[[OrchaPacket], Optional[str]]

# A fan-in merge function takes the gather node's input packet plus the list
# of completed child packets and returns the merged packet.
MergeFn = Callable[[OrchaPacket, List[OrchaPacket]], OrchaPacket]


class _BaseEdge:
    """Common fields for every edge type."""

    __slots__ = ("from_node", "to_node")

    def __init__(self, from_node: str, to_node: str) -> None:
        self.from_node = from_node
        self.to_node = to_node

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.from_node!r} → {self.to_node!r})"


class Edge(_BaseEdge):
    """Unconditional edge: from_node always transitions to to_node."""


class ConditionalEdge(_BaseEdge):
    """
    A conditional transition: the predicate inspects the packet and returns
    a label; the runtime looks up that label in ``routes`` to find the next
    node.

    The predicate may return None to mean "no transition" — the run then
    terminates with the current packet as the result (a soft END). This is
    how planner/verifier gate decisions are expressed without a retry flag.

    Parameters
    ----------
    from_node  Node whose output the predicate reads.
    routes     {label: destination_node_or_END}. Every possible label the
               predicate can return must be present here.
    predicate  fn(packet) -> label | None.
    """

    __slots__ = ("routes", "predicate")

    def __init__(
        self,
        from_node: str,
        routes: Dict[str, str],
        predicate: Predicate,
    ) -> None:
        if not routes:
            raise ValueError("ConditionalEdge requires at least one route")
        # The "to_node" of a conditional edge is informational only; routing
        # is decided dynamically. We store the routes dict for validation.
        super().__init__(from_node, to_node="<conditional>")
        self.routes: Dict[str, str] = dict(routes)
        self.predicate: Predicate = predicate

    def destinations(self) -> List[str]:
        """All possible destination nodes (excluding END)."""
        return [d for d in self.routes.values() if d != END]


class FanOutEdge(_BaseEdge):
    """
    Declares that ``from_node`` is a scatter node whose ``run`` returns a
    ``ScatterResult``. The runtime spawns one branch per scatter entry and
    routes every branch's terminal packet to ``to_node`` (the gather node).

    The edge itself carries no logic — it is a topology declaration the
    runtime uses to know where to send the scatter output and to validate
    that a matching FanInEdge exists.
    """

    __slots__ = ()


class FanInEdge(_BaseEdge):
    """
    Declares that ``to_node`` is a gather node. ``from_node`` here is the
    conceptual "join point" identifier (typically the scatter node's name)
    used to correlate with the matching FanOutEdge; the runtime supplies
    the gathered child packets to the node via the merge contract.

    The gather node's ``run`` receives the last packet that arrived (or a
    placeholder) plus the full list of gathered packets in the run metadata
    under ``__gathered__``; it returns the merged packet.
    """

    __slots__ = ("merge_fn",)

    def __init__(
        self, from_node: str, to_node: str, merge_fn: Optional[MergeFn] = None,
    ) -> None:
        super().__init__(from_node, to_node)
        self.merge_fn: Optional[MergeFn] = merge_fn


__all__ = [
    "END", "Predicate", "MergeFn",
    "Edge", "ConditionalEdge", "FanOutEdge", "FanInEdge",
]
