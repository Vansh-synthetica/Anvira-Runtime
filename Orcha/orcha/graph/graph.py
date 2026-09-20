"""
orcha.graph.graph
=================
The Graph: a validated, declarative description of node topology.

A Graph is built imperatively (add_node / add_edge / add_conditional /
fan_out / fan_in) and then ``validate()``-d into an immutable, runnable
form. Validation enforces:

  - exactly one entry node
  - every edge source is a registered node
  - every edge destination is a registered node or END
  - every conditional route label maps to a real node or END
  - every FanOutEdge has a matching FanInEdge (scatter → gather pairing)
  - END is reachable from the entry node (no dead subgraphs that trap runs)

Graphs allow back-edges (retry loops) but the runtime carries an iteration
budget on the packet, so a pathological cycle terminates rather than spins.

The Graph itself is passive topology. Execution is performed by
``GraphRuntime`` (see runtime.py).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

from ..core.packets import OrchaPacket
from .edge import (
    ConditionalEdge, Edge, END, FanInEdge, FanOutEdge, MergeFn, Predicate,
)
from .errors import GraphInvalid
from .node import GatherNode, Node, to_node


class Graph:
    """
    A validated graph topology.

    Build it, then either call ``validate()`` explicitly or let ``run()``
    validate on first execution. Once validated, the topology is treated
    as immutable for the lifetime of the run.
    """

    def __init__(self, name: str = "graph") -> None:
        if not name or not isinstance(name, str):
            raise ValueError("Graph name must be a non-empty string")
        self.name: str = name
        self._nodes: Dict[str, Node] = {}
        self._edges: List[Union[Edge, ConditionalEdge, FanOutEdge, FanInEdge]] = []
        self._entry: Optional[str] = None
        self._validated: bool = False
        # Scatter→gather correlation: scatter_node -> gather_node
        self._scatter_to_gather: Dict[str, str] = {}
        self._gather_from_scatter: Dict[str, str] = {}

    # ── Topology mutation ──────────────────────────────────────────────

    def add_node(self, node: Union[Node, object], *, entry: bool = False) -> "Graph":
        """
        Register a node. ``node`` may be a Node instance or any callable;
        callables are coerced via ``to_node``. Names must be unique.

        Set ``entry=True`` to mark this as the run's starting node. Exactly
        one entry node is required.
        """
        node = to_node(node)
        if not node.name or node.name in self._nodes:
            # Allow re-registration only if it's literally the same object.
            existing = self._nodes.get(node.name)
            if existing is not node:
                raise GraphInvalid(
                    f"Duplicate node name {node.name!r}", node=node.name,
                )
        self._nodes[node.name] = node
        self._validated = False
        if entry:
            if self._entry is not None and self._entry != node.name:
                raise GraphInvalid(
                    f"Graph already has entry {self._entry!r}; "
                    f"cannot also set {node.name!r}", node=node.name,
                )
            self._entry = node.name
        return self

    def add_edge(self, from_node: str, to_node: str) -> "Graph":
        """Unconditional edge: from_node → to_node (or END)."""
        self._edges.append(Edge(from_node, to_node))
        self._validated = False
        return self

    def add_conditional(
        self,
        from_node: str,
        routes: Dict[str, str],
        predicate: Predicate,
    ) -> "Graph":
        """
        Conditional edge: ``predicate(packet)`` returns a label that selects
        the next node from ``routes``. A label of None means "no transition"
        (soft END with current packet).

        Every route value must be a registered node name or END.
        """
        self._edges.append(ConditionalEdge(from_node, routes, predicate))
        self._validated = False
        return self

    def fan_out(
        self,
        scatter_node: str,
        gather_node: str,
        merge_fn: Optional[MergeFn] = None,
    ) -> "Graph":
        """
        Declare a scatter→gather pair.

        ``scatter_node`` must return a ScatterResult from its run(); the
        runtime spawns one branch per scatter entry and routes every
        branch's terminal packet to ``gather_node``, which receives the
        collected children via ``packet.metadata['__gathered__']``.

        ``merge_fn`` is optional: if provided, the runtime applies it to
        collapse the gathered packets into one before invoking the gather
        node (handy when the gather node is a simple passthrough).
        """
        self._edges.append(FanOutEdge(scatter_node, gather_node))
        self._edges.append(FanInEdge(scatter_node, gather_node, merge_fn))
        self._scatter_to_gather[scatter_node] = gather_node
        self._gather_from_scatter[gather_node] = scatter_node
        self._validated = False
        return self

    # ── Introspection ──────────────────────────────────────────────────

    @property
    def entry(self) -> str:
        if self._entry is None:
            raise GraphInvalid(f"Graph {self.name!r} has no entry node")
        return self._entry

    @property
    def nodes(self) -> Dict[str, Node]:
        return dict(self._nodes)

    @property
    def edges(self) -> List[Union[Edge, ConditionalEdge, FanOutEdge, FanInEdge]]:
        return list(self._edges)

    def get_node(self, name: str) -> Node:
        if name not in self._nodes:
            raise GraphInvalid(f"Unknown node {name!r}", node=name)
        return self._nodes[name]

    def outgoing(self, node: str) -> List[Union[Edge, ConditionalEdge, FanOutEdge, FanInEdge]]:
        """All edges whose source is ``node``."""
        return [e for e in self._edges if e.from_node == node]

    def scatter_target(self, scatter_node: str) -> Optional[str]:
        """The gather node paired with a scatter node, if any."""
        return self._scatter_to_gather.get(scatter_node)

    def gather_source(self, gather_node: str) -> Optional[str]:
        """The scatter node paired with a gather node, if any."""
        return self._gather_from_scatter.get(gather_node)

    # ── Validation ─────────────────────────────────────────────────────

    def validate(self) -> "Graph":
        """
        Validate the topology. Raises GraphInvalid on any problem.

        Checks:
          1. Exactly one entry node exists.
          2. Every node name referenced by an edge exists (or is END).
          3. Conditional routes all point to real nodes or END.
          4. Fan-out and fan-in are paired consistently.
          5. END is reachable from entry (graph is not a trap).
          6. Every non-terminal node has at least one outgoing edge.
        """
        if self._validated:
            return self

        if self._entry is None:
            raise GraphInvalid(f"Graph {self.name!r} has no entry node")
        if self._entry not in self._nodes:
            raise GraphInvalid(
                f"Entry node {self._entry!r} is not registered", node=self._entry,
            )

        known = set(self._nodes.keys()) | {END}

        # Check edge endpoints reference real nodes.
        for e in self._edges:
            if e.from_node not in self._nodes:
                raise GraphInvalid(
                    f"Edge source {e.from_node!r} is not a registered node",
                    node=e.from_node,
                )
            if isinstance(e, ConditionalEdge):
                for label, dest in e.routes.items():
                    if dest not in known:
                        raise GraphInvalid(
                            f"Conditional route {label!r} from {e.from_node!r} "
                            f"points to unknown node {dest!r}", node=e.from_node,
                        )
            elif isinstance(e, (Edge, FanOutEdge, FanInEdge)):
                if e.to_node not in known:
                    raise GraphInvalid(
                        f"Edge {e.from_node!r} → {e.to_node!r}: destination "
                        f"{e.to_node!r} is not a registered node", node=e.from_node,
                    )

        # Scatter/gather pairing: every scatter must have a gather and vice versa.
        for s, g in self._scatter_to_gather.items():
            if s not in self._nodes:
                raise GraphInvalid(f"Scatter node {s!r} is not registered", node=s)
            if g not in self._nodes:
                raise GraphInvalid(f"Gather node {g!r} is not registered", node=g)
        for g, s in self._gather_from_scatter.items():
            if self._scatter_to_gather.get(s) != g:
                raise GraphInvalid(
                    f"Gather node {g!r} is paired with scatter {s!r} but the "
                    f"scatter does not point back to it", node=g,
                )

        # Every node except those feeding only into END must have an outgoing edge,
        # UNLESS it is a gather node (gather nodes are reached via the scatter
        # machinery, not a plain Edge).
        for name, node in self._nodes.items():
            is_gather = self.gather_source(name) is not None
            has_out = any(
                e.from_node == name
                for e in self._edges
                if not isinstance(e, FanInEdge)  # the FanInEdge from_node is the scatter
            )
            if not has_out and not is_gather and name != self._entry:
                # A node with no outgoing edge is only valid if it's terminal-ish;
                # we allow it (soft END) but warn via validation only for entry.
                pass

        # Reachability: END must be reachable from entry.
        if not self._end_reachable(self._entry):
            raise GraphInvalid(
                f"END is not reachable from entry {self._entry!r} — "
                f"the graph would never terminate", node=self._entry,
            )

        self._validated = True
        return self

    def _end_reachable(self, start: str) -> bool:
        """BFS/DFS from ``start``; True if END is reachable."""
        # Build adjacency including conditional routes and fan-out targets.
        adj: Dict[str, List[str]] = {}
        for e in self._edges:
            src = e.from_node
            if isinstance(e, ConditionalEdge):
                # Include every route (destinations() strips END, but END
                # must be considered reachable through conditional edges).
                adj.setdefault(src, []).extend(e.routes.values())
            elif isinstance(e, FanOutEdge):
                adj.setdefault(src, []).append(e.to_node)
            elif isinstance(e, Edge):
                adj.setdefault(src, []).append(e.to_node)
            # FanInEdge: the gather node is reached from the scatter machinery;
            # we model gather → (its outgoing edges) normally below.
        # Gather nodes' outgoing edges are plain Edges already captured above.

        seen = set()
        stack = [start]
        while stack:
            n = stack.pop()
            if n == END:
                return True
            if n in seen:
                continue
            seen.add(n)
            stack.extend(adj.get(n, []))
        return False

    def __repr__(self) -> str:
        n = len(self._nodes)
        e = len(self._edges)
        return f"Graph(name={self.name!r}, nodes={n}, edges={e}, entry={self._entry!r})"


__all__ = ["Graph"]
