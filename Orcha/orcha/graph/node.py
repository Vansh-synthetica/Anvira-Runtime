"""
orcha.graph.node
================
The Node contract — the universal compute unit of an ORCHA3 graph.

A node is a typed coroutine ``(packet, ctx) -> packet`` with optional
lifecycle hooks, a per-node timeout, and a retry budget. The runtime
wraps every ``run`` call in:

  1. cancellation check
  2. timeout enforcement (asyncio.wait_for)
  3. fault normalization (any raised exception becomes NodeFailed)
  4. trace stamping
  5. event emission (node_start / node_end)
  6. checkpointing

so a node author only writes the transformation itself.

Adapters
--------
- ``to_node(fn)`` wraps any legacy ORCHA2 stage — sync or async, signature
  ``(packet)->packet`` — as a Node. This is how the existing 7 stages become
  first-class graph nodes with zero porting effort.
- ``GatherNode`` is the base for fan-in nodes; it reads the gathered child
  packets from the context and produces the merged packet.
"""
from abc import ABC, abstractmethod
import asyncio
import functools
import inspect
import time
from typing import Any, Awaitable, Callable, List, Optional, Union

from ..core.packets import OrchaPacket, PacketKind
from .context import RunContext
from .context import ScatterResult  # noqa: F401  (re-exported for convenience)


# A legacy stage function: sync or async, (packet) -> packet.
StageFn = Callable[[OrchaPacket], Union[OrchaPacket, Awaitable[OrchaPacket]]]

# Sentinel kind used when a node does not change the packet lifecycle kind.
PASS_THROUGH = None


class Node(ABC):
    """
    Base class for all graph nodes.

    Subclasses set ``name`` and implement ``run``. ``timeout_s`` and
    ``retries`` are read by the runtime; ``retries`` gives a node up to
    that many *additional* attempts (so retries=2 → up to 3 executions)
    on NodeFailed before the failure propagates.
    """

    name: str = "node"
    timeout_s: Optional[float] = None
    retries: int = 0
    # If True, the runtime treats this node as a scatter point and expects a
    # ScatterResult from run(). Declared via FanOutEdge, not usually here.
    is_scatter: bool = False
    is_gather: bool = False

    @abstractmethod
    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        """Transform the packet. Must return a forked child, never mutate."""
        raise NotImplementedError

    # ── Optional lifecycle hooks (defaults are no-ops) ──────────────────

    async def on_enter(self, packet: OrchaPacket, ctx: RunContext) -> None:
        """Called before run(). Useful for warming resources."""

    async def on_success(
        self, packet: OrchaPacket, result: OrchaPacket, ctx: RunContext,
    ) -> None:
        """Called after a successful run()."""

    async def on_failure(
        self, packet: OrchaPacket, exc: BaseException, ctx: RunContext,
    ) -> None:
        """Called if run() raises (after retries are exhausted)."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class _FunctionNode(Node):
    """
    Node wrapping a plain function (the legacy stage contract).

    The function may be sync or async and takes either ``(packet)`` or
    ``(packet, ctx)`` — the adapter inspects the signature so existing
    ORCHA2 stages (which take only packet) work unchanged, while new nodes
    can opt into the context.
    """

    def __init__(
        self,
        fn: StageFn,
        name: Optional[str] = None,
        timeout_s: Optional[float] = None,
        retries: int = 0,
    ) -> None:
        self._fn = fn
        self.name = name or getattr(fn, "__name__", None) or "fn_node"
        if timeout_s is not None:
            self.timeout_s = timeout_s
        self.retries = retries
        # Does the wrapped fn want the context? Inspect once.
        try:
            sig = inspect.signature(fn)
            self._wants_ctx = len(sig.parameters) >= 2
        except (TypeError, ValueError):
            self._wants_ctx = False
        functools.update_wrapper(self, fn, updated=())

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        result = self._fn(packet, ctx) if self._wants_ctx else self._fn(packet)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, OrchaPacket):
            raise TypeError(
                f"Node {self.name!r} returned {type(result).__name__}, "
                f"expected OrchaPacket"
            )
        return result


def to_node(
    fn_or_node: Union[StageFn, Node],
    name: Optional[str] = None,
    timeout_s: Optional[float] = None,
    retries: int = 0,
) -> Node:
    """
    Coerce a function or a Node into a Node.

    - Already a Node: returned as-is (name/timeout overrides applied if given).
    - Callable: wrapped in ``_FunctionNode``.

    This is the bridge that lets every ORCHA2 stage become a graph node
    without modification::

        from orcha.orchestration import get_decomposer
        DecomposeNode = to_node(get_decomposer().decompose, name="decompose")
    """
    if isinstance(fn_or_node, Node):
        if name is not None:
            fn_or_node.name = name
        if timeout_s is not None:
            fn_or_node.timeout_s = timeout_s
        if retries:
            fn_or_node.retries = retries
        return fn_or_node
    if callable(fn_or_node):
        return _FunctionNode(fn_or_node, name=name, timeout_s=timeout_s, retries=retries)
    raise TypeError(f"to_node() expected a callable or Node, got {type(fn_or_node).__name__}")


class GatherNode(Node):
    """
    Base class for fan-in (gather) nodes.

    Subclasses implement ``gather`` which receives the list of completed
    child packets from the upstream scatter and returns the merged packet.
    The runtime passes the gathered packets via ``ctx``-attached metadata
    on the incoming packet under the ``__gathered__`` key.
    """

    is_gather = True

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        gathered: List[OrchaPacket] = packet.payload.get("__gathered__", [])
        return await self.gather(packet, gathered, ctx)

    async def gather(
        self,
        packet: OrchaPacket,
        children: List[OrchaPacket],
        ctx: RunContext,
    ) -> OrchaPacket:
        """Merge child packets into one. Override in subclasses."""
        raise NotImplementedError


__all__ = ["Node", "to_node", "GatherNode", "ScatterResult", "StageFn"]
