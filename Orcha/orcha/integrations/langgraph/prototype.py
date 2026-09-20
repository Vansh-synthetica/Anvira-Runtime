"""
orcha.integrations.langgraph.prototype
======================================
Bridge Orcha Nodes onto LangGraph and a prototype graph builder.

``orcha_node_to_langgraph`` adapts any Orcha Node (``run(packet, ctx)
-> packet``) to a LangGraph node. The run context is injected per
invoke via ``config["configurable"]["orcha_ctx"]`` — it is never part
of the checkpointed state.

``build_prototype_graph`` chains a list of Orcha nodes into a linear
LangGraph StateGraph whose state is exactly ``{"packet": OrchaPacket}``.
It exists to prove the engine seam (Stage 3); the real graph builders
(Stage 4+) will construct their own StateGraphs over the same bridge.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ...core.packets import OrchaPacket
from ...graph.node import Node
from ..base import IntegrationUnavailable

__all__ = ["orcha_node_to_langgraph", "build_prototype_graph", "orcha_serde"]


def orcha_serde(extra_msgpack_modules=None):
    """
    A LangGraph serde that round-trips OrchaPacket and friends.

    Registers every class defined in ``orcha.core.packets`` in the
    msgpack allowlist so checkpoint (de)serialization never emits
    deprecation warnings or gets blocked when strict mode flips on.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    import orcha.core.packets as _packets

    modules = [
        (m.__module__, m.__name__)
        for m in vars(_packets).values()
        if isinstance(m, type) and getattr(m, "__module__", None) == _packets.__name__
    ]
    modules += list(extra_msgpack_modules or [])
    return JsonPlusSerializer(
        allowed_json_modules=True,
        allowed_msgpack_modules=modules,
    )


def _langgraph():
    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.checkpoint.memory import MemorySaver
        return END, START, StateGraph, MemorySaver
    except ImportError:
        raise IntegrationUnavailable(
            "LangGraph is not installed. Install with "
            "`pip install \"orcha[lang]\"`."
        ) from None


def orcha_node_to_langgraph(node: Node):
    """
    Adapt an Orcha Node to a LangGraph node.

    The returned node reads ``state["packet"]``, runs the Orcha node
    with the injected RunContext, and returns ``{"packet": ...}``.
    """
    END, START, StateGraph, MemorySaver = _langgraph()
    from langchain_core.runnables.config import RunnableConfig

    async def _lg_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        ctx = config["configurable"]["orcha_ctx"]
        packet = await node.run(state["packet"], ctx)
        return {"packet": packet}

    _lg_node.__name__ = node.name
    return _lg_node


def build_prototype_graph(
    nodes: Sequence[Node],
    *,
    interrupt_before: Optional[Sequence[str]] = None,
    checkpointer: Optional[Any] = None,
):
    """
    Chain Orcha nodes into a linear LangGraph StateGraph.

    Parameters
    ----------
    nodes              Orcha Nodes, run in order.
    interrupt_before   Node names to stop before (human-in-the-loop).
    checkpointer       LangGraph checkpointer (defaults to MemorySaver).

    Returns
    -------
    A compiled LangGraph ready for ``LangGraphEngine``.
    """
    END, START, StateGraph, MemorySaver = _langgraph()
    import os
    # OrchaPacket/PacketKind are not yet in langgraph's registered msgpack
    # types; keep the deprecation notice non-blocking on future upgrades.
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "false")

    def _packet_state():
        from typing import TypedDict

        class _PacketState(TypedDict):
            packet: OrchaPacket
        return _PacketState

    graph = StateGraph(_packet_state())
    for node in nodes:
        graph.add_node(node.name, orcha_node_to_langgraph(node))
    graph.add_edge(START, nodes[0].name)
    for prev, nxt in zip(nodes, nodes[1:]):
        graph.add_edge(prev.name, nxt.name)
    graph.add_edge(nodes[-1].name, END)

    return graph.compile(
        checkpointer=checkpointer or MemorySaver(serde=orcha_serde()),
        interrupt_before=list(interrupt_before) if interrupt_before else None,
    )