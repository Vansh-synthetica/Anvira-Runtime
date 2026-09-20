"""
orcha.graph
===========
The ORCHA3 graph execution engine.

Core components
--------------
- **Graph**: declarative topology builder (nodes, edges, conditional edges,
  scatter/gather).
- **GraphRuntime**: executes a Graph to completion with timeout, retry,
  checkpointing, event streaming, and budget enforcement.
- **Node**: abstract compute unit; ``to_node()`` wraps legacy functions.
- **GatherNode**: base for fan-in nodes.
- **RunContext**: frozen per-run handle (cancel token, store, emitter).
- **Store**: checkpoint durability interface (MemoryStore, FileStore, SqliteStore).

Quick start::

    from orcha.graph import Graph, GraphRuntime
    from orcha.graph.node import to_node

    async def my_node(pkt, ctx):
        return pkt.fork(pkt.kind, result="done")

    g = Graph(name="demo")
    g.add_node(to_node(my_node, name="worker"), entry=True)
    g.add_edge("worker", END)
    g.validate()

    rt = GraphRuntime(g)
    result = await rt.run("hello world")
    print(result.answer)
"""
from .context import (
    CancelToken,
    EventEmitter,
    EventSubscriber,
    RunContext,
    RunEvent,
    ScatterResult,
    current_trace_id,
    trace_id_var,
)
from .edge import (
    ConditionalEdge,
    Edge,
    END,
    FanInEdge,
    FanOutEdge,
    MergeFn,
    Predicate,
)
from .errors import (
    BudgetExceeded,
    Cancelled,
    GatherError,
    GraphError,
    GraphInvalid,
    NoRoute,
    NodeFailed,
    NodeTimeout,
)
from .graph import Graph
from .node import GatherNode, Node, to_node
from .runtime import CHECKPOINT_ALWAYS, GraphRuntime
from .store import (
    Checkpoint,
    FileStore,
    MemoryStore,
    SqliteStore,
    Store,
)

__all__ = [
    # Topology
    "Graph", "Node", "GatherNode", "to_node",
    "Edge", "ConditionalEdge", "FanOutEdge", "FanInEdge",
    "END", "Predicate", "MergeFn",
    # Execution
    "GraphRuntime", "CHECKPOINT_ALWAYS",
    # Context
    "RunContext", "RunEvent", "EventEmitter", "EventSubscriber",
    "CancelToken", "ScatterResult",
    "current_trace_id", "trace_id_var",
    # Errors
    "GraphError", "GraphInvalid", "NodeTimeout", "NodeFailed",
    "BudgetExceeded", "Cancelled", "GatherError", "NoRoute",
    # Durability
    "Store", "Checkpoint", "MemoryStore", "FileStore", "SqliteStore",
]
