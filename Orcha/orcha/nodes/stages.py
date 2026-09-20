"""
orcha.nodes.stages
==================
Graph-node wrappers for every ORCHA2 orchestration stage.

Each ``*Node`` holds a reference to the legacy stage object and delegates
``run()`` to it. The adapter is zero-overhead: it just calls the stage's
method with the packet and returns the result. Because every ORCHA2 stage
already returns a forked OrchaPacket, the contract is satisfied without any
translation.

Usage::

    from orcha.orchestration.decomposer import SmartDecomposer
    from orcha.nodes.stages import DecomposeNode

    node = DecomposeNode(SmartDecomposer())
    graph.add_node(node, entry=True)

Stage-to-Node mapping
----------------------
SmartDecomposer   → DecomposeNode   (decompose(packet) → packet)
BudgetPlanner     → PlanNode        (plan(packet) → packet)
ExpertSelector    → SelectNode      (select(packet) → packet)
ParallelExecutor  → ExecuteNode     (await execute(packet) → packet)
Aggregator        → AggregateNode   (await aggregate(packet) → packet)
Evaluator         → EvaluateNode    (evaluate(packet) → packet)
RetryController   → RetryNode       (decide(packet) → packet)
"""
from __future__ import annotations

from typing import Optional

from ..core.packets import OrchaPacket
from ..graph.context import RunContext
from ..graph.node import Node


class DecomposeNode(Node):
    """
    Wraps ``SmartDecomposer.decompose(packet)`` as a graph node.

    Parameters
    ----------
    stage      A ``SmartDecomposer`` instance.
    timeout_s  Per-node timeout (None = no timeout).
    retries    Retry budget (default 0).
    """

    def __init__(
        self,
        stage: object,
        timeout_s: Optional[float] = None,
        retries: int = 0,
    ) -> None:
        self.name = "decompose"
        self.timeout_s = timeout_s
        self.retries = retries
        self._stage = stage

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        return self._stage.decompose(packet)


class PlanNode(Node):
    """
    Wraps ``BudgetPlanner.plan(packet)`` as a graph node.

    Parameters
    ----------
    stage      A ``BudgetPlanner`` instance.
    timeout_s  Per-node timeout.
    retries    Retry budget.
    """

    def __init__(
        self,
        stage: object,
        timeout_s: Optional[float] = None,
        retries: int = 0,
    ) -> None:
        self.name = "plan"
        self.timeout_s = timeout_s
        self.retries = retries
        self._stage = stage

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        return self._stage.plan(packet)


class SelectNode(Node):
    """
    Wraps ``ExpertSelector.select(packet)`` as a graph node.

    Parameters
    ----------
    stage      An ``ExpertSelector`` instance.
    timeout_s  Per-node timeout.
    retries    Retry budget.
    """

    def __init__(
        self,
        stage: object,
        timeout_s: Optional[float] = None,
        retries: int = 0,
    ) -> None:
        self.name = "select"
        self.timeout_s = timeout_s
        self.retries = retries
        self._stage = stage

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        return self._stage.select(packet)


class ExecuteNode(Node):
    """
    Wraps ``ParallelExecutor.execute(packet)`` as a graph node.

    The executor is already async, so this is a thin pass-through.

    Parameters
    ----------
    stage      A ``ParallelExecutor`` instance.
    timeout_s  Per-node timeout (overrides executor's internal timeout).
    retries    Retry budget.
    """

    def __init__(
        self,
        stage: object,
        timeout_s: Optional[float] = None,
        retries: int = 0,
    ) -> None:
        self.name = "execute"
        self.timeout_s = timeout_s
        self.retries = retries
        self._stage = stage

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        return await self._stage.execute(packet)


class AggregateNode(Node):
    """
    Wraps ``Aggregator.aggregate(packet)`` as a graph node.

    The aggregator is already async, so this is a thin pass-through.

    Parameters
    ----------
    stage      An ``Aggregator`` instance.
    timeout_s  Per-node timeout.
    retries    Retry budget.
    """

    def __init__(
        self,
        stage: object,
        timeout_s: Optional[float] = None,
        retries: int = 0,
    ) -> None:
        self.name = "aggregate"
        self.timeout_s = timeout_s
        self.retries = retries
        self._stage = stage

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        return await self._stage.aggregate(packet)


class EvaluateNode(Node):
    """
    Wraps ``Evaluator.evaluate(packet)`` as a graph node.

    Parameters
    ----------
    stage      An ``Evaluator`` instance.
    timeout_s  Per-node timeout.
    retries    Retry budget.
    """

    def __init__(
        self,
        stage: object,
        timeout_s: Optional[float] = None,
        retries: int = 0,
    ) -> None:
        self.name = "evaluate"
        self.timeout_s = timeout_s
        self.retries = retries
        self._stage = stage

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        return self._stage.evaluate(packet)


class RetryNode(Node):
    """
    Wraps ``RetryController.decide(packet)`` as a graph node.

    This node produces a ``PacketKind.RETRY`` packet with ``retry`` (bool)
    and ``retry_reason`` in the payload. The graph's conditional edge reads
    ``retry`` to decide whether to loop back or terminate.

    Parameters
    ----------
    stage      A ``RetryController`` instance.
    timeout_s  Per-node timeout.
    retries    Retry budget.
    """

    def __init__(
        self,
        stage: object,
        timeout_s: Optional[float] = None,
        retries: int = 0,
    ) -> None:
        self.name = "retry"
        self.timeout_s = timeout_s
        self.retries = retries
        self._stage = stage

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        return self._stage.decide(packet)


__all__ = [
    "DecomposeNode", "PlanNode", "SelectNode",
    "ExecuteNode", "AggregateNode", "EvaluateNode", "RetryNode",
]
