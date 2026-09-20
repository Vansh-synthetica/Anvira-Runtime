"""
orcha.integrations.langgraph.engine
===================================
``LangGraphEngine`` — a LangGraph-backed implementation of Orcha's
``ExecutionEngine`` contract.

The engine executes an already-compiled LangGraph (StateGraph) whose
state carries the OrchaPacket. Nodes inside the graph are Orcha Nodes
(wrapped by ``orcha_node_to_langgraph``) or native LangGraph nodes —
the engine does not care which. Anvira, AICL, and the public API only
ever see the ``ExecutionEngine.execute(packet, ctx) -> packet`` seam.

Mapping to LangGraph concepts
-----------------------------
- thread id      → ``ctx.run_id`` (== packet.id == trace id). A run is a
                  thread; every ``execute`` on the same run_id resumes
                  that thread (checkpoints/resume for free).
- run context    → injected per-invoke via ``config["configurable"]
                  ["orcha_ctx"]``; nodes that declare a ``config``
                  parameter receive it. Context is never checkpointed —
                  it is re-injected on every invoke.
- interrupts     → a graph compiled with ``interrupt_before`` (or
                  calling ``interrupt()``) raises ``GraphInterrupt``;
                  the engine surfaces it as ``ExecutionInterrupt``
                  carrying the thread id + snapshot, and
                  ``resume(thread_id, value)`` continues the run.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ...core.packets import OrchaPacket
from ...graph.context import RunContext
from ...graph.store import MemoryStore
from ..base import ExecutionEngine, IntegrationUnavailable

__all__ = ["LangGraphEngine", "ExecutionInterrupt"]


class ExecutionInterrupt(RuntimeError):
    """
    Raised by ``execute`` when the LangGraph run stops for human input.

    Attributes
    ----------
    thread_id   The LangGraph thread (== Orcha run id) to resume.
    snapshot    The checkpoint state at the interrupt point.
    """

    def __init__(self, thread_id: str, snapshot: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(f"execution interrupted at thread {thread_id}")
        self.thread_id = thread_id
        self.snapshot = snapshot


def _config(thread_id: str, ctx: RunContext, **extra: Any) -> Dict[str, Any]:
    return {
        "configurable": {
            "thread_id": thread_id,
            "orcha_ctx": ctx,
        },
        **extra,
    }


class LangGraphEngine(ExecutionEngine):
    """
    Run an OrchaPacket through a compiled LangGraph.

    Parameters
    ----------
    graph       A compiled LangGraph (``StateGraph.compile(checkpointer=...)``).
                The graph state must expose a ``packet`` key typed as
                OrchaPacket.
    ctx_factory Optional callable that builds a RunContext when ``execute``
                is called without one (defaults to a minimal context keyed
                by the packet's id).
    """

    def __init__(self, graph: Any, ctx_factory=None) -> None:
        try:
            from langgraph.errors import GraphInterrupt  # noqa: F401
        except ImportError:
            raise IntegrationUnavailable(
                "LangGraph is not installed. Install with "
                "`pip install \"orcha[lang]\"`."
            ) from None
        self._graph = graph
        self._ctx_factory = ctx_factory or self._default_ctx

    @staticmethod
    def _default_ctx(packet: OrchaPacket) -> RunContext:
        from ...graph.context import CancelToken, EventEmitter
        return RunContext(
            run_id=packet.id,
            store=MemoryStore(),
            cancel=CancelToken(),
            emit=EventEmitter(run_id=packet.id),
            logger=logging.getLogger("orcha.integrations.langgraph"),
        )

    async def execute(
        self, packet: OrchaPacket, ctx: Optional[RunContext] = None,
        **config_extra: Any,
    ) -> OrchaPacket:
        ctx = ctx or self._ctx_factory(packet)
        config = _config(ctx.run_id, ctx, **config_extra)
        try:
            state = await self._graph.ainvoke({"packet": packet}, config)
            state = state or {}
        except Exception as exc:
            if self._unwrap_interrupt(exc) is not None:
                raise ExecutionInterrupt(ctx.run_id) from None
            raise
        if await self._is_paused(state, config):
            raise ExecutionInterrupt(ctx.run_id, snapshot=dict(state))
        out = state.get("packet")
        if not isinstance(out, OrchaPacket):
            raise TypeError(
                "LangGraph finished without a packet in state — expected "
                f"an OrchaPacket, got {type(out).__name__}"
            )
        return out

    async def execute_stream(
        self, packet: OrchaPacket, ctx: Optional[RunContext] = None,
        **config_extra: Any,
    ):
        """Yield LangGraph ``astream_events`` v2 while executing *packet*.

        Yields dicts with at least ``{"kind": ..., "event": ..., "name": ...}``.
        The final yielded item is ``{"kind": "result", "packet": <OrchaPacket>}``.
        Callers that don't need streaming can just ``await execute()`` instead.
        """
        ctx = ctx or self._ctx_factory(packet)
        config = _config(ctx.run_id, ctx, **config_extra)
        final_packet = None
        try:
            async for event in self._graph.astream_events(
                {"packet": packet}, config, version="v2",
            ):
                kind = event.get("event", "")
                # Forward LLM token events for streaming display
                if kind in ("on_chat_model_stream", "on_chat_model_end"):
                    yield {"kind": "model_event", "event": kind, "data": event.get("data", {})}
                elif kind.startswith("on_"):
                    yield {"kind": "node_event", "event": kind, "name": event.get("name", "")}
        except Exception as exc:
            if self._unwrap_interrupt(exc) is not None:
                raise ExecutionInterrupt(ctx.run_id) from None
            raise
        # Retrieve final state
        state = await self._graph.aget_state(config)
        values = getattr(state, "values", None)
        if isinstance(values, dict):
            final_packet = values.get("packet")
        if final_packet is None:
            # Fallback: try the snapshot
            snap = await self.snapshot(ctx.run_id)
            if snap:
                final_packet = snap.get("packet")
        if not isinstance(final_packet, OrchaPacket):
            raise TypeError(
                "LangGraph streaming finished without a packet in state — "
                f"expected an OrchaPacket, got {type(final_packet).__name__}"
            )
        yield {"kind": "result", "packet": final_packet}

    async def _is_paused(self, state: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """
        True when the thread still has work pending.

        langgraph v1 signals pauses two ways: ``interrupt()`` adds a
        ``__interrupt__`` key to the returned state; ``interrupt_before``
        halts silently with the next nodes still pending on the thread's
        checkpoint. Checking both keeps the engine semantics identical
        across langgraph versions.
        """
        if state.get("__interrupt__"):
            return True
        try:
            snap = await self._graph.aget_state(config)
            return bool(getattr(snap, "next", None))
        except Exception:
            return False

    async def resume(
        self, packet: OrchaPacket, thread_id: str, value: Any = None,
        ctx: Optional[RunContext] = None,
    ) -> OrchaPacket:
        """
        Continue an interrupted thread. ``value`` is delivered to the
        interrupt site (e.g. the human's approval decision).
        """
        from langgraph.types import Command

        ctx = ctx or self._ctx_factory(packet)
        state = await self._graph.ainvoke(
            Command(resume=value), _config(thread_id, ctx)
        )
        out = state.get("packet")
        if not isinstance(out, OrchaPacket):
            raise TypeError(
                "LangGraph resumed without a packet in state — expected "
                f"an OrchaPacket, got {type(out).__name__}"
            )
        return out

    async def snapshot(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Latest checkpoint state of a thread, or None when absent."""
        try:
            state = await self._graph.aget_state({"configurable": {"thread_id": thread_id}})
        except Exception:
            return None
        values = getattr(state, "values", None)
        return dict(values) if isinstance(values, dict) else None

    async def pending(self, thread_id: str) -> list:
        """Next nodes still pending on a thread.

        Returns ``[]`` for a completed thread or a thread that paused at its
        very end, and a non-empty list of node names for a thread with
        pending work (e.g. one paused at an interrupt). Used by runners to
        decide whether ``resume_from`` should continue an existing thread or
        start a fresh execution.
        """
        try:
            state = await self._graph.aget_state(
                {"configurable": {"thread_id": thread_id}}
            )
        except Exception:
            return []
        if state is None:
            return []
        return list(getattr(state, "next", None) or [])

    async def replay(self, thread_id: str) -> list:
        """Checkpoint history of a thread (each entry is a dict)."""
        from langgraph.checkpoint.base import CheckpointTuple  # noqa: F401
        history = []
        async for snap in self._graph.aget_state_history(
            {"configurable": {"thread_id": thread_id}}
        ):
            history.append({
                "checkpoint_id": getattr(snap, "config", {}).get("configurable", {}).get("checkpoint_id"),
                "next": list(getattr(snap, "next", []) or []),
                "values": dict(getattr(snap, "values", {}) or {}),
            })
        return history

    @staticmethod
    def _unwrap_interrupt(exc: Exception) -> Optional[Dict[str, Any]]:
        try:
            from langgraph.errors import GraphInterrupt
        except ImportError:  # pragma: no cover
            return None
        if isinstance(exc, GraphInterrupt):
            return {}
        # GraphInterrupt may wrap a nested exception chain.
        cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
        if cause is not None and isinstance(cause, GraphInterrupt):
            return {}
        return None