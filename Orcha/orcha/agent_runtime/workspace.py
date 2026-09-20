"""
orcha.agent_runtime.workspace
=============================
The Workspace: executes Actions and returns Observations.

Transport is the OrchaPacket bus — ``execute()`` receives an ACTION-kind
packet (see bus.py) and returns an OBSERVATION-kind packet, so no parallel
bus exists. The Conversation unwraps the returned Observation and appends
it to the EventLog.

- ``StubWorkspace`` — echo/no-op execution for the skeleton tests.
- ``ToolWorkspace`` — real execution: ToolCallActions are validated
  against the ToolRegistry and executed by the underlying Orcha tool
  machinery (filesystem/terminal capabilities), returning real
  ToolResultObservations. Unknown tools and schema mismatches become
  ErrorObservations — the agent loop never crashes.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from ..core.packets import OrchaPacket
from .bus import (
    action_from_packet, observation_packet,
)
from .diagnostics import diag_log
from .events import (
    ErrorAction, ErrorObservation, MessageAction, Observation,
    ToolCallAction, ToolResultObservation,
)
from .tools import ToolRegistry

logger = logging.getLogger("orcha.agent_runtime.workspace")


class Workspace(ABC):
    """
    Executes Actions, returns Observations.

    Contract: ``execute(packet)`` receives an ACTION-kind OrchaPacket and
    must return an OBSERVATION-kind OrchaPacket. Workspaces are expected to
    be side-effectful (that is their job) but deterministic in what they
    report back.
    """

    @abstractmethod
    def execute(self, packet: OrchaPacket) -> OrchaPacket:
        ...


class StubWorkspace(Workspace):
    """
    Echo/no-op tool execution for the skeleton: every ToolCallAction
    "succeeds" and reports its own name and arguments back verbatim.
    """

    def execute(self, packet: OrchaPacket) -> OrchaPacket:
        action = action_from_packet(packet)
        observation: Optional[Observation] = None

        if isinstance(action, ToolCallAction):
            observation = ToolResultObservation(
                tool_call_id=action.tool_call_id,
                content=(
                    f"[stub:{action.name}] executed with "
                    f"arguments={action.arguments}"
                ),
                success=True,
            )
        elif isinstance(action, ErrorAction):
            observation = ErrorObservation(message=action.message)
        elif isinstance(action, MessageAction):
            # MessageActions are user-facing chatter, not tool work; the
            # Conversation never routes them here.
            observation = ErrorObservation(
                message="workspace does not execute message actions"
            )

        if observation is None:
            observation = ErrorObservation(
                message=f"unsupported action type {type(action).__name__}"
            )

        seq = int(packet.payload.get("event_seq", 0) or 0)
        return observation_packet(observation, seq=seq, parent=packet)


class ToolWorkspace(Workspace):
    """
    Real tool execution against a ToolRegistry.

    Every ToolCallAction is routed through the registry's executor: schema
    validation happens first (a mismatch becomes an ErrorObservation —
    never a crash), then the underlying Orcha tool machinery runs it. The
    workspace is bound to the registry at construction; no mutable state is
    consulted per call, so results are deterministic for identical inputs.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    def execute(self, packet: OrchaPacket) -> OrchaPacket:
        action = action_from_packet(packet)
        observation: Optional[Observation] = None

        if isinstance(action, ToolCallAction):
            tool = self._registry.get(action.name)
            if tool is None:
                diag_log(
                    logger, "workspace", "unknown_tool",
                    level=logging.WARNING,
                    name=action.name,
                    available=",".join(self._registry.names()) or "(none)",
                )
                observation = ErrorObservation(
                    message=(
                        f"unknown tool '{action.name}' — available tools: "
                        f"{', '.join(self._registry.names()) or '(none)'}"
                    )
                )
            else:
                validation_error = tool.validate(action.arguments)
                if validation_error:
                    diag_log(
                        logger, "workspace", "validation_failed",
                        level=logging.WARNING,
                        name=action.name, error=validation_error,
                    )
                    observation = ErrorObservation(
                        message=(
                            f"tool '{action.name}' rejected arguments: "
                            f"{validation_error}"
                        )
                    )
                else:
                    observation = tool.run(
                        action.arguments, tool_call_id=action.tool_call_id,
                    )
        elif isinstance(action, ErrorAction):
            observation = ErrorObservation(message=action.message)
        elif isinstance(action, MessageAction):
            observation = ErrorObservation(
                message="workspace does not execute message actions"
            )

        if observation is None:
            observation = ErrorObservation(
                message=f"unsupported action type {type(action).__name__}"
            )

        seq = int(packet.payload.get("event_seq", 0) or 0)
        return observation_packet(observation, seq=seq, parent=packet)


__all__ = ["Workspace", "StubWorkspace", "ToolWorkspace"]
