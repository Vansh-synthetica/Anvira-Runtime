"""
orcha.agent_runtime.bus
=======================
Transport adapters between the AgentRuntime and Orcha's existing
OrchaPacket message bus.

Design contract
---------------
- The EventLog is the authoritative store of Events (see events.py).
- Actions and Observations cross component boundaries — Agent → Workspace,
  Workspace → Conversation — wrapped in OrchaPackets. No parallel bus is
  invented: an Action leaves the Conversation as an ``ACTION``-kind packet
  (payload ``{"action": {...}, "event_seq": n}``) and comes back from the
  Workspace as an ``OBSERVATION``-kind packet (payload
  ``{"observation": {...}}``). Tokens ride along in packet metadata so a
  full Event can round-trip through the bus losslessly.

Wire format
-----------
- ACTION packet payload:
    {"action": <Action.model_dump()>, "event_seq": <int>}
- OBSERVATION packet payload:
    {"observation": <Observation.model_dump()>, "event_seq": <int>}
- Event tokens are carried in ``packet.metadata["tokens"]``.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import TypeAdapter

from ..core.packets import OrchaPacket, PacketKind
from .errors import AgentRuntimeError
from .events import (
    Action, ActionPayload, Event, EventKind, Observation, ObservationPayload,
    kind_of,
)

ACTION_KEY = "action"
OBSERVATION_KEY = "observation"
EVENT_SEQ_KEY = "event_seq"
TOKENS_KEY = "tokens"

_ACTION_ADAPTER = TypeAdapter(ActionPayload)
_OBSERVATION_ADAPTER = TypeAdapter(ObservationPayload)


# ── Actions ────────────────────────────────────────────────────────────────

def action_packet(
    action: Action,
    *,
    query: str = "",
    seq: int = 0,
    tokens: int = 0,
    parent: Optional[OrchaPacket] = None,
    budget: Any = None,
) -> OrchaPacket:
    """Wrap an Action in an ACTION-kind OrchaPacket (the outbound leg of
    the bus, Conversation → Workspace)."""
    if parent is not None:
        return parent.fork(
            PacketKind.ACTION,
            **{ACTION_KEY: action.model_dump(), EVENT_SEQ_KEY: seq},
        )
    kwargs: dict = {
        "kind": PacketKind.ACTION,
        "query": query,
        "payload": {ACTION_KEY: action.model_dump(), EVENT_SEQ_KEY: seq},
        "metadata": {TOKENS_KEY: max(0, int(tokens))},
    }
    if budget is not None:
        kwargs["budget"] = budget
    return OrchaPacket(**kwargs)


def action_from_packet(packet: OrchaPacket) -> Action:
    """Unwrap an Action from an ACTION-kind OrchaPacket."""
    raw = packet.payload.get(ACTION_KEY)
    if raw is None:
        raise AgentRuntimeError(
            f"Packet {packet.id[:8]} kind={packet.kind} carries no action payload"
        )
    return _ACTION_ADAPTER.validate_python(raw)


def packet_to_action(packet: OrchaPacket) -> Action:
    """Alias of ``action_from_packet`` (bus reads Actions in)."""
    return action_from_packet(packet)


# ── Observations ───────────────────────────────────────────────────────────

def observation_packet(
    observation: Observation,
    *,
    query: str = "",
    seq: int = 0,
    tokens: int = 0,
    parent: Optional[OrchaPacket] = None,
    budget: Any = None,
) -> OrchaPacket:
    """Wrap an Observation in an OBSERVATION-kind OrchaPacket (the return
    leg of the bus, Workspace → Conversation)."""
    if parent is not None:
        return parent.fork(
            PacketKind.OBSERVATION,
            **{OBSERVATION_KEY: observation.model_dump(), EVENT_SEQ_KEY: seq},
        )
    kwargs: dict = {
        "kind": PacketKind.OBSERVATION,
        "query": query,
        "payload": {OBSERVATION_KEY: observation.model_dump(), EVENT_SEQ_KEY: seq},
        "metadata": {TOKENS_KEY: max(0, int(tokens))},
    }
    if budget is not None:
        kwargs["budget"] = budget
    return OrchaPacket(**kwargs)


def observation_from_packet(packet: OrchaPacket) -> Observation:
    """Unwrap an Observation from an OBSERVATION-kind OrchaPacket."""
    raw = packet.payload.get(OBSERVATION_KEY)
    if raw is None:
        raise AgentRuntimeError(
            f"Packet {packet.id[:8]} kind={packet.kind} carries no observation payload"
        )
    return _OBSERVATION_ADAPTER.validate_python(raw)


def packet_to_observation(packet: OrchaPacket) -> Observation:
    """Alias of ``observation_from_packet`` (bus reads Observations in)."""
    return observation_from_packet(packet)


# ── Events (full lossless round-trip) ──────────────────────────────────────

def event_to_packet(
    event: Event,
    *,
    query: str = "",
    parent: Optional[OrchaPacket] = None,
    budget: Any = None,
) -> OrchaPacket:
    """Wrap an Event in the bus packet its payload's type requires. The
    event seq is carried in ``event_seq`` and the token cost in metadata,
    so ``event_from_packet`` reconstructs the Event exactly."""
    seq = event.seq
    tokens = event.tokens
    if event.kind == EventKind.ACTION:
        return action_packet(
            event.payload, query=query, seq=seq, tokens=tokens,
            parent=parent, budget=budget,
        )
    return observation_packet(
        event.payload, query=query, seq=seq, tokens=tokens,
        parent=parent, budget=budget,
    )


def event_from_packet(packet: OrchaPacket) -> Event:
    """Reconstruct an Event from a bus packet (ACTION or OBSERVATION kind)."""
    payload: Any
    if packet.kind == PacketKind.ACTION or ACTION_KEY in packet.payload:
        payload = action_from_packet(packet)
    elif packet.kind == PacketKind.OBSERVATION or OBSERVATION_KEY in packet.payload:
        payload = observation_from_packet(packet)
    else:
        raise AgentRuntimeError(
            f"Packet {packet.id[:8]} kind={packet.kind} is not an "
            f"ACTION/OBSERVATION bus packet"
        )
    return Event(
        seq=int(packet.payload.get(EVENT_SEQ_KEY, 0) or 0),
        kind=kind_of(payload),
        payload=payload,
        tokens=int(packet.metadata.get(TOKENS_KEY, 0) or 0),
    )


__all__ = [
    "ACTION_KEY", "OBSERVATION_KEY", "EVENT_SEQ_KEY", "TOKENS_KEY",
    "action_packet", "action_from_packet", "packet_to_action",
    "observation_packet", "observation_from_packet", "packet_to_observation",
    "event_to_packet", "event_from_packet",
]
