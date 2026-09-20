"""
aicl.transport – Transport‑neutral communication layer
=====================================================

Provides abstract and concrete transport backends so that AICL packets
can be moved over any medium without the protocol layer needing to know
which one is in use.

Architecture::

    AICL Packet
       ↓
    AICL Binary Codec
       ↓
    Transport  (abstract interface)
       ↓
    [Unix Socket | Named Pipe | Shared Memory | TCP | QUIC]
       ↓
    Transport  (abstract interface)
       ↓
    AICL Binary Codec
       ↓
    AICL Packet

Usage::

    from aicl.transport import TransportSession, TransportAddress

    # Create a session over a Unix domain socket
    addr = TransportAddress("unix", "/tmp/aicl.sock")
    async with TransportSession(addr, role="client") as sess:
        await sess.send(packet_dict)
        resp = await sess.recv()

    # Or over a Windows named pipe
    addr = TransportAddress("pipe", r"\\\\.\\pipe\\aicl")
    async with TransportSession(addr, role="server") as sess:
        await sess.accept()
        ...
"""

from __future__ import annotations

# ── Address ──────────────────────────────────────────────────────────────────
from aicl.transport.address import TransportAddress

# ── Exceptions ───────────────────────────────────────────────────────────────
from aicl.transport.exceptions import (
    TransportError,
    ConnectionError,
    ChannelClosedError,
    ChannelResetError,
    ChannelTimeoutError,
    BackpressureError,
    ProtocolError,
)

# ── Abstract transport interface ──────────────────────────────────────────────
from aicl.transport.interface import (
    Transport,
    TransportRole,
    ConnectionMetadata,
)

# ── Session (codec + transport) ──────────────────────────────────────────────
from aicl.transport.session import TransportSession

# ── Factory ──────────────────────────────────────────────────────────────────
from aicl.transport.factory import create_transport, get_platform_transport

__all__ = [
    # Address
    "TransportAddress",
    # Exceptions
    "TransportError",
    "ConnectionError",
    "ChannelClosedError",
    "ChannelResetError",
    "ChannelTimeoutError",
    "BackpressureError",
    "ProtocolError",
    # Interface
    "Transport",
    "TransportRole",
    "ConnectionMetadata",
    # Session
    "TransportSession",
    # Factory
    "create_transport",
    "get_platform_transport",
]
