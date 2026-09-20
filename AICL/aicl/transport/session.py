"""
Transport session — codec + transport bound together.

Provides a high-level session that handles encoding/decoding
and delegates transport to the underlying backend.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from aicl.transport.address import TransportAddress
from aicl.transport.interface import Transport, TransportRole, ConnectionMetadata
from aicl.transport.exceptions import TransportError, ChannelClosedError


class TransportSession:
    """A session wrapping a transport with codec operations.

    Handles encode/decode and lifecycle management.
    """

    def __init__(
        self,
        address: TransportAddress,
        role: TransportRole = "client",
        transport: Optional[Transport] = None,
    ):
        self._address = address
        self._role = role
        self._transport = transport
        self._open = False

    @property
    def address(self) -> TransportAddress:
        return self._address

    @property
    def role(self) -> TransportRole:
        return self._role

    @property
    def is_open(self) -> bool:
        return self._open

    async def connect(self) -> None:
        """Establish the connection."""
        if self._transport is None:
            raise TransportError("No transport backend configured")
        self._open = True

    async def accept(self) -> None:
        """Accept an incoming connection (server role)."""
        if self._role != "server":
            raise TransportError("accept() only available for server role")
        self._open = True

    async def send(self, data: bytes) -> int:
        """Send raw bytes through the transport."""
        if not self._open:
            raise ChannelClosedError("Session is closed")
        if self._transport is None:
            raise TransportError("No transport backend")
        return await self._transport.send(data)

    async def recv(self) -> bytes:
        """Receive raw bytes from the transport."""
        if not self._open:
            raise ChannelClosedError("Session is closed")
        if self._transport is None:
            raise TransportError("No transport backend")
        return await self._transport.recv()

    async def close(self) -> None:
        """Close the session."""
        self._open = False

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()
