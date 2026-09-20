"""
Abstract transport interface – protocol‑layer agnostic communication.

This module defines the minimal surface that AICL uses to send and receive
packets over any medium (Unix sockets, named pipes, shared memory, TCP, QUIC,
etc.).  The protocol layer never knows or cares whether the underlying
mechanism is a Unix socket or a Windows named pipe.
"""

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional, Union


# ── Constants & Metadata ────────────────────────────────────────────────────


class TransportProtocol(IntEnum):
    """Protocol family this session uses.

    Values are ordered according to the priority list described in the
    design spec (Unix sockets highest priority, QUIC lowest).
    """
    UNIX_SOCKET = 1
    NAMED_PIPE = 2
    SHARED_MEMORY = 3
    TCP = 4
    QUIC = 5


@dataclass(slots=True, frozen=True)
class ConnectionMetadata:
    """Immutable metadata describing an open transport connection.

    Includes details like the local/remote address, protocol, and extra
    information provided by the concrete implementation.
    """
    protocol: TransportProtocol
    local_address: str
    remote_address: str
    is_accepting: bool = False
    extra: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            object.__setattr__(self, "extra", {})


class TransportRole(IntEnum):
    """Whether this transport is acting as a client (dialer) or server (listener)."""
    CLIENT = 1
    SERVER = 2


class Transport(abc.ABC):
    """
    Abstract base class for transport‑specific implementations.

    AICL protocol layer communicates solely through this interface:
        - send()/recv()          – atomic or streaming delivery
        - streaming support      – `stream_recv()` for high‑throughput bulk data
        - error handling         – throws transport‑specific exceptions
        - backpressure signaling – caller may inspect `metadata` for
          queue depth, flow‑control state, etc.

    The class is deliberately "low‑level": it works with raw bytes because
    the AICL‑BIN codec operates on byte streams, not on Python dicts.
    """

    @abc.abstractmethod
    def connect(self, address=None) -> ConnectionMetadata:
        """Open a connection to the given address.

        For clients, ``address`` describes a remote endpoint.
        For servers, ``address`` is the local bind/listen address.
        """
        pass

    @abc.abstractmethod
    async def accept(self) -> ConnectionMetadata:
        """Wait for and return an accepted connection (server mode)."""
        pass

    # ── Primitive transmission ───────────────────────────────────────────────

    @abc.abstractmethod
    def send(self, data: bytes, timeout: Optional[float] = None) -> None:
        """Blocking send of ``data``.

        ``timeout`` specifies the maximum time to wait for the data to be
        accepted by the kernel. If ``timeout`` is exceeded, a
        ``ChannelTimeoutError`` is raised.
        """
        pass

    @abc.abstractmethod
    def recv(self, max_bytes: int = 8192, timeout: Optional[float] = None) -> bytes:
        """Blocking receive of up to ``max_bytes`` bytes.

        ``timeout`` applies to the whole receive operation; ``recv`` may
        return fewer bytes than ``max_bytes`` if the remote side closes.
        """
        pass

    # ── Asynchronous helpers ─────────────────────────────────────────────────

    async def send_async(self, data: bytes, timeout: Optional[float] = None) -> None:
        """Non‑blocking version of ``send`` using asyncio."""
        raise NotImplementedError("Transport subclasses should override this")

    async def recv_async(self, max_bytes: int = 8192, timeout: Optional[float] = None) -> bytes:
        """Non‑blocking version of ``recv`` using asyncio."""
        raise NotImplementedError("Transport subclasses should override this")

    # ── Streaming support ────────────────────────────────────────────────────

    @abc.abstractmethod
    async def stream_recv(self, max_bytes: int = 65536) -> asyncio.AsyncGenerator[bytes, None]:
        """Async generator that yields chunks of incoming data indefinitely.

        ``max_bytes`` is the preferred chunk size; the implementation may
        choose larger or smaller chunks based on its internal buffering.
        """
        pass

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def close(self) -> None:
        """Terminate the connection and release all resources."""
        pass

    @abc.abstractmethod
    async def close_async(self) -> None:
        """Asynchronous variant of ``close`` for async APIs."""
        pass

    # ── Backpressure / metadata ──────────────────────────────────────────────

    @abc.abstractmethod
    def get_metadata(self) -> ConnectionMetadata:
        """Return metadata describing the current connection."""
        pass

    @abc.abstractmethod
    def can_send(self, bytes_hint: int = 0) -> bool:
        """True if the channel can accept a transmission of ``bytes_hint`` bytes.

        Implementations may inspect queue depth, OS send buffers, or other
        metrics to implement backpressure.
        """
        pass

    # ── Timeouts / cancellation ───────────────────────────────────────────────

    @abc.abstractmethod
    def set_timeout(self, timeout_seconds: Optional[float]) -> None:
        """Set a global timeout for subsequent read/write operations."""
        pass

    @abc.abstractmethod
    def cancel(self) -> None:
        """Cancel any pending operations (e.g., ``recv`` on a blocking call)."""
        pass
