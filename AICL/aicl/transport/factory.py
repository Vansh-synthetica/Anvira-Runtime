"""
Transport factory — create transport backends by scheme.

Provides `create_transport` for creating specific transports
and `get_platform_transport` for the default platform transport.
"""

from __future__ import annotations

from typing import Optional

from aicl.transport.address import TransportAddress
from aicl.transport.interface import Transport, TransportRole
from aicl.transport.exceptions import TransportError


def create_transport(
    address: TransportAddress,
    role: TransportRole = "client",
) -> Transport:
    """Create a transport backend for the given address.

    Supported schemes:
      - unix: Unix domain socket
      - pipe: Windows named pipe
      - shm: Shared memory ring buffer
      - tcp: TCP socket

    Raises:
        TransportError: If the scheme is not supported.
    """
    scheme = address.scheme.lower()

    if scheme == "unix":
        return _create_unix_transport(address)
    elif scheme == "pipe":
        return _create_pipe_transport(address)
    elif scheme == "shm":
        return _create_shm_transport(address)
    elif scheme == "tcp":
        return _create_tcp_transport(address)
    else:
        raise TransportError(f"Unsupported transport scheme: {scheme}")


def get_platform_transport(
    role: TransportRole = "client",
) -> Transport:
    """Get the default transport for the current platform.

    - Windows: Named pipe (\\\\.\\pipe\\aicl)
    - Unix: Unix domain socket (/tmp/aicl.sock)
    """
    import sys
    if sys.platform == "win32":
        addr = TransportAddress("pipe", r"\\.\pipe\aicl")
    else:
        addr = TransportAddress("unix", "/tmp/aicl.sock")
    return create_transport(addr, role)


def _create_unix_transport(address: TransportAddress) -> Transport:
    """Create a Unix domain socket transport."""
    try:
        from aicl.transport.unix_socket import UnixSocketTransport
        return UnixSocketTransport(address.path)
    except ImportError:
        raise TransportError(
            "Unix socket transport not yet implemented. "
            "Use shared_memory or tcp transport."
        )


def _create_pipe_transport(address: TransportAddress) -> Transport:
    """Create a Windows named pipe transport."""
    try:
        from aicl.transport.named_pipe import NamedPipeTransport
        return NamedPipeTransport(address.path)
    except ImportError:
        raise TransportError(
            "Named pipe transport not yet implemented. "
            "Use shared_memory or tcp transport."
        )


def _create_shm_transport(address: TransportAddress) -> Transport:
    """Create a shared memory ring buffer transport."""
    try:
        from aicl.transport.shared_memory_ring import SharedMemoryRingTransport
        return SharedMemoryRingTransport(address.path)
    except ImportError:
        raise TransportError(
            "Shared memory transport not yet implemented."
        )


def _create_tcp_transport(address: TransportAddress) -> Transport:
    """Create a TCP socket transport."""
    try:
        from aicl.transport.tcp_socket import TcpSocketTransport
        return TcpSocketTransport(address.host, address.port)
    except ImportError:
        raise TransportError(
            "TCP transport not yet implemented."
        )
