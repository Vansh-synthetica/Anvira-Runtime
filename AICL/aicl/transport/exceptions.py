"""
Transport‑layer exception hierarchy.

All transport exceptions inherit from the public ``TransportError`` base
class so that callers can catch any transport failure with a single
``except`` clause.
"""

from __future__ import annotations


class TransportError(Exception):
    """
    Root exception for the transport layer.

    All concrete transport errors (``ConnectionError``, ``ChannelClosedError``,
    etc.) inherit from this class.
    """
    pass


# ── Connection errors ────────────────────────────────────────────────────────

class ConnectionError(TransportError):
    """Failed to establish a transport connection."""
    pass


class ConnectionRefusedError(ConnectionError):
    """The peer actively refused the connection."""
    pass


class ConnectionTimeoutError(ConnectionError):
    """Connection establishment timed out."""
    pass


# ── Channel errors ───────────────────────────────────────────────────────────

class ChannelError(TransportError):
    """General channel (open connection) error."""
    pass


class ChannelClosedError(ChannelError):
    """The channel has been closed (by local or remote side)."""
    pass


class ChannelResetError(ChannelError):
    """The channel was abruptly reset by the peer."""
    pass


class ChannelTimeoutError(ChannelError):
    """A read or write operation timed out."""
    pass


class BackpressureError(ChannelError):
    """
    The receiver cannot keep up with the sender.

    This signals that the caller should pause sending and retry later.
    """
    pass


# ── Protocol errors ──────────────────────────────────────────────────────────

class ProtocolError(TransportError):
    """A transport‑ or codec‑level protocol violation was detected."""
    pass


class FramingError(ProtocolError):
    """Failed to decode a transport frame."""
    pass


class ChecksumError(ProtocolError):
    """Integrity check (CRC / checksum) failed."""
    pass
