"""
Transport address parsing and scheme resolution.

Each address is represented as a TransportAddress object that
decomposes a URI‑like string into scheme and path components.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# Valid transport schemes
VALID_SCHEMES: frozenset[str] = frozenset({
    "unix",     # Unix domain socket
    "pipe",     # Windows named pipe
    "shm",      # Shared memory region
    "tcp",      # TCP socket
    "quic",     # QUIC (future)
})


@dataclass(frozen=True, slots=True)
class TransportAddress:
    """
    Immutable, transport‑neutral endpoint descriptor.

    Parses addresses of the form::

        unix:///tmp/aicl.sock
        pipe://my_pipe
        shm://aicl_buffer
        tcp://127.0.0.1:7890
        quic://localhost:7891

    The leading ``//`` is optional for Unix / pipe / shm schemes.
    """

    scheme: str
    path:   str
    host:   Optional[str] = None
    port:   Optional[int] = None

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __post_init__(self):
        if self.scheme not in VALID_SCHEMES:
            raise ValueError(
                f"Unknown transport scheme {self.scheme!r}. "
                f"Valid schemes: {sorted(VALID_SCHEMES)}"
            )

    @classmethod
    def parse(cls, uri: str) -> "TransportAddress":
        """
        Parse a URI string into a TransportAddress.

        Examples::

            TransportAddress.parse("unix:///tmp/aicl.sock")
            TransportAddress.parse("tcp://127.0.0.1:8080")
            TransportAddress.parse("pipe://my_pipe")

        Raises ``ValueError`` if the URI is malformed.
        """
        if not uri:
            raise ValueError("Empty transport address")

        # Extract scheme (before ://)
        m = re.match(r"^([a-zA-Z][a-zA-Z0-9+-]*?)://(.*)$", uri)
        if not m:
            # No scheme – treat the whole string as a path for unix/pipe/shm
            scheme = "unix"
            path = uri
            host, port = None, None
        else:
            scheme = m.group(1).lower()
            rest = m.group(2)

            if scheme in ("tcp", "quic"):
                # Host:port format
                host_m = re.match(r"\[([^\]]+)\]:(\d+)$", rest)  # IPv6
                if host_m:
                    host, port_str = host_m.group(1), host_m.group(2)
                else:
                    host_m = re.match(r"([^:]+):(\d+)$", rest)    # IPv4 or hostname
                    if host_m:
                        host, port_str = host_m.group(1), host_m.group(2)
                    else:
                        raise ValueError(
                            f"Invalid {scheme} address {uri!r}: "
                            "expected host:port"
                        )
                try:
                    port = int(port_str)
                except ValueError:
                    raise ValueError(f"Invalid port in {uri!r}") from None
                path = ""

            elif scheme in ("unix", "pipe", "shm"):
                path = rest
                host, port = None, None
            else:
                raise ValueError(f"Unknown scheme {scheme!r}")

        return cls(scheme=scheme, path=path, host=host, port=port)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def is_local(self) -> bool:
        """True for Unix sockets, named pipes, and shared memory."""
        return self.scheme in ("unix", "pipe", "shm")

    @property
    def is_stream(self) -> bool:
        """True for byte‑stream transports (sockets, pipes)."""
        return self.scheme in ("unix", "pipe", "tcp", "quic")

    @property
    def is_datagram(self) -> bool:
        """True for message‑oriented transports (shared memory)."""
        return self.scheme == "shm"

    # ------------------------------------------------------------------ #
    # Representation
    # ------------------------------------------------------------------ #

    def __str__(self) -> str:
        if self.host is not None:
            return f"{self.scheme}://{self.host}:{self.port}"
        if self.path:
            return f"{self.scheme}://{self.path}"
        return f"{self.scheme}:"

    def __repr__(self) -> str:
        return f"TransportAddress({self.scheme!r}, {self.path!r}, host={self.host!r}, port={self.port!r})"
