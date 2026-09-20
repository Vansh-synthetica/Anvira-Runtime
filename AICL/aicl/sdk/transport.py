"""
Transport layer for the AICL SDK.

Provides:
  - NativeTransport: FFI bridge to the Rust core (default, always)
  - HttpTransport: HTTP compatibility layer (explicit opt-in only)

The default is always native. HTTP must be explicitly enabled.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, AsyncIterator, Callable, Optional, Union

from .errors import (
    HTTPCompatDisabledError,
    HTTPCompatError,
    NativeBridgeError,
    TransportError,
)
from ._ffi import (
    AiclContext,
    NativeBuffer,
    NativeError,
    get_version,
    get_isa_version,
)
from ._cancellation import CancellationToken, NULL_TOKEN


class NativeTransport:
    """Transport backed by the native Rust AICL core via FFI.

    This is the default transport. It uses:
      - Lock-free ring buffers for in-process communication
      - Shared memory for cross-process communication
      - Zero-copy codec for encode/decode
      - Direct dispatch (O(1) opcode lookup)

    Ownership:
      - The transport owns the AiclContext
      - Encoded data is copied into native buffers (one copy)
      - Decoded data borrows the wire buffer (zero-copy)
    """

    __slots__ = ("_ctx", "_lock", "_closed")

    def __init__(self):
        self._ctx = AiclContext()
        self._lock = threading.Lock()
        self._closed = False

    @property
    def is_native(self) -> bool:
        return True

    def encode(self, data: bytes) -> NativeBuffer:
        """Encode wire data into a native buffer (one copy into Rust allocation)."""
        if self._closed:
            raise TransportError("Transport is closed")
        return NativeBuffer(data)

    def decode(self, data: bytes) -> AiclPacketView:
        """Decode wire data into a packet view (zero-copy borrow of data)."""
        if self._closed:
            raise TransportError("Transport is closed")
        view_ptr = self._ctx.decode(data)
        return AiclPacketView(view_ptr, self._ctx)

    def send(self, data: bytes) -> None:
        """Send raw wire data through the transport.

        In the native path, this encodes into a native buffer.
        Actual transport (ring buffer, SHM) is handled by the Rust runtime.
        """
        if self._closed:
            raise TransportError("Transport is closed")
        # For now, this validates the wire data by decoding it
        # The actual send is handled by the Rust runtime's ring buffer
        self._ctx.decode(data)

    def recv(self) -> bytes:
        """Receive raw wire data from the transport.

        In the native path, this reads from the Rust runtime's ring buffer.
        """
        if self._closed:
            raise TransportError("Transport is closed")
        raise TransportError(
            "recv() not yet implemented for native transport; "
            "use the Rust runtime's async receive path"
        )

    def close(self) -> None:
        """Close the transport and free native resources."""
        self._closed = True
        self._ctx = None  # type: ignore

    def __del__(self):
        if not self._closed:
            self.close()


class AiclPacketView:
    """Zero-copy view of a decoded AICL packet.

    Borrows data from the native transport. Valid as long as the
    transport (or the underlying buffer) is alive.
    """

    __slots__ = ("_view_ptr", "_ctx", "_opcode", "_flags", "_operand_count")

    def __init__(self, view_ptr: int, ctx: AiclContext):
        self._view_ptr = view_ptr
        self._ctx = ctx
        self._opcode = AiclContext.view_opcode(view_ptr)
        self._flags = AiclContext.view_flags(view_ptr)
        self._operand_count = AiclContext.view_operand_count(view_ptr)

    @property
    def opcode(self) -> int:
        return self._opcode

    @property
    def flags(self) -> int:
        return self._flags

    @property
    def operand_count(self) -> int:
        return self._operand_count

    def __del__(self):
        if hasattr(self, "_view_ptr") and self._view_ptr and self._ctx:
            self._ctx.free_view(self._view_ptr)


class HttpTransport:
    """HTTP compatibility layer — explicit opt-in only.

    This transport wraps HTTP requests to an AICL HTTP gateway.
    It is NOT enabled by default and must be explicitly requested:

        runtime = AICLRuntime(http=True, http_url="http://localhost:8080")

    When enabled, all operations go through HTTP. The native FFI
    is still available for local operations.
    """

    __slots__ = ("_url", "_timeout", "_session", "_closed")

    def __init__(self, url: str = "http://localhost:8080", timeout: float = 30.0):
        self._url = url.rstrip("/")
        self._timeout = timeout
        self._session = None
        self._closed = False

    @property
    def is_native(self) -> bool:
        return False

    def encode(self, data: bytes) -> bytes:
        """Encode is a no-op for HTTP (data is sent as-is)."""
        if self._closed:
            raise TransportError("Transport is closed")
        return data

    def decode(self, data: bytes) -> dict[str, Any]:
        """Decode HTTP JSON response into a dict."""
        if self._closed:
            raise TransportError("Transport is closed")
        import json
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            raise HTTPCompatError(f"Invalid JSON response: {e}")

    def send(self, data: bytes) -> bytes:
        """Send data via HTTP POST and return the response."""
        if self._closed:
            raise TransportError("Transport is closed")

        import urllib.request
        import json

        req = urllib.request.Request(
            f"{self._url}/v1/aicl",
            data=data,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return resp.read()
        except urllib.error.URLError as e:
            raise HTTPCompatError(f"HTTP request failed: {e}")
        except TimeoutError:
            from .errors import TimeoutError as AICLTimeout
            raise AICLTimeout(f"HTTP request timed out after {self._timeout}s")

    def recv(self) -> bytes:
        raise TransportError("recv() not supported for HTTP transport; use send() for request/response")

    def close(self) -> None:
        self._closed = True

    def __del__(self):
        if not self._closed:
            self.close()
