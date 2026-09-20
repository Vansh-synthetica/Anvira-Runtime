"""
AICL Runtime — main entry point for the Python SDK.

The runtime manages connections to native modules, capability discovery,
and transport configuration. It is a developer-friendly facade over the
native Rust core.

Usage::

    from aicl.sdk import AICLRuntime

    # Default: native Rust transport, no HTTP
    runtime = AICLRuntime()

    # Connect to a model with a specific capability
    model = runtime.connect(capability="reasoning")

    # Sync call
    result = model.call("classify", text="hello world")
    print(result.unwrap())

    # Async call
    result = await model.acall("classify", text="hello world")

    # Streaming
    async for chunk in model.stream("generate", prompt="hello"):
        print(chunk.data)

    # Diagnostic
    print(runtime.diagnose().report())

    # HTTP compat (explicit opt-in)
    runtime_http = AICLRuntime(http=True, http_url="http://localhost:8080")
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Optional, Union

from .capabilities import Capability, CapabilitySet, BUILTIN_CAPABILITIES
from .diagnostic import TransportInfo, diagnose
from .errors import (
    AICLError,
    CapabilityError,
    HTTPCompatDisabledError,
    NativeBridgeError,
)
from .model import Model
from .result import StructuredResult
from .transport import NativeTransport, HttpTransport
from ._cancellation import CancellationToken
from ._ffi import get_version, get_isa_version, NativeError


class AICLRuntime:
    """Main entry point for the AICL Python SDK.

    The runtime is a facade over the native Rust core. It manages:
      - Transport selection (native vs HTTP)
      - Capability discovery
      - Model connections
      - Diagnostics

    Default behavior:
      - Transport: native (Rust FFI)
      - Codec: AICL-ISA (binary)
      - Zero-copy: active
      - HTTP: disabled

    To enable HTTP compatibility::

        runtime = AICLRuntime(http=True, http_url="http://localhost:8080")
    """

    __slots__ = (
        "_transport", "_http_enabled", "_http_url", "_timeout",
        "_native_available", "_capabilities", "_models", "_closed", "_lock",
    )

    def __init__(
        self,
        *,
        http: bool = False,
        http_url: str = "http://localhost:8080",
        timeout: float = 30.0,
        capabilities: Optional[CapabilitySet] = None,
    ):
        self._http_enabled = http
        self._http_url = http_url
        self._timeout = timeout
        self._models: dict[str, Model] = {}
        self._closed = False
        self._lock = threading.Lock()
        self._transport = None  # type: ignore

        # Detect native library
        self._native_available = False
        try:
            from ._ffi import get_lib
            get_lib()
            self._native_available = True
        except (OSError, NativeError):
            self._native_available = False

        # Select transport
        if self._native_available:
            self._transport = NativeTransport()
        elif http:
            self._transport = HttpTransport(url=http_url, timeout=timeout)
        else:
            raise AICLError(
                "No native AICL core library found and HTTP is not enabled. "
                "Either build the Rust core (cargo build --release in core-rust/) "
                "or enable HTTP: AICLRuntime(http=True)"
            )

        # Load capabilities
        self._capabilities = capabilities or BUILTIN_CAPABILITIES

    # ── Connection ───────────────────────────────────────────────────────────

    def connect(
        self,
        capability: str,
        *,
        name: str = "",
        timeout: Optional[float] = None,
    ) -> Model:
        """Connect to a model with a specific capability.

        Args:
            capability: Capability name (e.g., 'classify', 'generate').
            name: Optional name for the model connection.
            timeout: Override the default timeout.

        Returns:
            A Model handle for sending instructions.

        Raises:
            CapabilityError: If the capability is not available.
        """
        if self._closed:
            raise AICLError("Runtime is closed")

        # Find the capability
        caps = self._capabilities.find(capability)
        if not caps:
            raise CapabilityError(
                f"Capability '{capability}' not found. "
                f"Available: {self._capabilities.names()}"
            )

        cap = caps[0]
        model = Model(
            transport=self._transport,
            capability=cap,
            name=name,
            timeout=timeout or self._timeout,
        )

        with self._lock:
            self._models[cap.name] = model

        return model

    def connect_any(
        self,
        *capabilities: str,
        name: str = "",
        timeout: Optional[float] = None,
    ) -> Model:
        """Connect to the first available capability from the list."""
        for cap_name in capabilities:
            try:
                return self.connect(capability=cap_name, name=name, timeout=timeout)
            except CapabilityError:
                continue
        raise CapabilityError(
            f"None of the requested capabilities available: {list(capabilities)}. "
            f"Available: {self._capabilities.names()}"
        )

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def diagnose(self) -> TransportInfo:
        """Get transport diagnostic information.

        Returns a TransportInfo with:
          - transport: 'native' or 'http'
          - runtime language: 'rust' or 'http'
          - codec: 'aicl-isa', 'aicl-bin', or 'json'
          - zero-copy: active/inactive
          - http enabled: yes/no
          - native available: yes/no
          - library version: version string
          - isa version: version number
        """
        try:
            lib_version = get_version() if self._native_available else "N/A"
            isa_ver = get_isa_version() if self._native_available else 0
        except Exception:
            lib_version = "error"
            isa_ver = 0

        return diagnose(
            native_available=self._native_available,
            http_enabled=self._http_enabled,
            library_version=lib_version,
            isa_version=isa_ver,
        )

    def native_transport(self) -> TransportInfo:
        """Alias for diagnose(). Returns transport diagnostic info."""
        return self.diagnose()

    # ── Capability discovery ─────────────────────────────────────────────────

    def capabilities(self) -> CapabilitySet:
        """Get the set of available capabilities."""
        return self._capabilities

    def has_capability(self, name: str) -> bool:
        """Check if a capability is available."""
        return self._capabilities.has(name)

    def list_capabilities(self) -> list[str]:
        """List all available capability names."""
        return self._capabilities.names()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the runtime and all connected models."""
        with self._lock:
            for model in self._models.values():
                model.close()
            self._models.clear()
            self._closed = True
            if self._transport:
                self._transport.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if getattr(self, "_closed", True):
            return
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        mode = "native" if self._native_available else ("http" if self._http_enabled else "none")
        return f"<AICLRuntime mode={mode} capabilities={len(self._capabilities)}>"
