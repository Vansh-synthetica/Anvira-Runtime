"""
AICL Python SDK — developer-friendly facade over the native Rust core.

This package provides a clean Python API for AICL operations while
delegating all performance-critical work to the native Rust core.

# Default: Native Transport

    from aicl.sdk import AICLRuntime

    runtime = AICLRuntime()
    model = runtime.connect(capability="reasoning")
    result = model.call("classify", text="hello world")

# HTTP Compatibility (explicit opt-in)

    runtime = AICLRuntime(http=True, http_url="http://localhost:8080")
    model = runtime.connect(capability="classify")
    result = model.call("classify", text="hello world")

# Diagnostics

    info = runtime.diagnose()
    print(info.report())

# What Python does NOT implement:
#   - Native routing (delegated to Rust)
#   - Shared-memory synchronization (delegated to Rust)
#   - Packet transport (delegated to Rust)
#   - Binary codec hot path (delegated to Rust)
#   - Security enforcement (delegated to Rust)
#   - Native scheduling (delegated to Rust)
#
# Python provides:
#   - Developer-friendly API (this package)
#   - Async/sync interface
#   - Streaming
#   - Cancellation
#   - Structured results
#   - Capability discovery
#   - Diagnostics
#   - HTTP compatibility layer (opt-in)
"""

from __future__ import annotations

__version__ = "1.0.0"

from .runtime import AICLRuntime
from .model import Model
from .result import StructuredResult, StreamChunk, ResultStream, ok, err
from .errors import (
    AICLError,
    TransportError,
    CodecError,
    CapabilityError,
    TimeoutError,
    CancelledError,
    SecurityError,
    NativeBridgeError,
    HTTPCompatError,
    HTTPCompatDisabledError,
)
from .capabilities import Capability, CapabilitySet, BUILTIN_CAPABILITIES
from .diagnostic import TransportInfo
from ._cancellation import CancellationToken

__all__ = [
    # Core
    "AICLRuntime",
    "Model",
    # Results
    "StructuredResult",
    "StreamChunk",
    "ResultStream",
    "ok",
    "err",
    # Errors
    "AICLError",
    "TransportError",
    "CodecError",
    "CapabilityError",
    "TimeoutError",
    "CancelledError",
    "SecurityError",
    "NativeBridgeError",
    "HTTPCompatError",
    "HTTPCompatDisabledError",
    # Capabilities
    "Capability",
    "CapabilitySet",
    "BUILTIN_CAPABILITIES",
    # Diagnostics
    "TransportInfo",
    # Cancellation
    "CancellationToken",
]
