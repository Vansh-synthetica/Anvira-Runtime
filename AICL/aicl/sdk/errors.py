"""
AICL SDK error hierarchy.

All errors inherit from AICLError. HTTP-specific errors are only raised
when the HTTP compatibility layer is explicitly enabled.
"""

from __future__ import annotations

from typing import Any, Optional


class AICLError(Exception):
    """Base error for all AICL SDK operations."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ):
        self.code = code or "AICL_ERROR"
        self.context = context or {}
        super().__init__(message)


class TransportError(AICLError):
    """Transport-level error (connection, timeout, etc.)."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(message, code="TRANSPORT_ERROR", **kwargs)


class CodecError(AICLError):
    """Encoding/decoding error."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(message, code="CODEC_ERROR", **kwargs)


class CapabilityError(AICLError):
    """Requested capability is not available."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(message, code="CAPABILITY_ERROR", **kwargs)


class TimeoutError(AICLError):
    """Operation timed out."""

    def __init__(self, message: str = "Operation timed out", **kwargs: Any):
        super().__init__(message, code="TIMEOUT", **kwargs)


class CancelledError(AICLError):
    """Operation was cancelled."""

    def __init__(self, message: str = "Operation cancelled", **kwargs: Any):
        super().__init__(message, code="CANCELLED", **kwargs)


class SecurityError(AICLError):
    """Security policy violation."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(message, code="SECURITY_ERROR", **kwargs)


class NativeBridgeError(AICLError):
    """Error communicating with the native Rust core."""

    def __init__(self, message: str, native_code: int = 0, **kwargs: Any):
        self.native_code = native_code
        super().__init__(
            message, code="NATIVE_BRIDGE_ERROR",
            context={"native_code": native_code}, **kwargs
        )


class HTTPCompatError(AICLError):
    """Error in the HTTP compatibility layer.

    Only raised when HTTP mode is explicitly enabled.
    """

    def __init__(self, message: str, status_code: int = 0, **kwargs: Any):
        self.status_code = status_code
        super().__init__(
            message, code="HTTP_COMPAT_ERROR",
            context={"status_code": status_code}, **kwargs
        )


class HTTPCompatDisabledError(AICLError):
    """Attempted to use HTTP but it is not enabled.

    This is a programming error — the caller should either:
      1. Enable HTTP explicitly: AICLRuntime(http=True)
      2. Use the native transport (default)
    """

    def __init__(self, **kwargs: Any):
        super().__init__(
            "HTTP transport is not enabled. "
            "Pass http=True to AICLRuntime() to enable it, or use the default native transport.",
            code="HTTP_DISABLED",
            **kwargs,
        )
