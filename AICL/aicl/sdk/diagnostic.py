"""
Transport diagnostics for the AICL SDK.

Provides runtime information about the active transport, codec,
and zero-copy status. Used by `AICLRuntime.diagnose()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TransportInfo:
    """Diagnostic information about the active transport."""

    transport: str
    """Transport type: 'native' or 'http'."""

    runtime_language: str
    """Language of the runtime core: 'rust', 'python', or 'http'."""

    codec: str
    """Active codec: 'aicl-bin', 'aicl-isa', 'aicl-sl', or 'json'."""

    zero_copy: bool
    """Whether zero-copy is active."""

    http_enabled: bool
    """Whether HTTP compatibility layer is enabled."""

    native_available: bool
    """Whether the native Rust library is loadable."""

    library_version: str
    """Version string from the native library."""

    isa_version: int
    """ISA version number."""

    def report(self) -> str:
        """Human-readable diagnostic report."""
        lines = [
            "=== AICL Runtime Diagnostics ===",
            f"  transport:        {self.transport}",
            f"  runtime language: {self.runtime_language}",
            f"  codec:            {self.codec}",
            f"  zero-copy:        {'active' if self.zero_copy else 'inactive'}",
            f"  http enabled:     {'yes' if self.http_enabled else 'no (default)'}",
            f"  native available: {'yes' if self.native_available else 'no'}",
            f"  library version:  {self.library_version}",
            f"  isa version:      {self.isa_version}",
            "=" * 36,
        ]
        return "\n".join(lines)


def diagnose(
    *,
    native_available: bool = True,
    http_enabled: bool = False,
    library_version: str = "unknown",
    isa_version: int = 0,
) -> TransportInfo:
    """Build a TransportInfo snapshot."""
    if native_available:
        transport = "native"
        runtime_language = "rust"
        codec = "aicl-isa"
        zero_copy = True
    elif http_enabled:
        transport = "http"
        runtime_language = "http"
        codec = "json"
        zero_copy = False
    else:
        transport = "none"
        runtime_language = "python"
        codec = "aicl-bin"
        zero_copy = False

    return TransportInfo(
        transport=transport,
        runtime_language=runtime_language,
        codec=codec,
        zero_copy=zero_copy,
        http_enabled=http_enabled,
        native_available=native_available,
        library_version=library_version,
        isa_version=isa_version,
    )
