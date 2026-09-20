"""
Structured result types for AICL SDK operations.

Results carry the decoded payload along with metadata about the operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass
class StructuredResult(Generic[T]):
    """A structured result from an AICL operation.

    Carries the decoded payload plus operation metadata.
    """

    data: T
    """The decoded payload (type depends on the operation)."""

    opcode: int = 0
    """The opcode of the operation that produced this result."""

    message_id: str = ""
    """Unique message ID for request-response correlation."""

    correlation_id: str = ""
    """Correlation ID linking request to response."""

    latency_ms: float = 0.0
    """Round-trip latency in milliseconds."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata from the transport."""

    @property
    def ok(self) -> bool:
        """Check if the result is successful."""
        return not isinstance(self.data, Exception)

    def unwrap(self) -> T:
        """Get the data, or raise if it's an error."""
        if isinstance(self.data, Exception):
            raise self.data
        return self.data

    def map(self, fn: Any) -> StructuredResult:
        """Transform the data with a function."""
        if isinstance(self.data, Exception):
            return self
        return StructuredResult(
            data=fn(self.data),
            opcode=self.opcode,
            message_id=self.message_id,
            correlation_id=self.correlation_id,
            latency_ms=self.latency_ms,
            metadata=self.metadata,
        )

    def __repr__(self) -> str:
        if isinstance(self.data, Exception):
            return f"<StructuredResult error={self.data!r}>"
        return f"<StructuredResult ok data={self.data!r}>"


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""

    data: Any
    """Chunk data (type depends on the operation)."""

    chunk_index: int = 0
    """Index of this chunk in the stream."""

    is_final: bool = False
    """True if this is the last chunk."""

    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        marker = " [FINAL]" if self.is_final else ""
        return f"<StreamChunk idx={self.chunk_index}{marker}>"


class ResultStream:
    """Async iterator of StreamChunks.

    Wraps the native streaming API and provides an async iteration interface.
    """

    __init__ = None  # placeholder, defined below

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


# Fix: define __init__ properly
class ResultStream:
    """Async iterator of StreamChunks."""

    def __init__(self, chunks: list[StreamChunk]):
        self._chunks = chunks
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> StreamChunk:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk

    async def collect(self) -> list[StreamChunk]:
        """Collect all chunks into a list."""
        return [chunk async for chunk in self]

    async def to_bytes(self) -> bytes:
        """Collect all chunks and concatenate their data."""
        parts = []
        async for chunk in self:
            if isinstance(chunk.data, bytes):
                parts.append(chunk.data)
            else:
                parts.append(str(chunk.data).encode())
        return b"".join(parts)


# Convenience constructors
def ok(data: Any, **kwargs: Any) -> StructuredResult:
    """Create a successful result."""
    return StructuredResult(data=data, **kwargs)


def err(error: Exception, **kwargs: Any) -> StructuredResult:
    """Create an error result."""
    return StructuredResult(data=error, **kwargs)
