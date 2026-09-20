"""
Model handle for the AICL SDK.

Represents a connected model or module that can receive instructions
and return results. Provides both sync and async APIs.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, AsyncIterator, Callable, Optional, Union

from .capabilities import Capability, CapabilitySet
from .errors import CancelledError, CapabilityError, TimeoutError, TransportError
from ._cancellation import CancellationToken, NULL_TOKEN
from .result import ResultStream, StreamChunk, StructuredResult, ok, err
from .transport import NativeTransport, HttpTransport


class Model:
    """A connected model or module.

    Created by `AICLRuntime.connect()`. Provides both synchronous
    and asynchronous APIs for sending instructions and receiving results.

    Usage::

        # Sync
        result = model.call("classify", text="hello world")
        print(result.unwrap())

        # Async
        result = await model.acall("classify", text="hello world")

        # Streaming
        async for chunk in model.stream("generate", prompt="hello"):
            print(chunk.data)
    """

    __slots__ = (
        "_transport", "_capability", "_name", "_message_counter",
        "_lock", "_closed", "_timeout",
    )

    def __init__(
        self,
        transport: Union[NativeTransport, HttpTransport],
        capability: Capability,
        name: str = "",
        timeout: float = 30.0,
    ):
        self._transport = transport
        self._capability = capability
        self._name = name or capability.name
        self._message_counter = 0
        self._lock = threading.Lock()
        self._closed = False
        self._timeout = timeout

    @property
    def name(self) -> str:
        return self._name

    @property
    def capability(self) -> Capability:
        return self._capability

    @property
    def is_native(self) -> bool:
        return self._transport.is_native

    def _next_message_id(self) -> str:
        with self._lock:
            self._message_counter += 1
            return f"msg-{self._message_counter:08d}"

    # ── Synchronous API ─────────────────────────────────────────────────────

    def call(
        self,
        instruction: str,
        *,
        timeout: Optional[float] = None,
        cancellation: Optional[CancellationToken] = None,
        **kwargs: Any,
    ) -> StructuredResult:
        """Send a synchronous instruction and return a structured result.

        Args:
            instruction: The operation name (e.g., 'classify', 'generate').
            timeout: Override the default timeout (seconds).
            cancellation: Optional cancellation token.
            **kwargs: Additional arguments for the instruction.

        Returns:
            StructuredResult with the decoded response.
        """
        if self._closed:
            return err(TransportError("Model is closed"))

        timeout = timeout or self._timeout
        token = cancellation or NULL_TOKEN
        msg_id = self._next_message_id()

        start = time.monotonic()
        try:
            token.check()

            # Build the wire frame
            wire = self._encode_instruction(instruction, msg_id, kwargs)

            token.check()

            # Send and receive
            if isinstance(self._transport, HttpTransport):
                response_data = self._transport.send(wire)
            else:
                # Native path: encode + decode (simulated round-trip)
                native_buf = self._transport.encode(wire)
                response_data = native_buf.as_bytes()

            token.check()

            # Decode the response
            elapsed_ms = (time.monotonic() - start) * 1000
            if isinstance(self._transport, HttpTransport):
                decoded = self._transport.decode(response_data)
                return ok(
                    data=decoded,
                    message_id=msg_id,
                    latency_ms=elapsed_ms,
                )
            else:
                view = self._transport.decode(response_data)
                return StructuredResult(
                    data={"opcode": view.opcode, "operands": view.operand_count},
                    opcode=view.opcode,
                    message_id=msg_id,
                    latency_ms=elapsed_ms,
                )

        except CancelledError:
            return err(CancelledError())
        except TimeoutError:
            return err(TimeoutError(f"call() timed out after {timeout}s"))
        except Exception as e:
            return err(e)

    def classify(
        self,
        text: str,
        categories: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> StructuredResult:
        """Classify text into categories."""
        return self.call(
            "classify",
            text=text,
            categories=categories or [],
            **kwargs,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> StructuredResult:
        """Generate text from a prompt."""
        return self.call(
            "generate",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def reason(
        self,
        question: str,
        *,
        evidence: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> StructuredResult:
        """Chain-of-thought reasoning."""
        return self.call(
            "reason",
            question=question,
            evidence=evidence or [],
            **kwargs,
        )

    def summarize(
        self,
        text: str,
        *,
        max_length: int = 200,
        **kwargs: Any,
    ) -> StructuredResult:
        """Summarize text."""
        return self.call(
            "summarize",
            text=text,
            max_length=max_length,
            **kwargs,
        )

    # ── Asynchronous API ────────────────────────────────────────────────────

    async def acall(
        self,
        instruction: str,
        *,
        timeout: Optional[float] = None,
        cancellation: Optional[CancellationToken] = None,
        **kwargs: Any,
    ) -> StructuredResult:
        """Async version of call().

        Sends the instruction in a thread pool to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.call(
                instruction,
                timeout=timeout,
                cancellation=cancellation,
                **kwargs,
            ),
        )

    async def aclassify(
        self,
        text: str,
        categories: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> StructuredResult:
        return await self.acall("classify", text=text, categories=categories or [], **kwargs)

    async def agenerate(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> StructuredResult:
        return await self.acall(
            "generate", prompt=prompt, max_tokens=max_tokens,
            temperature=temperature, **kwargs,
        )

    # ── Streaming ───────────────────────────────────────────────────────────

    async def stream(
        self,
        instruction: str,
        *,
        cancellation: Optional[CancellationToken] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream results from an instruction.

        Yields StreamChunk objects as data becomes available.
        """
        msg_id = self._next_message_id()
        token = cancellation or NULL_TOKEN

        # Build and send the instruction
        wire = self._encode_instruction(instruction, msg_id, kwargs)
        token.check()

        # For now, simulate streaming with a single chunk
        # In production, this would read from the Rust streaming API
        if isinstance(self._transport, HttpTransport):
            response = self._transport.send(wire)
            yield StreamChunk(
                data=response,
                chunk_index=0,
                is_final=True,
                metadata={"message_id": msg_id},
            )
        else:
            native_buf = self._transport.encode(wire)
            view = self._transport.decode(native_buf.as_bytes())
            yield StreamChunk(
                data={"opcode": view.opcode, "operands": view.operand_count},
                chunk_index=0,
                is_final=True,
                metadata={"message_id": msg_id},
            )

    # ── Internal helpers ────────────────────────────────────────────────────

    def _encode_instruction(
        self, instruction: str, msg_id: str, kwargs: dict[str, Any]
    ) -> bytes:
        """Encode an instruction into wire bytes.

        This builds a minimal AICL-BIN frame for the instruction.
        """
        import struct

        # Build a simple frame: [4B magic][2B version][16B msg_id][2B flags][4B payload_len][payload]
        magic = b"AICL"
        version = struct.pack(">H", 1)  # ISA v1
        msg_id_bytes = msg_id.encode("utf-8").ljust(16, b"\0")[:16]
        flags = struct.pack(">H", 0)

        # Build payload: instruction + kwargs as simple key=value pairs
        payload_parts = [instruction.encode("utf-8")]
        for k, v in kwargs.items():
            payload_parts.append(f"{k}={v}".encode("utf-8"))
        payload = b"\x00".join(payload_parts)

        payload_len = struct.pack(">I", len(payload))

        return magic + version + msg_id_bytes + flags + payload_len + payload

    def close(self) -> None:
        """Close the model and release resources."""
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        mode = "native" if self.is_native else "http"
        return f"<Model name={self._name!r} capability={self._capability.name} mode={mode}>"
