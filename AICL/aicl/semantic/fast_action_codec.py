"""Fast semantic codec — eliminates UUID and intermediate representations.

Bypasses the BinPacket intermediate and encodes directly from ModelAction
fields to binary wire format. Uses atomic counters instead of UUID.

This is the hot path for edge AI: consumer CPU, 4GB-class GPU, 8-16GB RAM.

Wire format (optimized):
    [1B version][1B intent_code][2B flags][16B session_id][16B message_id]
    [16B correlation_id][2B target_len][target_utf8]
    [2B inputs_json_len][inputs_json][2B params_json_len][params_json]
    [2B result_json_len][result_json][4B deadline_ms][2B confidence_x1000]
    [2B metadata_json_len][metadata_json]

Usage:
    from aicl.semantic.fast_action_codec import FastActionEncoder, FastActionDecoder

    encoder = FastActionEncoder(origin="classifier")
    wire = encoder.encode(action)
    action = encoder.decode(wire)
"""
from __future__ import annotations

import json
import struct
import threading
from typing import Any, Dict, Optional, Tuple

from aicl.semantic.types import ModelIntent, ModelAction, SemanticError

__all__ = ["FastActionEncoder", "FastActionDecoder"]

_VERSION = 0x02  # Optimized v2

# Intent -> compact 1-byte code (using AICL-ISA opcodes directly)
_INTENT_TO_BYTE: Dict[ModelIntent, int] = {
    ModelIntent.CLASSIFY: 0x43,       # ISA: Classify
    ModelIntent.GENERATE: 0x46,       # ISA: Execute
    ModelIntent.REASON: 0x44,         # ISA: Reason
    ModelIntent.EMBED: 0x42,          # ISA: Embed
    ModelIntent.RANK: 0x47,           # vendor: rank
    ModelIntent.SUMMARIZE: 0x48,      # vendor: summarize
    ModelIntent.SYNTHESIZE: 0x49,     # vendor: synthesize
    ModelIntent.VERIFY: 0x4A,         # vendor: verify
    ModelIntent.EXTRACT: 0x4B,        # vendor: extract
    ModelIntent.TRANSFORM: 0x4C,      # vendor: transform
    ModelIntent.PLAN: 0x4D,           # vendor: plan
    ModelIntent.EVALUATE: 0x4E,       # vendor: evaluate
    ModelIntent.CALL_MODEL: 0x40,     # ISA: ModelCall
    ModelIntent.CALL_TOOL: 0x41,      # ISA: ToolCall
    ModelIntent.READ_MEMORY: 0x20,    # ISA: MemoryRead
    ModelIntent.WRITE_MEMORY: 0x21,   # ISA: MemoryWrite
    ModelIntent.DELETE_MEMORY: 0x22,  # ISA: MemoryDelete
    ModelIntent.INDEX_QUERY: 0x23,    # ISA: IndexQuery
    ModelIntent.INDEX_UPSERT: 0x24,   # ISA: IndexUpsert
    ModelIntent.ROUTE: 0x03,          # ISA: Discover
    ModelIntent.FILTER: 0x04,         # ISA: Ping
    ModelIntent.RETURN_RESULT: 0x11,  # ISA: Return
    ModelIntent.STREAM_RESULT: 0x12,  # ISA: Stream
    ModelIntent.CANCEL: 0x09,         # ISA: Cancel
}

_BYTE_TO_INTENT: Dict[int, ModelIntent] = {v: k for k, v in _INTENT_TO_BYTE.items()}

# Flag bits
_FLAG_REQUEST = 0x0001
_FLAG_STREAM = 0x0004
_FLAG_CANCEL = 0x0020
_FLAG_HAS_PARAMS = 0x0100
_FLAG_HAS_RESULT = 0x0200
_FLAG_HAS_METADATA = 0x0400


class _AtomicCounter:
    """Thread-safe atomic counter for message IDs (no UUID)."""

    def __init__(self) -> None:
        self._counter = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter


_counter = _AtomicCounter()


def _make_message_id() -> bytes:
    """Generate a 16-byte message ID from atomic counter (no UUID)."""
    n = _counter.next()
    return b"\x00" * 8 + n.to_bytes(8, "little")


def _encode_str16(value: str) -> bytes:
    b = value.encode("utf-8")
    return struct.pack(">H", len(b)) + b


def _decode_str16(data: memoryview, pos: int) -> Tuple[str, int]:
    length = struct.unpack_from(">H", data, pos)[0]
    pos += 2
    return data[pos:pos + length].tobytes().decode("utf-8"), pos + length


def _encode_json_small(obj: Any) -> bytes:
    if not obj:
        return b"\x00\x00"
    b = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return struct.pack(">H", len(b)) + b


def _decode_json_small(data: memoryview, pos: int) -> Tuple[Any, int]:
    length = struct.unpack_from(">H", data, pos)[0]
    pos += 2
    if length == 0:
        return {}, pos
    return json.loads(data[pos:pos + length].tobytes()), pos + length


class FastActionEncoder:
    """Encode ModelAction directly to binary without BinPacket overhead.

    Optimizations over standard path:
    1. No UUID generation (atomic counter, ~1ns vs ~200ns)
    2. No BinPacket intermediate representation
    3. No TLV symbol encoding (direct JSON for inputs/params)
    4. Single b''.join at the end (Python-optimized C path)
    """

    __slots__ = ("origin", "_session_id")

    def __init__(self, origin: str = "model", session_id: Optional[bytes] = None):
        self.origin = origin
        self._session_id = session_id or _make_message_id()

    def encode(self, action: ModelAction) -> bytes:
        """Encode a ModelAction to compact binary."""
        parts = []

        # Header: version(1) + intent(1) + flags(4) = 6 bytes
        intent_byte = _INTENT_TO_BYTE.get(action.intent, 0x00)
        flags = _FLAG_REQUEST
        if action.stream:
            flags |= _FLAG_STREAM
        if action.cancel:
            flags |= _FLAG_CANCEL
        if action.params:
            flags |= _FLAG_HAS_PARAMS
        if action.result:
            flags |= _FLAG_HAS_RESULT
        if action.metadata:
            flags |= _FLAG_HAS_METADATA
        parts.append(struct.pack("<BBI", _VERSION, intent_byte, flags))

        # Session ID (16 bytes, reused per-adapter instance)
        parts.append(self._session_id)

        # Message ID (atomic counter, no UUID)
        parts.append(_make_message_id())

        # Correlation ID (16 bytes)
        if action.correlation_id:
            parts.append(bytes.fromhex(action.correlation_id.replace("-", "")))
        else:
            parts.append(b"\x00" * 16)

        # Target
        parts.append(_encode_str16(action.target))

        # Inputs as compact JSON
        parts.append(_encode_json_small(action.inputs))

        # Params as compact JSON (only if present)
        if action.params:
            parts.append(_encode_json_small(action.params))

        # Result as compact JSON (only if present)
        if action.result:
            parts.append(_encode_json_small(action.result))

        # Deadline (4 bytes)
        parts.append(struct.pack("<I", action.deadline_ms & 0xFFFFFFFF))

        # Confidence as uint16 (x1000)
        conf_int = min(int(action.confidence * 1000), 65535)
        parts.append(struct.pack("<H", conf_int))

        # Metadata as compact JSON (only if present)
        if action.metadata:
            parts.append(_encode_json_small(action.metadata))

        return b"".join(parts)

    def encode_to_ring(self, action: ModelAction, ring_push_fn) -> bool:
        """Encode and push directly to a ring buffer."""
        wire = self.encode(action)
        return ring_push_fn(wire)


class FastActionDecoder:
    """Decode binary wire to ModelAction (fast path)."""

    def decode(self, data: bytes | memoryview) -> ModelAction:
        if isinstance(data, memoryview):
            mv = data
        else:
            mv = memoryview(data)

        pos = 0

        # Header: version(1) + intent(1) + flags(4)
        version, intent_byte, flags = struct.unpack_from("<BBI", mv, pos)
        pos += 6

        if version != _VERSION:
            raise SemanticError(f"Unknown fast codec version: {version}")

        intent = _BYTE_TO_INTENT.get(intent_byte)
        if intent is None:
            raise SemanticError(f"Unknown intent byte: 0x{intent_byte:02x}")

        # Session ID (skip)
        pos += 16

        # Message ID (skip)
        pos += 16

        # Correlation ID
        corr_bytes = mv[pos:pos + 16].tobytes()
        pos += 16
        correlation_id = corr_bytes.hex() if corr_bytes != b"\x00" * 16 else ""

        # Target
        target, pos = _decode_str16(mv, pos)

        # Inputs
        inputs, pos = _decode_json_small(mv, pos)

        # Params
        params = {}
        if flags & _FLAG_HAS_PARAMS:
            params, pos = _decode_json_small(mv, pos)

        # Result
        result = {}
        if flags & _FLAG_HAS_RESULT:
            result, pos = _decode_json_small(mv, pos)

        # Deadline
        deadline_ms = struct.unpack_from("<I", mv, pos)[0]
        pos += 4

        # Confidence
        conf_int = struct.unpack_from("<H", mv, pos)[0]
        pos += 2
        confidence = conf_int / 1000.0

        # Metadata
        metadata = {}
        if flags & _FLAG_HAS_METADATA:
            metadata, pos = _decode_json_small(mv, pos)

        return ModelAction(
            intent=intent,
            target=target,
            inputs=inputs,
            params=params,
            result=result,
            confidence=confidence,
            stream=bool(flags & _FLAG_STREAM),
            cancel=bool(flags & _FLAG_CANCEL),
            correlation_id=correlation_id,
            deadline_ms=deadline_ms,
            metadata=metadata,
        )
