"""AICL-SL debug formatter — binary to human-readable text."""
from __future__ import annotations
from typing import List
import aicl.bin.constants as C
import aicl.bin.extension_tags as X
from aicl.bin.symbol_types import SYMBOL_TYPE_NAMES, S_LIST, is_vendor
from aicl.bin.codec_view import PacketView


def format_view(view: PacketView, indent: int = 0) -> str:
    pad = "  " * indent
    lines: List[str] = []
    lines.append(f"{pad}── AICL-BIN Packet ──")
    lines.append(f"{pad}magic:     {view.raw[:4].tobytes()}")
    lines.append(f"{pad}version:   0x{view.version:04x}")
    lines.append(f"{pad}flags:     0x{view.flags:04x}  {_flag_names(view.flags)}")
    if view.session_id: lines.append(f"{pad}session:   {view.session_id.hex()}")
    lines.append(f"{pad}message:   {view.message_id.hex()}")
    lines.append(f"{pad}correlation: {view.correlation_id.hex()}")
    lines.append(f"{pad}deadline:  {view.deadline_ms}ms")
    lines.append(f"{pad}payload:   {view.payload_length} bytes")
    if view.origin: lines.append(f"{pad}origin:    {view.origin}")
    if view.targets: lines.append(f"{pad}targets:   {', '.join(view.targets)}")
    from aicl.bin.ops import OPERATION_NAMES
    op_name = OPERATION_NAMES.get(view.operation, f"0x{view.operation:02x}")
    lines.append(f"{pad}operation: [0x{view.operation:02x}] {op_name}")
    if view.intent: lines.append(f"{pad}intent:    {view.intent}")
    if view.symbols:
        lines.append(f"{pad}symbols:")
        for sym in view.symbols: lines.append(f"{pad}  {format_symbol(sym)}")
    if view.response_symbols:
        lines.append(f"{pad}response_symbols:")
        for sym in view.response_symbols: lines.append(f"{pad}  {format_symbol(sym)}")
    if view.confidence != 1.0: lines.append(f"{pad}confidence: {view.confidence:.4f}")
    if view.priority != 128: lines.append(f"{pad}priority:  {view.priority}")
    if view.capabilities: lines.append(f"{pad}capabilities: {', '.join(view.capabilities)}")
    if view.qos:
        q = view.qos
        lines.append(f"{pad}qos:       pc={q.priority_class} rel={q.reliability} ord={q.ordering}")
    if view.stream_id: lines.append(f"{pad}stream_id: {view.stream_id.hex()}")
    if view.partial_result: lines.append(f"{pad}partial:   {view.partial_result[:40]!r}...")
    if view.schema_id: lines.append(f"{pad}schema_id: {view.schema_id}")
    if view.error_info:
        e = view.error_info
        lines.append(f"{pad}error:     [{e.error_code}] {e.message}")
        if e.details: lines.append(f"{pad}  details: {e.details}")
    if view.backpressure:
        bp = view.backpressure
        lines.append(f"{pad}backpressure: qd={bp.queue_depth}/{bp.max_queue}")
    if view.metadata:
        lines.append(f"{pad}metadata:")
        for k, v in view.metadata.items(): lines.append(f"{pad}  {k}: {v}")
    return "\n".join(lines)

def format_symbol(sym) -> str:
    tag_name = SYMBOL_TYPE_NAMES.get(sym.tag, f"vendor:0x{sym.tag:02x}" if is_vendor(sym.tag) else f"0x{sym.tag:02x}")
    v = sym.value
    if sym.tag == S_LIST:
        inner = ", ".join(format_symbol(s) for s in (v or []))
        return f"{tag_name}[{inner}]"
    if isinstance(v, str):
        return f"{tag_name}:\"{v}\""
    if v is None:
        return f"{tag_name}"
    return f"{tag_name}:{v!r}"


def format_tlv(ext_tag: int, value: bytes) -> str:
    """Format a raw vendor-extension operand (see extension_tags.py) —
    named `format_tlv` for source compatibility with callers written
    before the payload-structure rewrite; these are operand-level vendor
    tags now, not TLV type codes."""
    names = {v: k[4:] for k, v in vars(X).items() if k.startswith("EXT_")}
    name = names.get(ext_tag, f"VENDOR_0x{ext_tag:02x}")
    if len(value) <= 16:
        try: return f"{name}: {value.decode()!r}"
        except UnicodeDecodeError: pass
    return f"{name}: {value.hex()[:40]}"


def format_packet(data: bytes, indent: int = 0) -> str:
    from aicl import decode
    return format_view(decode(data), indent)


def _flag_names(flags: int) -> str:
    parts = []
    if flags & C.FLAG_REQUEST: parts.append("REQUEST")
    if flags & C.FLAG_RESPONSE: parts.append("RESPONSE")
    if flags & C.FLAG_STREAM_CHUNK: parts.append("STREAM_CHUNK")
    if flags & C.FLAG_STREAM_END: parts.append("STREAM_END")
    if flags & C.FLAG_ERROR: parts.append("ERROR")
    if flags & C.FLAG_CANCEL: parts.append("CANCEL")
    if flags & C.FLAG_HEARTBEAT: parts.append("HEARTBEAT")
    if flags & C.FLAG_COMPRESSED: parts.append("COMPRESSED")
    if flags & C.FLAG_ENCRYPTED: parts.append("ENCRYPTED")
    if flags & C.FLAG_TRACED: parts.append("TRACED")
    if flags & C.FLAG_PRIORITY_HIGH: parts.append("PRIORITY_HIGH")
    if flags & C.FLAG_PRIORITY_LOW: parts.append("PRIORITY_LOW")
    if flags & C.FLAG_EXTENDED: parts.append("EXTENDED")
    if flags & C.FLAG_HAS_BUFFER_REF: parts.append("HAS_BUFFER_REF")
    if flags & C.FLAG_FROZEN: parts.append("FROZEN")
    if flags & C.FLAG_HAS_TRAILER: parts.append("HAS_TRAILER")
    return "|".join(parts) if parts else "none"
