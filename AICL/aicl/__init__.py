"""aicl - Adaptive Inter-Module Communication Language.

AICL-BIN binary codec + AICL-SL debug formatter + Semantic model layer.

High-level API:
    from aicl import encode, decode, Packet, PacketView, StreamingDecoder
    from aicl.sl import format_packet
    from aicl.semantic import ModelIntent, ModelAction, SemanticAdapter, AICLGrammar
"""
from __future__ import annotations
from aicl.bin.codec_api import encode, decode
from aicl.bin.codec_packet import Packet
from aicl.bin.codec_view import PacketView
from aicl.bin.streaming import StreamingDecoder
from aicl.bin.exceptions import (
    AICLBinError, HeaderError, ChecksumError, SymbolError, TLVError,
    EncodeError, DecodeError, TruncatedPacketError, InvalidPacketError,
)
from aicl.semantic import (
    ModelIntent, ModelAction, ModelObservation, ModelRequest, ModelResult,
    SemanticAdapter, AICLGrammar, GateCheck, GateResult,
    SemanticError, ValidationError, INTENT_TO_OPCODE,
)
__all__ = [
    "encode", "decode", "Packet", "PacketView", "StreamingDecoder",
    "AICLBinError", "HeaderError", "ChecksumError", "SymbolError", "TLVError",
    "EncodeError", "DecodeError", "TruncatedPacketError", "InvalidPacketError",
    "ModelIntent", "ModelAction", "ModelObservation", "ModelRequest", "ModelResult",
    "SemanticAdapter", "AICLGrammar", "GateCheck", "GateResult",
    "SemanticError", "ValidationError", "INTENT_TO_OPCODE",
]

