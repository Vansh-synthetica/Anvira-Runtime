"""AICL-SL — Symbolic debug formatter for AICL-BIN binary packets.

Maps a binary packet to a human-readable text representation.
NEVER used on the wire — this is a debugging/diagnostic tool.

Format (per line):
    FIELD:VALUE
"""
from aicl.sl.formatter import format_packet, format_tlv, format_symbol, format_view

__all__ = ["format_packet", "format_tlv", "format_symbol", "format_view"]