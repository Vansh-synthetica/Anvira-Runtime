"""Minimal GGUF header reader (port of ``readGgufMetadata`` in llamaServer.cjs).

Reads only what launch heuristics need: layer count, KV heads, embedding
width. Never loads tensors, so it is safe on multi-GB files.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_WANTED_SUFFIXES = (".block_count", ".attention.head_count_kv", ".attention.head_count",
                    ".embedding_length", ".context_length")
_MAX_KV = 100_000


class GgufError(Exception):
    pass


def _read(fh: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    data = fh.read(size)
    if len(data) != size:
        raise GgufError("truncated GGUF header")
    return struct.unpack(fmt, data)[0]


def _string(fh: BinaryIO) -> str:
    n = _read(fh, "<Q")
    if n > 1 << 24:
        raise GgufError("implausible string length")
    return fh.read(n).decode("utf-8", errors="replace")


def _value(fh: BinaryIO, vtype: int, keep: bool) -> Any:
    if vtype in _SCALAR:
        return _read(fh, _SCALAR[vtype])
    if vtype == 8:
        if keep:
            return _string(fh)
        fh.seek(_read(fh, "<Q"), 1)
        return None
    if vtype == 9:  # array
        etype, count = _read(fh, "<I"), _read(fh, "<Q")
        if etype in _SCALAR:
            fh.seek(struct.calcsize(_SCALAR[etype]) * count, 1)
        else:
            for _ in range(count):
                _value(fh, etype, False)
        return None
    raise GgufError(f"unknown value type {vtype}")


def read_metadata(path: Path) -> dict[str, Any] | None:
    """Return ``{architecture, n_layer, n_head, n_head_kv, n_embd, context_length}`` or None."""
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != b"GGUF":
                return None
            _read(fh, "<I")           # version
            _read(fh, "<Q")           # tensor count
            kv_count = _read(fh, "<Q")
            if kv_count > _MAX_KV:
                return None
            arch, found = None, {}
            for _ in range(kv_count):
                key = _string(fh)
                vtype = _read(fh, "<I")
                keep = key == "general.architecture" or key.endswith(_WANTED_SUFFIXES)
                val = _value(fh, vtype, keep)
                if key == "general.architecture":
                    arch = val
                elif keep:
                    found[key] = val
    except (OSError, GgufError, struct.error):
        return None

    def pick(suffix: str) -> int | None:
        for k, v in found.items():
            if k.endswith(suffix) and isinstance(v, int):
                return v
        return None

    return {"architecture": arch, "n_layer": pick(".block_count"), "n_head": pick(".attention.head_count"),
            "n_head_kv": pick(".attention.head_count_kv"), "n_embd": pick(".embedding_length"),
            "context_length": pick(".context_length")}
