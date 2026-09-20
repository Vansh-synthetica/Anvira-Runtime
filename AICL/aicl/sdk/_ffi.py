"""
Low-level FFI bridge to the native AICL Rust core.

This module loads the compiled Rust shared library (aicl_core.dll/.so/.dylib)
and exposes raw C-callable functions. It never falls back to localhost HTTP.

Ownership contract:
  - All handles returned by Rust are opaque pointers
  - Python MUST call the corresponding _free function when done
  - Buffer references are non-owning: the caller must keep the memory alive
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path
from typing import Optional

# ── Library discovery ────────────────────────────────────────────────────────

_LIB_NAME = "aicl_core"
_SEARCH_DIRS = [
    Path(__file__).parent.parent.parent / "core-rust" / "target" / "release",
    Path(__file__).parent.parent.parent / "core-rust" / "target" / "debug",
    Path(__file__).parent.parent / "lib",
    Path(__file__).parent,
]


def _find_library() -> ctypes.CDLL:
    """Locate and load the native AICL core library.

    Searches in order:
      1. AICL_CORE_LIB environment variable (explicit path)
      2. core-rust/target/release/ (optimized build)
      3. core-rust/target/debug/ (debug build)
      4. aicl/lib/ (installed location)
      5. System library path (ctypes.util.find_library)
    """
    # 1. Explicit env var
    env_path = os.environ.get("AICL_CORE_LIB")
    if env_path:
        return ctypes.CDLL(env_path)

    # 2-4. Project-relative search
    if sys.platform == "win32":
        lib_file = f"{_LIB_NAME}.dll"
    elif sys.platform == "darwin":
        lib_file = f"lib{_LIB_NAME}.dylib"
    else:
        lib_file = f"lib{_LIB_NAME}.so"

    for search_dir in _SEARCH_DIRS:
        candidate = search_dir / lib_file
        if candidate.exists():
            return ctypes.CDLL(str(candidate))

    # 5. System path
    found = ctypes.util.find_library(_LIB_NAME)
    if found:
        return ctypes.CDLL(found)

    raise OSError(
        f"Cannot find AICL native library '{lib_file}'.\n"
        f"Searched: {[str(d) for d in _SEARCH_DIRS]}\n"
        f"Set AICL_CORE_LIB environment variable to the full path, or\n"
        f"build with: cargo build --release (in core-rust/)"
    )


_lib: Optional[ctypes.CDLL] = None


def get_lib() -> ctypes.CDLL:
    """Get the native library, loading it lazily on first access."""
    global _lib
    if _lib is None:
        _lib = _find_library()
        _setup_signatures()
    return _lib


# ── Opaque handle types ─────────────────────────────────────────────────────

class _AiclCtx(ctypes.Structure):
    _fields_ = [("_opaque", ctypes.c_char * 0)]


class _AiclPktView(ctypes.Structure):
    _fields_ = [("_opaque", ctypes.c_char * 0)]


class _AiclHandle(ctypes.Structure):
    _fields_ = [("_opaque", ctypes.c_char * 0)]


class _AiclBuffer(ctypes.Structure):
    _fields_ = [("_opaque", ctypes.c_char * 0)]


class _AiclStream(ctypes.Structure):
    _fields_ = [("_opaque", ctypes.c_char * 0)]


# ── C-compatible structs ────────────────────────────────────────────────────

class BufRef(ctypes.Structure):
    """C-compatible buffer reference (16 bytes)."""
    _fields_ = [
        ("ptr", ctypes.c_void_p),
        ("length", ctypes.c_uint32),
        ("alignment", ctypes.c_uint16),
        ("tag", ctypes.c_uint16),
        ("ownership", ctypes.c_uint8),
        ("lifetime", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 2),
    ]


class StatusCode(ctypes.c_int):
    """C-compatible status codes."""
    OK = 0
    BAD_HEADER = 1
    UNSUPPORTED_VERSION = 2
    TRUNCATED = 3
    CHECKSUM = 4
    PAYLOAD_TOO_LARGE = 5
    OPCODE_INVALID = 6
    OPERAND_COUNT = 7
    UNKNOWN_TYPE = 8
    NO_HANDLER = 9
    EMPTY = 10
    BUFFER_FULL = 11
    TRANSPORT_CLOSED = 12
    NULL_POINTER = 13
    INVALID_LEN = 14
    INTERNAL = 100


_STATUS_NAMES = {
    0: "OK",
    1: "BAD_HEADER",
    2: "UNSUPPORTED_VERSION",
    3: "TRUNCATED",
    4: "CHECKSUM",
    5: "PAYLOAD_TOO_LARGE",
    6: "OPCODE_INVALID",
    7: "OPERAND_COUNT",
    8: "UNKNOWN_TYPE",
    9: "NO_HANDLER",
    10: "EMPTY",
    11: "BUFFER_FULL",
    12: "TRANSPORT_CLOSED",
    13: "NULL_POINTER",
    14: "INVALID_LEN",
    100: "INTERNAL",
}


# ── Signature setup ─────────────────────────────────────────────────────────

def _setup_signatures() -> None:
    """Set argument and return types for all FFI functions."""
    lib = _lib

    # Context lifecycle
    lib.aicl_context_new.argtypes = []
    lib.aicl_context_new.restype = ctypes.POINTER(_AiclCtx)

    lib.aicl_context_free.argtypes = [ctypes.POINTER(_AiclCtx)]
    lib.aicl_context_free.restype = None

    # Decode
    lib.aicl_decode.argtypes = [
        ctypes.POINTER(_AiclCtx),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.POINTER(_AiclPktView)),
    ]
    lib.aicl_decode.restype = ctypes.c_int

    # Packet view accessors
    lib.aicl_pkt_view_opcode.argtypes = [ctypes.POINTER(_AiclPktView)]
    lib.aicl_pkt_view_opcode.restype = ctypes.c_uint

    lib.aicl_pkt_view_flags.argtypes = [ctypes.POINTER(_AiclPktView)]
    lib.aicl_pkt_view_flags.restype = ctypes.c_uint

    lib.aicl_pkt_view_operand_count.argtypes = [ctypes.POINTER(_AiclPktView)]
    lib.aicl_pkt_view_operand_count.restype = ctypes.c_uint

    lib.aicl_pkt_view_free.argtypes = [ctypes.POINTER(_AiclPktView)]
    lib.aicl_pkt_view_free.restype = None

    # Buffer ref operations
    lib.aicl_bufref_from_ptr.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.aicl_bufref_from_ptr.restype = BufRef

    lib.aicl_bufref_to_ptr.argtypes = [
        ctypes.POINTER(BufRef),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.aicl_bufref_to_ptr.restype = ctypes.c_int

    lib.aicl_bufref_is_empty.argtypes = [ctypes.POINTER(BufRef)]
    lib.aicl_bufref_is_empty.restype = ctypes.c_int

    # Buffer lifecycle
    lib.aicl_buffer_from_ptr.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.aicl_buffer_from_ptr.restype = ctypes.POINTER(_AiclBuffer)

    lib.aicl_buffer_borrow.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.aicl_buffer_borrow.restype = ctypes.POINTER(_AiclBuffer)

    lib.aicl_buffer_from_ptr_shared.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.aicl_buffer_from_ptr_shared.restype = ctypes.POINTER(_AiclBuffer)

    lib.aicl_buffer_clone.argtypes = [ctypes.POINTER(_AiclBuffer)]
    lib.aicl_buffer_clone.restype = ctypes.POINTER(_AiclBuffer)

    lib.aicl_buffer_free.argtypes = [ctypes.POINTER(_AiclBuffer)]
    lib.aicl_buffer_free.restype = None

    lib.aicl_buffer_len.argtypes = [ctypes.POINTER(_AiclBuffer)]
    lib.aicl_buffer_len.restype = ctypes.c_uint32

    lib.aicl_buffer_data.argtypes = [ctypes.POINTER(_AiclBuffer)]
    lib.aicl_buffer_data.restype = ctypes.c_void_p

    # Handle lifecycle
    lib.aicl_handle_free.argtypes = [ctypes.POINTER(_AiclHandle)]
    lib.aicl_handle_free.restype = None

    lib.aicl_handle_clone.argtypes = [ctypes.POINTER(_AiclHandle)]
    lib.aicl_handle_clone.restype = ctypes.POINTER(_AiclHandle)

    lib.aicl_handle_refcount.argtypes = [ctypes.POINTER(_AiclHandle)]
    lib.aicl_handle_refcount.restype = ctypes.c_uint

    # Stream operations
    lib.aicl_stream_new.argtypes = []
    lib.aicl_stream_new.restype = ctypes.POINTER(_AiclStream)

    lib.aicl_stream_push.argtypes = [ctypes.POINTER(_AiclStream), ctypes.POINTER(_AiclBuffer)]
    lib.aicl_stream_push.restype = ctypes.c_int

    lib.aicl_stream_total_len.argtypes = [ctypes.POINTER(_AiclStream)]
    lib.aicl_stream_total_len.restype = ctypes.c_uint32

    lib.aicl_stream_chunk_count.argtypes = [ctypes.POINTER(_AiclStream)]
    lib.aicl_stream_chunk_count.restype = ctypes.c_uint

    lib.aicl_stream_free.argtypes = [ctypes.POINTER(_AiclStream)]
    lib.aicl_stream_free.restype = None

    # Version
    lib.aicl_version.argtypes = []
    lib.aicl_version.restype = ctypes.c_char_p

    lib.aicl_isa_version.argtypes = []
    lib.aicl_isa_version.restype = ctypes.c_uint


# ── High-level helpers ───────────────────────────────────────────────────────

class NativeError(Exception):
    """Error from the native AICL core."""

    def __init__(self, code: int, message: str = ""):
        name = _STATUS_NAMES.get(code, f"UNKNOWN({code})")
        self.code = code
        self.name = name
        super().__init__(f"AICL native error: {name} ({code}){': ' + message if message else ''}")


def check_status(code: int, context: str = "") -> None:
    """Raise NativeError if status code is not OK."""
    if code != StatusCode.OK:
        raise NativeError(code, context)


class AiclContext:
    """Managed handle to an AICL codec context."""

    __slots__ = ("_ptr",)

    def __init__(self):
        self._ptr = get_lib().aicl_context_new()
        if not self._ptr:
            raise NativeError(-1, "Failed to create AICL context")

    def decode(self, data: bytes) -> int:
        """Decode wire data and return a raw view pointer (as Python int).

        The caller is responsible for freeing the view via `free_view`.
        """
        buf = ctypes.create_string_buffer(data)
        view_ptr = ctypes.POINTER(_AiclPktView)()
        status = get_lib().aicl_decode(
            self._ptr,
            buf,
            len(data),
            ctypes.byref(view_ptr),
        )
        check_status(status, "decode failed")
        return ctypes.cast(view_ptr, ctypes.c_void_p).value

    @staticmethod
    def free_view(view_ptr: int) -> None:
        """Free a decoded packet view."""
        if view_ptr:
            ptr = ctypes.cast(ctypes.c_void_p(view_ptr), ctypes.POINTER(_AiclPktView))
            get_lib().aicl_pkt_view_free(ptr)

    @staticmethod
    def view_opcode(view_ptr: int) -> int:
        """Get the opcode byte from a view."""
        ptr = ctypes.cast(ctypes.c_void_p(view_ptr), ctypes.POINTER(_AiclPktView))
        return get_lib().aicl_pkt_view_opcode(ptr)

    @staticmethod
    def view_flags(view_ptr: int) -> int:
        """Get the flags from a view."""
        ptr = ctypes.cast(ctypes.c_void_p(view_ptr), ctypes.POINTER(_AiclPktView))
        return get_lib().aicl_pkt_view_flags(ptr)

    @staticmethod
    def view_operand_count(view_ptr: int) -> int:
        """Get the operand count from a view."""
        ptr = ctypes.cast(ctypes.c_void_p(view_ptr), ctypes.POINTER(_AiclPktView))
        return get_lib().aicl_pkt_view_operand_count(ptr)

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            get_lib().aicl_context_free(self._ptr)


class NativeBuffer:
    """Managed native buffer with automatic cleanup."""

    __slots__ = ("_ptr",)

    def __init__(self, data: bytes, shared: bool = False):
        buf = ctypes.create_string_buffer(data)
        lib = get_lib()
        if shared:
            self._ptr = lib.aicl_buffer_from_ptr_shared(buf, len(data))
        else:
            self._ptr = lib.aicl_buffer_from_ptr(buf, len(data))
        if not self._ptr:
            raise NativeError(-1, "Failed to create native buffer")

    @property
    def length(self) -> int:
        return get_lib().aicl_buffer_len(self._ptr)

    @property
    def data_ptr(self) -> int:
        """Raw pointer to the buffer data."""
        return get_lib().aicl_buffer_data(self._ptr)

    def as_bytes(self) -> bytes:
        """Copy data out of the native buffer into a Python bytes object."""
        ptr = self.data_ptr
        length = self.length
        if not ptr or length == 0:
            return b""
        buf_type = ctypes.c_uint8 * length
        buf = buf_type.from_address(ptr)
        return bytes(buf)

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            get_lib().aicl_buffer_free(self._ptr)


def get_version() -> str:
    """Get the native AICL library version string."""
    ver = get_lib().aicl_version()
    return ver.decode("utf-8") if ver else "unknown"


def get_isa_version() -> int:
    """Get the ISA version number."""
    return get_lib().aicl_isa_version()
