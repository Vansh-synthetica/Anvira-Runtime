"""
Shared Memory Ring Buffer for AICL
==================================

A lock-free single-producer/single-consumer ring buffer over genuine
OS-backed shared memory (`multiprocessing.shared_memory.SharedMemory`),
for AICL packet transport between two separate processes on the same
machine — no sockets, no serialization framework, no network stack.

Design
------
Layout: [HEADER][slot 0][slot 1]...[slot N-1]

Each slot: [used: 1 byte][length: u32 big-endian][payload: up to
slot_size - 5 bytes]

Correctness relies on a single invariant per slot, not on shared
head/tail counters: the `used` byte is only ever written 0->1 by the
producer and 1->0 by the consumer. Each side tracks its own read/write
cursor as a private, non-shared local integer. A single-byte write is
atomic at the hardware level on every platform Python runs on, so no
locks, atomics library, or head/tail synchronization across the shared
segment are needed for the SPSC case this class supports.

ZERO-COPY NOTE: the ring buffer's slots hold real AICL-BIN wire bytes.
Producer/consumer still each pay one copy (into/out of the shared
segment) plus AICL-BIN's own encode/decode cost — there is no cross-
process zero-copy for Python (unlike the Rust ring buffer, which can
hand back a borrowed view into the segment directly).
"""

from __future__ import annotations

import struct
import time
from multiprocessing import shared_memory
from typing import Optional

__all__ = ["SharedMemoryRing", "RingFull", "RingEmpty"]

_HEADER_FMT = ">4sQQ"  # magic, capacity, slot_size
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_SLOT_USED_OFFSET = 0
_SLOT_LEN_OFFSET = 1
_SLOT_DATA_OFFSET = 5
_MAGIC = b"SHMR"


class RingFull(Exception):
    """Raised by push_nowait() when the slot the producer needs is still in use."""


class RingEmpty(Exception):
    """Raised by pop_nowait() when the slot the consumer needs has no data yet."""


class SharedMemoryRing:
    """
    One end of a shared-memory ring buffer.

    Create the buffer once with `SharedMemoryRing.create(name, capacity,
    slot_size)`, then attach a second `SharedMemoryRing.attach(name)` from
    the other process. Exactly one side should call `push`, the other
    `pop` — this is a single-producer/single-consumer channel, not a
    general-purpose queue.
    """

    def __init__(self, shm: "shared_memory.SharedMemory", capacity: int, slot_size: int, owner: bool):
        self._shm = shm
        self._buf = shm.buf
        self.capacity = capacity
        self.slot_size = slot_size
        self._owner = owner
        self._write_idx = 0
        self._read_idx = 0

    @classmethod
    def create(cls, name: str, capacity: int, slot_size: int) -> "SharedMemoryRing":
        total = _HEADER_SIZE + capacity * slot_size
        try:
            existing = shared_memory.SharedMemory(name=name)
            existing.close()
            existing.unlink()
        except FileNotFoundError:
            pass
        shm = shared_memory.SharedMemory(name=name, create=True, size=total)
        struct.pack_into(_HEADER_FMT, shm.buf, 0, _MAGIC, capacity, slot_size)
        for i in range(capacity):
            shm.buf[cls._slot_offset(capacity, slot_size, i) + _SLOT_USED_OFFSET] = 0
        return cls(shm, capacity, slot_size, owner=True)

    @classmethod
    def attach(cls, name: str, retry_s: float = 5.0) -> "SharedMemoryRing":
        deadline = time.monotonic() + retry_s
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                shm = shared_memory.SharedMemory(name=name)
                magic, capacity, slot_size = struct.unpack_from(_HEADER_FMT, shm.buf, 0)
                if magic != _MAGIC:
                    raise ValueError(f"bad ring magic: {magic!r}")
                return cls(shm, capacity, slot_size, owner=False)
            except FileNotFoundError as exc:
                last_err = exc
                time.sleep(0.01)
        raise TimeoutError(f"shared memory {name!r} never appeared") from last_err

    @staticmethod
    def _slot_offset(capacity: int, slot_size: int, idx: int) -> int:
        return _HEADER_SIZE + (idx % capacity) * slot_size

    def _slot(self, idx: int) -> int:
        return self._slot_offset(self.capacity, self.slot_size, idx)

    def push_nowait(self, data: bytes) -> None:
        """Write one message into the next slot. Raises RingFull if the
        consumer hasn't drained that slot yet."""
        max_payload = self.slot_size - _SLOT_DATA_OFFSET
        if len(data) > max_payload:
            raise ValueError(f"payload {len(data)} > slot capacity {max_payload}")
        off = self._slot(self._write_idx)
        if self._buf[off + _SLOT_USED_OFFSET] == 1:
            raise RingFull()
        struct.pack_into(">I", self._buf, off + _SLOT_LEN_OFFSET, len(data))
        self._buf[off + _SLOT_DATA_OFFSET:off + _SLOT_DATA_OFFSET + len(data)] = data
        self._buf[off + _SLOT_USED_OFFSET] = 1  # publish — must be the last write
        self._write_idx += 1

    def push(self, data: bytes, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                return self.push_nowait(data)
            except RingFull:
                if time.monotonic() > deadline:
                    raise TimeoutError("ring buffer full — consumer not keeping up")

    def pop_nowait(self) -> bytes:
        """Read one message. Raises RingEmpty if the producer hasn't
        published to that slot yet."""
        off = self._slot(self._read_idx)
        if self._buf[off + _SLOT_USED_OFFSET] == 0:
            raise RingEmpty()
        length = struct.unpack_from(">I", self._buf, off + _SLOT_LEN_OFFSET)[0]
        data = bytes(self._buf[off + _SLOT_DATA_OFFSET:off + _SLOT_DATA_OFFSET + length])
        self._buf[off + _SLOT_USED_OFFSET] = 0  # release — must be the last write
        self._read_idx += 1
        return data

    def pop(self, timeout_s: float = 5.0) -> bytes:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                return self.pop_nowait()
            except RingEmpty:
                if time.monotonic() > deadline:
                    raise TimeoutError("ring buffer empty — producer not sending")

    def close(self) -> None:
        self._shm.close()
        if self._owner:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
