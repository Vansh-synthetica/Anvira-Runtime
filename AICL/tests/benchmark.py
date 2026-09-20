"""Benchmark: AICL-BIN encode/decode latency and throughput."""
from __future__ import annotations
import time
import uuid
import tracemalloc
from aicl import encode, decode, Packet, StreamingDecoder
from aicl.bin.types import Identity, Symbol, QoS
from aicl.bin.symbol_types import S_STRING, S_F64, S_I64, S_BOOL

__all__ = []


def bm_simple(n=50_000):
    """Minimal packet — only required header fields."""
    pkt = Packet()
    t0 = time.perf_counter()
    for _ in range(n):
        data = encode(pkt)
        view = decode(data)
    t1 = time.perf_counter()
    print(f"  Simple packet:       {n:,} rounds in {t1-t0:.3f}s  "
          f"= {n/(t1-t0):,.0f} pkt/s")


def bm_small(n=30_000):
    """Small packet — origin, targets, operation, 1 symbol."""
    pkt = Packet(
        flags=1, operation=3,
        origin=Identity(0x01, "agent1"),
        targets=["classifier"],
        symbols=[Symbol(S_STRING, "I love this!")],
        confidence=0.95,
    )
    t0 = time.perf_counter()
    for _ in range(n):
        data = encode(pkt)
        view = decode(data)
        _ = view.operation, view.targets, view.symbols[0].value
    t1 = time.perf_counter()
    print(f"  Small packet:       {n:,} rounds in {t1-t0:.3f}s  "
          f"= {n/(t1-t0):,.0f} pkt/s")


def bm_medium(n=20_000):
    """Medium packet — multiple targets, 5 symbols, metadata."""
    pkt = Packet(
        flags=1, operation=3,
        origin=Identity(0x01, "orchestrator"),
        targets=["sentiment", "intent", "ner"],
        symbols=[
            Symbol(S_STRING, "What is the weather in San Francisco?"),
            Symbol(S_F64, 0.95),
            Symbol(S_I64, 42),
            Symbol(S_BOOL, True),
            Symbol(S_STRING, "weather_query"),
        ],
        confidence=0.9,
        priority=64,
        capabilities=["sentiment", "ner", "qa"],
        metadata={"model": "gpt-4", "temp": "0.7"},
    )
    t0 = time.perf_counter()
    for _ in range(n):
        data = encode(pkt)
        view = decode(data)
        _ = (view.operation, view.targets, view.capabilities,
             view.metadata, view.symbols)
    t1 = time.perf_counter()
    print(f"  Medium packet:      {n:,} rounds in {t1-t0:.3f}s  "
          f"= {n/(t1-t0):,.0f} pkt/s")


def bm_large(n=5_000):
    """Large packet — 100 symbols, 10 targets."""
    pkt = Packet(
        flags=1, operation=3,
        origin=Identity(0x01, "orchestrator"),
        targets=[f"module_{i}" for i in range(10)],
        symbols=[
            Symbol(S_STRING, f"document_chunk_{i}_text_content")
            for i in range(100)
        ],
        confidence=0.99,
        priority=32,
        capabilities=[f"cap_{i}" for i in range(5)],
    )
    t0 = time.perf_counter()
    for _ in range(n):
        data = encode(pkt)
        view = decode(data)
        _ = len(view.symbols), len(view.targets)
    t1 = time.perf_counter()
    print(f"  Large packet:       {n:,} rounds in {t1-t0:.3f}s  "
          f"= {n/(t1-t0):,.0f} pkt/s")


def bm_checksum(n=20_000):
    """Packet with CRC32 checksum."""
    pkt = Packet(
        flags=1, operation=3,
        origin=Identity(0x01, "agent"),
        symbols=[Symbol(S_STRING, "checksum test")],
    )
    t0 = time.perf_counter()
    for _ in range(n):
        data = encode(pkt, checksum=True)
        view = decode(data)
        assert view.has_trailer
    t1 = time.perf_counter()
    print(f"  With checksum:      {n:,} rounds in {t1-t0:.3f}s  "
          f"= {n/(t1-t0):,.0f} pkt/s")


def bm_streaming(n=10_000):
    """Streaming decoder: feed N packets one byte at a time."""
    pkt = encode(Packet(
        flags=1, operation=3,
        symbols=[Symbol(S_STRING, "streaming test message")],
    ))
    decoder = StreamingDecoder()
    t0 = time.perf_counter()
    for _ in range(n):
        decoder.clear()
        for byte in pkt:
            results = decoder.feed(bytes([byte]))
        results = decoder.feed(b"")
        assert len(results) == 1
    t1 = time.perf_counter()
    print(f"  Streaming (byte-by-byte): {n:,} pkts in {t1-t0:.3f}s  "
          f"= {n/(t1-t0):,.0f} pkt/s")


def bm_alloc(n=5_000):
    """Allocation profiling."""
    pkt = Packet(
        flags=1, operation=3,
        origin=Identity(0x01, "agent"),
        targets=["a", "b", "c"],
        symbols=[Symbol(S_STRING, f"sym_{i}") for i in range(10)],
    )
    tracemalloc.start()
    for _ in range(n):
        data = encode(pkt)
        view = decode(data)
        _ = view.symbols
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  Allocations ({n:,} rounds): current={cur/1024:.1f}KB  peak={peak/1024:.1f}KB")


def bm_packet_sizes():
    """Report encoded packet sizes."""
    sizes = {}
    for name, pkt in [
        ("empty", Packet()),
        ("small", Packet(flags=1, operation=3, targets=["m"],
                         symbols=[Symbol(S_STRING, "hi")])),
        ("medium", Packet(flags=1, operation=3, targets=["a","b","c"],
                          symbols=[Symbol(S_STRING, f"s{i}") for i in range(5)])),
        ("large", Packet(symbols=[Symbol(S_STRING, f"x{i}") for i in range(100)])),
    ]:
        data = encode(pkt)
        sizes[name] = len(data)

    print("\n  Packet sizes:")
    for name, size in sizes.items():
        print(f"    {name:8s}: {size:,} bytes")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    print("\n=== AICL-BIN Benchmark ===")
    bm_simple()
    bm_small()
    bm_medium()
    bm_large()
    bm_checksum()
    bm_streaming()
    bm_alloc()
    bm_packet_sizes()
    print("\nDone.")
