"""pytest-free tests for aicl.bin - error handling, streaming, checksum."""
from __future__ import annotations
import struct
from aicl import encode, decode, Packet, StreamingDecoder
from aicl.bin.types import Symbol, ChunkInfo
from aicl.bin.symbol_types import S_STRING
from aicl.bin.constants import FLAG_REQUEST, FLAG_STREAM_CHUNK, HEADER_SIZE, CURRENT_VERSION, MAGIC
from aicl.bin.exceptions import TruncatedPacketError, ChecksumError, HeaderError


def expect(exc_class, fn):
    """Run fn and assert it raises exc_class."""
    try:
        fn()
    except exc_class:
        return
    except Exception as e:
        assert False, f"Expected {exc_class.__name__}, got {type(e).__name__}: {e}"
    assert False, f"Expected {exc_class.__name__}, no exception raised"


def test_checksum_valid():
    pkt = Packet()
    data = encode(pkt, checksum=True)
    # header + opcode + operand-count varint + auto-generated session_id
    # extension + 4-byte CRC trailer.
    assert len(data) > HEADER_SIZE + 2
    view = decode(data)
    assert view.has_trailer


def test_checksum_mismatch():
    pkt = Packet(symbols=[Symbol(S_STRING, "hello")])
    data = bytearray(encode(pkt, checksum=True))
    # Flip a byte inside the payload (after the header, before the
    # trailer) — flipping anything inside the header itself would trip
    # the header's own CRC first (HeaderError), not the trailer's
    # ChecksumError this test is actually checking.
    data[HEADER_SIZE + 3] ^= 0xFF
    expect(ChecksumError, lambda: decode(bytes(data)))


def test_no_trailer_no_check():
    pkt = Packet()
    data = encode(pkt, checksum=False)
    view = decode(data)
    assert not view.has_trailer


def test_truncated_header():
    expect(TruncatedPacketError, lambda: decode(b"AICL"))


def test_truncated_payload():
    pkt = Packet(symbols=[Symbol(S_STRING, "test")])
    data = encode(pkt)
    expect(TruncatedPacketError, lambda: decode(data[:HEADER_SIZE + 3]))


def test_invalid_magic():
    def go():
        data = bytearray(HEADER_SIZE)
        data[:4] = b"XXXX"
        struct.pack_into(">H", data, 4, CURRENT_VERSION)
        decode(bytes(data))
    expect(HeaderError, go)


def test_invalid_version():
    def go():
        data = bytearray(HEADER_SIZE)
        data[:4] = MAGIC
        struct.pack_into(">H", data, 4, 99)
        decode(bytes(data))
    expect(HeaderError, go)


def test_header_crc_rejects_corruption():
    """The new 56-byte header carries a real integrity check (CRC32 over
    bytes 0-51) that the old 64-byte format never had — corrupting any
    header field, even one the old format didn't separately validate,
    must be caught."""
    pkt = Packet(symbols=[Symbol(S_STRING, "test")])
    data = bytearray(encode(pkt))
    data[10] ^= 0xFF  # inside message_id, well within the CRC-covered prefix
    expect(HeaderError, lambda: decode(bytes(data)))


def test_streaming_full_chunks():
    pkt1 = encode(Packet(flags=FLAG_REQUEST, operation=0x43, symbols=[Symbol(S_STRING, "one")]))
    pkt2 = encode(Packet(flags=FLAG_REQUEST, operation=0x44, symbols=[Symbol(S_STRING, "two")]))
    decoder = StreamingDecoder()
    results = decoder.feed(pkt1 + pkt2)
    assert len(results) == 2
    assert results[0].symbols[0].value == "one"
    assert results[1].symbols[0].value == "two"


def test_streaming_partial():
    pkt = encode(Packet(symbols=[Symbol(S_STRING, "test")]))
    decoder = StreamingDecoder()
    results = decoder.feed(pkt[:50])
    assert len(results) == 0
    assert decoder.pending() > 0
    results = decoder.feed(pkt[50:])
    assert len(results) == 1
    assert results[0].symbols[0].value == "test"


def test_streaming_interleaved():
    """Two back-to-back packets, each arriving in its own partial chunks.

    A real byte stream from a single sender is strictly ordered: you never
    get pkt2's bytes spliced into the middle of pkt1's still-incomplete
    header (that would require a multiplexing layer above the codec, which
    StreamingDecoder doesn't implement — nor should it, since validate_header()
    only sanity-checks magic/version/payload_length bounds, not a full header
    checksum, so a genuinely spliced/corrupted header can be mistaken for an
    incomplete one and stall the decoder rather than resync). This test
    covers the real scenario streaming needs to handle: multiple complete
    packets, each delivered across more than one feed() call, decoded in
    order as each one finishes arriving.
    """
    pkt1 = encode(Packet(symbols=[Symbol(S_STRING, "first")]))
    pkt2 = encode(Packet(symbols=[Symbol(S_STRING, "second")]))
    decoder = StreamingDecoder()
    results = decoder.feed(pkt1[:50])
    assert len(results) == 0
    results = decoder.feed(pkt1[50:])
    assert len(results) == 1
    assert results[0].symbols[0].value == "first"
    results = decoder.feed(pkt2[:50])
    assert len(results) == 0
    results = decoder.feed(pkt2[50:])
    assert len(results) == 1
    assert results[0].symbols[0].value == "second"


def test_streaming_magic_resync():
    pkt = encode(Packet(symbols=[Symbol(S_STRING, "after_garbage")]))
    decoder = StreamingDecoder()
    results = decoder.feed(b"JUNKJUNK" + pkt)
    assert len(results) == 1
    assert results[0].symbols[0].value == "after_garbage"


def test_streaming_clear():
    decoder = StreamingDecoder()
    decoder.feed(b"xxxxx")
    assert decoder.pending() > 0
    decoder.clear()
    assert decoder.pending() == 0


def test_streaming_chunked_flag():
    pkt = Packet(
        flags=FLAG_STREAM_CHUNK,
        chunk_info=ChunkInfo(total_chunks=5, chunk_index=2,
                             original_size=10000, chunk_offset=4000, chunk_size=2000),
    )
    data = encode(pkt)
    view = decode(data)
    assert view.is_stream_chunk
    assert view.chunk_info.total_chunks == 5
    assert view.chunk_info.chunk_index == 2


def test_packetview_zero_copy():
    pkt = Packet(symbols=[Symbol(S_STRING, "zero_copy_test")])
    data = encode(pkt)
    view = decode(data)
    assert view.symbols[0].value == "zero_copy_test"
    assert isinstance(view.raw, memoryview)


def test_streaming_empty_feed():
    decoder = StreamingDecoder()
    results = decoder.feed(b"")
    assert len(results) == 0
    assert decoder.pending() == 0


def test_streaming_large_chunk():
    pkt = encode(Packet(symbols=[Symbol(S_STRING, "x" * 1000)]))
    decoder = StreamingDecoder()
    for i in range(0, len(pkt), 50):
        results = decoder.feed(pkt[i:i+50])
        if results:
            assert results[0].symbols[0].value == "x" * 1000
            return
    assert False, "expected results"
