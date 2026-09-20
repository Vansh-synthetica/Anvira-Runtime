"""pytest tests for aicl.bin codec - Part 1: roundtrip."""
from __future__ import annotations
import uuid

from aicl import encode, decode, Packet
from aicl.bin.types import Identity, Symbol, QoS, ErrorInfo, Backpressure
from aicl.bin.symbol_types import (
    S_STRING, S_F64, S_I64, S_BOOL,
    S_LIST, S_UUID, S_NULL, S_BYTES,
)
from aicl.bin.constants import (
    FLAG_REQUEST, FLAG_RESPONSE, FLAG_ERROR, HEADER_SIZE, CURRENT_VERSION,
)


def test_roundtrip_minimal():
    pkt = Packet()
    data = encode(pkt)
    # header + opcode + operand-count varint + an auto-generated session_id
    # extension (Packet always assigns one in __post_init__, so even a
    # "default" packet's payload is never just the bare 2-byte minimum).
    assert len(data) > HEADER_SIZE
    view = decode(data)
    assert view.version == CURRENT_VERSION
    assert view.session_id == pkt.session_id


def test_roundtrip_full():
    pkt = Packet(
        flags=FLAG_REQUEST,
        session_id=uuid.uuid4().bytes,
        message_id=uuid.uuid4().bytes,
        correlation_id=uuid.uuid4().bytes,
        origin=Identity(0x01, "orchestrator"),
        targets=["sentiment_classifier", "intent_parser"],
        operation=0x43,  # OP_CLASSIFY
        intent="Classify sentiment of user text",
        symbols=[
            Symbol(S_STRING, "I love this product!"),
            Symbol(S_F64, 0.95),
            Symbol(S_I64, 42),
            Symbol(S_BOOL, True),
            Symbol(S_STRING, "positive"),
        ],
        confidence=0.95, priority=64,
        deadline_ms=123456789,
        capabilities=["sentiment", "multilingual"],
        schema_id="sentiment-v1",
    )
    data = encode(pkt)
    assert len(data) > HEADER_SIZE
    view = decode(data)
    assert view.operation == 0x43
    assert view.targets == ["sentiment_classifier", "intent_parser"]
    assert view.confidence == 0.95
    assert view.origin.name == "orchestrator"
    assert view.deadline_ms == 123456789
    assert len(view.symbols) == 5
    assert view.symbols[0].value == "I love this product!"
    assert view.symbols[3].value is True
    assert view.symbols[4].value == "positive"


def test_roundtrip_lists():
    pkt = Packet(
        flags=FLAG_REQUEST,
        symbols=[Symbol(S_LIST, [
            Symbol(S_LIST, [Symbol(S_STRING, "a"), Symbol(S_STRING, "b")]),
            Symbol(S_STRING, "c"),
        ])]
    )
    data = encode(pkt)
    view = decode(data)
    outer = view.symbols[0]
    assert outer.tag == S_LIST
    assert outer.value[0].value[0].value == "a"


def test_roundtrip_bytes():
    blob_data = bytes(range(256))
    pkt = Packet(symbols=[Symbol(S_BYTES, blob_data)])
    data = encode(pkt)
    view = decode(data)
    assert view.symbols[0].value == blob_data


def test_roundtrip_uuid():
    uid = uuid.uuid4().bytes
    pkt = Packet(symbols=[Symbol(S_UUID, uid)])
    data = encode(pkt)
    view = decode(data)
    assert view.symbols[0].value == uid


def test_roundtrip_null():
    pkt = Packet(symbols=[Symbol(S_NULL, None)])
    data = encode(pkt)
    view = decode(data)
    assert view.symbols[0].tag == S_NULL
    assert view.symbols[0].value is None


def test_roundtrip_string_as_reference():
    # core-rust has no dedicated "reference" operand tag distinct from a
    # plain string — see symbol_types.py's note on the old S_REFERENCE tag.
    pkt = Packet(symbols=[Symbol(S_STRING, "nlp.sentiment.output.label")])
    data = encode(pkt)
    view = decode(data)
    assert view.symbols[0].value == "nlp.sentiment.output.label"


def test_roundtrip_string_as_json():
    # Same for the old S_JSON tag — JSON text is just a string on the wire.
    pkt = Packet(symbols=[Symbol(S_STRING, '{"key": "value"}')])
    data = encode(pkt)
    view = decode(data)
    assert view.symbols[0].value == '{"key": "value"}'


def test_response_packet():
    pkt = Packet(
        flags=FLAG_RESPONSE,
        correlation_id=uuid.uuid4().bytes,
        response_symbols=[Symbol(S_STRING, "positive"), Symbol(S_F64, 0.95)],
    )
    data = encode(pkt)
    view = decode(data)
    assert view.is_response
    assert len(view.response_symbols) == 2


def test_error_packet():
    pkt = Packet(
        flags=FLAG_RESPONSE | FLAG_ERROR,
        error_info=ErrorInfo(error_code=4, severity=1, message="Not found",
                              details="classifier unavailable"),
    )
    data = encode(pkt)
    view = decode(data)
    assert view.is_error
    assert view.error_info.message == "Not found"


def test_session_correlation_ids():
    s = uuid.uuid4().bytes
    c = uuid.uuid4().bytes
    pkt = Packet(session_id=s, correlation_id=c)
    data = encode(pkt)
    view = decode(data)
    assert view.session_id == s
    assert view.correlation_id == c


def test_qos():
    pkt = Packet(qos=QoS(1, 2, 1, 1, 3, 500))
    data = encode(pkt)
    view = decode(data)
    assert view.qos.max_retries == 3


def test_backpressure():
    pkt = Packet(backpressure=Backpressure(500, 1000, 2, 250))
    data = encode(pkt)
    view = decode(data)
    assert view.backpressure.queue_depth == 500


def test_metadata():
    pkt = Packet(metadata={"k1": "v1", "k2": "v2"})
    data = encode(pkt)
    view = decode(data)
    assert view.metadata == {"k1": "v1", "k2": "v2"}


def test_symbol_long_256():
    pkt = Packet(symbols=[Symbol(S_STRING, "x" * 256)])
    data = encode(pkt)
    view = decode(data)
    assert view.symbols[0].value == "x" * 256


def test_symbol_long_65535():
    pkt = Packet(symbols=[Symbol(S_STRING, "y" * 65535)])
    data = encode(pkt)
    view = decode(data)
    assert len(view.symbols[0].value) == 65535
