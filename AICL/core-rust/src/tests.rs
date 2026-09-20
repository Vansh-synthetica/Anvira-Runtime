//! End-to-end smoke tests for aicl-core.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use crate::codec::AiclCodec;
use crate::error::ErrorCode;
use crate::operand::Operand;
use crate::opcode::{AiclFlags, AiclOpcode, ISA_VERSION};
use crate::packet::AiclPacket;
use crate::router::{RoutingStrategy, Router};

#[test]
fn encode_decode_nop() {
    let codec = AiclCodec::new();
    let pkt = AiclPacket::new(AiclOpcode::Nop);
    let bytes = codec.encode(&pkt).expect("encode");
    let view = codec.decode(&bytes).expect("decode");
    assert_eq!(view.opcode(), AiclOpcode::Nop.to_u8());
}

#[test]
fn encode_decode_ping_pong() {
    let codec = AiclCodec::new();
    let pkt = AiclPacket::ping(0xDEAD_BEEF_CAFE_BABE);
    let bytes = codec.encode(&pkt).expect("encode");
    let view = codec.decode(&bytes).expect("decode");
    assert_eq!(view.opcode(), AiclOpcode::Ping.to_u8());
    let pong = AiclPacket::pong(0xDEAD_BEEF_CAFE_BABE);
    let bytes2 = codec.encode(&pong).expect("encode pong");
    let view2 = codec.decode(&bytes2).expect("decode pong");
    assert_eq!(view2.opcode(), AiclOpcode::Pong.to_u8());
}

#[test]
fn encode_decode_string_operand() {
    let codec = AiclCodec::new();
    let pkt = AiclPacket::new(AiclOpcode::Hello)
        .with(Operand::Str("hello, world".to_string()));
    let bytes = codec.encode(&pkt).expect("encode");
    let view = codec.decode(&bytes).expect("decode");
    let op = view.operand(0).expect("op0");
    if let Operand::Str(s) = op {
        assert_eq!(s, "hello, world");
    } else {
        panic!("expected Str operand");
    }
}

#[test]
fn encode_decode_map_operand() {
    let codec = AiclCodec::new();
    let mut m = BTreeMap::new();
    m.insert("name".to_string(), Operand::Str("AICL".to_string()));
    m.insert("version".to_string(), Operand::U32(1));
    let pkt = AiclPacket::new(AiclOpcode::Call)
        .with(Operand::Map(m));
    let bytes = codec.encode(&pkt).expect("encode");
    let view = codec.decode(&bytes).expect("decode");
    let op = view.operand(0).expect("op0");
    if let Operand::Map(map) = op {
        assert_eq!(map.len(), 2);
        assert!(map.contains_key("name"));
        assert!(map.contains_key("version"));
    } else {
        panic!("expected Map operand");
    }
}

#[test]
fn encode_decode_list_operands() {
    let codec = AiclCodec::new();
    let pkt = AiclPacket::new(AiclOpcode::Hello)
        .with(Operand::U32(1))
        .with(Operand::U32(2))
        .with(Operand::U32(3));
    let bytes = codec.encode(&pkt).expect("encode");
    let view = codec.decode(&bytes).expect("decode");
    assert_eq!(view.operand_count().expect("count"), 3);
    let op0 = view.operand(0).expect("op0");
    if let Operand::U32(v) = op0 { assert_eq!(v, 1); } else { panic!(); }
}

#[test]
fn bad_header_rejected() {
    let codec = AiclCodec::new();
    let bytes = [0u8; 8];
    let result = codec.decode(&bytes);
    assert!(result.is_err());
}

#[test]
fn bad_magic_rejected() {
    let codec = AiclCodec::new();
    let mut bytes = vec![0u8; 56];
    bytes[0..4].copy_from_slice(b"XXXX");
    let result = codec.decode(&bytes);
    assert!(result.is_err());
}

#[test]
fn opcode_round_trip() {
    for &op in &[
        AiclOpcode::Nop, AiclOpcode::Hello, AiclOpcode::Ping,
        AiclOpcode::Call, AiclOpcode::Return, AiclOpcode::Classify,
    ] {
        assert_eq!(AiclOpcode::from_u8(op.to_u8()), Some(op));
    }
}

#[test]
fn isa_version_constant() {
    assert_eq!(ISA_VERSION, 0x0100);
}

#[test]
fn router_round_robin() {
    let mut router = Router::round_robin();
    let h1: crate::router::Handler = alloc::boxed::Box::new(|_pkt| Ok(AiclPacket::new(AiclOpcode::Ack)));
    let h2: crate::router::Handler = alloc::boxed::Box::new(|_pkt| Ok(AiclPacket::new(AiclOpcode::Nak)));
    router.add_handler(h1);
    router.add_handler(h2);
    let p1 = AiclPacket::new(AiclOpcode::Ping);
    let p2 = AiclPacket::new(AiclOpcode::Ping);
    let r1 = router.dispatch(p1).expect("d1");
    let r2 = router.dispatch(p2).expect("d2");
    assert_eq!(r1.opcode, AiclOpcode::Ack);
    assert_eq!(r2.opcode, AiclOpcode::Nak);
}

#[test]
fn router_direct() {
    let mut router = Router::direct();
    let h1: crate::router::Handler = alloc::boxed::Box::new(|_pkt| Ok(AiclPacket::new(AiclOpcode::Ack)));
    router.add_capability_handler(7, h1);
    let p = AiclPacket::new(AiclOpcode::Call)
        .with(Operand::Ref(7));
    let r = router.dispatch(p).expect("dispatch");
    assert_eq!(r.opcode, AiclOpcode::Ack);
}

#[test]
fn error_code_round_trip() {
    for &code in &[
        ErrorCode::Ok, ErrorCode::BadHeader, ErrorCode::Truncated,
        ErrorCode::Checksum, ErrorCode::InvalidUtf8, ErrorCode::InvalidBool,
        ErrorCode::PayloadTooLarge, ErrorCode::UnsupportedVersion,
    ] {
        let v = code.code();
        assert_eq!(ErrorCode::from_u32(v), code);
    }
}

#[test]
fn flags_insert_contains() {
    let mut flags = AiclFlags::default();
    assert!(!flags.contains(AiclFlags::REQUEST));
    flags.insert(AiclFlags::REQUEST);
    assert!(flags.contains(AiclFlags::REQUEST));
    flags.remove(AiclFlags::REQUEST);
    assert!(!flags.contains(AiclFlags::REQUEST));
}

/// Regression test: corrupting a byte in correlation_id (bytes 24-39) must cause
/// decode to fail with a BAD_HEADER CRC error.
///
/// Before the 52-byte CRC-scope fix, bytes 30-51 were outside the CRC scope, so
/// corrupting correlation_id would silently pass. After the fix, the CRC covers
/// all bytes 0-51, so any corruption in the header prefix is detected.
#[test]
fn header_crc_scope_catches_correlation_id_corruption() {
    let codec = AiclCodec::new();
    let pkt = AiclPacket::ping(0xDEAD_BEEF_CAFE_BABE);
    let mut bytes = codec.encode(&pkt).expect("encode");

    // Byte 30 is inside correlation_id (offset 24..40 in the 56-byte header).
    // Flip one bit; the stored CRC was computed over bytes 0..51 (including this
    // byte), so recomputing it will produce a different value.
    bytes[30] ^= 0xFF;

    let result = codec.decode(&bytes);
    let err = result.expect_err("decode should fail after CRC scope fix");
    assert_eq!(err.code, ErrorCode::BadHeader,
        "expected BAD_HEADER (CRC mismatch), got {:?}", err.code);
    assert!(err.message.contains("CRC mismatch"),
        "error message should mention CRC mismatch: {}", err.message);
}

/// Regression test: an unassigned opcode byte (0x99) must cause decode to fail
/// with UNKNOWN_OPCODE per ISA §8 step 4, not silently decode as Nop.
///
/// Before the unknown-opcode fix, `AiclOpcode::from_u8` returned Nop for any
/// unrecognised byte. After the fix it returns None and the decoder propagates
/// ErrorCode::UnknownOpcode.
#[test]
fn unknown_opcode_rejected() {
    let codec = AiclCodec::new();
    let pkt = AiclPacket::ping(0x1234_5678_9ABC_DEF0);
    let mut bytes = codec.encode(&pkt).expect("encode");

    // Opcode is the first byte of the payload (header is 56 bytes).
    // Replace it with 0x99 (unassigned - not in any ISA range, not >= 0xF0).
    bytes[56] = 0x99;

    let result = codec.decode(&bytes);
    let err = result.expect_err("decode should fail for unknown opcode 0x99");
    assert_eq!(err.code, ErrorCode::UnknownOpcode,
        "expected UNKNOWN_OPCODE, got {:?}", err.code);
    assert!(err.message.contains("0x99"),
        "error message should mention the unknown byte: {}", err.message);
}
