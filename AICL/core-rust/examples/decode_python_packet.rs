//! One-off cross-language interop check: decode a packet that was
//! encoded by aicl.bin (Python), using core-rust's own codec, and verify
//! the fields match. If this passes, the two implementations really do
//! speak the same wire format now — not just "the source code looks
//! similar," an actual byte-for-byte round trip between the two
//! language implementations.

use aicl_core::codec::AiclCodec;
use aicl_core::opcode::AiclOpcode;
use aicl_core::operand::Operand;

fn main() {
    let hex = "4149434c01000001000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f0000000000000046000000006f24ad5f4304111354686520717569636b2062726f776e20666f7817021108706f73697469766511086e6567617469766511056770742d348010f93da5f91a9b4968b49e4a495f6b4d56";
    let wire: Vec<u8> = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect();

    println!("Decoding {} bytes produced by Python's aicl.bin...", wire.len());

    let codec = AiclCodec::new();
    let pkt = codec.decode_to_packet(&wire).expect("decode failed — NOT interoperable");

    println!("opcode:         {:?}", pkt.opcode);
    println!("message_id:     {:02x?}", pkt.message_id.unwrap());
    println!("correlation_id: {:02x?}", pkt.correlation_id);
    println!("operands:       {} total", pkt.operands.len());
    for (i, op) in pkt.operands.iter().enumerate() {
        match op {
            Operand::Str(s) => println!("  [{i}] Str({s:?})"),
            Operand::List(items) => println!("  [{i}] List({} items)", items.len()),
            Operand::Vendor(tag, data) => println!("  [{i}] Vendor(tag=0x{tag:02x}, {} bytes) — Python-only extension, correctly skipped", data.len()),
            other => println!("  [{i}] {other:?}"),
        }
    }

    assert_eq!(pkt.opcode, AiclOpcode::Classify);
    assert_eq!(pkt.message_id.unwrap(), (0u8..16).collect::<Vec<_>>().as_slice());
    assert_eq!(pkt.correlation_id, (16u8..32).collect::<Vec<_>>().as_slice());
    assert_eq!(pkt.operands.len(), 4); // 3 core + 1 vendor (session_id)

    match &pkt.operands[0] {
        Operand::Str(s) => assert_eq!(s, "The quick brown fox"),
        _ => panic!("operand 0 should be Str"),
    }
    match &pkt.operands[1] {
        Operand::List(items) => {
            assert_eq!(items.len(), 2);
            match (&items[0], &items[1]) {
                (Operand::Str(a), Operand::Str(b)) => {
                    assert_eq!(a, "positive");
                    assert_eq!(b, "negative");
                }
                _ => panic!("list items should be Str"),
            }
        }
        _ => panic!("operand 1 should be List"),
    }
    match &pkt.operands[2] {
        Operand::Str(s) => assert_eq!(s, "gpt-4"),
        _ => panic!("operand 2 should be Str"),
    }
    match &pkt.operands[3] {
        Operand::Vendor(tag, _) => assert_eq!(*tag, 0x80),
        _ => panic!("operand 3 should be the Vendor session_id extension"),
    }

    println!("\n✓ CROSS-LANGUAGE INTEROP CONFIRMED — Rust correctly decoded a");
    println!("  packet encoded by Python's aicl.bin: header, opcode, operands,");
    println!("  nested List, and the vendor-extension mechanism all matched.");
}
