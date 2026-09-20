//! One-off cross-language interop check, reverse direction: encode a
//! packet with core-rust and print the hex so Python's aicl.bin can
//! decode it.

use aicl_core::codec::AiclCodec;
use aicl_core::packet::AiclPacket;

fn main() {
    let pkt = AiclPacket::classify(
        "Rust encoded this packet",
        vec!["positive".into(), "negative".into(), "neutral".into()],
        "gpt-4",
    );
    let codec = AiclCodec::new();
    let wire = codec.encode(&pkt).expect("encode failed");
    println!("{}", wire.iter().map(|b| format!("{:02x}", b)).collect::<String>());
}
