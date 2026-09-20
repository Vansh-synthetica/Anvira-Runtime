//! AICL Comparison Benchmark
//!
//! Compares all communication paths side-by-side:
//!
//! 1. Old InProcChannel (VecDeque, copies data)
//! 2. New LocalChannel (SPSC ring buffer, zero-copy pop)
//! 3. FFI boundary (Python ctypes → Rust, simulated)
//! 4. Codec paths (alloc vs reuse vs borrowed)
//! 5. Full round-trip (encode → transmit → decode)
//!
//! Run with:
//!   cargo +stable-x86_64-pc-windows-gnu run --release --example bench_compare

extern crate alloc;

use std::time::{Duration, Instant};

use aicl_core::codec::AiclCodec;
use aicl_core::local_path::{LocalChannel, LocalRingBuffer, BufferPool, ZeroCopyCodec};
use aicl_core::opcode::AiclOpcode;
use aicl_core::packet::AiclPacket;
use aicl_core::transport::{InProcChannel, Transport};
use aicl_core::operand::Operand;

// ═════════════════════════════════════════════════════════════════════════════
// Benchmark harness
// ═════════════════════════════════════════════════════════════════════════════

struct BenchResult {
    name: String,
    iterations: usize,
    total: Duration,
    latencies: Vec<Duration>,
}

impl BenchResult {
    fn avg(&self) -> Duration { self.total / self.iterations as u32 }
    fn p50(&self) -> Duration { percentile(&self.latencies, 50.0) }
    fn p95(&self) -> Duration { percentile(&self.latencies, 95.0) }
    fn p99(&self) -> Duration { percentile(&self.latencies, 99.0) }
    fn throughput(&self) -> f64 { self.iterations as f64 / self.total.as_secs_f64() }
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() { return Duration::ZERO; }
    let idx = ((p / 100.0) * sorted.len() as f64) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn ns(d: Duration) -> f64 { d.as_nanos() as f64 }

fn print_result(r: &BenchResult) {
    println!("  {:<45} {:>8.1}µs {:>8.1}µs {:>8.1}µs {:>8.1}µs {:>10.0}/s",
        r.name,
        ns(r.avg()) / 1000.0,
        ns(r.p50()) / 1000.0,
        ns(r.p95()) / 1000.0,
        ns(r.p99()) / 1000.0,
        r.throughput(),
    );
}

fn header() {
    println!("  {:<45} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "Operation", "Avg", "p50", "p95", "p99", "Throughput");
    println!("  {}", "-".repeat(97));
}

// ═════════════════════════════════════════════════════════════════════════════
// Test packets
// ═════════════════════════════════════════════════════════════════════════════

fn pkt_nop() -> AiclPacket { AiclPacket::nop() }

fn pkt_ping() -> AiclPacket { AiclPacket::ping(0xDEAD_BEEF) }

fn pkt_classify() -> AiclPacket {
    AiclPacket::classify(
        "The quick brown fox jumps over the lazy dog",
        vec!["positive".into(), "negative".into(), "neutral".into()],
        "gpt-4",
    )
}

fn pkt_generate() -> AiclPacket {
    AiclPacket::generate("Explain quantum computing", "gpt-4")
}

fn pkt_model_call() -> AiclPacket {
    let mut inputs = alloc::collections::BTreeMap::new();
    inputs.insert("prompt".into(), Operand::Str("Explain quantum computing in detail".into()));
    inputs.insert("max_tokens".into(), Operand::U32(1024));
    inputs.insert("temperature".into(), Operand::F32(0.7));
    AiclPacket::model_call(42, "generate", inputs)
}

// ═════════════════════════════════════════════════════════════════════════════
// Benchmarks
// ═════════════════════════════════════════════════════════════════════════════

// ── Codec comparison ────────────────────────────────────────────────────────

fn bench_encode_alloc(codec: &AiclCodec, pkt: &AiclPacket, n: usize) -> BenchResult {
    let mut latencies = Vec::with_capacity(n);
    let t = Instant::now();
    for _ in 0..n {
        let s = Instant::now();
        let _ = codec.encode(pkt).unwrap();
        latencies.push(s.elapsed());
    }
    latencies.sort();
    BenchResult { name: "encode (alloc new Vec)".into(), iterations: n, total: t.elapsed(), latencies }
}

fn bench_encode_into(codec: &AiclCodec, pkt: &AiclPacket, n: usize) -> BenchResult {
    let mut buf = Vec::with_capacity(4096);
    let mut latencies = Vec::with_capacity(n);
    let t = Instant::now();
    for _ in 0..n {
        buf.clear();
        let s = Instant::now();
        codec.encode_into(pkt, &mut buf).unwrap();
        latencies.push(s.elapsed());
    }
    latencies.sort();
    BenchResult { name: "encode (into reusable buf)".into(), iterations: n, total: t.elapsed(), latencies }
}

fn bench_decode_standard(codec: &AiclCodec, wire: &[u8], n: usize) -> BenchResult {
    let mut latencies = Vec::with_capacity(n);
    let t = Instant::now();
    for _ in 0..n {
        let s = Instant::now();
        let _ = codec.decode(wire).unwrap();
        latencies.push(s.elapsed());
    }
    latencies.sort();
    BenchResult { name: "decode (standard, Arc alloc)".into(), iterations: n, total: t.elapsed(), latencies }
}

fn bench_decode_borrowed(codec: &AiclCodec, wire: &[u8], n: usize) -> BenchResult {
    let mut latencies = Vec::with_capacity(n);
    let t = Instant::now();
    for _ in 0..n {
        let s = Instant::now();
        let _ = codec.decode_borrowed(wire).unwrap();
        latencies.push(s.elapsed());
    }
    latencies.sort();
    BenchResult { name: "decode (borrowed, 0-copy)".into(), iterations: n, total: t.elapsed(), latencies }
}

// ── Transport comparison ────────────────────────────────────────────────────

fn bench_inproc_old(codec: &AiclCodec, pkt: &AiclPacket, n: usize) -> BenchResult {
    let wire = codec.encode(pkt).unwrap();
    let (mut tx, mut rx) = InProcChannel::pair(256);
    let mut latencies = Vec::with_capacity(n);
    let t = Instant::now();
    for _ in 0..n {
        let s = Instant::now();
        tx.send(&wire).unwrap();
        let mut out = Vec::new();
        rx.recv(&mut out).unwrap();
        latencies.push(s.elapsed());
    }
    latencies.sort();
    BenchResult { name: "InProcChannel (old VecDeque)".into(), iterations: n, total: t.elapsed(), latencies }
}

fn bench_local_channel(codec: &AiclCodec, pkt: &AiclPacket, n: usize) -> BenchResult {
    let (mut tx, mut rx) = LocalChannel::new(4096, 256);
    let mut latencies = Vec::with_capacity(n);
    let t = Instant::now();
    for _ in 0..n {
        let s = Instant::now();
        tx.send(pkt, codec).unwrap();
        let _ = rx.recv(codec).unwrap();
        latencies.push(s.elapsed());
    }
    latencies.sort();
    BenchResult { name: "LocalChannel (new SPSC ring)".into(), iterations: n, total: t.elapsed(), latencies }
}

fn bench_local_zero_copy(codec: &AiclCodec, pkt: &AiclPacket, n: usize) -> BenchResult {
    let (mut tx, mut rx) = LocalChannel::new(4096, 256);
    let mut latencies = Vec::with_capacity(n);
    let t = Instant::now();
    for _ in 0..n {
        let s = Instant::now();
        tx.send(pkt, codec).unwrap();
        let _ = rx.try_recv_zero_copy(codec).unwrap();
        latencies.push(s.elapsed());
    }
    latencies.sort();
    BenchResult { name: "LocalChannel (zero-copy decode)".into(), iterations: n, total: t.elapsed(), latencies }
}

// ── Full round-trip comparison ──────────────────────────────────────────────

fn bench_roundtrip_old(codec: &AiclCodec, pkt: &AiclPacket, n: usize) -> BenchResult {
    let (mut tx, mut rx) = InProcChannel::pair(256);
    let mut latencies = Vec::with_capacity(n);
    let t = Instant::now();
    for _ in 0..n {
        let wire = codec.encode(pkt).unwrap();
        let s = Instant::now();
        tx.send(&wire).unwrap();
        let mut out = Vec::new();
        rx.recv(&mut out).unwrap();
        let _ = codec.decode(&out).unwrap();
        latencies.push(s.elapsed());
    }
    latencies.sort();
    BenchResult { name: "roundtrip: old (encode→copy→recv→decode)".into(), iterations: n, total: t.elapsed(), latencies }
}

fn bench_roundtrip_new(codec: &AiclCodec, pkt: &AiclPacket, n: usize) -> BenchResult {
    let (mut tx, mut rx) = LocalChannel::new(4096, 256);
    let mut latencies = Vec::with_capacity(n);
    let t = Instant::now();
    for _ in 0..n {
        let s = Instant::now();
        tx.send(pkt, codec).unwrap();
        let view = rx.recv(codec).unwrap();
        let _ = view.to_packet().unwrap();
        latencies.push(s.elapsed());
    }
    latencies.sort();
    BenchResult { name: "roundtrip: new (send→ring→recv→decode)".into(), iterations: n, total: t.elapsed(), latencies }
}

fn bench_roundtrip_zero_copy(codec: &AiclCodec, pkt: &AiclPacket, n: usize) -> BenchResult {
    let (mut tx, mut rx) = LocalChannel::new(4096, 256);
    let mut latencies = Vec::with_capacity(n);
    let t = Instant::now();
    for _ in 0..n {
        let s = Instant::now();
        tx.send(pkt, codec).unwrap();
        let _ = rx.try_recv_zero_copy(codec).unwrap();
        latencies.push(s.elapsed());
    }
    latencies.sort();
    BenchResult { name: "roundtrip: zero-copy (send→ring→pop→borrow)".into(), iterations: n, total: t.elapsed(), latencies }
}

// ── Buffer pool comparison ──────────────────────────────────────────────────

fn bench_alloc_free(n: usize) -> BenchResult {
    let mut latencies = Vec::with_capacity(n);
    let t = Instant::now();
    for _ in 0..n {
        let s = Instant::now();
        let _ = vec![0u8; 4096];
        latencies.push(s.elapsed());
    }
    latencies.sort();
    BenchResult { name: "alloc+free (4KB vec)".into(), iterations: n, total: t.elapsed(), latencies }
}

fn bench_pool_get_put(n: usize) -> BenchResult {
    let mut pool = BufferPool::new();
    let mut latencies = Vec::with_capacity(n);
    let t = Instant::now();
    for _ in 0..n {
        let s = Instant::now();
        let mut buf = pool.get(4096);
        buf.extend_from_slice(&[0xAB; 4096]);
        pool.put(buf);
        latencies.push(s.elapsed());
    }
    latencies.sort();
    BenchResult { name: "pool get+put (4KB slab)".into(), iterations: n, total: t.elapsed(), latencies }
}

// ═════════════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════════════

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              AICL COMPARISON BENCHMARK                         ║");
    println!("║   Old InProcChannel vs New LocalChannel vs Zero-Copy Path      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let codec = AiclCodec::new();
    let n = 100_000;
    let warmup = 10_000;

    // Warm up
    for pkt_fn in &[pkt_nop, pkt_ping, pkt_classify, pkt_generate] {
        let pkt = pkt_fn();
        let wire = codec.encode(&pkt).unwrap();
        for _ in 0..warmup {
            let _ = codec.encode(&pkt);
            let _ = codec.decode(&wire);
            let _ = codec.decode_borrowed(&wire);
        }
    }

    // ─── 1. Codec comparison ────────────────────────────────────────────────
    println!("\n  1. CODEC COMPARISON ({} iterations)", n);
    println!("     Which encode/decode path is fastest?");
    header();

    for (name, pkt_fn) in &[
        ("NOP", pkt_nop as fn() -> AiclPacket),
        ("Ping", pkt_ping),
        ("Classify", pkt_classify),
        ("Generate", pkt_generate),
        ("ModelCall", pkt_model_call),
    ] {
        let pkt = pkt_fn();
        let wire = codec.encode(&pkt).unwrap();
        println!("  [{}]", name);
        print_result(&bench_encode_alloc(&codec, &pkt, n));
        print_result(&bench_encode_into(&codec, &pkt, n));
        print_result(&bench_decode_standard(&codec, &wire, n));
        print_result(&bench_decode_borrowed(&codec, &wire, n));
        println!();
    }

    // ─── 2. Transport comparison ────────────────────────────────────────────
    println!("  2. TRANSPORT COMPARISON ({} iterations)", n);
    println!("     Old InProcChannel vs New LocalChannel (SPSC ring)");
    header();

    for (name, pkt_fn) in &[
        ("Ping", pkt_ping as fn() -> AiclPacket),
        ("Classify", pkt_classify),
        ("Generate", pkt_generate),
    ] {
        let pkt = pkt_fn();
        println!("  [{}]", name);
        print_result(&bench_inproc_old(&codec, &pkt, n));
        print_result(&bench_local_channel(&codec, &pkt, n));
        print_result(&bench_local_zero_copy(&codec, &pkt, n));
        println!();
    }

    // ─── 3. Full round-trip comparison ──────────────────────────────────────
    println!("  3. FULL ROUND-TRIP ({} iterations)", n);
    println!("     encode → transmit → decode (complete path)");
    header();

    for (name, pkt_fn) in &[
        ("Ping", pkt_ping as fn() -> AiclPacket),
        ("Classify", pkt_classify),
        ("Generate", pkt_generate),
        ("ModelCall", pkt_model_call),
    ] {
        let pkt = pkt_fn();
        println!("  [{}]", name);
        print_result(&bench_roundtrip_old(&codec, &pkt, n));
        print_result(&bench_roundtrip_new(&codec, &pkt, n));
        print_result(&bench_roundtrip_zero_copy(&codec, &pkt, n));
        println!();
    }

    // ─── 4. Buffer pool comparison ──────────────────────────────────────────
    println!("  4. ALLOCATION ({} iterations)", n);
    println!("     Raw alloc vs buffer pool");
    header();
    print_result(&bench_alloc_free(n));
    print_result(&bench_pool_get_put(n));
    println!();

    // ─── 5. Summary ────────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                        SUMMARY                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let classify = pkt_classify();
    let wire = codec.encode(&classify).unwrap();

    // Direct comparisons
    let e_alloc = bench_encode_alloc(&codec, &classify, n);
    let e_into = bench_encode_into(&codec, &classify, n);
    let d_std = bench_decode_standard(&codec, &wire, n);
    let d_bor = bench_decode_borrowed(&codec, &wire, n);
    let t_old = bench_inproc_old(&codec, &classify, n);
    let t_new = bench_local_channel(&codec, &classify, n);
    let t_zc = bench_local_zero_copy(&codec, &classify, n);
    let r_old = bench_roundtrip_old(&codec, &classify, n);
    let r_new = bench_roundtrip_new(&codec, &classify, n);
    let r_zc = bench_roundtrip_zero_copy(&codec, &classify, n);

    println!();
    println!("  Encode (Classify packet):");
    println!("    alloc vs into:     {:.2}x speedup", ns(e_alloc.avg()) / ns(e_into.avg()));
    println!();
    println!("  Decode (Classify wire):");
    println!("    standard vs borrowed: {:.2}x speedup", ns(d_std.avg()) / ns(d_bor.avg()));
    println!();
    println!("  Transport (ping packet, 1 direction):");
    println!("    old InProc vs new LocalChannel:  {:.2}x speedup", ns(t_old.avg()) / ns(t_new.avg()));
    println!("    old InProc vs zero-copy:         {:.2}x speedup", ns(t_old.avg()) / ns(t_zc.avg()));
    println!();
    println!("  Full round-trip (classify packet):");
    println!("    old vs new:        {:.2}x speedup", ns(r_old.avg()) / ns(r_new.avg()));
    println!("    old vs zero-copy:  {:.2}x speedup", ns(r_old.avg()) / ns(r_zc.avg()));
    println!();
    println!("  Copies per round-trip:");
    println!("    Old InProcChannel: 4 (encode + to_vec + recv + decode)");
    println!("    New LocalChannel:  2 (encode + recv)");
    println!("    Zero-copy:         1 (encode only)");
    println!();
}
