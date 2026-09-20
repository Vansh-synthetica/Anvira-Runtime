//! CPU-optimized local path benchmarks for AICL.
//!
//! Measures:
//! - p50/p95/p99 latency (encode + transmit + decode)
//! - Throughput (messages/sec)
//! - Allocations per message
//! - Copies per message
//! - Synchronization overhead
//!
//! Run with: cargo +stable-x86_64-pc-windows-gnu run --release --example bench_local_path
//!
//! Note: For accurate benchmarks, build with `--release` to enable optimizations.

extern crate alloc;

use std::time::{Duration, Instant};

use aicl_core::codec::AiclCodec;
use aicl_core::local_path::{LocalChannel, LocalRingBuffer, BufferPool, DispatchTable, CACHE_LINE, ZeroCopyCodec};
use aicl_core::opcode::{AiclFlags, AiclOpcode};
use aicl_core::packet::AiclPacket;
use aicl_core::operand::Operand;

// ═════════════════════════════════════════════════════════════════════════════
// Benchmark harness
// ═════════════════════════════════════════════════════════════════════════════

struct BenchResult {
    name: String,
    iterations: usize,
    total_duration: Duration,
    latencies: Vec<Duration>,
    allocations: usize,
    copies: usize,
}

impl BenchResult {
    fn p50(&self) -> Duration { percentile(&self.latencies, 50.0) }
    fn p95(&self) -> Duration { percentile(&self.latencies, 95.0) }
    fn p99(&self) -> Duration { percentile(&self.latencies, 99.0) }
    fn throughput(&self) -> f64 {
        self.iterations as f64 / self.total_duration.as_secs_f64()
    }
    fn avg_latency(&self) -> Duration {
        let total: Duration = self.latencies.iter().sum();
        total / self.latencies.len() as u32
    }
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() { return Duration::ZERO; }
    let idx = ((p / 100.0) * sorted.len() as f64) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Format a Duration in whichever unit keeps at least 3 significant digits
/// — sub-microsecond operations (most of what's measured here) previously
/// got truncated to whole microseconds via `as_micros()`, which rounds
/// anything under 1000ns down to "0.0µs" for every percentile alike and
/// hides all real tail-latency variance. Report nanoseconds until the value
/// is large enough that microseconds are actually more readable.
fn fmt_duration(d: Duration) -> String {
    let ns = d.as_nanos();
    if ns < 10_000 {
        format!("{}ns", ns)
    } else {
        format!("{:.1}us", d.as_secs_f64() * 1_000_000.0)
    }
}

fn print_result(r: &BenchResult) {
    println!("  {:<40} {:>10} {:>10} {:>10} {:>10} {:>10.0}",
        r.name,
        fmt_duration(r.avg_latency()),
        fmt_duration(r.p50()),
        fmt_duration(r.p95()),
        fmt_duration(r.p99()),
        r.throughput(),
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Test packets of various sizes
// ═════════════════════════════════════════════════════════════════════════════

fn pkt_nop() -> AiclPacket {
    AiclPacket::nop()
}

fn pkt_ping() -> AiclPacket {
    AiclPacket::ping(0xDEAD_BEEF_CAFE_BABE)
}

fn pkt_classify() -> AiclPacket {
    AiclPacket::classify(
        "The quick brown fox jumps over the lazy dog. \
         This is a moderately sized text for classification testing.",
        vec!["positive".into(), "negative".into(), "neutral".into()],
        "gpt-4",
    )
}

fn pkt_model_call() -> AiclPacket {
    let mut inputs = alloc::collections::BTreeMap::new();
    inputs.insert("prompt".into(), Operand::Str("Explain quantum computing".into()));
    inputs.insert("max_tokens".into(), Operand::U32(1024));
    inputs.insert("temperature".into(), Operand::F32(0.7));
    AiclPacket::model_call(42, "generate", inputs)
}

fn pkt_memory_write() -> AiclPacket {
    AiclPacket::memory_write("session/context/current", Operand::Str(
        "User asked about AICL optimization. We implemented a lock-free SPSC ring buffer \
         with cache-line alignment, zero-copy codec, buffer pool, and direct dispatch table."
            .into()
    ))
}

// ═════════════════════════════════════════════════════════════════════════════
// Benchmarks
// ═════════════════════════════════════════════════════════════════════════════

fn bench_encode_only(codec: &AiclCodec, pkt: &AiclPacket, iterations: usize) -> BenchResult {
    let mut latencies = Vec::with_capacity(iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        let t0 = Instant::now();
        let _wire = codec.encode(pkt).unwrap();
        latencies.push(t0.elapsed());
    }
    let total = start.elapsed();
    latencies.sort();
    BenchResult {
        name: "encode (alloc)".into(),
        iterations, total_duration: total, latencies,
        allocations: iterations, // one Vec per encode
        copies: 0,
    }
}

fn bench_encode_reuse(codec: &AiclCodec, pkt: &AiclPacket, iterations: usize) -> BenchResult {
    let mut latencies = Vec::with_capacity(iterations);
    let mut buf = Vec::with_capacity(4096);
    let start = Instant::now();
    for _ in 0..iterations {
        let t0 = Instant::now();
        codec.encode_reuse(pkt, &mut buf).unwrap();
        latencies.push(t0.elapsed());
    }
    let total = start.elapsed();
    latencies.sort();
    BenchResult {
        name: "encode (reuse buf)".into(),
        iterations, total_duration: total, latencies,
        allocations: 0,
        copies: 0,
    }
}

fn bench_decode_borrowed(codec: &AiclCodec, wire: &[u8], iterations: usize) -> BenchResult {
    let mut latencies = Vec::with_capacity(iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        let t0 = Instant::now();
        let _view = codec.decode_borrowed(wire).unwrap();
        latencies.push(t0.elapsed());
    }
    let total = start.elapsed();
    latencies.sort();
    BenchResult {
        name: "decode (borrowed, 0-copy)".into(),
        iterations, total_duration: total, latencies,
        allocations: 0,
        copies: 0,
    }
}

fn bench_decode_standard(codec: &AiclCodec, wire: &[u8], iterations: usize) -> BenchResult {
    let mut latencies = Vec::with_capacity(iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        let t0 = Instant::now();
        let _view = codec.decode(wire).unwrap();
        latencies.push(t0.elapsed());
    }
    let total = start.elapsed();
    latencies.sort();
    BenchResult {
        name: "decode (standard, Arc copy)".into(),
        iterations, total_duration: total, latencies,
        allocations: iterations, // Arc<Vec<u8>> per decode
        copies: iterations,      // data.to_vec()
    }
}

fn bench_ring_push_pop(capacity: u64, slot_size: u64, iterations: usize) -> BenchResult {
    let mut ring = LocalRingBuffer::new(capacity, slot_size);
    let data = vec![0xABu8; 256]; // 256-byte message
    let mut latencies = Vec::with_capacity(iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        let t0 = Instant::now();
        ring.push(&data).unwrap();
        let _borrow = ring.try_pop().unwrap();
        latencies.push(t0.elapsed());
    }
    let total = start.elapsed();
    latencies.sort();
    BenchResult {
        name: alloc::format!("ring push+pop (cap={}, slot={})", capacity, slot_size),
        iterations, total_duration: total, latencies,
        allocations: 0,
        copies: 2, // push copies data in, pop borrows (0 copies for borrow)
    }
}

fn bench_channel_send_recv(iterations: usize) -> BenchResult {
    let codec = AiclCodec::new();
    let (mut tx, mut rx) = LocalChannel::new(4096, 256);
    let pkt = pkt_ping();
    let mut latencies = Vec::with_capacity(iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        let t0 = Instant::now();
        tx.send(&pkt, &codec).unwrap();
        let _view = rx.recv(&codec).unwrap();
        latencies.push(t0.elapsed());
    }
    let total = start.elapsed();
    latencies.sort();
    BenchResult {
        name: "channel send+recv (ping)".into(),
        iterations, total_duration: total, latencies,
        allocations: iterations, // decode scratch copy
        copies: iterations,      // encode + decode
    }
}

fn bench_channel_zero_copy(iterations: usize) -> BenchResult {
    let codec = AiclCodec::new();
    let (mut tx, mut rx) = LocalChannel::new(4096, 256);
    let pkt = pkt_ping();
    let mut latencies = Vec::with_capacity(iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        let t0 = Instant::now();
        tx.send(&pkt, &codec).unwrap();
        let _view = rx.try_recv_zero_copy(&codec).unwrap();
        latencies.push(t0.elapsed());
    }
    let total = start.elapsed();
    latencies.sort();
    BenchResult {
        name: "channel send+recv (zero-copy)".into(),
        iterations, total_duration: total, latencies,
        allocations: 0,
        copies: 1, // encode copies into ring slot
    }
}

fn bench_dispatch_table(iterations: usize) -> BenchResult {
    let mut table = DispatchTable::new();
    fn nop_handler(_wire: &[u8]) -> aicl_core::AiclResult<Vec<u8>> {
        Ok(Vec::new())
    }
    table.register(AiclOpcode::Nop, nop_handler);

    let codec = AiclCodec::new();
    let pkt = pkt_nop();
    let wire = codec.encode(&pkt).unwrap();

    let mut latencies = Vec::with_capacity(iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        let t0 = Instant::now();
        let _result = table.dispatch(&wire).unwrap();
        latencies.push(t0.elapsed());
    }
    let total = start.elapsed();
    latencies.sort();
    BenchResult {
        name: "dispatch table (O(1) lookup)".into(),
        iterations, total_duration: total, latencies,
        allocations: 0,
        copies: 0,
    }
}

fn bench_buffer_pool(iterations: usize) -> BenchResult {
    let mut pool = BufferPool::new();
    let mut latencies = Vec::with_capacity(iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        let t0 = Instant::now();
        let mut buf = pool.get(4096);
        buf.extend_from_slice(&[0xAB; 4096]);
        pool.put(buf);
        latencies.push(t0.elapsed());
    }
    let total = start.elapsed();
    latencies.sort();
    BenchResult {
        name: "buffer pool get+put".into(),
        iterations, total_duration: total, latencies,
        allocations: 1, // first get allocates
        copies: 0,
    }
}

fn bench_inproc_channel_original(iterations: usize) -> BenchResult {
    use aicl_core::transport::{InProcChannel, Transport};
    let codec = AiclCodec::new();
    let pkt = pkt_ping();
    let wire = codec.encode(&pkt).unwrap();
    let (mut tx, mut rx) = InProcChannel::pair(256);

    let mut latencies = Vec::with_capacity(iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        let t0 = Instant::now();
        tx.send(&wire).unwrap();
        let mut out = Vec::new();
        rx.recv(&mut out).unwrap();
        latencies.push(t0.elapsed());
    }
    let total = start.elapsed();
    latencies.sort();
    BenchResult {
        name: "InProcChannel (original)".into(),
        iterations, total_duration: total, latencies,
        allocations: iterations * 2, // send to_vec + recv extend
        copies: iterations * 2,
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════════════

fn main() {
    println!("AICL Local Path CPU Benchmarks");
    println!("==============================");
    println!();

    let codec = AiclCodec::new();
    let iterations = 100_000;
    let warmup = 10_000;

    // Warm up
    for pkt_fn in &[pkt_nop, pkt_ping, pkt_classify, pkt_model_call] {
        let pkt = pkt_fn();
        let wire = codec.encode(&pkt).unwrap();
        for _ in 0..warmup {
            let _ = codec.encode(&pkt);
            let _ = codec.decode_borrowed(&wire);
            let _ = codec.decode(&wire);
        }
    }

    // ─── Codec benchmarks ───────────────────────────────────────────────────
    println!("Codec Performance ({} iterations):", iterations);
    println!("  {:<40} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Operation", "Avg", "p50", "p95", "p99", "Throughput");
    println!("  {}", "-".repeat(92));

    for (name, pkt_fn) in &[
        ("NOP (minimal)", pkt_nop as fn() -> AiclPacket),
        ("Ping (1 operand)", pkt_ping),
        ("Classify (3 operands)", pkt_classify),
        ("ModelCall (map)", pkt_model_call),
        ("MemoryWrite (string)", pkt_memory_write),
    ] {
        let pkt = pkt_fn();
        let wire = codec.encode(&pkt).unwrap();

        let r = bench_encode_only(&codec, &pkt, iterations);
        print_result(&r);

        let r = bench_encode_reuse(&codec, &pkt, iterations);
        print_result(&r);

        let r = bench_decode_borrowed(&codec, &wire, iterations);
        print_result(&r);

        let r = bench_decode_standard(&codec, &wire, iterations);
        print_result(&r);

        println!();
    }

    // ─── Ring buffer benchmarks ─────────────────────────────────────────────
    println!("Ring Buffer Performance ({} iterations):", iterations);
    println!("  {:<40} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Configuration", "Avg", "p50", "p95", "p99", "Throughput");
    println!("  {}", "-".repeat(92));

    for (cap, slot) in &[
        // slot_size must leave room for SlotHeader::SIZE on top of the
        // 256-byte test message below — 256 alone leaves no headroom.
        (16, 512),
        (64, 1024),
        (256, 4096),
        (1024, 4096),
    ] {
        let r = bench_ring_push_pop(*cap, *slot, iterations);
        print_result(&r);
    }

    println!();

    // ─── Channel benchmarks ─────────────────────────────────────────────────
    println!("Channel Performance ({} iterations):", iterations);
    println!("  {:<40} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Channel", "Avg", "p50", "p95", "p99", "Throughput");
    println!("  {}", "-".repeat(92));

    let r = bench_inproc_channel_original(iterations);
    print_result(&r);

    let r = bench_channel_send_recv(iterations);
    print_result(&r);

    let r = bench_channel_zero_copy(iterations);
    print_result(&r);

    println!();

    // ─── Dispatch & pool benchmarks ─────────────────────────────────────────
    println!("Dispatch & Pool ({} iterations):", iterations);
    println!("  {:<40} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Operation", "Avg", "p50", "p95", "p99", "Throughput");
    println!("  {}", "-".repeat(92));

    let r = bench_dispatch_table(iterations);
    print_result(&r);

    let r = bench_buffer_pool(iterations);
    print_result(&r);

    println!();

    // ─── Summary ────────────────────────────────────────────────────────────
    println!("Cache line size: {} bytes", CACHE_LINE);
    println!("Ring header size: {} bytes ({} cache lines)",
        core::mem::size_of::<aicl_core::local_path::RingHeader>(),
        core::mem::size_of::<aicl_core::local_path::RingHeader>() / CACHE_LINE);

    // Key comparisons
    println!();
    println!("=== Key Comparisons ===");

    let pkt = pkt_ping();
    let wire = codec.encode(&pkt).unwrap();

    // Encode: alloc vs reuse
    let r_alloc = bench_encode_only(&codec, &pkt, iterations);
    let r_reuse = bench_encode_reuse(&codec, &pkt, iterations);
    println!("Encode: reuse buf is {:.1}x faster than alloc",
        r_alloc.avg_latency().as_nanos() as f64 / r_reuse.avg_latency().as_nanos() as f64);

    // Decode: standard vs borrowed
    let r_std = bench_decode_standard(&codec, &wire, iterations);
    let r_bor = bench_decode_borrowed(&codec, &wire, iterations);
    println!("Decode: borrowed (0-copy) is {:.1}x faster than standard",
        r_std.avg_latency().as_nanos() as f64 / r_bor.avg_latency().as_nanos() as f64);

    // Channel: original vs local
    let r_orig = bench_inproc_channel_original(iterations);
    let r_local = bench_channel_zero_copy(iterations);
    println!("Channel: local (zero-copy) is {:.1}x faster than InProcChannel",
        r_orig.avg_latency().as_nanos() as f64 / r_local.avg_latency().as_nanos() as f64);
}
