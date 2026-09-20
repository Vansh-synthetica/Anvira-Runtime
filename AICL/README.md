# AICL — Adaptive Inter-Module Communication Language

AICL is a compact binary wire protocol for AI-module-to-AI-module
communication: a typed packet format, a streaming decoder for partial
buffers, a semantic layer mapping model intents/actions onto the wire, and
a Python SDK that can talk either natively (via a compiled Rust core) or
over HTTP.

```python
from aicl import encode, decode, Packet
from aicl.bin.types import Symbol
from aicl.bin.symbol_types import S_STRING

pkt = Packet(symbols=[Symbol(S_STRING, "hello")])
wire = encode(pkt)
assert decode(wire).symbols[0].value == "hello"
```

## Install

```bash
pip install localhousellm-aicl
```

Zero third-party dependencies — pure standard library. For development:

```bash
git clone https://github.com/LocalHouseLLM/AICL
cd AICL
python -m venv .venv
.venv\Scripts\activate      # .venv/bin/activate on Linux/macOS
pip install -e ".[dev]"
pytest
```

## What's in the box

- **`aicl.bin`** — the binary codec: `encode`/`decode`, `Packet`,
  `PacketView`, `StreamingDecoder` for partial/chunked buffers, typed
  symbols, and a CRC32 trailer for payload integrity.
- **`aicl.semantic`** — `ModelIntent`/`ModelAction`/`ModelObservation`/
  `ModelRequest`/`ModelResult`, a `SemanticAdapter` bridging model output to
  the wire, `AICLGrammar` for intent validation, and a `GateCheck`/
  `GateResult` safety gate for inbound/outbound packets.
- **`aicl.sl`** — a human-readable debug formatter (`format_packet`,
  `format_symbol`, `format_tlv`, `format_view`) for inspecting packets on
  the wire — never emitted on the wire itself, display-only.
- **`aicl.sdk`** — the developer-facing runtime (`AICLRuntime`, `Model`,
  structured results, cancellation, diagnostics). Native transport
  delegates the hot path (codec, routing, shared-memory sync, scheduling)
  to a compiled Rust core; HTTP transport is available as an explicit
  opt-in fallback that needs nothing compiled.
- **`aicl.transport`** — connection/session/address plumbing shared by both
  transports.
- **`core-rust`** / **`core-cpp`** / **`native-cpp`** — native
  implementations. `core-rust` is the reference the ISA spec is written
  against.

## Native transport requires a compiled Rust core

`aicl.sdk`'s native (default) transport loads `aicl_core.dll`/`.so`/`.dylib`
via FFI (`aicl/sdk/_ffi.py`) — it looks in `core-rust/target/release/`,
`core-rust/target/debug/`, or `$AICL_CORE_LIB`. Build it with:

```bash
cd core-rust
cargo build --release
cargo test --release   # 40 tests
```

Without a compiled core, use the HTTP transport instead
(`AICLRuntime(http=True, http_url="...")`) or work directly against
`aicl.bin`/`aicl.semantic`, which are pure Python and need nothing compiled.

## Benchmarks

Two real OS processes, genuine shared-memory IPC, the same `aicl.bin`
codec this repo ships, compared against FastAPI over HTTPS doing the same
job:

| Scenario | AICL | FastAPI/HTTPS | Difference |
|---|---|---|---|
| Request/response (cross-process, no inference) | 26.3µs avg | 1,480µs avg | ~56× faster |
| Streaming, 51 chunks/response (cross-process, no inference) | 546µs/stream | 7,161µs/stream | ~13× faster |
| End-to-end, real inference, AICL over HTTP (40 real tokens, 60 rounds) | 1,173.5ms avg | 1,176.2ms avg | essentially tied |
| **End-to-end, real inference, AICL fully in-process (no network at all)** | **1,104.0ms avg (−89.4ms vs. no-relay)** | — | **further ~68ms faster than AICL-over-HTTP** |

The pure-transport rows isolate transport cost from model compute — real,
but they overstate what a user actually feels. The third row is what
matters with the network still involved at all: at 60 rounds, AICL and
HTTPS/SSE are both statistically indistinguishable from calling the model
directly — the transport difference that's dramatic in isolation
genuinely disappears against ~1 second of real inference. The **last row
answers a different question — can the network be removed entirely** —
by having the relay load the model directly (via `llama-cpp-python`, no
HTTP/TCP/`localhost` anywhere) instead of calling a separate server. That
one still measurably wins, consistently, on top of everything else.

(The pure-transport numbers above are slower than an earlier version of
this table — a later rewrite made `aicl.bin` match Rust's wire format
exactly for real cross-language interop, which came with genuine added
per-message cost: a real header CRC32 integrity check the old format
never had, among other things. Full honest breakdown in
[BENCHMARKS.md §7](BENCHMARKS.md#7-re-measured-after-the-wire-format-rewrite).)

Full methodology, every caveat, and the debugging story behind that last
number are in [BENCHMARKS.md](BENCHMARKS.md) — including two real,
fixable inefficiencies found along the way (DNS-resolving `"localhost"`
instead of using `127.0.0.1`, and reconstructing a fresh HTTP client per
call instead of reusing one) that turned out to cost every path in this
benchmark ~40% of its total time, unrelated to AICL vs. HTTPS at all.

## Wire format status — Python and Rust now speak the same protocol

**`aicl.bin` (Python) and `core-rust` implement the same wire format**,
matching the ISA spec's own layout: a 56-byte header (magic, version,
flags, message_id, correlation_id, deadline_ms, payload_length, a
header-integrity CRC32), the same opcode taxonomy, the same operand type
tags, and the same `opcode + varint(operand_count) + operands` payload
structure. This isn't a claim taken on faith — `core-rust/examples/
decode_python_packet.rs` and `core-rust/examples/encode_for_python.rs`
are real, runnable proof: a packet encoded by Python decodes correctly in
Rust, and vice versa, byte for byte, including nested `List` operands.

Python's rich object model (`origin`, `qos`, `trace`, `error_info`,
`metadata`, and everything else `Packet` exposes beyond the core
opcode+operands) has no equivalent in `core-rust`'s wire format — rather
than drop any of it, each field rides on the wire as an `Operand::Vendor`
extension (tags `0x80`-`0xFF`, part of the ISA spec's own extensibility
story). A Rust decoder sees a normal operand list where a few entries
carry an opaque vendor payload it can skip; Python recognizes its own
extension tags and reconstructs the rich fields. See
`aicl/bin/extension_tags.py` for the tag assignments and
[`WIRE_FORMAT_AUDIT.md`](WIRE_FORMAT_AUDIT.md) for the full history of
how the two implementations drifted before this rewrite (kept as a
historical record — the audit predates this fix).

## Testing

```bash
pytest tests/
```

92 Python tests, covering the binary codec, error handling
(truncated/corrupt packets, checksum mismatches, streaming with partial
buffers), the semantic layer, the SDK surface (capabilities, cancellation,
results, transports), and the debug formatter — pure Python, no native
build required. Plus 40 Rust tests (`cd core-rust && cargo test --release`)
covering the native codec, the lock-free ring buffer, the FFI boundary, and
capability dispatch.

## License

MPL-2.0 — see [LICENSE](LICENSE).
