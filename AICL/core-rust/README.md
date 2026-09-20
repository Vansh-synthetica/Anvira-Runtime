# aicl-core

**AICL Native Rust Core — Trusted Systems Implementation**

This crate is the canonical implementation of the [AICL Instruction Set
Architecture (AICL-ISA v1.0)](../../protocol/AICL-ISA.md). It is the **trusted
systems implementation** of the AICL communication language.

## Design constraints

- **No Python, HTTP, REST, `/v1`, JSON wire, or localhost dependency.**
- Pure Rust, `no_std` capable.
- Deterministic, bounded allocations.
- Zero-copy parsing where realistic.
- Panic-free: every public API returns `Result<_, AiclError>`.

## Crate layout

```
src/
├── lib.rs        — public re-exports
├── error.rs      — AiclError + ErrorCode
├── opcode.rs     — opcode taxonomy, flag bits, validation
├── operand.rs    — typed operand model
├── packet.rs     — AiclPacket (typed semantic layer)
├── codec.rs      — AiclBinaryCodec (encode / decode, zero-copy views)
├── instruction.rs— AiclInstruction (parsed payload)
├── capability.rs — AiclCapability + registry
├── gate.rs       — AiclGate (pre/post validation pipeline)
├── endpoint.rs   — AiclEndpoint (named participant)
├── session.rs    — AiclSession (correlated request/response)
├── channel.rs    — AiclChannel (in-process channel)
├── runtime.rs    — AiclRuntime (owns the world)
└── tests         — see ../tests/ for cross-crate conformance
```

## Wire format (v1.0)

32-byte fixed header + TLV payload + optional CRC32 trailer. See the ISA
document for the full specification.

## Quick start

```rust
use aicl_core::{AiclRuntime, AiclPacket, AiclOpcode, AiclFlags, AiclCapability};
use std::sync::Arc;

// Build a runtime with default settings.
let runtime = AiclRuntime::new();

// Register a capability.
runtime.registry().register(
    "sentiment",
    AiclCapability::new("sentiment.classify")
        .with_param("labels", "positive,negative")
).unwrap();

// Compose a packet.
let pkt = AiclPacket::call("sentiment.classify")
    .with_input("text", "I love this product")
    .with_input("labels", "positive,negative")
    .with_deadline_ms(500)
    .build();

// Encode to bytes.
let bytes = runtime.codec().encode(&pkt).unwrap();

// Decode back.
let view = runtime.codec().decode(&bytes).unwrap();
assert_eq!(view.packet().opcode, AiclOpcode::Call);
```

## License

MPL-2.0
