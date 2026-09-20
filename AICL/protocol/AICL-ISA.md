# AICL-ISA — Instruction Set Architecture

**Status:** Normative specification, v1.0
**Audience:** Implementers of AICL runtimes in Rust, C++, or any language binding.
**Non-goal:** AICL-ISA is not a transport protocol. It defines the meaning of bytes inside a single AICL message, not how those bytes travel.

---

## 1. Design Philosophy

AICL-ISA is a compact, deterministic, language-independent instruction format:

1. **No natural language on the wire.** Every byte has structural meaning.
2. **Model-facing and machine-facing layers are separate.** LLMs emit typed semantic instructions; the runtime encodes them as binary.
3. **Byte-identical codec implementations** in Rust, C++, and Python — verified by conformance tests.
4. **LLMs do not generate raw binary.** They emit typed instructions; the runtime encodes them.
5. **Parsing is deterministic and bounded.** All allocations are pre-declared. A malformed packet is rejected with an error code, never panics.

```
   Semantic layer (typed, model-facing)
   "Classify text 'I love it' with labels pos,neg"
                 ↓ encode
   ISA layer (opcode + typed operands)
                 ↓ serialize
   Wire layer (56B header + TLV payload + optional trailer)
```

---

## 2. Instruction Taxonomy

Five categories, each with a reserved opcode range:

| Range | Category | Purpose |
|---|---|---|
| 0x00–0x0F | System / lifecycle | Hello, Discover, Ping, Ack, Error, Cancel, Goodbye |
| 0x10–0x1F | Call / return | Call, Return, Stream, StreamEnd |
| 0x20–0x2F | Memory / state | MemoryRead, MemoryWrite, MemoryDelete, IndexQuery, IndexUpsert |
| 0x30–0x3F | Capability | CapabilityAdvertise, CapabilityWithdraw, CapabilityQuery |
| 0x40–0x4F | AI primitives | ModelCall, ToolCall, Embed, Classify, Reason, Retrieve, Execute |
| 0x50–0x5F | Tracing / diagnostics | Trace, Stats, Health |
| 0x60–0x6F | Buffer / reference | BufferRef, BufferRelease, BufferShare |
| 0x70–0xEF | Reserved | Future use |
| 0xF0–0xFF | Vendor / extension | Opaque to core |

### 2.1 System (0x00–0x0F)

| Opcode | Mnemonic | Purpose |
|---|---|---|
| 0x00 | NOP | No-op; keep-alive alignment |
| 0x01 | HELLO | Endpoint announces itself (id, capabilities, version) |
| 0x02 | GOODBYE | Graceful shutdown announcement |
| 0x03 | DISCOVER | Query endpoints by capability |
| 0x04 | PING | Liveness probe; answered by PONG |
| 0x05 | PONG | Reply to PING |
| 0x06 | ACK | Acknowledge receipt |
| 0x07 | NAK | Negative acknowledgement |
| 0x08 | ERROR | Terminal error with code and message |
| 0x09 | CANCEL | Cancel a pending call by correlation_id |
| 0x0A | VERSION | Negotiate protocol version |
| 0x0B–0x0F | reserved | — |

### 2.2 Call / Return (0x10–0x1F)

| Opcode | Mnemonic | Purpose |
|---|---|---|
| 0x10 | CALL | Invoke a capability: target, inputs, deadline, flags |
| 0x11 | RETURN | Successful response with correlation_id, outputs |
| 0x12 | STREAM | Streaming chunk: correlation_id, chunk, stream_seq |
| 0x13 | STREAM_END | Last chunk; total_chunks, optional summary |
| 0x14–0x1F | reserved | — |

### 2.3 Memory / State (0x20–0x2F)

| Opcode | Mnemonic | Purpose |
|---|---|---|
| 0x20 | MEMORY_READ | Read a key from a memory region |
| 0x21 | MEMORY_WRITE | Write key/value to a memory region |
| 0x22 | MEMORY_DELETE | Delete a key |
| 0x23 | INDEX_QUERY | Query an index |
| 0x24 | INDEX_UPSERT | Insert or update an index entry |
| 0x25–0x2F | reserved | — |

### 2.4 Capability (0x30–0x3F)

| Opcode | Mnemonic | Purpose |
|---|---|---|
| 0x30 | CAPABILITY_ADVERTISE | Announce one or more capabilities |
| 0x31 | CAPABILITY_WITHDRAW | Revoke a previously advertised capability |
| 0x32 | CAPABILITY_QUERY | Ask: who can do X? |
| 0x33–0x3F | reserved | — |

### 2.5 AI Primitives (0x40–0x4F)

| Opcode | Mnemonic | Purpose |
|---|---|---|
| 0x40 | MODEL_CALL | Invoke a model with named inputs and params |
| 0x41 | TOOL_CALL | Invoke a tool by name with JSON parameters |
| 0x42 | EMBED | Produce vector embedding |
| 0x43 | CLASSIFY | Classify input into named labels |
| 0x44 | REASON | Perform a reasoning step / chain |
| 0x45 | RETRIEVE | Retrieve documents by query |
| 0x46 | EXECUTE | Execute arbitrary code (sandboxed) |
| 0x47–0x4F | reserved | — |

### 2.6 Tracing / Diagnostics (0x50–0x5F)

| Opcode | Mnemonic | Purpose |
|---|---|---|
| 0x50 | TRACE | Append a trace entry to provenance |
| 0x51 | STATS | Request or return counters |
| 0x52 | HEALTH | Health summary |
| 0x53–0x5F | reserved | — |

### 2.7 Buffer / Reference (0x60–0x6F)

| Opcode | Mnemonic | Purpose |
|---|---|---|
| 0x60 | BUFFER_REF | Reference to a shared-memory buffer (zero-copy) |
| 0x61 | BUFFER_RELEASE | Release a buffer handle |
| 0x62 | BUFFER_SHARE | Grant another endpoint access to a buffer |
| 0x63–0x6F | reserved | — |

---

## 3. Operand Type System

Closed type system; tag set is fixed.

| Tag | Type | Width | Notes |
|---|---|---|---|
| 0x01 | I8 | 1B | Signed |
| 0x02 | I16 | 2B BE | Signed |
| 0x03 | I32 | 4B BE | Signed |
| 0x04 | I64 | 8B BE | Signed |
| 0x05 | U8 | 1B | Unsigned |
| 0x06 | U16 | 2B BE | Unsigned |
| 0x07 | U32 | 4B BE | Unsigned |
| 0x08 | U64 | 8B BE | Unsigned |
| 0x09 | F32 | 4B BE | IEEE-754 binary32 |
| 0x0A | F64 | 8B BE | IEEE-754 binary64 |
| 0x10 | BOOL | 1B | 0x00=false, 0x01=true. Others invalid. |
| 0x11 | STR | varint+UTF-8 | Bounded by MAX_STR (default 64 KiB) |

---

## 5. Opcode Registry

Formal operand schemas (see §3 for type definitions):

| Opcode | Mnemonic | Operand schema |
|---|---|---|
| 0x00 | NOP | — |
| 0x01 | HELLO | [STR name, U32 version, LIST capabilities, MAP attrs] |
| 0x02 | GOODBYE | [STR reason] |
| 0x03 | DISCOVER | [STR capability_filter, U32 limit] |
| 0x04 | PING | [U64 nonce] |
| 0x05 | PONG | [U64 nonce] |
| 0x06 | ACK | [UUID correlation_id] |
| 0x07 | NAK | [UUID correlation_id, U32 error_code, STR reason] |
| 0x08 | ERROR | [U32 error_code, STR message, MAP details] |
| 0x09 | CANCEL | [UUID correlation_id, STR reason] |
| 0x0A | VERSION | [U32 min_version, U32 max_version] |
| 0x10 | CALL | [REF target_capability, MAP inputs, U32 deadline_ms, U16 flags] |
| 0x11 | RETURN | [UUID correlation_id, MAP outputs] |
| 0x12 | STREAM | [UUID correlation_id, U32 stream_seq, BYTES chunk, BOOL is_partial] |
| 0x13 | STREAM_END | [UUID correlation_id, U32 total_chunks, MAP summary] |
| 0x20 | MEMORY_READ | [STR key, REF region] |
| 0x21 | MEMORY_WRITE | [STR key, BYTES value, REF region, DURATION_MS ttl] |
| 0x22 | MEMORY_DELETE | [STR key, REF region] |
| 0x23 | INDEX_QUERY | [STR query, U32 top_k, REF index] |
| 0x24 | INDEX_UPSERT | [STR key, BYTES vector, REF index] |
| 0x30 | CAPABILITY_ADVERTISE | [LIST capabilities] |
| 0x31 | CAPABILITY_WITHDRAW | [STR capability_id] |
| 0x32 | CAPABILITY_QUERY | [STR capability_filter] |
| 0x40 | MODEL_CALL | [STR model, MAP params, MAP inputs, U32 deadline_ms] |
| 0x41 | TOOL_CALL | [STR tool, MAP params, U32 deadline_ms] |
| 0x42 | EMBED | [STR model, LIST texts] |
| 0x43 | CLASSIFY | [STR text, LIST labels, STR model] |
| 0x44 | REASON | [STR prompt, LIST premises, STR strategy, U32 depth] |
| 0x45 | RETRIEVE | [STR query, U32 top_k, REF index] |
| 0x46 | EXECUTE | [REF capability, MAP inputs, U32 deadline_ms] |
| 0x50 | TRACE | [STR actor, STR action, TS_MS timestamp, DURATION_MS duration] |
| 0x51 | STATS | [MAP counters] |
| 0x52 | HEALTH | [U8 status, MAP metrics] |
| 0x60 | BUFFER_REF | [BUF_REF descriptor, STR tag, U8 access] |
| 0x61 | BUFFER_RELEASE | [BUF_REF descriptor] |
| 0x62 | BUFFER_SHARE | [BUF_REF descriptor, REF peer, U8 access] |

---

## 6. Flags (16-bit, OR-combined)

| Bit | Name | Meaning |
|---|---|---|
| 0x0001 | REQUEST | Packet is a request (expects response) |
| 0x0002 | RESPONSE | Packet is a response |
| 0x0004 | STREAM_CHUNK | Payload is one chunk of a stream |
| 0x0008 | STREAM_END | Last chunk of a stream |
| 0x0010 | ERROR | Packet is an error |
| 0x0020 | CANCEL | Cancellation notice |
| 0x0040 | HEARTBEAT | Liveness ping |
| 0x0080 | COMPRESSED | Payload is compressed (zstd) |
| 0x0100 | ENCRYPTED | Payload is encrypted (AEAD) |
| 0x0200 | TRACED | Packet has a TRACE operand |
| 0x0400 | PRIORITY_HIGH | Hint to scheduler |
| 0x0800 | PRIORITY_LOW | Hint to scheduler |
| 0x1000 | EXTENDED | Followed by extension TLVs |
| 0x2000 | BUFFER_REF | Contains a BUF_REF operand |
| 0x4000 | FROZEN | Packet is immutable; receivers must not mutate |
| 0x8000 | reserved | — |

---

## 7. Wire Format

Every AICL message:

```
┌──────────────────────────────────────┐
│  HEADER   (56 bytes, fixed)          │
│    offset  size  field                │
│    0       4B    magic          = "AICL"│
│    4       2B    version (u16 BE)      │
│    6       2B    flags (u16 BE)        │
│    8       16B   message_id (UUID raw) │
│    24      16B   correlation_id (UUID raw)│
│    40      4B    deadline_ms (u32 BE; 0=none)│
│    44      4B    payload_length (u32 BE)│
│    48      4B    padding (0; aligns CRC)│
│    52      4B    header_crc32 (u32 BE) │
├──────────────────────────────────────┤
│  PAYLOAD  (variable, max 16 MiB)      │
│    opcode         u8                  │
│    operand_count varint               │
│    operand_0      tag + typed value  │
│    operand_1      tag + typed value  │
│    ...                               │
│    extension_tlvs (if EXTENDED flag)  │
├──────────────────────────────────────┤
│  TRAILER  (optional)                  │
│    payload_crc32 u32 BE               │
└──────────────────────────────────────┘
```

**Header is 56 bytes total.** Field-width arithmetic: 4 (magic) + 2 (version) + 2 (flags) + 16 (message_id) + 16 (correlation_id) + 4 (deadline_ms) + 4 (payload_length) + 4 (padding) + 4 (header_crc32) = 56. The 4-byte padding word at offset 48–51 is reserved (MUST be zero) and aligns `header_crc32` to a 4-byte boundary. `header_crc32` covers **bytes 0..52** (the full 52-byte prefix before the CRC field itself) using CRC-32 (IEEE 802.3 polynomial, `crc32fast`). `payload_crc32` covers the entire header + payload (i.e. the first 56 + payload_length bytes of the message).


---

## 9. Execution Semantics

In-flight CALL operations keyed by correlation_id:

```
CALL ──► pending ──► done
            │
            │ timeout / cancel
            ▼
         cancelled
```

- CALL with REQUEST flag expects RETURN (single) or STREAM+STREAM_END (streamed) or ERROR.
- Correlation: receiver uses correlation_id == sender.message_id.
- If deadline_ms > 0 and deadline passes, runtime MAY emit CANCEL to executor then ERROR to caller with TIMEOUT.
- STREAM chunks share correlation_id; reassembled in stream_seq order. Out-of-order buffered up to bound; beyond: ERROR_STREAM_OUT_OF_ORDER.

---

## 10. Error Codes

u32. Core range 0x0000–0x00FF. Vendor 0x0100–0xFFFF.

| Code | Symbol | Meaning |
|---|---|---|
| 0x0000 | OK | No error |
| 0x0001 | BAD_HEADER | Magic/version/CRC failure |
| 0x0002 | TRUNCATED | Declared length > actual |
| 0x0003 | CHECKSUM | Trailer CRC mismatch |
| 0x0004 | UNKNOWN_OPCODE | Opcode not in supported set |
| 0x0005 | UNKNOWN_TYPE | Operand tag not in supported set |
| 0x0006 | OPERAND_COUNT | Count mismatch |
| 0x0007 | SCHEMA_MISMATCH | Types don't match schema |
| 0x0008 | INVALID_UTF8 | STR not valid UTF-8 |
| 0x0009 | INVALID_BOOL | BOOL not 0/1 |
| 0x000A | PAYLOAD_TOO_LARGE | Field exceeds limit |
| 0x000B | VARINT_OVERFLOW | varint > 4 bytes |
| 0x000C | MAP_KEY_TYPE | Non-STR key in MAP |
| 0x000D | INVALID_VALUE | Semantic value check failed |
| 0x000E | UNSUPPORTED_VERSION | Version mismatch |
| 0x000F | CAPABILITY_NOT_FOUND | REF does not resolve |
| 0x0010 | TIMEOUT | Deadline exceeded |
| 0x0011 | CANCELLED | Operation cancelled |
| 0x0012 | BACKPRESSURE | Receiver asks sender to slow |
| 0x0013 | BUFFER_NOT_FOUND | BUF_REF does not resolve |
| 0x0014 | BUFFER_ACCESS_DENIED | Required access not granted |
| 0x0015 | STREAM_OUT_OF_ORDER | stream_seq violation |
| 0x0016 | EXTENSION_NOT_UNDERSTOOD | Cannot pass extension |
| 0x0017–0x00FF | reserved | Future core errors |
| 0x0100–0xFFFF | vendor | Vendor-defined |

---

## 11. Versioning

- Header version: u16 = (major << 8) | minor. v1.0 = 0x0100.
- Runtime MUST accept any packet with same major (forward-compatible).
- Runtime MUST reject different major. Error: UNSUPPORTED_VERSION.
- VERSION opcode (0x0A) on connect for negotiation.
- Adding opcode within major: only in reserved ranges, once assigned, permanent.
- Adding operand tag: only in 0x1C–0x7F, permanent.
- Incompatible changes: new major.

---

## 12. Extension Mechanism

EXTENDED flag → payload ends with extension TLVs:
- type: u8 (0x00–0x7F core-reserved, 0x80–0xFF vendor)
- length: varint
- value: length bytes

Core that doesn't understand an extension MUST NOT fail. Pass through (when relaying) or drop silently (when consuming). Emit EXTENSION_NOT_UNDERSTOOD only if asked explicitly. Extensions not covered by ISA schema — they are opaque TLVs.

---

## 13. Reserved Ranges

| Item | Reserved range | Notes |
|---|---|---|
| Opcodes | 0x70–0xEF | Future categories |
| Opcodes | 0xF0–0xFF | Vendor |
| Operand tags | 0x1C–0x7F | Future types |
| Operand tags | 0x80–0xFF | Vendor |
| Error codes | 0x0017–0x00FF | Future core errors |
| Error codes | 0x0100–0xFFFF | Vendor |
| Flags | 0x8000 | Reserved |
| Header version major | 0x00 | Pre-1.0 dev |
| Header version major | > 0x01 | Future |

---

## 14. Security Implications

1. No untrusted input reaches the executor without full validation (all 9 steps).
2. Bounded allocations — all STR/BYTES/LIST lengths bounded by config.
3. Deadlines honored — CALL without deadline gets a configurable default cap.
4. REF is an index into a local capability table, not a free-form string.
5. BUF_REF checked for bounds and access mode.
6. No string concatenation into wire — all encoding is length-prefixed.
7. Version negotiation is explicit — no silent downgrade.
8. Panic-free codec: Rust Result<AiclError, Never>, C++ aicl_status_t, Python AiclError.
9. No implicit telemetry — only STATS/HEALTH opcodes.
10. No network implied — ISA does not address endpoints.

---

## 15. Conformance Requirements

A runtime claiming AICL-ISA v1.0 conformance MUST:

1. Accept every opcode in §5 and validate per §8.
2. Reject malformed input with a documented error code, never panic.
3. Reject packets with different major version.
4. Honor deadline_ms for CALL operations.
5. Preserve correlation_id end-to-end.
6. Support BUF_REF and zero-copy payload exchange in-memory.
7. Support EXTENDED flag (pass-through is acceptable).
8. Pass the conformance test suite.

A runtime MAY:
- Implement only a subset of opcodes (unsupported → UNKNOWN_OPCODE).
- Add vendor opcodes in 0xF0–0xFF and tags in 0x80–0xFF.
- Add vendor error codes in 0x0100–0xFFFF.

---

## 8. Validation Algorithm

On receive, the runtime MUST execute in order:

1. **Frame check.** Verify magic, version, header_crc32. FAIL: BAD_HEADER.
2. **Length check.** payload_length vs actual. FAIL: TRUNCATED.
3. **Trailer check.** If trailer present, verify payload_crc32. FAIL: CHECKSUM.
4. **Opcode check.** In supported set. FAIL: UNKNOWN_OPCODE.
5. **Operand count check.** Matches declared. FAIL: OPERAND_COUNT.
6. **Operand type check.** Each tag in supported set. FAIL: UNKNOWN_TYPE.
7. **Schema check.** Count and types match opcode schema (§5). FAIL: SCHEMA_MISMATCH.
8. **Bounds check.** STR/BYTES within limits. FAIL: PAYLOAD_TOO_LARGE.
9. **Semantic check.** BOOL=0/1, UTF-8 validity, etc. FAIL: INVALID_VALUE.

The runtime MUST NOT proceed past any failed step. No partial interpretation.

| 0x12 | BYTES | varint+bytes | Bounded by MAX_BYTES (default 16 MiB) |
| 0x13 | UUID | 16B | RFC 4122 raw |
| 0x14 | HANDLE | 8B | Opaque endpoint/capability handle |
| 0x15 | REF | 2B BE | Short reference index (0..65535) into capability table |
| 0x16 | BUF_REF | 16B | Buffer: {id:u64, offset:u32, length:u32} |
| 0x17 | LIST | varint+operands | Heterogeneous list |
| 0x18 | MAP | varint+(k,v) | String-keyed map |
| 0x19 | NULL | 0B | Explicit null |
| 0x1A | TS_MS | 8B BE | Unix epoch milliseconds |
| 0x1B | DURATION_MS | 4B BE | Unsigned milliseconds |
| 0x1C–0x7F | reserved | — | Future types |
| 0x80–0xFF | vendor | — | Opaque to core |

Validation rules:
- STR must be valid UTF-8; invalid → ERROR_INVALID_UTF8
- BOOL only 0x00 or 0x01; others → ERROR_INVALID_BOOL
- LIST may be empty (count=0)
- MAP keys must be STR; non-string keys → ERROR_MAP_KEY_TYPE
- BUF_REF offset+length must fit inside the referenced buffer

---

## 4. Variable-Length Integers (varint)

STR/BYTES/LIST/MAP lengths use unsigned LEB128 with 4-byte max:

| Encoding | Range | Bytes |
|---|---|---|
| 0xxxxxxx | 0..127 | 1 |
| 1xxxxxxx 0xxxxxxx | 128..16383 | 2 |
| 1xxxxxxx 1xxxxxxx 0xxxxxxx | 16384..2097151 | 3 |
| 1xxxxxxx 1xxxxxxx 1xxxxxxx 0xxxxxxx | 2097152..536870911 | 4 |

Overflow past 4 bytes → ERROR_VARINT_OVERFLOW. Value exceeding limit → ERROR_PAYLOAD_TOO_LARGE.

