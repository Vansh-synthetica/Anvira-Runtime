# Wire-Format Audit

> **RESOLVED.** This audit's core finding — that `aicl/bin/` (Python) and
> `core-rust` implemented genuinely different, incompatible wire formats —
> has been fixed. `aicl/bin/` was rewritten to match `core-rust`'s ISA-spec
> layout exactly (56-byte header, same opcode taxonomy, same operand type
> tags, same `opcode + varint + operands` payload structure), with a
> vendor-extension mechanism (`Operand::Vendor`, tags `0x80`-`0xFF`)
> carrying Python's rich object-model fields that have no core-rust
> equivalent. Verified with real, bidirectional, byte-for-byte round trips
> — see `core-rust/examples/decode_python_packet.rs` and
> `core-rust/examples/encode_for_python.rs`, and README.md's "Wire format
> status" section. This document is kept as-is below as the historical
> record of what was wrong and how badly the two implementations had
> drifted — it does **not** describe the current state.
>
> One correction to this document's own findings, caught during the fix:
> §4.8's table claims core-rust's header CRC covers only the first 28
> bytes, diverging from the spec's implied 52-byte scope. Direct
> inspection of `core-rust/src/codec.rs` (`crc32(&data[..52])`, both
> encode and decode) shows it actually covers 52 bytes, matching the spec
> — the "28B" claim below was incorrect (or accurate against an older
> version of the code) at the time it was written.

**Scope:** Compare the three wire formats that currently coexist in the AICL repository, and verify the Rust implementation against the normative spec.

**Sources audited:**
1. `protocol/AICL-ISA.md` — the normative specification.
2. `aicl/bin/header.py`, `aicl/bin/codec_packet.py`, `aicl/bin/codec_view.py`, `aicl/bin/tlv.py`, `aicl/bin/constants.py`, `aicl/bin/ops.py`, `aicl/bin/symbol_types.py`, `aicl/bin/codec_api.py` — the AICL-BIN implementation.
3. `aicl/legacy_sl/packet.py`, `aicl/legacy_sl/registry.py`, `aicl/legacy_sl/router.py`, `aicl/legacy_sl/safety.py`, `lang/encoder.py`, `lang/decoder.py` — the AICL-SL text implementation.
4. `core-rust/src/codec.rs`, `core-rust/src/packet.rs`, `core-rust/src/opcode.rs` — the Rust implementation.

No code was modified. This document is facts only.

---

## 1. Header-Field Cross-Table

The three formats carry different fields at different positions. Most fields exist in only one or two of them.

| Field / concept | ISA spec (`AICL-ISA.md`) | AICL-BIN (`aicl/bin/header.py`) | AICL-SL (`aicl/legacy_sl/packet.py` + `lang/encoder.py`) |
|---|---|---|---|
| **Total header size** | 32 bytes (per §7) - note: 56B implied by diagram field widths; see §6 | 64 bytes (`HEADER_SIZE = 64`, `aicl/bin/constants.py:5`) | 0 bytes (text string, no fixed header) |
| **Magic** | 4B `"AICL"` (§7) | 4B `"AICL"` (`MAGIC = b"AICL"`, `constants.py:4`) | prefix literal `AICL/` followed by version text |
| **Version** | `u16 BE` major+minor (§7); spec defines v1.0 = `0x0100` | `u16 BE` slot, value constrained to `{1}` (`SUPPORTED_VERSIONS = {1}`, `constants.py:7`) | text `1.0` after the slash |
| **Flags** | 16-bit OR-combined (§6) | 16-bit (`H` in `HEADER_STRUCT`) | implicit (presence of fields like `SAF:`, `TRC:`, `CNF:`) |
| **message_id** | 16B UUID raw (§7) | 16B (`message_id` in `Header` namedtuple) | `SID:<id>` field, opaque string `s-<ms>-<hex>` (`_gen_sid()`) |
| **correlation_id** | 16B UUID raw (§7) | 16B (in header) | not present in the wire format |
| **deadline_ms** | `u32 BE` (§7) | not in fixed header; carried as TLV (`TYPE_DEADLINE = 0x0A`, `constants.py:42`), encoded `>q` int64 in `codec_packet.py:65` | not present |
| **payload_length** | `u32 BE` (§7) | 24-bit (`I` in `HEADER_STRUCT`, capped at `0xFFFFFF`, `header.py:57`) | implicit in string length |
| **header_crc32** | `u32 BE` (§7) | not present (only an optional *payload* CRC trailer) | not present |
| **session_id** | not in spec | 16B (in header) | not present (`SID:` doubles as session id, opaque string, not 16B) |
| **target_count** | not in spec | `u16` (`HH` in `HEADER_STRUCT`, capped at `0xFFFF`, `header.py:60`) | not present (targets carried in `TGT:mod1,mod2`) |
| **reserved** | not in spec | 2B (`HH` slot in `HEADER_STRUCT`, hard-coded 0) | n/a |
| **Optional trailer** | `payload_crc32 u32 BE` (§7) | 4B CRC32, gated by `FLAG_HAS_TRAILER = 0x0100` (`constants.py:25`); `TRAILER_MAGIC = 0x5CA1AB13` *defined* (`constants.py:14`) but never written by `codec_api.py` (encoder just appends raw CRC32) | n/a |
| **Payload data model** | `opcode u8` + `operand_count varint` + sequence of `(tag, typed value)` (§7) | sequence of TLVs: `type u8` + `length u24 BE` + `value` (`tlv.py:20-38`); types in `constants.py:33-58` | pipe-delimited key/value fields, e.g. `OP:CLS`, `SYM:S:"hello"`, `META:` |
| **Opcode model** | Yes, full taxonomy in §2 (`NOP`, `HELLO`, `CALL`, `MEMORY_*`, `CLASSIFY`, etc.) | Single `operation: int = 0` (one byte, 0-255) in a TLV (`TYPE_OPERATION = 0x03`); mnemonics in `aicl/bin/ops.py` (`OP_REQ`, `OP_CLS`, `OP_RSN`, `OP_MEM_*`, `OP_EXE_*`, `OP_SYS_*`) | `OP:<code>` field on the wire; mnemonics defined in `lang/symbols.py:OPERATORS` |
| **Operand / typed-value tags** | closed set: I8/I16/I32/I64, U8/U16/U32/U64, F32/F64, BOOL, STR, BYTES, UUID, HANDLE, REF, BUF_REF, LIST, MAP, NULL, TS_MS, DURATION_MS (§3) | TLV `TYPE_*` IDs in `constants.py:33-58`; inner symbol tags in `symbol_types.py:5-41` (`S_STRING=0x01`..`S_BLOB=0x0D`) | text prefixes on values: `S:"text"`, `N:42`, `T:label`, `B:true`, `V:[...]`, `J:{...}` (see `lang/encoder.py:_encode_symbol_value`) |
| **Header CRC scope** | CRC32 of pre-CRC prefix; placed at bytes 52-55 (§7) | not implemented; only payload CRC trailer | not present |
| **MAX payload size** | 16 MiB | 16 MiB (`MAX_TLV_VALUE_SIZE = 0xFFFFFF`) | unbounded string |

### 1.1 Field-count summary

- Fields defined in **ISA spec only**: none. The spec's fields are a strict subset by intent, but the wire-level encoding differs (see §2).
- Fields defined in **AICL-BIN only**: `session_id` (16B), `target_count`, `reserved`.
- Fields defined in **AICL-SL only**: no binary fields; everything is text-keyed.
- Fields present in **ISA + AICL-BIN**: `magic`, `version`, `flags`, `message_id`, `correlation_id`, `payload_length` (different width: ISA u32 BE, AICL-BIN 24-bit).
- Fields present in **ISA + AICL-SL**: `magic` (literal `AICL/` vs `AICL` bytes), `version` (text vs binary), `message_id` vs `SID:` (different format).
- Fields present in **all three (by name)**: `magic`, `version`, `flags`. Representation differs.
- Fields in **ISA only, missing in AICL-BIN header**: `deadline_ms` (moved to TLV in AICL-BIN), `header_crc32` (absent in AICL-BIN).
- Fields in **AICL-BIN header, missing in ISA**: `session_id` (16B), `target_count` (u16), `reserved` (u16), and the 64B vs 32B/56B header-size split.

---

## 2. Does `aicl/bin/` Implement the ISA Opcode/Operand Model?

**Plain answer: No.** `aicl/bin/` implements a different structure than `protocol/AICL-ISA.md`.

### 2.1 Header-level divergence

| Property | ISA spec | AICL-BIN (`aicl/bin/header.py`, `constants.py`) |
|---|---|---|
| Header size | 32 bytes (§7) - note: 56B implied by diagram field widths + padding | 64 bytes (`HEADER_SIZE = 64`) |
| Magic | 4B `"AICL"` | 4B `"AICL"` (same) |
| `deadline_ms` | in fixed header (`u32 BE`) | NOT in header; carried as a TLV (`TYPE_DEADLINE`, encoded as `>q` int64 in `codec_packet.py:65`) |
| `header_crc32` | in fixed header (`u32 BE`, bytes 52-55) | NOT present; only an optional payload CRC trailer |
| `session_id` | not in spec | in header (16B) |
| `target_count` | not in spec | in header (u16) |
| `reserved` | not in spec | 2B hard-coded 0 |
| Trailer | `payload_crc32 u32 BE` | 4B CRC32, gated by `FLAG_HAS_TRAILER`; `TRAILER_MAGIC = 0x5CA1AB13` is *defined* (`constants.py:14`) but **never written** by `codec_api.py` |
| Version field | `u16 BE` (major.minor) | `u16 BE` slot, but value constrained to `{1}` (effectively u8) |

**Concrete byte-layout evidence (Rust, for the ISA-matched variant):**
`core-rust/src/codec.rs:62-73` writes the 56-byte ISA header:
```
out[start..start+4]     = MAGIC                       //  4B
out[start+4..start+6]   = ISA_VERSION (u16 BE)        //  2B  (0x0100)
out[start+6..start+8]   = pkt.flags (u16 BE)          //  2B
out[start+8..start+24]  = msg_id (16B)                // 16B
out[start+24..start+40] = correlation_id (16B)        // 16B
out[start+40..start+44] = deadline_ms (u32 BE)        //  4B
out[start+44..start+48] = payload_len (u32 BE)        //  4B
out[start+48..start+52] = padding (zero)               //  4B  (gap for CRC alignment)
out[start+52..start+56] = hdr_crc (u32 BE)            //  4B
```
This is 56 bytes total, consistent with the spec diagram and `core-rust/src/opcode.rs:HEADER_SIZE = 56`.

**Concrete byte-layout evidence (Python AICL-BIN):**
`aicl/bin/constants.py:15` declares `HEADER_STRUCT = ">4sHH16s16s16sIHH"` which expands to: 4 (magic) + 2 (version) + 2 (flags) + 16 (session_id) + 16 (message_id) + 16 (correlation_id) + 4 (payload_length slot) + 2 (target_count) + 2 (reserved) = **64 bytes**. AICL-BIN is 8B longer than the ISA header and places different fields in the "extended" region.

### 2.2 Payload-level divergence

**ISA:** payload = `opcode u8` + `operand_count varint` + sequence of `{tag, typed value}` where tag is one of the ISA operand types (I8/I16/.../BOOL/STR/BYTES/UUID/HANDLE/REF/BUF_REF/LIST/MAP/NULL/TS_MS/DURATION_MS).

**AICL-BIN:** payload = sequence of TLVs where each TLV is `type u8 + length u24 BE + value`, and the `type` is one of the `TYPE_*` constants in `aicl/bin/constants.py:33-58` (`TYPE_ORIGIN`, `TYPE_TARGETS`, `TYPE_OPERATION`, `TYPE_INTENT`, `TYPE_SYMBOLS`, `TYPE_RESPONSE_SYMBOLS`, `TYPE_METADATA`, `TYPE_CONFIDENCE`, `TYPE_PRIORITY`, `TYPE_DEADLINE`, `TYPE_CAPABILITIES`, `TYPE_QOS`, `TYPE_SECURITY`, `TYPE_TRACE`, `TYPE_CHUNK_INFO`, `TYPE_BLOB_REF`, `TYPE_BACKPRESSURE`, `TYPE_ERROR_INFO`, `TYPE_ACK_INFO`, `TYPE_TARGET_CAPABILITIES`, `TYPE_SCHEMA_ID`, `TYPE_STREAM_ID`, `TYPE_PARTIAL_RESULT`, `TYPE_MODEL_INVOCATION`, `TYPE_TOOL_INVOCATION`, `TYPE_CHECKSUM`).

AICL-BIN has no `opcode` byte and no `varint` operand count. It has `TYPE_OPERATION = 0x03` (a single-byte TLV), and `operation: int = 0` in `aicl/bin/codec_packet.py:22`.

### 2.3 Opcode-set divergence

The ISA defines one taxonomy (`AiclOpcode` in `core-rust/src/opcode.rs:34-80`):
`NOP, HELLO, GOODBYE, DISCOVER, PING, PONG, ACK, NAK, ERROR, CANCEL, VERSION, CALL, RETURN, STREAM, STREAM_END, MEMORY_READ, MEMORY_WRITE, MEMORY_DELETE, INDEX_QUERY, INDEX_UPSERT, CAPABILITY_ADVERTISE, CAPABILITY_WITHDRAW, CAPABILITY_QUERY, MODEL_CALL, TOOL_CALL, EMBED, CLASSIFY, REASON, RETRIEVE, EXECUTE, TRACE, STATS, HEALTH, BUFFER_REF, BUFFER_RELEASE, BUFFER_SHARE, Vendor`.

AICL-BIN defines a *different* set in `aicl/bin/ops.py:4-34`:
`OP_REQUEST, OP_RESPONSE, OP_CLS, OP_GEN, OP_RSN, OP_EMB, OP_RNK, OP_SUM, OP_SYN, OP_VRF, OP_XTR, OP_TRN, OP_PLN, OP_EVL, OP_TRN_MODEL, OP_MEM_READ, OP_MEM_WRITE, OP_MEM_DELETE, OP_IDX_QUERY, OP_IDX_UPSERT, OP_RTE_DISCOVER, OP_RTE_ROUTE, OP_RTE_STATS, OP_EXE_RUN, OP_EXE_TOOL, OP_EXE_MODEL, OP_DBG_INFO, OP_SYS_HBT, OP_SYS_ERR, OP_SYS_ACK, OP_SYS_CANCEL`.

The mnemonics overlap partially (e.g. `OP_MEM_READ` approx `MEMORY_READ`, `OP_CLS` approx `CLASSIFY`, `OP_RSN` approx `REASON`, `OP_EMB` approx `EMBED`) but the numeric values and the surrounding taxonomy are different.

### 2.4 Symbol/operand-type-tag divergence

ISA operand tags (0x01-0x1C, then 0x80-0xFF vendor) cover I8/I16/I32/I64/U8/U16/U32/U64/F32/F64/BOOL/STR/BYTES/UUID/HANDLE/REF/BUF_REF/LIST/MAP/NULL/TS_MS/DURATION_MS.

AICL-BIN symbol tags (`aicl/bin/symbol_types.py:5-41`) cover a *different* set: `S_STRING=0x01, S_NUMBER=0x02, S_INTEGER=0x03, S_BOOLEAN=0x04, S_TAG=0x05, S_KEY=0x06, S_VECTOR=0x07, S_REFERENCE=0x08, S_JSON=0x09, S_DATETIME=0x0A, S_UUID=0x0B, S_NULL=0x0C, S_BLOB=0x0D`. No integer-width distinction, no IEEE-754 width distinction, no LIST/MAP, no HANDLE/REF/BUF_REF, no TS_MS/DURATION_MS.

### 2.5 Conclusion for `aicl/bin/`

`aicl/bin/` is **not** an implementation of the ISA spec. It is an independent wire format (header 64B, TLV-based payload, single-byte operation code in a TLV, AICL-BIN-specific opcode/symbol tag tables) that shares only the magic word `"AICL"`, the version-flag width (u16), and a CRC32 trailer concept.

---

## 3. Files Importing `aicl/bin/` and `aicl/legacy_sl/` (Blast Radius)

This is the file-level set of code that would need to change if one of the two formats were picked over the other (or removed).

### 3.1 Files that import `aicl.bin.*` (AICL-BIN)

| Importer file | Imports |
|---|---|
| `aicl/__init__.py` | `aicl.bin.codec_api`, `aicl.bin.codec_packet`, `aicl.bin.codec_view`, `aicl.bin.streaming`, `aicl.bin.exceptions` |
| `aicl/sl/formatter.py` | `aicl.bin.constants`, `aicl.bin.tlv`, `aicl.bin.symbols`, `aicl.bin.symbol_types`, `aicl.bin.codec_view`, `aicl.bin.ops` (lazy) |
| `tests/test_codec.py` | `aicl.bin.types`, `aicl.bin.symbol_types`, `aicl.bin.constants` |
| `tests/test_bin_errors.py` | `aicl.bin.types`, `aicl.bin.symbol_types`, `aicl.bin.constants`, `aicl.bin.exceptions` |
| `tests/test_sl.py` | `aicl.bin.types`, `aicl.bin.symbol_types`, `aicl.bin.constants`, `aicl.bin.ops` |
| `tests/benchmark.py` | `aicl.bin.types`, `aicl.bin.symbol_types` |

Total importers: **6 files** (1 package init, 1 formatter, 4 tests). Plus the package's own internal cross-imports inside `aicl/bin/` (codec_packet imports constants/tlv/symbols; codec_view imports header/types; etc.).

### 3.2 Files that import `aicl.legacy_sl.*` (AICL-SL)

| Importer file | Imports |
|---|---|
| `__init__.py` (repo root) | `aicl.legacy_sl.packet`, `aicl.legacy_sl.router`, `aicl.legacy_sl.registry`, `aicl.legacy_sl.safety` |
| `aicl/legacy_sl/router.py` | `aicl.legacy_sl.packet`, `aicl.legacy_sl.registry` |
| `aicl/legacy_sl/safety.py` | `aicl.legacy_sl.packet` |
| `bend/bridge.py` | `aicl.legacy_sl.packet` |
| `lang/encoder.py` | `aicl.legacy_sl.packet` (TYPE_CHECKING only) |
| `lang/decoder.py` | `aicl.legacy_sl.packet` (TYPE_CHECKING + runtime) |

Total importers: **6 files** (1 repo-root init, 1 sibling inside `legacy_sl/`, 1 optional Bend bridge, 1 encoder, 1 decoder, plus the `lang/__init__.py` re-export via the root init).

### 3.3 Files that import via `from aicl import ...` (touch both)

`tests/test_codec.py`, `tests/test_bin_errors.py`, `tests/test_sl.py`, `tests/benchmark.py`, and `aicl/sl/formatter.py` (lazy) all do `from aicl import encode, decode, ...`. The `aicl/__init__.py` re-exports from `aicl.bin.*` only - it does **not** re-export anything from `aicl.legacy_sl.*`. So these consumers are bound to AICL-BIN, not to AICL-SL.

### 3.4 Combined blast-radius summary

| If you... | Files that change |
|---|---|
| Remove `aicl/bin/` (AICL-BIN) | 6 importers above + the 18 files inside `aicl/bin/` itself, plus `aicl/__init__.py`, `aicl/sl/formatter.py` |
| Remove `aicl/legacy_sl/` (AICL-SL) | 6 importers above + the 4 files inside `aicl/legacy_sl/`, the root `__init__.py`, `lang/encoder.py`, `lang/decoder.py`, `lang/symbols.py`, `lang/grammar.py`, `bend/bridge.py` |
| Keep both | no change (current state) |

`core-rust/src/*` and `core-cpp/src/*` are **independent** of both `aicl/bin/` and `aicl/legacy_sl/`. They are not in the import graph of either Python package.

---

## 4. Rust Implementation: Match Against `AICL-ISA.md`

**Files audited:** `core-rust/src/codec.rs`, `core-rust/src/packet.rs`, `core-rust/src/opcode.rs`.

### 4.1 Header layout (byte offsets)

The Rust encoder at `core-rust/src/codec.rs:62-73` writes the header in this order:

| Offset | Field | Size | Source | Matches spec? |
|---|---|---|---|---|
| 0-3 | `magic` | 4B | `MAGIC = b"AICL"` (`opcode.rs:11`) | yes |
| 4-5 | `version` | u16 BE | `ISA_VERSION = 0x0100` (`opcode.rs:6`) | yes |
| 6-7 | `flags` | u16 BE | `pkt.flags.bits().to_be_bytes()` (`codec.rs:65`) | yes |
| 8-23 | `message_id` | 16B | `msg_id` UUID raw (`codec.rs:66-67`) | yes |
| 24-39 | `correlation_id` | 16B | `pkt.correlation_id` (`codec.rs:68`) | yes |
| 40-43 | `deadline_ms` | u32 BE | `pkt.deadline_ms.to_be_bytes()` (`codec.rs:69`) | yes |
| 44-47 | `payload_length` | u32 BE | `payload_len.to_be_bytes()` (`codec.rs:70`) | yes |
| 48-51 | (padding/zero) | 4B | implicit buffer padding (brings header to 56B) | spec text says 32B; diagram implies 56B with 4B gap before CRC |
| 52-55 | `header_crc32` | u32 BE | `crc32(&out[start..start+28]).to_be_bytes()` (`codec.rs:72-73`) | scope discrepancy - see §4.2 |

`HEADER_SIZE = 56` (`opcode.rs:14`) is consistent with the diagram field widths.

### 4.2 Header-CRC scope - discrepancy

The Rust decoder at `core-rust/src/codec.rs:97` computes the header CRC over **only the first 28 bytes**:

```rust
let computed = crc32(&data[..28]);
```

The spec places `header_crc32` at byte offset 52, implying the CRC covers the entire 52-byte prefix (4+2+2+16+16+4+4+4 = 52). The Rust implementation CRC scope is **28 bytes** (magic+version+flags+message_id). Self-consistent internally, but diverges from the spec implied scope.

### 4.3 Flags - match

`AiclFlags` in `core-rust/src/opcode.rs:144-202` defines exactly the 15 flag bits listed in the spec §6 (`REQUEST=0x0001` through `FROZEN=0x4000`) plus leaves `0x8000` reserved. Matches the spec table bit-for-bit.

### 4.4 Opcode taxonomy - match

`AiclOpcode` in `core-rust/src/opcode.rs:34-80` enumerates all opcodes from spec §2 with the exact numeric values (`Nop=0x00`..`BufferShare=0x62`, plus `Vendor(u8)` for 0xF0-0xFF). Matches the spec.

### 4.5 Payload (opcode + operands) - match

`AiclCodec::encode_into` (`codec.rs:48-54`) writes: `opcode (u8)` + `operand_count (varint, LEB128)` + sequence of operands. Operand encoding (`encode_operand`, `codec.rs:585-637`) uses: 1B tag, varint length prefix for STR/BYTES/LIST/MAP, fixed-width BE for integers/floats/UUID/HANDLE/REF/BufRef/TsMs/DurationMs. Matches spec §3 and §4.

### 4.6 Trailing-bytes validation - match

`codec.rs:144-147` rejects payloads where operand bytes leave trailing unconsumed data (`TrailingBytes` error). Matches the spec bounded-allocation requirement.

### 4.7 Decode-time semantics - match (with one caveat)

`decode_varint` (`codec.rs:569-583`) enforces a 4-byte cap (`VarintOverflow` error), matching spec §4. `decode` rejects empty payloads (`codec.rs:136-138`), matching spec §8 step 2. Unknown opcodes decode as `Nop` via `from_u8` fallback (`opcode.rs:99`) rather than failing decode - permissive; the spec §8 step 4 requires `UNKNOWN_OPCODE`.

### 4.8 Summary of Rust-vs-ISA discrepancies

| Item | Spec | Rust | Status |
|---|---|---|---|
| Header byte layout (0-47) | diagram implies 52B prefix before CRC | 52B prefix, matches layout | matches |
| Header CRC scope | implied 52B (full prefix before CRC field) | first 28B only (`codec.rs:97`) | **DISCREPANCY - self-consistent internally, diverges from spec** |
| `HEADER_SIZE` constant | "32 bytes fixed" in §7 text; diagram implies 52B prefix + 4B padding = 56B | `56` (`opcode.rs:14`) | consistent with diagram arithmetic; spec text says 32B which appears to be a typo |
| Flag bits (0x0001-0x4000) | 15 named bits in §6 | 15 named bits | matches |
| Opcode taxonomy | §2 table | `AiclOpcode` enum | matches |
| Operand tag set (0x01-0x1C, 0x80-0xFF) | §3 table | `OperandType` | matches |
| Varint (LEB128, 4B cap) | §4 | `encode_varint`/`decode_varint` (`codec.rs:557-583`) | matches |
| Trailer (payload_crc32 u32 BE) | §7, optional | `if has_trailer { ... crc32 }` (`codec.rs:117-132`) | matches |
| Unknown opcode handling | §8 step 4: reject | `from_u8` falls back to `Nop` (`opcode.rs:99`) | **DISCREPANCY - permissive fallback instead of reject** |
| Trailing-byte rejection | §8 step 7 | `TrailingBytes` error (`codec.rs:144-147`) | matches |

**Net:** Two discrepancies: (a) header CRC scope is 28B instead of the spec implied 52B; (b) unknown opcodes decode as `Nop` instead of being rejected. All other items match.

---

## 5. AICL-SL (`aicl/legacy_sl/`) Wire Format — REMOVED

**Historical record only.** `aicl/legacy_sl/`, the root-level `lang/` and
`utils/` packages, and `bend/` (which depended on this stack) were removed
from the repository as dead code: zero test coverage, unused by any other
module, and structurally broken (`aicl.lang` didn't exist at the path the
code imported it from — it lived at root-level `lang/`, not `aicl/lang/`).
The section below describes what existed before removal, kept for anyone
tracing the protocol's history.

For completeness, the AICL-SL text format (`aicl/legacy_sl/packet.py` + `lang/encoder.py` / `lang/decoder.py`):

- Pipe-delimited: `FIELD1:VALUE1|FIELD2:VALUE2|...`
- Begins with literal `AICL/<version>` (e.g. `AICL/1.0`).
- Recognized fields: `SID`, `ORI`, `TGT`, `OP`, `SYM`, `RSP`, `INT`, `CNF`, `META`, `SAF`, `TRC`.
- Symbols typed by single-letter prefix in value: `S:"..."`, `N:42`, `T:label`, `B:true|false`, `V:[...]`, `J:{...}`, `K:...`, `R:...` (`lang/encoder.py:_encode_symbol_value`).
- `OP` mnemonics from `lang/symbols.py:OPERATORS` - a *third* opcode set, distinct from both the ISA taxonomy and from `aicl/bin/ops.py`.
- No binary header, no magic bytes, no length field, no CRC, no deadline, no correlation_id.

Shares only the `AICL/1.0` prefix-literal and an "OP" field-by-name with the binary formats.

---

## 6. Note on the ISA Spec Text vs Diagram

The spec text in §7 says the header is 32 bytes, but the field-width arithmetic (4+2+2+16+16+4+4+4 = 52) plus the 4B gap to place `header_crc32` at offset 52 is 56 bytes total. The Rust implementation uses 56 (`HEADER_SIZE = 56` in `opcode.rs:14`). The spec 32-byte figure appears to be a typo or shorthand; the byte diagram implies 52-56 bytes. The Python AICL-BIN header is 64 bytes regardless of this.

---

*End of audit. No code modified. No recommendation made.*
