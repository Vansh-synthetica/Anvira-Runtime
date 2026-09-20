//! AiclInstruction — lowest-level raw (opcode, operand bytes) representation.
//!
//! This is the encoding-agnostic, wire-format unit of AICL. It sits between
//! `AiclPacket` (typed, semantic) and the raw byte stream produced by
//! `AiclCodec`.

use alloc::vec::Vec;

use crate::codec::AiclCodec;
use crate::error::{AiclError, AiclResult, ErrorCode};
use crate::opcode::AiclOpcode;
use crate::packet::AiclPacket;

/// Raw instruction: opcode + raw operand bytes (already-encoded).
#[derive(Debug, Clone)]
pub struct AiclInstruction {
    /// Opcode byte value.
    pub opcode: AiclOpcode,
    /// Serialised operand data following the ISA operand encoding.
    /// This is exactly the payload portion of the wire format (without the
    /// leading opcode byte and varint operand-count prefix — those are part of
    /// `operand_bytes` too, so this is the full payload starting with the
    /// operand-count varint).
    pub operand_bytes: Vec<u8>,
}

impl AiclInstruction {
    /// Construct a raw instruction from an opcode and pre-encoded operand bytes.
    pub fn new(opcode: AiclOpcode, operand_bytes: Vec<u8>) -> Self {
        Self { opcode, operand_bytes }
    }

    /// Return a NOP instruction (opcode 0x00, empty operand bytes).
    pub fn nop() -> Self {
        Self { opcode: AiclOpcode::Nop, operand_bytes: Vec::new() }
    }

    /// Return the opcode as a raw byte.
    #[inline]
    pub fn opcode_byte(&self) -> u8 {
        self.opcode.to_u8()
    }

    /// Decode the operand-count varint from the start of `operand_bytes`.
    ///
    /// The ISA encodes the operand count as a leading varint immediately after
    /// the opcode byte in the payload. This method reads and returns that count.
    pub fn operand_count_from_bytes(&self) -> AiclResult<u32> {
        decode_varint(&self.operand_bytes)
            .map(|(v, _)| v)
            .map_err(|e| AiclError::new(ErrorCode::VarintOverflow,
                alloc::format!("instruction operand count: {}", e.message)))
    }

    /// Validate that the total encoded size does not exceed `max` bytes.
    pub fn validate_max_size(&self, max: usize) -> AiclResult<()> {
        let total = 1 // opcode byte
            + self.operand_bytes.len();
        if total > max {
            return Err(AiclError::new(ErrorCode::PayloadTooLarge,
                alloc::format!("instruction {} > {} bytes", total, max)));
        }
        Ok(())
    }
}

/// Convert a typed `AiclPacket` into a raw `AiclInstruction` by encoding it.
impl From<AiclPacket> for AiclInstruction {
    fn from(pkt: AiclPacket) -> Self {
        let codec = AiclCodec::new();
        // Encode the packet — this gives us the full wire bytes (header + payload).
        let wire = codec.encode(&pkt).unwrap_or_else(|_| {
            // On encode error return a NOP; callers should validate first.
            AiclPacket::nop().encode().unwrap()
        });
        // The ISA wire payload starts after the fixed header (56 bytes) and
        // contains: [opcode byte][varint operand_count][operands...].
        // We strip the header so operand_bytes = wire[56..].
        const HDR: usize = 56;
        let payload = if wire.len() > HDR {
            wire[HDR..].to_vec()
        } else {
            Vec::new()
        };
        Self { opcode: pkt.opcode, operand_bytes: payload }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Decode a varint from the start of `data`. Returns `(value, bytes_consumed)`.
fn decode_varint(data: &[u8]) -> Result<(u32, usize), AiclError> {
    let mut value: u32 = 0;
    let mut shift = 0;
    for (i, &byte) in data.iter().enumerate() {
        if i >= 4 {
            return Err(AiclError::new(ErrorCode::VarintOverflow,
                alloc::string::String::from("varint exceeds 4 bytes")));
        }
        value |= ((byte & 0x7F) as u32) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, i + 1));
        }
        shift += 7;
    }
    Err(AiclError::new(ErrorCode::Truncated,
        alloc::string::String::from("varint truncated")))
}
