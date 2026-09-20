//! Safety limits and conformance checks.

use alloc::string::String;

use crate::error::{AiclError, AiclResult, ErrorCode};

/// Safety policy — a set of bounds enforced on every packet.
#[derive(Debug, Clone)]
pub struct SafetyPolicy {
    /// Maximum payload size in bytes.
    pub max_payload_size: usize,
    /// Maximum operand count per packet.
    pub max_operand_count: u32,
    /// Maximum string length.
    pub max_string_length: u32,
    /// Maximum bytes length.
    pub max_bytes_length: u32,
    /// Maximum nesting depth (List/Map).
    pub max_nesting_depth: u32,
}

impl Default for SafetyPolicy {
    fn default() -> Self {
        Self {
            max_payload_size: 16 * 1024 * 1024, // 16 MB
            max_operand_count: 1024,
            max_string_length: 64 * 1024,
            max_bytes_length: 16 * 1024 * 1024,
            max_nesting_depth: 32,
        }
    }
}

impl SafetyPolicy {
    /// Validate a string length.
    pub fn check_string(&self, s: &str) -> AiclResult<()> {
        if s.len() > self.max_string_length as usize {
            return Err(AiclError::new(ErrorCode::PayloadTooLarge,
                format!("string len {} > {}", s.len(), self.max_string_length)));
        }
        Ok(())
    }

    /// Validate a bytes length.
    pub fn check_bytes(&self, b: &[u8]) -> AiclResult<()> {
        if b.len() > self.max_bytes_length as usize {
            return Err(AiclError::new(ErrorCode::PayloadTooLarge,
                format!("bytes len {} > {}", b.len(), self.max_bytes_length)));
        }
        Ok(())
    }
}
