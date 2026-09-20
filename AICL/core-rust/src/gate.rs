//! AiclGate — security boundary enforcing capability-based access control.
//!
//! The gate inspects incoming packets and decides whether they are allowed to
//! pass through based on the current [`CapabilityTable`] and [`SafetyPolicy`].

use alloc::vec::Vec;

use crate::capability::{Capability, CapabilityTable};
use crate::error::{AiclError, AiclResult, ErrorCode};
use crate::packet::AiclPacket;
use crate::safety::SafetyPolicy;

/// Access control decision returned by the gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessDecision {
    /// The packet is allowed through unconditionally.
    Allow,
    /// The packet is denied unconditionally.
    Deny,
    /// The packet is allowed but a record should be written to an audit log.
    Audit,
}

/// Gate — enforces capability-based access control on packets.
pub struct AiclGate {
    /// Local capability table used for resolution.
    cap_table: CapabilityTable,
    /// Safety limits applied during validation.
    policy: SafetyPolicy,
    /// When true, allowed packets that touch registered capabilities are logged.
    audit_enabled: bool,
}

impl Default for AiclGate {
    fn default() -> Self { Self::new() }
}

impl AiclGate {
    /// Construct a new gate with an empty capability table and the default
    /// safety policy.
    pub fn new() -> Self {
        Self {
            cap_table: CapabilityTable::new(65535),
            policy: SafetyPolicy::default(),
            audit_enabled: false,
        }
    }

    /// Set a custom safety policy (builder pattern).
    #[inline]
    pub fn with_policy(mut self, policy: SafetyPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Enable or disable audit logging (builder pattern).
    #[inline]
    pub fn with_audit(mut self, enabled: bool) -> Self {
        self.audit_enabled = enabled;
        self
    }

    /// Make an access-control decision for a packet without performing deep
    /// safety validation.
    ///
    /// Returns [`AccessDecision::Deny`] if the packet references a capability
    /// index that is not registered. Returns [`AccessDecision::Audit`] for
    /// allowed packets when audit mode is enabled.
    pub fn check(&self, pkt: &AiclPacket) -> AccessDecision {
        // Inspect REF operands and verify each referenced cap is registered.
        for op in &pkt.operands {
            if let crate::operand::Operand::Ref(idx) = op {
                if self.cap_table.get(*idx).is_none() {
                    return AccessDecision::Deny;
                }
            }
        }
        if self.audit_enabled {
            AccessDecision::Audit
        } else {
            AccessDecision::Allow
        }
    }

    /// Perform a full capability check and safety validation.
    ///
    /// This combines [`check()`](Self::check) with [`validate_packet()`](Self::validate_packet)
    /// and returns the decision or an error if validation fails.
    pub fn check_and_validate(&self, pkt: &AiclPacket) -> AiclResult<AccessDecision> {
        let decision = self.check(pkt);
        if decision == AccessDecision::Deny {
            return Err(AiclError::new(ErrorCode::CapabilityNotFound,
                alloc::format!("capability not found for opcode {:?}", pkt.opcode)));
        }
        self.validate_packet(pkt)?;
        Ok(decision)
    }

    /// Register a capability in the gate's local capability table.
    ///
    /// Returns the 16-bit capability index assigned to the new entry.
    pub fn register_capability(&mut self, cap: Capability) -> AiclResult<u16> {
        self.cap_table.register(cap)
    }

    /// Validate a packet against the gate's safety policy.
    ///
    /// Checks operand count, string lengths, bytes lengths, and nesting depth.
    pub fn validate_packet(&self, pkt: &AiclPacket) -> AiclResult<()> {
        // Check operand count limit.
        if pkt.operands.len() as u32 > self.policy.max_operand_count {
            return Err(AiclError::new(ErrorCode::OperandCount,
                alloc::format!("operand count {} > limit {}",
                    pkt.operands.len(), self.policy.max_operand_count)));
        }

        // Recursively validate each operand.
        for op in &pkt.operands {
            self.validate_operand(op, 0)?;
        }
        Ok(())
    }

    /// Recursively validate an operand against the safety policy.
    fn validate_operand(&self, op: &crate::operand::Operand, depth: u32) -> AiclResult<()> {
        if depth > self.policy.max_nesting_depth {
            return Err(AiclError::new(ErrorCode::PayloadTooLarge,
                alloc::format!("nesting depth {} > limit {}", depth, self.policy.max_nesting_depth)));
        }
        match op {
            crate::operand::Operand::Str(s) => {
                self.policy.check_string(s)
            }
            crate::operand::Operand::Bytes(b) => {
                self.policy.check_bytes(b)
            }
            crate::operand::Operand::List(items) => {
                for item in items {
                    self.validate_operand(item, depth + 1)?;
                }
                Ok(())
            }
            crate::operand::Operand::Map(entries) => {
                for (k, v) in entries {
                    self.policy.check_string(k)?;
                    self.validate_operand(v, depth + 1)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}
