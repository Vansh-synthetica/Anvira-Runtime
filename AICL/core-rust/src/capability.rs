//! Capability table — manages 16-bit capability references.
//!
//! Capabilities are 16-bit indices into a fixed table. Each entry has a
//! name, a `CapabilityKind`, and a payload (encoded by the kind).
//!
//! Note: capabilities are local to a process; remote agents do not see
//! the same indices.

use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU16, Ordering};

use crate::error::{AiclError, AiclResult, ErrorCode};

/// Maximum capability index (16-bit: 0..=65535).
pub const MAX_CAPABILITY: u16 = u16::MAX;

/// Capability kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityKind {
    /// A function implemented in Rust.
    Function,
    /// A local resource.
    Resource,
    /// A remote agent.
    Remote,
    /// A model reference (e.g. AI/ML).
    Model,
    /// Vendor-defined.
    Vendor(u8),
}

impl CapabilityKind {
    /// Tag byte for the kind.
    pub fn tag(self) -> u8 {
        match self {
            Self::Function => 0x01,
            Self::Resource => 0x02,
            Self::Remote => 0x03,
            Self::Model => 0x04,
            Self::Vendor(v) => v,
        }
    }
}

/// A capability record.
#[derive(Debug, Clone)]
pub struct Capability {
    /// Stable name (e.g. "calculator.add").
    pub name: String,
    /// Capability kind.
    pub kind: CapabilityKind,
    /// Optional payload (e.g. an "endpoint id" for a remote cap).
    pub payload: Option<u64>,
}

/// Capability table — fixed-capacity, lock-free allocation.
///
/// Slot 0 is reserved as the invalid/null slot.
pub struct CapabilityTable {
    slots: Vec<Option<Capability>>,
    next: AtomicU16,
}

impl CapabilityTable {
    /// Create a table with the given capacity (max 65535 entries).
    pub fn new(capacity: usize) -> Self {
        let cap = core::cmp::min(capacity, MAX_CAPABILITY as usize);
        let mut slots = Vec::with_capacity(cap);
        slots.push(None); // slot 0 reserved
        for _ in 1..cap { slots.push(None); }
        Self {
            slots,
            next: AtomicU16::new(1),
        }
    }

    /// Number of registered (non-empty) capabilities.
    pub fn len(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// True if the table is empty.
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Look up a capability by index.
    pub fn get(&self, idx: u16) -> Option<&Capability> {
        self.slots.get(idx as usize).and_then(|s| s.as_ref())
    }

    /// Look up a capability by name.
    pub fn find_by_name(&self, name: &str) -> Option<u16> {
        for (i, s) in self.slots.iter().enumerate() {
            if let Some(c) = s {
                if c.name == name { return Some(i as u16); }
            }
        }
        None
    }

    /// Register a capability. Returns its 16-bit index.
    pub fn register(&mut self, cap: Capability) -> AiclResult<u16> {
        // Try to find a free slot first; then bump the watermark.
        for i in 1..self.slots.len() {
            if self.slots[i].is_none() {
                self.slots[i] = Some(cap);
                return Ok(i as u16);
            }
        }
        if self.slots.len() >= MAX_CAPABILITY as usize {
            return Err(AiclError::new(ErrorCode::CapabilityOverflow,
                String::from("capability table full")));
        }
        let idx = self.next.fetch_add(1, Ordering::Relaxed);
        if (idx as usize) >= self.slots.len() {
            self.slots.push(Some(cap));
        } else {
            self.slots[idx as usize] = Some(cap);
        }
        Ok(idx)
    }

    /// Unregister a capability.
    pub fn unregister(&mut self, idx: u16) -> AiclResult<Capability> {
        if idx == 0 {
            return Err(AiclError::new(ErrorCode::CapabilityInvalid, "cannot unregister null cap"));
        }
        let slot = self.slots.get_mut(idx as usize)
            .ok_or_else(|| AiclError::new(ErrorCode::CapabilityInvalid,
                alloc::format!("capability index {} out of range", idx)))?;
        slot.take().ok_or_else(|| AiclError::new(ErrorCode::CapabilityInvalid,
            alloc::format!("capability index {} not registered", idx)))
    }

    /// List all registered capabilities as `(idx, name, kind)` tuples.
    pub fn list(&self) -> Vec<(u16, &str, CapabilityKind)> {
        let mut out = Vec::new();
        for (i, s) in self.slots.iter().enumerate() {
            if let Some(c) = s {
                out.push((i as u16, c.name.as_str(), c.kind));
            }
        }
        out
    }
}
