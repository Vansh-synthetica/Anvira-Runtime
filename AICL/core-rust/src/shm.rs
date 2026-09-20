//! Shared Memory Transport - native same-machine AICL IPC.
//! Zero-copy, lock-free transport using shared memory ring buffer.
//! No HTTP, localhost, or REST.

#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

use alloc::string::String;

/// SHM magic bytes.
pub const SHM_MAGIC: &[u8; 4] = b"SHMM";
/// SHM protocol version.
pub const SHM_VERSION: u16 = 1;
/// Default shared memory header size.
pub const SHM_HEADER_SIZE: usize = 64;
/// Default slot size (4 KiB).
pub const DEFAULT_SLOT_SIZE: usize = 4096;
/// Default slot count (256).
pub const DEFAULT_SLOT_COUNT: usize = 256;
/// Maximum slot size (1 MiB).
pub const MAX_SLOT_SIZE: usize = 1024 * 1024;

/// SHM error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShmError {
    CreateFailed,
    OpenFailed,
    MapFailed,
    InvalidRegion,
    NotProducer,
    NotConsumer,
    PayloadTooLarge,
    Backpressure,
    Empty,
    Corrupted,
    Timeout,
    System(i32),
}

/// SHM result type.
pub type ShmResult<T> = Result<T, ShmError>;

/// Shared memory header layout.
#[derive(Debug)]
#[repr(C)]
pub struct ShmHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub slot_size: u32,
    pub slot_count: u32,
    pub head: core::sync::atomic::AtomicU32,
    pub tail: core::sync::atomic::AtomicU32,
}

impl ShmHeader {
    /// Create a new header.
    pub fn new(slot_size: u64, slot_count: u64) -> Self {
        use core::sync::atomic::AtomicU32;
        Self {
            magic: *SHM_MAGIC,
            version: SHM_VERSION,
            slot_size: slot_size as u32,
            slot_count: slot_count as u32,
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
        }
    }

    /// Check if header is valid.
    pub fn is_valid(&self) -> bool {
        self.magic == *SHM_MAGIC && self.version == SHM_VERSION
    }

    /// Number of free slots.
    pub fn slots_free(&self) -> u32 {
        let h = self.head.load(core::sync::atomic::Ordering::Acquire);
        let t = self.tail.load(core::sync::atomic::Ordering::Acquire);
        self.slot_count - (t.wrapping_sub(h))
    }

    /// Get the byte offset for a slot index.
    pub fn slot_offset(&self, index: u32) -> usize {
        (index as usize % self.slot_count as usize) * self.slot_size as usize
    }
}

/// Descriptor for large data stored in shared memory.
#[derive(Debug, Clone, Default)]
#[repr(C)]
pub struct ShmBufferRef {
    pub id: u64,
    pub offset: u32,
    pub length: u32,
    pub alignment: u16,
    pub buffer_type: u16,
    pub ownership: u8,
    pub lifetime: u8,
    pub reserved: [u8; 2],
}

impl ShmBufferRef {
    pub const SIZE: usize = 24;
    pub fn new(id: u64, offset: u32, length: u32) -> Self {
        Self { id, offset, length, alignment: 64, buffer_type: 0, ownership: 0, lifetime: 0, reserved: [0; 2] }
    }
    pub fn is_valid(&self) -> bool { self.length > 0 && self.offset > 0 }
}

/// Configuration for the shared memory transport.
#[derive(Debug, Clone)]
pub struct ShmConfig {
    pub name: String,
    pub slot_size: usize,
    pub slot_count: usize,
}

impl Default for ShmConfig {
    fn default() -> Self {
        Self {
            name: String::from("aicl-default"),
            slot_size: DEFAULT_SLOT_SIZE,
            slot_count: DEFAULT_SLOT_COUNT,
        }
    }
}

impl ShmConfig {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), ..Default::default() }
    }
    pub fn with_slot_size(mut self, size: usize) -> Self {
        self.slot_size = size.min(MAX_SLOT_SIZE);
        self
    }
    pub fn region_size(&self) -> usize {
        SHM_HEADER_SIZE + (self.slot_size * self.slot_count)
    }
}

/// Event kind for wait operations.
#[derive(Debug, Clone, Copy)]
pub enum ShmEventKind { Read, Write }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_buffer_ref_size() { assert_eq!(core::mem::size_of::<ShmBufferRef>(), ShmBufferRef::SIZE); }
    #[test]
    fn test_header_valid() { let h = ShmHeader::new(4096, 256); assert!(h.is_valid()); }
    #[test]
    fn test_slot_offset() {
        let h = ShmHeader::new(1024, 16);
        assert_eq!(h.slot_offset(0), 0);
        assert_eq!(h.slot_offset(1), 1024);
        assert_eq!(h.slot_offset(16), 0);
    }
}
