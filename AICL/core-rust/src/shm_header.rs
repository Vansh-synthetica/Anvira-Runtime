//! Shared memory region header and constants.
use core::sync::atomic::{AtomicU64, Ordering};

/// Magic bytes identifying AICL shared memory region.
pub const SHM_MAGIC: &[u8; 4] = b"AICL";
/// Protocol version for shared memory transport.
pub const SHM_VERSION: u16 = 1;
/// Header size (must be 64-byte aligned).
pub const SHM_HEADER_SIZE: usize = 128;
/// Default slot size for small messages.
pub const DEFAULT_SLOT_SIZE: usize = 4096;
/// Default number of slots.
pub const DEFAULT_SLOT_COUNT: usize = 256;
/// Maximum slot size.
pub const MAX_SLOT_SIZE: usize = 16 * 1024 * 1024;

/// Packed header for the shared memory region.
/// Must be exactly 128 bytes and 64-byte cache-line aligned.
#[repr(C, align(64))]
pub struct ShmHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub flags: u16,
    pub capacity: u64,
    pub slot_size: u64,
    pub slot_count: u64,
    pub head: AtomicU64,
    pub tail: AtomicU64,
    pub event_read: u64,
    pub event_write: u64,
    pub header_crc: u32,
    pub reserved: [u8; 64],
}

impl ShmHeader {
    pub fn new(slot_size: u64, slot_count: u64) -> Self {
        let capacity = slot_size * slot_count;
        Self {
            magic: *SHM_MAGIC, version: SHM_VERSION, flags: 0,
            capacity, slot_size, slot_count,
            head: AtomicU64::new(0), tail: AtomicU64::new(0),
            event_read: 0, event_write: 0, header_crc: 0, reserved: [0; 64],
        }
    }

    pub fn is_valid(&self) -> bool { &self.magic == SHM_MAGIC && self.version == SHM_VERSION }
    pub fn slot_offset(&self, index: u64) -> usize { (index % self.slot_count) as usize * self.slot_size as usize }
    
    pub fn slots_used(&self) -> u64 {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        tail.wrapping_sub(head)
    }
    
    pub fn slots_free(&self) -> u64 { self.slot_count - self.slots_used() }
}
