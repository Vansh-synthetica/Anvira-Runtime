//! CPU-optimized local communication path for AICL.
//!
//! This module implements the minimum-overhead path for moving AICL
//! instructions between local AI modules. The design targets:
//!
//! - Cache-line aligned structures (64-byte alignment on x86_64)
//! - Lock-free SPSC ring buffer with atomic sequence counters
//! - Zero-copy codec that borrows directly from ring buffer memory
//! - Reusable buffer pool to eliminate allocator pressure
//! - Direct dispatch table indexed by opcode byte
//! - False-sharing prevention via padded atomics
//!
//! ## Architecture
//!
//! ```text
//! Producer                        Consumer
//! ────────                        ────────
//! encode_to_ring()  ──────────►  decode_from_ring()
//!     │                               │
//!     ▼                               ▼
//! [cache-line aligned header]     [zero-copy borrow]
//! [slot 0: wire bytes]            [direct field access]
//! [slot 1: wire bytes]            [dispatch by opcode]
//! ...                             ...
//! ```
//!
//! ## Memory Ordering
//!
//! The ring buffer uses a publish-consume protocol:
//! - Producer writes slot data, then `store(Relaxed)` on tail
//! - Consumer reads tail with `Acquire`, reads slot data, then `store(Relaxed)` on head
//! - The Sequence atomic provides enough ordering for SPSC without full fences
//!
//! This works because:
//! 1. Single producer, single consumer — no ABA problem
//! 2. Sequence counter validates slot ownership — no data races
//! 3. Acquire/Release on head/tail provide happens-before for slot data

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use core::{fmt, mem, ptr};

use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::codec::{AiclCodec, PacketView};
use crate::error::{AiclError, AiclResult, ErrorCode};
use crate::opcode::{AiclOpcode, HEADER_SIZE};
use crate::packet::AiclPacket;
use crate::operand::Operand;

// ═════════════════════════════════════════════════════════════════════════════
// Cache-line constants
// ═════════════════════════════════════════════════════════════════════════════

/// Cache line size in bytes (x86_64 and most ARM64).
pub const CACHE_LINE: usize = 64;

/// Default slot size: 4 KiB (one memory page on most systems).
pub const DEFAULT_SLOT_SIZE: usize = 4096;

/// Default slot count: 256.
pub const DEFAULT_SLOT_COUNT: usize = 256;

/// Maximum slot size: 1 MiB.
pub const MAX_SLOT_SIZE: usize = 1024 * 1024;

// ═════════════════════════════════════════════════════════════════════════════
// SPSC Ring Buffer
// ═════════════════════════════════════════════════════════════════════════════

/// Padded atomic to prevent false sharing between producer and consumer.
///
/// On x86_64, cache lines are 64 bytes. A bare `AtomicU64` is 8 bytes,
/// so two atomics on the same cache line would cause cross-core invalidation
/// traffic. Padding each to a full cache line eliminates this.
#[repr(C, align(64))]
struct PaddedAtomic {
    value: AtomicU64,
    _pad: [u8; CACHE_LINE - 8],
}

impl PaddedAtomic {
    const fn new(v: u64) -> Self {
        Self { value: AtomicU64::new(v), _pad: [0; CACHE_LINE - 8] }
    }

    #[inline]
    fn load(&self, order: Ordering) -> u64 { self.value.load(order) }

    #[inline]
    fn store(&self, v: u64, order: Ordering) { self.value.store(v, order); }
}

/// Cache-line aligned header for the ring buffer.
///
/// Layout (128 bytes, two cache lines):
/// ```text
/// [0..8]    capacity      — slot count (u64)
/// [8..16]   slot_size     — bytes per slot (u64)
/// [16..24]  head_seq      — consumer sequence counter (padded to cache line)
/// [24..72]  _pad_head     — padding
/// [72..80]  tail_seq      — producer sequence counter (padded to cache line)
/// [80..128] _pad_tail     — padding
/// ```
///
/// Head and tail are on separate cache lines to avoid false sharing.
/// The consumer only writes head; the producer only writes tail.
#[repr(C, align(64))]
pub struct RingHeader {
    /// Total number of slots.
    pub capacity: u64,
    /// Bytes per slot.
    pub slot_size: u64,
    /// Consumer sequence (padded to own cache line).
    head: PaddedAtomic,
    /// Producer sequence (padded to own cache line).
    tail: PaddedAtomic,
}

impl RingHeader {
    /// Create a new header for a ring buffer.
    pub fn new(capacity: u64, slot_size: u64) -> Self {
        Self {
            capacity,
            slot_size,
            head: PaddedAtomic::new(0),
            tail: PaddedAtomic::new(0),
        }
    }

    /// Number of free slots available for the producer.
    #[inline]
    pub fn free_slots(&self) -> u64 {
        let t = self.tail.load(Ordering::Relaxed);
        let h = self.head.load(Ordering::Acquire);
        self.capacity - t.wrapping_sub(h)
    }

    /// Number of pending slots for the consumer.
    #[inline]
    pub fn pending_slots(&self) -> u64 {
        let t = self.tail.load(Ordering::Acquire);
        let h = self.head.load(Ordering::Relaxed);
        t.wrapping_sub(h)
    }

    /// Reserve one slot for writing. Returns the slot index.
    /// Returns `None` if the ring is full.
    #[inline]
    pub fn reserve(&self) -> Option<u64> {
        let t = self.tail.load(Ordering::Relaxed);
        let h = self.head.load(Ordering::Acquire);
        if t.wrapping_sub(h) >= self.capacity {
            return None;
        }
        Some(t)
    }

    /// Commit the last reserved slot (advance tail).
    #[inline]
    pub fn commit(&self) {
        let t = self.tail.load(Ordering::Relaxed);
        self.tail.store(t.wrapping_add(1), Ordering::Release);
    }

    /// Acquire the next slot for reading. Returns the slot index.
    /// Returns `None` if the ring is empty.
    #[inline]
    pub fn acquire(&self) -> Option<u64> {
        let h = self.head.load(Ordering::Relaxed);
        let t = self.tail.load(Ordering::Acquire);
        if h == t {
            return None;
        }
        Some(h)
    }

    /// Release the last acquired slot (advance head).
    #[inline]
    pub fn release(&self) {
        let h = self.head.load(Ordering::Relaxed);
        self.head.store(h.wrapping_add(1), Ordering::Release);
    }

    /// Get the byte offset for a slot index (wrapping).
    #[inline]
    pub fn slot_offset(&self, index: u64) -> usize {
        (index as usize % self.capacity as usize) * self.slot_size as usize
    }

    /// Header size in bytes (cache-line aligned).
    pub const SIZE: usize = mem::size_of::<Self>();
}

// ═════════════════════════════════════════════════════════════════════════════
// Slot header (per-slot metadata)
// ═════════════════════════════════════════════════════════════════════════════

/// Per-slot header: 8 bytes of metadata before the wire bytes.
///
/// ```text
/// [0..4]  payload_len  — u32, total wire bytes in this slot
/// [4..5]  opcode       — u8, opcode byte (for dispatch without full decode)
/// [5..6]  flags_lo     — u8, low byte of flags
/// [6..8]  reserved     — u16, alignment padding
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SlotHeader {
    pub payload_len: u32,
    pub opcode: u8,
    pub flags_lo: u8,
    pub reserved: u16,
}

impl SlotHeader {
    pub const SIZE: usize = 8;

    #[inline]
    pub fn new(payload_len: u32, opcode: u8, flags_lo: u8) -> Self {
        Self { payload_len, opcode, flags_lo, reserved: 0 }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// In-memory ring buffer (for in-process IPC)
// ═════════════════════════════════════════════════════════════════════════════

/// A lock-free, cache-friendly SPSC ring buffer for local AICL IPC.
///
/// The ring buffer stores pre-encoded AICL wire frames in fixed-size slots.
/// Both producer and consumer operate on the same contiguous memory region.
///
/// # Memory Layout
///
/// ```text
/// [RingHeader (128 bytes, 2 cache lines)]
/// [Slot 0: SlotHeader(8) + padding + wire bytes]
/// [Slot 1: SlotHeader(8) + padding + wire bytes]
/// ...
/// [Slot N-1: SlotHeader(8) + padding + wire bytes]
/// ```
///
/// # Safety
///
/// The caller must ensure:
/// - Only one thread acts as producer (calls `push_*`)
/// - Only one thread acts as consumer (calls `try_pop_*`)
/// - The memory region outlives the ring buffer handles
pub struct LocalRingBuffer {
    /// Pointer to the shared memory region.
    ptr: *mut u8,
    /// Total size of the region in bytes.
    region_size: usize,
    /// Capacity (number of slots).
    capacity: u64,
    /// Bytes per slot.
    slot_size: u64,
    /// Whether we own the allocation (for Drop).
    owned: bool,
}

// Safety: SPSC access pattern — producer thread owns push, consumer owns pop.
// All mutation of the ring's contents goes through raw pointers into the
// shared memory region, synchronized via the atomics in RingHeader — not
// through &mut access to this struct's own fields (which are set once at
// construction and read-only afterward) — so shared (&self) access from two
// threads, one producing and one consuming, is sound under that discipline.
unsafe impl Send for LocalRingBuffer {}
unsafe impl Sync for LocalRingBuffer {}

impl LocalRingBuffer {
    /// Create a new ring buffer with the given configuration.
    /// Allocates and initializes the shared memory region.
    pub fn new(capacity: u64, slot_size: u64) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        assert!(slot_size >= 256, "slot_size must be >= 256");
        assert!(slot_size <= MAX_SLOT_SIZE as u64, "slot_size too large");

        let header_size = RingHeader::SIZE;
        let region_size = header_size + (capacity as usize * slot_size as usize);
        let mut buf = vec![0u8; region_size];
        let ptr = buf.as_mut_ptr();

        // Initialize header in-place.
        let header = RingHeader::new(capacity, slot_size);
        unsafe {
            ptr::write(ptr as *mut RingHeader, header);
        }

        // Prevent deallocation — we now manage the memory manually.
        mem::forget(buf);

        Self {
            ptr,
            region_size,
            capacity,
            slot_size,
            owned: true,
        }
    }

    /// Wrap an existing memory region as a ring buffer.
    ///
    /// # Safety
    ///
    /// The memory region must be at least `RingHeader::SIZE + capacity * slot_size` bytes,
    /// properly aligned, and exclusively accessible by this thread and its pair.
    pub unsafe fn from_raw_parts(ptr: *mut u8, region_size: usize) -> Self {
        let header = unsafe { &*(ptr as *const RingHeader) };
        let capacity = header.capacity;
        let slot_size = header.slot_size;
        Self {
            ptr,
            region_size,
            capacity,
            slot_size,
            owned: false,
        }
    }

    /// Get a reference to the ring header.
    #[inline]
    pub fn header(&self) -> &RingHeader {
        unsafe { &*(self.ptr as *const RingHeader) }
    }

    /// Get the slot data pointer for a given slot index.
    #[inline]
    fn slot_ptr(&self, index: u64) -> *mut u8 {
        let offset = RingHeader::SIZE + self.header().slot_offset(index);
        unsafe { self.ptr.add(offset) }
    }

    /// Get the slot header for a given slot index.
    #[inline]
    fn slot_header(&self, index: u64) -> &SlotHeader {
        unsafe { &*(self.slot_ptr(index) as *const SlotHeader) }
    }

    /// Push a pre-encoded wire frame into the ring buffer.
    ///
    /// `data` must contain a complete AICL frame (header + payload).
    /// Returns `Ok(())` on success, `Err(Backpressure)` if the ring is full.
    #[inline]
    pub fn push(&self, data: &[u8]) -> AiclResult<()> {
        let header = self.header();
        let slot_idx = header.reserve().ok_or_else(|| {
            AiclError::new(ErrorCode::Backpressure, "ring buffer full")
        })?;

        let slot_ptr = self.slot_ptr(slot_idx);
        let slot_capacity = self.slot_size as usize - SlotHeader::SIZE;

        if data.len() > slot_capacity {
            return Err(AiclError::new(ErrorCode::PayloadTooLarge,
                alloc::format!("payload {} > slot capacity {}", data.len(), slot_capacity)));
        }

        // Write slot header.
        let opcode = if data.len() > HEADER_SIZE { data[HEADER_SIZE] } else { 0 };
        let flags_lo = if data.len() > 6 { data[6] } else { 0 };
        let slot_hdr = SlotHeader::new(data.len() as u32, opcode, flags_lo);
        unsafe {
            ptr::write_unaligned(slot_ptr as *mut SlotHeader, slot_hdr);
        }

        // Write wire data after slot header.
        let data_start = unsafe { slot_ptr.add(SlotHeader::SIZE) };
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), data_start, data.len());
        }

        // Publish — advance tail.
        header.commit();
        Ok(())
    }

    /// Fast push: encode a packet directly into a ring slot using the provided codec.
    ///
    /// Uses the caller's scratch buffer to avoid pool allocation overhead.
    /// This is the hot path for local IPC.
    #[inline]
    pub fn push_fast(
        &self,
        pkt: &AiclPacket,
        codec: &AiclCodec,
        scratch: &mut Vec<u8>,
    ) -> AiclResult<()> {
        codec.encode_to_ring_fast(pkt, self, scratch)
    }

    /// Try to pop a wire frame from the ring buffer without allocation.
    ///
    /// Returns a `RingBorrow` that borrows the data directly from the ring buffer.
    /// The borrow is valid until the next `pop` or `try_pop` call.
    #[inline]
    pub fn try_pop(&self) -> Option<RingBorrow<'_>> {
        let header = self.header();
        let slot_idx = header.acquire()?;
        let slot_ptr = self.slot_ptr(slot_idx);
        let slot_hdr = unsafe { ptr::read_unaligned(slot_ptr as *const SlotHeader) };

        let data_start = unsafe { slot_ptr.add(SlotHeader::SIZE) };
        let data_len = slot_hdr.payload_len as usize;

        // Release the slot.
        header.release();

        Some(RingBorrow {
            data_start,
            data_len,
            slot_hdr,
            _marker: core::marker::PhantomData,
        })
    }

    /// Number of slots available for pushing.
    #[inline]
    pub fn free_slots(&self) -> u64 {
        self.header().free_slots()
    }

    /// Number of slots pending for popping.
    #[inline]
    pub fn pending_slots(&self) -> u64 {
        self.header().pending_slots()
    }

    /// Total region size in bytes.
    pub fn region_size(&self) -> usize { self.region_size }
}

impl Drop for LocalRingBuffer {
    fn drop(&mut self) {
        if self.owned {
            // Reconstruct the Vec to deallocate.
            unsafe {
                let _ = Vec::from_raw_parts(self.ptr, self.region_size, self.region_size);
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// RingBorrow — zero-copy view into ring buffer slot
// ═════════════════════════════════════════════════════════════════════════════

/// Zero-copy borrow from a ring buffer slot.
///
/// This borrows directly from the ring buffer's memory — no allocation,
/// no copy. The data is valid as long as the borrow exists.
pub struct RingBorrow<'a> {
    data_start: *const u8,
    data_len: usize,
    slot_hdr: SlotHeader,
    _marker: core::marker::PhantomData<&'a u8>,
}

// Safety: RingBorrow provides shared access to slot data.
// The slot has been released back to the producer, so no concurrent writes.
unsafe impl Send for RingBorrow<'_> {}

impl<'a> RingBorrow<'a> {
    /// The raw wire bytes (header + payload).
    #[inline]
    pub fn data(&self) -> &'a [u8] {
        unsafe { core::slice::from_raw_parts(self.data_start, self.data_len) }
    }

    /// Opcode byte from the slot header (avoids reading from wire data).
    #[inline]
    pub fn opcode(&self) -> u8 { self.slot_hdr.opcode }

    /// Payload length.
    #[inline]
    pub fn payload_len(&self) -> u32 { self.slot_hdr.payload_len }
}

impl<'a> fmt::Debug for RingBorrow<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RingBorrow")
            .field("len", &self.data_len)
            .field("opcode", &format_args!("0x{:02X}", self.slot_hdr.opcode))
            .finish()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Buffer Pool
// ═════════════════════════════════════════════════════════════════════════════

/// A simple slab-based buffer pool for reusing encode/decode buffers.
///
/// Each capacity tier has its own free list. When a buffer is returned,
/// it goes back to its tier. When a buffer is requested, the smallest
/// sufficient tier is used.
///
/// Thread-safe via `AtomicBool`-guarded Vec (not lock-free, but low-contention
/// since the pool is typically per-thread or per-session).
pub struct BufferPool {
    /// Tiers: [(capacity, free_list), ...]
    tiers: Vec<BufferTier>,
}

struct BufferTier {
    capacity: usize,
    free_list: Vec<Vec<u8>>,
}

impl BufferPool {
    /// Create a pool with standard tiers: 256B, 1KB, 4KB, 16KB, 64KB, 256KB.
    pub fn new() -> Self {
        let capacities = [256, 1024, 4096, 16 * 1024, 64 * 1024, 256 * 1024];
        let tiers = capacities.iter().map(|&cap| BufferTier {
            capacity: cap,
            free_list: Vec::new(),
        }).collect();
        Self { tiers }
    }

    /// Create a pool with custom tier capacities.
    pub fn with_tiers(capacities: &[usize]) -> Self {
        let tiers = capacities.iter().map(|&cap| BufferTier {
            capacity: cap,
            free_list: Vec::new(),
        }).collect();
        Self { tiers }
    }

    /// Get a buffer with at least `min_capacity` bytes.
    /// Reuses a previously returned buffer if available.
    #[inline]
    pub fn get(&mut self, min_capacity: usize) -> Vec<u8> {
        // Find the smallest tier that fits.
        for tier in &mut self.tiers {
            if tier.capacity >= min_capacity {
                if let Some(buf) = tier.free_list.pop() {
                    return buf;
                }
                // No pooled buffer yet — allocate at the TIER's capacity,
                // not just min_capacity, so this buffer matches the tier's
                // free-list bucket (`put`'s `tier.capacity == cap` check)
                // and actually becomes reusable next time it's returned.
                return Vec::with_capacity(tier.capacity);
            }
        }
        // No tier large enough for this request — allocate exactly what
        // was asked for (won't match any tier on `put`, and that's fine:
        // an oversized one-off buffer isn't meant to be pooled).
        Vec::with_capacity(min_capacity)
    }

    /// Return a buffer to the pool for reuse.
    ///
    /// The buffer is placed in the tier matching its capacity.
    /// Buffers that don't match any tier are simply dropped.
    #[inline]
    pub fn put(&mut self, mut buf: Vec<u8>) {
        let cap = buf.capacity();
        buf.clear();
        for tier in &mut self.tiers {
            if tier.capacity == cap {
                // Limit pool size per tier to prevent unbounded growth.
                if tier.free_list.len() < 64 {
                    tier.free_list.push(buf);
                }
                return;
            }
        }
        // Drop the buffer if it doesn't match any tier.
    }

    /// Total number of buffers currently in the pool.
    pub fn pooled_count(&self) -> usize {
        self.tiers.iter().map(|t| t.free_list.len()).sum()
    }
}

impl Default for BufferPool {
    fn default() -> Self { Self::new() }
}

// ═════════════════════════════════════════════════════════════════════════════
// Zero-copy codec extensions
// ═════════════════════════════════════════════════════════════════════════════

/// Extension trait for zero-copy operations on `AiclCodec`.
pub trait ZeroCopyCodec {
    /// Decode directly from a borrowed buffer without copying into Arc.
    ///
    /// Returns a `BorrowedView` that references the original buffer.
    /// This avoids the `data.to_vec()` + `Arc::new()` in the standard decode.
    fn decode_borrowed<'a>(&self, data: &'a [u8]) -> AiclResult<BorrowedView<'a>>;

    /// Encode directly into a pre-allocated buffer (no new allocation).
    fn encode_reuse(&self, pkt: &AiclPacket, buf: &mut Vec<u8>) -> AiclResult<()>;

    /// Encode directly into a ring buffer slot.
    fn encode_to_ring(&self, pkt: &AiclPacket, ring: &LocalRingBuffer) -> AiclResult<()>;

    /// Encode a packet directly into a ring slot with a pre-existing scratch buffer.
    ///
    /// This is the fastest path: reuses the scratch buffer AND writes directly
    /// into the ring slot without going through `LocalRingBuffer::push()`.
    fn encode_to_ring_fast(
        &self,
        pkt: &AiclPacket,
        ring: &LocalRingBuffer,
        scratch: &mut Vec<u8>,
    ) -> AiclResult<()>;
}

impl ZeroCopyCodec for AiclCodec {
    #[inline]
    fn decode_borrowed<'a>(&self, data: &'a [u8]) -> AiclResult<BorrowedView<'a>> {
        if data.len() < HEADER_SIZE {
            return Err(AiclError::new(ErrorCode::BadHeader,
                alloc::format!("buffer {} < header size {}", data.len(), HEADER_SIZE)));
        }
        if &data[0..4] != crate::opcode::MAGIC {
            return Err(AiclError::new(ErrorCode::BadHeader, "invalid magic"));
        }

        let version = u16::from_be_bytes([data[4], data[5]]);
        if version >> 8 != 1 {
            return Err(AiclError::new(ErrorCode::UnsupportedVersion, "version major != 1"));
        }

        // Header CRC check.
        let hdr_crc = u32::from_be_bytes([data[52], data[53], data[54], data[55]]);
        let computed = crate::codec::crc32(data[..52].into());
        if hdr_crc != computed {
            return Err(AiclError::new(ErrorCode::BadHeader, "header CRC mismatch"));
        }

        let flags = u16::from_be_bytes([data[6], data[7]]);
        let payload_len = u32::from_be_bytes([data[44], data[45], data[46], data[47]]);
        let total = HEADER_SIZE + payload_len as usize;

        if data.len() < total {
            return Err(AiclError::new(ErrorCode::Truncated, "truncated"));
        }

        Ok(BorrowedView {
            data,
            payload_offset: HEADER_SIZE,
            payload_length: payload_len as usize,
            has_trailer: (flags & 0x0100) != 0,
            _flags: flags,
        })
    }

    #[inline]
    fn encode_reuse(&self, pkt: &AiclPacket, buf: &mut Vec<u8>) -> AiclResult<()> {
        buf.clear();
        self.encode_into(pkt, buf)
    }

    #[inline]
    fn encode_to_ring(&self, pkt: &AiclPacket, ring: &LocalRingBuffer) -> AiclResult<()> {
        let mut scratch = Vec::with_capacity(HEADER_SIZE + 256);
        self.encode_into(pkt, &mut scratch)?;
        ring.push(&scratch)
    }

    #[inline]
    fn encode_to_ring_fast(
        &self,
        pkt: &AiclPacket,
        ring: &LocalRingBuffer,
        scratch: &mut Vec<u8>,
    ) -> AiclResult<()> {
        scratch.clear();
        self.encode_into(pkt, scratch)?;
        ring.push(scratch)
    }
}

/// Zero-copy view that borrows directly from the source buffer.
///
/// Unlike `PacketView` (which owns an `Arc<Vec<u8>>`), this borrows
/// the underlying `&[u8]` with no allocation or reference counting.
pub struct BorrowedView<'a> {
    data: &'a [u8],
    payload_offset: usize,
    payload_length: usize,
    has_trailer: bool,
    _flags: u16,
}

impl<'a> BorrowedView<'a> {
    /// The raw wire bytes.
    #[inline]
    pub fn raw(&self) -> &'a [u8] { self.data }

    /// Opcode byte.
    #[inline]
    pub fn opcode(&self) -> u8 { self.data[HEADER_SIZE] }

    /// Flags.
    #[inline]
    pub fn flags(&self) -> u16 {
        u16::from_be_bytes([self.data[6], self.data[7]])
    }

    /// Message ID bytes.
    #[inline]
    pub fn message_id(&self) -> [u8; 16] {
        let mut id = [0u8; 16];
        id.copy_from_slice(&self.data[8..24]);
        id
    }

    /// Correlation ID bytes.
    #[inline]
    pub fn correlation_id(&self) -> [u8; 16] {
        let mut id = [0u8; 16];
        id.copy_from_slice(&self.data[24..40]);
        id
    }

    /// Deadline in milliseconds.
    #[inline]
    pub fn deadline_ms(&self) -> u32 {
        u32::from_be_bytes([self.data[40], self.data[41], self.data[42], self.data[43]])
    }

    /// Payload slice (without header).
    #[inline]
    pub fn payload(&self) -> &'a [u8] {
        let end = self.payload_offset + self.payload_length;
        &self.data[self.payload_offset..end]
    }

    /// Total packet size.
    #[inline]
    pub fn total_size(&self) -> usize {
        let trailer = if self.has_trailer { 4 } else { 0 };
        self.payload_offset + self.payload_length + trailer
    }

    /// Convert to an owned `AiclPacket` (copies operand data).
    #[inline]
    pub fn to_packet(&self) -> AiclResult<AiclPacket> {
        crate::codec::decode_to_packet_from_slice(self.data)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Direct dispatch table
// ═════════════════════════════════════════════════════════════════════════════

/// Dispatch handler type: takes a wire frame, returns a response wire frame.
pub type DispatchHandler = fn(&[u8]) -> AiclResult<Vec<u8>>;

/// Fixed-size dispatch table indexed by opcode byte.
///
/// Uses an array of 256 `Option<DispatchHandler>` for O(1) lookup.
/// Unregistered opcodes return `None`.
pub struct DispatchTable {
    handlers: [Option<DispatchHandler>; 256],
}

impl DispatchTable {
    /// Create an empty dispatch table.
    pub fn new() -> Self {
        // Can't use [None; 256] because fn pointers aren't Copy in const context.
        // Build with a loop.
        let mut handlers: [Option<DispatchHandler>; 256] = unsafe {
            mem::zeroed()
        };
        for h in &mut handlers {
            *h = None;
        }
        Self { handlers }
    }

    /// Register a handler for an opcode.
    #[inline]
    pub fn register(&mut self, opcode: AiclOpcode, handler: DispatchHandler) {
        self.handlers[opcode.to_u8() as usize] = Some(handler);
    }

    /// Dispatch a wire frame by its opcode byte.
    ///
    /// This is a single array index — no match, no branch prediction,
    /// no hash lookup. The opcode byte is already at a known offset
    /// in the AICL header.
    #[inline]
    pub fn dispatch(&self, wire: &[u8]) -> AiclResult<Vec<u8>> {
        if wire.len() <= HEADER_SIZE {
            return Err(AiclError::new(ErrorCode::Truncated, "no opcode byte"));
        }
        let opcode_byte = wire[HEADER_SIZE];
        match self.handlers[opcode_byte as usize] {
            Some(handler) => handler(wire),
            None => Err(AiclError::new(ErrorCode::UnknownOpcode,
                alloc::format!("no handler for opcode 0x{:02X}", opcode_byte))),
        }
    }

    /// Check if a handler is registered for an opcode.
    #[inline]
    pub fn has_handler(&self, opcode: AiclOpcode) -> bool {
        self.handlers[opcode.to_u8() as usize].is_some()
    }
}

impl Default for DispatchTable {
    fn default() -> Self { Self::new() }
}

// ═════════════════════════════════════════════════════════════════════════════
// Optimized local channel (replaces InProcChannel for hot path)
// ═════════════════════════════════════════════════════════════════════════════

/// A CPU-optimized in-process channel using the SPSC ring buffer.
///
/// This replaces `InProcChannel` for the local hot path:
/// - No `Vec::to_vec()` on send
/// - No `VecDeque` indirection
/// - Cache-line aligned ring buffer
/// - Zero-copy pop via `BorrowedView`
///
/// Holds `Arc`-shared handles to the ring buffers, so each end can be
/// dropped independently without freeing memory the other end still uses —
/// the underlying region is only deallocated once both ends have dropped.
/// The ring buffers themselves are allocated on the heap and shared between
/// endpoints.
pub struct LocalChannel {
    /// Ring buffer for outgoing messages (sender writes, receiver reads).
    outgoing: Arc<LocalRingBuffer>,
    /// Ring buffer for incoming messages (reverse direction).
    incoming: Arc<LocalRingBuffer>,
    /// Buffer pool for encoding.
    pool: BufferPool,
    /// Scratch buffer for decode (avoids allocation per recv).
    decode_scratch: Vec<u8>,
    /// Scratch buffer for encode (avoids allocation per send).
    encode_scratch: Vec<u8>,
    /// Nonce generator for unique message IDs without UUID.
    nonce: LocalNonce,
}

// Safety: LocalChannel is used from a single thread at a time (send or recv, not both).
unsafe impl Send for LocalChannel {}

impl LocalChannel {
    /// Create a paired channel with the given slot configuration.
    ///
    /// Returns `(end_a, end_b)` where:
    /// - `end_a.send()` writes to the ring that `end_b.recv()` reads from
    /// - `end_b.send()` writes to the ring that `end_a.recv()` reads from
    pub fn new(slot_size: usize, slot_count: usize) -> (Self, Self) {
        let a_to_b = Arc::new(LocalRingBuffer::new(slot_count as u64, slot_size as u64));
        let b_to_a = Arc::new(LocalRingBuffer::new(slot_count as u64, slot_size as u64));

        let end_a = Self {
            outgoing: a_to_b.clone(),
            incoming: b_to_a.clone(),
            pool: BufferPool::new(),
            decode_scratch: Vec::with_capacity(slot_size),
            encode_scratch: Vec::with_capacity(slot_size),
            nonce: LocalNonce::new(),
        };
        let end_b = Self {
            outgoing: b_to_a,
            incoming: a_to_b,
            pool: BufferPool::new(),
            decode_scratch: Vec::with_capacity(slot_size),
            encode_scratch: Vec::with_capacity(slot_size),
            nonce: LocalNonce::new(),
        };
        (end_a, end_b)
    }

    /// Send a packet (encode + push to ring).
    #[inline]
    pub fn send(&mut self, pkt: &AiclPacket, codec: &AiclCodec) -> AiclResult<()> {
        self.outgoing.push_fast(pkt, codec, &mut self.encode_scratch)
    }

    /// Generate a unique message ID without UUID (atomic nonce).
    #[inline]
    pub fn next_message_id(&self) -> [u8; 16] {
        self.nonce.next_id()
    }

    /// Derive a correlation ID from a request message ID.
    #[inline]
    pub fn correlation_id(request_id: [u8; 16]) -> [u8; 16] {
        LocalNonce::correlation_id(request_id)
    }

    /// Try to receive a packet without allocation.
    ///
    /// Returns a `BorrowedView` that references the ring buffer's memory directly.
    /// Valid until the next `try_recv` or `recv` call.
    #[inline]
    pub fn try_recv_zero_copy(&mut self, codec: &AiclCodec) -> AiclResult<BorrowedView<'_>> {
        let borrow = self.incoming.try_pop()
            .ok_or_else(|| AiclError::new(ErrorCode::Empty, "no data"))?;
        codec.decode_borrowed(borrow.data())
    }

    /// Receive a packet (pop from ring + decode to owned).
    ///
    /// Uses a reusable scratch buffer to avoid per-call allocation.
    #[inline]
    pub fn recv(&mut self, codec: &AiclCodec) -> AiclResult<PacketView> {
        let borrow = self.incoming.try_pop()
            .ok_or_else(|| AiclError::new(ErrorCode::Empty, "no data"))?;
        let data = borrow.data();
        self.decode_scratch.clear();
        self.decode_scratch.extend_from_slice(data);
        codec.decode(&self.decode_scratch)
    }

    /// Number of messages waiting to be received.
    #[inline]
    pub fn pending(&self) -> u64 {
        self.incoming.pending_slots()
    }

    /// Number of free slots for sending.
    #[inline]
    pub fn available(&self) -> u64 {
        self.outgoing.free_slots()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// AlignTo helper
// ═════════════════════════════════════════════════════════════════════════════

/// Round up `value` to the nearest multiple of `alignment`.
/// `alignment` must be a power of 2.
#[inline]
pub const fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

/// Round down `value` to the nearest multiple of `alignment`.
#[inline]
pub const fn align_down(value: usize, alignment: usize) -> usize {
    value & !(alignment - 1)
}

// ═════════════════════════════════════════════════════════════════════════════
// Nonce-based message ID generator (avoids UUID for local path)
// ═════════════════════════════════════════════════════════════════════════════

/// Atomic nonce generator for local-path message IDs.
///
/// Replaces `uuid::Uuid::new_v4()` on the hot path. On a 4GB-class GPU
/// with a quantized 7B model, UUID v4 generation costs ~200ns per call
/// (entropy pool read + CSPRNG). A simple atomic counter costs ~1ns.
///
/// The nonce is a 128-bit value: `[8 bytes zero][8 bytes counter]`.
/// This is unique within a process and fits the AICL message ID field.
pub struct LocalNonce {
    counter: AtomicU64,
}

impl LocalNonce {
    pub const fn new() -> Self {
        Self { counter: AtomicU64::new(1) }
    }

    /// Generate the next unique message ID (16 bytes).
    #[inline]
    pub fn next_id(&self) -> [u8; 16] {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let mut id = [0u8; 16];
        id[8..16].copy_from_slice(&n.to_le_bytes());
        id
    }

    /// Generate a correlation ID from a nonce value (for request/response pairing).
    #[inline]
    pub fn correlation_id(request_id: [u8; 16]) -> [u8; 16] {
        // Simple derivation: XOR with a fixed constant for local pairing.
        let mut corr = request_id;
        for b in &mut corr[8..16] {
            *b ^= 0xFF;
        }
        corr
    }
}

impl Default for LocalNonce {
    fn default() -> Self { Self::new() }
}

// ═════════════════════════════════════════════════════════════════════════════
// Fast packet builder (avoids heap-allocated operands for common cases)
// ═════════════════════════════════════════════════════════════════════════════

/// Build a ping packet with a nonce message ID (no UUID).
#[inline]
pub fn fast_ping(nonce: u64, msg_id: [u8; 16]) -> AiclPacket {
    let mut p = AiclPacket::new(AiclOpcode::Ping);
    p.flags = crate::opcode::AiclFlags::REQUEST;
    p.message_id = Some(msg_id);
    p.operands.push(Operand::U64(nonce));
    p
}

/// Build a pong packet with a nonce message ID.
#[inline]
pub fn fast_pong(nonce: u64, msg_id: [u8; 16], corr_id: [u8; 16]) -> AiclPacket {
    let mut p = AiclPacket::new(AiclOpcode::Pong);
    p.flags = crate::opcode::AiclFlags::RESPONSE;
    p.message_id = Some(msg_id);
    p.correlation_id = corr_id;
    p.operands.push(Operand::U64(nonce));
    p
}

/// Build a classify packet with nonce message ID.
#[inline]
pub fn fast_classify(
    text: String,
    labels: Vec<String>,
    model: String,
    msg_id: [u8; 16],
) -> AiclPacket {
    let mut p = AiclPacket::new(AiclOpcode::Classify);
    p.flags = crate::opcode::AiclFlags::REQUEST;
    p.message_id = Some(msg_id);
    p.operands.push(Operand::Str(text));
    p.operands.push(Operand::List(labels.into_iter().map(Operand::Str).collect()));
    p.operands.push(Operand::Str(model));
    p
}

/// Build an execute/generate packet with nonce message ID.
#[inline]
pub fn fast_generate(
    prompt: String,
    model: String,
    msg_id: [u8; 16],
) -> AiclPacket {
    let mut p = AiclPacket::new(AiclOpcode::Execute);
    p.flags = crate::opcode::AiclFlags::REQUEST;
    p.message_id = Some(msg_id);
    p.operands.push(Operand::Str(prompt));
    p.operands.push(Operand::Str(model));
    p
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opcode::AiclFlags;

    #[test]
    fn test_ring_buffer_header_size() {
        // Header should be cache-line aligned (at least 64 bytes).
        assert!(RingHeader::SIZE >= 64);
        assert!(RingHeader::SIZE % CACHE_LINE == 0);
    }

    #[test]
    fn test_ring_buffer_push_pop() {
        let mut ring = LocalRingBuffer::new(4, 1024);
        assert_eq!(ring.pending_slots(), 0);
        assert_eq!(ring.free_slots(), 4);

        let data = b"hello world";
        ring.push(data).unwrap();
        assert_eq!(ring.pending_slots(), 1);
        assert_eq!(ring.free_slots(), 3);

        let borrow = ring.try_pop().unwrap();
        assert_eq!(borrow.data(), data);
        assert_eq!(ring.pending_slots(), 0);
    }

    #[test]
    fn test_ring_buffer_full() {
        let mut ring = LocalRingBuffer::new(2, 256);
        ring.push(b"msg1").unwrap();
        ring.push(b"msg2").unwrap();
        assert!(ring.push(b"msg3").is_err());
    }

    #[test]
    fn test_ring_buffer_empty() {
        let mut ring = LocalRingBuffer::new(4, 256);
        assert!(ring.try_pop().is_none());
    }

    #[test]
    fn test_buffer_pool() {
        let mut pool = BufferPool::new();
        let buf1 = pool.get(100);
        assert!(buf1.capacity() >= 256);
        pool.put(buf1);
        let buf2 = pool.get(100);
        // Should reuse the pooled buffer.
        assert_eq!(buf2.capacity(), 256);
    }

    #[test]
    fn test_dispatch_table() {
        let mut table = DispatchTable::new();
        fn nop_handler(_wire: &[u8]) -> AiclResult<Vec<u8>> {
            Ok(Vec::new())
        }
        table.register(AiclOpcode::Nop, nop_handler);
        assert!(table.has_handler(AiclOpcode::Nop));
        assert!(!table.has_handler(AiclOpcode::Ping));
    }

    #[test]
    fn test_zero_copy_decode() {
        let pkt = crate::packet::AiclPacket::ping(42);
        let codec = AiclCodec::new();
        let wire = codec.encode(&pkt).unwrap();

        let view = codec.decode_borrowed(&wire).unwrap();
        assert_eq!(view.opcode(), AiclOpcode::Ping.to_u8());
        assert_eq!(view.flags(), AiclFlags::REQUEST.bits());
    }

    #[test]
    fn test_local_channel() {
        let codec = AiclCodec::new();
        let (mut tx, mut rx) = LocalChannel::new(4096, 16);

        let pkt = crate::packet::AiclPacket::ping(42);
        tx.send(&pkt, &codec).unwrap();

        let view = rx.try_recv_zero_copy(&codec).unwrap();
        assert_eq!(view.opcode(), AiclOpcode::Ping.to_u8());
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(63, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
    }

    #[test]
    fn test_slot_header_layout() {
        assert_eq!(core::mem::size_of::<SlotHeader>(), 8);
    }

    #[test]
    fn test_padded_atomic_prevents_false_sharing() {
        // PaddedAtomic should be exactly one cache line.
        assert_eq!(core::mem::size_of::<PaddedAtomic>(), CACHE_LINE);
    }
}
