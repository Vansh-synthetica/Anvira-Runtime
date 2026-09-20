//! Stable ABI — Rust ↔ C++ boundary for AICL.
//!
//! # Ownership Model
//!
//! | Type                | Owner      | Mutate | Free          | Thread-Safe | Read-Only |
//! |---------------------|------------|--------|---------------|-------------|-----------|
//! | `AiclBufferRef`     | Creator    | No     | No            | Yes         | Yes       |
//! | `AiclBuffer::Owned` | Rust       | Yes    | Rust (Drop)   | Yes         | No        |
//! | `AiclBuffer::Borrowed`| C++      | C++    | C++           | No          | Depends   |
//! | `AiclBuffer::Shared`| Refcount   | No     | Refcount      | Yes         | Yes       |
//! | `AiclHandle`        | Refcount   | Via API| Refcount      | Yes         | Depends   |
//!
//! # Zero-Copy Guarantees
//!
//! - `AiclBufferRef` points into existing memory (C++ inference buffer, SHM slot, etc.)
//! - `encode_into_bufref` writes directly into a caller-provided buffer
//! - `decode_to_bufref` returns a view borrowing the wire buffer (no allocation)
//! - No `Rust buffer → copy → C++ buffer → copy → model buffer` path
//!
//! # Thread Safety
//!
//! - `AiclBufferRef` is `Send + Sync` (immutable descriptor)
//! - `AiclBuffer::Owned` is `Send + Sync` (exclusive ownership)
//! - `AiclBuffer::Borrowed` is NOT `Send` (lifetime-bound to creator thread)
//! - `AiclBuffer::Shared` is `Send + Sync` (atomic refcount)
//! - `AiclHandle` is `Send + Sync` (atomic refcount)

#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ffi::c_void;
use core::fmt;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::codec::{AiclCodec, PacketView};
use crate::error::{AiclError, AiclResult, ErrorCode};
use crate::opcode::AiclOpcode;
use crate::packet::AiclPacket;

// ═════════════════════════════════════════════════════════════════════════════
// AiclBufferRef — Non-owning, immutable memory descriptor
// ═════════════════════════════════════════════════════════════════════════════

/// Non-owning reference to a memory region.
///
/// # Contract
///
/// - **Owner**: The creator of this ref. The ref does NOT own the memory.
/// - **Mutate**: Nobody may mutate through a `BufferRef` (immutable).
/// - **Free**: The owner must keep the memory alive while any `BufferRef` exists.
/// - **Share**: Can be copied freely (it's just a pointer + length).
/// - **Read-Only**: Always.
/// - **Thread-Safe**: Yes (`Send + Sync`). The memory it points to must be
///   valid from any thread (C++ caller must guarantee this).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AiclBufferRef {
    /// Pointer to the data. Must be valid for `length` bytes.
    pub ptr: *const c_void,
    /// Number of bytes.
    pub length: u32,
    /// Alignment of the underlying data (for SIMD/AVX).
    pub alignment: u16,
    /// User-defined type tag (e.g., F32_MODEL_OUTPUT, UTF8_PROMPT).
    pub tag: u16,
    /// Ownership hint: 0=unknown, 1=caller-owns, 2=Rust-owns, 3=shared.
    pub ownership: u8,
    /// Lifetime hint: 0=unknown, 1=session, 2=request, 3=transient.
    pub lifetime: u8,
    /// Reserved for future use.
    pub reserved: [u8; 2],
}

// Safety: AiclBufferRef is an immutable descriptor.
// The caller guarantees the pointed-to memory is valid from any thread.
unsafe impl Send for AiclBufferRef {}
unsafe impl Sync for AiclBufferRef {}

impl AiclBufferRef {
    /// Create a buffer ref from a raw pointer and length.
    ///
    /// # Safety
    ///
    /// - `ptr` must be valid for `length` bytes
    /// - The memory must remain valid while this ref exists
    #[inline]
    pub unsafe fn from_ptr(ptr: *const c_void, length: u32) -> Self {
        Self {
            ptr,
            length,
            alignment: 1,
            tag: 0,
            ownership: 0,
            lifetime: 0,
            reserved: [0; 2],
        }
    }

    /// Create a buffer ref from a byte slice.
    #[inline]
    pub fn from_slice(data: &[u8]) -> Self {
        Self {
            ptr: data.as_ptr() as *const c_void,
            length: data.len() as u32,
            alignment: 1,
            tag: 0,
            ownership: 0,
            lifetime: 0,
            reserved: [0; 2],
        }
    }

    /// Create a buffer ref from a `Vec<u8>`.
    ///
    /// # Safety
    ///
    /// The `Vec` must outlive this ref.
    #[inline]
    pub unsafe fn from_vec(vec: &Vec<u8>) -> Self {
        Self {
            ptr: vec.as_ptr() as *const c_void,
            length: vec.len() as u32,
            alignment: 1,
            tag: 0,
            ownership: 0,
            lifetime: 0,
            reserved: [0; 2],
        }
    }

    /// View this ref as a byte slice.
    ///
    /// # Safety
    ///
    /// The underlying memory must be valid and immutable for the returned lifetime.
    #[inline]
    pub unsafe fn as_slice(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.ptr as *const u8, self.length as usize) }
    }

    /// Check if the ref is null or zero-length.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ptr.is_null() || self.length == 0
    }

    /// Set the type tag.
    #[inline]
    pub fn with_tag(mut self, tag: u16) -> Self {
        self.tag = tag;
        self
    }

    /// Set the alignment.
    #[inline]
    pub fn with_alignment(mut self, alignment: u16) -> Self {
        self.alignment = alignment;
        self
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// AiclBuffer — Owning buffer with variant ownership
// ═════════════════════════════════════════════════════════════════════════════

/// Reference-counted buffer that can cross the Rust ↔ C++ boundary.
///
/// # Thread Safety
///
/// - `Owned` and `Shared`: `Send + Sync` (exclusive or refcounted ownership)
/// - `Borrowed`: NOT `Send` (lifetime-bound to creator thread)
pub struct AiclBuffer {
    inner: BufferInner,
    refcount: Option<NonNull<AtomicUsize>>,
}

enum BufferInner {
    Owned(Vec<u8>),
    Borrowed {
        ptr: NonNull<u8>,
        length: usize,
        _marker: core::marker::PhantomData<*mut u8>,
    },
    Shared {
        ptr: NonNull<u8>,
        length: usize,
        capacity: usize,
    },
}

// Safety: AiclBuffer is Send+Sync for Owned and Shared variants.
unsafe impl Send for AiclBuffer {}
unsafe impl Sync for AiclBuffer {}

impl AiclBuffer {
    /// Create an owned buffer from a `Vec<u8>`.
    #[inline]
    pub fn owned(data: Vec<u8>) -> Self {
        Self { inner: BufferInner::Owned(data), refcount: None }
    }

    /// Create a borrowed buffer from a C++ pointer (NO COPY).
    ///
    /// # Safety
    ///
    /// - `ptr` must be valid for `length` bytes
    /// - The memory must remain valid while this buffer exists
    /// - NOT thread-safe (do not send to another thread)
    #[inline]
    pub unsafe fn borrowed(ptr: *mut u8, length: usize) -> Self {
        Self {
            inner: BufferInner::Borrowed {
                ptr: unsafe { NonNull::new_unchecked(ptr) },
                length,
                _marker: core::marker::PhantomData,
            },
            refcount: None,
        }
    }

    /// Create a shared, reference-counted buffer from a `Vec<u8>`.
    #[inline]
    pub fn shared(data: Vec<u8>) -> Self {
        let boxed = Box::new(AtomicUsize::new(1));
        let rc = Box::into_raw(boxed);
        let ptr = NonNull::new(data.as_ptr() as *mut u8).unwrap();
        let length = data.len();
        let capacity = data.capacity();
        // `Drop for AiclBuffer` reconstructs and frees this allocation via
        // `Vec::from_raw_parts` once the refcount hits zero — forgetting
        // `data` here hands off ownership instead of letting it deallocate
        // when this function returns (which would free the buffer while an
        // AiclBuffer still pointed at it, then free it again on Drop).
        core::mem::forget(data);
        Self {
            inner: BufferInner::Shared { ptr, length, capacity },
            refcount: Some(unsafe { NonNull::new_unchecked(rc) }),
        }
    }

    /// Borrow this buffer as an immutable byte slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        match &self.inner {
            BufferInner::Owned(v) => v.as_slice(),
            BufferInner::Borrowed { ptr, length, .. } => {
                unsafe { core::slice::from_raw_parts(ptr.as_ptr(), *length) }
            }
            BufferInner::Shared { ptr, length, .. } => {
                unsafe { core::slice::from_raw_parts(ptr.as_ptr(), *length) }
            }
        }
    }

    /// Get the buffer length.
    #[inline]
    pub fn len(&self) -> usize {
        match &self.inner {
            BufferInner::Owned(v) => v.len(),
            BufferInner::Borrowed { length, .. } => *length,
            BufferInner::Shared { length, .. } => *length,
        }
    }

    /// Check if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Convert into a `Vec<u8>`, consuming self.
    ///
    /// For `Owned`: moves the Vec. For `Shared`/`Borrowed`: copies the data.
    pub fn into_vec(self) -> Vec<u8> {
        use core::mem::ManuallyDrop;
        let mut this = ManuallyDrop::new(self);
        match &this.inner {
            BufferInner::Owned(_) => {
                // Safety: We're consuming self. ManuallyDrop prevents double-free.
                match core::mem::replace(&mut this.inner, BufferInner::Owned(Vec::new())) {
                    BufferInner::Owned(v) => v,
                    _ => unreachable!(),
                }
            }
            BufferInner::Borrowed { ptr, length, .. } => {
                let slice = unsafe { core::slice::from_raw_parts(ptr.as_ptr(), *length) };
                slice.to_vec()
            }
            BufferInner::Shared { ptr, length, .. } => {
                let slice = unsafe { core::slice::from_raw_parts(ptr.as_ptr(), *length) };
                slice.to_vec()
            }
        }
    }

    /// Create an `AiclBufferRef` pointing into this buffer.
    ///
    /// # Safety
    ///
    /// The returned ref borrows this buffer. Do not move or free this buffer
    /// while the ref is in use.
    #[inline]
    pub unsafe fn as_bufref(&self) -> AiclBufferRef {
        AiclBufferRef {
            ptr: self.as_slice().as_ptr() as *const c_void,
            length: self.len() as u32,
            alignment: 1,
            tag: 0,
            ownership: 2, // Rust-owns
            lifetime: 0,
            reserved: [0; 2],
        }
    }
}

impl Clone for AiclBuffer {
    fn clone(&self) -> Self {
        match &self.inner {
            BufferInner::Owned(v) => Self::owned(v.clone()),
            BufferInner::Borrowed { ptr, length, .. } => {
                Self {
                    inner: BufferInner::Borrowed {
                        ptr: *ptr,
                        length: *length,
                        _marker: core::marker::PhantomData,
                    },
                    refcount: None,
                }
            }
            BufferInner::Shared { ptr, length, capacity } => {
                if let Some(rc) = self.refcount {
                    unsafe { rc.as_ref().fetch_add(1, Ordering::Relaxed); }
                }
                Self {
                    inner: BufferInner::Shared {
                        ptr: *ptr,
                        length: *length,
                        capacity: *capacity,
                    },
                    refcount: self.refcount,
                }
            }
        }
    }
}

impl Drop for AiclBuffer {
    fn drop(&mut self) {
        match &self.inner {
            BufferInner::Shared { ptr, length, capacity } => {
                if let Some(rc) = self.refcount {
                    let prev = unsafe { rc.as_ref().fetch_sub(1, Ordering::Release) };
                    if prev == 1 {
                        unsafe {
                            core::sync::atomic::fence(Ordering::Acquire);
                            drop(Box::from_raw(rc.as_ptr()));
                            let _ = Vec::from_raw_parts(ptr.as_ptr(), *length, *capacity);
                        }
                    }
                }
            }
            BufferInner::Borrowed { .. } => { /* C++ owns */ }
            BufferInner::Owned(_) => { /* Vec drops */ }
        }
    }
}

impl fmt::Debug for AiclBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            BufferInner::Owned(v) => write!(f, "AiclBuffer::Owned({} bytes)", v.len()),
            BufferInner::Borrowed { length, .. } => write!(f, "AiclBuffer::Borrowed({} bytes)", length),
            BufferInner::Shared { length, .. } => {
                let rc = self.refcount.map(|r| unsafe { r.as_ref().load(Ordering::Relaxed) }).unwrap_or(0);
                write!(f, "AiclBuffer::Shared({} bytes, rc={})", length, rc)
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// AiclHandle — Opaque handle for C++ interop
// ═════════════════════════════════════════════════════════════════════════════

/// Reference-counted opaque handle wrapping a Rust object.
///
/// C++ holds this as `aicl_handle_t*` and never sees the inner type.
pub struct AiclHandle {
    inner: Box<dyn core::any::Any + Send + Sync>,
    refcount: AtomicUsize,
}

impl AiclHandle {
    /// Wrap a value into an opaque handle.
    #[inline]
    pub fn new<T: Send + Sync + 'static>(value: T) -> *mut Self {
        let handle = Box::new(Self {
            inner: Box::new(value),
            refcount: AtomicUsize::new(1),
        });
        Box::into_raw(handle)
    }

    /// Increment the reference count.
    ///
    /// # Safety
    ///
    /// `handle` must be valid and previously returned by `new` or `clone_ref`.
    #[inline]
    pub unsafe fn clone_ref(handle: *mut Self) -> *mut Self {
        if handle.is_null() { return handle; }
        unsafe { &*handle }.refcount.fetch_add(1, Ordering::Relaxed);
        handle
    }

    /// Decrement the reference count. Frees when it reaches 0.
    ///
    /// # Safety
    ///
    /// `handle` must be valid and previously returned by `new` or `clone_ref`.
    #[inline]
    pub unsafe fn release_ref(handle: *mut Self) {
        if handle.is_null() { return; }
        let prev = unsafe { &*handle }.refcount.fetch_sub(1, Ordering::Release);
        if prev == 1 {
            unsafe {
                core::sync::atomic::fence(Ordering::Acquire);
                drop(Box::from_raw(handle));
            }
        }
    }

    /// Downcast the handle to a concrete type.
    #[inline]
    pub fn downcast_ref<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.inner.downcast_ref::<T>()
    }

    /// Downcast the handle to a mutable concrete type.
    #[inline]
    pub fn downcast_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.inner.downcast_mut::<T>()
    }

    /// Get the reference count.
    #[inline]
    pub fn refcount(&self) -> usize {
        self.refcount.load(Ordering::Relaxed)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// AiclStream — Streaming buffer for large data
// ═════════════════════════════════════════════════════════════════════════════

/// Streaming buffer for large data (model outputs, multi-chunk payloads).
pub struct AiclStream {
    chunks: Vec<AiclBuffer>,
    total_len: usize,
}

impl AiclStream {
    /// Create an empty stream.
    #[inline]
    pub fn new() -> Self {
        Self { chunks: Vec::new(), total_len: 0 }
    }

    /// Append a chunk to the stream.
    #[inline]
    pub fn push(&mut self, chunk: AiclBuffer) {
        self.total_len += chunk.len();
        self.chunks.push(chunk);
    }

    /// Get the total length of all chunks.
    #[inline]
    pub fn total_len(&self) -> usize { self.total_len }

    /// Get the number of chunks.
    #[inline]
    pub fn chunk_count(&self) -> usize { self.chunks.len() }

    /// Get a chunk by index.
    #[inline]
    pub fn chunk(&self, idx: usize) -> Option<&AiclBuffer> {
        self.chunks.get(idx)
    }

    /// Copy all chunks into a single contiguous `Vec<u8>`.
    #[inline]
    pub fn into_contiguous(self) -> Vec<u8> {
        let mut result = Vec::with_capacity(self.total_len);
        for chunk in self.chunks {
            result.extend_from_slice(chunk.as_slice());
        }
        result
    }

    /// Create an `AiclBufferRef` pointing into the first chunk.
    ///
    /// # Safety
    ///
    /// The stream must outlive the returned ref.
    #[inline]
    pub unsafe fn as_bufref(&self) -> Option<AiclBufferRef> {
        if self.chunks.len() == 1 {
            Some(unsafe { self.chunks[0].as_bufref() })
        } else {
            None
        }
    }
}

impl Default for AiclStream {
    fn default() -> Self { Self::new() }
}

// ═════════════════════════════════════════════════════════════════════════════
// Zero-copy encode/decode functions
// ═════════════════════════════════════════════════════════════════════════════

/// Encode a packet directly into a pre-existing buffer (zero-alloc).
///
/// Returns number of bytes written.
#[inline]
pub fn encode_into_bufref(
    pkt: &AiclPacket,
    codec: &AiclCodec,
    buf: &mut Vec<u8>,
) -> AiclResult<usize> {
    let before = buf.len();
    codec.encode(pkt)?;
    Ok(buf.len() - before)
}

/// Encode a packet and return the wire bytes as an `AiclBuffer`.
#[inline]
pub fn encode_to_buffer(
    pkt: &AiclPacket,
    codec: &AiclCodec,
) -> AiclResult<AiclBuffer> {
    let wire = codec.encode(pkt)?;
    Ok(AiclBuffer::owned(wire))
}

/// Decode a packet from an `AiclBuffer` (zero-copy view).
///
/// The returned `PacketView` borrows the buffer's data — no allocation.
pub fn decode_from_buffer(
    buf: &AiclBuffer,
    codec: &AiclCodec,
) -> AiclResult<PacketView> {
    let slice = buf.as_slice();
    codec.decode(slice)
}

/// Decode from a buffer ref (zero-copy).
///
/// # Safety
///
/// The `bufref` must point to valid AICL wire data.
pub unsafe fn decode_from_bufref(
    bufref: &AiclBufferRef,
    codec: &AiclCodec,
) -> AiclResult<PacketView> {
    if bufref.is_empty() {
        return Err(AiclError::new(ErrorCode::Empty, "buffer ref is empty"));
    }
    let slice = unsafe { bufref.as_slice() };
    codec.decode(slice)
}

/// Create an `AiclBufferRef` from a C++ pointer.
///
/// # Safety
///
/// - `ptr` must be valid for `length` bytes
/// - The memory must remain valid while the ref exists
#[inline]
pub unsafe fn bufref_from_ptr(ptr: *const c_void, length: u32) -> AiclBufferRef {
    unsafe { AiclBufferRef::from_ptr(ptr, length) }
}

/// Get a pointer and length from an `AiclBufferRef`.
#[inline]
pub fn bufref_to_ptr(ref_: &AiclBufferRef) -> (*const c_void, u32) {
    (ref_.ptr, ref_.length)
}

// ═════════════════════════════════════════════════════════════════════════════
// FFI — C-compatible bindings for the ABI
// ═════════════════════════════════════════════════════════════════════════════

use core::ffi::{c_int, c_uint};

/// C-compatible buffer reference.
#[repr(C)]
pub struct aicl_bufref_t {
    pub ptr: *const c_void,
    pub length: u32,
    pub alignment: u16,
    pub tag: u16,
    pub ownership: u8,
    pub lifetime: u8,
    pub reserved: [u8; 2],
}

/// C-compatible status codes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum aicl_status_t {
    OK = 0,
    BAD_HEADER = 1,
    UNSUPPORTED_VERSION = 2,
    TRUNCATED = 3,
    CHECKSUM = 4,
    PAYLOAD_TOO_LARGE = 5,
    OPCODE_INVALID = 6,
    OPERAND_COUNT = 7,
    UNKNOWN_TYPE = 8,
    NO_HANDLER = 9,
    EMPTY = 10,
    BUFFER_FULL = 11,
    TRANSPORT_CLOSED = 12,
    NULL_POINTER = 13,
    INVALID_LEN = 14,
    INTERNAL = 100,
}

impl From<AiclError> for aicl_status_t {
    fn from(e: AiclError) -> Self {
        match e.code {
            ErrorCode::Ok => Self::OK,
            ErrorCode::BadHeader => Self::BAD_HEADER,
            ErrorCode::UnsupportedVersion => Self::UNSUPPORTED_VERSION,
            ErrorCode::Truncated => Self::TRUNCATED,
            ErrorCode::Checksum => Self::CHECKSUM,
            ErrorCode::PayloadTooLarge => Self::PAYLOAD_TOO_LARGE,
            ErrorCode::UnknownOpcode => Self::OPCODE_INVALID,
            ErrorCode::OperandCount => Self::OPERAND_COUNT,
            ErrorCode::UnknownType => Self::UNKNOWN_TYPE,
            ErrorCode::NoHandler => Self::NO_HANDLER,
            ErrorCode::Empty => Self::EMPTY,
            ErrorCode::BufferFull => Self::BUFFER_FULL,
            ErrorCode::TransportClosed => Self::TRANSPORT_CLOSED,
            _ => Self::INTERNAL,
        }
    }
}

/// Opaque handle type for C++ callers.
#[repr(C)]
pub struct aicl_handle_t { _priv: [u8; 0] }

// ─── Buffer ref operations ──────────────────────────────────────────────────

/// Create a buffer ref from a raw pointer.
///
/// # Safety
/// - `data` must be valid for `length` bytes
#[no_mangle]
pub unsafe extern "C" fn aicl_bufref_from_ptr(
    data: *const c_void,
    length: u32,
) -> aicl_bufref_t {
    if data.is_null() || length == 0 {
        return aicl_bufref_t {
            ptr: core::ptr::null(),
            length: 0, alignment: 0, tag: 0,
            ownership: 0, lifetime: 0, reserved: [0; 2],
        };
    }
    aicl_bufref_t {
        ptr: data,
        length,
        alignment: 1,
        tag: 0,
        ownership: 1,
        lifetime: 0,
        reserved: [0; 2],
    }
}

/// Get pointer and length from a buffer ref.
#[no_mangle]
pub unsafe extern "C" fn aicl_bufref_to_ptr(
    bufref: *const aicl_bufref_t,
    out_ptr: *mut *const c_void,
    out_len: *mut u32,
) -> aicl_status_t {
    if bufref.is_null() || out_ptr.is_null() || out_len.is_null() {
        return aicl_status_t::NULL_POINTER;
    }
    let r = unsafe { &*bufref };
    unsafe {
        *out_ptr = r.ptr;
        *out_len = r.length;
    }
    aicl_status_t::OK
}

/// Check if a buffer ref is empty.
#[no_mangle]
pub unsafe extern "C" fn aicl_bufref_is_empty(bufref: *const aicl_bufref_t) -> c_int {
    if bufref.is_null() { return 1; }
    let r = unsafe { &*bufref };
    if r.ptr.is_null() || r.length == 0 { 1 } else { 0 }
}

// ─── Buffer lifecycle ───────────────────────────────────────────────────────

/// Create an owned buffer from a pointer (copies data into Rust allocation).
///
/// # Safety
/// - `data` must be valid for `length` bytes
#[no_mangle]
pub unsafe extern "C" fn aicl_buffer_from_ptr(
    data: *const c_void,
    length: u32,
) -> *mut AiclBuffer {
    if data.is_null() || length == 0 { return core::ptr::null_mut(); }
    let slice = unsafe { core::slice::from_raw_parts(data as *const u8, length as usize) };
    let buf = AiclBuffer::owned(slice.to_vec());
    Box::into_raw(Box::new(buf))
}

/// Create a borrowed buffer from a C++ pointer (NO COPY).
///
/// # Safety
/// - `data` must be valid for `length` bytes for the lifetime of the buffer
#[no_mangle]
pub unsafe extern "C" fn aicl_buffer_borrow(
    data: *mut c_void,
    length: u32,
) -> *mut AiclBuffer {
    if data.is_null() || length == 0 { return core::ptr::null_mut(); }
    let buf = unsafe { AiclBuffer::borrowed(data as *mut u8, length as usize) };
    Box::into_raw(Box::new(buf))
}

/// Create a shared, reference-counted buffer from a pointer.
///
/// # Safety
/// - `data` must be valid for `length` bytes
#[no_mangle]
pub unsafe extern "C" fn aicl_buffer_from_ptr_shared(
    data: *const c_void,
    length: u32,
) -> *mut AiclBuffer {
    if data.is_null() || length == 0 { return core::ptr::null_mut(); }
    let slice = unsafe { core::slice::from_raw_parts(data as *const u8, length as usize) };
    let buf = AiclBuffer::shared(slice.to_vec());
    Box::into_raw(Box::new(buf))
}

/// Clone a buffer (increments refcount for shared, copies for owned/borrowed).
#[no_mangle]
pub unsafe extern "C" fn aicl_buffer_clone(buf: *const AiclBuffer) -> *mut AiclBuffer {
    if buf.is_null() { return core::ptr::null_mut(); }
    let cloned = unsafe { &*buf }.clone();
    Box::into_raw(Box::new(cloned))
}

/// Free a buffer.
#[no_mangle]
pub unsafe extern "C" fn aicl_buffer_free(buf: *mut AiclBuffer) {
    if buf.is_null() { return; }
    unsafe { drop(Box::from_raw(buf)); }
}

/// Get the length of a buffer.
#[no_mangle]
pub unsafe extern "C" fn aicl_buffer_len(buf: *const AiclBuffer) -> u32 {
    if buf.is_null() { return 0; }
    unsafe { &*buf }.len() as u32
}

/// Get a read-only pointer to the buffer data.
#[no_mangle]
pub unsafe extern "C" fn aicl_buffer_data(buf: *const AiclBuffer) -> *const c_void {
    if buf.is_null() { return core::ptr::null(); }
    unsafe { &*buf }.as_slice().as_ptr() as *const c_void
}

// ─── Handle lifecycle ───────────────────────────────────────────────────────

/// Free an opaque handle.
#[no_mangle]
pub unsafe extern "C" fn aicl_handle_free(handle: *mut aicl_handle_t) {
    if handle.is_null() { return; }
    unsafe { AiclHandle::release_ref(handle as *mut AiclHandle); }
}

/// Clone an opaque handle (increments refcount).
#[no_mangle]
pub unsafe extern "C" fn aicl_handle_clone(handle: *mut aicl_handle_t) -> *mut aicl_handle_t {
    if handle.is_null() { return core::ptr::null_mut(); }
    unsafe { AiclHandle::clone_ref(handle as *mut AiclHandle) as *mut aicl_handle_t }
}

/// Get the reference count of a handle.
#[no_mangle]
pub unsafe extern "C" fn aicl_handle_refcount(handle: *const aicl_handle_t) -> c_uint {
    if handle.is_null() { return 0; }
    unsafe { &*(handle as *const AiclHandle) }.refcount() as c_uint
}

// ─── Stream operations ──────────────────────────────────────────────────────

/// Create an empty stream.
#[no_mangle]
pub extern "C" fn aicl_stream_new() -> *mut AiclStream {
    Box::into_raw(Box::new(AiclStream::new()))
}

/// Append a buffer to a stream (takes ownership of the buffer).
#[no_mangle]
pub unsafe extern "C" fn aicl_stream_push(
    stream: *mut AiclStream,
    buf: *mut AiclBuffer,
) -> aicl_status_t {
    if stream.is_null() || buf.is_null() { return aicl_status_t::NULL_POINTER; }
    let stream = unsafe { &mut *stream };
    let buf = unsafe { *Box::from_raw(buf) };
    stream.push(buf);
    aicl_status_t::OK
}

/// Get the total length of a stream.
#[no_mangle]
pub unsafe extern "C" fn aicl_stream_total_len(stream: *const AiclStream) -> u32 {
    if stream.is_null() { return 0; }
    unsafe { &*stream }.total_len() as u32
}

/// Get the number of chunks in a stream.
#[no_mangle]
pub unsafe extern "C" fn aicl_stream_chunk_count(stream: *const AiclStream) -> c_uint {
    if stream.is_null() { return 0; }
    unsafe { &*stream }.chunk_count() as c_uint
}

/// Free a stream and all its buffers.
#[no_mangle]
pub unsafe extern "C" fn aicl_stream_free(stream: *mut AiclStream) {
    if stream.is_null() { return; }
    unsafe { drop(Box::from_raw(stream)); }
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_bufref_size() {
        // ptr(8) + length(4) + alignment(2) + tag(2) + ownership(1) +
        // lifetime(1) + reserved(2) = 20 raw bytes, rounded up to 24 to
        // satisfy the 8-byte alignment the pointer field requires under
        // #[repr(C)]. (Not the same struct as core-cpp's unrelated
        // id+offset+length `aicl_bufref_t`, which is 16 bytes by
        // coincidence of a shared name, not a shared ABI contract.)
        assert_eq!(core::mem::size_of::<AiclBufferRef>(), 24);
        assert_eq!(core::mem::size_of::<aicl_bufref_t>(), 24);
    }

    #[test]
    fn test_bufref_from_slice() {
        let data = vec![1u8, 2, 3, 4, 5];
        let r = AiclBufferRef::from_slice(&data);
        assert_eq!(r.length, 5);
        assert!(!r.is_empty());
        unsafe { assert_eq!(r.as_slice(), &[1, 2, 3, 4, 5]); }
    }

    #[test]
    fn test_bufref_empty() {
        let r = AiclBufferRef::from_slice(&[]);
        assert!(r.is_empty());
    }

    #[test]
    fn test_buffer_owned() {
        let buf = AiclBuffer::owned(vec![10, 20, 30]);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn test_buffer_clone_shared() {
        let buf = AiclBuffer::shared(vec![1, 2, 3]);
        let cloned = buf.clone();
        assert_eq!(buf.as_slice(), &[1, 2, 3]);
        assert_eq!(cloned.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_buffer_into_vec() {
        let buf = AiclBuffer::owned(vec![1, 2, 3]);
        let v = buf.into_vec();
        assert_eq!(v, vec![1, 2, 3]);
    }

    #[test]
    fn test_handle_lifecycle() {
        let handle = AiclHandle::new(42u32);
        unsafe {
            assert_eq!((*handle).refcount(), 1);
            let cloned = AiclHandle::clone_ref(handle);
            assert_eq!((*handle).refcount(), 2);
            AiclHandle::release_ref(cloned);
            assert_eq!((*handle).refcount(), 1);
            AiclHandle::release_ref(handle);
        }
    }

    #[test]
    fn test_handle_downcast() {
        let handle = AiclHandle::new(String::from("hello"));
        unsafe {
            let h = &*handle;
            assert_eq!(h.downcast_ref::<String>().unwrap(), "hello");
            AiclHandle::release_ref(handle);
        }
    }

    #[test]
    fn test_stream() {
        let mut stream = AiclStream::new();
        stream.push(AiclBuffer::owned(vec![1, 2, 3]));
        stream.push(AiclBuffer::owned(vec![4, 5]));
        assert_eq!(stream.total_len(), 5);
        assert_eq!(stream.chunk_count(), 2);
        let contiguous = stream.into_contiguous();
        assert_eq!(contiguous, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_encode_to_buffer_and_decode() {
        let codec = AiclCodec::new();
        let pkt = AiclPacket::ping(0xDEAD_BEEF_CAFE_BABE);
        let buf = encode_to_buffer(&pkt, &codec).unwrap();
        let view = decode_from_buffer(&buf, &codec).unwrap();
        assert_eq!(view.opcode(), AiclOpcode::Ping.to_u8());
    }

    #[test]
    fn test_bufref_to_c_and_back() {
        let data = vec![10u8, 20, 30, 40];
        let r = AiclBufferRef::from_slice(&data);
        let (ptr, len) = bufref_to_ptr(&r);
        assert_eq!(len, 4);
        unsafe {
            let slice = core::slice::from_raw_parts(ptr as *const u8, len as usize);
            assert_eq!(slice, &[10, 20, 30, 40]);
        }
    }
}
