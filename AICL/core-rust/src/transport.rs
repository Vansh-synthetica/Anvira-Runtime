//! Transport — abstract interface for sending/receiving bytes.

use alloc::vec::Vec;

use crate::error::AiclResult;

/// Endpoint address (transport-specific).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Endpoint {
    /// Opaque bytes — depends on the transport kind.
    pub bytes: Vec<u8>,
}

impl Endpoint {
    /// Build a new endpoint from bytes.
    pub fn from_bytes(b: Vec<u8>) -> Self { Self { bytes: b } }

    /// Build from a human-readable string.
    pub fn from_str(s: &str) -> Self { Self { bytes: s.as_bytes().to_vec() } }
}

/// A transport implementation.
pub trait Transport: Send {
    /// Send a packet buffer.
    fn send(&mut self, bytes: &[u8]) -> AiclResult<usize>;

    /// Receive a packet buffer into `out`. Returns the number of bytes written.
    fn recv(&mut self, out: &mut Vec<u8>) -> AiclResult<usize>;

    /// Close the transport.
    fn close(&mut self) -> AiclResult<()>;

    /// True if the transport is currently open.
    fn is_open(&self) -> bool;
}

// ─────────────────────────────────────────────────────────────────────────────
// In-process channel
// ─────────────────────────────────────────────────────────────────────────────

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, Ordering};

/// A spinlock-guarded queue, shared between both ends of an `InProcChannel`
/// pair. Dependency-free (no `std::sync::Mutex`) to keep this crate's
/// `no_std`-capable design intact.
struct LockedQueue {
    lock: AtomicBool,
    queue: UnsafeCell<VecDeque<Vec<u8>>>,
}

// Safety: access to `queue` is only ever made while `lock` is held.
unsafe impl Sync for LockedQueue {}

impl LockedQueue {
    fn new(cap: usize) -> Self {
        Self { lock: AtomicBool::new(false), queue: UnsafeCell::new(VecDeque::with_capacity(cap)) }
    }

    fn with_locked<R>(&self, f: impl FnOnce(&mut VecDeque<Vec<u8>>) -> R) -> R {
        while self.lock.compare_exchange_weak(
            false, true, Ordering::Acquire, Ordering::Relaxed,
        ).is_err() {
            core::hint::spin_loop();
        }
        let result = f(unsafe { &mut *self.queue.get() });
        self.lock.store(false, Ordering::Release);
        result
    }
}

/// A bounded in-process channel between two endpoints.
pub struct InProcChannel {
    /// Queue this end writes to (the other end's `incoming`).
    outgoing: Arc<LockedQueue>,
    /// Queue this end reads from (the other end's `outgoing`).
    incoming: Arc<LockedQueue>,
    cap: usize,
    open: AtomicBool,
}

impl InProcChannel {
    /// Create a pair of channels (a, b) where `a.send` -> `b.recv` and vice-versa.
    pub fn pair(cap: usize) -> (Self, Self) {
        let a_to_b = Arc::new(LockedQueue::new(cap));
        let b_to_a = Arc::new(LockedQueue::new(cap));
        let a = Self {
            outgoing: a_to_b.clone(), incoming: b_to_a.clone(),
            cap, open: AtomicBool::new(true),
        };
        let b = Self {
            outgoing: b_to_a, incoming: a_to_b,
            cap, open: AtomicBool::new(true),
        };
        (a, b)
    }

    /// Push bytes to the outgoing queue (readable by the paired endpoint).
    pub fn push(&mut self, bytes: Vec<u8>) -> bool {
        self.outgoing.with_locked(|q| {
            if q.len() >= self.cap { return false; }
            q.push_back(bytes);
            true
        })
    }

    /// Pop bytes sent by the paired endpoint.
    pub fn pop(&mut self) -> Option<Vec<u8>> {
        self.incoming.with_locked(|q| q.pop_front())
    }
}

impl Transport for InProcChannel {
    fn send(&mut self, bytes: &[u8]) -> AiclResult<usize> {
        if !self.open.load(Ordering::Relaxed) {
            return Err(crate::error::AiclError::new(
                crate::error::ErrorCode::TransportClosed, "channel closed"));
        }
        let sent = self.outgoing.with_locked(|q| {
            if q.len() >= self.cap { return false; }
            q.push_back(bytes.to_vec());
            true
        });
        if !sent {
            return Err(crate::error::AiclError::new(
                crate::error::ErrorCode::BufferFull, "channel full"));
        }
        Ok(bytes.len())
    }

    fn recv(&mut self, out: &mut Vec<u8>) -> AiclResult<usize> {
        match self.incoming.with_locked(|q| q.pop_front()) {
            Some(b) => {
                let n = b.len();
                out.extend_from_slice(&b);
                Ok(n)
            }
            None => Err(crate::error::AiclError::new(
                crate::error::ErrorCode::Empty, "queue empty")),
        }
    }

    fn close(&mut self) -> AiclResult<()> {
        self.open.store(false, Ordering::Relaxed);
        Ok(())
    }

    fn is_open(&self) -> bool { self.open.load(Ordering::Relaxed) }
}
