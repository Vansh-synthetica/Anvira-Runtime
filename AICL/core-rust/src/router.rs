//! Router — distributes incoming calls to handlers.
//!
//! Two strategies are supported:
//!  - **Round-robin** ([`Router::round_robin`])
//!  - **Direct-by-capability** ([`Router::direct`])
//!
//! Handlers are added with `add_handler(opcode, handler)` or
//! `add_capability_handler(ref, handler)`. The router chooses based on
//! the opcode/operand type.

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::error::AiclResult;
use crate::opcode::AiclOpcode;
use crate::packet::AiclPacket;

/// A handler closure type. It receives the call packet and returns
/// a result packet.
pub type Handler = alloc::boxed::Box<dyn FnMut(AiclPacket) -> AiclResult<AiclPacket> + Send>;

/// Routing strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingStrategy {
    /// Distribute calls evenly across handlers.
    RoundRobin,
    /// Direct to a specific handler by capability reference.
    Direct,
}

/// Router — stateful distribution of calls.
pub struct Router {
    strategy: RoutingStrategy,
    handlers: Vec<Handler>,
    cursor: AtomicUsize,
    /// Capability -> handler index map (Direct).
    cap_table: Vec<u16>,
}

impl Router {
    /// Construct a round-robin router.
    pub fn round_robin() -> Self {
        Self {
            strategy: RoutingStrategy::RoundRobin,
            handlers: Vec::new(),
            cursor: AtomicUsize::new(0),
            cap_table: Vec::new(),
        }
    }

    /// Construct a direct router.
    pub fn direct() -> Self {
        Self {
            strategy: RoutingStrategy::Direct,
            handlers: Vec::new(),
            cursor: AtomicUsize::new(0),
            cap_table: Vec::new(),
        }
    }

    /// Add a round-robin handler.
    pub fn add_handler(&mut self, h: Handler) -> usize {
        let idx = self.handlers.len();
        self.handlers.push(h);
        self.cap_table.push(0);
        idx
    }

    /// Add a direct (by capability) handler.
    pub fn add_capability_handler(&mut self, cap_ref: u16, h: Handler) -> usize {
        let idx = self.handlers.len();
        self.handlers.push(h);
        self.cap_table.push(cap_ref);
        idx
    }

    /// Return the number of registered handlers.
    pub fn len(&self) -> usize { self.handlers.len() }

    /// True if no handlers are registered.
    pub fn is_empty(&self) -> bool { self.handlers.is_empty() }

    /// Pick a handler for a packet.
    pub fn pick(&self, pkt: &AiclPacket) -> Option<usize> {
        if self.handlers.is_empty() { return None; }
        match self.strategy {
            RoutingStrategy::RoundRobin => {
                let i = self.cursor.fetch_add(1, Ordering::Relaxed) % self.handlers.len();
                Some(i)
            }
            RoutingStrategy::Direct => {
                // Look for a REF operand containing the cap index.
                for op in &pkt.operands {
                    if let crate::operand::Operand::Ref(cap) = op {
                        for (i, c) in self.cap_table.iter().enumerate() {
                            if c == cap { return Some(i); }
                        }
                    }
                }
                None
            }
        }
    }

    /// Dispatch a packet to the appropriate handler.
    pub fn dispatch(&mut self, pkt: AiclPacket) -> AiclResult<AiclPacket> {
        let idx = self.pick(&pkt)
            .ok_or_else(|| crate::error::AiclError::new(
                crate::error::ErrorCode::NoHandler,
                alloc::format!("no handler for opcode {:?}", pkt.opcode)))?;
        let mut h = self.handlers.remove(idx);
        let result = (h.as_mut())(pkt);
        self.handlers.insert(idx, h);
        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Connection
// ─────────────────────────────────────────────────────────────────────────────

/// A connection endpoint. Holds a queue of incoming packets and a router.
pub struct Connection {
    router: Router,
    incoming: VecDeque<AiclPacket>,
    max_inflight: usize,
}

impl Connection {
    /// Build a new connection with the given routing strategy.
    pub fn new(router: Router) -> Self {
        Self {
            router,
            incoming: VecDeque::new(),
            max_inflight: 1024,
        }
    }

    /// Enqueue an incoming packet.
    pub fn enqueue(&mut self, pkt: AiclPacket) -> Result<(), String> {
        if self.incoming.len() >= self.max_inflight {
            return Err(String::from("queue full"));
        }
        self.incoming.push_back(pkt);
        Ok(())
    }

    /// Drain and dispatch queued packets, calling `on_response` for each.
    pub fn drain(&mut self, mut on_response: impl FnMut(AiclPacket)) -> AiclResult<usize> {
        let mut n = 0;
        while let Some(pkt) = self.incoming.pop_front() {
            let result = self.router.dispatch(pkt)?;
            on_response(result);
            n += 1;
        }
        Ok(n)
    }
}
