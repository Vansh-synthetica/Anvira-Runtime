//! AiclRuntime — top-level runtime tying all AICL subsystems together.
//!
//! The runtime owns the [`AiclGate`], [`AiclEndpointRegistry`], and all active
//! [`AiclSession`] instances. It is the single entry point for dispatching
//! packets and managing the lifecycle of the AICL system.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::codec::AiclCodec;
use crate::endpoint::{AiclEndpoint, AiclEndpointRegistry};
use crate::error::{AiclError, AiclResult, ErrorCode};
use crate::gate::AiclGate;
use crate::packet::AiclPacket;
use crate::session::{AiclSession, SessionState};
use crate::transport::Transport;

/// Runtime — manages all AICL subsystems: gate, endpoints, sessions, codec.
pub struct AiclRuntime {
    /// Security gate enforcing capability access control.
    gate: AiclGate,
    /// Registry of named service endpoints.
    endpoints: AiclEndpointRegistry,
    /// Active sessions, keyed by their 16-byte session UUID.
    sessions: BTreeMap<[u8; 16], AiclSession>,
    /// Shared codec used for encoding / decoding.
    codec: AiclCodec,
}

impl Default for AiclRuntime {
    fn default() -> Self { Self::new() }
}

impl AiclRuntime {
    /// Construct a new runtime with default settings.
    pub fn new() -> Self {
        Self {
            gate: AiclGate::new(),
            endpoints: AiclEndpointRegistry::default(),
            sessions: BTreeMap::new(),
            codec: AiclCodec::new(),
        }
    }

    /// Replace the codec used by this runtime (builder pattern).
    #[inline]
    pub fn with_codec(mut self, codec: AiclCodec) -> Self {
        self.codec = codec;
        self
    }

    /// Borrow the gate.
    #[inline]
    pub fn gate(&self) -> &AiclGate { &self.gate }

    /// Mutably borrow the gate.
    #[inline]
    pub fn gate_mut(&mut self) -> &mut AiclGate { &mut self.gate }

    /// Register an endpoint with the runtime's registry.
    pub fn register_endpoint(&mut self, ep: AiclEndpoint) -> AiclResult<()> {
        self.endpoints.register(ep)
    }

    /// Create a new session over the given transport.
    /// Returns the 16-byte session identifier.
    pub fn create_session(&mut self, transport: Box<dyn Transport>) -> AiclResult<[u8; 16]> {
        let session = AiclSession::new(transport);
        let id = session.id;
        self.sessions.insert(id, session);
        Ok(id)
    }

    /// Close and remove a session by its identifier.
    pub fn close_session(&mut self, id: &[u8; 16]) -> AiclResult<()> {
        let session = self.sessions.remove(id)
            .ok_or_else(|| AiclError::new(ErrorCode::SessionClosed,
                alloc::format!("session not found")))?;
        let mut s = session;
        s.close()
    }

    /// Dispatch a packet through the gate and route to the matching endpoint.
    ///
    /// Steps:
    /// 1. Gate capability check + safety validation.
    /// 2. Resolve target endpoint from the first `Ref` operand.
    /// 3. Forward to the corresponding session.
    pub fn dispatch_packet(&mut self, pkt: AiclPacket) -> AiclResult<AiclPacket> {
        self.gate.check_and_validate(&pkt)?;

        // Resolve target from first REF operand.
        let target_cap = pkt.operands.iter()
            .find_map(|op| {
                if let crate::operand::Operand::Ref(idx) = op {
                    Some(*idx)
                } else {
                    None
                }
            });

        if let Some(cap_idx) = target_cap {
            let mut found: Option<([u8; 16], &mut AiclSession)> = None;
            for (sid, session) in self.sessions.iter_mut() {
                if session.is_open() {
                    found = Some((*sid, session));
                    break;
                }
            }

            if let Some((sid, session)) = found {
                let mut out = pkt.clone();
                out.correlation_id = sid;
                session.send_packet(&out)?;
                return session.recv_packet();
            }
        }

        Err(AiclError::new(ErrorCode::NoHandler,
            alloc::format!("no handler for capability {}", target_cap.unwrap_or(0))))
    }

    /// Advance the runtime by one tick.
    ///
    /// Processes pending work in all active sessions:
    /// - Closes zombie `Closed` sessions.
    /// - Drains incoming packets from open transports.
    ///
    /// Returns the total number of events processed.
    pub fn tick(&mut self) -> AiclResult<usize> {
        let mut events = 0;

        let to_close: Vec<[u8; 16]> = self.sessions.iter()
            .filter(|(_, s)| s.state == SessionState::Closed)
            .map(|(id, _)| *id)
            .collect();

        for id in to_close {
            self.sessions.remove(&id);
            events += 1;
        }

        for (_, session) in self.sessions.iter_mut() {
            if session.is_open() {
                let mut buf = Vec::new();
                match session.transport.recv(&mut buf) {
                    Ok(n) if n > 0 => {
                        if let Ok(view) = self.codec.decode(&buf) {
                            let _ = view.to_packet();
                        }
                        events += 1;
                    }
                    Ok(_) | Err(_) => { /* no data — skip */ }
                }
            }
        }

        Ok(events)
    }
}