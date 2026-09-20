//! AiclSession — logical communication channel between two peers.
//!
//! A session manages codec state, sequencing, and transport to provide a
//! reliable, ordered channel abstraction over an arbitrary `Transport`.

use alloc::boxed::Box;
use alloc::vec::Vec;
use uuid::Uuid;

use crate::codec::AiclCodec;
use crate::error::{AiclError, AiclResult, ErrorCode};
use crate::packet::AiclPacket;
use crate::transport::Transport;

/// Session state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Session created, no handshake sent yet.
    Init,
    /// Local handshake (HELLO) has been sent; waiting for remote HELLO.
    HandshakeSent,
    /// Both sides have exchanged hellos; session is fully established.
    Established,
    /// Close has been initiated locally.
    Closing,
    /// Session is fully closed.
    Closed,
}

/// A session — a logical communication channel over a `Transport`.
pub struct AiclSession {
    /// 16-byte session identifier (UUIDv4).
    pub id: [u8; 16],
    /// Current session state.
    pub state: SessionState,
    codec: AiclCodec,
    /// Underlying transport (pub(crate) so the runtime can poll it).
    pub(crate) transport: Box<dyn Transport>,
    /// Sequence number for outgoing packets.
    seq_sent: u64,
    /// Last sequence number received from the remote peer.
    seq_received: u64,
}

impl AiclSession {
    /// Create a new session over the given transport.
    ///
    /// The session starts in [`SessionState::Init`].
    pub fn new(transport: Box<dyn Transport>) -> Self {
        let id = {
            let uuid = Uuid::new_v4();
            let mut b = [0u8; 16];
            b.copy_from_slice(uuid.as_bytes());
            b
        };
        Self {
            id,
            state: SessionState::Init,
            codec: AiclCodec::new(),
            transport,
            seq_sent: 0,
            seq_received: 0,
        }
    }

    /// Send a typed packet over this session.
    ///
    /// The packet is encoded using the session's codec and written to the
    /// underlying transport. Returns an error if the transport is closed or
    /// encoding fails.
    pub fn send_packet(&mut self, pkt: &AiclPacket) -> AiclResult<()> {
        if self.state == SessionState::Closed {
            return Err(AiclError::new(ErrorCode::SessionClosed,
                alloc::format!("session {:02X?} closed", &self.id[..4])));
        }
        if !self.transport.is_open() {
            return Err(AiclError::new(ErrorCode::TransportClosed,
                alloc::format!("transport closed for session {:02X?}", &self.id[..4])));
        }
        let wire = self.codec.encode(pkt)?;
        self.transport.send(&wire)?;
        self.seq_sent += 1;
        Ok(())
    }

    /// Receive the next packet from this session.
    ///
    /// Reads raw bytes from the transport, decodes them, and returns the
    /// resulting `AiclPacket`. Returns an error if the transport is closed or
    /// decoding fails.
    pub fn recv_packet(&mut self) -> AiclResult<AiclPacket> {
        if self.state == SessionState::Closed {
            return Err(AiclError::new(ErrorCode::SessionClosed,
                alloc::format!("session {:02X?} closed", &self.id[..4])));
        }
        if !self.transport.is_open() {
            return Err(AiclError::new(ErrorCode::TransportClosed,
                alloc::format!("transport closed for session {:02X?}", &self.id[..4])));
        }
        let mut wire = Vec::with_capacity(4096);
        let n = self.transport.recv(&mut wire)?;
        if n == 0 {
            return Err(AiclError::new(ErrorCode::Truncated,
                alloc::string::String::from("recv returned 0 bytes")));
        }
        wire.truncate(n);
        let view = self.codec.decode_owned(wire)?;
        let pkt = view.to_packet()?;
        self.seq_received += 1;
        Ok(pkt)
    }

    /// Initiate graceful session closure.
    ///
    /// This sets the session state to [`SessionState::Closing`] and closes
    /// the underlying transport.
    pub fn close(&mut self) -> AiclResult<()> {
        self.state = SessionState::Closing;
        self.transport.close()?;
        self.state = SessionState::Closed;
        Ok(())
    }

    /// True if the session is open (Init, HandshakeSent, or Established).
    #[inline]
    pub fn is_open(&self) -> bool {
        matches!(self.state, SessionState::Init | SessionState::HandshakeSent | SessionState::Established)
    }
}
