//! aicl-core — AICL Native Rust Core
//!
//! Canonical implementation of AICL-ISA v1.0. See the [ISA spec](../../protocol/AICL-ISA.md).
//!
//! ## Design constraints
//!
//! - No Python, HTTP, REST, `/v1`, JSON wire, or localhost.
//! - Pure Rust, `no_std` capable (with `alloc`).
//! - Deterministic, bounded allocations.
//! - Zero-copy parsing where realistic.
//! - Panic-free: every public API returns `Result<_, AiclError>`.

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::pedantic)]

extern crate alloc;

pub mod capability;
pub mod codec;
pub mod error;
pub mod ffi;
pub mod opcode;
pub mod operand;
pub mod packet;
pub mod router;
pub mod safety;
pub mod endpoint;pub mod gate;pub mod instruction;pub mod runtime;
pub mod session;
pub mod transport;
pub mod local_path;
pub mod abi;

// Public re-exports — the stable API surface.
pub use abi::{
    AiclBufferRef, AiclBuffer, AiclHandle, AiclStream,
    encode_into_bufref, encode_to_buffer,
    decode_from_bufref, decode_from_buffer,
    bufref_from_ptr, bufref_to_ptr,
};
pub use abi::{
    aicl_bufref_t, aicl_status_t, aicl_handle_t,
    aicl_bufref_from_ptr, aicl_bufref_to_ptr, aicl_bufref_is_empty,
    aicl_buffer_from_ptr, aicl_buffer_borrow, aicl_buffer_from_ptr_shared,
    aicl_buffer_clone, aicl_buffer_free, aicl_buffer_len, aicl_buffer_data,
    aicl_handle_free, aicl_handle_clone, aicl_handle_refcount,
    aicl_stream_new, aicl_stream_push, aicl_stream_total_len,
    aicl_stream_chunk_count, aicl_stream_free,
};
pub use capability::{Capability, CapabilityKind, CapabilityTable, MAX_CAPABILITY};
pub use codec::{AiclCodec, PacketView};
pub mod shm;
pub use error::{AiclError, AiclResult, ErrorCode};
pub use ffi::{
    aicl_context_free, aicl_context_new, aicl_decode, aicl_isa_version,
    aicl_pkt_view_free, aicl_pkt_view_opcode, aicl_pkt_view_flags,
    aicl_pkt_view_operand_count, aicl_version,
    aicl_ctx, aicl_pkt_view, aicl_error_code,
};
pub use opcode::{AiclFlags, AiclOpcode, ISA_VERSION};
pub use operand::{BufRef, Operand, OperandType};
pub use packet::{AiclPacket, AiclPacketBuilder};
pub use router::{Connection, Handler, RoutingStrategy, Router};
pub use safety::SafetyPolicy;
pub use transport::{Endpoint, InProcChannel, Transport};pub use endpoint::{AiclEndpoint, AiclEndpointRegistry};pub use gate::{AccessDecision, AiclGate};pub use instruction::AiclInstruction;pub use runtime::AiclRuntime;pub use session::{AiclSession, SessionState};
pub use local_path::{
    LocalChannel, LocalRingBuffer, LocalNonce,
    BufferPool, DispatchTable, ZeroCopyCodec,
    fast_ping, fast_pong, fast_classify, fast_generate,
};

#[cfg(test)]
mod tests;


