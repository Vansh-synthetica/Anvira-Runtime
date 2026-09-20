//! FFI — C-compatible bindings for AICL.

use core::ffi::{c_char, c_uint, c_void};

use crate::codec::AiclCodec;
use crate::error::ErrorCode;
use crate::packet::AiclPacket;

/// Opaque context handle (C callers see this).
#[repr(C)]
pub struct aicl_ctx { _priv: [u8; 0] }

/// Opaque packet-view handle (C callers see this).
#[repr(C)]
pub struct aicl_pkt_view { _priv: [u8; 0] }

/// C error codes — mirror ErrorCode.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum aicl_error_code {
    NONE = 0,
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
    CAPABILITY_INVALID = 13,
    CAPABILITY_OVERFLOW = 14,
    INVALID_BOOL = 15,
    INVALID_UTF8 = 16,
    VARINT_OVERFLOW = 17,
    TRAILING_BYTES = 18,
    INTERNAL = 100,
}

impl From<ErrorCode> for aicl_error_code {
    fn from(code: ErrorCode) -> Self {
        match code {
            ErrorCode::Ok => Self::NONE,
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
            ErrorCode::CapabilityInvalid => Self::CAPABILITY_INVALID,
            ErrorCode::CapabilityOverflow => Self::CAPABILITY_OVERFLOW,
            ErrorCode::InvalidBool => Self::INVALID_BOOL,
            ErrorCode::InvalidUtf8 => Self::INVALID_UTF8,
            ErrorCode::VarintOverflow => Self::VARINT_OVERFLOW,
            ErrorCode::TrailingBytes => Self::TRAILING_BYTES,
            ErrorCode::Internal => Self::INTERNAL,
            _ => Self::INTERNAL,
        }
    }
}

/// Allocate a new AICL context.
#[no_mangle]
pub unsafe extern "C" fn aicl_context_new() -> *mut aicl_ctx {
    let codec = AiclCodec::new();
    let leaked = alloc::boxed::Box::leak(alloc::boxed::Box::new(codec));
    leaked as *mut AiclCodec as *mut aicl_ctx
}

/// Free a context.
#[no_mangle]
pub unsafe extern "C" fn aicl_context_free(ctx: *mut aicl_ctx) {
    if ctx.is_null() { return; }
    unsafe { drop(alloc::boxed::Box::from_raw(ctx as *mut AiclCodec)); }
}

/// Decode a packet view. Returns error code.
#[no_mangle]
pub unsafe extern "C" fn aicl_decode(
    ctx: *mut aicl_ctx,
    data: *const c_void,
    len: usize,
    view_out: *mut *mut aicl_pkt_view,
) -> aicl_error_code {
    if ctx.is_null() || data.is_null() || view_out.is_null() {
        return aicl_error_code::INTERNAL;
    }
    let codec = unsafe { &*(ctx as *const AiclCodec) };
    let slice = unsafe { core::slice::from_raw_parts(data as *const u8, len) };

    match codec.decode(slice) {
        Ok(view) => {
            let leaked = alloc::boxed::Box::leak(alloc::boxed::Box::new(view));
            unsafe { *view_out = leaked as *mut _ as *mut aicl_pkt_view };
            aicl_error_code::NONE
        }
        Err(e) => aicl_error_code::from(e.code),
    }
}

/// Get the opcode byte from a view.
#[no_mangle]
pub unsafe extern "C" fn aicl_pkt_view_opcode(v: *const aicl_pkt_view) -> c_uint {
    if v.is_null() { return 0xFF; }
    let view = unsafe { &*(v as *const crate::codec::PacketView) };
    view.opcode() as c_uint
}

/// Get the flags field from a view.
#[no_mangle]
pub unsafe extern "C" fn aicl_pkt_view_flags(v: *const aicl_pkt_view) -> c_uint {
    if v.is_null() { return 0; }
    let view = unsafe { &*(v as *const crate::codec::PacketView) };
    view.flags() as c_uint
}

/// Get the number of operands in a view.
#[no_mangle]
pub unsafe extern "C" fn aicl_pkt_view_operand_count(v: *const aicl_pkt_view) -> c_uint {
    if v.is_null() { return 0; }
    let view = unsafe { &*(v as *const crate::codec::PacketView) };
    view.operand_count().unwrap_or(0) as c_uint
}

/// Free a packet view.
#[no_mangle]
pub unsafe extern "C" fn aicl_pkt_view_free(v: *mut aicl_pkt_view) {
    if v.is_null() { return; }
    unsafe { drop(alloc::boxed::Box::from_raw(v as *mut crate::codec::PacketView)); }
}

/// Return the library version as a static C string.
#[no_mangle]
pub extern "C" fn aicl_version() -> *const c_char {
    "1.0.0\0".as_ptr() as *const c_char
}

/// Return the ISA version.
#[no_mangle]
pub extern "C" fn aicl_isa_version() -> c_uint {
    crate::opcode::ISA_VERSION as c_uint
}

