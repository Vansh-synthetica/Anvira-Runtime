//! Error types for the AICL core.

use core::fmt;

/// AICL error code — u32 matching the ISA spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ErrorCode {
    Ok                      = 0x0000,
    BadHeader               = 0x0001,
    Truncated               = 0x0002,
    Checksum                = 0x0003,
    UnknownOpcode           = 0x0004,
    UnknownType             = 0x0005,
    OperandCount            = 0x0006,
    SchemaMismatch          = 0x0007,
    InvalidUtf8             = 0x0008,
    InvalidBool             = 0x0009,
    PayloadTooLarge         = 0x000A,
    VarintOverflow          = 0x000B,
    MapKeyType              = 0x000C,
    InvalidValue            = 0x000D,
    UnsupportedVersion      = 0x000E,
    CapabilityNotFound     = 0x000F,
    Timeout                 = 0x0010,
    Cancelled               = 0x0011,
    Backpressure            = 0x0012,
    BufferNotFound          = 0x0013,
    BufferAccessDenied      = 0x0014,
    StreamOutOfOrder       = 0x0015,
    ExtensionNotUnderstood  = 0x0016,
    CapabilityExists        = 0x0017,
    CapabilityTableFull     = 0x0018,
    InvalidInstruction      = 0x0019,
    SessionClosed           = 0x001A,
    ChannelClosed           = 0x001B,
    UnknownExtension        = 0x001C,
    ExtensionTooLarge       = 0x001D,
    NoHandler               = 0x001E,
    Empty                   = 0x001F,
    BufferFull              = 0x0020,
    TransportClosed         = 0x0021,
    CapabilityInvalid       = 0x0022,
    CapabilityOverflow      = 0x0023,
    TrailingBytes           = 0x0024,
    Internal                = 0x0064,
    Vendor(u32),
}

impl ErrorCode {
    #[inline]
    pub const fn code(&self) -> u32 {
        match self {
            Self::Ok => 0x0000, Self::BadHeader => 0x0001,
            Self::Truncated => 0x0002, Self::Checksum => 0x0003,
            Self::UnknownOpcode => 0x0004, Self::UnknownType => 0x0005,
            Self::OperandCount => 0x0006, Self::SchemaMismatch => 0x0007,
            Self::InvalidUtf8 => 0x0008, Self::InvalidBool => 0x0009,
            Self::PayloadTooLarge => 0x000A, Self::VarintOverflow => 0x000B,
            Self::MapKeyType => 0x000C, Self::InvalidValue => 0x000D,
            Self::UnsupportedVersion => 0x000E, Self::CapabilityNotFound => 0x000F,
            Self::Timeout => 0x0010, Self::Cancelled => 0x0011,
            Self::Backpressure => 0x0012, Self::BufferNotFound => 0x0013,
            Self::BufferAccessDenied => 0x0014, Self::StreamOutOfOrder => 0x0015,
            Self::ExtensionNotUnderstood => 0x0016, Self::CapabilityExists => 0x0017,
            Self::CapabilityTableFull => 0x0018, Self::InvalidInstruction => 0x0019,
            Self::SessionClosed => 0x001A, Self::ChannelClosed => 0x001B,
            Self::UnknownExtension => 0x001C, Self::ExtensionTooLarge => 0x001D,
            Self::NoHandler => 0x001E, Self::Empty => 0x001F,
            Self::BufferFull => 0x0020, Self::TransportClosed => 0x0021,
            Self::CapabilityInvalid => 0x0022, Self::CapabilityOverflow => 0x0023,
            Self::TrailingBytes => 0x0024, Self::Internal => 0x0064,
            Self::Vendor(v) => *v,
        }
    }

    #[inline]
    pub const fn is_ok(&self) -> bool { matches!(self, Self::Ok) }

    #[inline]
    pub const fn from_u32(v: u32) -> Self {
        match v {
            0x0000=>Self::Ok, 0x0001=>Self::BadHeader, 0x0002=>Self::Truncated,
            0x0003=>Self::Checksum, 0x0004=>Self::UnknownOpcode, 0x0005=>Self::UnknownType,
            0x0006=>Self::OperandCount, 0x0007=>Self::SchemaMismatch,
            0x0008=>Self::InvalidUtf8, 0x0009=>Self::InvalidBool,
            0x000A=>Self::PayloadTooLarge, 0x000B=>Self::VarintOverflow,
            0x000C=>Self::MapKeyType, 0x000D=>Self::InvalidValue,
            0x000E=>Self::UnsupportedVersion, 0x000F=>Self::CapabilityNotFound,
            0x0010=>Self::Timeout, 0x0011=>Self::Cancelled,
            0x0012=>Self::Backpressure, 0x0013=>Self::BufferNotFound,
            0x0014=>Self::BufferAccessDenied, 0x0015=>Self::StreamOutOfOrder,
            0x0016=>Self::ExtensionNotUnderstood, 0x0017=>Self::CapabilityExists,
            0x0018=>Self::CapabilityTableFull, 0x0019=>Self::InvalidInstruction,
            0x001A=>Self::SessionClosed, 0x001B=>Self::ChannelClosed,
            0x001C=>Self::UnknownExtension, 0x001D=>Self::ExtensionTooLarge,
            0x001E=>Self::NoHandler, 0x001F=>Self::Empty,
            0x0020=>Self::BufferFull, 0x0021=>Self::TransportClosed,
            0x0022=>Self::CapabilityInvalid, 0x0023=>Self::CapabilityOverflow,
            0x0024=>Self::TrailingBytes, 0x0064=>Self::Internal,
            v => Self::Vendor(v),
        }
    }
}


impl ErrorCode {
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Ok => "OK", Self::BadHeader => "BAD_HEADER",
            Self::Truncated => "TRUNCATED", Self::Checksum => "CHECKSUM",
            Self::UnknownOpcode => "UNKNOWN_OPCODE", Self::UnknownType => "UNKNOWN_TYPE",
            Self::OperandCount => "OPERAND_COUNT", Self::SchemaMismatch => "SCHEMA_MISMATCH",
            Self::InvalidUtf8 => "INVALID_UTF8", Self::InvalidBool => "INVALID_BOOL",
            Self::PayloadTooLarge => "PAYLOAD_TOO_LARGE",
            Self::VarintOverflow => "VARINT_OVERFLOW",
            Self::MapKeyType => "MAP_KEY_TYPE", Self::InvalidValue => "INVALID_VALUE",
            Self::UnsupportedVersion => "UNSUPPORTED_VERSION",
            Self::CapabilityNotFound => "CAPABILITY_NOT_FOUND",
            Self::Timeout => "TIMEOUT", Self::Cancelled => "CANCELLED",
            Self::Backpressure => "BACKPRESSURE",
            Self::BufferNotFound => "BUFFER_NOT_FOUND",
            Self::BufferAccessDenied => "BUFFER_ACCESS_DENIED",
            Self::StreamOutOfOrder => "STREAM_OUT_OF_ORDER",
            Self::ExtensionNotUnderstood => "EXTENSION_NOT_UNDERSTOOD",
            Self::CapabilityExists => "CAPABILITY_EXISTS",
            Self::CapabilityTableFull => "CAPABILITY_TABLE_FULL",
            Self::InvalidInstruction => "INVALID_INSTRUCTION",
            Self::SessionClosed => "SESSION_CLOSED",
            Self::ChannelClosed => "CHANNEL_CLOSED",
            Self::UnknownExtension => "UNKNOWN_EXTENSION",
            Self::ExtensionTooLarge => "EXTENSION_TOO_LARGE",
            Self::NoHandler => "NO_HANDLER", Self::Empty => "EMPTY",
            Self::BufferFull => "BUFFER_FULL", Self::TransportClosed => "TRANSPORT_CLOSED",
            Self::CapabilityInvalid => "CAPABILITY_INVALID",
            Self::CapabilityOverflow => "CAPABILITY_OVERFLOW",
            Self::TrailingBytes => "TRAILING_BYTES", Self::Internal => "INTERNAL",
            Self::Vendor(_) => "VENDOR",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [0x{:04X}]", self.name(), self.code())
    }
}

impl core::error::Error for ErrorCode {}

/// The canonical AICL error type.
#[derive(Debug, Clone)]
pub struct AiclError {
    pub code: ErrorCode,
    pub message: alloc::string::String,
    pub context: Option<alloc::string::String>,
}

impl AiclError {
    #[inline]
    pub fn new(code: ErrorCode, message: impl Into<alloc::string::String>) -> Self {
        Self { code, message: message.into(), context: None }
    }

    #[inline]
    pub fn with_context(mut self, ctx: impl Into<alloc::string::String>) -> Self {
        self.context = Some(ctx.into());
        self
    }

    #[inline]
    pub fn wire_code(&self) -> u32 { self.code.code() }
}

impl fmt::Display for AiclError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref ctx) = self.context {
            write!(f, "{}: {} ({})", self.code, self.message, ctx)
        } else {
            write!(f, "{}: {}", self.code, self.message)
        }
    }
}

impl core::error::Error for AiclError {}

/// Convenience result type for the AICL core.
pub type AiclResult<T> = Result<T, AiclError>;

