//! Opcode taxonomy and flag definitions per AICL-ISA v1.0.

use core::fmt;

/// ISA version: major << 8 | minor. v1.0 = 0x0100.
pub const ISA_VERSION: u16 = 0x0100;
pub const ISA_VERSION_MAJOR: u8 = 1;
pub const ISA_VERSION_MINOR: u8 = 0;

/// Magic bytes at the start of every AICL header.
pub const MAGIC: &[u8; 4] = b"AICL";

/// Fixed header size in bytes (ISA spec: 56 bytes).
pub const HEADER_SIZE: usize = 56;

/// Maximum payload length (16 MiB).
pub const MAX_PAYLOAD_SIZE: usize = 16 * 1024 * 1024;

/// Maximum string length (64 KiB).
pub const MAX_STR_SIZE: usize = 64 * 1024;

/// Maximum bytes field length (16 MiB).
pub const MAX_BYTES_SIZE: usize = 16 * 1024 * 1024;

/// Maximum capability table size.
pub const MAX_CAPABILITY_TABLE: usize = 65_536;

#[cfg(test)]
const _: () = assert!(HEADER_SIZE == 56);

/// AICL opcodes matching the ISA spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum AiclOpcode {
    // System 0x00-0x0F
    Nop             = 0x00,
    Hello           = 0x01,
    Goodbye         = 0x02,
    Discover        = 0x03,
    Ping            = 0x04,
    Pong            = 0x05,
    Ack             = 0x06,
    Nak             = 0x07,
    Error           = 0x08,
    Cancel          = 0x09,
    Version         = 0x0A,
    // Call/Return 0x10-0x1F
    Call            = 0x10,
    Return          = 0x11,
    Stream          = 0x12,
    StreamEnd       = 0x13,
    // Memory 0x20-0x2F
    MemoryRead      = 0x20,
    MemoryWrite     = 0x21,
    MemoryDelete    = 0x22,
    IndexQuery      = 0x23,
    IndexUpsert     = 0x24,
    // Capability 0x30-0x3F
    CapabilityAdvertise = 0x30,
    CapabilityWithdraw  = 0x31,
    CapabilityQuery    = 0x32,
    // AI Primitives 0x40-0x4F
    ModelCall       = 0x40,
    ToolCall        = 0x41,
    Embed           = 0x42,
    Classify        = 0x43,
    Reason          = 0x44,
    Retrieve        = 0x45,
    Execute         = 0x46,
    // Tracing 0x50-0x5F
    Trace           = 0x50,
    Stats           = 0x51,
    Health          = 0x52,
    // Buffer 0x60-0x6F
    BufferRef       = 0x60,
    BufferRelease   = 0x61,
    BufferShare     = 0x62,
    /// Vendor-defined opcode in 0xF0-0xFF.
    Vendor(u8)      = 0xF0,
}

impl AiclOpcode {
    /// Map a raw byte to an `AiclOpcode`. Returns `None` for unassigned opcodes
    /// (0x0B–0x0F, 0x14–0x1F, 0x25–0x2F, 0x33–0x3F, 0x47–0x4F, 0x53–0x5F,
    /// 0x63–0x6F, 0x70–0xEF). Use `to_u8` to serialise; use this to validate received
    /// bytes per ISA §8 step 4.
    #[inline]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x00=>Some(Self::Nop), 0x01=>Some(Self::Hello), 0x02=>Some(Self::Goodbye),
            0x03=>Some(Self::Discover), 0x04=>Some(Self::Ping), 0x05=>Some(Self::Pong),
            0x06=>Some(Self::Ack), 0x07=>Some(Self::Nak), 0x08=>Some(Self::Error),
            0x09=>Some(Self::Cancel), 0x0A=>Some(Self::Version),
            0x10=>Some(Self::Call), 0x11=>Some(Self::Return),
            0x12=>Some(Self::Stream), 0x13=>Some(Self::StreamEnd),
            0x20=>Some(Self::MemoryRead), 0x21=>Some(Self::MemoryWrite),
            0x22=>Some(Self::MemoryDelete), 0x23=>Some(Self::IndexQuery),
            0x24=>Some(Self::IndexUpsert),
            0x30=>Some(Self::CapabilityAdvertise), 0x31=>Some(Self::CapabilityWithdraw),
            0x32=>Some(Self::CapabilityQuery),
            0x40=>Some(Self::ModelCall), 0x41=>Some(Self::ToolCall),
            0x42=>Some(Self::Embed), 0x43=>Some(Self::Classify),
            0x44=>Some(Self::Reason), 0x45=>Some(Self::Retrieve),
            0x46=>Some(Self::Execute),
            0x50=>Some(Self::Trace), 0x51=>Some(Self::Stats), 0x52=>Some(Self::Health),
            0x60=>Some(Self::BufferRef), 0x61=>Some(Self::BufferRelease),
            0x62=>Some(Self::BufferShare),
            v if v >= 0xF0 => Some(Self::Vendor(v)),
            _ => None,
        }
    }

    #[inline]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::Nop => 0x00, Self::Hello => 0x01, Self::Goodbye => 0x02,
            Self::Discover => 0x03, Self::Ping => 0x04, Self::Pong => 0x05,
            Self::Ack => 0x06, Self::Nak => 0x07, Self::Error => 0x08,
            Self::Cancel => 0x09, Self::Version => 0x0A,
            Self::Call => 0x10, Self::Return => 0x11,
            Self::Stream => 0x12, Self::StreamEnd => 0x13,
            Self::MemoryRead => 0x20, Self::MemoryWrite => 0x21,
            Self::MemoryDelete => 0x22, Self::IndexQuery => 0x23,
            Self::IndexUpsert => 0x24,
            Self::CapabilityAdvertise => 0x30, Self::CapabilityWithdraw => 0x31,
            Self::CapabilityQuery => 0x32,
            Self::ModelCall => 0x40, Self::ToolCall => 0x41,
            Self::Embed => 0x42, Self::Classify => 0x43,
            Self::Reason => 0x44, Self::Retrieve => 0x45,
            Self::Execute => 0x46,
            Self::Trace => 0x50, Self::Stats => 0x51, Self::Health => 0x52,
            Self::BufferRef => 0x60, Self::BufferRelease => 0x61,
            Self::BufferShare => 0x62,
            Self::Vendor(v) => v,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Nop=>"NOP", Self::Hello=>"HELLO", Self::Goodbye=>"GOODBYE",
            Self::Discover=>"DISCOVER", Self::Ping=>"PING", Self::Pong=>"PONG",
            Self::Ack=>"ACK", Self::Nak=>"NAK", Self::Error=>"ERROR",
            Self::Cancel=>"CANCEL", Self::Version=>"VERSION",
            Self::Call=>"CALL", Self::Return=>"RETURN",
            Self::Stream=>"STREAM", Self::StreamEnd=>"STREAM_END",
            Self::MemoryRead=>"MEMORY_READ", Self::MemoryWrite=>"MEMORY_WRITE",
            Self::MemoryDelete=>"MEMORY_DELETE", Self::IndexQuery=>"INDEX_QUERY",
            Self::IndexUpsert=>"INDEX_UPSERT",
            Self::CapabilityAdvertise=>"CAPABILITY_ADVERTISE",
            Self::CapabilityWithdraw=>"CAPABILITY_WITHDRAW",
            Self::CapabilityQuery=>"CAPABILITY_QUERY",
            Self::ModelCall=>"MODEL_CALL", Self::ToolCall=>"TOOL_CALL",
            Self::Embed=>"EMBED", Self::Classify=>"CLASSIFY",
            Self::Reason=>"REASON", Self::Retrieve=>"RETRIEVE", Self::Execute=>"EXECUTE",
            Self::Trace=>"TRACE", Self::Stats=>"STATS", Self::Health=>"HEALTH",
            Self::BufferRef=>"BUFFER_REF", Self::BufferRelease=>"BUFFER_RELEASE",
            Self::BufferShare=>"BUFFER_SHARE",
            Self::Vendor(_) => "VENDOR",
        }
    }

    #[inline]
    pub const fn is_vendor(self) -> bool { matches!(self, Self::Vendor(_)) }
}

impl fmt::Display for AiclOpcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [0x{:02X}]", self.name(), self.to_u8())
    }
}


/// Packet flags — 16-bit, OR-combined.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct AiclFlags(pub u16);

impl AiclFlags {
    pub const REQUEST:        Self = Self(0x0001);
    pub const RESPONSE:       Self = Self(0x0002);
    pub const STREAM_CHUNK:   Self = Self(0x0004);
    pub const STREAM_END:     Self = Self(0x0008);
    pub const ERROR_FLAG:     Self = Self(0x0010);
    pub const CANCEL:         Self = Self(0x0020);
    pub const HEARTBEAT:      Self = Self(0x0040);
    pub const COMPRESSED:     Self = Self(0x0080);
    pub const ENCRYPTED:      Self = Self(0x0100);
    pub const TRACED:         Self = Self(0x0200);
    pub const PRIORITY_HIGH:  Self = Self(0x0400);
    pub const PRIORITY_LOW:   Self = Self(0x0800);
    pub const EXTENDED:       Self = Self(0x1000);
    pub const HAS_BUFFER_REF: Self = Self(0x2000);
    pub const FROZEN:         Self = Self(0x4000);

    #[inline]
    pub const fn new(bits: u16) -> Self { Self(bits) }
    #[inline]
    pub const fn bits(self) -> u16 { self.0 }
    #[inline]
    pub const fn contains(self, flag: Self) -> bool { (self.0 & flag.0) == flag.0 }
    #[inline]
    pub fn insert(&mut self, flag: Self) { self.0 |= flag.0; }
    #[inline]
    pub fn remove(&mut self, flag: Self) { self.0 &= !flag.0; }
}

impl fmt::Display for AiclFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        let mut write_flag = |name: &'static str| -> fmt::Result {
            if !first { write!(f, "|")?; }
            first = false; write!(f, "{}", name)
        };
        if self.contains(Self::REQUEST)        { write_flag("REQUEST")?; }
        if self.contains(Self::RESPONSE)       { write_flag("RESPONSE")?; }
        if self.contains(Self::STREAM_CHUNK)   { write_flag("STREAM_CHUNK")?; }
        if self.contains(Self::STREAM_END)     { write_flag("STREAM_END")?; }
        if self.contains(Self::ERROR_FLAG)     { write_flag("ERROR")?; }
        if self.contains(Self::CANCEL)         { write_flag("CANCEL")?; }
        if self.contains(Self::HEARTBEAT)     { write_flag("HEARTBEAT")?; }
        if self.contains(Self::COMPRESSED)     { write_flag("COMPRESSED")?; }
        if self.contains(Self::ENCRYPTED)      { write_flag("ENCRYPTED")?; }
        if self.contains(Self::TRACED)         { write_flag("TRACED")?; }
        if self.contains(Self::PRIORITY_HIGH)  { write_flag("PRIORITY_HIGH")?; }
        if self.contains(Self::PRIORITY_LOW)   { write_flag("PRIORITY_LOW")?; }
        if self.contains(Self::EXTENDED)       { write_flag("EXTENDED")?; }
        if self.contains(Self::HAS_BUFFER_REF) { write_flag("BUFFER_REF")?; }
        if self.contains(Self::FROZEN)         { write_flag("FROZEN")?; }
        if first { write!(f, "NONE")?; }
        Ok(())
    }
}

