//! Typed operand model per AICL-ISA v1.0.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::error::{AiclError, AiclResult, ErrorCode};
use crate::opcode::{MAX_BYTES_SIZE, MAX_STR_SIZE};

/// Operand type tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum OperandType {
    I8    = 0x01,
    I16   = 0x02,
    I32   = 0x03,
    I64   = 0x04,
    U8    = 0x05,
    U16   = 0x06,
    U32   = 0x07,
    U64   = 0x08,
    F32   = 0x09,
    F64   = 0x0A,
    Bool  = 0x10,
    Str   = 0x11,
    Bytes = 0x12,
    Uuid  = 0x13,
    Handle = 0x14,
    Ref   = 0x15,
    BufRef = 0x16,
    List  = 0x17,
    Map   = 0x18,
    Null  = 0x19,
    TsMs  = 0x1A,
    DurationMs = 0x1B,
    Vendor(u8) = 0x80,
}

impl OperandType {
    #[inline]
    pub const fn from_u8(v: u8) -> Self {
        match v {
            0x01=>Self::I8, 0x02=>Self::I16, 0x03=>Self::I32, 0x04=>Self::I64,
            0x05=>Self::U8, 0x06=>Self::U16, 0x07=>Self::U32, 0x08=>Self::U64,
            0x09=>Self::F32, 0x0A=>Self::F64,
            0x10=>Self::Bool, 0x11=>Self::Str, 0x12=>Self::Bytes, 0x13=>Self::Uuid,
            0x14=>Self::Handle, 0x15=>Self::Ref, 0x16=>Self::BufRef,
            0x17=>Self::List, 0x18=>Self::Map, 0x19=>Self::Null,
            0x1A=>Self::TsMs, 0x1B=>Self::DurationMs,
            v if v >= 0x80 => Self::Vendor(v),
            _ => Self::Null,
        }
    }

    #[inline]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::I8 => 0x01, Self::I16 => 0x02, Self::I32 => 0x03, Self::I64 => 0x04,
            Self::U8 => 0x05, Self::U16 => 0x06, Self::U32 => 0x07, Self::U64 => 0x08,
            Self::F32 => 0x09, Self::F64 => 0x0A,
            Self::Bool => 0x10, Self::Str => 0x11, Self::Bytes => 0x12, Self::Uuid => 0x13,
            Self::Handle => 0x14, Self::Ref => 0x15, Self::BufRef => 0x16,
            Self::List => 0x17, Self::Map => 0x18, Self::Null => 0x19,
            Self::TsMs => 0x1A, Self::DurationMs => 0x1B,
            Self::Vendor(v) => v,
        }
    }

    #[inline]
    pub const fn is_core(self) -> bool { !matches!(self, Self::Vendor(_)) }
}

impl fmt::Display for OperandType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}[0x{:02X}]", self, self.to_u8())
    }
}

/// Buffer reference descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufRef {
    pub id: u64,
    pub offset: u32,
    pub length: u32,
}

/// Typed operand value.
#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    I8(i8), I16(i16), I32(i32), I64(i64),
    U8(u8), U16(u16), U32(u32), U64(u64),
    F32(f32), F64(f64),
    Bool(bool),
    Str(String),
    Bytes(Vec<u8>),
    Uuid([u8; 16]),
    Handle(u64),
    Ref(u16),
    BufRef(BufRef),
    List(Vec<Operand>),
    Map(BTreeMap<String, Operand>),
    Null,
    TsMs(i64),
    DurationMs(u32),
    Vendor(u8, Vec<u8>),
}

impl Operand {
    /// Return the operand type tag.
    #[inline]
    pub fn kind(&self) -> OperandType {
        match self {
            Self::I8(_) => OperandType::I8, Self::I16(_) => OperandType::I16,
            Self::I32(_) => OperandType::I32, Self::I64(_) => OperandType::I64,
            Self::U8(_) => OperandType::U8, Self::U16(_) => OperandType::U16,
            Self::U32(_) => OperandType::U32, Self::U64(_) => OperandType::U64,
            Self::F32(_) => OperandType::F32, Self::F64(_) => OperandType::F64,
            Self::Bool(_) => OperandType::Bool, Self::Str(_) => OperandType::Str,
            Self::Bytes(_) => OperandType::Bytes, Self::Uuid(_) => OperandType::Uuid,
            Self::Handle(_) => OperandType::Handle, Self::Ref(_) => OperandType::Ref,
            Self::BufRef(_) => OperandType::BufRef, Self::List(_) => OperandType::List,
            Self::Map(_) => OperandType::Map, Self::Null => OperandType::Null,
            Self::TsMs(_) => OperandType::TsMs, Self::DurationMs(_) => OperandType::DurationMs,
            Self::Vendor(t, _) => OperandType::Vendor(*t),
        }
    }

    /// Convenience constructors.
    #[inline]
    pub fn str_val(s: impl Into<String>) -> Self { Self::Str(s.into()) }

    #[inline]
    pub fn bytes_val(b: impl Into<Vec<u8>>) -> Self { Self::Bytes(b.into()) }

    #[inline]
    pub fn null() -> Self { Self::Null }

    /// Validate size constraints.
    pub fn validate(&self) -> AiclResult<()> {
        match self {
            Self::Str(s) if s.len() > MAX_STR_SIZE => {
                Err(AiclError::new(ErrorCode::PayloadTooLarge,
                    format!("STR {} > limit {}", s.len(), MAX_STR_SIZE)))
            }
            Self::Bytes(b) if b.len() > MAX_BYTES_SIZE => {
                Err(AiclError::new(ErrorCode::PayloadTooLarge,
                    format!("BYTES {} > limit {}", b.len(), MAX_BYTES_SIZE)))
            }
            Self::List(items) => {
                for item in items { item.validate()?; }
                Ok(())
            }
            Self::Map(entries) => {
                for (k, v) in entries {
                    if k.len() > MAX_STR_SIZE {
                        return Err(AiclError::new(ErrorCode::PayloadTooLarge,
                            format!("MAP key '{}' > limit", k)));
                    }
                    v.validate()?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}


impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I8(v) => write!(f, "I8({})", v),
            Self::I16(v) => write!(f, "I16({})", v),
            Self::I32(v) => write!(f, "I32({})", v),
            Self::I64(v) => write!(f, "I64({})", v),
            Self::U8(v) => write!(f, "U8({})", v),
            Self::U16(v) => write!(f, "U16({})", v),
            Self::U32(v) => write!(f, "U32({})", v),
            Self::U64(v) => write!(f, "U64({})", v),
            Self::F32(v) => write!(f, "F32({})", v),
            Self::F64(v) => write!(f, "F64({})", v),
            Self::Bool(v) => write!(f, "Bool({})", v),
            Self::Str(v) => write!(f, "Str({:?})", v),
            Self::Bytes(v) => write!(f, "Bytes({} bytes)", v.len()),
            Self::Uuid(v) => write!(f, "Uuid({:02X?})", v),
            Self::Handle(v) => write!(f, "Handle(0x{:016X})", v),
            Self::Ref(v) => write!(f, "Ref({})", v),
            Self::BufRef(b) => write!(f, "BufRef(id={},off={},len={})", b.id, b.offset, b.length),
            Self::List(v) => write!(f, "List({} items)", v.len()),
            Self::Map(m) => write!(f, "Map({} entries)", m.len()),
            Self::Null => write!(f, "Null"),
            Self::TsMs(v) => write!(f, "TsMs({})", v),
            Self::DurationMs(v) => write!(f, "DurationMs({})", v),
            Self::Vendor(t, b) => write!(f, "Vendor(0x{:02X},{} bytes)", t, b.len()),
        }
    }
}

