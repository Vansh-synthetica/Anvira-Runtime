//! Shared memory transport error types.
use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShmError {
    CreateFailed, OpenFailed, MapFailed, InvalidRegion, Corrupted,
    PayloadTooLarge, Backpressure, Empty, NotProducer, NotConsumer,
    Timeout, Cancelled, BufferNotFound, InvalidBuffer, System(i32),
}

pub type ShmResult<T> = Result<T, ShmError>;

impl fmt::Display for ShmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CreateFailed => write!(f, "failed to create shared memory region"),
            Self::OpenFailed => write!(f, "failed to open shared memory region"),
            Self::MapFailed => write!(f, "failed to map shared memory"),
            Self::InvalidRegion => write!(f, "shared memory region is invalid"),
            Self::Corrupted => write!(f, "shared memory region is corrupted"),
            Self::PayloadTooLarge => write!(f, "payload too large for slot"),
            Self::Backpressure => write!(f, "ring buffer is full (backpressure)"),
            Self::Empty => write!(f, "ring buffer is empty"),
            Self::NotProducer => write!(f, "operation requires producer role"),
            Self::NotConsumer => write!(f, "operation requires consumer role"),
            Self::Timeout => write!(f, "operation timed out"),
            Self::Cancelled => write!(f, "operation was cancelled"),
            Self::BufferNotFound => write!(f, "buffer not found"),
            Self::InvalidBuffer => write!(f, "invalid buffer offset/length"),
            Self::System(e) => write!(f, "system error: {}", e),
        }
    }
}

impl core::error::Error for ShmError {}
