//! AiclPacket — the typed semantic representation of an AICL message.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::error::{AiclError, AiclResult, ErrorCode};
use crate::opcode::{AiclFlags, AiclOpcode, ISA_VERSION};
use crate::operand::Operand;

/// A fully-formed AICL packet.
#[derive(Debug, Clone)]
pub struct AiclPacket {
    pub opcode: AiclOpcode,
    pub flags: AiclFlags,
    pub message_id: Option<[u8; 16]>,
    pub correlation_id: [u8; 16],
    pub deadline_ms: u32,
    pub operands: Vec<Operand>,
}

impl AiclPacket {
    pub fn new(opcode: AiclOpcode) -> Self {
        Self {
            opcode, flags: AiclFlags::default(),
            message_id: None, correlation_id: [0u8; 16],
            deadline_ms: 0, operands: Vec::new(),
        }
    }

    pub fn nop() -> Self { Self::new(AiclOpcode::Nop) }

    pub fn ping(nonce: u64) -> Self {
        let mut p = Self::new(AiclOpcode::Ping);
        p.flags.insert(AiclFlags::REQUEST);
        p.operands.push(Operand::U64(nonce));
        p
    }

    pub fn pong(nonce: u64) -> Self {
        let mut p = Self::new(AiclOpcode::Pong);
        p.flags.insert(AiclFlags::RESPONSE);
        p.operands.push(Operand::U64(nonce));
        p
    }

    pub fn hello(name: impl Into<String>, capabilities: Vec<String>) -> Self {
        let mut p = Self::new(AiclOpcode::Hello);
        p.flags.insert(AiclFlags::REQUEST);
        p.operands.push(Operand::Str(name.into()));
        p.operands.push(Operand::U32(ISA_VERSION as u32));
        let list = capabilities.into_iter().map(Operand::Str).collect();
        p.operands.push(Operand::List(list));
        p.operands.push(Operand::Map(BTreeMap::new()));
        p
    }

    pub fn call(target: u16, deadline_ms: u32) -> Self {
        let mut p = Self::new(AiclOpcode::Call);
        p.flags.insert(AiclFlags::REQUEST);
        p.operands.push(Operand::Ref(target));
        p.operands.push(Operand::Map(BTreeMap::new()));
        p.operands.push(Operand::U32(deadline_ms));
        p.operands.push(Operand::U16(0));
        p
    }

    pub fn return_response(correlation_id: [u8; 16]) -> Self {
        let mut p = Self::new(AiclOpcode::Return);
        p.flags.insert(AiclFlags::RESPONSE);
        p.correlation_id = correlation_id;
        p.operands.push(Operand::Uuid(correlation_id));
        p.operands.push(Operand::Map(BTreeMap::new()));
        p
    }

    pub fn error(code: ErrorCode, message: impl Into<String>) -> Self {
        let mut p = Self::new(AiclOpcode::Error);
        p.flags.insert(AiclFlags::ERROR_FLAG);
        p.operands.push(Operand::U32(code.code()));
        p.operands.push(Operand::Str(message.into()));
        p.operands.push(Operand::Map(BTreeMap::new()));
        p
    }
}
// ─────────────────────────────────────────────────────────────────────────────
// Remaining AiclPacket methods
// ─────────────────────────────────────────────────────────────────────────────
impl AiclPacket {
    pub fn cancel(correlation_id: [u8; 16], reason: impl Into<String>) -> Self {
        let mut p = Self::new(AiclOpcode::Cancel);
        p.flags.insert(AiclFlags::CANCEL);
        p.operands.push(Operand::Uuid(correlation_id));
        p.operands.push(Operand::Str(reason.into()));
        p
    }

    pub fn classify(text: impl Into<String>, labels: Vec<String>, model: impl Into<String>) -> Self {
        let mut p = Self::new(AiclOpcode::Classify);
        p.flags.insert(AiclFlags::REQUEST);
        p.operands.push(Operand::Str(text.into()));
        let list = labels.into_iter().map(Operand::Str).collect();
        p.operands.push(Operand::List(list));
        p.operands.push(Operand::Str(model.into()));
        p
    }

    pub fn embed(model: impl Into<String>, texts: Vec<String>) -> Self {
        let mut p = Self::new(AiclOpcode::Embed);
        p.flags.insert(AiclFlags::REQUEST);
        p.operands.push(Operand::Str(model.into()));
        let list = texts.into_iter().map(Operand::Str).collect();
        p.operands.push(Operand::List(list));
        p
    }

    /// Generate text/code/structured data.
    pub fn generate(prompt: impl Into<String>, model: impl Into<String>) -> Self {
        let mut p = Self::new(AiclOpcode::Execute);
        p.flags.insert(AiclFlags::REQUEST);
        p.operands.push(Operand::Str(prompt.into()));
        p.operands.push(Operand::Str(model.into()));
        p
    }

    /// Reason over a problem or chain-of-thought.
    pub fn reason(problem: impl Into<String>) -> Self {
        let mut p = Self::new(AiclOpcode::Reason);
        p.flags.insert(AiclFlags::REQUEST);
        p.operands.push(Operand::Str(problem.into()));
        p
    }

    /// Rank candidates by relevance.
    pub fn rank(query: impl Into<String>, candidates: Vec<String>) -> Self {
        let mut p = Self::new(AiclOpcode::Execute);
        p.flags.insert(AiclFlags::REQUEST);
        p.operands.push(Operand::Str(query.into()));
        let list = candidates.into_iter().map(Operand::Str).collect();
        p.operands.push(Operand::List(list));
        p
    }

    /// Summarize text.
    pub fn summarize(text: impl Into<String>, max_length: u32) -> Self {
        let mut p = Self::new(AiclOpcode::Execute);
        p.flags.insert(AiclFlags::REQUEST);
        p.operands.push(Operand::Str(text.into()));
        p.operands.push(Operand::U32(max_length));
        p
    }

    /// Verify a claim for correctness.
    pub fn verify(claim: impl Into<String>) -> Self {
        let mut p = Self::new(AiclOpcode::Reason);
        p.flags.insert(AiclFlags::REQUEST);
        p.operands.push(Operand::Str(claim.into()));
        p
    }

    /// Extract structured fields from unstructured input.
    pub fn extract(text: impl Into<String>, fields: Vec<String>) -> Self {
        let mut p = Self::new(AiclOpcode::Execute);
        p.flags.insert(AiclFlags::REQUEST);
        p.operands.push(Operand::Str(text.into()));
        let list = fields.into_iter().map(Operand::Str).collect();
        p.operands.push(Operand::List(list));
        p
    }

    /// Transform input to a different format.
    pub fn transform(input: impl Into<String>, format: impl Into<String>) -> Self {
        let mut p = Self::new(AiclOpcode::Execute);
        p.flags.insert(AiclFlags::REQUEST);
        p.operands.push(Operand::Str(input.into()));
        p.operands.push(Operand::Str(format.into()));
        p
    }

    /// Plan a multi-step sequence.
    pub fn plan(goal: impl Into<String>) -> Self {
        let mut p = Self::new(AiclOpcode::Execute);
        p.flags.insert(AiclFlags::REQUEST);
        p.operands.push(Operand::Str(goal.into()));
        p
    }

    /// Evaluate quality of input.
    pub fn evaluate(input: impl Into<String>, criteria: impl Into<String>) -> Self {
        let mut p = Self::new(AiclOpcode::Execute);
        p.flags.insert(AiclFlags::REQUEST);
        p.operands.push(Operand::Str(input.into()));
        p.operands.push(Operand::Str(criteria.into()));
        p
    }

    /// Call a model by capability reference.
    pub fn model_call(cap_ref: u16, task: impl Into<String>, inputs: BTreeMap<String, Operand>) -> Self {
        let mut p = Self::new(AiclOpcode::ModelCall);
        p.flags.insert(AiclFlags::REQUEST);
        p.operands.push(Operand::Ref(cap_ref));
        p.operands.push(Operand::Str(task.into()));
        p.operands.push(Operand::Map(inputs));
        p
    }

    /// Call an external tool.
    pub fn tool_call(tool_name: impl Into<String>, params: BTreeMap<String, Operand>) -> Self {
        let mut p = Self::new(AiclOpcode::ToolCall);
        p.flags.insert(AiclFlags::REQUEST);
        p.operands.push(Operand::Str(tool_name.into()));
        p.operands.push(Operand::Map(params));
        p
    }

    /// Read from persistent memory.
    pub fn memory_read(key: impl Into<String>) -> Self {
        let mut p = Self::new(AiclOpcode::MemoryRead);
        p.flags.insert(AiclFlags::REQUEST);
        p.operands.push(Operand::Str(key.into()));
        p
    }

    /// Write to persistent memory.
    pub fn memory_write(key: impl Into<String>, value: Operand) -> Self {
        let mut p = Self::new(AiclOpcode::MemoryWrite);
        p.flags.insert(AiclFlags::REQUEST);
        p.operands.push(Operand::Str(key.into()));
        p.operands.push(value);
        p
    }

    /// Delete from persistent memory.
    pub fn memory_delete(key: impl Into<String>) -> Self {
        let mut p = Self::new(AiclOpcode::MemoryDelete);
        p.flags.insert(AiclFlags::REQUEST);
        p.operands.push(Operand::Str(key.into()));
        p
    }

    /// Query the index.
    pub fn index_query(query: impl Into<String>) -> Self {
        let mut p = Self::new(AiclOpcode::IndexQuery);
        p.flags.insert(AiclFlags::REQUEST);
        p.operands.push(Operand::Str(query.into()));
        p
    }

    /// Upsert into the index.
    pub fn index_upsert(key: impl Into<String>, value: Operand) -> Self {
        let mut p = Self::new(AiclOpcode::IndexUpsert);
        p.flags.insert(AiclFlags::REQUEST);
        p.operands.push(Operand::Str(key.into()));
        p.operands.push(value);
        p
    }

    pub fn with(mut self, op: Operand) -> Self {
        self.operands.push(op);
        self
    }

    pub fn with_correlation(mut self, id: [u8; 16]) -> Self {
        self.correlation_id = id;
        self
    }

    pub fn with_deadline_ms(mut self, ms: u32) -> Self {
        self.deadline_ms = ms;
        self
    }

    pub fn with_flag(mut self, flag: AiclFlags) -> Self {
        self.flags.insert(flag);
        self
    }

    pub fn map_get(&self, key: &str) -> Option<&Operand> {
        for op in &self.operands {
            if let Operand::Map(m) = op {
                if let Some(v) = m.get(key) { return Some(v); }
            }
        }
        None
    }

    pub fn validate(&self) -> AiclResult<()> {
        if self.opcode.is_vendor() { return Ok(()); }
        for op in &self.operands { op.validate()?; }
        Ok(())
    }

    /// Encode this packet to bytes using a default codec.
    pub fn encode(&self) -> AiclResult<Vec<u8>> {
        crate::codec::AiclCodec::new().encode(self)
    }
}

/// Fluent builder for AiclPacket.
pub struct AiclPacketBuilder {
    pkt: AiclPacket,
}

impl AiclPacketBuilder {
    pub fn new(opcode: AiclOpcode) -> Self { Self { pkt: AiclPacket::new(opcode) } }
    pub fn ping(nonce: u64) -> Self { Self { pkt: AiclPacket::ping(nonce) } }
    pub fn call(target: u16, deadline_ms: u32) -> Self { Self { pkt: AiclPacket::call(target, deadline_ms) } }
    pub fn classify(text: impl Into<String>, labels: Vec<String>, model: impl Into<String>) -> Self {
        Self { pkt: AiclPacket::classify(text, labels, model) }
    }

    pub fn generate(prompt: impl Into<String>, model: impl Into<String>) -> Self {
        Self { pkt: AiclPacket::generate(prompt, model) }
    }

    pub fn reason(problem: impl Into<String>) -> Self {
        Self { pkt: AiclPacket::reason(problem) }
    }

    pub fn tool_call(tool_name: impl Into<String>, params: BTreeMap<String, Operand>) -> Self {
        Self { pkt: AiclPacket::tool_call(tool_name, params) }
    }

    pub fn model_call(cap_ref: u16, task: impl Into<String>, inputs: BTreeMap<String, Operand>) -> Self {
        Self { pkt: AiclPacket::model_call(cap_ref, task, inputs) }
    }

    pub fn memory_read(key: impl Into<String>) -> Self {
        Self { pkt: AiclPacket::memory_read(key) }
    }

    pub fn memory_write(key: impl Into<String>, value: Operand) -> Self {
        Self { pkt: AiclPacket::memory_write(key, value) }
    }

    pub fn embed(model: impl Into<String>, texts: Vec<String>) -> Self {
        Self { pkt: AiclPacket::embed(model, texts) }
    }

    /// Add a key/value into the first MAP operand; create one if absent.
    pub fn input(mut self, key: impl Into<String>, value: impl Into<Operand>) -> Self {
        let k = key.into();
        let v = value.into();
        for op in &mut self.pkt.operands {
            if let Operand::Map(m) = op {
                m.insert(k.clone(), v);
                return self;
            }
        }
        let mut m = BTreeMap::new();
        m.insert(k, v);
        self.pkt.operands.push(Operand::Map(m));
        self
    }

    pub fn flag(mut self, f: AiclFlags) -> Self { self.pkt.flags.insert(f); self }
    pub fn deadline_ms(mut self, ms: u32) -> Self { self.pkt.deadline_ms = ms; self }
    pub fn correlation(mut self, id: [u8; 16]) -> Self { self.pkt.correlation_id = id; self }
    pub fn build(self) -> AiclPacket { self.pkt }
}

