//! Binary codec — encode / decode AICL packets.
//!
//! The codec converts between `AiclPacket` (typed) and the on-the-wire
//! byte representation defined in the AICL-ISA v1.0 spec. Decoding is
//! non-copying: see [`PacketView`].

use alloc::collections::BTreeMap;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use crc32fast::Hasher;

use crate::error::{AiclError, AiclResult, ErrorCode};
use crate::opcode::{AiclFlags, AiclOpcode, HEADER_SIZE, ISA_VERSION, MAGIC, MAX_PAYLOAD_SIZE};
use crate::operand::{BufRef, Operand, OperandType};
use crate::packet::AiclPacket;

/// Encode + decode operations for AICL packets.
pub struct AiclCodec {
    max_payload: usize,
    check_crc: bool,
}

impl Default for AiclCodec {
    fn default() -> Self { Self::new() }
}

impl AiclCodec {
    pub fn new() -> Self { Self { max_payload: MAX_PAYLOAD_SIZE, check_crc: true } }
    pub fn max_payload(mut self, v: usize) -> Self { self.max_payload = v; self }
    pub fn check_crc(mut self, v: bool) -> Self { self.check_crc = v; self }

    /// Encode a packet to a fresh byte vector.
    pub fn encode(&self, pkt: &AiclPacket) -> AiclResult<Vec<u8>> {
        let mut buf = Vec::with_capacity(HEADER_SIZE + 256);
        self.encode_into(pkt, &mut buf)?;
        Ok(buf)
    }

    /// Encode a packet into a pre-allocated buffer.
    pub fn encode_into(&self, pkt: &AiclPacket, out: &mut Vec<u8>) -> AiclResult<()> {
        let start = out.len();
        out.resize(start + HEADER_SIZE, 0);

        let payload_start = out.len();
        out.push(pkt.opcode.to_u8());
        encode_varint(pkt.operands.len() as u32, out);

        for op in &pkt.operands {
            encode_operand(op, out)?;
        }

        let payload_len = (out.len() - payload_start) as u32;
        if payload_len as usize > self.max_payload {
            return Err(AiclError::new(ErrorCode::PayloadTooLarge,
                format!("payload {} > limit {}", payload_len, self.max_payload)));
        }

        let out = out.as_mut_slice();
        out[start..start + 4].copy_from_slice(MAGIC);
        out[start + 4..start + 6].copy_from_slice(&ISA_VERSION.to_be_bytes());
        out[start + 6..start + 8].copy_from_slice(&pkt.flags.bits().to_be_bytes());
        let msg_id = pkt.message_id.unwrap_or_else(uuid_v4_bytes);
        out[start + 8..start + 24].copy_from_slice(&msg_id);
        out[start + 24..start + 40].copy_from_slice(&pkt.correlation_id);
        out[start + 40..start + 44].copy_from_slice(&pkt.deadline_ms.to_be_bytes());
        out[start + 44..start + 48].copy_from_slice(&payload_len.to_be_bytes());

        // CRC-32 of the 52-byte header prefix (bytes 0..52):
        // magic(4) + version(2) + flags(2) + message_id(16) + correlation_id(16) + deadline_ms(4) + payload_length(4) + padding(4)
        let hdr_crc = crc32(&out[start..start + 52]);
        out[start + 52..start + 56].copy_from_slice(&hdr_crc.to_be_bytes());
        Ok(())
    }
}

impl AiclCodec {
    /// Decode bytes into a zero-copy `PacketView`.
    ///
    /// If you own the buffer and want to avoid a copy, use [`decode_owned`](Self::decode_owned).
    pub fn decode(&self, data: &[u8]) -> AiclResult<PacketView> {
        Self::validate_header(data, self.max_payload, self.check_crc)?;
        let payload_len = u32::from_be_bytes([data[44], data[45], data[46], data[47]]) as usize;
        let flags = u16::from_be_bytes([data[6], data[7]]);
        let has_trailer = (flags & 0x0100) != 0;

        Ok(PacketView {
            data: Arc::new(data.to_vec()),
            payload_offset: HEADER_SIZE,
            payload_length: payload_len,
            has_trailer,
            _flags: flags,
        })
    }

    /// Decode an owned buffer into a `PacketView` without copying.
    ///
    /// Takes ownership of the `Vec<u8>` — zero copies on the decode path.
    /// The `PacketView` will own the buffer via `Arc`.
    pub fn decode_owned(&self, data: Vec<u8>) -> AiclResult<PacketView> {
        Self::validate_header(&data, self.max_payload, self.check_crc)?;
        let payload_len = u32::from_be_bytes([data[44], data[45], data[46], data[47]]) as usize;
        let flags = u16::from_be_bytes([data[6], data[7]]);
        let has_trailer = (flags & 0x0100) != 0;

        Ok(PacketView {
            data: Arc::new(data),
            payload_offset: HEADER_SIZE,
            payload_length: payload_len,
            has_trailer,
            _flags: flags,
        })
    }

    /// Fast validate header without allocating. Returns Ok(()) if valid.
    #[inline]
    fn validate_header(data: &[u8], max_payload: usize, check_crc: bool) -> AiclResult<()> {
        if data.len() < HEADER_SIZE {
            return Err(AiclError::new(ErrorCode::BadHeader,
                format!("buffer {} < header size {}", data.len(), HEADER_SIZE)));
        }
        if &data[0..4] != MAGIC {
            return Err(AiclError::new(ErrorCode::BadHeader,
                format!("invalid magic: {:02X?}", &data[0..4])));
        }
        let version = u16::from_be_bytes([data[4], data[5]]);
        if version >> 8 != 1 {
            return Err(AiclError::new(ErrorCode::UnsupportedVersion,
                format!("version major {} != 1", version >> 8)));
        }

        let hdr_crc = u32::from_be_bytes([data[52], data[53], data[54], data[55]]);
        let computed = crc32(&data[..52]);
        if hdr_crc != computed {
            return Err(AiclError::new(ErrorCode::BadHeader,
                format!("header CRC mismatch: expected {:08X}, got {:08X}", computed, hdr_crc)));
        }

        let flags = u16::from_be_bytes([data[6], data[7]]);
        let payload_len = u32::from_be_bytes([data[44], data[45], data[46], data[47]]);
        if payload_len as usize > max_payload {
            return Err(AiclError::new(ErrorCode::PayloadTooLarge,
                format!("payload {} > max {}", payload_len, max_payload)));
        }

        let total = HEADER_SIZE + payload_len as usize;
        if data.len() < total {
            return Err(AiclError::new(ErrorCode::Truncated,
                format!("declared {} + {} = {} but buffer is {}",
                    HEADER_SIZE, payload_len, total, data.len())));
        }

        let has_trailer = (flags & 0x0100) != 0;
        if has_trailer {
            if data.len() < total + 4 {
                return Err(AiclError::new(ErrorCode::Truncated, "trailer missing"));
            }
            if check_crc {
                let expected_crc = u32::from_be_bytes([
                    data[total], data[total + 1], data[total + 2], data[total + 3]]);
                let computed = crc32(&data[..total]);
                if expected_crc != computed {
                    return Err(AiclError::new(ErrorCode::Checksum,
                        format!("CRC mismatch: expected {:08X}, got {:08X}",
                            expected_crc, computed)));
                }
            }
        }

        let opcode_byte = data[HEADER_SIZE];
        let _opcode = AiclOpcode::from_u8(opcode_byte)
            .ok_or_else(|| AiclError::new(ErrorCode::UnknownOpcode,
                format!("unknown opcode byte: 0x{:02X}", opcode_byte)))?;

        let payload = &data[HEADER_SIZE..HEADER_SIZE + payload_len as usize];
        if payload.is_empty() {
            return Err(AiclError::new(ErrorCode::Truncated, "payload empty"));
        }
        // decode_varint's second return value counts bytes consumed *within
        // the payload[1..] slice* (i.e. relative to skipping the opcode
        // byte) — it must be offset by 1 to become an absolute index back
        // into `payload`, or every operand decode starts one byte early.
        let (count, varint_len) = decode_varint(&payload[1..])?;
        let mut pos = 1 + varint_len;
        for i in 0..count {
            // decode_operand's returned length is relative to the slice it
            // was given (&payload[pos..]), so it must be added to the
            // running absolute `pos`, not used to replace it.
            let (_, consumed) = decode_operand(&payload[pos..], i)?;
            pos += consumed;
        }
        if pos != payload.len() {
            return Err(AiclError::new(ErrorCode::TrailingBytes,
                format!("operand payload consumed {} of {} bytes", pos, payload.len())));
        }

        Ok(())
    }

    /// Decode bytes into an owned `AiclPacket` (copies operands).
    pub fn decode_to_packet(&self, data: &[u8]) -> AiclResult<AiclPacket> {
        let view = self.decode(data)?;
        view.to_packet()
    }
}


/// A zero-copy view into a decoded packet.
///
/// Backed by an `Arc<Vec<u8>>` so the underlying buffer stays alive
/// as long as the view is alive. No copies are made during decoding.
#[derive(Debug, Clone)]
pub struct PacketView {
    data: Arc<Vec<u8>>,
    payload_offset: usize,
    payload_length: usize,
    has_trailer: bool,
    _flags: u16,
}

impl PacketView {
    /// Backed buffer (full frame including header + payload + trailer).
    #[inline]
    pub fn raw(&self) -> &[u8] { &self.data }

    /// Opcode byte.
    #[inline]
    pub fn opcode(&self) -> u8 { self.data[HEADER_SIZE] }

    /// Flags.
    #[inline]
    pub fn flags(&self) -> u16 {
        u16::from_be_bytes([self.data[6], self.data[7]])
    }

    /// Message ID bytes.
    #[inline]
    pub fn message_id(&self) -> [u8; 16] {
        let mut id = [0u8; 16];
        id.copy_from_slice(&self.data[8..24]);
        id
    }

    /// Correlation ID bytes.
    #[inline]
    pub fn correlation_id(&self) -> [u8; 16] {
        let mut id = [0u8; 16];
        id.copy_from_slice(&self.data[24..40]);
        id
    }

    /// Deadline in milliseconds.
    #[inline]
    pub fn deadline_ms(&self) -> u32 {
        u32::from_be_bytes([self.data[40], self.data[41], self.data[42], self.data[43]])
    }

    /// Payload slice (without header).
    #[inline]
    pub fn payload(&self) -> &[u8] {
        let end = self.payload_offset + self.payload_length;
        &self.data[self.payload_offset..end]
    }

    /// Total packet size (header + payload + trailer).
    #[inline]
    pub fn total_size(&self) -> usize {
        let trailer = if self.has_trailer { 4 } else { 0 };
        self.payload_offset + self.payload_length + trailer
    }

    /// True if the packet has a CRC trailer.
    #[inline]
    pub fn has_trailer(&self) -> bool { self.has_trailer }

    /// Number of operands.
    pub fn operand_count(&self) -> AiclResult<usize> {
        let payload = self.payload();
        if payload.is_empty() { return Ok(0); }
        let (count, _) = decode_varint(&payload[1..])?;
        Ok(count as usize)
    }

    /// Decode operand at index `idx`.
    pub fn operand(&self, idx: usize) -> AiclResult<Operand> {
        let payload = self.payload();
        if payload.is_empty() {
            return Err(AiclError::new(ErrorCode::Truncated, "payload empty"));
        }
        let (count, varint_len) = decode_varint(&payload[1..])?;
        let mut pos = 1 + varint_len;
        if (idx as u32) >= count {
            return Err(AiclError::new(ErrorCode::OperandCount,
                format!("operand index {} >= count {}", idx, count)));
        }
        for i in 0..count {
            let (op, consumed) = decode_operand(&payload[pos..], i)?;
            if i == idx as u32 {
                return Ok(op);
            }
            pos += consumed;
        }
        unreachable!()
    }

    /// Convert to an owned AiclPacket (copies operand data).
    pub fn to_packet(&self) -> AiclResult<AiclPacket> {
        let payload = self.payload();
        let opcode_byte = self.opcode();
        let opcode = AiclOpcode::from_u8(opcode_byte)
            .ok_or_else(|| AiclError::new(ErrorCode::UnknownOpcode,
                format!("unknown opcode byte: 0x{:02X}", opcode_byte)))?;
        let flags = AiclFlags::new(self.flags());

        let (count, varint_len) = decode_varint(&payload[1..])?;
        let mut pos = 1 + varint_len;
        let mut operands = Vec::with_capacity(count as usize);
        for i in 0..count {
            let (op, consumed) = decode_operand(&payload[pos..], i)?;
            operands.push(op);
            pos += consumed;
        }

        Ok(AiclPacket {
            opcode, flags,
            message_id: Some(self.message_id()),
            correlation_id: self.correlation_id(),
            deadline_ms: self.deadline_ms(),
            operands,
        })
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Internal encoding helpers
// ─────────────────────────────────────────────────────────────────────────────

fn encode_varint(mut value: u32, out: &mut Vec<u8>) {
    // LEB128: the continuation bit belongs on every byte EXCEPT the last
    // one still to come — i.e. set it when more bytes will follow, not
    // when a byte isn't the first. decode_varint (below) stops as soon as
    // it sees a byte with the continuation bit clear, so getting this
    // backwards silently truncates every multi-byte varint to just its
    // first byte on decode (anything encoding a value >= 128 — payload
    // lengths, operand counts, etc. — corrupts everything after it).
    let mut buf = [0u8; 4];
    let mut len = 0;
    loop {
        let more = value > 0x7F;
        buf[len] = (value as u8 & 0x7F) | if more { 0x80 } else { 0 };
        len += 1;
        if !more { break; }
        value >>= 7;
        if len == 4 { break; }
    }
    out.extend_from_slice(&buf[..len]);
}

fn decode_varint(data: &[u8]) -> AiclResult<(u32, usize)> {
    let mut value: u32 = 0;
    let mut shift = 0;
    for (i, &byte) in data.iter().enumerate() {
        if i >= 4 {
            return Err(AiclError::new(ErrorCode::VarintOverflow, "varint > 4 bytes"));
        }
        value |= ((byte & 0x7F) as u32) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, i + 1));
        }
        shift += 7;
    }
    Err(AiclError::new(ErrorCode::Truncated, "varint truncated"))
}

fn encode_operand(op: &Operand, out: &mut Vec<u8>) -> AiclResult<()> {
    out.push(op.kind().to_u8());
    match op {
        Operand::Null => {}
        Operand::I8(v) => { out.push(*v as u8); }
        Operand::I16(v) => { out.write_i16::<BigEndian>(*v).unwrap(); }
        Operand::I32(v) => { out.write_i32::<BigEndian>(*v).unwrap(); }
        Operand::I64(v) => { out.write_i64::<BigEndian>(*v).unwrap(); }
        Operand::U8(v) => { out.push(*v); }
        Operand::U16(v) => { out.write_u16::<BigEndian>(*v).unwrap(); }
        Operand::U32(v) => { out.write_u32::<BigEndian>(*v).unwrap(); }
        Operand::U64(v) => { out.write_u64::<BigEndian>(*v).unwrap(); }
        Operand::F32(v) => { out.write_f32::<BigEndian>(*v).unwrap(); }
        Operand::F64(v) => { out.write_f64::<BigEndian>(*v).unwrap(); }
        Operand::Bool(v) => { out.push(if *v { 1 } else { 0 }); }
        Operand::Str(s) => {
            encode_varint(s.len() as u32, out);
            out.extend_from_slice(s.as_bytes());
        }
        Operand::Bytes(b) => {
            encode_varint(b.len() as u32, out);
            out.extend_from_slice(b);
        }
        Operand::Uuid(id) => { out.extend_from_slice(id); }
        Operand::Handle(v) => { out.write_u64::<BigEndian>(*v).unwrap(); }
        Operand::Ref(v) => { out.write_u16::<BigEndian>(*v).unwrap(); }
        Operand::BufRef(b) => {
            out.write_u64::<BigEndian>(b.id).unwrap();
            out.write_u32::<BigEndian>(b.offset).unwrap();
            out.write_u32::<BigEndian>(b.length).unwrap();
        }
        Operand::List(items) => {
            encode_varint(items.len() as u32, out);
            for item in items { encode_operand(item, out)?; }
        }
        Operand::Map(m) => {
            encode_varint(m.len() as u32, out);
            for (k, v) in m {
                encode_varint(k.len() as u32, out);
                out.extend_from_slice(k.as_bytes());
                encode_operand(v, out)?;
            }
        }
        Operand::TsMs(v) => { out.write_i64::<BigEndian>(*v).unwrap(); }
        Operand::DurationMs(v) => { out.write_u32::<BigEndian>(*v).unwrap(); }
        Operand::Vendor(tag, data) => {
            encode_varint(data.len() as u32, out);
            out.push(*tag);
            out.extend_from_slice(data);
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Operand decoding
// ─────────────────────────────────────────────────────────────────────────────

fn need_bytes(data: &[u8], pos: usize, n: usize, idx: u32) -> AiclResult<()> {
    if data.len() < pos + n {
        return Err(AiclError::new(ErrorCode::Truncated,
            format!("operand {}: need {} bytes, have {}", idx, n, data.len() - pos)));
    }
    Ok(())
}
fn decode_operand(data: &[u8], idx: u32) -> AiclResult<(Operand, usize)> {
    if data.is_empty() {
        return Err(AiclError::new(ErrorCode::Truncated,
            format!("operand {}: no tag byte", idx)));
    }
    let tag = OperandType::from_u8(data[0]);
    let mut pos = 1;

    let op = match tag {
        OperandType::Null => Operand::Null,
        OperandType::I8 => {
            need_bytes(data, pos, 1, idx)?;
            let v = data[pos] as i8; pos += 1;
            Operand::I8(v)
        }
        OperandType::I16 => {
            need_bytes(data, pos, 2, idx)?;
            let v = (&data[pos..pos+2]).read_i16::<BigEndian>().unwrap(); pos += 2;
            Operand::I16(v)
        }
        OperandType::I32 => {
            need_bytes(data, pos, 4, idx)?;
            let v = (&data[pos..pos+4]).read_i32::<BigEndian>().unwrap(); pos += 4;
            Operand::I32(v)
        }
        OperandType::I64 => {
            need_bytes(data, pos, 8, idx)?;
            let v = (&data[pos..pos+8]).read_i64::<BigEndian>().unwrap(); pos += 8;
            Operand::I64(v)
        }
        OperandType::U8 => {
            need_bytes(data, pos, 1, idx)?;
            let v = data[pos]; pos += 1;
            Operand::U8(v)
        }
        OperandType::U16 => {
            need_bytes(data, pos, 2, idx)?;
            let v = (&data[pos..pos+2]).read_u16::<BigEndian>().unwrap(); pos += 2;
            Operand::U16(v)
        }
        OperandType::U32 => {
            need_bytes(data, pos, 4, idx)?;
            let v = (&data[pos..pos+4]).read_u32::<BigEndian>().unwrap(); pos += 4;
            Operand::U32(v)
        }
        OperandType::U64 => {
            need_bytes(data, pos, 8, idx)?;
            let v = (&data[pos..pos+8]).read_u64::<BigEndian>().unwrap(); pos += 8;
            Operand::U64(v)
        }
        OperandType::F32 => {
            need_bytes(data, pos, 4, idx)?;
            let v = (&data[pos..pos+4]).read_f32::<BigEndian>().unwrap(); pos += 4;
            Operand::F32(v)
        }
        OperandType::F64 => {
            need_bytes(data, pos, 8, idx)?;
            let v = (&data[pos..pos+8]).read_f64::<BigEndian>().unwrap(); pos += 8;
            Operand::F64(v)
        }
        OperandType::Bool => {
            need_bytes(data, pos, 1, idx)?;
            if data[pos] > 1 {
                return Err(AiclError::new(ErrorCode::InvalidBool,
                    format!("operand {}: invalid BOOL {}", idx, data[pos])));
            }
            let v = data[pos] != 0; pos += 1;
            Operand::Bool(v)
        }
        OperandType::Str | OperandType::Bytes => {
            let (len, consumed) = decode_varint(&data[pos..])?;
            pos += consumed;
            let end = pos + len as usize;
            need_bytes(data, pos, len as usize, idx)?;
            if tag == OperandType::Str {
                let s = core::str::from_utf8(&data[pos..end])
                    .map_err(|_| AiclError::new(ErrorCode::InvalidUtf8,
                        format!("operand {}: invalid UTF-8", idx)))?;
                pos = end;
                Operand::Str(s.to_string())
            } else {
                pos = end;
                Operand::Bytes(data[pos - len as usize..pos].to_vec())
            }
        }
        OperandType::Uuid => {
            need_bytes(data, pos, 16, idx)?;
            let mut id = [0u8; 16];
            id.copy_from_slice(&data[pos..pos+16]); pos += 16;
            Operand::Uuid(id)
        }
        OperandType::Handle => {
            need_bytes(data, pos, 8, idx)?;
            let v = (&data[pos..pos+8]).read_u64::<BigEndian>().unwrap(); pos += 8;
            Operand::Handle(v)
        }
        OperandType::Ref => {
            need_bytes(data, pos, 2, idx)?;
            let v = (&data[pos..pos+2]).read_u16::<BigEndian>().unwrap(); pos += 2;
            Operand::Ref(v)
        }
        OperandType::BufRef => {
            need_bytes(data, pos, 16, idx)?;
            let id = (&data[pos..pos+8]).read_u64::<BigEndian>().unwrap();
            let offset = (&data[pos+8..pos+12]).read_u32::<BigEndian>().unwrap();
            let length = (&data[pos+12..pos+16]).read_u32::<BigEndian>().unwrap();
            pos += 16;
            Operand::BufRef(BufRef { id, offset, length })
        }
        OperandType::List => {
            let (count, consumed) = decode_varint(&data[pos..])?;
            pos += consumed;
            let mut items = Vec::with_capacity(count as usize);
            for i in 0..count {
                let (item, new_pos) = decode_operand(&data[pos..], idx * 1000 + i)?;
                items.push(item);
                pos += new_pos;
            }
            Operand::List(items)
        }
        OperandType::Map => {
            let (count, consumed) = decode_varint(&data[pos..])?;
            pos += consumed;
            let mut map = BTreeMap::new();
            for i in 0..count {
                let (key_len, kc) = decode_varint(&data[pos..])?;
                pos += kc;
                let key_end = pos + key_len as usize;
                need_bytes(data, pos, key_len as usize, idx)?;
                let key = core::str::from_utf8(&data[pos..key_end])
                    .map_err(|_| AiclError::new(ErrorCode::InvalidUtf8,
                        format!("map key {}: invalid UTF-8", i)))?;
                pos = key_end;
                let (value, vp) = decode_operand(&data[pos..], idx * 1000 + i)?;
                map.insert(key.to_string(), value);
                pos += vp;
            }
            Operand::Map(map)
        }
        OperandType::TsMs => {
            need_bytes(data, pos, 8, idx)?;
            let v = (&data[pos..pos+8]).read_i64::<BigEndian>().unwrap(); pos += 8;
            Operand::TsMs(v)
        }
        OperandType::DurationMs => {
            need_bytes(data, pos, 4, idx)?;
            let v = (&data[pos..pos+4]).read_u32::<BigEndian>().unwrap(); pos += 4;
            Operand::DurationMs(v)
        }
        OperandType::Vendor(v) => {
            let (len, consumed) = decode_varint(&data[pos..])?;
            pos += consumed;
            let end = pos + len as usize;
            need_bytes(data, pos, len as usize, idx)?;
            pos = end;
            Operand::Vendor(v, data[pos - len as usize..pos].to_vec())
        }
        _ => return Err(AiclError::new(ErrorCode::UnknownType,
            format!("operand {}: unknown type tag 0x{:02X}", idx, tag.to_u8()))),
    };

    Ok((op, pos))
}



// ─────────────────────────────────────────────────────────────────────────────
// CRC + UUID helpers
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) fn crc32(data: &[u8]) -> u32 {
    let mut h = Hasher::new();
    h.update(data);
    h.finalize()
}

/// Decode a wire frame directly into an owned `AiclPacket`.
///
/// This is the inner implementation used by both `PacketView::to_packet()`
/// and `BorrowedView::to_packet()`.
pub(crate) fn decode_to_packet_from_slice(data: &[u8]) -> AiclResult<AiclPacket> {
    if data.len() < HEADER_SIZE {
        return Err(AiclError::new(ErrorCode::BadHeader, "truncated header"));
    }
    let opcode_byte = data[HEADER_SIZE];
    let opcode = AiclOpcode::from_u8(opcode_byte)
        .ok_or_else(|| AiclError::new(ErrorCode::UnknownOpcode,
            alloc::format!("unknown opcode: 0x{:02X}", opcode_byte)))?;
    let flags = AiclFlags::new(u16::from_be_bytes([data[6], data[7]]));
    let payload_len = u32::from_be_bytes([data[44], data[45], data[46], data[47]]) as usize;
    let payload = &data[HEADER_SIZE..HEADER_SIZE + payload_len];

    if payload.is_empty() {
        return Err(AiclError::new(ErrorCode::Truncated, "payload empty"));
    }

    let (count, varint_len) = decode_varint(&payload[1..])?;
    let mut pos = 1 + varint_len;
    let mut operands = Vec::with_capacity(count as usize);
    for i in 0..count {
        let (op, consumed) = decode_operand(&payload[pos..], i)?;
        operands.push(op);
        pos += consumed;
    }

    let mut msg_id = [0u8; 16];
    msg_id.copy_from_slice(&data[8..24]);
    let mut corr_id = [0u8; 16];
    corr_id.copy_from_slice(&data[24..40]);

    Ok(AiclPacket {
        opcode, flags,
        message_id: Some(msg_id),
        correlation_id: corr_id,
        deadline_ms: u32::from_be_bytes([data[40], data[41], data[42], data[43]]),
        operands,
    })
}

fn uuid_v4_bytes() -> [u8; 16] {
    let mut b = [0u8; 16];
    let uuid = uuid::Uuid::new_v4();
    b.copy_from_slice(uuid.as_bytes());
    b
}

