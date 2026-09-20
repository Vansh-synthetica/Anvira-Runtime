"""Model adapter: bridges model output to AICL binary protocol.

Outbound pipeline (model -> wire):
    Model output -> ModelAction -> validate -> Gate -> Packet -> binary

Inbound pipeline (wire -> model):
    binary -> decode -> PacketView -> Gate -> ModelObservation -> ModelRequest

The model never sees raw bytes. The adapter handles all translation.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aicl.bin.codec_api import encode as bin_encode
from aicl.bin.codec_api import decode as bin_decode
from aicl.bin.codec_packet import Packet as BinPacket
from aicl.bin.codec_view import PacketView
from aicl.bin.types import Identity, Symbol, ErrorInfo, ToolInvocation, ModelInvocation
from aicl.bin.symbol_types import S_STRING, S_F64, S_I64, S_BOOL, S_LIST, S_NULL
from aicl.semantic.types import (
    ModelIntent,
    ModelAction,
    ModelObservation,
    ModelRequest,
    ModelResult,
    SemanticError,
    ValidationError,
    INTENT_TO_OPCODE,
    _OPCODE_TO_INTENT,
)
from aicl.semantic.grammar import AICLGrammar

__all__ = [
    "SemanticAdapter",
    "GateCheck",
    "GateResult",
]

# Map ModelIntent -> AICL-SL operator code
_INTENT_TO_SL_OP = {
    ModelIntent.CLASSIFY: "CLS",
    ModelIntent.GENERATE: "GEN",
    ModelIntent.REASON: "RSN",
    ModelIntent.EMBED: "EMB",
    ModelIntent.RANK: "RNK",
    ModelIntent.SUMMARIZE: "SUM",
    ModelIntent.SYNTHESIZE: "SYN",
    ModelIntent.VERIFY: "VRF",
    ModelIntent.EXTRACT: "XTR",
    ModelIntent.TRANSFORM: "TRN",
    ModelIntent.PLAN: "PLN",
    ModelIntent.EVALUATE: "EVL",
    ModelIntent.CALL_MODEL: "EXE_MODEL",
    ModelIntent.CALL_TOOL: "EXE_TOOL",
    ModelIntent.READ_MEMORY: "MEM_READ",
    ModelIntent.WRITE_MEMORY: "MEM_WRITE",
    ModelIntent.DELETE_MEMORY: "MEM_DELETE",
    ModelIntent.INDEX_QUERY: "IDX_QUERY",
    ModelIntent.INDEX_UPSERT: "IDX_UPSERT",
    ModelIntent.ROUTE: "RTE",
    ModelIntent.FILTER: "FLT",
    ModelIntent.RETURN_RESULT: "RES",
    ModelIntent.STREAM_RESULT: "GEN",
    ModelIntent.CANCEL: "CANCEL",
}


# ──────────────────────────────────────────────────────────────
# Gate interface
# ──────────────────────────────────────────────────────────────

@dataclass(slots=True)
class GateResult:
    """Result of a gate check."""
    allowed: bool
    reason: str = ""
    audit_entry: Optional[Dict[str, Any]] = None


class GateCheck:
    """Abstract gate for capability-based access control.

    Implement this to plug in your own gate logic. The default
    implementation allows everything.
    """

    def check_outbound(self, action: ModelAction) -> GateResult:
        """Check if an outbound action is allowed."""
        return GateResult(allowed=True)

    def check_inbound(self, packet_view: PacketView) -> GateResult:
        """Check if an inbound packet is allowed."""
        return GateResult(allowed=True)


class _AllowAllGate(GateCheck):
    """Default gate that allows everything."""


# ──────────────────────────────────────────────────────────────
# Symbol helpers (semantic <-> binary)
# ──────────────────────────────────────────────────────────────

def _to_binary_symbol(value: Any, type_prefix: str = "") -> Symbol:
    """Convert a Python value to a binary Symbol.

    Dicts encode as JSON text under S_STRING — core-rust's operand set has
    no dedicated JSON tag (the old S_JSON tag was always just a UTF-8
    string on the wire anyway; "this string is JSON" was purely an
    application-level convention, not a real wire distinction), so this
    matches what Rust would do with the same data.
    """
    if isinstance(value, str):
        return Symbol(tag=S_STRING, value=value)
    elif isinstance(value, bool):
        return Symbol(tag=S_BOOL, value=value)
    elif isinstance(value, int):
        return Symbol(tag=S_I64, value=value)
    elif isinstance(value, float):
        return Symbol(tag=S_F64, value=value)
    elif isinstance(value, (list, tuple)):
        return Symbol(tag=S_LIST, value=[_to_binary_symbol(v) for v in value])
    elif isinstance(value, dict):
        return Symbol(tag=S_STRING, value=json.dumps(value, separators=(",", ":")))
    elif value is None:
        return Symbol(tag=S_NULL, value=None)
    else:
        return Symbol(tag=S_STRING, value=str(value))


# ──────────────────────────────────────────────────────────────
# SemanticAdapter
# ──────────────────────────────────────────────────────────────

class SemanticAdapter:
    """Bridges model output to the AICL binary protocol.

    This is the single entry point for the model-facing semantic layer.
    It owns the outbound and inbound pipelines.

    Usage:
        adapter = SemanticAdapter(origin="classifier_model")
        adapter.set_gate(MyGate())

        # Outbound: model output -> binary
        action = ModelAction(
            intent=ModelIntent.CLASSIFY,
            target="orchestrator",
            inputs={"text": "I love this"},
            result={"label": "pos", "confidence": 0.95},
        )
        wire_bytes = adapter.action_to_bytes(action)

        # Inbound: binary -> model input
        observation = adapter.bytes_to_observation(wire_bytes)
        # Feed observation to model...
    """

    def __init__(
        self,
        origin: str = "model",
        gate: Optional[GateCheck] = None,
        identity_type: int = 0x04,  # IDENTITY_MODEL
    ):
        self.origin = origin
        self._gate: GateCheck = gate or _AllowAllGate()
        self._identity_type = identity_type
        self._session_counter: int = 0

    def set_gate(self, gate: GateCheck) -> None:
        self._gate = gate

    # ──────────────────────────────────────────────────────────
    # Outbound: ModelAction -> binary
    # ──────────────────────────────────────────────────────────

    def action_to_packet(self, action: ModelAction) -> BinPacket:
        """Convert a ModelAction to a binary AICL Packet.

        Pipeline:
            ModelAction -> validate -> Gate -> BinPacket
        """
        errors = action.validate()
        if errors:
            raise ValidationError(errors, action)

        gate_result = self._gate.check_outbound(action)
        if not gate_result.allowed:
            raise SemanticError(f"Gate rejected action: {gate_result.reason}")

        opcode = INTENT_TO_OPCODE.get(action.intent, 0x01)
        symbols = self._build_symbols(action)
        intent_str = action.intent.value

        session_id = uuid.uuid4().bytes
        msg_id = uuid.uuid4().bytes
        corr_id = (
            uuid.UUID(action.correlation_id).bytes
            if action.correlation_id
            else b"\x00" * 16
        )

        origin_identity = Identity(
            id_type=self._identity_type,
            name=self.origin,
        )

        flags = 0x0001  # FLAG_REQUEST
        if action.stream:
            flags |= 0x0080  # FLAG_STREAM_CHUNK
        if action.cancel:
            flags |= 0x0020  # FLAG_CANCEL

        pkt = BinPacket(
            flags=flags,
            session_id=session_id,
            message_id=msg_id,
            correlation_id=corr_id,
            origin=origin_identity,
            targets=[action.target] if action.target else [],
            operation=opcode,
            intent=intent_str,
            symbols=symbols,
            confidence=action.confidence,
            deadline_ms=action.deadline_ms,
            metadata={"semantic_intent": action.intent.value, **action.metadata},
        )

        if action.intent == ModelIntent.CALL_TOOL and action.params.get("tool_name"):
            pkt.tool_invocation = ToolInvocation(
                tool_name=action.params["tool_name"],
                parameters_json=json.dumps(action.params.get("parameters", {})),
                timeout_ms=action.params.get("timeout_ms", 5000),
            )

        if action.intent == ModelIntent.CALL_MODEL and action.target:
            pkt.model_invocation = ModelInvocation(
                model_name=action.target,
                task=action.params.get("task", action.intent.value),
                parameters_json=json.dumps(action.params),
                timeout_ms=action.params.get("timeout_ms", 5000),
            )

        return pkt

    def action_to_bytes(self, action: ModelAction, checksum: bool = False) -> bytes:
        """Convert a ModelAction directly to wire bytes."""
        pkt = self.action_to_packet(action)
        return bin_encode(pkt, checksum=checksum)

    def action_to_sl(self, action: ModelAction) -> str:
        """Convert a ModelAction to AICL-SL text (for debugging).

        Builds the wire string directly to avoid import chain issues
        with the root-level lang/ directory.
        """
        import time
        parts = ["AICL/1.0"]
        sid = f"s-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
        parts.append(f"SID:{sid}")
        parts.append(f"ORI:{self.origin}")
        if action.target:
            parts.append(f"TGT:{action.target}")
        parts.append(f"OP:{_INTENT_TO_SL_OP.get(action.intent, action.intent.value)}")

        syms = self._build_sl_symbols(action)
        if syms:
            encoded = []
            for prefix, val in syms:
                if prefix == "S":
                    escaped = str(val).replace('"', '\\"')
                    encoded.append(f'S:"{escaped}"')
                elif prefix == "N":
                    encoded.append(f"N:{val}")
                elif prefix == "B":
                    encoded.append(f"B:{'true' if val else 'false'}")
                elif prefix == "T":
                    encoded.append(f"T:{val}")
                elif prefix == "J":
                    import json as _json
                    encoded.append(f"J:{_json.dumps(val)}")
                else:
                    encoded.append(f"S:{val}")
            parts.append(f"SYM:{','.join(encoded)}")

        parts.append(f"CNF:{action.confidence}")

        meta_parts = []
        for k, v in action.metadata.items():
            meta_parts.append(f"{k}={v}")
        meta_parts.append(f"semantic_intent={action.intent.value}")
        parts.append(f"META:{';'.join(meta_parts)}")

        return "|".join(parts)

    # ──────────────────────────────────────────────────────────
    # Inbound: binary -> ModelObservation / ModelRequest
    # ──────────────────────────────────────────────────────────

    def bytes_to_observation(self, data: bytes) -> ModelObservation:
        """Decode binary bytes into a ModelObservation.

        Pipeline:
            binary -> decode -> PacketView -> Gate -> ModelObservation
        """
        pv = bin_decode(data)
        gate_result = self._gate.check_inbound(pv)
        if not gate_result.allowed:
            raise SemanticError(f"Gate rejected inbound: {gate_result.reason}")

        return self._packet_view_to_observation(pv)

    def packet_view_to_observation(self, pv: PacketView) -> ModelObservation:
        """Convert a PacketView to a ModelObservation (public alias)."""
        gate_result = self._gate.check_inbound(pv)
        if not gate_result.allowed:
            raise SemanticError(f"Gate rejected inbound: {gate_result.reason}")
        return self._packet_view_to_observation(pv)

    def _packet_view_to_observation(self, pv: PacketView) -> ModelObservation:
        """Internal: PacketView -> ModelObservation."""
        intent_str = pv.intent or ""
        try:
            intent = ModelIntent(intent_str)
        except ValueError:
            intent = ModelIntent.RETURN_RESULT

        task = intent_str
        inputs: Dict[str, Any] = {}
        params: Dict[str, Any] = {}

        syms = pv.symbols
        for sym in syms:
            if isinstance(sym, tuple) and len(sym) == 2:
                tag, val = sym
                if tag == "S":
                    if "text" not in inputs:
                        inputs["text"] = val
                    else:
                        inputs.setdefault("values", []).append(val)
                elif tag == "J":
                    try:
                        inputs.update(json.loads(val) if isinstance(val, str) else val)
                    except (json.JSONDecodeError, TypeError):
                        inputs["json_data"] = val
                elif tag == "K":
                    params[str(val)] = True
                elif tag == "T":
                    inputs.setdefault("tags", []).append(val)
                elif tag == "N":
                    inputs.setdefault("numbers", []).append(val)

        metadata = pv.metadata if hasattr(pv, "metadata") and pv.metadata else {}

        return ModelObservation(
            task=task,
            context={"intent": intent_str},
            inputs=inputs,
            params=params,
            schema_id=pv.schema_id if hasattr(pv, "schema_id") else "",
            trace_id=pv.session_id.hex() if pv.session_id else "",
            origin=pv.origin.name if pv.origin else "",
            confidence=pv.confidence,
            deadline_ms=pv.deadline_ms if hasattr(pv, "deadline_ms") else 0,
            metadata=metadata,
        )

    def observation_to_request(
        self,
        obs: ModelObservation,
        model_name: str = "",
        grammar: Optional[AICLGrammar] = None,
    ) -> ModelRequest:
        """Convert a ModelObservation to a ModelRequest for inference."""
        prompt_parts: List[str] = []
        task = obs.task.upper() if obs.task else "GENERATE"

        if obs.inputs.get("text"):
            prompt_parts.append(obs.inputs["text"])
        elif obs.inputs:
            for k, v in obs.inputs.items():
                if isinstance(v, str):
                    prompt_parts.append(f"{k}: {v}")
                else:
                    prompt_parts.append(f"{k}: {json.dumps(v)}")

        prompt = "\n".join(prompt_parts) if prompt_parts else ""

        return ModelRequest(
            target_model=model_name or obs.context.get("model", ""),
            task=task,
            prompt=prompt,
            inputs=obs.inputs,
            params=obs.params,
            grammar=grammar.bnf if grammar else None,
            correlation_id=obs.trace_id,
            deadline_ms=obs.deadline_ms,
            metadata=obs.metadata,
        )

    # ──────────────────────────────────────────────────────────
    # Result -> binary response
    # ──────────────────────────────────────────────────────────

    def result_to_bytes(
        self,
        result: ModelResult,
        correlation_id: str = "",
        session_id: Optional[bytes] = None,
    ) -> bytes:
        """Convert a ModelResult to a binary response packet."""
        opcode = INTENT_TO_OPCODE.get(result.intent, 0x02)
        symbols = []

        if result.output is not None:
            if isinstance(result.output, dict):
                symbols.append(_to_binary_symbol(result.output, "J"))
            elif isinstance(result.output, str):
                symbols.append(_to_binary_symbol(result.output))
            else:
                symbols.append(_to_binary_symbol(str(result.output)))

        if result.error:
            symbols.append(_to_binary_symbol(result.error, "S"))

        corr_id = (
            uuid.UUID(correlation_id).bytes
            if correlation_id
            else b"\x00" * 16
        )

        pkt = BinPacket(
            flags=0x0002,  # FLAG_RESPONSE
            session_id=session_id or uuid.uuid4().bytes,
            message_id=uuid.uuid4().bytes,
            correlation_id=corr_id,
            origin=Identity(id_type=self._identity_type, name=self.origin),
            operation=opcode,
            intent=result.intent.value,
            symbols=symbols,
            confidence=result.confidence,
            metadata={
                "semantic_intent": result.intent.value,
                "tokens_used": result.tokens_used,
                "latency_ms": result.latency_ms,
                **result.metadata,
            },
        )

        if result.error:
            pkt.error_info = ErrorInfo(
                error_code=0x0001,
                severity=1,
                message=result.error,
            )

        return bin_encode(pkt)

    # ──────────────────────────────────────────────────────────
    # Convenience: parse raw model text output
    # ──────────────────────────────────────────────────────────

    def parse_model_output(self, raw: str) -> ModelAction:
        """Parse raw model text output into a ModelAction.

        Tries JSON first (structured generation output), then falls
        back to a simple parser for unconstrained output.
        """
        raw = raw.strip()
        if not raw:
            raise SemanticError("Empty model output")

        # Try JSON (structured generation)
        try:
            d = json.loads(raw)
            if isinstance(d, dict):
                return ModelAction.from_dict(d)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: extract intent keyword from text
        upper = raw.upper()
        for intent in ModelIntent:
            if intent.value in upper:
                return ModelAction(
                    intent=intent,
                    inputs={"text": raw},
                )

        # Default: treat as generation result
        return ModelAction(
            intent=ModelIntent.RETURN_RESULT,
            result={"text": raw},
        )

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _build_symbols(self, action: ModelAction) -> list:
        """Build binary symbols from a ModelAction."""
        symbols = []

        if action.inputs.get("text"):
            symbols.append(_to_binary_symbol(action.inputs["text"]))

        for k, v in action.inputs.items():
            if k == "text":
                continue
            symbols.append(_to_binary_symbol(v))

        if action.params:
            symbols.append(_to_binary_symbol(action.params, "J"))

        if action.result:
            symbols.append(_to_binary_symbol(action.result, "J"))

        return symbols

    def _build_sl_symbols(self, action: ModelAction) -> list:
        """Build AICL-SL typed symbols from a ModelAction."""
        symbols = []

        if action.inputs.get("text"):
            symbols.append(("S", action.inputs["text"]))

        for k, v in action.inputs.items():
            if k == "text":
                continue
            if isinstance(v, str):
                symbols.append(("S", v))
            elif isinstance(v, (int, float)):
                symbols.append(("N", v))
            elif isinstance(v, bool):
                symbols.append(("B", v))
            elif isinstance(v, dict):
                symbols.append(("J", v))

        if action.result:
            symbols.append(("J", action.result))

        return symbols
