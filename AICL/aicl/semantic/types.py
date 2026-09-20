"""Model-facing semantic types for AICL.

These types form the boundary between model output and the AICL binary
protocol. A model never sees raw bytes — it emits and receives these
structured types. The adapter converts between these and AICL packets.

Design constraints:
    - No natural language on the wire.
    - Structured generation produces valid ModelAction JSON.
    - Grammar-constrained decoding enforces valid output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "ModelIntent",
    "ModelAction",
    "ModelObservation",
    "ModelRequest",
    "ModelResult",
    "SemanticError",
    "ValidationError",
]


class ModelIntent(Enum):
    """What the model wants to do.

    Each intent maps to an AICL opcode:
        CLASSIFY       -> OP_CLS
        GENERATE       -> OP_GEN
        REASON         -> OP_RSN
        EMBED          -> OP_EMB
        RANK           -> OP_RNK
        SUMMARIZE      -> OP_SUM
        SYNTHESIZE     -> OP_SYN
        VERIFY         -> OP_VRF
        EXTRACT        -> OP_XTR
        TRANSFORM      -> OP_TRN
        PLAN           -> OP_PLN
        EVALUATE       -> OP_EVL
        CALL_MODEL     -> OP_EXE_MODEL
        CALL_TOOL      -> OP_EXE_TOOL
        READ_MEMORY    -> OP_MEM_READ
        WRITE_MEMORY   -> OP_MEM_WRITE
        DELETE_MEMORY  -> OP_MEM_DELETE
        INDEX_QUERY    -> OP_IDX_QUERY
        INDEX_UPSERT   -> OP_IDX_UPSERT
        ROUTE          -> OP_RTE_ROUTE
        FILTER         -> OP_RTE_DISCOVER
        RETURN_RESULT  -> OP_RESPONSE
        STREAM_RESULT  -> OP_GEN (with STREAM_CHUNK flag)
        CANCEL         -> OP_SYS_CANCEL
    """

    CLASSIFY = "CLASSIFY"
    GENERATE = "GENERATE"
    REASON = "REASON"
    EMBED = "EMBED"
    RANK = "RANK"
    SUMMARIZE = "SUMMARIZE"
    SYNTHESIZE = "SYNTHESIZE"
    VERIFY = "VERIFY"
    EXTRACT = "EXTRACT"
    TRANSFORM = "TRANSFORM"
    PLAN = "PLAN"
    EVALUATE = "EVALUATE"
    CALL_MODEL = "CALL_MODEL"
    CALL_TOOL = "CALL_TOOL"
    READ_MEMORY = "READ_MEMORY"
    WRITE_MEMORY = "WRITE_MEMORY"
    DELETE_MEMORY = "DELETE_MEMORY"
    INDEX_QUERY = "INDEX_QUERY"
    INDEX_UPSERT = "INDEX_UPSERT"
    ROUTE = "ROUTE"
    FILTER = "FILTER"
    RETURN_RESULT = "RETURN_RESULT"
    STREAM_RESULT = "STREAM_RESULT"
    CANCEL = "CANCEL"


# Intent -> AICL opcode mapping (defined in aicl.bin.ops)
# Opcode values match core-rust's AiclOpcode exactly (see aicl/bin/ops.py)
# — the ISA-spec-aligned rewrite of this codebase's wire format. Where
# core-rust has no dedicated opcode for a given intent (it collapses many
# "do a language-model thing" intents into the general-purpose Execute
# opcode — see core-rust/src/packet.rs's rank/summarize/extract/
# transform/plan/evaluate constructors, which all use AiclOpcode::Execute
# too), this mapping does the same rather than inventing an opcode Rust
# wouldn't recognize.
INTENT_TO_OPCODE: Dict[ModelIntent, int] = {
    ModelIntent.CLASSIFY: 0x43,       # OP_CLASSIFY
    ModelIntent.GENERATE: 0x46,       # OP_EXECUTE (matches Rust's generate())
    ModelIntent.REASON: 0x44,         # OP_REASON
    ModelIntent.EMBED: 0x42,          # OP_EMBED
    ModelIntent.RANK: 0x46,           # OP_EXECUTE (matches Rust's rank())
    ModelIntent.SUMMARIZE: 0x46,      # OP_EXECUTE (matches Rust's summarize())
    ModelIntent.SYNTHESIZE: 0x46,     # OP_EXECUTE (no dedicated Rust opcode)
    ModelIntent.VERIFY: 0x44,         # OP_REASON (matches Rust's verify())
    ModelIntent.EXTRACT: 0x46,        # OP_EXECUTE (matches Rust's extract())
    ModelIntent.TRANSFORM: 0x46,      # OP_EXECUTE (matches Rust's transform())
    ModelIntent.PLAN: 0x46,           # OP_EXECUTE (matches Rust's plan())
    ModelIntent.EVALUATE: 0x46,       # OP_EXECUTE (matches Rust's evaluate())
    ModelIntent.CALL_MODEL: 0x40,     # OP_MODEL_CALL
    ModelIntent.CALL_TOOL: 0x41,      # OP_TOOL_CALL
    ModelIntent.READ_MEMORY: 0x20,    # OP_MEMORY_READ
    ModelIntent.WRITE_MEMORY: 0x21,   # OP_MEMORY_WRITE
    ModelIntent.DELETE_MEMORY: 0x22,  # OP_MEMORY_DELETE
    ModelIntent.INDEX_QUERY: 0x23,    # OP_INDEX_QUERY
    ModelIntent.INDEX_UPSERT: 0x24,   # OP_INDEX_UPSERT
    ModelIntent.ROUTE: 0x03,          # OP_DISCOVER (closest match; Rust has no dedicated routing opcode)
    ModelIntent.FILTER: 0x46,         # OP_EXECUTE (no dedicated Rust opcode)
    ModelIntent.RETURN_RESULT: 0x11,  # OP_RETURN (matches Rust's return_response())
    ModelIntent.STREAM_RESULT: 0x12,  # OP_STREAM
    ModelIntent.CANCEL: 0x09,         # OP_CANCEL
}

# Reverse mapping
_OPCODE_TO_INTENT: Dict[int, ModelIntent] = {v: k for k, v in INTENT_TO_OPCODE.items()}


class SemanticError(Exception):
    """Base exception for semantic layer errors."""


class ValidationError(SemanticError):
    """Raised when a ModelAction fails validation."""

    def __init__(self, errors: List[str], action: Optional["ModelAction"] = None):
        self.errors = errors
        self.action = action
        super().__init__(f"Validation failed: {'; '.join(errors)}")


@dataclass(slots=True)
class ModelAction:
    """Structured action a model wants to perform.

    This is what the model emits. The adapter converts it to an AICL
    binary packet. The model never sees raw bytes.

    The action is a tagged union: the intent determines which fields
    are relevant.

    Example (classifier output):
        ModelAction(
            intent=ModelIntent.CLASSIFY,
            target="sentiment_classifier",
            inputs={"text": "I love this product"},
            params={"labels": ["positive", "negative"]},
            result={"label": "positive", "confidence": 0.95},
        )
    """

    intent: ModelIntent
    target: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    stream: bool = False
    cancel: bool = False
    correlation_id: str = ""
    deadline_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate this action. Returns list of errors (empty = valid)."""
        errors: List[str] = []
        if not isinstance(self.intent, ModelIntent):
            errors.append(f"Invalid intent: {self.intent!r}")
        if not (0.0 <= self.confidence <= 1.0):
            errors.append(f"Confidence out of range: {self.confidence}")
        if self.deadline_ms < 0:
            errors.append(f"Negative deadline: {self.deadline_ms}")
        if self.intent in (
            ModelIntent.CLASSIFY,
            ModelIntent.CALL_MODEL,
            ModelIntent.CALL_TOOL,
            ModelIntent.READ_MEMORY,
            ModelIntent.WRITE_MEMORY,
        ) and not self.target:
            errors.append(f"{self.intent.value} requires a target")
        if self.intent == ModelIntent.CALL_TOOL and "tool_name" not in self.params:
            errors.append("CALL_TOOL requires params.tool_name")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"intent": self.intent.value}
        if self.target:
            d["target"] = self.target
        if self.inputs:
            d["inputs"] = self.inputs
        if self.params:
            d["params"] = self.params
        if self.result:
            d["result"] = self.result
        if self.confidence != 1.0:
            d["confidence"] = self.confidence
        if self.stream:
            d["stream"] = True
        if self.cancel:
            d["cancel"] = True
        if self.correlation_id:
            d["correlation_id"] = self.correlation_id
        if self.deadline_ms:
            d["deadline_ms"] = self.deadline_ms
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelAction":
        intent_str = d.get("intent", "")
        try:
            intent = ModelIntent(intent_str)
        except ValueError:
            raise SemanticError(f"Unknown intent: {intent_str!r}")
        return cls(
            intent=intent,
            target=d.get("target", ""),
            inputs=d.get("inputs", {}),
            params=d.get("params", {}),
            result=d.get("result", {}),
            confidence=d.get("confidence", 1.0),
            stream=d.get("stream", False),
            cancel=d.get("cancel", False),
            correlation_id=d.get("correlation_id", ""),
            deadline_ms=d.get("deadline_ms", 0),
            metadata=d.get("metadata", {}),
        )


@dataclass(slots=True)
class ModelObservation:
    """What a model receives as input context.

    This is the inbound side: binary -> AICL instruction -> Gate ->
    ModelObservation -> model adapter -> model.

    The observation carries the task, context, and any prior results
    the model needs to process.
    """

    task: str
    context: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    schema_id: str = ""
    trace_id: str = ""
    origin: str = ""
    confidence: float = 1.0
    deadline_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"task": self.task}
        if self.context:
            d["context"] = self.context
        if self.inputs:
            d["inputs"] = self.inputs
        if self.params:
            d["params"] = self.params
        if self.schema_id:
            d["schema_id"] = self.schema_id
        if self.trace_id:
            d["trace_id"] = self.trace_id
        if self.origin:
            d["origin"] = self.origin
        if self.confidence != 1.0:
            d["confidence"] = self.confidence
        if self.deadline_ms:
            d["deadline_ms"] = self.deadline_ms
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelObservation":
        return cls(
            task=d.get("task", ""),
            context=d.get("context", {}),
            inputs=d.get("inputs", {}),
            params=d.get("params", {}),
            schema_id=d.get("schema_id", ""),
            trace_id=d.get("trace_id", ""),
            origin=d.get("origin", ""),
            confidence=d.get("confidence", 1.0),
            deadline_ms=d.get("deadline_ms", 0),
            metadata=d.get("metadata", {}),
        )


@dataclass(slots=True)
class ModelRequest:
    """Outbound request to a model.

    This is what gets sent to a model for inference. The model adapter
    constructs this from a ModelObservation + available capabilities.
    """

    target_model: str
    task: str
    prompt: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 512
    temperature: float = 0.7
    stop: List[str] = field(default_factory=list)
    stream: bool = False
    grammar: Optional[str] = None
    correlation_id: str = ""
    deadline_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "target_model": self.target_model,
            "task": self.task,
        }
        if self.prompt:
            d["prompt"] = self.prompt
        if self.inputs:
            d["inputs"] = self.inputs
        if self.params:
            d["params"] = self.params
        if self.max_tokens != 512:
            d["max_tokens"] = self.max_tokens
        if self.temperature != 0.7:
            d["temperature"] = self.temperature
        if self.stop:
            d["stop"] = self.stop
        if self.stream:
            d["stream"] = True
        if self.grammar:
            d["grammar"] = self.grammar
        if self.correlation_id:
            d["correlation_id"] = self.correlation_id
        if self.deadline_ms:
            d["deadline_ms"] = self.deadline_ms
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelRequest":
        return cls(
            target_model=d.get("target_model", ""),
            task=d.get("task", ""),
            prompt=d.get("prompt", ""),
            inputs=d.get("inputs", {}),
            params=d.get("params", {}),
            max_tokens=d.get("max_tokens", 512),
            temperature=d.get("temperature", 0.7),
            stop=d.get("stop", []),
            stream=d.get("stream", False),
            grammar=d.get("grammar"),
            correlation_id=d.get("correlation_id", ""),
            deadline_ms=d.get("deadline_ms", 0),
            metadata=d.get("metadata", {}),
        )


@dataclass(slots=True)
class ModelResult:
    """Result from a model.

    This is what comes back after inference. The adapter converts it
    to an AICL binary response packet.
    """

    intent: ModelIntent
    output: Any = None
    confidence: float = 1.0
    tokens_used: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None
    done: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "done": self.done,
        }
        if self.output is not None:
            d["output"] = self.output
        if self.tokens_used:
            d["tokens_used"] = self.tokens_used
        if self.latency_ms:
            d["latency_ms"] = self.latency_ms
        if self.error:
            d["error"] = self.error
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelResult":
        intent_str = d.get("intent", "")
        try:
            intent = ModelIntent(intent_str)
        except ValueError:
            intent = ModelIntent.RETURN_RESULT
        return cls(
            intent=intent,
            output=d.get("output"),
            confidence=d.get("confidence", 1.0),
            tokens_used=d.get("tokens_used", 0),
            latency_ms=d.get("latency_ms", 0.0),
            error=d.get("error"),
            done=d.get("done", True),
            metadata=d.get("metadata", {}),
        )
