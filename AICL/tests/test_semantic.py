"""Tests for aicl.semantic — model-facing semantic layer."""
from __future__ import annotations

import json

from aicl.semantic.types import (
    ModelIntent, ModelAction, ModelObservation, ModelRequest, ModelResult,
    SemanticError, ValidationError, INTENT_TO_OPCODE, _OPCODE_TO_INTENT,
)
from aicl.semantic.adapter import SemanticAdapter, GateCheck, GateResult
from aicl.semantic.grammar import AICLGrammar
from aicl.bin.constants import HEADER_SIZE
from aicl.bin.ops import OP_CLASSIFY, OP_EXECUTE, OP_TOOL_CALL, OP_MEMORY_READ


# ──────────────────────────────────────────────────────────────
# ModelAction tests
# ──────────────────────────────────────────────────────────────

def test_action_roundtrip_dict():
    action = ModelAction(
        intent=ModelIntent.CLASSIFY,
        target="sentiment",
        inputs={"text": "I love this"},
        params={"labels": ["pos", "neg"]},
        result={"label": "pos", "confidence": 0.95},
        confidence=0.95,
    )
    d = action.to_dict()
    action2 = ModelAction.from_dict(d)
    assert action2.intent == ModelIntent.CLASSIFY
    assert action2.target == "sentiment"
    assert action2.inputs["text"] == "I love this"
    assert action2.params["labels"] == ["pos", "neg"]
    assert action2.result["label"] == "pos"
    assert action2.confidence == 0.95


def test_action_validate_classify_requires_target():
    action = ModelAction(
        intent=ModelIntent.CLASSIFY,
        inputs={"text": "hello"},
    )
    errors = action.validate()
    assert any("target" in e for e in errors)


def test_action_validate_tool_requires_tool_name():
    action = ModelAction(
        intent=ModelIntent.CALL_TOOL,
        target="router",
        params={},
    )
    errors = action.validate()
    assert any("tool_name" in e for e in errors)


def test_action_validate_confidence_range():
    action = ModelAction(
        intent=ModelIntent.GENERATE,
        confidence=1.5,
    )
    errors = action.validate()
    assert any("Confidence" in e for e in errors)


def test_action_is_valid():
    action = ModelAction(
        intent=ModelIntent.CLASSIFY,
        target="cls",
        inputs={"text": "test"},
    )
    assert action.is_valid()


def test_action_all_intents():
    for intent in ModelIntent:
        action = ModelAction(intent=intent)
        d = action.to_dict()
        action2 = ModelAction.from_dict(d)
        assert action2.intent == intent


# ──────────────────────────────────────────────────────────────
# ModelObservation tests
# ──────────────────────────────────────────────────────────────

def test_observation_roundtrip():
    obs = ModelObservation(
        task="CLASSIFY",
        inputs={"text": "hello"},
        origin="orchestrator",
        confidence=0.9,
        params={"k": "v"},
    )
    d = obs.to_dict()
    obs2 = ModelObservation.from_dict(d)
    assert obs2.task == "CLASSIFY"
    assert obs2.inputs["text"] == "hello"
    assert obs2.origin == "orchestrator"
    assert obs2.confidence == 0.9
    assert obs2.params.get("k") == "v"


# ──────────────────────────────────────────────────────────────
# ModelRequest tests
# ──────────────────────────────────────────────────────────────

def test_request_roundtrip():
    req = ModelRequest(
        target_model="llama-3",
        task="GENERATE",
        prompt="Hello world",
        max_tokens=256,
        grammar="root ::= ...",
    )
    d = req.to_dict()
    req2 = ModelRequest.from_dict(d)
    assert req2.target_model == "llama-3"
    assert req2.task == "GENERATE"
    assert req2.prompt == "Hello world"
    assert req2.max_tokens == 256
    assert req2.grammar == "root ::= ..."


# ──────────────────────────────────────────────────────────────
# ModelResult tests
# ──────────────────────────────────────────────────────────────

def test_result_roundtrip():
    result = ModelResult(
        intent=ModelIntent.CLASSIFY,
        output={"label": "pos", "confidence": 0.95},
        confidence=0.95,
        tokens_used=42,
        latency_ms=12.5,
    )
    d = result.to_dict()
    result2 = ModelResult.from_dict(d)
    assert result2.intent == ModelIntent.CLASSIFY
    assert result2.output["label"] == "pos"
    assert result2.tokens_used == 42


def test_result_is_error():
    result = ModelResult(
        intent=ModelIntent.GENERATE,
        error="Model overloaded",
    )
    assert result.is_error


# ──────────────────────────────────────────────────────────────
# SemanticAdapter tests
# ──────────────────────────────────────────────────────────────

def test_adapter_action_to_bytes():
    adapter = SemanticAdapter(origin="test_model")
    action = ModelAction(
        intent=ModelIntent.CLASSIFY,
        target="orchestrator",
        inputs={"text": "I love this product"},
        params={"labels": ["pos", "neg"]},
        result={"label": "pos", "confidence": 0.95},
    )
    wire = adapter.action_to_bytes(action)
    assert isinstance(wire, bytes)
    assert len(wire) >= HEADER_SIZE
    assert wire[:4] == b"AICL"


def test_adapter_roundtrip_bytes():
    adapter = SemanticAdapter(origin="test_model")
    action = ModelAction(
        intent=ModelIntent.GENERATE,
        target="orchestrator",
        inputs={"prompt": "Explain AICL"},
        params={"max_tokens": 128},
    )
    wire = adapter.action_to_bytes(action)
    obs = adapter.bytes_to_observation(wire)
    assert obs.task == "GENERATE"
    assert obs.origin == "test_model"


def test_adapter_action_to_sl():
    adapter = SemanticAdapter(origin="test_model")
    action = ModelAction(
        intent=ModelIntent.CLASSIFY,
        target="orchestrator",
        inputs={"text": "hello"},
    )
    sl = adapter.action_to_sl(action)
    assert sl.startswith("AICL/1.0")
    assert "CLS" in sl


def test_adapter_result_to_bytes():
    adapter = SemanticAdapter(origin="test_model")
    result = ModelResult(
        intent=ModelIntent.CLASSIFY,
        output={"label": "pos", "confidence": 0.95},
        confidence=0.95,
        tokens_used=10,
        latency_ms=5.0,
    )
    wire = adapter.result_to_bytes(result)
    assert isinstance(wire, bytes)
    assert len(wire) >= HEADER_SIZE


def test_adapter_parse_json_output():
    adapter = SemanticAdapter(origin="test_model")
    raw = json.dumps({
        "intent": "CLASSIFY",
        "target": "sentiment",
        "inputs": {"text": "I love this"},
        "result": {"label": "pos", "confidence": 0.95},
    })
    action = adapter.parse_model_output(raw)
    assert action.intent == ModelIntent.CLASSIFY
    assert action.target == "sentiment"
    assert action.result["label"] == "pos"


def test_adapter_parse_empty_raises():
    adapter = SemanticAdapter(origin="test_model")
    try:
        adapter.parse_model_output("")
        assert False, "Should have raised SemanticError"
    except SemanticError:
        pass


def test_adapter_observation_to_request():
    adapter = SemanticAdapter(origin="test_model")
    obs = ModelObservation(
        task="CLASSIFY",
        inputs={"text": "hello world"},
        metadata={"model": "llama-3"},
    )
    grammar = AICLGrammar.for_classify(labels=["pos", "neg"])
    req = adapter.observation_to_request(obs, model_name="llama-3", grammar=grammar)
    assert req.target_model == "llama-3"
    assert req.task == "CLASSIFY"
    assert "hello world" in req.prompt
    assert req.grammar is not None


# ──────────────────────────────────────────────────────────────
# Gate tests
# ──────────────────────────────────────────────────────────────

class RejectAllGate(GateCheck):
    def check_outbound(self, action):
        return GateResult(allowed=False, reason="test rejection")

    def check_inbound(self, packet_view):
        return GateResult(allowed=False, reason="test rejection")


def test_gate_rejects_outbound():
    adapter = SemanticAdapter(origin="test", gate=RejectAllGate())
    action = ModelAction(
        intent=ModelIntent.CLASSIFY,
        target="test",
        inputs={"text": "hello"},
    )
    try:
        adapter.action_to_bytes(action)
        assert False, "Should have raised SemanticError"
    except SemanticError as e:
        assert "Gate rejected" in str(e)


def test_gate_rejects_inbound():
    adapter = SemanticAdapter(origin="test", gate=RejectAllGate())
    action = ModelAction(
        intent=ModelIntent.CLASSIFY,
        target="test",
        inputs={"text": "hello"},
    )
    wire = SemanticAdapter(origin="test").action_to_bytes(action)
    try:
        adapter.bytes_to_observation(wire)
        assert False, "Should have raised SemanticError"
    except SemanticError as e:
        assert "Gate rejected" in str(e)


# ──────────────────────────────────────────────────────────────
# Grammar tests
# ──────────────────────────────────────────────────────────────

def test_grammar_classify():
    g = AICLGrammar.for_classify(labels=["pos", "neg"])
    assert "CLASSIFY" in g.bnf
    assert g.json_schema is not None
    assert "CLASSIFY" in g.json_schema["properties"]["intent"]["enum"]


def test_grammar_generate():
    g = AICLGrammar.for_generate()
    assert "GENERATE" in g.bnf


def test_grammar_tool():
    g = AICLGrammar.for_tool(tool_name="weather")
    assert "CALL_TOOL" in g.bnf
    assert g.json_schema is not None


def test_grammar_all_intents():
    g = AICLGrammar.for_all()
    assert len(g.intents) == len(ModelIntent)
    for intent in ModelIntent:
        assert f'"{intent.value}"' in g.bnf


def test_grammar_intents():
    g = AICLGrammar.for_intents([ModelIntent.CLASSIFY, ModelIntent.GENERATE])
    assert len(g.intents) == 2
    assert "CLASSIFY" in g.bnf
    assert "GENERATE" in g.bnf


# ──────────────────────────────────────────────────────────────
# Opcode mapping tests
# ──────────────────────────────────────────────────────────────

def test_opcode_mapping_all_intents():
    for intent in ModelIntent:
        assert intent in INTENT_TO_OPCODE, f"Missing opcode for {intent}"


def test_reverse_opcode_mapping():
    for intent, opcode in INTENT_TO_OPCODE.items():
        assert opcode in _OPCODE_TO_INTENT


def test_classifier_opcodes():
    assert INTENT_TO_OPCODE[ModelIntent.CLASSIFY] == OP_CLASSIFY
    # GENERATE maps to the general-purpose Execute opcode, matching
    # core-rust's own generate() constructor — see semantic/types.py.
    assert INTENT_TO_OPCODE[ModelIntent.GENERATE] == OP_EXECUTE
    assert INTENT_TO_OPCODE[ModelIntent.CALL_TOOL] == OP_TOOL_CALL
    assert INTENT_TO_OPCODE[ModelIntent.READ_MEMORY] == OP_MEMORY_READ
