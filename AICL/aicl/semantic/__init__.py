"""aicl.semantic — Model-facing semantic layer for AICL.

Provides structured types that models emit and receive, without
exposing raw binary protocol details.

Usage:
    from aicl.semantic import (
        ModelIntent, ModelAction, ModelObservation, ModelRequest, ModelResult,
        SemanticAdapter, AICLGrammar,
    )

    # Constrained generation grammar for llama.cpp
    grammar = AICLGrammar.for_classify(labels=["pos", "neg"])

    # Adapter converts model output -> binary
    adapter = SemanticAdapter(origin="my_model")
    wire_bytes = adapter.action_to_bytes(action)

    # Binary -> model input
    obs = adapter.bytes_to_observation(wire_bytes)
    request = adapter.observation_to_request(obs, grammar=grammar)
"""
from aicl.semantic.types import (
    ModelIntent,
    ModelAction,
    ModelObservation,
    ModelRequest,
    ModelResult,
    SemanticError,
    ValidationError,
    INTENT_TO_OPCODE,
)
from aicl.semantic.adapter import SemanticAdapter, GateCheck, GateResult
from aicl.semantic.grammar import AICLGrammar

__all__ = [
    "ModelIntent",
    "ModelAction",
    "ModelObservation",
    "ModelRequest",
    "ModelResult",
    "SemanticError",
    "ValidationError",
    "INTENT_TO_OPCODE",
    "SemanticAdapter",
    "GateCheck",
    "GateResult",
    "AICLGrammar",
]
