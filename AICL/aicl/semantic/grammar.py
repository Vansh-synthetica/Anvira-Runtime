"""Grammar-constrained generation for AICL semantic actions.

Generates BNF grammars that force model output to be valid AICL
ModelAction JSON. Supports llama.cpp grammar-constrained decoding.

Usage:
    grammar = AICLGrammar.for_classify(labels=["pos", "neg"])
    # grammar.bnf  -> BNF string for llama.cpp
    # grammar.json_schema -> JSON Schema for validation

    # With llama.cpp Python bindings:
    result = llm.create_completion(
        prompt="Classify: I love this product\n",
        grammar=grammar.bnf,
    )
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aicl.semantic.types import ModelIntent

__all__ = [
    "AICLGrammar",
    "GRAMMAR_ACTION_TEMPLATE",
    "GRAMMAR_CLASSIFY_TEMPLATE",
    "GRAMMAR_GENERATE_TEMPLATE",
    "GRAMMAR_TOOL_TEMPLATE",
]


# ──────────────────────────────────────────────────────────────
# BNF Grammar Templates
# ──────────────────────────────────────────────────────────────

# Root grammar: a complete ModelAction in JSON
GRAMMAR_ACTION_TEMPLATE = """root   ::= "{" action_fields "}"
action_fields ::= "\"intent\"" ":" intent_value ("," other_fields)*

intent_value  ::= "{intent_values}"

other_fields  ::= field_target | field_inputs | field_params | field_result
                | field_confidence | field_stream | field_cancel
                | field_correlation_id | field_deadline_ms | field_metadata

field_target  ::= "\"target\"" ":" string
field_inputs  ::= "\"inputs\"" ":" object
field_params  ::= "\"params\"" ":" object
field_result  ::= "\"result\"" ":" object
field_confidence ::= "\"confidence\"" ":" number
field_stream  ::= "\"stream\"" ":" bool
field_cancel  ::= "\"cancel\"" ":" bool
field_correlation_id ::= "\"correlation_id\"" ":" string
field_deadline_ms ::= "\"deadline_ms\"" ":" integer
field_metadata ::= "\"metadata\"" ":" object

string  ::= "\"" chars "\""
chars   ::= [^"]*
number  ::= [0-9]+ ("." [0-9]+)?
integer ::= [0-9]+
bool    ::= "true" | "false"
object  ::= "{" (string ":" value ("," string ":" value)*)? "}"
value   ::= string | number | integer | bool | object | array
array   ::= "[" (value ("," value)*)? "]"
"""


GRAMMAR_CLASSIFY_TEMPLATE = """root   ::= "{" classify_fields "}"
classify_fields ::= "\"intent\"" ":" "\"CLASSIFY\"" "," "\"target\"" ":" string "," "\"inputs\"" ":" inputs_obj ("," other_fields)*

other_fields ::= field_params | field_result | field_confidence | field_metadata
field_params ::= "\"params\"" ":" params_obj
field_result ::= "\"result\"" ":" result_obj
field_confidence ::= "\"confidence\"" ":" number
field_metadata ::= "\"metadata\"" ":" object

inputs_obj  ::= "{" "\"text\"" ":" string "}"
params_obj  ::= "{" "\"labels\"" ":" labels_array ("," string ":" value)* "}"
labels_array ::= "[" string ("," string)* "]"
result_obj  ::= "{" "\"label\"" ":" string "," "\"confidence\"" ":" number "}"

string  ::= "\"" chars "\""
chars   ::= [^"]*
number  ::= [0-9]+ ("." [0-9]+)?
integer ::= [0-9]+
bool    ::= "true" | "false"
object  ::= "{" (string ":" value ("," string ":" value)*)? "}"
value   ::= string | number | integer | bool | object | array
array   ::= "[" (value ("," value)*)? "]"
"""


GRAMMAR_GENERATE_TEMPLATE = """root   ::= "{" gen_fields "}"
gen_fields ::= "\"intent\"" ":" "\"GENERATE\"" ("," other_fields)*

other_fields ::= field_target | field_inputs | field_params | field_result
                | field_confidence | field_stream | field_metadata

field_target  ::= "\"target\"" ":" string
field_inputs  ::= "\"inputs\"" ":" object
field_params  ::= "\"params\"" ":" object
field_result  ::= "\"result\"" ":" object
field_confidence ::= "\"confidence\"" ":" number
field_stream  ::= "\"stream\"" ":" bool
field_metadata ::= "\"metadata\"" ":" object

string  ::= "\"" chars "\""
chars   ::= [^"]*
number  ::= [0-9]+ ("." [0-9]+)?
integer ::= [0-9]+
bool    ::= "true" | "false"
object  ::= "{" (string ":" value ("," string ":" value)*)? "}"
value   ::= string | number | integer | bool | object | array
array   ::= "[" (value ("," value)*)? "]"
"""


GRAMMAR_TOOL_TEMPLATE = """root   ::= "{" tool_fields "}"
tool_fields ::= "\"intent\"" ":" "\"CALL_TOOL\"" "," "\"target\"" ":" string "," "\"params\"" ":" params_obj ("," other_fields)*

other_fields ::= field_inputs | field_result | field_confidence | field_metadata
field_inputs  ::= "\"inputs\"" ":" object
field_result  ::= "\"result\"" ":" object
field_confidence ::= "\"confidence\"" ":" number
field_metadata ::= "\"metadata\"" ":" object

params_obj ::= "{" "\"tool_name\"" ":" string ("," string ":" value)* "}"

string  ::= "\"" chars "\""
chars   ::= [^"]*
number  ::= [0-9]+ ("." [0-9]+)?
integer ::= [0-9]+
bool    ::= "true" | "false"
object  ::= "{" (string ":" value ("," string ":" value)*)? "}"
value   ::= string | number | integer | bool | object | array
array   ::= "[" (value ("," value)*)? "]"
"""


# ──────────────────────────────────────────────────────────────
# Intent values for grammar
# ──────────────────────────────────────────────────────────────

_INTENT_VALUES: Dict[ModelIntent, str] = {
    ModelIntent.CLASSIFY: '"CLASSIFY"',
    ModelIntent.GENERATE: '"GENERATE"',
    ModelIntent.REASON: '"REASON"',
    ModelIntent.EMBED: '"EMBED"',
    ModelIntent.RANK: '"RANK"',
    ModelIntent.SUMMARIZE: '"SUMMARIZE"',
    ModelIntent.SYNTHESIZE: '"SYNTHESIZE"',
    ModelIntent.VERIFY: '"VERIFY"',
    ModelIntent.EXTRACT: '"EXTRACT"',
    ModelIntent.TRANSFORM: '"TRANSFORM"',
    ModelIntent.PLAN: '"PLAN"',
    ModelIntent.EVALUATE: '"EVALUATE"',
    ModelIntent.CALL_MODEL: '"CALL_MODEL"',
    ModelIntent.CALL_TOOL: '"CALL_TOOL"',
    ModelIntent.READ_MEMORY: '"READ_MEMORY"',
    ModelIntent.WRITE_MEMORY: '"WRITE_MEMORY"',
    ModelIntent.DELETE_MEMORY: '"DELETE_MEMORY"',
    ModelIntent.INDEX_QUERY: '"INDEX_QUERY"',
    ModelIntent.INDEX_UPSERT: '"INDEX_UPSERT"',
    ModelIntent.ROUTE: '"ROUTE"',
    ModelIntent.FILTER: '"FILTER"',
    ModelIntent.RETURN_RESULT: '"RETURN_RESULT"',
    ModelIntent.STREAM_RESULT: '"STREAM_RESULT"',
    ModelIntent.CANCEL: '"CANCEL"',
}


@dataclass
class AICLGrammar:
    """Grammar specification for constrained model output.

    Generates BNF for llama.cpp and JSON Schema for validation.
    """

    bnf: str
    description: str = ""
    intents: List[ModelIntent] = field(default_factory=list)
    json_schema: Optional[Dict[str, Any]] = None

    @classmethod
    def for_intents(
        cls,
        intents: Sequence[ModelIntent],
        description: str = "",
    ) -> "AICLGrammar":
        """Generate a grammar that accepts any of the given intents."""
        intent_values = " | ".join(
            _INTENT_VALUES[i] for i in intents if i in _INTENT_VALUES
        )
        bnf = GRAMMAR_ACTION_TEMPLATE.replace("{intent_values}", intent_values)
        schema = _build_json_schema(intents)
        return cls(
            bnf=bnf,
            description=description or f"AICL actions: {', '.join(i.value for i in intents)}",
            intents=list(intents),
            json_schema=schema,
        )

    @classmethod
    def for_classify(
        cls,
        labels: Optional[List[str]] = None,
    ) -> "AICLGrammar":
        """Grammar for classification output only."""
        return cls(
            bnf=GRAMMAR_CLASSIFY_TEMPLATE,
            description=f"AICL classify action (labels: {labels})",
            intents=[ModelIntent.CLASSIFY],
            json_schema=_build_classify_schema(labels),
        )

    @classmethod
    def for_generate(cls) -> "AICLGrammar":
        """Grammar for text generation output only."""
        return cls(
            bnf=GRAMMAR_GENERATE_TEMPLATE,
            description="AICL generate action",
            intents=[ModelIntent.GENERATE],
        )

    @classmethod
    def for_tool(
        cls,
        tool_name: Optional[str] = None,
    ) -> "AICLGrammar":
        """Grammar for tool invocation output only."""
        return cls(
            bnf=GRAMMAR_TOOL_TEMPLATE,
            description=f"AICL tool call action (tool: {tool_name})",
            intents=[ModelIntent.CALL_TOOL],
            json_schema=_build_tool_schema(tool_name),
        )

    @classmethod
    def for_all(cls) -> "AICLGrammar":
        """Grammar that accepts any valid AICL intent."""
        return cls.for_intents(list(ModelIntent), "All AICL intents")


def _build_json_schema(intents: Sequence[ModelIntent]) -> Dict[str, Any]:
    """Build a JSON Schema for the given intents."""
    intent_enum = [i.value for i in intents if i in _INTENT_VALUES]
    return {
        "type": "object",
        "required": ["intent"],
        "properties": {
            "intent": {
                "type": "string",
                "enum": intent_enum,
            },
            "target": {"type": "string"},
            "inputs": {"type": "object"},
            "params": {"type": "object"},
            "result": {"type": "object"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "stream": {"type": "boolean"},
            "cancel": {"type": "boolean"},
            "correlation_id": {"type": "string"},
            "deadline_ms": {"type": "integer", "minimum": 0},
            "metadata": {"type": "object"},
        },
        "additionalProperties": False,
    }


def _build_classify_schema(labels: Optional[List[str]] = None) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "required": ["intent", "target", "inputs", "result"],
        "properties": {
            "intent": {"type": "string", "enum": ["CLASSIFY"]},
            "target": {"type": "string"},
            "inputs": {
                "type": "object",
                "required": ["text"],
                "properties": {"text": {"type": "string"}},
            },
            "result": {
                "type": "object",
                "required": ["label", "confidence"],
                "properties": {
                    "label": {"type": "string"},
                    "confidence": {"type": "number"},
                },
            },
        },
    }
    if labels:
        schema["properties"]["result"]["properties"]["label"]["enum"] = labels
    return schema


def _build_tool_schema(tool_name: Optional[str] = None) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "required": ["intent", "target", "params"],
        "properties": {
            "intent": {"type": "string", "enum": ["CALL_TOOL"]},
            "target": {"type": "string"},
            "params": {
                "type": "object",
                "required": ["tool_name"],
                "properties": {
                    "tool_name": {"type": "string"},
                    "parameters": {"type": "object"},
                },
            },
        },
    }
    if tool_name:
        schema["properties"]["params"]["properties"]["tool_name"]["enum"] = [tool_name]
    return schema
