"""
orcha.nodes.tool
================
Tool nodes — invoke external tools, scripts, or APIs as graph nodes.

A ToolNode wraps a single callable as a graph node. The callable receives
the packet's relevant data (configurable via input extraction) and returns
a string or structured result that gets written back into the packet.

Tools can be:
  - Simple functions (e.g., calculator, web search wrapper)
  - Shell commands (with safety controls)
  - API calls (HTTP, database queries)
  - Any Python callable respecting the contract

The ToolSpec is a lightweight descriptor that the AgentNode uses to
discover and invoke tools. It can also be used standalone as a ToolNode.

Contract
--------
Input packet payload:
  - ``tool_input`` (Any, optional): direct input to the tool.
  - Or the tool reads from any packet field via ``input_key``.

Output packet payload:
  - ``tool_output`` (str): the tool's result.
  - ``tool_name`` (str): which tool ran.
  - ``tool_duration_ms`` (float): execution time.
  - ``tool_success`` (bool): True if no exception.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from ..core.packets import OrchaPacket, PacketKind
from ..graph.context import RunContext
from ..graph.node import Node


# ── Tool specification ────────────────────────────────────────────────────────

ToolFunction = Callable[[Any], Any]

# Alias for the OpenAI-compatible JSON schema of a tool's arguments.
ToolSchema = Dict[str, Any]


@dataclass
class ToolSpec:
    """
    Describes a tool that can be invoked by an AgentNode or used as a ToolNode.

    Attributes
    ----------
    name        Tool identifier (used for routing and logging).
    description Human-readable description (shown to the agent in prompts).
    fn          The tool function. Takes a single argument (the input) and
                returns a result (string, dict, or any serialisable value).
    input_key   Packet payload key to read input from. If None, reads from
                ``payload["tool_input"]`` or falls back to the query.
    output_key  Packet payload key to write the result to. Default "tool_output".
    category    Optional category tag (e.g., "search", "math", "io").
    parameters  OpenAI-compatible JSON Schema for the tool's keyword arguments
                (e.g. {"type": "object", "properties": {...}}). When set, the
                tool supports native function-calling (kwargs-based invocation).
    kwargs_fn   Optional callable invoked with keyword arguments when a model
                calls this tool via function calling. Prefer over ``fn`` for
                structured tools.
    """
    name: str
    description: str
    fn: Optional[ToolFunction] = None
    input_key: Optional[str] = None
    output_key: str = "tool_output"
    category: str = "general"
    parameters: Optional[ToolSchema] = None
    kwargs_fn: Optional[Callable[..., Any]] = None
    # ── Capability-system metadata ────────────────────────────────────────
    permissions: List[str] = field(
        default_factory=lambda: ["read"],
        metadata={"help": "Subset of read/write/execute/network/dangerous."},
    )
    safety_level: str = field(
        default="safe",
        metadata={"help": "safe | cautious | dangerous"},
    )
    validate_fn: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None
    result_format: Optional[Dict[str, Any]] = None
    capability: Optional[str] = None
    requires_approval: Optional[bool] = None
    # ── Contract metadata (hardened tool contract) ─────────────────────────
    # Explicit overrides; when None both derive from ``permissions``:
    # read-only ⇔ no write/execute/network/dangerous permission, and a
    # read-only tool is concurrency-safe by default. Executors may run
    # concurrency-safe tools in parallel; anything else runs strictly alone.
    read_only: Optional[bool] = None
    concurrency_safe: Optional[bool] = None
    # Free-form provenance/metadata surfaced to hosts (e.g. a prompt skill's
    # allowed-tools rule strings, argument hint). Never interpreted here.
    meta: Optional[Dict[str, Any]] = None

    def __call__(self, input_data: Any) -> Any:
        """Invoke the tool directly (positional input form)."""
        if self.fn is None:
            raise RuntimeError(f"Tool {self.name!r} has no function attached")
        return self.fn(input_data)

    def invoke_kwargs(self, **kwargs: Any) -> Any:
        """
        Invoke the tool with keyword arguments (native function-calling form).
        Falls back to ``fn(kwargs)`` / ``fn()`` when no ``kwargs_fn`` is set.
        """
        if self.kwargs_fn is not None:
            return self.kwargs_fn(**kwargs)
        if self.fn is None:
            raise RuntimeError(f"Tool {self.name!r} has no function attached")
        if kwargs:
            return self.fn(kwargs)
        return self.fn()

    def to_openai_schema(self) -> ToolSchema:
        """Return the OpenAI-compatible tool schema for this spec."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or "",
                "parameters": self.parameters
                or {"type": "object", "properties": {}, "required": []},
            },
        }

    def validate_kwargs(self, **kwargs: Any) -> Optional[str]:
        """
        Validate keyword arguments. Returns an error message, or None if the
        call is acceptable. Checks required parameters from the JSON schema
        and, when present, the tool's explicit ``validate_fn``.
        """
        if self.validate_fn is not None:
            try:
                return self.validate_fn(dict(kwargs))
            except Exception as exc:
                return f"Validation failed: {exc}"
        if self.parameters:
            required = self.parameters.get("required") or []
            missing = [k for k in required if kwargs.get(k) in (None, "")]
            if missing:
                return self._argument_error(missing, bool(kwargs))
        return None

    def _argument_error(self, missing: List[str], had_any_args: bool) -> str:
        """A validation failure the model can actually act on.

        The bare "Missing required argument: path" this replaced told a
        model *that* it was wrong without ever showing it what a correct
        call looks like. Confirmed live on a 3B local model: it emitted
        write_file and run_command with completely empty arguments, got
        that message back, and simply repeated the same empty call until
        the loop guard disabled the tool — two of three files never got
        written. Small models fail most often on the envelope's
        double-encoded `arguments` string, so the recovery path has to
        restate the exact shape rather than just name the offending key.
        """
        props = (self.parameters or {}).get("properties") or {}
        required = (self.parameters or {}).get("required") or []
        example_parts = []
        for key in required:
            spec = props.get(key) or {}
            kind = spec.get("type", "string")
            placeholder = {
                "integer": "123",
                "number": "123",
                "boolean": "true",
                "array": "[]",
                "object": "{}",
            }.get(kind, f'"<{key}>"')
            example_parts.append(f'"{key}": {placeholder}')
        example = "{" + ", ".join(example_parts) + "}"

        lead = (
            f"You called {self.name} with no arguments at all."
            if not had_any_args
            else f"Missing required argument(s): {', '.join(missing)}."
        )
        detail = ""
        for key in missing:
            desc = (props.get(key) or {}).get("description")
            if desc:
                detail += f"\n  {key}: {desc}"
        return (
            f"{lead} {self.name} requires: {', '.join(required) or 'no arguments'}."
            f"{detail}\nRe-issue the call with arguments shaped exactly like: {example}"
        )

    def needs_approval(self) -> bool:
        """
        True when this tool should require approval in approval/partial access
        mode: explicitly flagged, or flagged via dangerous/execute/network
        permissions.
        """
        if self.requires_approval is not None:
            return self.requires_approval
        return bool({"dangerous", "execute", "network"} & set(self.permissions))

    _UNSAFE_FOR_READONLY = frozenset({"write", "execute", "network", "dangerous"})

    def is_read_only(self) -> bool:
        """
        True when the tool cannot mutate the world. Explicit ``read_only``
        wins; otherwise derived from permissions (any write/execute/network/
        dangerous permission ⇒ not read-only).
        """
        if self.read_only is not None:
            return self.read_only
        return not (self._UNSAFE_FOR_READONLY & set(self.permissions))

    def is_concurrency_safe(self) -> bool:
        """
        True when this call may run in parallel with other concurrency-safe
        calls. Explicit ``concurrency_safe`` wins; otherwise read-only tools
        are safe and everything else serializes.
        """
        if self.concurrency_safe is not None:
            return self.concurrency_safe
        return self.is_read_only()

    def to_prompt_description(self) -> str:
        """Human-readable description for agent prompts."""
        return f"TOOL:{self.name} — {self.description}"


# ── ToolNode ──────────────────────────────────────────────────────────────────

class ToolNode(Node):
    """
    A graph node that invokes a single tool.

    The tool reads its input from the packet (configurable), executes,
    and writes the result back. If the tool raises, the node catches the
    exception, writes it to the output, and marks ``tool_success=False`` —
    so tool failures are fault-isolated (the graph continues).

    Parameters
    ----------
    spec       A ToolSpec describing the tool.
    name       Node name (default: spec.name).
    timeout_s  Per-node timeout (None = no timeout).
    retries    Retry budget for transient failures.
    """

    def __init__(
        self,
        spec: ToolSpec,
        name: Optional[str] = None,
        timeout_s: Optional[float] = None,
        retries: int = 0,
    ) -> None:
        self.name = name or spec.name
        self.timeout_s = timeout_s
        self.retries = retries
        self._spec = spec

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        t0 = time.perf_counter()

        # Extract input.
        input_key = self._spec.input_key or "tool_input"
        tool_input = packet.payload.get(input_key)
        if tool_input is None:
            tool_input = packet.query

        # Execute the tool.
        success = True
        error_msg: Optional[str] = None
        try:
            result = self._spec(tool_input)
            if not isinstance(result, str):
                result = str(result)
        except Exception as exc:
            success = False
            error_msg = str(exc)
            result = f"[Tool error: {error_msg}]"
            ctx.logger.warning(
                "tool_error tool=%s error=%s", self._spec.name, exc,
            )

        duration_ms = (time.perf_counter() - t0) * 1000

        return packet.fork(
            packet.kind,
            **{self._spec.output_key: result},
            tool_name=self._spec.name,
            tool_duration_ms=round(duration_ms, 2),
            tool_success=success,
            tool_error=error_msg,
        )


# ── Pre-built tool specs ─────────────────────────────────────────────────────

def tool_spec_from_fn(
    fn: ToolFunction,
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: str = "general",
    parameters: Optional[ToolSchema] = None,
    kwargs_fn: Optional[Callable[..., Any]] = None,
) -> ToolSpec:
    """
    Convenience: build a ToolSpec from a plain function.

    Name and description are inferred from the function if not provided.
    """
    return ToolSpec(
        name=name or getattr(fn, "__name__", "unnamed_tool"),
        description=description or getattr(fn, "__doc__", "") or f"Tool: {fn.__name__}",
        fn=fn,
        category=category,
        parameters=parameters,
        kwargs_fn=kwargs_fn,
    )


def tool_node_from_fn(
    fn: ToolFunction,
    name: Optional[str] = None,
    description: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> ToolNode:
    """
    Convenience: build a ToolNode directly from a function.
    """
    spec = tool_spec_from_fn(fn, name=name, description=description)
    return ToolNode(spec, timeout_s=timeout_s)


__all__ = ["ToolNode", "ToolSpec", "ToolFunction", "ToolSchema", "tool_spec_from_fn", "tool_node_from_fn"]
