"""
orcha.integrations.langchain.tools
==================================
Conversion between Orcha tools and LangChain tools.

Two directions, with a deliberate asymmetry:

- ``orcha_tool_to_langchain`` — expose an Orcha Tool to a LangChain model
  as a StructuredTool. Execution STAYS in Orcha: the wrapper calls the
  Orcha ``Tool.run`` path, so validation, permission policy, approval
  flow, and the capability system all keep working. LangChain only sees
  the schema (model-facing), never executes anything itself.

- ``langchain_tool_to_orcha`` — import an ecosystem tool INTO Orcha as a
  ToolSpec, registered via the existing ToolRegistry. This is how MCP /
  LangChain community tools become first-class Orcha tools, still gated
  by Orcha permissions.

Orcha remains the permission boundary in both directions.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ...agent_runtime.tools import Tool
from ...nodes.tool import ToolSpec
from ..base import IntegrationUnavailable

__all__ = ["orcha_tool_to_langchain", "langchain_tool_to_orcha"]


def orcha_tool_to_langchain(tool: Tool):
    """
    Wrap an Orcha Tool as a LangChain StructuredTool.

    The returned tool is schema-compatible with LangChain chat models
    (``bind_tools``) but every invocation routes back through the Orcha
    Tool's own validation + permission + execution path. The result is
    the tool's Observation content rendered as a string.
    """
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        raise IntegrationUnavailable(
            "LangChain is not installed. Install with "
            "`pip install \"orcha[lang]\"`."
        ) from None

    schema: Dict[str, Any] = tool.parameters or {
        "type": "object", "properties": {}, "required": [],
    }

    def _run(**kwargs: Any) -> str:
        observation = tool.run(kwargs)
        content = getattr(observation, "content", None)
        if content is None:
            content = str(getattr(observation, "message", observation))
        return content

    return StructuredTool.from_function(
        func=_run,
        name=tool.name,
        description=tool.description,
        args_schema=schema,
        response_format="content",
    )


def langchain_tool_to_orcha(lang_tool, permissions: Optional[list] = None) -> ToolSpec:
    """
    Import a LangChain/ecosystem tool as an Orcha ToolSpec.

    The schema comes from the tool's own args schema; invocation goes
    through the tool's synchronous ``invoke`` path. Register the result
    with ``ToolRegistry.register_spec``.

    Parameters
    ----------
    lang_tool    Any LangChain BaseTool (StructuredTool, tool-decorated
                 function, MCP-adapted tool, …).
    permissions  Orcha permission vocabulary for the tool. Defaults to
                 ``["read"]`` — ecosystem tools start restricted and are
                 explicitly widened per workspace policy.
    """
    try:
        from langchain_core.tools import BaseTool
    except ImportError:
        raise IntegrationUnavailable(
            "LangChain is not installed. Install with "
            "`pip install \"orcha[lang]\"`."
        ) from None
    if not isinstance(lang_tool, BaseTool):
        raise TypeError(
            f"expected a langchain_core BaseTool, got {type(lang_tool).__name__}"
        )

    # ── Schema ─────────────────────────────────────────────────────────
    parameters: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    args_schema = getattr(lang_tool, "args_schema", None)
    if isinstance(args_schema, dict):
        # langchain v1: args_schema IS the JSON schema.
        parameters = args_schema
    elif args_schema is not None and hasattr(args_schema, "model_json_schema"):
        # langchain 0.3.x: pydantic model.
        parameters = args_schema.model_json_schema()
    else:
        raw_args = getattr(lang_tool, "args", None)
        if isinstance(raw_args, dict):
            parameters = {"type": "object", "properties": raw_args, "required": []}

    # ── Invocation ─────────────────────────────────────────────────────
    def _invoke(**kwargs: Any) -> Any:
        try:
            result = lang_tool.invoke(kwargs)
        except Exception as exc:
            return {"error": type(exc).__name__, "message": str(exc)}
        if isinstance(result, str):
            return result
        if hasattr(result, "content") and result.content is not None:
            return result.content
        return result

    return ToolSpec(
        name=lang_tool.name,
        description=getattr(lang_tool, "description", "") or "",
        kwargs_fn=_invoke,
        parameters=parameters,
        permissions=list(permissions or ["read"]),
        safety_level="cautious",
        capability="langchain",
    )
