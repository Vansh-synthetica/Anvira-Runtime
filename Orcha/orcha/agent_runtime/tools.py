"""
orcha.agent_runtime.tools
=========================
The flat, typed tool system of the AgentRuntime (smolagents-style
minimalism — no heavyweight schema framework):

- ``Tool`` — name, description, typed JSON-schema args, ``run(args) ->
  Observation``. A thin adapter over Orcha's existing ``ToolSpec`` +
  ``ToolExecutor`` (validation, permission policy, structured errors), so
  the AgentRuntime reuses the production filesystem/terminal machinery
  instead of reimplementing it.
- ``ToolRegistry`` — flat name → Tool registration. The AgentRuntime
  lists available tools and hands their JSON schemas to the model at call
  time; the Workspace executes calls against the same registry.
- ``build_default_tools`` — the shipped toolset: sandboxed shell exec,
  file read, file write, and an Orcha-native ``query_experts`` tool backed
  by ``ExpertSelector``.

Tools execute with zero side effects on the agent: ``run`` never touches
the EventLog, the bus, or the model.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from ..capabilities.base import (
    CAUTIOUS, DANGEROUS_LEVEL, READ, SAFE, WRITE,
    CapabilityContext, ToolExecutor, spec,
)
from ..capabilities.filesystem import build_tools as build_filesystem_tools
from ..capabilities.terminal import build_tools as build_terminal_tools
from ..capabilities.git import build_tools as build_git_tools
from ..capabilities.search import build_tools as build_search_tools
from ..capabilities.web import build_tools as build_web_tools
from ..capabilities.workspace import build_tools as build_workspace_tools
from ..capabilities.diagnostics import build_tools as build_diagnostics_tools
from ..capabilities.code_intelligence import build_tools as build_code_intelligence_tools
from ..capabilities.readstate import ReadStateCache
from ..core.packets import OrchaPacket, PacketKind
from ..nodes.tool import ToolSchema, ToolSpec
from .diagnostics import diag_log
from .events import ErrorObservation, Observation, ToolResultObservation

logger = logging.getLogger("orcha.agent_runtime.tools")


# ── The Tool ─────────────────────────────────────────────────────────────────

class Tool:
    """
    Flat, typed tool bound to the AgentRuntime contract.

    Attributes
    ----------
    name         Tool identifier (must be unique in a registry).
    description  Human-readable description shown to the model.
    parameters   JSON Schema for the tool's arguments.
    run()        Execute ``args`` and return an Observation — never raises;
                 every failure becomes an ErrorObservation so the agent
                 loop can keep going.
    """

    def __init__(self, spec: ToolSpec) -> None:
        self._spec = spec
        self._executor = ToolExecutor([spec])

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def description(self) -> str:
        return self._spec.description

    @property
    def parameters(self) -> ToolSchema:
        return self._spec.parameters or {
            "type": "object", "properties": {}, "required": [],
        }

    @property
    def permissions(self) -> List[str]:
        return list(self._spec.permissions)

    @property
    def safety_level(self) -> str:
        return self._spec.safety_level

    def is_read_only(self) -> bool:
        """True when the tool cannot mutate the world (parallel-friendly)."""
        return self._spec.is_read_only()

    def is_concurrency_safe(self) -> bool:
        """True when this tool may run alongside other safe tools."""
        return self._spec.is_concurrency_safe()

    @property
    def spec(self) -> ToolSpec:
        """The underlying Orcha ToolSpec (schema, kwargs_fn, metadata)."""
        return self._spec

    # ── Model-facing ───────────────────────────────────────────────────

    def schema(self) -> Dict[str, Any]:
        """OpenAI-compatible function schema for the model's tool list."""
        return self._spec.to_openai_schema()

    def to_prompt_line(self) -> str:
        return f"- {self.name}: {self.description}"

    # ── Execution ──────────────────────────────────────────────────────

    def validate(self, args: Dict[str, Any]) -> Optional[str]:
        """Return an error message if ``args`` violates the JSON schema
        (missing required keys), else None."""
        return self._spec.validate_kwargs(**dict(args or {}))

    def run(
        self, args: Dict[str, Any], tool_call_id: Optional[str] = None,
    ) -> Observation:
        """
        Execute the tool with validated keyword args.

        Returns a ToolResultObservation on success (content rendered from
        the structured result) or an ErrorObservation on validation
        failure / runtime failure. Never raises.
        """
        t0 = time.perf_counter()
        args_preview = json.dumps(args or {}, default=str, ensure_ascii=False)
        if len(args_preview) > 600:
            args_preview = args_preview[:600] + f"... ({len(args_preview)} chars)"
        try:
            result = self._executor.invoke(self.name, **(args or {}))
        except Exception as exc:
            diag_log(
                logger, "tool", "exception", level=logging.ERROR,
                name=self.name, exception=type(exc).__name__,
                message=str(exc),
            )
            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            return ErrorObservation(
                message=f"Tool {self.name} failed: {exc}",
                tool=self.name,
            )
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        if result.ok:
            value = result.value
            file_changes = []
            candidate = getattr(value, "file_changes", None)
            if isinstance(candidate, list):
                file_changes = [item for item in candidate if isinstance(item, dict)]
                content = result.to_message()
            elif isinstance(value, dict):
                candidate = value.get("file_changes")
                if isinstance(candidate, list):
                    file_changes = [item for item in candidate if isinstance(item, dict)]
                # Keep the model-facing result concise while exposing the
                # structured change record separately to execution viewers.
                content = str(value.get("message") or result.to_message())
            else:
                content = result.to_message()
            diag_log(
                logger, "tool", "result",
                name=self.name, ok=True, duration_ms=duration_ms,
                output_len=len(content),
                output_preview=content[:300],
            )
            return ToolResultObservation(
                tool_call_id=tool_call_id,
                content=content,
                success=True,
                file_changes=file_changes,
            )
        code = (result.error or {}).get("code", "tool_error")
        message = (result.error or {}).get("message", "tool failed")
        detail = (result.error or {}).get("detail")
        diag_log(
            logger, "tool", "result",
            level=logging.WARNING,
            name=self.name, ok=False, duration_ms=duration_ms,
            error_code=code, error_message=message,
            error_detail=detail,
        )
        return ErrorObservation(message=f"[{code}] {message}")

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r}, safety={self.safety_level!r})"


# ── The registry ─────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Flat registration of Tools. No plugin ceremony: ``register`` a Tool
    (or ``register_spec`` an existing Orcha ToolSpec) and the registry is
    ready. The same registry serves both the agent (schemas for the model)
    and the workspace (execution).
    """

    def __init__(self, tools: Optional[List[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> "ToolRegistry":
        """Register a Tool under its name. Re-registration replaces it."""
        if not tool or not tool.name:
            raise ValueError("cannot register a tool without a name")
        self._tools[tool.name] = tool
        return self

    def register_spec(self, spec: ToolSpec) -> Tool:
        """Wrap an existing Orcha ToolSpec as a Tool and register it.
        Returns the wrapped Tool."""
        tool = Tool(spec)
        self.register(tool)
        return tool

    def remove(self, name: str) -> bool:
        """Unregister a tool (e.g. a loop guard marking it unavailable).
        Returns True when a tool was actually removed."""
        return self._tools.pop(name, None) is not None

    # ── Reads ──────────────────────────────────────────────────────────

    def names(self) -> List[str]:
        return list(self._tools)

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def tools(self) -> List[Tool]:
        return list(self._tools.values())

    def schemas(self) -> List[Dict[str, Any]]:
        """OpenAI-compatible tool schemas for the model (call time)."""
        return [t.schema() for t in self._tools.values()]

    def describe(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "permissions": t.permissions,
                "safety_level": t.safety_level,
                "read_only": t.is_read_only(),
                "concurrency_safe": t.is_concurrency_safe(),
                "parameters": t.parameters,
            }
            for t in self._tools.values()
        ]

    def system_prompt_tools_section(self) -> str:
        """A concise tool listing for the agent's system prompt.

        Groups tools by category and provides a short workflow guide.
        Kept compact to save tokens — the detailed workflow examples
        live in the text-protocol prompt (server.py) where they matter
        most (small/local models).
        """
        if not self._tools:
            return ""

        lines = [
            "RULES: You have real tools. CALL them to do work — do NOT just describe what you would do.",
            "ALWAYS read_file BEFORE edit_file (edit needs exact text match).",
            "If a tool fails, read the error and retry with corrected args.",
            "",
        ]

        # Tool catalog grouped by category
        categories = [
            ("EXPLORE", ["directory_tree", "list_directory", "current_workspace",
                         "workspace_tree", "project_summary", "exists", "file_info"]),
            ("READ", ["read_file", "read_multiple_files"]),
            ("FIND", ["search_text", "grep", "regex_search", "glob_search",
                      "filename_search", "symbol_search", "workspace_search"]),
            ("WEB", ["web_search", "web_fetch"]),
            ("WRITE", ["create_file", "write_file", "edit_file", "append_file", "replace_text"]),
            ("MANAGE", ["delete_file", "rename_file", "copy_file", "move_file",
                        "create_directory", "delete_directory"]),
            ("COMMANDS", ["run_command", "stream_output", "stop_process", "running_processes"]),
            ("GIT", ["git_status", "git_diff", "git_add", "git_commit",
                     "git_log", "git_branch", "git_checkout", "git_restore", "git_push"]),
            ("BUILD", ["run_build", "run_tests", "run_linter", "type_check", "read_logs"]),
            ("CODE INTEL", ["find_symbol", "find_references", "rename_symbol",
                            "document_symbol", "outline_file"]),
        ]

        for cat_label, cat_tools in categories:
            available = [n for n in cat_tools if n in self._tools]
            if not available:
                continue
            lines.append(f"[{cat_label}] " + ", ".join(available))

        lines.append("")
        lines.append("WORKFLOW: explore -> read -> edit -> build/test -> git add/commit")

        return "\nAvailable tools:\n" + "\n".join(lines)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())


# ── Orcha-native tool: query the ExpertSelector ─────────────────────────────

def _build_query_experts_tool(selector: Any) -> ToolSpec:
    """Tool that asks Orcha's ExpertSelector which registered experts best
    fit a query. Degrades gracefully when no selector is configured."""

    def query_experts(query: str, limit: int = 3) -> Dict[str, Any]:
        limit = max(1, min(int(limit or 3), 10))
        if selector is None:
            return {
                "experts": [],
                "count": 0,
                "note": "no expert selector configured in this runtime",
            }
        packet = OrchaPacket(
            kind=PacketKind.QUERY,
            query=query,
            payload={"parallel_width": limit, "force_all_experts": False},
        )
        out = selector.select(packet)
        slots = out.get_selected_experts()
        return {
            "experts": [
                {
                    "name": s.name,
                    "domain": s.domain,
                    "score": round(s.score, 3),
                    "description": s.description,
                }
                for s in slots[:limit]
            ],
            "count": len(slots),
        }
    return spec(
        "query_experts",
        "Ask Orcha's expert selector which of the registered local models "
        "are best suited to answer a question. Returns ranked expert names, "
        "domains and confidence scores.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question or task to route."},
                "limit": {"type": "integer", "description": "Max experts to return (1-10). Default 3."},
            },
            "required": ["query"],
        },
        query_experts,
        permissions=[READ],
        safety_level=SAFE,
        capability="orcha",
        result_format={"type": "object"},
    )


# ── The default toolset ─────────────────────────────────────────────────────

def build_default_tools(
    roots: List[str],
    selector: Any = None,
    read_state: Optional["ReadStateCache"] = None,
) -> List[Tool]:
    """
    The shipped AgentRuntime toolset (sandboxed to ``roots``) — every tool
    defined across the capability modules, so the model's actual registry
    always matches what the system prompt promises it can do:

    - ``filesystem`` — read/write/edit/append/create/delete/rename/copy/
      move for files and directories, directory listing/tree, existence
      and metadata checks, glob/text search, multi-file read.
    - ``terminal`` — run a command, start/stop a background process,
      inspect running processes and command history.
    - ``git`` — status/diff/add/commit/branch/checkout/log/show/restore/
      pull/push.
    - ``search`` — grep, regex, filename, symbol, and workspace search.
    - ``web`` — web_search (DuckDuckGo, no API key) and web_fetch (download
      + extract readable text from a URL) — the only tools here that reach
      outside the local workspace. Both are NETWORK/CAUTIOUS, so
      PermissionPolicy gates them exactly like run_command/git_push:
      auto-allowed in action/full/auto access modes, held for approval in
      approval/partial modes.
    - ``workspace`` — current workspace, workspace tree, project list,
      attached files/folders, active file, recent files, project summary.
    - ``diagnostics`` — run build/tests/linter, type-check, dependency
      check, read logs.
    - ``code_intelligence`` — find/rename symbols, find references,
      document symbol, outline file.
    - ``query_experts`` — Orcha-native ExpertSelector routing, ONLY when a
      selector is actually configured. Without a selector the tool is not
      registered at all, so the model can never call (or loop on) a tool
      that cannot answer.

    Every tool here is already tagged SAFE/CAUTIOUS/DANGEROUS_LEVEL at the
    capability layer (see ``orcha.capabilities.base``) and runs through the
    same ``ToolExecutor`` as the rest of Orcha — registering more tools
    does not weaken that; it was the ONLY gate they already had. This
    runtime's ``Conversation``/``ToolWorkspace`` loop has no interactive
    approval step of its own (unlike the task_executor.py pipeline), so a
    DANGEROUS_LEVEL tool here (git_push, delete_file, run_command, ...)
    still executes the moment the model calls it — that's a product
    decision for whoever wires this runtime into a live endpoint, not
    something to silently work around by hiding tools from the model.

    ``selector`` is an optional ``ExpertSelector`` instance for the
    Orcha-native tool.

    ``read_state`` optionally attaches a
    :class:`~orcha.capabilities.readstate.ReadStateCache`, enabling
    stale-write protection: mutating tools then refuse to modify files the
    agent has not fully read (or that changed on disk since the read).
    Default None keeps legacy behavior.
    """
    ctx = CapabilityContext(roots=list(roots or []), read_state=read_state)
    by_name: Dict[str, ToolSpec] = {}
    for builder in (
        build_filesystem_tools,
        build_terminal_tools,
        build_git_tools,
        build_search_tools,
        build_web_tools,
        build_workspace_tools,
        build_diagnostics_tools,
        build_code_intelligence_tools,
    ):
        for tool_spec in builder(ctx):
            by_name[tool_spec.name] = tool_spec

    specs: List[ToolSpec] = list(by_name.values())
    if selector is not None:
        specs.append(_build_query_experts_tool(selector))

    return [Tool(s) for s in specs]


def _build_thinking_tool() -> ToolSpec:
    """Tool that lets the model think step-by-step before acting.

    Small models especially benefit from explicit chain-of-thought:
    they state what they've observed, what they plan to do, and why,
    then get the thought back so they can refine their reasoning
    before committing to a tool call.
    """

    def thinking(thought: str) -> str:
        return (
            "Thinking recorded. Now proceed with your next tool call or "
            "final answer based on the reasoning above."
        )

    return spec(
        "thinking",
        "Think step-by-step before acting. Use this to reason about what "
        "you've observed, plan your approach, weigh options, or decide "
        "which tool to call next. Your thought is recorded and you will "
        "get it back — then you MUST proceed with an actual tool call or "
        "final answer. Use this before complex tasks, when unsure which "
        "tool to use, or when you need to organize your approach.",
        {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": (
                        "Your step-by-step reasoning. Include: (1) What you know "
                        "from the conversation so far, (2) What the user wants, "
                        "(3) Which tool(s) you should call and why, (4) Any risks "
                        "or things to watch out for."
                    ),
                },
            },
            "required": ["thought"],
        },
        thinking,
        permissions=[],
        safety_level=SAFE,
        capability="agent_runtime",
    )


__all__ = [
    "Tool", "ToolRegistry", "build_default_tools",
]