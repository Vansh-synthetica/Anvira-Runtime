"""
orcha.agent_runtime.agent
=========================
The STATELESS Agent (OpenHands V1 model).

Hard rule of this system: the Agent carries no mutable state. Given an
immutable ``AgentConfig`` and the current EventLog slice, ``step()``
returns the next Action. Everything the agent needs to decide is either in
the config or in the events — never in the agent object. Every future
prompt in this series must preserve this contract.

Agents
------
- ``StubAgent``       — deterministic canned answer; used by skeleton
                        tests and as a drop-in when no backend is wired.
- ``ToolCallingAgent``— the real model-backed agent. Calls a ModelBackend
                        with the conversation plus the ToolRegistry's
                        schemas, then hands the model's choice back as
                        actions. Stateless: the backend is injected, and
                        all per-call state lives in the config, the log
                        slice and the returned actions.

Token accounting
----------------
Actions carry optional ``tokens`` / ``reasoning_tokens`` (default 0). The
Conversation copies them onto the Event when appending, so reasoning
tokens reported by the backend fold into the SAME budget accounting as
everything else — a backend that reports none simply contributes zero.
"""
from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from pydantic import BaseModel, ConfigDict

from .backend import (
    GenerationConfig, ModelBackend, ModelResponse, ToolCallParseError,
)
from .diagnostics import diag_log, diag_log_exc
from .events import (
    Action, Event, ErrorAction, EventKind, FinishAction, MessageAction,
    ToolCallAction,
)
from .memory import ContextMemory, render_memory_section
from .tools import ToolRegistry

logger = logging.getLogger("orcha.agent_runtime.agent")

# Context length used for the rendered history (overridable in AgentConfig).
DEFAULT_MAX_CONTEXT_CHARS = 32_000

_TOOL_DIRECTIVE = (
    "CRITICAL: You have real tools that execute actual commands on the system.\n"
    "When the user asks you to DO something (create, edit, read, search, run,\n"
    "delete, build, test, commit, etc.) you MUST call the appropriate tool.\n"
    "Do NOT describe what you would do — actually do it by calling a tool.\n"
    "\n"
    "To call a tool, respond with a JSON tool-call object:\n"
    "  {\"name\": \"<tool_name>\", \"arguments\": {\"<param>\": \"<value>\"}}\n"
    "\n"
    "Common tool calls:\n"
    "- Read a file: {\"name\": \"read_file\", \"arguments\": {\"path\": \"src/app.py\"}}\n"
    "- Edit a file: {\"name\": \"edit_file\", \"arguments\": {\"path\": \"src/app.py\", \"old_string\": \"...\", \"new_string\": \"...\"}}\n"
    "- Search code: {\"name\": \"search_text\", \"arguments\": {\"query\": \"functionName\"}}\n"
    "- List files: {\"name\": \"directory_tree\", \"arguments\": {\"path\": \".\", \"depth\": 3}}\n"
    "- Run a command: {\"name\": \"run_command\", \"arguments\": {\"command\": \"npm test\"}}\n"
    "- Git status: {\"name\": \"git_status\", \"arguments\": {}}\n"
    "\n"
    "NEVER invent tool names. NEVER claim a tool ran if it didn't.\n"
    "If a tool fails, read the error and try again with corrected arguments.\n"
    "For plain text answers (no tool needed), respond normally.\n"
)


class AgentConfig(BaseModel):
    """
    Immutable configuration for an Agent. Frozen by contract: no field can
    be mutated after construction.
    """
    model_config = ConfigDict(frozen=True)

    model: str = "stub-model"   # model identifier (display / metadata)
    system_prompt: str = (
        "You are an autonomous agent. Use the conversation history to "
        "decide your next action."
    )
    # Per-step generation options forwarded to the backend; a backend is
    # free to ignore them (see GenerationConfig).
    generation: Optional[GenerationConfig] = None
    # Guardrail: the model's free-text replies are truncated to this many
    # characters before they are attached to actions.
    max_content_chars: int = 8_000
    # Guardrail: the system prompt is truncated to fit.
    max_system_prompt_chars: int = 12_000
    # Guardrail: rendered history truncation (chars).
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
    # Working-memory guardrail: the windowed summary of recent steps
    # (step memory) may cost at most this many tokens of context — charged
    # as tokens + reasoning tokens per step (the SAME accounting as the
    # budget), so context stays bounded across long tool-calling sessions
    # whether or not the backend reports reasoning tokens.
    max_context_tokens: int = 8_000


class Agent(ABC):
    """
    Stateless agent contract.

    ``step()`` MUST be a pure function of (config, log_slice): the same
    inputs must always yield the same Action, and it must not read or
    write any mutable state (no self.foo, no files, no clocks, no network).
    This is what makes the whole runtime deterministic and replayable.
    """

    @abstractmethod
    async def step(self, config: AgentConfig, log_slice: Sequence[Event]) -> Action:
        """
        Decide the next Action from the current EventLog slice.

        Parameters
        ----------
        config      The immutable agent configuration.
        log_slice   The ordered events of the current turn (everything
                    after the last user message), read-only.

        Returns
        -------
        The next Action. A FinishAction terminates the conversation; a
        ToolCallAction is handed to the Workspace.
        """

    async def step_batch(
        self,
        config: AgentConfig,
        log_slice: Sequence[Event],
        memory: Optional[ContextMemory] = None,
    ) -> List[Action]:
        """
        Decide the next batch of Actions from the current EventLog slice.

        ``memory`` is an immutable working-memory snapshot (short-term step
        records + durable facts) the agent may fold into the context it
        hands the model. It is a projection of the EventLog, so the agent
        stays stateless: same inputs → same actions.

        Defaults to a single step; agents that emit multiple actions per
        model call (e.g. several tool calls at once) override this.

        The Conversation executes batch actions sequentially, appending an
        event per action — so multiple tool calls per turn cost exactly ONE
        model call.
        """
        return [await self.step(config, log_slice)]


class StubAgent(Agent):
    """
    Deterministic skeleton agent: always finishes immediately with a canned
    answer. No model calls, no streaming — kept for the skeleton tests and
    as the default when no backend is configured.
    """

    async def step(self, config: AgentConfig, log_slice: Sequence[Event]) -> Action:
        return FinishAction(
            content=f"Stub agent ({config.model}): no model calls yet.",
            reason="done",
        )


# ── The real agent ───────────────────────────────────────────────────────────

class ToolCallingAgent(Agent):
    """
    Model-backed, tool-using agent.

    Drives the loop statelessly:

    1. Builds the message list from the config and the log slice.
    2. Offers the registry's tool schemas to the backend.
    3. Interprets the response:
       - tool calls → one ``ToolCallAction`` per call (multiple tool calls
         per turn are returned together via ``step_batch``).
       - free text  → ``FinishAction`` when replying to a tool result,
         ``MessageAction`` on intermediate turns (keeps working).
    4. One repair round: if the backend raised ``ToolCallParseError``
       (malformed tool-call JSON), the agent re-prompts once with the parse
       error and re-interprets; a second failure fails the step cleanly
       with an ``ErrorAction``.
    5. Usage from the backend lands on the returned actions' ``tokens`` /
       ``reasoning_tokens`` (with a deterministic estimate for free-text
       content when the backend reports no usage).

    Streaming: tool rounds are never streamed unless the backend declares
    support (see ``GenerationConfig.should_stream``); the conservative
    default streams only the final free-text turn.
    """

    def __init__(
        self,
        backend: ModelBackend,
        tools: Optional[ToolRegistry] = None,
    ) -> None:
        self._backend = backend
        self._tools = tools or ToolRegistry()
        # Cache tool schemas and system prompt — they never change within a run.
        self._cached_schemas: Optional[List[Dict[str, Any]]] = None
        self._cached_system_prompt: Optional[str] = None
        self._cached_prompt_config_id: Optional[int] = None
        self._cached_prompt_tools_id: Optional[int] = None

    @property
    def backend(self) -> ModelBackend:
        return self._backend

    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    @property
    def model_name(self) -> str:
        return self._backend.model_name

    # ── Rendering helpers (pure) ──────────────────────────────────────

    @staticmethod
    def system_prompt(config: AgentConfig, tools: ToolRegistry, *, native_tools: bool = True) -> str:
        """
        Build the system prompt for the model.

        When ``native_tools=True`` (the default), tool schemas are sent via
        the ``tools`` parameter — the plain-text listing and directive are
        redundant and waste ~225 tokens per call.  When ``native_tools=False``
        (text-only mode), include the listing + directive as fallback.
        """
        prompt = config.system_prompt
        if len(prompt) > config.max_system_prompt_chars:
            prompt = prompt[: config.max_system_prompt_chars]
        if native_tools:
            # Tool schemas are sent via the `tools` parameter — no listing needed.
            return prompt
        section = tools.system_prompt_tools_section()
        if section:
            prompt = f"{prompt}\n{section}\n{_TOOL_DIRECTIVE}"
        return prompt

    def _cached_system_prompt_for(self, config: AgentConfig) -> str:
        """Cache system prompt — same config + tools = same prompt."""
        config_id = id(config)
        tools_id = id(self._tools)
        if (
            self._cached_system_prompt is not None
            and self._cached_prompt_config_id == config_id
            and self._cached_prompt_tools_id == tools_id
        ):
            return self._cached_system_prompt
        # native_tools=False: always include the plain-text tool listing
        # and JSON directive too, even though native `tools` schemas are
        # also sent. Belt-and-suspenders — costs a few hundred tokens but
        # means local/edge models that ignore the `tools` field (common
        # for smaller llama.cpp/Ollama models) still get told how to ask
        # for a tool in a shape the backend's text fallback parser can
        # recover (see openai_compat._extract_text_tool_call).
        self._cached_system_prompt = self.system_prompt(
            config, self._tools, native_tools=False,
        )
        self._cached_prompt_config_id = config_id
        self._cached_prompt_tools_id = tools_id
        return self._cached_system_prompt

    @staticmethod
    def render_history(
        log_slice: Sequence[Event], max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> str:
        """
        Render the turn's events into a deterministic transcript.

        Strategy (research-backed):
        - Render all events to compute per-event sizes.
        - Keep the **most recent** events first (they carry the most
          decision-relevant context) and drop the oldest when the
          character budget is exhausted.
        - If even the first event overflows, render it truncated with
          an ellipsis marker so the model still sees *something*.
        """
        rendered: List[tuple] = []  # (event, line_text, line_len)
        for ev in log_slice:
            line = _render_event_line(ev)
            if not line:
                continue
            rendered.append((ev, line, len(line)))

        if not rendered:
            return ""

        total = sum(r[2] for r in rendered)
        if total <= max_chars:
            return "\n".join(r[1] for r in rendered)

        # Budget exceeded — keep the most recent events, drop oldest.
        kept: List[str] = []
        used = 0
        for ev, line, length in reversed(rendered):
            if used + length > max_chars:
                break
            kept.append(line)
            used += length
        kept.reverse()

        if not kept:
            # Even the single most recent event exceeds the budget —
            # truncate it with an ellipsis so the model gets context.
            last_line = rendered[-1][1]
            kept = [last_line[:max_chars] + "… [truncated]"]

        return "[history truncated]\n" + "\n".join(kept)

    # ── The step ──────────────────────────────────────────────────────

    async def step(self, config: AgentConfig, log_slice: Sequence[Event]) -> Action:
        actions = await self.step_batch(config, log_slice)
        return actions[0]

    async def step_batch(
        self,
        config: AgentConfig,
        log_slice: Sequence[Event],
        memory: Optional[ContextMemory] = None,
    ) -> List[Action]:
        gen = config.generation or GenerationConfig(max_tokens=2048)
        # Apply anti-repetition penalties if not already set by caller.
        # Small models (≤3B params) are extremely prone to repetition
        # loops; these penalties are a first-line defence.
        if gen.frequency_penalty is None or gen.presence_penalty is None:
            from dataclasses import replace
            gen = replace(
                gen,
                frequency_penalty=gen.frequency_penalty if gen.frequency_penalty is not None else 0.3,
                presence_penalty=gen.presence_penalty if gen.presence_penalty is not None else 0.15,
            )
        # Cache tool schemas — they don't change within a run.
        if self._cached_schemas is None:
            self._cached_schemas = self._tools.schemas()
        schemas = self._cached_schemas

        memory_section = render_memory_section(
            memory, max_context_tokens=config.max_context_tokens,
        )
        transcript = self.render_history(log_slice, config.max_context_chars)
        if transcript:
            transcript = f"\n[Turn events]\n{transcript}"

        # ── KV-cache-aware prompt structure ────────────────────────────
        # Research shows prefix caching gives 85-95% cost savings when the
        # first N tokens are identical across requests.  We split the user
        # content into two messages so the MEMORY section (which changes
        # slowly) sits earlier in the token stream than the TRANSCRIPT
        # (which rebuilds every step).  This extends the cacheable prefix
        # by the memory section's token count on every step where the
        # memory hasn't changed yet.
        if memory_section:
            messages = [
                {"role": "system", "content": self._cached_system_prompt_for(config)},
                {"role": "user", "content": memory_section},
                {"role": "user", "content": f"{transcript}\n[Decide your next action.]".lstrip("\n")},
            ]
        else:
            messages = [
                {"role": "system", "content": self._cached_system_prompt_for(config)},
                {"role": "user", "content": f"{transcript}\n[Decide your next action.]".lstrip("\n")},
            ]

        response: Optional[ModelResponse] = None
        # Cache-fingerprint: hash of the system prompt (stable prefix).
        # Loggers can track this to verify prefix cache hit rates —
        # identical hashes across steps mean the KV cache is reusable.
        _prefix_hash = hashlib.sha256(
            messages[0]["content"].encode()
        ).hexdigest()[:12]
        for attempt in (0, 1):  # one repair round
            try:
                response = self._backend.complete(
                    messages, tools=schemas, config=gen,
                )
                break
            except ToolCallParseError as exc:
                diag_log(
                    logger, "agent", "repair",
                    attempt=attempt + 1, error=exc.message,
                )
                if attempt == 1:
                    return [ErrorAction(
                        message=(
                            "The model returned malformed tool calls twice; "
                            f"aborting this step. ({exc.message})"
                        )
                    )]
                messages.append({
                    "role": "user",
                    "content": self._backend.render_tool_calls_for_repair(exc),
                })
            except Exception as exc:  # transport/API failure → clean error
                diag_log_exc(logger, "agent", "model_call_failed", exc)
                return [ErrorAction(message=f"Model call failed: {exc}")]

        if response is None:  # pragma: no cover — the loop always breaks/returns
            return [ErrorAction(message="Model call failed unexpectedly.")]

        actions = self._interpret(
            response=response,
            is_final_turn=_is_final_turn(log_slice),
            max_content_chars=config.max_content_chars,
        )
        diag_log(
            logger, "agent", "decision",
            actions=",".join(a.kind for a in actions),
            tools=",".join(tc.name for tc in response.tool_calls) or None,
            text_len=len(response.content),
            usage=f"{response.usage.prompt_tokens}/{response.usage.completion_tokens}"
                  f"/{response.usage.reasoning_tokens}",
            finish_reason=response.finish_reason,
            prefix_hash=_prefix_hash,
        )
        return actions

    # ── Interpretation (pure) ─────────────────────────────────────────

    def _interpret(
        self,
        response: ModelResponse,
        is_final_turn: bool,
        max_content_chars: int,
    ) -> List[Action]:
        usage = response.usage
        n = len(response.tool_calls)

        if n:
            # Multiple tool calls per turn → one ToolCallAction each. The
            # whole step's usage is attributed to the first call so the
            # budget is never double-counted.
            actions: List[Action] = []
            for i, tc in enumerate(response.tool_calls):
                actions.append(ToolCallAction(
                    name=tc.name,
                    arguments=tc.arguments,
                    tool_call_id=tc.id or None,
                    tokens=usage.step_tokens if i == 0 else 0,
                    reasoning_tokens=usage.reasoning_tokens if i == 0 else 0,
                ))
            return actions

        content = response.content
        if len(content) > max_content_chars:
            content = content[:max_content_chars]

        # Deterministic estimate so text turns always count against the
        # budget even when the backend reports no usage.
        estimated = max(usage.step_tokens, _estimate_chars_tokens(content))

        if is_final_turn:
            return [FinishAction(
                content=content,
                reason="done",
                tokens=estimated,
                reasoning_tokens=usage.reasoning_tokens,
            )]
        return [MessageAction(
            content=content,
            tokens=estimated,
            reasoning_tokens=usage.reasoning_tokens,
        )]


def _render_event_line(ev: Event) -> str:
    p = ev.payload
    kind = p.kind
    if kind == "user_message":
        return f"user: {p.content}"
    if kind == "tool_call":
        return f"tool call: {p.name}({p.arguments})"
    if kind == "tool_result":
        # Truncate very large tool results to save context tokens.
        # The model doesn't need 10K chars of file contents — a concise
        # summary with the first/last lines is more useful.
        content = p.content or ""
        _TOOL_RESULT_MAX = 800
        if len(content) > _TOOL_RESULT_MAX:
            head = content[:_TOOL_RESULT_MAX // 2]
            tail = content[-(_TOOL_RESULT_MAX // 2):]
            content = f"{head}\n... [{len(p.content)} chars total, truncated] ...\n{tail}"
        return f"tool result: {content}"
    if kind == "error":
        return f"error: {p.message}"
    if kind == "message":
        return f"assistant: {p.content}"
    if kind == "finish":
        return f"assistant (done): {p.content}"
    if kind == "fact":
        # Durable facts are injected via the working-memory section
        # ([Durable facts]), not the raw turn transcript.
        return ""
    return f"[{kind}]"


def _is_final_turn(log_slice: Sequence[Event]) -> bool:
    """
    A turn is final when the model is replying to a tool outcome (a tool
    result OR a tool error): free text then means the agent is done, so it
    becomes a FinishAction. A turn whose last event is the user message
    keeps working — that is the "let me check that" intermediate-reply
    semantic.

    Exception: if the model already produced assistant text earlier in this
    round (the last event is an assistant message), a SECOND text-only reply
    is the answer — otherwise a backend that replies in prose without ever
    calling a tool loops to the step cap ("canned reply" dead loop).
    """
    if not log_slice:
        return False
    last = log_slice[-1]
    if last.kind != EventKind.OBSERVATION:
        # Round already contains an assistant text turn: the next text is
        # the answer, never another "intermediate" reply.
        return last.payload.kind == "message"
    return last.payload.kind != "user_message"


def _estimate_chars_tokens(text: str) -> int:
    """Rough deterministic estimate (~1 token per 4 chars)."""
    return max(0, len(text) // 4)


__all__ = [
    "AgentConfig", "Agent", "StubAgent", "ToolCallingAgent",
    "DEFAULT_MAX_CONTEXT_CHARS",
]
