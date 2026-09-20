"""
orcha.nodes.agent
=================
Agent nodes — autonomous reasoning units that can invoke tools, maintain
a scratchpad, and iterate on subtasks.

An AgentNode wraps a model-backed agent loop: given a task description
and optional context, it repeatedly calls a model with a system prompt,
inspects the output for tool calls, executes those tools, and feeds
the results back until the agent signals completion.

The agent is a graph node, so it participates in checkpointing, timeout,
retry, cancellation, and event streaming like any other node.

Contract
--------
Input packet payload:
  - ``task`` (str): the task for the agent to perform.
  - ``agent_context`` (str, optional): additional context from upstream nodes.
  - ``agent_tools`` (list[ToolSpec], optional): tools available to the agent.

Output packet payload:
  - ``agent_output`` (str): the agent's final answer.
  - ``agent_steps`` (list[dict]): step-by-step trace of the agent loop.
  - ``agent_tool_calls`` (list[dict]): tool invocations made.
  - ``agent_iterations`` (int): number of model calls.
  - ``agent_completed`` (bool): True if the agent finished (vs timed out).
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.packets import OrchaPacket, PacketKind
from ..graph.context import RunContext, EVT_TEXT_DELTA
from ..graph.node import Node

# Consecutive model-call failures before the agent loop gives up and the
# node fails (surfacing the error instead of answering with an error string).
_MAX_CONSECUTIVE_MODEL_ERRORS = 2

# Verification-gate tool classes: writing code without ever executing it is
# the #1 failure mode of small local models (they answer with INVENTED
# output). When the loop sees code written but no command run, it injects
# one corrective turn demanding real execution.
from ..agent_runtime import completion_checks, scaffold  # noqa: E402  (deterministic finished-work checks; per-file project builds)

_WRITE_CODE_TOOLS = frozenset({"write_file", "create_file", "append_file", "replace_text", "apply_patch", "edit_file"})
_EXECUTE_TOOLS = frozenset({"run_command", "stream_output"})
# Tools that only ever look at files already in the local workspace — see
# the empty-result hint below, where a zero-match result from one of these
# gets an actionable nudge toward web_search when that tool is registered.
_LOCAL_SEARCH_TOOLS = frozenset({"search_text", "grep", "regex_search", "glob_search", "symbol_search", "workspace_search"})


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """
    Configuration for an AgentNode.

    Attributes
    ----------
    system_prompt       The agent's system prompt template. Use {task} and
                        {context} as placeholders.
    max_iterations      Maximum model calls before the agent is forced to stop.
    model_fn            The model call function: async (prompt, system) -> str.
                        This is how the agent talks to any model (local or remote).
    completion_fn       Optional async (messages, system, tools) -> dict that
                        returns the full assistant message (content + tool_calls).
                        When set, the agent uses NATIVE function-calling
                        instead of the text "TOOL:name:arg" protocol.
    stop_phrases         Phrases that signal the agent is done (e.g., "FINAL ANSWER:").
    scratchpad_max_len   Maximum scratchpad length (chars). Older entries are
                        trimmed to stay within budget.
    tools                List of ToolSpec objects the agent can invoke.
    executor             Optional ToolExecutor. When set, every function call is
                        validated, policy-checked and executed through it, and
                        approval results are relayed as tool messages.
    capabilities         Declared capability names (informational + surfaced to
                        the builder so it can assemble the executor).
    reasoning_level      Reasoning level (fast/light/medium/high/max) used by the
                        orchestrator to configure planner/retrieval/verification.
    """
    system_prompt: str = (
        "You are an autonomous coding agent. Complete the following task "
        "using the tools available to you.\n\n"
        "If the task is a plain filesystem operation and not code (creating "
        "a folder, copying/moving/renaming files, listing a directory), use "
        "the dedicated filesystem tool for it directly (e.g. "
        "create_directory, copy_file, move_file, list_directory) instead of "
        "write_file — write_file is for creating or editing a file's actual "
        "text content, never for creating a directory or as a placeholder "
        "with empty content.\n\n"
        "If the task needs information from the internet, a specific "
        "website, current events, or anything that is not already part of "
        "this project's own files — even if the word \"search\" or \"find\" "
        "appears in the task — you MUST use web_search (then web_fetch on a "
        "promising result) when those tools are available. search_text, "
        "grep, and glob_search ONLY look at files already in this "
        "workspace; they cannot see the internet at all and will return "
        "misleading empty results for a request that actually needed the "
        "web. When in doubt about whether something is \"in this project\" "
        "or \"on the internet\", prefer web_search — a wasted web_search "
        "costs one extra tool call; a wrongly-local search can silently "
        "answer from nothing.\n\n"
        "Never answer a specific, checkable fact (an exact name, version "
        "number, current status, or anything that could be wrong from "
        "memory) from recollection alone when a tool that could verify it "
        "is available — call the tool first, then answer from what it "
        "actually returned. A guess that sounds right but is not "
        "verified is worse than the extra tool call it would have cost, "
        "the same way answering about a file you never actually read "
        "would be. This includes web_search results specifically: a "
        "search result's title/snippet is NOT the page's actual content — "
        "web_search alone tells you a page exists and roughly what it's "
        "about, nothing more. If the task needs the page's real content "
        "(what a README says, what a doc explains), call web_fetch on it "
        "before answering. Never say something was \"found in\" or \"says\" "
        "on a specific page unless you actually called web_fetch on that "
        "exact page and it's in what came back.\n\n"
        "Workflow for code tasks (follow this exact order):\n"
        "0. NEVER run a command for a file that does not exist yet. Create "
        "every file the task requires FIRST, and only then execute.\n"
        "1. To create or change code, call write_file with the COMPLETE "
        "file content (all branches, no placeholders). This is the ONLY "
        "way to create files.\n"
        "2. Then execute it by calling run_command (e.g. 'python "
        "filename.py'). NEVER invent command output — report only what "
        "the tool actually returned in stdout/stderr.\n"
        "3. If it failed or output is wrong, call write_file again with a "
        "fix, then run_command again. Repeat until correct.\n"
        "4. Before answering about code you wrote, check yourself: if you "
        "have NOT called run_command since your last write_file, do it now "
        "instead of answering. An answer that predicts output without "
        "running the code is wrong even when the prediction looks right.\n"
        "5. Only after verifying real output, reply with 'FINAL ANSWER:' "
        "followed by what you did and the actual output observed.\n\n"
        "Brevity rules: keep prose minimal between tool calls. When "
        "calling write_file, emit ONLY the tool call with the COMPLETE "
        "file content — never abbreviate, truncate, or add comments like "
        "'rest unchanged'. Never paste file contents into your prose.\n\n"
        "Example of correct tool use:\n"
        "user: Create app.py that prints hello, then run it.\n"
        "assistant: [calls write_file(path='app.py', content='print(hello)')]\n"
        "assistant: [calls run_command(command='python app.py')]\n"
        "[tool result: exit_code 0, stdout 'hello']\n"
        "assistant: FINAL ANSWER: Created app.py and ran it. Real output: hello\n\n"
        "Task: {task}\nContext: {context}"
    )
    max_iterations: int = 5
    model_fn: Optional[Callable] = None
    completion_fn: Optional[Callable] = None
    stop_phrases: List[str] = field(
        default_factory=lambda: ["FINAL ANSWER:", "FINAL:", "ANSWER:"]
    )
    scratchpad_max_len: int = 4000
    seed_messages: List[Dict[str, Any]] = field(
        default_factory=list,
        metadata={
            "help": "Prior conversation (role/content dicts) to seed the agent's "
                    "message list so it remembers earlier turns in the session."
        },
    )
    tools: List[object] = field(default_factory=list)  # List[ToolSpec]
    executor: Optional[Any] = None  # ToolExecutor
    capabilities: List[str] = field(default_factory=list)
    workspace_roots: List[str] = field(default_factory=list)  # lets the completion gates look at the project itself
    reasoning_level: Optional[str] = None
    approval_broker: Optional[Any] = None  # ApprovalBroker for interactive tool approval
    approval_timeout_s: float = 300.0  # how long to wait for a user decision


# ── AgentNode ─────────────────────────────────────────────────────────────────

class AgentNode(Node):
    """
    A graph node that runs an autonomous agent loop.

    The agent:
      1. Builds a system prompt from the task and context.
      2. Calls the model with the prompt + scratchpad.
      3. Parses the model output for tool calls.
      4. Executes matched tool calls and appends results to scratchpad.
      5. Repeats until a stop phrase is found or max_iterations is hit.

    If no ``model_fn`` is provided, the node operates in "dry run" mode:
    it produces a synthetic response for testing without requiring a model.

    Parameters
    ----------
    name       Node name (default "agent").
    config     Agent configuration.
    timeout_s  Per-node timeout (the entire agent loop must complete within this).
    retries    Retry budget.
    """

    def __init__(
        self,
        name: str = "agent",
        config: Optional[AgentConfig] = None,
        timeout_s: Optional[float] = 120.0,
        retries: int = 0,
    ) -> None:
        self.name = name
        self.timeout_s = timeout_s
        self.retries = retries
        self.config = config or AgentConfig()

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        task: str = packet.payload.get("task", packet.query)
        context: str = packet.payload.get("agent_context", "")
        # Literal placeholder substitution — str.format() here would treat
        # ANY brace in the prompt as a replacement field. The prompt now
        # routinely embeds attached project files (JSON, dicts, f-strings),
        # so {"name": x} inside an attachment crashed every run with
        # KeyError('"name"').
        system_prompt = (
            self.config.system_prompt
            .replace("{task}", task)
            .replace("{context}", context)
        )

        scaffold_plan = self._scaffold_plan(task)
        if scaffold_plan is not None:
            steps, tool_calls, final_output, completed = await self._run_scaffold(scaffold_plan, task, ctx)
        elif self.config.completion_fn is not None:
            steps, tool_calls, final_output, completed = await self._run_native_loop(
                system_prompt, task, ctx, packet
            )
        else:
            steps, tool_calls, final_output, completed = await self._run_text_loop(
                system_prompt, task, ctx, packet
            )

        if not final_output:
            if completed:
                final_output = (
                    f"The run finished after {len(steps)} iteration(s) and "
                    f"{len(tool_calls)} tool call(s) without producing a "
                    "final answer."
                )
            else:
                final_output = (
                    f"I wasn't able to complete that within "
                    f"{self.config.max_iterations} iterations — try "
                    "rephrasing the request."
                )

        return packet.fork(
            packet.kind,
            agent_output=final_output,
            agent_steps=steps,
            agent_tool_calls=tool_calls,
            agent_iterations=len(steps),
            agent_completed=completed,
        )

    def _scaffold_plan(self, task: str):
        """An explicit multi-file build request that this agent can carry out file by file (see agent_runtime/scaffold.py)."""
        ex = self.config.executor
        if self.config.completion_fn is None or ex is None or not self.config.workspace_roots:
            return None
        try:
            if "write_file" not in {t.name for t in ex.tools()}:
                return None
        except Exception:
            return None
        return scaffold.plan_files(task)

    async def _run_scaffold(self, plan, task: str, ctx: RunContext):
        root = str(self.config.workspace_roots[0])
        executor = self.config.executor

        async def write(path: str, content: str):
            return await self._invoke_tool(executor, "write_file", {"path": path, "content": content}, ctx.run_id, f"scaffold_{path}")
        res = await scaffold.run_scaffold(task=task, plan=plan, completion_fn=self.config.completion_fn, write=write, root=root,
                                          cancelled=lambda: bool(ctx.cancelled))
        return res.steps, res.tool_calls, res.answer, res.completed

    async def _run_native_loop(
        self,
        system_prompt: str,
        task: str,
        ctx: RunContext,
        packet: OrchaPacket,
    ):
        """
        Native function-calling agent loop. Each model completion may return
        ``tool_calls``; each is executed via the configured ToolExecutor (or the
        legacy ToolSpec map) and the results are fed back as ``role=tool``
        messages until the model answers with plain text or the iteration
        budget runs out. Approval-mode results are relayed verbatim so the
        agent surfaces the approval request to the user.
        """

        from ..nodes.tool import ToolSpec
        from ..capabilities.base import ToolExecutor as _ToolExecutor

        messages: List[Dict[str, Any]] = list(self.config.seed_messages or [])
        # The current query MUST be the final user turn of the conversation.
        # With seeded history the list otherwise ends on an assistant turn,
        # and chat templates then CONTINUE that turn (verbatim parroting)
        # instead of answering the new task — the query previously lived
        # only inside the system prompt, which renders BEFORE all messages.
        if not (
            messages
            and messages[-1].get("role") == "user"
            and messages[-1].get("content") == task
        ):
            messages.append({"role": "user", "content": task})
        tool_schemas: List[Dict[str, Any]] = []
        by_name: Dict[str, ToolSpec] = {}
        executor = self.config.executor if isinstance(self.config.executor, _ToolExecutor) else None
        if executor is not None:
            tool_schemas = executor.schemas()
            by_name = {t.name: t for t in executor.tools()}
            # Bind the policy to this run so approval grants are consumed
            # strictly for the run that requested them.
            policy = getattr(executor, "policy", None)
            if policy is not None and hasattr(policy, "set_run_id"):
                policy.set_run_id(ctx.run_id)
        else:
            for spec in self.config.tools:
                if isinstance(spec, ToolSpec):
                    by_name[spec.name] = spec
                    tool_schemas.append(spec.to_openai_schema())

        steps: List[Dict[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []
        final_output = ""
        completed = False
        consecutive_errors = 0
        wrote_code_file = False
        ran_command = False
        last_written_name = ""
        nudged_run = 0  # escalation counter (max 2 corrective turns)
        mutated = False        # did any call actually change the workspace?
        nudged_change = 0      # "nothing was changed" gate (max 2)
        nudged_rename = 0      # "rename not finished" gate (max 3)
        force_next_tool = False  # after a gate rejected a prose answer, the next turn MUST be a tool call (schema-level)
        pending_cmd_failure = False
        nudged_fix = 0
        # Loop guard: identical (tool, args) calls are counted; a call
        # repeated beyond the threshold is answered with an "unavailable"
        # result instead of being executed again, so the model must pick
        # another tool or answer directly.
        tool_call_counts: Dict[Tuple[str, str], int] = {}
        max_tool_repeats = 2

        for iteration in range(1, self.config.max_iterations + 1):
            if ctx.cancelled:
                break

            t0 = time.perf_counter()
            # Force a real tool call on the first turn when tools are on the
            # table and none has been made yet — same mechanism and same
            # reasoning as task_executor.py's FORCE_TOOL_CALL_MARKER
            # (imported, not reimplemented, so the schema-level "final"
            # removal stays in one place): by the time this node runs at
            # all, the intent gate has already decided the task genuinely
            # needs tools (see orcha/nodes/intent.py — CHAT is finalized
            # before ever reaching here), so a first turn that skips
            # straight to a text answer is answering a question the intent
            # gate already settled. Confirmed live this was a real gap, not
            # a hypothetical one: asked to research something specific via
            # web_search, a 3B local model sometimes just answered from
            # memory instead — plausible-looking but wrong (invented GGUF
            # quantization names that don't exist) — with zero tool calls
            # to show for it. A prompt-only rule (added separately in this
            # same system_prompt) didn't reliably prevent that; this closes
            # it at the grammar level, the same way task_executor.py's own
            # module comment already documented prompting alone failing to.
            # Forced until a call actually SUCCEEDS, not merely attempted:
            # confirmed live that a failed/malformed attempt (missing a
            # required argument) still counts as "a tool call happened" if
            # this only checked `not tool_calls` — the very next turn was
            # then free to give up in text instead of retrying with
            # corrected arguments, undermining the whole point of forcing
            # turn 1 in the first place. Deliberately stricter than
            # task_executor.py's own equivalent gate (state.tool_call_count,
            # incremented in Packet.record_tool_call as soon as a call is
            # PARSED, before validation/execution — so a same-shaped
            # malformed-first-call gap likely exists there too) rather than
            # copying that exact condition.
            call_system_prompt = system_prompt
            if tool_schemas and (force_next_tool or not any(tc.get("result_type") == "ok" for tc in tool_calls)):
                from .task_executor import FORCE_TOOL_CALL_MARKER
                call_system_prompt = system_prompt + FORCE_TOOL_CALL_MARKER
            force_next_tool = False
            try:
                # Use streaming variant when available to emit text deltas
                # for real-time token-by-token display in the UI.
                _stream_fn = getattr(self.config.completion_fn, "_streaming", None)
                if _stream_fn is not None and ctx.emit is not None:
                    async def _emit_delta(text: str) -> None:
                        await ctx.emit.emit(EVT_TEXT_DELTA, "agent", packet, text=text)
                    message = await _stream_fn(
                        messages, call_system_prompt, tool_schemas or None,
                        on_text_delta=_emit_delta,
                    )
                else:
                    message = await self.config.completion_fn(
                        messages, call_system_prompt, tool_schemas or None
                    )
                consecutive_errors = 0
            except Exception as exc:
                # A transient model failure must not become the agent's final
                # answer. Record it, nudge the model to retry (bounded), and
                # only fail the node after repeated consecutive errors.
                consecutive_errors += 1
                steps.append({
                    "iteration": iteration,
                    "response_preview": "",
                    "duration_ms": 0.0,
                    "status": "error",
                    "note": f"model call failed: {exc}",
                })
                if consecutive_errors >= _MAX_CONSECUTIVE_MODEL_ERRORS:
                    raise RuntimeError(
                        f"Model call failed {consecutive_errors} times in a row: {exc}"
                    ) from exc
                messages.append({
                    "role": "user",
                    "content": "The previous model call failed. Please retry the task.",
                })
                continue

            call_duration_ms = (time.perf_counter() - t0) * 1000
            content = message.get("content") or ""
            calls = message.get("tool_calls") or []

            # A completion that hit the token ceiling mid-tool-call produces
            # PLAUSIBLE but incomplete payloads (files cut at a random
            # line). Never execute those — force a clean re-issue instead.
            call_truncated = message.get("finish_reason") == "length"

            step = {
                "iteration": iteration,
                "response_preview": content[:200],
                "duration_ms": round(call_duration_ms, 2),
            }

            if calls:
                step["tool_calls_found"] = len(calls)
                step["status"] = "continued"
                steps.append(step)

                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": calls,
                }
                messages.append(assistant_msg)

                for i, call in enumerate(calls):
                    fn = call.get("function") or {}
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments", {})
                    args: Dict[str, Any] = {}
                    parse_error = ""
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args, strict=False) if raw_args.strip() else {}
                        except json.JSONDecodeError as exc:
                            parse_error = str(exc)
                    else:
                        args = dict(raw_args or {})

                    # Resolve the tool-call id and write it back into the
                    # stored assistant message (same dicts are referenced by
                    # assistant_msg) so every tool message carries a matching
                    # tool_call_id. Missing/zero/duplicate server ids would
                    # otherwise produce a 400 on the model's next request.
                    call_id = str(call.get("id") or f"call_{iteration}_{i}")
                    call["id"] = call_id

                    record: Dict[str, Any] = {
                        "name": name, "arguments": args, "iteration": iteration,
                        "tool_call_id": call_id,
                    }

                    if executor is not None and name and name not in by_name:
                        # A plausible-but-wrong tool name (find_and_replace, save_file, ...): resolve it through the alias table the
                        # executor already owns, instead of answering "no such tool" and spending a round-trip.
                        from ..capabilities.tool_aliases import resolve_tool_name
                        resolved = resolve_tool_name(name, set(by_name))
                        if resolved:
                            record["requested_name"] = name
                            record["name"] = name = resolved
                    if name in _EXECUTE_TOOLS and not args and not parse_error and self.config.workspace_roots:
                        # A small model that calls run_command with `{}` has no arguments to tolerate - supply the obvious one.
                        guessed = completion_checks.guess_run_command(self.config.workspace_roots, last_written_name)
                        if guessed:
                            args = {"command": guessed}
                            record["arguments"] = args
                            record["note"] = "no command was given; ran the project's tests / the file just written"
                    spec = by_name.get(name)
                    exec_exit_value = None
                    if call_truncated and name in _WRITE_CODE_TOOLS:
                        # Never write a half-generated file.
                        result_str = (
                            "[Tool call was cut off by the token limit before "
                            "the file completed. Re-issue the SAME write_file "
                            "with the COMPLETE content, and keep any prose to "
                            "one short sentence so it fits.]"
                        )
                        record["result_type"] = "error"
                        record["error_code"] = "truncated_call"
                        record["result"] = result_str[:2000]
                        tool_calls.append(record)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": str(call.get("id") or f"call_{iteration}_{i}"),
                            "content": result_str,
                        })
                        continue
                    if parse_error:
                        # Truncated/invalid arguments must not silently become
                        # an empty-args call that runs with no content.
                        result_str = (
                            f"[Tool call arguments could not be parsed as JSON "
                            f"({parse_error}); the call was NOT executed. "
                            "Re-issue the call with valid JSON arguments.]"
                        )
                        record["result_type"] = "error"
                        record["error_code"] = "malformed_arguments"
                    elif spec is None:
                        result_str = f"[No tool '{name}' available]"
                    else:
                        # Loop guard: identical repeated calls are not
                        # re-executed — the model gets an "unavailable"
                        # result so it must change strategy or answer.
                        #
                        # An EMPTY-argument call is exempted from the normal
                        # limit because it is a different failure with a
                        # different fix: the model picked the right tool but
                        # botched the envelope's double-encoded `arguments`
                        # string, so the answer is "call this same tool
                        # properly", not "give up on this tool". Confirmed
                        # live on a 3B local model: write_file/run_command
                        # were called with {} repeatedly, the guard disabled
                        # run_command after 3 tries, and the task failed with
                        # two of its three files never written. Validation
                        # failures now also return the tool's exact required
                        # shape (see ToolSpec._argument_error), so the extra
                        # attempts are genuinely recoverable rather than the
                        # same doomed call landing again.
                        empty_args = not args
                        repeat_key = (name, json.dumps(args, sort_keys=True, default=str))
                        tool_call_counts[repeat_key] = tool_call_counts.get(repeat_key, 0) + 1
                        effective_limit = max_tool_repeats + 3 if empty_args else max_tool_repeats
                        if tool_call_counts[repeat_key] > effective_limit:
                            result_str = (
                                f"[tool unavailable] {name} was called "
                                f"{tool_call_counts[repeat_key]} times with "
                                + (
                                    "no arguments and never ran. Either issue "
                                    f"{name} WITH its required arguments filled "
                                    "in, or use a different tool."
                                    if empty_args
                                    else "the same arguments and did not help. "
                                    "Choose another tool or answer directly."
                                )
                            )
                            record["result_type"] = "error"
                            record["error_code"] = "tool_loop_guard"
                        elif executor is not None:
                            result = await self._invoke_tool(executor, name, args, ctx.run_id, call_id)
                            result_str = result.to_message(max_chars=2000)
                            record["result_type"] = "ok" if result.ok else "error"
                            if not result.ok and result.error:
                                record["error_code"] = result.error.get("code")
                            # A local-search tool coming back genuinely empty
                            # looks identical to "I searched and there's
                            # nothing there" whether the model asked the
                            # right question of the wrong data source (the
                            # workspace) or the right one — confirmed live: a
                            # weak local model asked to "search the web" for
                            # something called search_text instead, with a
                            # clean query bearing no trace of the original
                            # "web" framing, and got back "No matches found."
                            # with nothing to correct course from. A system-
                            # prompt rule alone didn't reliably prevent the
                            # wrong call in the first place; this repairs it
                            # AFTER the fact by attaching an actionable hint
                            # to the empty result itself, costing one extra
                            # iteration instead of a wrong or absent answer.
                            if (
                                result.ok
                                and name in _LOCAL_SEARCH_TOOLS
                                and "web_search" in by_name
                                and "no matches found" in result_str.lower()
                            ):
                                result_str += (
                                    " (This only searched the local workspace. "
                                    "If the request needs information from the "
                                    "internet rather than this project's own "
                                    "files, call web_search instead.)"
                                )
                            # Command-level outcome (run_command/stream_output):
                            # a tool can execute fine while the COMMAND it ran
                            # failed — exit_code lives in the structured value.
                            exec_exit_value = getattr(result, "value", None)
                        else:
                            try:
                                result = await asyncio.to_thread(spec.invoke_kwargs, **args)
                                result_str = result if isinstance(result, str) else str(result)
                            except Exception as exc:
                                result_str = f"[Tool error: {exc}]"
                    record["result"] = result_str[:2000]
                    tool_calls.append(record)
                    if name in completion_checks.MUTATING_TOOLS and record.get("result_type") != "error" and spec is not None:
                        mutated = True
                    # Track the verification-gate state: did this call
                    # actually write code, and has anything been executed?
                    if name in _WRITE_CODE_TOOLS:
                        if record.get("result_type") != "error":
                            wrote_code_file = True
                            try:
                                last_written_name = str(
                                    (args or {}).get("path") or ""
                                ).replace(chr(92), "/").split("/")[-1]
                            except Exception:
                                pass
                            # A successful write CHANGES the world: rerunning
                            # a previously-blocked command is now meaningful
                            # (fix → re-verify). Clear exec repeats so the
                            # loop guard never deadlocks the fix cycle.
                            tool_call_counts.clear()
                    elif name in _EXECUTE_TOOLS:
                        exit_code = None
                        if isinstance(exec_exit_value, dict):
                            raw = exec_exit_value.get("exit_code")
                            if isinstance(raw, int):
                                exit_code = raw
                        if record.get("error_code") in ("validation_error", "malformed_arguments", "tool_loop_guard"):
                            pass  # the command never ran: the tool result already says what to fix; do not send the model rewriting files
                        elif record.get("result_type") == "error" or (
                            exit_code is not None and exit_code != 0
                        ):
                            pending_cmd_failure = True
                        else:
                            ran_command = True
                            pending_cmd_failure = False

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": result_str,
                    })
                continue

            # No tool calls → the model produced its final answer.
            if content:
                final_output = (final_output + content) if final_output else content

            # If the answer was cut off by the model's token budget
            # (finish_reason == "length"), don't declare it complete — feed
            # the partial text back and ask the model to continue so long
            # answers aren't truncated mid-sentence.
            if (
                message.get("finish_reason") == "length"
                and final_output
                and iteration < self.config.max_iterations
            ):
                step["status"] = "continued"
                step["note"] = "truncated; continuing"
                steps.append(step)
                if content:
                    messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was cut off. Continue exactly "
                        "from where you stopped, without repeating earlier text."
                    ),
                })
                continue

            # Fix gate: the last executed command FAILED (non-zero exit or
            # tool error) and the model is trying to talk instead of fixing.
            # Narrating a failure as if it were success is the single worst
            # small-model failure mode — force one corrective turn that must
            # change files and re-run with a different attempt.
            if (
                pending_cmd_failure
                and nudged_fix < 3
                and iteration < self.config.max_iterations
                and executor is not None
            ):
                nudged_fix += 1
                step["status"] = "continued"
                step["note"] = f"fix-gate: last command failed (non-zero exit), attempt {nudged_fix}"
                steps.append(step)
                # Replace the prose-only reply with a short rejection
                # marker: keeping SOME assistant turn preserves role
                # alternation (chat templates merge adjacent same-role
                # messages into one confusing blob), while dropping the
                # narration stops it from teaching the model that prose
                # is an acceptable gate response.
                final_output = ""
                if not wrote_code_file:
                    fix_msg = (
                        "The command failed because the files do not exist "
                        "yet. Do NOT run any command again first. Your next "
                        "action must be a write_file tool call that creates "
                        "the required file with its complete content. After "
                        "ALL files exist, call run_command. Never repeat an "
                        "identical failing command."
                    )
                else:
                    # Surface the actual failure text so a truncated write
                    # (SyntaxError/IndentationError) is recognized as such.
                    last_err = ""
                    for rec in reversed(tool_calls):
                        if rec.get("result_type") in ("error",) or (
                            isinstance(rec.get("result"), str)
                            and ("Error" in rec["result"] or "exit_code" in rec["result"])
                        ):
                            last_err = str(rec.get("result", ""))[:300]
                            break
                    fix_msg = (
                        "Your last command FAILED. Failure output:\n"
                        f"{last_err}\n"
                        "Fix the actual bug, then run_command again. If the "
                        "error is a Syntax/Indentation error, your previous "
                        "write_file was likely cut off — rewrite the ENTIRE "
                        "file with write_file in one complete tool call."
                    )
                if nudged_fix >= 3:
                    fix_msg = (
                        "STILL FAILING, and your previous reply contained no "
                        "tool call — that is not acceptable for this task. "
                        "Respond with ONLY a write_file tool call that fixes "
                        "the bug, followed by a run_command call. No prose."
                    )
                messages.append({"role": "user", "content": fix_msg})
                force_next_tool = True
                continue

            # Completion gates. Small models declare victory early: they read a file and answer without changing anything, or rename
            # a symbol in one file and stop. Both are checked mechanically (no extra model call) and answered with a concrete
            # instruction, in the same style as the fix/verification gates around them.
            if executor is not None and iteration < self.config.max_iterations:
                can_mutate = any(n in by_name for n in completion_checks.MUTATING_TOOLS)
                step["completion_check"] = {"needs_change": completion_checks.task_requires_change(task), "can_mutate": can_mutate,
                                            "mutated": mutated, "tools": sorted(by_name)[:60], "task": task[:80]}
                if can_mutate and not mutated and nudged_change < 2 and completion_checks.task_requires_change(task):
                    nudged_change += 1
                    step["status"] = "continued"
                    step["note"] = f"completion-gate: the task needs a change but nothing was changed, attempt {nudged_change}"
                    steps.append(step)
                    final_output = ""
                    messages.append({"role": "user", "content": completion_checks.nothing_changed_message()})
                    force_next_tool = True
                    continue
                rename = completion_checks.parse_rename(task)
                if rename and can_mutate and nudged_rename < 3 and self.config.workspace_roots:
                    left = completion_checks.find_remaining(self.config.workspace_roots, rename[0])
                    if left:
                        nudged_rename += 1
                        step["status"] = "continued"
                        step["note"] = f"completion-gate: rename unfinished ({len(left)} occurrence(s) of {rename[0]} left), attempt {nudged_rename}"
                        steps.append(step)
                        final_output = ""
                        messages.append({"role": "user", "content": completion_checks.rename_unfinished_message(rename[0], rename[1], left)})
                        force_next_tool = True
                        continue

            # Verification gate: the agent wrote code but is about to answer
            # without executing anything — i.e. its "output" would be a
            # prediction, not a measurement. Inject ONE corrective turn that
            # forces a real run_command before the final answer (deterministic
            # enforcement; small models ignore prompt-only instructions).
            if (
                wrote_code_file
                and not ran_command
                and nudged_run < 2
                and iteration < self.config.max_iterations
                and "run_command" in by_name
                and executor is not None
            ):
                nudged_run += 1
                step["status"] = "continued"
                step["note"] = f"verification: code written but never executed, attempt {nudged_run}"
                steps.append(step)
                # Same rejection-marker pattern as the fix gate: preserve
                # role alternation, drop the predicted-output narration.
                final_output = ""
                target = last_written_name or "the file you wrote"
                run_msg = (
                    "You have written code but have NOT executed it. "
                    "Call run_command now to run what you just wrote, "
                    "then use the real stdout/stderr you receive for "
                    "your final answer. Do not predict output. "
                    'Example response: {"action": "tool", "name": '
                    '"run_command", "arguments": {"command": '
                    '"python " + target + ""}}'
                )
                if nudged_run >= 2:
                    run_msg = (
                        "Your previous reply contained no tool call. Respond "
                        "with ONLY a run_command tool call that executes the "
                        "code you wrote (e.g. python filename.py). No prose."
                    )
                messages.append({"role": "user", "content": run_msg})
                force_next_tool = True
                continue

            step["status"] = "completed" if final_output else "continued"
            steps.append(step)

            if final_output:
                completed = True
            break

        return steps, tool_calls, final_output, completed

    @staticmethod
    def _parse_text_tool_args(raw: str) -> Dict[str, Any]:
        """Leniently parse the argument portion of a ``TOOL:name:...`` call.

        Tries, in order:
          1. A single JSON object (the recommended format for small models).
          2. ``key=value, key=value`` pairs (unquoted values; quotes optional).
          3. A bare string wrapped as ``{"value": raw}``.

        Never raises — on total failure we return the raw text so the executor
        can surface a clear validation error the model can recover from.
        """
        raw = (raw or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"value": parsed}
        except (json.JSONDecodeError, ValueError):
            pass
        # key=value pairs (quoted / bracketed / bare values all accepted).
        if "=" in raw:
            args: Dict[str, Any] = {}
            for m in re.finditer(
                r'(\w+)\s*=\s*("(?:[^"]*)"|\[[^\]]*\]|[^\s,]+)', raw
            ):
                k = m.group(1)
                v = m.group(2).strip().strip('"')
                if v.lstrip("-").replace(".", "", 1).isdigit():
                    v = float(v) if "." in v else int(v)
                args[k] = v
            if args:
                return args
        return {"value": raw}

    async def _invoke_tool(
        self, executor: Any, name: str, args: Dict[str, Any], run_id: str, tool_call_id: str,
    ) -> Any:
        """
        Execute one tool call without blocking the event loop.

        When the permission policy requires interactive approval, the call
        is registered with the configured ApprovalBroker (scoped to this
        run + tool call) and this coroutine waits (bounded by
        ``approval_timeout_s``) for the user's decision before executing or
        relaying the denial.
        """
        result = await asyncio.to_thread(executor.invoke, name, **args)
        if result.ok or (result.error or {}).get("code") != "approval_required":
            return result

        broker = self.config.approval_broker
        if broker is None:
            return result

        entry = broker.request(run_id, tool_call_id, name, args)
        decision = await broker.wait(entry["id"], timeout=self.config.approval_timeout_s)
        # Feed the permission circuit breaker: a human answered (yes OR no)
        # resets it; an unanswered timeout ticks toward fast-denial mode.
        policy = getattr(executor, "policy", None)
        if policy is not None and hasattr(policy, "record_wait_outcome"):
            policy.record_wait_outcome(decided=decision is not None)
        if decision is not True:
            # Rejected or no decision in time — relay the structured denial
            # so the agent asks the user instead of guessing.
            return result
        # The broker has recorded a run-scoped, one-shot grant; the policy
        # consumes it on re-execution, so only THIS run's identical call is
        # auto-approved — never another run's.
        return await asyncio.to_thread(executor.invoke, name, **args)

    async def _run_text_loop(
        self,
        system_prompt: str,
        task: str,
        ctx: RunContext,
        packet: "OrchaPacket",
    ):
        """Legacy text-protocol agent loop (TOOL:name:arg) — kept for backward
        compatibility when no completion_fn is configured."""
        scratchpad: List[str] = []

        steps: List[Dict[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []
        final_output = ""
        completed = False
        _tool_re = re.compile(r"TOOL:(\w+):(.+?)(?=TOOL:|$)", re.DOTALL)
        # Same loop guard _run_native_loop has — without it, a model that
        # falls back to this text protocol (weaker/free-tier models
        # without native tool-calling support) can re-issue an identical
        # no-op call every iteration with nothing telling it to stop. Each
        # iteration here is a full model round-trip, and on a slow free
        # model that's 60-100+ seconds burned per repeat for zero
        # progress — confirmed directly: a live run against
        # nvidia/nemotron-3-ultra-550b-a55b:free called `current_workspace`
        # twice in a row with identical (empty) arguments before the test
        # harness gave up waiting.
        tool_call_counts: Dict[Tuple[str, str], int] = {}
        max_tool_repeats = 2

        for iteration in range(1, self.config.max_iterations + 1):
            if ctx.cancelled:
                break

            # Build the prompt.
            prompt_parts = []
            if scratchpad:
                prompt_parts.append("Scratchpad:\n" + "\n".join(scratchpad[-10:]))
            prompt_parts.append("Continue working on the task.")
            prompt = "\n\n".join(prompt_parts)

            # Call the model.
            t0 = time.perf_counter()
            if self.config.model_fn is not None:
                try:
                    response = await self.config.model_fn(prompt, system_prompt)
                except Exception as exc:
                    response = f"[Model error: {exc}]"
            else:
                # Dry run: produce a synthetic response.
                response = f"[Dry run iteration {iteration}] Working on: {task[:50]}..."

            call_duration_ms = (time.perf_counter() - t0) * 1000

            step = {
                "iteration": iteration,
                "response_preview": response[:200],
                "duration_ms": round(call_duration_ms, 2),
            }

            # Check for stop phrases.
            for phrase in self.config.stop_phrases:
                if phrase in response:
                    final_output = response.split(phrase, 1)[-1].strip()
                    completed = True
                    step["stopped_by"] = phrase
                    break

            if completed:
                step["status"] = "completed"
                steps.append(step)
                # Stream the final answer token-by-token so the UI animates it.
                if ctx.emit is not None and final_output:
                    words = final_output.split(" ")
                    for i, w in enumerate(words):
                        chunk = w + (" " if i < len(words) - 1 else "")
                        await ctx.emit.emit(EVT_TEXT_DELTA, "agent", packet, text=chunk)
                        await asyncio.sleep(0.012)
                break

            # Check for tool calls (free-text protocol: "TOOL:name:{json}").
            tool_matches = _tool_re.findall(response)
            for tool_name, tool_arg in tool_matches:
                tool_name = tool_name.strip()
                tool_arg = tool_arg.strip()
                tool_call = {
                    "name": tool_name,
                    "arg": tool_arg[:200],
                    "iteration": iteration,
                }
                # Prefer the capability executor (filesystem / MCP / skills).
                result_str = None
                executor = self.config.executor
                if executor is not None and executor.has(tool_name):
                    args = self._parse_text_tool_args(tool_arg)
                    repeat_key = (tool_name, json.dumps(args, sort_keys=True, default=str))
                    tool_call_counts[repeat_key] = tool_call_counts.get(repeat_key, 0) + 1
                    if tool_call_counts[repeat_key] > max_tool_repeats:
                        result_str = (
                            f"[tool unavailable] {tool_name} was called "
                            f"{tool_call_counts[repeat_key]} times with the same "
                            "arguments and did not help. Choose another tool or "
                            "answer directly."
                        )
                    else:
                        try:
                            res = await asyncio.to_thread(executor.invoke, tool_name, **args)
                            result_str = (res.to_message(max_chars=2000)
                                          if hasattr(res, "to_message")
                                          else str(res))
                        except Exception as exc:
                            result_str = f"[Tool error: {exc}]"
                else:
                    # Legacy ToolSpec.fns (backward compatibility).
                    for spec in self.config.tools:
                        if spec.name == tool_name and spec.fn is not None:
                            try:
                                result_str = spec.fn(tool_arg)
                                result_str = result_str if isinstance(result_str, str) else str(result_str)
                            except Exception as exc:
                                result_str = f"[Tool error: {exc}]"
                            break
                if result_str is None:
                    result_str = f"[No tool '{tool_name}' available]"
                tool_call["result"] = result_str[:500]
                tool_calls.append(tool_call)
                scratchpad.append(f"Tool {tool_name} → {result_str[:200]}")

            step["status"] = "continued"
            step["tool_calls_found"] = len(tool_matches)
            steps.append(step)

            # Trim scratchpad if needed.
            total_len = sum(len(s) for s in scratchpad)
            if total_len > self.config.scratchpad_max_len:
                while scratchpad and total_len > self.config.scratchpad_max_len * 0.7:
                    removed = scratchpad.pop(0)
                    total_len -= len(removed)

        return steps, tool_calls, final_output, completed


__all__ = ["AgentNode", "AgentConfig"]
