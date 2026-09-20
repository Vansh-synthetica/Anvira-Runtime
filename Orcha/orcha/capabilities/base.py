"""
orcha.capabilities.base
=======================
Shared building blocks for the capability/tool system.

This module defines the vocabulary every capability uses:

- permission kinds (read / write / execute / network / dangerous)
- safety levels (safe / cautious / dangerous)
- structured errors (ToolError)
- structured results (ToolResult)
- the permission policy + runtime executor that runs tool calls safely
- the compact :func:`spec` builder that turns an implementation into a
  full ToolSpec (name, description, JSON schema, permissions, safety
  level, validation, result format)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..nodes.tool import ToolSchema, ToolSpec
from .outputstore import OutputStore
from .readstate import ReadStateCache
from .rules import PermissionRules
from .tool_aliases import resolve_tool_name

# ── Permission vocabulary ─────────────────────────────────────────────────────
READ = "read"
WRITE = "write"
EXECUTE = "execute"
NETWORK = "network"
DANGEROUS = "dangerous"
PERMISSIONS: tuple = (READ, WRITE, EXECUTE, NETWORK, DANGEROUS)

# ── Safety levels ─────────────────────────────────────────────────────────────
SAFE = "safe"
CAUTIOUS = "cautious"
DANGEROUS_LEVEL = "dangerous"

# A destructive/mutating action is dangerous when it permanently changes
# or removes files; these tools require approval in approval mode.
_DANGEROUS_PERMS = {DANGEROUS, EXECUTE, NETWORK}


class ToolError(Exception):
    """
    Structured tool failure. ``code`` is machine-readable (and stable for
    tests), ``message`` is model/UI-facing, ``detail`` is free-form.
    """

    def __init__(self, code: str, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.detail is not None:
            d["detail"] = self.detail
        return d


@dataclass
class ToolResult:
    """
    Uniform result envelope returned by the executor for every tool call.

    ``ok=True`` means the tool produced ``value`` (string, dict, list…).
    ``ok=False`` means the call failed; ``error`` carries the structured
    error (code + message + detail). ``to_message()`` renders the result
    the way the model should see it.
    """

    ok: bool
    value: Any = None
    error: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, value: Any = None, **metadata: Any) -> "ToolResult":
        return cls(ok=True, value=value, metadata=metadata)

    @classmethod
    def failure(cls, code: str, message: str, detail: Any = None) -> "ToolResult":
        error: Dict[str, Any] = {"code": code, "message": message}
        if detail is not None:
            error["detail"] = detail
        return cls(ok=False, error=error)

    def to_message(self, max_chars: int = 16000) -> str:
        if self.ok:
            value = self.value
            if isinstance(value, str):
                text = value
            else:
                try:
                    text = json.dumps(value, default=str, ensure_ascii=False, indent=2)
                except Exception:
                    text = str(value)
        else:
            text = json.dumps(self.error, default=str, ensure_ascii=False)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... (truncated {len(text) - max_chars} chars)"
        return text


@dataclass
class CapabilityContext:
    """
    Everything a capability builder needs to bind its tools.

    ``roots`` is the allow-list of absolute workspace directories. All other
    fields describe the current workspace session (project, attachments,
    active/recent files) and are used by the Workspace capability.
    """

    roots: List[str] = field(default_factory=list)
    project_path: Optional[str] = None
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    active_file: Optional[str] = None
    recent_files: List[str] = field(default_factory=list)
    # When set, filesystem tools record what the agent read and refuse to
    # modify files that are unread, partially seen, or changed since read
    # (stale-write protection). None = advisory-free legacy behavior.
    read_state: Optional[ReadStateCache] = None


# ── Permission policy + executor ──────────────────────────────────────────────

class PermissionPolicy:
    """
    Decides whether a tool call may run, needs approval, or is denied.

    Evaluation order (strict precedence — see ``evaluate``):

    1. rule strings (:class:`~orcha.capabilities.rules.PermissionRules`) —
       deny beats ask beats allow, always;
    2. trusted tool / capability pre-authorization → allow;
    3. access mode:

    Modes
    -----
    action   — run everything the agent asks (the user has explicitly put
               the agent in action mode / full trust).
    full     — same as action.
    auto     — like action for safe tools, but anything needing approval
               consults the interactive broker; repeated unanswered
               requests degrade to fast denials (circuit breaker) so an
               absent human cannot stall a desktop run forever.
    approval — dangerous/execute/network calls are not auto-approved; they
               return a structured ``approval_required`` result the agent
               relays to the user.
    partial  — like approval, dangerous calls need approval.

    ``trusted_tools`` / ``trusted_capabilities`` can pre-authorize specific
    tools or whole capabilities even in approval mode.
    """

    # Consecutive unanswered (timeout) approval waits before auto mode
    # stops blocking on the broker and denies fast instead.
    _MAX_UNANSWERED = 3

    def __init__(
        self,
        access_mode: str = "action",
        trusted_tools: Optional[List[str]] = None,
        trusted_capabilities: Optional[List[str]] = None,
        broker: Optional["ApprovalBroker"] = None,
        rules: Optional["PermissionRules"] = None,
    ) -> None:
        self.access_mode = access_mode or "action"
        self.trusted_tools = set(trusted_tools or [])
        self.trusted_capabilities = set(trusted_capabilities or [])
        self._broker = broker
        self.rules = rules or PermissionRules()
        # Circuit-breaker counters (session-scoped).
        self._unanswered_waits = 0
        self.interactive_exhausted = False
        # Bound per run by the agent loop; grants are consumed strictly for
        # the run that requested them (see ApprovalBroker.has_approval).
        self._run_id: Optional[str] = None

    def set_run_id(self, run_id: Optional[str]) -> None:
        """Bind this policy to one run so approval grants can never leak
        across runs sharing the same broker."""
        self._run_id = run_id

    def evaluate(self, spec: ToolSpec, kwargs: Dict[str, Any]) -> str:
        """
        The full pipeline: ``"allow" | "ask" | "deny"``. Rule strings have
        absolute precedence inside each kind and bypass mode/trust defaults
        entirely: an explicit ``ask`` rule forces interactivity even on a
        safe tool, and only when NO rule matches do trust lists and the
        access mode decide. Deny decisions are final and bypass-immune.
        """
        resolved = self.rules.evaluate(spec.name, kwargs)
        if resolved is not None:
            if resolved.decision == "deny":
                return "deny"
            if resolved.decision == "ask":
                return self._resolve_ask(spec, forced=True)
            return "allow"
        if spec.name in self.trusted_tools or spec.capability in self.trusted_capabilities:
            return "allow"
        if self.access_mode in ("action", "full"):
            return "allow"
        return self._resolve_ask(spec)

    def _resolve_ask(self, spec: ToolSpec, *, forced: bool = False) -> str:
        """Mode semantics for calls that need approval. ``forced=True``
        comes from an explicit ask RULE, which applies regardless of the
        tool's own danger level."""
        if forced or spec.needs_approval():
            if (
                self.access_mode == "auto"
                and self.interactive_exhausted
            ):
                # Nobody answered the last N interactive requests: stop
                # stalling the run — deny fast instead of waiting again.
                return "deny"
            return "ask"
        return "allow"

    def requires_approval(self, spec: ToolSpec, kwargs: Dict[str, Any]) -> bool:
        """Backward-compatible boolean view of :meth:`evaluate`."""
        return self.evaluate(spec, kwargs) == "ask"

    def record_wait_outcome(self, decided: bool) -> None:
        """
        Feed the circuit breaker after one interactive wait: ``decided``
        False means the request timed out unanswered. After
        ``_MAX_UNANSWERED`` consecutive timeouts the policy degrades to
        fast denials (see :meth:`_resolve_ask`). Any decision resets the
        counter.
        """
        if decided:
            self._unanswered_waits = 0
            self.interactive_exhausted = False
            return
        self._unanswered_waits += 1
        if self._unanswered_waits >= self._MAX_UNANSWERED:
            self.interactive_exhausted = True

    def approve(self, spec: ToolSpec, kwargs: Dict[str, Any]) -> bool:
        # Without an interactive approval channel, approval-mode calls are
        # denied with a structured result the agent can surface.
        if self._broker is None or self._run_id is None:
            return False
        # An approved request leaves a one-shot grant the broker consumes
        # here, so re-executing the same tool+args proceeds exactly once —
        # and only for the run that obtained the approval.
        return self._broker.has_approval(self._run_id, spec.name, kwargs)


class ApprovalBroker:
    """
    Run-scoped registry of pending tool-approval requests.

    When the permission policy denies a call with ``approval_required``,
    the agent loop registers a request here (scoped to run_id + tool_call_id)
    and waits (bounded) for a human decision via the API. An approved request
    leaves a short-lived, one-shot grant that the policy consumes when the
    tool call is re-executed. Grants and pending entries are strictly keyed
    by run: a decision obtained by one run can never authorize an identical
    call in a different run, and ``clear_run`` removes all state when a run
    terminates.
    """

    # Grants are only valid for a short window: they exist solely to bridge
    # the wait→re-execute gap inside one agent loop, never across runs.
    _GRANT_TTL_S = 120.0

    def __init__(self) -> None:
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._events: Dict[str, asyncio.Event] = {}
        self._grants: Dict[str, Tuple[bool, float]] = {}
        # Structured "don't ask again" updates recorded with decisions,
        # drained by the API layer into PermissionRules (data, not
        # hardcoded branches).
        self._updates: List[Dict[str, Any]] = []

    def _key(self, tool: str, arguments: Dict[str, Any]) -> str:
        payload = json.dumps(arguments, sort_keys=True, default=str)
        return hashlib.sha256(f"{tool}|{payload}".encode("utf-8")).hexdigest()

    def _grant_key(self, run_id: str, tool: str, arguments: Dict[str, Any]) -> str:
        # Run-scoped grant keys: an approval obtained by one run can never
        # authorize another run's identical call, even in the same workspace
        # with the same agent and the same tool arguments.
        return f"{run_id}|{self._key(tool, arguments)}"

    def request(
        self,
        run_id: str,
        tool_call_id: str,
        tool: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Register (or reuse) a pending approval request for a tool call.

        The request is scoped to ``run_id`` + ``tool_call_id``; a pending
        request from a different run is never reused.
        """
        key = self._key(tool, arguments)
        for entry in self._pending.values():
            if (
                entry["status"] == "pending"
                and entry["run_id"] == run_id
                and entry["key"] == key
            ):
                return entry
        approval_id = str(uuid.uuid4())
        entry = {
            "id": approval_id,
            "run_id": run_id,
            "tool_call_id": tool_call_id,
            "tool": tool,
            "arguments": arguments,
            "status": "pending",
            "key": key,
            "created_at": time.time(),
        }
        self._pending[approval_id] = entry
        self._events[approval_id] = asyncio.Event()
        return entry

    async def wait(self, approval_id: str, timeout: float) -> Optional[bool]:
        """Wait for a decision. True=approved, False=rejected, None=timeout."""
        event = self._events.get(approval_id)
        if event is None:
            return None
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        return self._pending.get(approval_id, {}).get("status") == "approved"

    def decide(
        self,
        approval_id: str,
        approved: bool,
        run_id: Optional[str] = None,
        update: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Resolve a pending request. Returns False if unknown/already decided.

        The pending-status check IS the atomic claim: under asyncio's
        single-threaded execution, exactly one ``decide`` call can observe
        ``status == "pending"`` per approval, so concurrent deciders (local
        UI, remote bridge, automation) race safely and the first wins —
        every loser gets False instead of a double resolution.

        ``update`` optionally carries a structured permission update
        (e.g. {"kind": "add_rule", "rule": "run_command(git *)",
        "source": "session"}) recorded alongside the approval for the API
        layer to persist into :class:`PermissionRules`.
        """
        entry = self._pending.get(approval_id)
        if entry is None or entry["status"] != "pending":
            return False
        if run_id is not None and entry["run_id"] != run_id:
            return False
        entry["status"] = "approved" if approved else "rejected"
        if approved:
            self._grants[self._grant_key(entry["run_id"], entry["tool"], entry["arguments"])] = (
                True,
                time.time(),
            )
        if update is not None and isinstance(update, dict):
            self._updates.append({
                "run_id": entry["run_id"],
                "approval_id": approval_id,
                "tool": entry["tool"],
                "approved": approved,
                "update": update,
            })
        event = self._events.get(approval_id)
        if event is not None:
            event.set()
        return True

    def drain_updates(self) -> List[Dict[str, Any]]:
        """Pop every recorded decision update (for rule persistence)."""
        out, self._updates = self._updates, []
        return out

    def has_approval(self, run_id: str, tool: str, arguments: Dict[str, Any]) -> bool:
        """One-shot check: the policy consumes the grant when re-executing.

        Strictly scoped to ``run_id`` — a grant recorded for another run can
        never satisfy this call.
        """
        grant = self._grants.pop(self._grant_key(run_id, tool, arguments), None)
        if grant is None:
            return False
        granted, granted_at = grant
        if time.time() - granted_at > self._GRANT_TTL_S:
            return False
        return granted

    def clear_run(self, run_id: str) -> None:
        """Drop every pending request, event, and grant belonging to a run.

        Called when the run terminates so a stale pending approval can never
        linger into (or be decided by) a later run, and so its grant can
        never authorize an identical call in a new run.
        """
        stale_ids = [
            entry_id
            for entry_id, entry in self._pending.items()
            if entry.get("run_id") == run_id
        ]
        for entry_id in stale_ids:
            self._pending.pop(entry_id, None)
            self._events.pop(entry_id, None)
        self._updates = [u for u in self._updates if u.get("run_id") != run_id]
        prefix = f"{run_id}|"
        for grant_key in [k for k in self._grants if k.startswith(prefix)]:
            self._grants.pop(grant_key, None)

    def list_pending(self, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        out = []
        for entry in self._pending.values():
            if entry["status"] != "pending":
                continue
            if run_id is not None and entry["run_id"] != run_id:
                continue
            out.append({
                "id": entry["id"],
                "run_id": entry["run_id"],
                "tool": entry["tool"],
                "arguments": entry["arguments"],
                "created_at": entry["created_at"],
            })
        return out


# Known wrong-but-plausible argument names local/edge models guess even
# when the schema documents the correct one — confirmed empirically with a
# small local model that alternated between the correct and an incorrect
# name for the same tool across otherwise-identical runs, and didn't always
# self-correct after a validation error. Keyed by the REAL parameter name.
_ARG_NAME_ALIASES: Dict[str, tuple] = {
    "path": ("filename", "file_path", "filepath", "directory", "dir", "dirname", "folder_path", "folder"),
    "old_path": ("source_path", "src_path", "from_path"),
    "new_path": ("dest_path", "destination_path", "to_path"),
    "source": ("src", "from_path", "old_path", "source_path"),
    "target": ("dest", "destination", "to_path", "new_path", "target_path"),
    "content": ("text", "file_content", "data", "body"),
    "command": ("cmd", "shell_command", "cmdline"),
    # edit_file / apply_patch: names small models (and opencode-trained ones) reach for
    "old_string": ("old_str", "oldString", "old_text", "oldText", "old", "search", "find", "target_string", "original"),
    "new_string": ("new_str", "newString", "new_text", "newText", "new", "replace", "replacement", "replace_with"),
    "replace_all": ("replaceAll", "all", "global"),
    "patch_text": ("patch", "patchText", "patch_content", "patch_body", "diff", "input", "changes", "content", "text"),
}


def _normalize_arg_aliases(kwargs: Dict[str, Any], real_properties: set) -> Dict[str, Any]:
    """
    Deterministically rename a known wrong-but-common argument alias (e.g.
    "filename") to the tool's real parameter name (e.g. "path") whenever
    the model used the alias, the real name isn't already supplied, and the
    real name is actually one of this tool's parameters — so a call that
    would otherwise fail validation over a purely cosmetic naming mismatch
    just works, instead of depending entirely on the model noticing and
    fixing its own mistake.
    """
    if not real_properties:
        return kwargs
    result = dict(kwargs)
    for real_name, aliases in _ARG_NAME_ALIASES.items():
        if real_name not in real_properties:
            continue
        has_real = real_name in result
        for alias in aliases:
            if alias not in result:
                continue
            if has_real:
                # Both supplied — the real name wins; drop the stray alias
                # rather than passing it through as an unexpected kwarg.
                result.pop(alias)
            else:
                result[real_name] = result.pop(alias)
                has_real = True
    return result


class ToolExecutor:
    """
    Runtime executor for a set of ToolSpecs. This is the single place where
    a function call is validated, policy-checked, executed, and wrapped into
    a ToolResult — every agent loop and every API surface uses it.
    """

    def __init__(
        self,
        tools: List[ToolSpec],
        policy: Optional[PermissionPolicy] = None,
        output_store: Optional[OutputStore] = None,
    ) -> None:
        self._tools = list(tools)
        self._by_name: Dict[str, ToolSpec] = {t.name: t for t in self._tools}
        self._policy = policy or PermissionPolicy()
        # When set, successful results larger than the store's threshold are
        # persisted to disk; the model sees a preview + saved-file path.
        self._output_store = output_store

    @property
    def policy(self) -> PermissionPolicy:
        return self._policy

    @property
    def output_store(self) -> Optional[OutputStore]:
        return self._output_store

    def tools(self) -> List[ToolSpec]:
        return list(self._tools)

    def has(self, name: str) -> bool:
        return name in self._by_name

    def names(self) -> List[str]:
        return list(self._by_name)

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools]

    def describe(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "capability": t.capability,
                "permissions": t.permissions,
                "safety_level": t.safety_level,
                "requires_approval": t.needs_approval(),
                "read_only": t.is_read_only(),
                "concurrency_safe": t.is_concurrency_safe(),
                "parameters": t.parameters or {},
            }
            for t in self._tools
        ]

    def get_spec(self, name: str) -> Optional[ToolSpec]:
        return self._by_name.get(name)

    def is_concurrency_safe(self, name: str) -> bool:
        """Whether ``name`` may run in parallel with other safe calls."""
        spec = self._by_name.get(name)
        return spec is not None and spec.is_concurrency_safe()

    def plan_parallel(
        self, calls: List[Tuple[str, Dict[str, Any]]],
    ) -> List[List[Tuple[str, Dict[str, Any]]]]:
        """
        Batch tool calls for execution: concurrency-safe calls group into one
        wave; any unsafe call forms a single-call wave (strictly alone), so
        writes/execution never race reads mid-flight.
        """
        waves: List[List[Tuple[str, Dict[str, Any]]]] = []
        batch: List[Tuple[str, Dict[str, Any]]] = []
        for call in calls:
            if self.is_concurrency_safe(call[0]):
                batch.append(call)
            else:
                if batch:
                    waves.append(batch)
                    batch = []
                waves.append([call])
        if batch:
            waves.append(batch)
        return waves

    def _reroute_edit_family(self, tool_name: str, kwargs: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
        if tool_name == "apply_patch" and "edit_file" in self._by_name and "patch_text" not in kwargs:
            probe = _normalize_arg_aliases(kwargs, {"path", "old_string", "new_string", "replace_all"})
            if {"path", "old_string", "new_string"} <= set(probe) and not any(
                    isinstance(v, str) and v.lstrip().startswith("*** Begin Patch") for v in kwargs.values()):
                return "edit_file", kwargs
        if tool_name == "edit_file" and "apply_patch" in self._by_name:
            for value in kwargs.values():
                if isinstance(value, str) and value.lstrip().startswith("*** Begin Patch"):
                    return "apply_patch", {"patch_text": value}
        return None

    def invoke(self, tool_name: str, **kwargs: Any) -> ToolResult:
        spec = self._by_name.get(tool_name)
        if spec is None:
            # Try to resolve a plausible-but-wrong guess (a curated alias
            # like "write_tool" for "write_file", or a close typo) to the
            # real tool BEFORE giving up — this is what used to cost a
            # whole extra round-trip (or, for a weaker model that never
            # notices the "available tools" hint below, the task silently
            # failing outright). See orcha/capabilities/tool_aliases.py.
            resolved_name = resolve_tool_name(tool_name, set(self._by_name))
            if resolved_name is not None:
                spec = self._by_name[resolved_name]
                tool_name = resolved_name
        if spec is None:
            # List the real available names so the model can self-correct a
            # plausible-but-wrong guess (e.g. "copy_files" when the actual
            # tool is "copy_file") on its next turn instead of giving up and
            # telling the user the capability doesn't exist at all.
            available = ", ".join(sorted(self._by_name)) or "none"
            return ToolResult.failure(
                "unknown_tool",
                f"No tool named '{tool_name}' is available. Available tools: {available}.",
            )

        # A model that picks the wrong member of the edit family still gets its work done: an edit-style call
        # (path/old_string/new_string) sent to apply_patch runs as edit_file, and a patch envelope sent to
        # edit_file runs as apply_patch. Both are unambiguous, so no retry round-trip is wasted.
        rerouted = self._reroute_edit_family(tool_name, kwargs)
        if rerouted is not None:
            tool_name, kwargs = rerouted
            spec = self._by_name[tool_name]

        real_properties = set((spec.parameters or {}).get("properties") or {})
        kwargs = _normalize_arg_aliases(kwargs, real_properties)

        validation_error = spec.validate_kwargs(**kwargs)
        if validation_error:
            return ToolResult.failure(
                "validation_error",
                validation_error,
                detail={"tool": tool_name, "arguments": _redact(kwargs)},
            )

        # Full permission pipeline: rule strings first (deny/ask/allow),
        # then trust lists, then access mode. Denials are final and
        # bypass-immune; asks flow into the interactive broker.
        decision = self._policy.evaluate(spec, kwargs)
        if decision == "deny":
            resolved = self._policy.rules.evaluate(tool_name, kwargs)
            reason = (
                f"denied by rule '{resolved.rule.display}'"
                if resolved is not None and resolved.rule is not None
                else "denied by policy"
            )
            return ToolResult.failure(
                "permission_denied",
                f"Use of '{tool_name}' was denied ({reason}). This decision "
                "is final — choose a different approach or ask the user.",
                detail={"tool": tool_name, "rule": resolved.rule.raw if resolved and resolved.rule else None},
            )
        if decision == "ask" and not self._policy.approve(spec, kwargs):
            resolved = self._policy.rules.evaluate(tool_name, kwargs)
            ask_rule = resolved.rule.raw if resolved is not None and resolved.rule else None
            return ToolResult.failure(
                "approval_required",
                f"Approval is required before using '{tool_name}' "
                f"(permissions: {', '.join(spec.permissions)}, "
                f"safety: {spec.safety_level}). Explain the step and ask the "
                "user to approve it before continuing.",
                detail={
                    "tool": tool_name,
                    "permissions": spec.permissions,
                    "safety_level": spec.safety_level,
                    "ask_rule": ask_rule,
                },
            )

        try:
            if spec.kwargs_fn is not None:
                value = spec.kwargs_fn(**kwargs)
            elif spec.fn is not None:
                value = spec.fn(kwargs) if kwargs else spec.fn()
            else:
                raise ToolError("not_implemented", f"Tool '{tool_name}' has no implementation attached.")
            # Oversized-output persistence: huge results go to disk, the
            # model sees a bounded preview + the saved path. Short results
            # (including _FileChangeResult strings carrying .file_changes)
            # pass through untouched.
            if self._output_store is not None and not isinstance(value, BaseException):
                attr_changes = getattr(value, "file_changes", None)
                if not isinstance(attr_changes, list):
                    value, persisted = self._output_store.wrap_if_oversized(tool_name, value)
                    if persisted:
                        return ToolResult.success(
                            value, tool=tool_name, persisted_output=persisted,
                        )
            return ToolResult.success(value, tool=tool_name)
        except ToolError as exc:
            return ToolResult.failure(exc.code, exc.message, exc.detail)
        except Exception as exc:  # never crash the runtime
            return ToolResult.failure(
                "tool_error",
                f"{tool_name} failed: {exc}",
                detail={"tool": tool_name, "error_type": type(exc).__name__},
            )


def _redact(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep error payloads small and avoid echoing huge content bodies."""
    out: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if isinstance(v, str) and len(v) > 500:
            out[k] = v[:500] + f"... ({len(v)} chars)"
        else:
            out[k] = v
    return out


# ── Compact ToolSpec builder ──────────────────────────────────────────────────

def spec(
    name: str,
    description: str,
    parameters: ToolSchema,
    impl: Callable[..., Any],
    *,
    permissions: List[str] = ("read",),
    safety_level: str = SAFE,
    capability: Optional[str] = None,
    result_format: Optional[Dict[str, Any]] = None,
    validate: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
    requires_approval: Optional[bool] = None,
    read_only: Optional[bool] = None,
    concurrency_safe: Optional[bool] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> ToolSpec:
    """Build a fully-annotated ToolSpec from an implementation callable."""
    return ToolSpec(
        name=name,
        description=description,
        parameters=parameters,
        kwargs_fn=impl,
        permissions=list(permissions),
        safety_level=safety_level,
        capability=capability,
        result_format=result_format,
        validate_fn=validate,
        requires_approval=requires_approval,
        read_only=read_only,
        concurrency_safe=concurrency_safe,
        meta=meta,
        category=capability or "general",
    )


__all__ = [
    "READ", "WRITE", "EXECUTE", "NETWORK", "DANGEROUS", "PERMISSIONS",
    "SAFE", "CAUTIOUS", "DANGEROUS_LEVEL",
    "ToolError", "ToolResult", "CapabilityContext",
    "PermissionPolicy", "ToolExecutor", "spec",
    "ReadStateCache", "OutputStore", "PermissionRules",
]
