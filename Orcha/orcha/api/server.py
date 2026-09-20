"""
orcha.api.server
================
FastAPI server that exposes Orcha over HTTP and serves the web UI.

The orchestrator is built inside a FastAPI ``lifespan`` handler, NOT at
module import time. Importing this module (for ``TestClient``, tooling,
or programmatic use) is therefore instant and never blocks on a network
probe or triggers an event-loop conflict.

Versioning
----------
The stable public API lives under ``/v1`` (e.g. ``POST /v1/query``).
The un-versioned paths (``/query``, ``/health``, …) are retained as
thin redirects for backwards compatibility but are considered
deprecated and will be removed in a future major release.

ORCHA3 graph endpoints
-----------------------
- ``POST /v1/run``           Run a graph for a query (durable, returns run_id).
- ``GET  /v1/run/{id}``      Get the latest checkpoint / result for a run.
- ``GET  /v1/run/{id}/events`` SSE stream of events for a run.
- ``POST /v1/run/{id}/resume`` Resume a crashed/interrupted run.
- ``POST /v1/run/{id}/replay`` Deterministically replay a run from history.
- ``GET  /v1/runs``          List all known run ids.
- ``POST /v1/run/{id}/approval`` Decide a paused final-answer approval
  (LangGraph agent engine with ``require_approval``).

Errors
------
All error responses share one envelope::

    {"error": {"type": "...", "message": "...", "status": 500, "trace_id": "..."}}

so clients can parse failures uniformly. Internal tracebacks are never
leaked to the client.

Start:
    uvicorn orcha.api.server:app --reload --port 8420

Then open http://localhost:8420 for the dashboard.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import os
import platform
import re
import secrets
import sys
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from pathlib import Path
from .. import __version__ as ORCHA_VERSION
from ..agent_runtime.diagnostics import get_diagnostics_store
from ..builders import ApprovalPending
from ..core.packets import OrchaPacket
from ..experts.local_chat import LocalChatExpert, extract_text_tool_calls
from ..experts.mock import load_mock_experts
from ..experts.ollama import list_ollama_models
from ..experts.registry import LocalModelRegistry
from ..graph.context import EventSubscriber, RunEvent
from ..graph.errors import GraphError
from ..graph.runtime import GraphRuntime
from ..graph.store import FileStore, MemoryStore, Store
from ..observability import configure_logging, get_logger
from ..orchestrator import Orchestrator
from ..result import RunResult
from ..settings import settings
from ..capabilities.base import ApprovalBroker
from ..capabilities.reasoning import (
    effort_for_level,
    model_supports_native_reasoning,
    pipeline_config,
)
from ..context import assemble_system_prompt, trim_history

API_VERSION = "0.4.0"
_TOOL_SEQ = [0]
_log = get_logger("orcha.api")

# Agent model calls get a larger token budget than the 1024-token chat
# default so multi-step tool loops and long final answers don't get silently
# truncated mid-sentence. The _post_with_retry truncation handler can grow
# this up to _MAX_GENERATION_TOKENS (8192), so starting at 8192 gives the
# agent a full budget on the first pass and avoids unnecessary retries.
_AGENT_MAX_TOKENS = 8192

# Agent-loop sampling: near-greedy. The default expert sampling (temp 0.7,
# repeat_penalty 1.15) is tuned for conversational variety, but a tool loop
# needs DETERMINISTIC protocol adherence — small models at high temperature
# skip write steps, narrate instead of calling tools, and hallucinate output.
_AGENT_SAMPLING = {
    "temperature": 0.35,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
}

# Text-protocol agent mode (ORCHA_AGENT_PROTOCOL=text) for small / local
# models: a free-text "TOOL:name:{json}" + "FINAL ANSWER:" format that tiny
# models follow far more reliably than the strict JSON-schema envelope. Lower
# temperature + a smaller token budget so they don't ramble or loop forever.
_AGENT_TEXT_MAX_TOKENS = 512
_AGENT_TEXT_SAMPLING = {
    "temperature": 0.25,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}


def _agent_protocol(base_url: str = "") -> str:
    """Agent tool protocol: ``"envelope"`` (strict JSON schema) or ``"text"``
    (free-text TOOL:/FINAL ANSWER: protocol for small models).

    Auto-detects: cloud base URLs (OpenRouter, OpenAI, etc.) fall back to
    ``"text"`` automatically because they don't support ``json_schema``
    response format.  Local endpoints (llama.cpp, Ollama) keep the stricter
    ``"envelope"`` protocol unless overridden via env var.
    """
    env = os.environ.get("ORCHA_AGENT_PROTOCOL", "").lower()
    if env in ("envelope", "text"):
        return env
    # Auto-detect: cloud base URLs -> text, local -> envelope
    if base_url and not any(
        h in base_url.lower()
        for h in ("localhost", "127.0.0.1", "0.0.0.0")
    ):
        return "text"
    return "envelope"


# Below this many billion parameters, the planner/executor switch to the
# compact prompts and tighter context windows in small_model_prompts.py.
# 8B matches the "CPU-friendly" ceiling the Model Library already uses for
# the same judgement call (see CPU_FRIENDLY_MAX_PARAMS_B in
# src/utils/modelSize.ts) — a model at or under this is the class those
# prompts were written for.
_SMALL_MODEL_MAX_PARAMS_B = 8.0
_PARAM_COUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b(?![a-z0-9])", re.I)

# Complexity threshold for small local models. Well below the 0.5 default
# so multi-step work actually reaches the decompose -> plan -> execute ->
# verify pipeline instead of being handed to the single-shot agent, which
# is where a small model silently drops half the request. 0.2 is set to
# catch a plain two-step task ("create A, then run it") while still
# leaving genuinely single-action asks ("read config.json", score 0.0) on
# the fast path.
#
# OFF BY DEFAULT, deliberately. Routing small models here is the correct
# destination, and the original blocker is gone: the pipeline no longer
# executes nothing. Measured on a 3B local model, the same two-file task
# went from ZERO files written to both files written with correct code,
# after three execution-layer fixes (the force-tool-call gate keying on
# attempted rather than succeeded calls; the stuck/replan branch
# pre-empting the recovery path; and a bounded corrective nudge on a
# premature "CANNOT COMPLETE", which one successful read_file is NOT
# evidence against). Plan-level routing was fixed after that: a
# dependency on a skipped step, a dangling dependency id, and a
# misparsed COMPLETE from the observer each used to end a run early
# while reporting success.
#
# It stays opt-in anyway, because the bar for flipping it is a task
# carried end to end - every step, through verification, with the files
# where they belong - and that has not been observed yet on a 3B model.
# Turning this on before then would route more traffic into a pipeline
# that finishes partially rather than one that at least finishes.
SMALL_MODEL_COMPLEXITY_THRESHOLD = 0.2
ENABLE_SMALL_MODEL_DECOMPOSITION = (
    os.environ.get("ORCHA_SMALL_MODEL_DECOMPOSITION", "").lower() in ("1", "true", "yes")
)


def _is_small_model(model_name: str) -> bool:
    """Whether the active model is small enough to need the compact
    agent prompts.

    GGUF/repo naming spells the parameter count out in the name itself
    ("qwen2.5-coder-3b-instruct-q4_k_m"), which is the only structured
    signal available here — there is no parameter-count field on an
    OpenAI-compatible endpoint. Takes the LARGEST such token, since a name
    can also carry a quantization or context number that happens to end in
    "b". Unknown/unparseable names return False, so a cloud or
    unrecognized model keeps the full-size prompts rather than being
    silently downgraded.
    """
    matches = [float(m) for m in _PARAM_COUNT_RE.findall(model_name or "")]
    sizes = [n for n in matches if 0 < n < 1000]
    if not sizes:
        return False
    return max(sizes) <= _SMALL_MODEL_MAX_PARAMS_B


# Strict tool envelope: when the backend supports structured output
# (llama.cpp response_format json_schema), agent completions are CONSTRAINED
# at generation time to a tiny JSON envelope — the model physically cannot
# emit malformed tool calls or prose instead of a call. This is what lets
# 1.5B-3B models run agentic loops reliably.
_AGENT_TOOL_ENVELOPE_SCHEMA = {
    # `arguments` is a JSON-encoded STRING, not a nested object: OpenAI's
    # strict json_schema validator requires every property to be listed in
    # `required` and every object (including nested ones) to declare
    # `additionalProperties: false` — which is incompatible with a
    # genuinely free-form "any tool's arguments" object. Encoding it as a
    # string sidesteps that recursive constraint while still round-tripping
    # arbitrary tool arguments (see _apply_agent_envelope, which decodes it).
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["tool", "final"]},
        "name": {"type": "string"},
        "arguments": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["action", "name", "arguments", "answer"],
    "additionalProperties": False,
}

_AGENT_ENVELOPE_INSTRUCTION = (
    "You MUST respond with a single JSON object and nothing else. All four "
    "keys are always required -- use empty strings for whichever don't apply.\n"
    "\n"
    "TOOL CALL FORMAT:\n"
    '{"action": "tool", "name": "<tool_name>", '
    '"arguments": "<JSON string of arguments>", "answer": ""}\n'
    "\n"
    "FINAL ANSWER FORMAT:\n"
    '{"action": "final", "name": "", "arguments": "", '
    '"answer": "<your answer to the user>"}\n'
    "\n"
    "ARGUMENTS must be a JSON-encoded string with escaped quotes. Examples:\n"
    '  read_file: {"path":"src/app.py"}\n'
    '  edit_file: {"path":"src/app.py","old_string":"foo","new_string":"bar"}\n'
    '  search_text: {"query":"useState"}\n'
    '  run_command: {"command":"npm test"}\n'
    '  git_status: {}\n'
    '  create_file: {"path":"src/new.ts","content":"export const x = 1;"}\n'
    "\n"
    "WORKFLOW: explore -> read -> edit -> build/test -> git add/commit\n"
    "\n"
    "RULES:\n"
    "- Always read a file BEFORE editing it (edit_file needs exact text match)\n"
    "- If a tool fails, read the error and try again with corrected arguments\n"
    "- Never claim to have done something unless the tool actually succeeded\n"
    "- No markdown, no prose outside the JSON object"
)

# Forced-tool variant: no "final" option in the enum at all, so the model
# is structurally incapable of describing/refusing instead of acting.
# Triggered by FORCE_TOOL_CALL_MARKER (orcha/nodes/task_executor.py) on the
# system prompt for a task-execution turn that hasn't made a real tool call
# yet. Confirmed live that the normal envelope's "action": "final" escape
# hatch doesn't stop this: Qwen2.5-Coder 3B/7B q4 repeatedly chose "final"
# with a prose description in "answer" over actually calling the tool, even
# after an explicit corrective retry telling it to call the tool. The JSON
# was always syntactically valid — the model was making a genuine choice
# the schema still allowed, so prompting alone can't close the gap.
# Removing "final" from the enum closes it at the grammar level instead.
_AGENT_TOOL_ENVELOPE_SCHEMA_FORCED = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["tool"]},
        "name": {"type": "string"},
        "arguments": {"type": "string"},
    },
    "required": ["action", "name", "arguments"],
    "additionalProperties": False,
}

_AGENT_ENVELOPE_INSTRUCTION_FORCED = (
    "You MUST call a tool right now — this response can ONLY be a real "
    "tool call, nothing else. Do not explain, describe, summarize, or "
    "say what you would do; there is no way to answer in text at this "
    "step. Respond with exactly this JSON object, with the real tool "
    "name and real arguments filled in:\n"
    '{"action": "tool", "name": "<tool_name>", '
    '"arguments": "<JSON string of arguments>"}\n'
    "\n"
    "ARGUMENTS must be a JSON-encoded string with escaped quotes. Examples:\n"
    '  write_file: {"path":"snake.html","content":"<!DOCTYPE html>..."}\n'
    '  read_file: {"path":"src/app.py"}\n'
    '  edit_file: {"path":"src/app.py","old_string":"foo","new_string":"bar"}\n'
    "\n"
    "Fill \"arguments\" with the COMPLETE content needed to make real "
    "progress — a partial or placeholder value is not acceptable."
)

# Streaming cannot re-run mid-stream like the non-streaming path's retry, so
# when the model's last streamed turn hit its token budget we transparently
# stream a bounded number of follow-up "continue" turns instead of leaving the
# user with a half-finished sentence.
_MAX_STREAM_CONTINUATIONS = 2

# Safety cap on concurrently-running background graph runs. Each run spawns
# its own agent loops and model calls; an unbounded number would exhaust the
# local machine's memory and GPU. Excess starts get a clean 429.
_MAX_CONCURRENT_BG_RUNS = 8

# Eviction cap for run_errors / cancelled_runs dicts. Entries older than this
# are evicted to prevent unbounded memory growth from long-lived sessions.
_MAX_RUN_HISTORY = 200


# ── Evicting dict (bounded LRU for run state) ───────────────────────────────

class _BoundedDict(OrderedDict):
    """An OrderedDict that evicts the oldest entry when it exceeds ``maxsize``."""

    def __init__(self, maxsize: int = _MAX_RUN_HISTORY):
        super().__init__()
        self._maxsize = maxsize

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self._maxsize:
            self.popitem(last=False)


# ── Orchestrator builder (single source of truth) ─────────────────────────────

async def _build_registry_from_ollama() -> Optional[LocalModelRegistry]:
    """
    Probe the local Ollama daemon and register every pulled model.

    Returns None if Ollama is not reachable or has no models — never raises.
    The caller then falls back to mock experts so the server always starts.

    This is an ``async`` function on purpose: it is always called from
    inside an event loop (the FastAPI lifespan or an async route handler),
    so it must ``await`` the discovery coroutine directly rather than call
    ``asyncio.run()``, which raises inside an already-running loop and
    would silently break discovery.
    """
    try:
        models = await list_ollama_models(settings.ollama_base_url)
    except Exception:
        return None
    if not models:
        return None

    registry = LocalModelRegistry()
    for m in models:
        if any(p in m.lower() for p in settings.ollama_skip_patterns):
            continue
        registry.add_ollama(m, base_url=settings.ollama_base_url)
    return registry


async def build_orchestrator() -> Orchestrator:
    """
    Discover experts from every configured source and combine them:
      1. A local OpenAI-compatible server (llama.cpp/LM Studio/vLLM/etc.)
         serving a model downloaded via Anvira, if LOCAL_SERVER_MODEL is set.
      2. Ollama auto-discovery, if the daemon is reachable.
    Falls back to mock experts only if neither source yields anything, so
    the server always starts cleanly even with no local models installed.

    This is the ONE place the startup/reload logic lives — ``/reload``
    delegates here too, so they can never drift apart.
    """
    registry = LocalModelRegistry()

    # Anvira-downloaded model served locally (llama.cpp server, LM Studio, etc.)
    # This is the "primary" local model — kept as the synthesizer by default.
    # A BYOK cloud connection (OpenRouter, Groq, OpenAI, ...) registers
    # through this exact same path — it's still just an OpenAI-compatible
    # base_url, only now with a real bearer api_key instead of "not-needed".
    if settings.local_server_model:
        registry.add_local_server(
            settings.local_server_model,
            base_url=settings.local_server_base_url,
            api_key=settings.local_server_api_key or "not-needed",
        )

    # Additional models running concurrently (multiple Anvira-activated
    # models, and/or BYOK cloud connections, at once).
    for extra in settings.local_server_extra_models:
        registry.add_local_server(
            extra["model"],
            base_url=extra["base_url"],
            api_key=extra.get("api_key") or "not-needed",
        )

    ollama_registry = await _build_registry_from_ollama()
    if ollama_registry is not None:
        for expert in ollama_registry.build().values():
            registry.add(expert)

    if registry.names():
        experts     = registry.build()
        synthesizer = registry.pick_synthesizer()
        source      = "local" if settings.local_server_model else "ollama"
        run_all     = settings.run_all_experts or len(settings.local_server_extra_models) > 0
    else:
        # No real model is available — start the server with mock experts
        # so the UI always works (users can browse, download, and configure
        # models even without one active). Chat will return simulated
        # responses until a real model is activated.
        _log.warning("No local model configured; using mock experts for demo mode")
        experts     = load_mock_experts()
        synthesizer = "mock_synthesizer"
        source      = "mock"
        run_all     = True

    orc = Orchestrator(
        experts=experts,
        synthesizer_expert=synthesizer,
        run_all_experts=run_all,
        max_cost=settings.max_cost,
        max_latency_s=settings.max_latency_s,
        max_iterations=settings.max_iterations,
        use_embeddings=settings.use_embeddings,
    )
    orc._source = source  # type: ignore[attr-defined]  # informational only
    _log.info("orchestrator.built source=%s experts=%d synthesizer=%s",
              source, len(experts), synthesizer)
    return orc


# ── App state (built lazily at startup, not import) ───────────────────────────

class AppState:
    """Mutable holder for the live orchestrator. Cheap to swap on /reload."""
    orc: Optional[Orchestrator] = None
    # ORCHA3 graph runtime state.
    store: Store = None  # type: ignore[assignment]
    active_runs: Dict[str, asyncio.Task] = {}  # run_id -> background task
    run_runtimes: Dict[str, "GraphRuntime"] = {}  # run_id -> runtime (live SSE events)
    run_errors: _BoundedDict = _BoundedDict()  # run_id -> error message (failed background runs)
    # run_id -> message for runs cancelled via POST /v1/run/{id}/cancel.
    # Kept separate from run_errors so get_run can report a distinct
    # status="cancelled" instead of a false "failed".
    cancelled_runs: _BoundedDict = _BoundedDict()
    approvals: Optional["ApprovalBroker"] = None  # shared interactive tool-approval broker
    # LangGraph-engine runs paused at the final-answer approval gate:
    # run_id -> {"query": ..., "request": ..., "runner": AgentLangGraphRunner}.
    # The runner must stay alive until the decision arrives so its thread
    # checkpointer can resume the paused run. When the server restarts, the
    # record is restored from disk with ``runner=None`` and the runner is
    # rebuilt from the persisted request at decision time (the thread itself
    # survives in the durable Sqlite checkpointer).
    pending_approvals: Dict[str, Dict[str, Any]] = {}
    # run_id -> serialized RunRequest — kept so the approval endpoint can
    # rebuild the runner even when the pending_approvals record was lost.
    run_requests: Dict[str, Dict[str, Any]] = {}
    langgraph_checkpointer: Any = None


_state = AppState()


# ── LangGraph durable state (shared checkpointer + pending approvals) ─────────

def _get_langgraph_checkpointer():
    """
    The server's shared durable LangGraph checkpointer.

    Compiled into every langgraph-engine runner built by this server, so
    threads (including paused approval threads) survive restarts. Path:
    ``ORCHA_LANGGRAPH_DB`` or ``~/.orcha/langgraph_threads.db``.
    """
    cp = getattr(_state, "langgraph_checkpointer", None)
    if cp is None:
        from ..integrations.langgraph import build_sqlite_checkpointer
        cp = build_sqlite_checkpointer()
        _state.langgraph_checkpointer = cp
    return cp

def _pending_file() -> Path:
    root = Path(os.environ.get("ORCHA_STATE_DIR") or (Path.home() / ".orcha"))
    root.mkdir(parents=True, exist_ok=True)
    return root / "pending_approvals.json"
def _read_pending_records() -> Dict[str, Dict[str, Any]]:
    try:
        with open(_pending_file(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}
def _persist_pending_approvals() -> None:
    """Best-effort write of the paused-approval records to disk."""
    try:
        records = {
            rid: {
                "query": rec.get("query", ""),
                "request": rec.get("request"),
                "ts": rec.get("ts"),
            }
            for rid, rec in _state.pending_approvals.items()
            if rec.get("request")
        }
        with open(_pending_file(), "w", encoding="utf-8") as f:
            json.dump(records, f)
    except Exception:  # noqa: BLE001 — persistence must never break a run
        _log.warning("pending_approval persist failed", exc_info=True)
def _restore_persisted_pending_approvals() -> None:
    """Restore paused-approval records after a restart (runner rebuilt on
    demand at decision time from the persisted request)."""
    for rid, rec in _read_pending_records().items():
        if rid not in _state.pending_approvals:
            _state.pending_approvals[rid] = {
                "query": rec.get("query", ""),
                "request": rec.get("request"),
                "runner": None,
                "ts": rec.get("ts"),
            }
            _log.info("restored pending approval run=%s from disk", rid[:8])
def _drop_persisted_pending(run_id: str) -> None:
    """Remove one record from the persisted pending-approval file."""
    records = _read_pending_records()
    if run_id not in records:
        return
    records.pop(run_id)
    try:
        with open(_pending_file(), "w", encoding="utf-8") as f:
            json.dump(records, f)
    except Exception:  # noqa: BLE001
        _log.warning("pending_approval persist failed", exc_info=True)
# Strong-reference holder for fire-and-forget startup tasks (see lifespan's
# connection pre-warm below) — asyncio.create_task()'s return value is only
# weakly referenced by the loop, so a task nothing else holds onto can be
# garbage-collected before it finishes.
_prewarm_tasks: set = set()
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the orchestrator when the server starts, not on import."""
    configure_logging()
    _state.orc = await build_orchestrator()
    # Initialize the durable store for graph runs.
    _state.store = FileStore()
    # Restore paused approval records (threads live in the durable
    # LangGraph checkpointer, which needs no startup step).
    _restore_persisted_pending_approvals()
    # Fire-and-forget: open the BYOK connection(s) now so the app's first
    # real chat doesn't also pay a cold TCP+TLS handshake on top of the
    # model's own response time. Not awaited — readiness must not wait on
    # a third-party endpoint, and a real request will open the connection
    # itself if this hasn't finished (or fails) by the time one arrives.
    # asyncio only holds a WEAK reference to a task's own return value, so
    # a task with nothing else referencing it is eligible for GC mid-flight
    # (documented asyncio footgun) — _prewarm_tasks keeps a strong one until
    # each finishes, then the done-callback drops it.
    for _expert in _state.orc.experts.values():
        if isinstance(_expert, LocalChatExpert):
            _task = asyncio.create_task(_expert.prewarm_connection())
            _prewarm_tasks.add(_task)
            _task.add_done_callback(_prewarm_tasks.discard)
    try:
        yield
    finally:
        # Cancel any in-flight background runs.
        for rid, task in list(_state.active_runs.items()):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        _state.orc = None
# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Orcha API",
    version=API_VERSION,
    description=(
        "Local-first multi-model orchestration. Runs all your local LLMs "
        "in parallel and synthesizes one refined answer."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Desktop launches spawn Orcha with ORCHA_API_TOKEN set and require the
# X-Orcha-Token header on every request, so a random webpage visited in
# the user's browser cannot drive the local agent (CORS "*" + zero auth
# would otherwise allow exactly that). Health stays open: it exposes no
# data and is used by the launcher's readiness probes.
@app.middleware("http")
async def _enforce_api_token(request: Request, call_next):
    # Preflights carry no custom headers by design: the browser asks
    # permission BEFORE sending X-Orcha-Token. Let OPTIONS through so the
    # CORS middleware (registered inside this one) can answer it with the
    # proper allow-origin headers — otherwise every browser fetch from the
    # app is rejected as a CORS failure before the token is even checked.
    if request.method == "OPTIONS":
        return await call_next(request)
    token = os.environ.get("ORCHA_API_TOKEN", "")
    if token and not request.url.path.endswith("/health"):
        provided = request.headers.get("x-orcha-token", "")
        if not secrets.compare_digest(provided, token):
            # This middleware runs OUTSIDE CORSMiddleware (registered after
            # it, so it wraps it), so returning a response directly here —
            # instead of via call_next — never passes through CORS header
            # injection. Without the header below, the browser can't read
            # this response at all: fetch() rejects with a generic "Failed
            # to fetch" instead of exposing the actual 401, so a real token
            # mismatch looks like Orcha is unreachable rather than telling
            # the caller (and whatever error message it shows the user)
            # what's actually wrong.
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"},
                headers={"Access-Control-Allow-Origin": request.headers.get("origin", "*")},
            )
    return await call_next(request)
@app.middleware("http")
async def _stamp_trace_id(request: Request, call_next):
    """Every request carries a trace id so error envelopes are linkable."""
    request.state.trace_id = uuid.uuid4().hex[:12]
    return await call_next(request)
def _orc() -> Orchestrator:
    """
    Return the live orchestrator. It is built during the lifespan startup
    event, so by the time any route runs it is guaranteed to be present.
    """
    if _state.orc is None:
        # Defensive: only reachable if a route somehow fires before startup
        # completed. Log a warning and return a mock-backed orchestrator
        # so the server stays responsive during the startup window.
        import logging
        logging.getLogger("orcha.api").warning(
            "_orc() called before orchestrator built — returning mock fallback"
        )
        orc = Orchestrator(
            experts=load_mock_experts(),
            synthesizer_expert="mock_synthesizer",
            run_all_experts=True,
            max_cost=settings.max_cost,
            max_latency_s=settings.max_latency_s,
            max_iterations=settings.max_iterations,
        )
        orc._source = "mock"  # type: ignore[attr-defined]
        _state.orc = orc
    return _state.orc
def _approvals() -> ApprovalBroker:
    """Return the shared interactive tool-approval broker (lazily created)."""
    if _state.approvals is None:
        _state.approvals = ApprovalBroker()
    return _state.approvals


def _mcp_manager() -> "McpManager":
    """Return the shared MCP server registry (lazily created + loaded)."""
    global _mcp_manager_instance
    if _mcp_manager_instance is None:
        from .mcp_manager import McpManager

        _mcp_manager_instance = McpManager()
        _mcp_manager_instance.load()
    return _mcp_manager_instance


_mcp_manager_instance: Optional["McpManager"] = None


def _skills_manager() -> "SkillsManager":
    """Return the shared SKILL.md registry (lazily created + scanned)."""
    global _skills_manager_instance
    if _skills_manager_instance is None:
        from .skills_manager import SkillsManager

        _skills_manager_instance = SkillsManager()
        try:
            _skills_manager_instance.reload()
        except Exception as exc:
            logging.getLogger("orcha.api").warning("skills scan failed: %s", exc)
    return _skills_manager_instance


_skills_manager_instance: Optional["SkillsManager"] = None
# The diagnostic session for the /v1/run currently executing on this task
# (set by run_graph for the duration of a run; the background task inherits
# it via asyncio's context copy). Used by model_fn/completion_fn to record
# each model call without threading a session through the graph nodes.
_RUN_DIAG: contextvars.ContextVar = contextvars.ContextVar(
    "orcha_run_diag", default=None
)
# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    """One prior turn in a conversation, sent alongside the current query."""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=20000)
class QueryRequest(BaseModel):
    query:           str = Field(..., min_length=1, max_length=20000)
    system_prompt:   Optional[str] = Field(
        default=None,
        max_length=8000,
        description=(
            "Optional persona/instructions prepended to every model call "
            "for this query. When set, it overrides each local model's "
            "system message so a user-configured persona actually shapes "
            "responses. Empty/None = use each model's own defaults."
        ),
    )
    messages:        Optional[List[ChatMessage]] = Field(
        default=None,
        max_length=100,
        description=(
            "Prior conversation turns (role/content) to give the model "
            "memory of the session. When provided, the current query is "
            "appended as the final user turn."
        ),
    )
    max_cost:        Optional[float] = Field(default=None, ge=0.0)
    max_iterations:  Optional[int]   = Field(default=None, ge=0)
    run_all_experts: Optional[bool]  = None
    reasoning:       Optional[str]   = Field(
        default=None,
        description=(
            "Reasoning level: fast | light | medium | high | max. Fast = one-pass "
            "direct answer with minimal planning/tools; Max = multi-agent planning, "
            "verification and synthesis. When set, the orchestrator tunes its "
            "iteration budget, planner width, quality threshold and (for streaming) "
            "generation budget to match."
        ),
    )
class QueryResponse(BaseModel):
    answer:        str
    confidence:    float
    quality_score: float
    synthesized:   bool
    agg_mode:      str
    iterations:    int
    cost:          float
    latency_s:     float
    contributors:  List[str]
    primary:       Optional[str]
    domains:       List[str]
    trace:         List[Dict[str, Any]]
class ExpertInfo(BaseModel):
    name:            str
    domain:          str
    description:     str
    version:         str
    is_synthesizer:  bool
class HealthResponse(BaseModel):
    status:          str
    version:         str
    source:          str
    synthesizer:     Optional[str]
    run_all_experts: bool
    experts:         List[str]
# ── Structured error envelope ─────────────────────────────────────────────────
class OrchaError(Exception):
    """Internal exception type carrying a normalized error payload."""
    def __init__(self, type_: str, message: str, status: int = 500):
        self.type = type_
        self.message = message
        self.status = status
        super().__init__(message)
@app.exception_handler(OrchaError)
async def _orcha_error_handler(request: Request, exc: OrchaError) -> JSONResponse:
    return _error_response(exc.type, exc.message, exc.status, request)
@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    messages = []
    for err in errors:
        loc = " -> ".join(str(l) for l in err.get("loc", []) if l != "body")
        msg = err.get("msg", "Invalid value")
        messages.append(f"{loc}: {msg}" if loc else msg)
    summary = "; ".join(messages) or "Validation failed"
    return _error_response("invalid_request", summary, 422, request)
@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    type_map = {
        404: "not_found",
        405: "method_not_allowed",
        422: "invalid_request",
        429: "rate_limited",
    }
    error_type = type_map.get(exc.status_code, "http_error")
    return _error_response(error_type, str(exc.detail), exc.status_code, request)
@app.exception_handler(Exception)
async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Log the full traceback internally; never leak it to the client.
    _log.exception("unhandled_error path=%s", request.url.path)
    return _error_response(
        "internal_error",
        "An unexpected error occurred while processing the request.",
        500,
        request,
    )
def _error_response(
    type_: str, message: str, status: int, request: Request
) -> JSONResponse:
    envelope = {
        "error": {
            "type":    type_,
            "message": message,
            "status":  status,
            "trace_id": getattr(request.state, "trace_id", None),
        }
    }
    return JSONResponse(status_code=status, content=envelope)
# ── Versioned routes (/v1) — the stable public API ────────────────────────────
@app.get("/v1/health", response_model=HealthResponse)
async def health():
    orc = _orc()
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        source=getattr(orc, "_source", "unknown"),
        synthesizer=orc.synthesizer_expert,
        run_all_experts=orc.run_all_experts,
        experts=list(orc.experts.keys()),
    )
@app.get("/v1/experts", response_model=List[ExpertInfo])
async def list_experts():
    return [ExpertInfo(**e) for e in _orc().list_experts()]
@app.post("/v1/query", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    base = _orc()
    try:
        # Resolve the reasoning level into concrete pipeline knobs. Invalid
        # levels raise a clean 400 instead of failing mid-run.
        try:
            pcfg = pipeline_config(req.reasoning)
        except ValueError as exc:
            raise OrchaError("invalid_reasoning", str(exc), 400)
        # History-aware fast path: when the caller supplies prior conversation,
        # run the designated synthesizer directly with the full message list —
        # the same single-expert semantics as the streaming endpoint — because
        # the multi-expert pipeline cannot thread a per-turn history into every
        # expert call. Falls back to the full pipeline when no messages.
        if req.messages:
            expert_name = base.synthesizer_expert or next(iter(base.experts), None)
            if expert_name is None or expert_name not in base.experts:
                raise OrchaError(
                    "no_expert_available",
                    "No expert is configured to handle this query.",
                    503,
                )
            expert = base.experts[expert_name]
            if not isinstance(expert, LocalChatExpert):
                raise OrchaError(
                    "no_local_model",
                    "Conversation history requires an active local model. "
                    "Activate a model in Anvira first.",
                    400,
                )
            wants_prompt = bool(req.system_prompt and req.system_prompt.strip())
            max_tokens = expert.max_tokens
            temperature = expert.temperature
            extra_body = dict(expert.extra_body)
            if req.reasoning and pcfg:
                max_tokens = max(32, int(round(expert.max_tokens * pcfg["output_scale"])))
                temperature = max(0.0, min(1.5, expert.temperature + pcfg["temperature_delta"]))
                if model_supports_native_reasoning(expert.model):
                    extra_body["reasoning_effort"] = effort_for_level(req.reasoning)
                if "frequency_penalty" not in extra_body:
                    extra_body["frequency_penalty"] = pcfg.get("frequency_penalty", 0.3)
                if "presence_penalty" not in extra_body:
                    extra_body["presence_penalty"] = pcfg.get("presence_penalty", 0.15)
            caller = LocalChatExpert(
                model=expert.model,
                base_url=expert.base_url,
                system_prompt=req.system_prompt.strip() if wants_prompt else expert.system_prompt,
                temperature=temperature,
                top_p=expert.top_p,
                top_k=expert.top_k,
                repeat_penalty=expert.repeat_penalty,
                max_tokens=max_tokens,
                stop=expert.stop,
                api_key=expert.api_key,
                timeout_s=expert.timeout_s,
                extra_body=extra_body,
            )
            history = trim_history([m.model_dump() for m in req.messages])
            output = await caller.execute(req.query, messages=history)
            return QueryResponse(
                answer=output.answer,
                confidence=output.confidence,
                quality_score=output.confidence,
                synthesized=False,
                agg_mode="direct_chat",
                iterations=1,
                cost=0.0,
                latency_s=output.latency_s,
                contributors=[expert_name],
                primary=expert_name,
                domains=[],
                trace=[],
            )
        # When a system prompt is supplied (Anvira's Settings -> system
        # prompt), apply it to every LocalChatExpert for this request only
        # by building a throwaway experts dict. This avoids mutating the
        # shared live pool, so the persona can't leak into concurrent
        # calls or the streaming endpoint. Construction is cheap (config
        # assignment only; no network connection until execute() runs).
        request_experts = base.experts
        if req.system_prompt and req.system_prompt.strip():
            prompt = req.system_prompt.strip()
            request_experts = {
                name: (
                    LocalChatExpert(
                        model=e.model,
                        base_url=e.base_url,
                        system_prompt=prompt,
                        temperature=e.temperature,
                        top_p=e.top_p,
                        top_k=e.top_k,
                        repeat_penalty=e.repeat_penalty,
                        max_tokens=e.max_tokens,
                        stop=e.stop,
                        api_key=e.api_key,
                        timeout_s=e.timeout_s,
                    )
                    if isinstance(e, LocalChatExpert)
                    else e
                )
                for name, e in base.experts.items()
            }
        # Per-request overrides use explicit None checks — a client asking
        # for max_cost=0 (local-only, spend nothing) or max_iterations=0
        # must NOT be silently coerced to the settings default by a
        # truthy/falsy `or` fallback. Reasoning levels contribute their own
        # defaults when the client didn't pin the value explicitly.
        has_override = any(
            v is not None
            for v in (req.max_cost, req.max_iterations, req.run_all_experts, req.reasoning)
        ) or request_experts is not base.experts
        if has_override:
            max_iterations = req.max_iterations
            if max_iterations is None:
                max_iterations = (
                    pcfg["max_iterations"]
                    if pcfg["max_iterations"] is not None
                    else settings.max_iterations
                )
            run_all = req.run_all_experts
            if run_all is None:
                run_all = (
                    pcfg["run_all_experts"]
                    if pcfg["run_all_experts"] is not None
                    else base.run_all_experts
                )
            orc = Orchestrator(
                experts=request_experts,
                synthesizer_expert=base.synthesizer_expert,
                run_all_experts=run_all,
                max_cost=(
                    req.max_cost
                    if req.max_cost is not None
                    else settings.max_cost
                ),
                max_latency_s=settings.max_latency_s,
                max_iterations=max_iterations,
                reasoning=req.reasoning,
            )
        else:
            orc = base
        result = await orc.run_async(req.query)
        return QueryResponse(**result.to_dict())
    except OrchaError:
        raise
    except Exception as exc:
        _log.exception("query_failed query=%r", req.query[:80])
        raise OrchaError("query_failed", str(exc), 500)
@app.post("/v1/query/stream")
async def run_query_stream(req: QueryRequest):
    """
    Token-by-token streaming for interactive chat.
    Streams directly from the orchestrator's designated synthesizer
    expert — the single largest/most capable local model — rather than
    running the full multi-expert pipeline. Synthesis fundamentally
    can't be streamed (it needs every expert's complete output before it
    can combine them), so this is a genuinely different, faster mode:
    direct generation for a responsive chat feel. Use /v1/query when you
    want the full multi-expert synthesis instead, at the cost of only
    seeing the result once everything finishes.
    Emits SSE events:
        data: {"type": "chunk", "content": "..."}      (repeated)
        data: {"type": "done", "model": "...", "latency_s": ...}
        data: {"type": "error", "message": "..."}
    """
    orc = _orc()
    expert_name = orc.synthesizer_expert or next(iter(orc.experts), None)
    if expert_name is None or expert_name not in orc.experts:
        raise OrchaError("no_expert_available", "No expert is configured to handle this query.", 503)
    expert = orc.experts[expert_name]
    msgs = trim_history([m.model_dump() for m in req.messages]) if req.messages else None
    async def _stream(exp, name, query):
        t0 = time.perf_counter()
        try:
            # A truncated streamed turn can't be re-run mid-stream, so when the
            # model hits its token budget we stream bounded follow-up turns that
            # ask it to continue exactly where it stopped — making streaming
            # answers complete the same way the non-streaming retry does.
            continue_turns = 0
            partial = ""
            history: List[Dict[str, Any]] = list(msgs or [])
            truncated_final = False
            while True:
                if isinstance(exp, LocalChatExpert):
                    stream = exp.execute_stream(query, messages=history or None)
                else:
                    stream = exp.execute_stream(query)
                async for piece in stream:
                    partial += piece
                    event = {"type": "chunk", "content": piece}
                    yield f"data: {json.dumps(event)}\n\n"
                truncated = getattr(exp, "last_finish_reason", "stop") == "length"
                if (
                    truncated
                    and partial
                    and continue_turns < _MAX_STREAM_CONTINUATIONS
                    and isinstance(exp, LocalChatExpert)
                ):
                    continue_turns += 1
                    history = history + [
                        {"role": "assistant", "content": partial},
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was cut off by the token "
                                "limit. Continue exactly from where you stopped, "
                                "without repeating earlier text."
                            ),
                        },
                    ]
                    query = "continue"
                    continue
                truncated_final = truncated
                break
            done_event = {
                "type": "done",
                "model": name,
                "latency_s": time.perf_counter() - t0,
                "truncated": truncated_final,
            }
            yield f"data: {json.dumps(done_event)}\n\n"
        except Exception as exc:
            _log.exception("stream_query_failed query=%r", query[:80])
            error_event = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(error_event)}\n\n"
    # When the caller passes a system prompt (Anvira's Settings -> system
    # prompt field) or a reasoning level, apply it per-request rather than
    # mutating the shared live expert's state (which would leak the persona
    # into unrelated multi-expert synthesis calls). For LocalChatExpert this
    # is cheap: construction is just config assignment, no connection is
    # opened until execute_stream() runs, and we reuse the exact same model +
    # base_url the active expert already has.
    wants_prompt = bool(req.system_prompt and req.system_prompt.strip())
    wants_reasoning = bool(req.reasoning)
    wants_messages = bool(req.messages)
    if isinstance(expert, LocalChatExpert) and (wants_prompt or wants_reasoning or wants_messages):
        try:
            pcfg = pipeline_config(req.reasoning) if wants_reasoning else None
        except ValueError as exc:
            raise OrchaError("invalid_reasoning", str(exc), 400)
        # Emulation knobs: scale the output-token budget and nudge sampling
        # so higher levels spend more compute per reply. For models with
        # native chain-of-thought, hand the level to the server as a
        # `reasoning_effort` field instead.
        max_tokens = expert.max_tokens
        temperature = expert.temperature
        extra_body = dict(expert.extra_body)
        if pcfg:
            max_tokens = max(32, int(round(expert.max_tokens * pcfg["output_scale"])))
            temperature = max(0.0, min(1.5, expert.temperature + pcfg["temperature_delta"]))
            if model_supports_native_reasoning(expert.model):
                extra_body["reasoning_effort"] = effort_for_level(req.reasoning)
            # Anti-repetition penalties — critical for small models.
            # Fast/Light levels have higher temperature, so they need
            # stronger penalties to prevent degenerate loops.
            if "frequency_penalty" not in extra_body:
                extra_body["frequency_penalty"] = pcfg.get("frequency_penalty", 0.3)
            if "presence_penalty" not in extra_body:
                extra_body["presence_penalty"] = pcfg.get("presence_penalty", 0.15)
        streaming_expert = LocalChatExpert(
            model=expert.model,
            base_url=expert.base_url,
            system_prompt=req.system_prompt.strip() if wants_prompt else expert.system_prompt,
            temperature=temperature,
            top_p=expert.top_p,
            top_k=expert.top_k,
            repeat_penalty=expert.repeat_penalty,
            max_tokens=max_tokens,
            stop=expert.stop,
            api_key=expert.api_key,
            timeout_s=expert.timeout_s,
            extra_body=extra_body,
        )
        return StreamingResponse(
            _stream(streaming_expert, expert_name, req.query),
            media_type="text/event-stream",
        )
    return StreamingResponse(_stream(expert, expert_name, req.query), media_type="text/event-stream")
@app.post("/v1/reload")
async def reload_experts():
    """Re-discover Ollama models. Call after `ollama pull <model>`."""
    _state.orc = await build_orchestrator()
    orc = _state.orc
    assert orc is not None
    return {
        "reloaded": True,
        "source":   getattr(orc, "_source", "unknown"),
        "experts":  list(orc.experts.keys()),
    }
class LocalModelRequest(BaseModel):
    model:    str = Field(..., min_length=1)
    base_url: str = Field(default="http://localhost:8080/v1")
    # BYOK support: any OpenAI-compatible cloud provider (OpenRouter, Groq,
    # OpenAI, Together, DeepSeek, ...) registers through this same request
    # shape — api_key is the bearer token, label is a display-only name
    # ("OpenRouter — Llama 3.3 70B") shown in place of the raw model id.
    # Both optional so the existing local-llama-server callers are unaffected.
    api_key:  Optional[str] = None
    label:    Optional[str] = None
@app.post("/v1/local-model")
async def set_local_model(payload: LocalModelRequest):
    """
    Point Orcha at a model served locally (e.g. by Anvira's llama-server)
    or a BYOK cloud connection, then rebuild the orchestrator so it takes
    effect immediately. Anvira runs as a separate process from Orcha, so
    it can't set LOCAL_SERVER_MODEL in Orcha's environment after the fact
    — this endpoint is the runtime equivalent, mutating the same settings
    singleton that build_orchestrator() reads from.
    Note: this REPLACES the primary local model. To run multiple models
    at once, use POST /v1/local-models/add for the additional ones
    instead of calling this repeatedly.
    """
    settings.local_server_model = payload.model
    settings.local_server_base_url = payload.base_url
    settings.local_server_api_key = payload.api_key or ""
    _state.orc = await build_orchestrator()
    orc = _state.orc
    assert orc is not None
    return {
        "reloaded": True,
        "source":   getattr(orc, "_source", "unknown"),
        "experts":  list(orc.experts.keys()),
    }
@app.post("/v1/local-models/add")
async def add_local_model(payload: LocalModelRequest):
    """
    Register an ADDITIONAL model running concurrently, without displacing
    the primary local_server_model or any other already-registered model.
    This is what enables running multiple Anvira-activated models (local
    and/or BYOK cloud connections) at once for multi-expert synthesis.
    A local model must already be reachable at its own base_url (its own
    llama-server instance/port); a cloud connection just needs a valid
    api_key — this endpoint only tells Orcha about it, it doesn't start
    anything.
    """
    already_registered = (
        payload.model == settings.local_server_model
        or any(m["model"] == payload.model for m in settings.local_server_extra_models)
    )
    if not already_registered:
        settings.local_server_extra_models.append({
            "model": payload.model,
            "base_url": payload.base_url,
            "api_key": payload.api_key or "",
            "label": payload.label,
        })
    _state.orc = await build_orchestrator()
    orc = _state.orc
    assert orc is not None
    return {
        "reloaded": True,
        "source":   getattr(orc, "_source", "unknown"),
        "experts":  list(orc.experts.keys()),
    }
@app.delete("/v1/local-models/{model_name:path}")
async def remove_local_model(model_name: str):
    """
    Stop using a specific local model or BYOK connection (e.g. after it's
    been ejected/removed in Anvira). Rebuilds the orchestrator so the
    change takes effect immediately. If the removed model was the primary
    local_server_model, the next remaining extra model (if any) is
    promoted to primary so the orchestrator doesn't lose its designated
    synthesizer.
    """
    removed = False
    if settings.local_server_model == model_name:
        settings.local_server_model = ""
        settings.local_server_base_url = "http://localhost:8080/v1"
        settings.local_server_api_key = ""
        removed = True
        # Promote the next extra model to primary, if one exists, so
        # there's still a synthesizer candidate.
        if settings.local_server_extra_models:
            promoted = settings.local_server_extra_models.pop(0)
            settings.local_server_model = promoted["model"]
            settings.local_server_base_url = promoted["base_url"]
            settings.local_server_api_key = promoted.get("api_key") or ""
    before_count = len(settings.local_server_extra_models)
    settings.local_server_extra_models = [
        m for m in settings.local_server_extra_models if m["model"] != model_name
    ]
    if len(settings.local_server_extra_models) < before_count:
        removed = True
    if not removed:
        raise OrchaError("model_not_found", f"'{model_name}' is not currently registered.", 404)
    _state.orc = await build_orchestrator()
    orc = _state.orc
    assert orc is not None
    return {
        "reloaded": True,
        "source":   getattr(orc, "_source", "unknown"),
        "experts":  list(orc.experts.keys()),
    }
@app.get("/v1/local-models")
async def list_local_models():
    """Lists every locally-registered model currently in the orchestrator.
    Never returns api_key — callers only need to know a connection exists
    and how to label it, not the secret itself."""
    models = []
    if settings.local_server_model:
        models.append({
            "model": settings.local_server_model,
            "base_url": settings.local_server_base_url,
            "primary": True,
            "isRemote": bool(settings.local_server_api_key),
        })
    for extra in settings.local_server_extra_models:
        models.append({
            "model": extra["model"],
            "base_url": extra["base_url"],
            "label": extra.get("label"),
            "primary": False,
            "isRemote": bool(extra.get("api_key")),
        })
    return {"models": models}
@app.get("/v1/performance")
async def performance():
    """Per-expert selector performance history."""
    return _orc().selector_performance()
# ── ORCHA3 graph endpoints ────────────────────────────────────────────────────
def _build_default_graph_for_orc(orc: Orchestrator):
    """Build a default graph using the live orchestrator's expert pool."""
    from ..builders.default import build_default_graph, DefaultGraphConfig
    return build_default_graph(
        DefaultGraphConfig(
            experts=dict(orc.experts),
            synthesizer_expert=orc.synthesizer_expert,
            run_all_experts=orc.run_all_experts,
            max_cost=orc.max_cost,
            max_latency_s=orc.max_latency_s,
            max_iterations=orc.max_iterations,
            use_embeddings=getattr(orc.decomposer, "use_embeddings", False),
        )
    )
class RunRequest(BaseModel):
    """Request body for POST /v1/run."""
    query: str = Field(..., min_length=1, max_length=20000)
    graph: str = Field(
        default="default",
        description="Graph name: 'default', 'research', or 'multi_agent'.",
    )
    mode: Optional[str] = Field(default=None)
    access_mode: Optional[str] = Field(default=None)
    allow_tools: Optional[bool] = Field(default=None)
    require_approval: Optional[bool] = Field(default=None)
    system_prompt: Optional[str] = Field(
        default=None,
        description=(
            "For graph='multi_agent': overrides the agent's system prompt "
            "(use {task} and {context} as placeholders), letting a chosen "
            "agent persona's instructions actually apply to how agents "
            "reason, not just what task they're given. Ignored for other "
            "graph types."
        ),
    )
    max_cost: Optional[float] = Field(default=None, ge=0.0)
    max_iterations: Optional[int] = Field(default=None, ge=0)
    workspace_roots: Optional[List[str]] = Field(
        default=None,
        description=(
            "Absolute directories the agent's workspace tools are allowed to "
            "read/write/edit. When empty, no file tools are exposed."
        ),
    )
    tools: Optional[List[str]] = Field(
        default=None,
        description="Whitelist of tool names to expose (None = all workspace tools).",
    )
    allow_rules: Optional[List[str]] = Field(
        default=None,
        description=(
            "Permission rule strings auto-ALLOWED for this run, e.g. "
            "'run_command(git *)' — persisted client-side from previous "
            "\"approve + don't ask again\" decisions."
        ),
    )
    deny_rules: Optional[List[str]] = Field(
        default=None,
        description="Permission rule strings always DENIED for this run.",
    )
    ask_rules: Optional[List[str]] = Field(
        default=None,
        description=(
            "Permission rule strings that force an interactive approval "
            "for this run even when the access mode would allow them."
        ),
    )
    capabilities: Optional[List[str]] = Field(
        default=None,
        description=(
            "Declared capability names for the agent (filesystem, workspace, "
            "search, web, terminal, git, diagnostics, code_intelligence). When set, "
            "the agent's tools are built from these capabilities via the "
            "CapabilityRegistry. When empty/None, legacy workspace tools are "
            "used for backward compatibility."
        ),
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Reasoning level for graph='multi_agent': fast | light | medium | high | max.",
    )
    messages: Optional[List[ChatMessage]] = Field(
        default=None,
        max_length=100,
        description=(
            "Prior conversation turns (role/content) to seed the agent's "
            "memory for this run."
        ),
    )
    prompt_parts: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Named system-prompt parts ({name, text, priority}) to assemble "
            "the agent's system prompt via priority-aware truncation, instead "
            "of one pre-concatenated string. Higher priority survives budget "
            "pressure; lower-priority parts are dropped/truncated first."
        ),
    )
    stream: bool = Field(
        default=False,
        description="If True, run in background and stream via SSE.",
    )
class ApprovalDecisionRequest(BaseModel):
    """Body for POST /v1/run/{run_id}/approvals/{approval_id}."""
    approved: bool = Field(
        ...,
        description="True to approve the pending tool call, False to reject it.",
    )
    update: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional structured permission update recorded with this "
            "decision — e.g. {'kind': 'add_rule', 'rule': 'run_command(git "
            "*)', 'source': 'session'} for a 'yes, and don't ask again' "
            "answer. Retrieved via GET /v1/rules/updates for persistence."
        ),
    )
class AgentApprovalDecisionRequest(BaseModel):
    """Body for POST /v1/run/{run_id}/approval (LangGraph final-answer gate)."""
    approved: bool = Field(
        ...,
        description="True to accept the proposed final answer, False to reject it.",
    )
    note: Optional[str] = Field(
        default=None,
        description="Optional reason surfaced to the agent when rejecting.",
    )
class RunResponse(BaseModel):
    """Response for POST /v1/run (non-streaming) or run status."""
    run_id: str
    graph_name: str
    answer: str = ""
    confidence: float = 0.0
    quality_score: float = 0.0
    synthesized: bool = False
    agg_mode: str = "unknown"
    iterations: int = 0
    cost: float = 0.0
    latency_s: float = 0.0
    contributors: List[str] = []
    primary: Optional[str] = None
    domains: List[str] = []
    status: str = "completed"
    error: Optional[str] = None
    trace: List[Dict[str, Any]] = []
    agent_steps: List[Dict[str, Any]] = []
    agent_tool_calls: List[Dict[str, Any]] = []
    agent_completed: bool = False
    approval: Optional[Dict[str, Any]] = None
def _run_result_to_response(result: RunResult, run_id: str, status: str = "completed") -> RunResponse:
    d = result.to_dict()
    return RunResponse(
        run_id=run_id,
        graph_name=d.get("graph_name", ""),
        answer=d.get("answer", ""),
        confidence=d.get("confidence", 0.0),
        quality_score=d.get("quality_score", 0.0),
        synthesized=d.get("synthesized", False),
        agg_mode=d.get("agg_mode", "unknown"),
        iterations=d.get("iterations", 0),
        cost=d.get("cost", 0.0),
        latency_s=d.get("latency_s", 0.0),
        contributors=d.get("contributors", []),
        primary=d.get("primary"),
        domains=d.get("domains", []),
        status=status,
        trace=d.get("trace", []),
        agent_steps=d.get("agent_steps", []),
        agent_tool_calls=d.get("agent_tool_calls", []),
        agent_completed=d.get("agent_completed", False),
    )
def _build_agent_model_fn(orc: Orchestrator):
    """
    Builds the model_fn callable that AgentNode/build_multi_agent_graph
    need to make real model calls, instead of running in dry-run mode
    (which produces fake synthetic "[Dry run iteration N]..." text with
    zero actual inference).
    Reuses the connection details (model name + base_url) of whichever
    local expert is currently the designated synthesizer — the same
    model already active for regular chat — rather than requiring a
    separate agent-specific model configuration. A fresh LocalChatExpert
    is constructed per call so each agent iteration can carry its own
    system prompt; this is cheap since construction is just config
    assignment, no network connection is opened until execute() runs.
    """
    synthesizer_name = orc.synthesizer_expert
    reference_expert = orc.experts.get(synthesizer_name) if synthesizer_name else None
    if not isinstance(reference_expert, LocalChatExpert):
        # No local model is currently active — agents have nothing real
        # to call. Return None so callers can detect this and respond
        # with a clear error instead of silently falling back to
        # dry-run/mock text that looks like a real answer but isn't.
        return None
    model = reference_expert.model
    base_url = reference_expert.base_url
    api_key = getattr(reference_expert, "api_key", None)
    async def model_fn(prompt: str, system: str) -> str:
        diag = _RUN_DIAG.get()
        if diag is not None:
            diag.record(
                "model", "call",
                model=model, base_url=base_url, backend="openai_compat",
                prompt_chars=len(prompt), prompt_preview=prompt[:600],
                system_chars=len(system), system_preview=system[:3000],
            )
        caller = LocalChatExpert(
            model=model,
            base_url=base_url,
            api_key=api_key,
            system_prompt=system,
            max_tokens=(_AGENT_TEXT_MAX_TOKENS if _agent_protocol(base_url) == "text"
                        else _AGENT_MAX_TOKENS),
            **(_AGENT_TEXT_SAMPLING if _agent_protocol(base_url) == "text"
                else _AGENT_SAMPLING),
        )
        try:
            output = await caller.execute(prompt)
        except Exception as exc:
            if diag is not None:
                diag.record_exception("model", exc, message=str(exc))
            raise
        if diag is not None:
            diag.record(
                "model", "response",
                model=model, content_preview=(output.answer or "")[:600],
            )
        return output.answer
    return model_fn

def _apply_agent_envelope(result: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Parse the strict-tools JSON envelope (``action: tool`` / ``action: final``)
    returned by the model under constrained mode into a native assistant message
    with ``tool_calls`` or plain ``content``. Shared by the streaming and
    non-streaming agent completion paths so both behave identically.
    """
    if not isinstance(result, dict):
        return result
    raw_content = str(result.get("content") or "").strip()
    if not raw_content:
        return result
    envelope = None
    try:
        envelope = json.loads(raw_content)
    except json.JSONDecodeError:
        bm = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if bm:
            try:
                envelope = json.loads(bm.group(0))
            except json.JSONDecodeError:
                envelope = None
    if isinstance(envelope, dict) and envelope.get("action") == "tool":
        tname = str(envelope.get("name") or "")
        targs = envelope.get("arguments")
        # `arguments` is normally a JSON-encoded string (see the schema
        # above) but tolerate a raw object too, in case a backend that
        # doesn't enforce the schema strictly hands one back anyway.
        # strict=False: a model writing a FILE into a JSON string argument emits
        # real newlines and tabs inside it, which strict JSON rejects as invalid
        # control characters. Measured live on a 3B model: create_file arrived
        # with a correct path and 285 characters of correct Python, the strict
        # parse raised, the except branch swallowed it to {}, and the model was
        # told "you called create_file with no arguments at all" - a false
        # accusation it could not act on, repeated until the step exhausted its
        # iterations. This was the single largest source of failed steps.
        if isinstance(targs, str):
            try:
                targs = json.loads(targs, strict=False) if targs.strip() else {}
            except json.JSONDecodeError:
                targs = {}
        if not isinstance(targs, dict):
            targs = {}
        if tname:
            _TOOL_SEQ[0] += 1
            return {
                "role": "assistant",
                "content": "",
                "finish_reason": result.get("finish_reason", "stop"),
                "tool_calls": [{
                    "id": f"call_env_{_TOOL_SEQ[0]}",
                    "type": "function",
                    "function": {
                        "name": tname,
                        "arguments": json.dumps(targs),
                    },
                }],
            }
    if isinstance(envelope, dict) and envelope.get("action") == "final":
        return {
            "role": "assistant",
            "content": str(envelope.get("answer") or ""),
            "finish_reason": result.get("finish_reason", "stop"),
        }
    # Envelope mode failed to parse a structured {action, name, arguments}
    # response — some models (MiniMax observed) ignore the schema
    # instruction entirely and emit their own native tool-call syntax
    # instead (e.g. Anthropic-style <invoke name="..."><parameter ...>
    # XML, optionally wrapped in a namespaced <minimax:tool_call> tag).
    # Recover it the same way the legacy text-tool-call path already does
    # for other backends, rather than leaking the raw markup to the user
    # as if it were the model's actual answer.
    if envelope is None:
        recovered = extract_text_tool_calls(raw_content)
        if recovered:
            return {
                "role": "assistant",
                "content": "",
                "finish_reason": result.get("finish_reason", "stop"),
                "tool_calls": recovered,
            }
    # Unparseable under strict mode: surface raw text so the loop's existing
    # recovery parser gets a chance before giving up.
    result["content"] = raw_content
    return result


async def _emit_text_delta_typed(on_text_delta, text: str, delay: float = 0.012):
    """Emit ``text`` word-by-word so the UI can render a token-by-token
    typing animation. Used for answers that are only known once an envelope
    (or fallback) completion fully resolves, not streamed live."""
    if not on_text_delta or not text:
        return
    parts = text.split(" ")
    for i, part in enumerate(parts):
        chunk = part + (" " if i < len(parts) - 1 else "")
        await on_text_delta(chunk)
        await asyncio.sleep(delay)

def _build_agent_completion_fn(orc: Orchestrator):
    """
    Builds the native function-calling completion callable for agents:
    async (messages, system, tools) -> dict.
    Mirrors _build_agent_model_fn but returns the full assistant message
    (including ``tool_calls``) so AgentNode can run the native tool loop.
    """
    synthesizer_name = orc.synthesizer_expert
    reference_expert = orc.experts.get(synthesizer_name) if synthesizer_name else None
    if not isinstance(reference_expert, LocalChatExpert):
        return None
    model = reference_expert.model
    base_url = reference_expert.base_url
    api_key = getattr(reference_expert, "api_key", None)
    from ..nodes.task_executor import FORCE_TOOL_CALL_MARKER
    async def completion_fn(messages, system, tools=None):
        force_tool = isinstance(system, str) and system.endswith(FORCE_TOOL_CALL_MARKER)
        if force_tool:
            system = system[: -len(FORCE_TOOL_CALL_MARKER)]
        diag = _RUN_DIAG.get()
        if diag is not None:
            last_user = next(
                (m for m in reversed(messages)
                 if isinstance(m, dict) and m.get("role") == "user"),
                None,
            )
            diag.record(
                "model", "call",
                model=model, base_url=base_url, backend="openai_compat",
                system_chars=len(system or ""),
                system_preview=(system or "")[:3000],
                prompt_chars=sum(
                    len(str(m.get("content") or "")) for m in messages
                ),
                prompt_preview=(
                    str(last_user.get("content") or "")[:600]
                    if last_user else ""
                ),
                tools=len(tools or []),
            )
        caller = LocalChatExpert(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_tokens=_AGENT_MAX_TOKENS,
            **_AGENT_SAMPLING,
        )
        # Structured-output constraint: only when tools exist AND the
        # endpoint advertises llama.cpp-style json_schema support is
        # unknown here — attempt constrained mode and fall back cleanly.
        use_envelope = bool(tools) and os.environ.get("ORCHA_STRICT_TOOLS", "1") != "0"
        call_kwargs = {}
        eff_system = system
        if use_envelope:
            schema = _AGENT_TOOL_ENVELOPE_SCHEMA_FORCED if force_tool else _AGENT_TOOL_ENVELOPE_SCHEMA
            instruction = _AGENT_ENVELOPE_INSTRUCTION_FORCED if force_tool else _AGENT_ENVELOPE_INSTRUCTION
            call_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_tool_envelope",
                    "strict": True,
                    "schema": schema,
                },
            }
            eff_system = (system or "") + "\n\n" + instruction
        try:
            result = await caller.chat_completion(
                messages, system=eff_system, tools=None if use_envelope else tools,
                **call_kwargs,
            )
        except Exception as exc:
            if diag is not None:
                diag.record_exception("model", exc, message=str(exc))
            # Constrained request rejected by the backend (e.g. OpenRouter
            # free models don't support json_schema response_format, or the
            # endpoint doesn't support structured tool output). Fall back
            # through progressive degradation:
            #   1. native tools (no envelope) — most capable fallback
            #   2. plain text (no tools at all) — last resort
            # Never re-raise on the first failure when use_envelope was true:
            # the caller expects the fallback to be attempted.
            if use_envelope:
                try:
                    result = await caller.chat_completion(
                        messages, system=system, tools=tools
                    )
                except Exception as exc2:
                    if diag is not None:
                        diag.record_exception("model", exc2, message=str(exc2))
                    # Both envelope AND native tools failed — try plain text
                    # as the absolute last resort so the agent still produces
                    # an answer instead of a raw error.
                    try:
                        result = await caller.chat_completion(
                            messages, system=system, tools=None
                        )
                    except Exception as exc3:
                        if diag is not None:
                            diag.record_exception("model", exc3, message=str(exc3))
                        raise
            else:
                raise
        if use_envelope and isinstance(result, dict):
            result = _apply_agent_envelope(result, model)
        if diag is not None:
            content = ""
            finish_reason = None
            if isinstance(result, dict):
                content = str(result.get("content") or "")
                finish_reason = result.get("finish_reason")
            diag.record(
                "model", "response",
                model=model, content_preview=content[:600],
                finish_reason=finish_reason,
            )
        return result

    async def completion_fn_streaming(messages, system, tools=None, on_text_delta=None):
        """Streaming variant: yields text deltas via on_text_delta callback
        while accumulating the full response for tool-call parsing."""
        import httpx as _httpx
        force_tool = isinstance(system, str) and system.endswith(FORCE_TOOL_CALL_MARKER)
        if force_tool:
            system = system[: -len(FORCE_TOOL_CALL_MARKER)]
        diag = _RUN_DIAG.get()
        if diag is not None:
            last_user = next(
                (m for m in reversed(messages)
                 if isinstance(m, dict) and m.get("role") == "user"),
                None,
            )
            diag.record(
                "model", "stream_call",
                model=model, base_url=base_url, backend="openai_compat",
                system_chars=len(system or ""),
                prompt_chars=sum(
                    len(str(m.get("content") or "")) for m in messages
                ),
                tools=len(tools or []),
            )
        use_envelope = bool(tools) and os.environ.get("ORCHA_STRICT_TOOLS", "1") != "0"
        call_kwargs = {}
        eff_system = system
        if use_envelope:
            schema = _AGENT_TOOL_ENVELOPE_SCHEMA_FORCED if force_tool else _AGENT_TOOL_ENVELOPE_SCHEMA
            instruction = _AGENT_ENVELOPE_INSTRUCTION_FORCED if force_tool else _AGENT_ENVELOPE_INSTRUCTION
            call_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_tool_envelope",
                    "strict": True,
                    "schema": schema,
                },
            }
            eff_system = (system or "") + "\n\n" + instruction
        body = {
            "model": model,
            "messages": [{"role": "system", "content": eff_system or ""}] + list(messages),
            "stream": True,
            "max_tokens": _AGENT_MAX_TOKENS,
            **_AGENT_SAMPLING,
            **call_kwargs,
        }
        if tools and not use_envelope:
            body["tools"] = list(tools)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        _SSE_DATA = re.compile(r"^data:\s*(.+)$")
        acc_content = ""
        acc_tool_calls: Dict[int, Dict[str, Any]] = {}
        finish_reason = None
        try:
            async with _httpx.AsyncClient(timeout=_httpx.Timeout(120)) as client:
                async with client.stream(
                    "POST",
                    f"{base_url.rstrip('/')}/chat/completions",
                    json=body, headers=headers,
                ) as resp:
                    if resp.status_code >= 400:
                        err_text = (await resp.aread()).decode("utf-8", "replace")
                        raise RuntimeError(f"model API error {resp.status_code}: {err_text[:200]}")
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        m = _SSE_DATA.match(line.strip())
                        if not m:
                            continue
                        payload = m.group(1).strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            import logging
                            logging.getLogger(__name__).debug(
                                "SSE: skipping malformed JSON chunk: %s", payload[:100]
                            )
                            continue
                        choice = (chunk.get("choices") or [{}])[0]
                        delta = choice.get("delta") or {}
                        finish_reason = choice.get("finish_reason") or finish_reason
                        delta_text = delta.get("content") or ""
                        if delta_text:
                            acc_content += delta_text
                            # In envelope mode the model returns JSON; streaming
                            # it raw would flash raw JSON in the UI, so we hold
                            # off and emit only the clean final answer later.
                            if on_text_delta and not use_envelope:
                                await on_text_delta(delta_text)
                        for tc_delta in delta.get("tool_calls") or []:
                            idx = tc_delta.get("index", 0)
                            if idx not in acc_tool_calls:
                                acc_tool_calls[idx] = {
                                    "id": tc_delta.get("id", f"call_{idx}"),
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            tc = acc_tool_calls[idx]
                            fn = tc_delta.get("function") or {}
                            if fn.get("name"):
                                tc["function"]["name"] = fn["name"]
                            if fn.get("arguments"):
                                tc["function"]["arguments"] += fn["arguments"]
                            if tc_delta.get("id"):
                                tc["id"] = tc_delta["id"]
        except Exception as exc:
            if diag is not None:
                diag.record_exception("model_stream", exc, message=str(exc))
            # Streaming failed — fall back to non-streaming with progressive
            # degradation: native tools first, then plain text. The envelope
            # call may have failed because the backend (e.g. OpenRouter free
            # models) doesn't support json_schema response_format.
            fallback = LocalChatExpert(
                model=model,
                base_url=base_url,
                api_key=api_key,
                max_tokens=_AGENT_MAX_TOKENS,
                **_AGENT_SAMPLING,
            )
            try:
                message = await fallback.chat_completion(
                    messages, system=system, tools=tools,
                )
            except Exception as exc2:
                if diag is not None:
                    diag.record_exception("model_fallback", exc2, message=str(exc2))
                # Native tools also failed — plain text last resort
                message = await fallback.chat_completion(
                    messages, system=system, tools=None,
                )
            if on_text_delta and message.get("content"):
                await _emit_text_delta_typed(on_text_delta, str(message["content"]))
            return message
        result: Dict[str, Any] = {"role": "assistant", "content": acc_content}
        if acc_tool_calls:
            result["tool_calls"] = [acc_tool_calls[i] for i in sorted(acc_tool_calls)]
        if finish_reason:
            result["finish_reason"] = finish_reason
        if use_envelope:
            result = _apply_agent_envelope(result, model)
            # We suppressed raw-JSON streaming above; now emit only the clean
            # final answer (envelope "final" actions surface the answer in
            # content) so the UI animates the reply instead of flashing JSON.
            if on_text_delta and result.get("content"):
                await _emit_text_delta_typed(on_text_delta, str(result["content"]))
        if diag is not None:
            diag.record(
                "model", "stream_response",
                model=model, content_preview=acc_content[:600],
                finish_reason=finish_reason,
            )
        return result

    completion_fn._streaming = completion_fn_streaming  # type: ignore[attr-defined]
    return completion_fn
def _build_agent_tools(
    workspace_roots: Optional[List[str]],
    capabilities: Optional[List[str]],
    tools: Optional[List[str]],
    access_mode: Optional[str],
    require_approval: Optional[bool],
    allow_rules: Optional[List[str]] = None,
    deny_rules: Optional[List[str]] = None,
    ask_rules: Optional[List[str]] = None,
):
    """
    Assemble the agent's tool surface.
    When ``capabilities`` are declared, builds a ToolExecutor from the
    CapabilityRegistry (root-guarded, policy-checked). Otherwise falls back
    to the legacy workspace tools for backward compatibility. Returns
    ``{"executor": ..., "tools": [...]}``.
    """
    from ..capabilities.base import CapabilityContext, PermissionPolicy, ToolExecutor
    from ..capabilities.registry import CapabilityRegistry
    from ..capabilities.rules import ASK, ALLOW, DENY, PermissionRules
    if not capabilities:
        from ..tools.workspace import build_workspace_tools
        return {
            "executor": None,
            "tools": build_workspace_tools(roots=workspace_roots, allow=tools),
        }
    mode = "approval" if require_approval else (access_mode or "action")
    rules = PermissionRules()
    for kind, rule_list in ((ALLOW, allow_rules), (DENY, deny_rules), (ASK, ask_rules)):
        for raw in rule_list or []:
            rules.add(kind, str(raw), source="session")
    policy = PermissionPolicy(access_mode=mode, broker=_approvals(), rules=rules)
    ctx = CapabilityContext(roots=workspace_roots or [])
    specs = CapabilityRegistry().register_defaults().build(capabilities, ctx=ctx, policy=policy).tools()
    # Managed MCP servers contribute their connected, namespaced tools
    # (mcp__<server>__<tool>) to EVERY run — an MCP tool is just a Tool.
    try:
        mcp_specs = _mcp_manager().active_tool_specs()
        if mcp_specs:
            seen = {s.name for s in specs}
            for mcp_spec in mcp_specs:
                if mcp_spec.name not in seen:
                    specs.append(mcp_spec)
                    seen.add(mcp_spec.name)
    except Exception:  # MCP must never break a plain run
        pass
    # Local SKILL.md packages join every run too (prompt skills expand to
    # instructions; script skills execute lazily behind the Tool contract).
    try:
        skill_specs = _skills_manager().active_tool_specs()
        if skill_specs:
            seen = {s.name for s in specs}
            for skill_spec in skill_specs:
                if skill_spec.name not in seen:
                    specs.append(skill_spec)
                    seen.add(skill_spec.name)
    except Exception:  # skills must never break a plain run
        pass
    executor = ToolExecutor(specs, policy=policy)
    if tools:
        keep = set(tools)
        executor = ToolExecutor(
            [t for t in executor.tools() if t.name in keep], policy=policy
        )
    return {"executor": executor, "tools": executor.tools()}
_NO_FABRICATION_RULE = (
    "Never claim to have created, edited, copied, moved, deleted, or ran "
    "anything unless you actually issued the matching tool call in THIS "
    "conversation and it returned success. If a tool isn't available, if a "
    "tool call fails, or if you are unsure whether something happened, say "
    "so plainly instead of describing the outcome as if it occurred.\n\n"
)


def _agent_access_prefix(
    access_mode: Optional[str],
    allow_tools: Optional[bool],
    require_approval: Optional[bool],
) -> str:
    """Guidance text prepended to the agent system prompt per access mode."""
    if require_approval or access_mode == "approval":
        return _NO_FABRICATION_RULE + (
            "Approval is required for impactful actions. Explain the intended "
            "step and ask for confirmation before doing anything risky.\n\n"
        )
    if access_mode == "partial":
        return _NO_FABRICATION_RULE + (
            "Partial access is enabled. Prefer to ask before making impactful "
            "changes, but continue with safe reasoning and low-risk steps.\n\n"
        )
    if access_mode == "full" or allow_tools:
        return _NO_FABRICATION_RULE + (
            "Full access is enabled. When the user asks you to create, edit, "
            "copy, move, or delete something, you MUST use a tool to do it — "
            "do not just describe doing it. Use available tools whenever they "
            "help, while still being careful.\n\n"
        )
    return _NO_FABRICATION_RULE
def _assemble_text_agent_system_prompt(
    prompt_parts: Optional[List[Dict[str, Any]]],
    system_prompt: Optional[str],
    executor: Any,
    agent_tools: List[object],
) -> str:
    """System prompt for the free-text agent protocol used by small / local
    models (``ORCHA_AGENT_PROTOCOL=text``).

    Teaches the model to emit tool calls as a single flat line
    ``TOOL:name:{"arg":"value"}`` and the final answer as ``FINAL ANSWER:
    ...`` — a deterministic, brace-light format tiny models follow far more
    reliably than the strict JSON-schema envelope.
    """
    specs = executor.tools() if executor is not None else list(agent_tools or [])
    tool_lines = []
    for spec in specs:
        params = spec.parameters if isinstance(getattr(spec, "parameters", None), dict) else {}
        props = params.get("properties", {}) if isinstance(params.get("properties"), dict) else {}
        req = params.get("required", []) if isinstance(params.get("required"), list) else []
        pdesc = ", ".join(
            f"{k}{'' if k in req else '?'}" for k in props.keys()
        )
        tool_lines.append(f"- {spec.name}: {spec.description}  (params: {pdesc or 'none'})")
    tools_block = "\n".join(tool_lines) if tool_lines else "(no tools — answer directly)"

    task_line = assemble_system_prompt(prompt_parts)[:1500] if prompt_parts else ""

    examples = (
        "Examples of the EXACT format you must use:\n"
        "\n"
        "STEP 1 — Read a file:\n"
        'TOOL:read_file:{"path":"src/app.py"}\n'
        "\n"
        "STEP 2 — Edit it (you MUST read first to get exact text):\n"
        'TOOL:edit_file:{"path":"src/app.py","old_string":"function oldName","new_string":"function newName"}\n'
        "\n"
        "STEP 3 — Search for code:\n"
        'TOOL:search_text:{"query":"useState"}\n'
        "\n"
        "STEP 4 — Run a build:\n"
        'TOOL:run_command:{"command":"npm run build"}\n'
        "\n"
        "STEP 5 — Run tests:\n"
        'TOOL:run_command:{"command":"npm test"}\n'
        "\n"
        "STEP 6 — Check git status:\n"
        'TOOL:git_status:{}\n'
        "\n"
        "STEP 7 — Stage and commit:\n"
        'TOOL:git_add:{"paths":["src/app.ts"]}\n'
        'TOOL:git_commit:{"message":"fix: rename oldName to newName"}\n'
        "\n"
        "STEP 8 — Final answer:\n"
        "FINAL ANSWER: I renamed oldName to newName in src/app.py, verified the build passes, and committed the change.\n"
        "\n"
        "To create a new file:\n"
        'TOOL:create_file:{"path":"src/utils.ts","content":"export function helper() { return 42; }"}\n'
        "\n"
        "To list directory contents:\n"
        'TOOL:list_directory:{"path":"src"}\n'
        "\n"
        "To show the project tree:\n"
        'TOOL:directory_tree:{"path":".","depth":3}\n'
    )

    base = (
        "You are a precise tool-using agent. Communicate using ONLY these two "
        "line formats, with no markdown fences and no extra commentary.\n"
        "\n"
        "RULES:\n"
        "1) To use a tool, write exactly one line:\n"
        '   TOOL:<tool_name>:{"param":"value",...}\n'
        "   The arguments MUST be a single valid JSON object on that one line.\n"
        "2) When the task is fully done, write exactly one line:\n"
        "   FINAL ANSWER: <your concise answer>\n"
        "\n"
        "IMPORTANT:\n"
        "- ALWAYS read a file BEFORE editing it (edit_file needs exact text)\n"
        "- After each tool result, emit the next TOOL: line or FINAL ANSWER:\n"
        "- If a tool fails, read the error and retry with corrected args\n"
        "- Never invent tool output — only use what the tool returns\n"
        "- Never claim to have done something unless the tool succeeded\n"
        "\n"
        "TOOL SELECTION GUIDE:\n"
        "- To explore the project: directory_tree, list_directory, project_summary\n"
        "- To read code: read_file, read_multiple_files\n"
        "- To find code: search_text, grep, glob_search, symbol_search\n"
        "- To create/edit: create_file, write_file, edit_file, append_file\n"
        "- To run commands: run_command\n"
        "- To use git: git_status, git_diff, git_add, git_commit, git_log\n"
        "- To build/test: run_build, run_tests, run_linter, type_check\n"
        "\n"
        + examples +
        "\nAvailable tools:\n" + tools_block + "\n"
    )
    if task_line:
        base += "\nTask:\n" + task_line + "\n"
    if system_prompt and system_prompt.strip():
        base += "\n" + system_prompt.strip() + "\n"
    return base


def _assemble_agent_system_prompt(
    prompt_parts: Optional[List[Dict[str, Any]]],
    system_prompt: Optional[str],
    mode: Optional[str],
    access_mode: Optional[str],
    allow_tools: Optional[bool],
    require_approval: Optional[bool],
) -> str:
    """
    Assemble the agent's base system prompt.
    When ``prompt_parts`` are supplied they are composed via
    :func:`assemble_system_prompt` (priority-aware truncation) as the core;
    an explicit ``system_prompt`` persona is layered on top. Otherwise the
    legacy single-string path is preserved (AgentConfig default + optional
    persona). Mode and access-mode guidance are always applied.
    """
    from ..nodes.agent import AgentConfig
    plan_prefix = (
        "You are in plan mode. First outline the approach, constraints, and "
        "next steps before execution.\n\n"
        if mode == "plan"
        else ""
    )
    if prompt_parts:
        core = assemble_system_prompt(prompt_parts)
        if not core:
            core = AgentConfig().system_prompt
        prefix = _agent_access_prefix(access_mode, allow_tools, require_approval)
        composed = prefix + core
        if system_prompt and system_prompt.strip():
            composed = system_prompt.strip() + "\n\n" + composed
        return plan_prefix + composed
    base = AgentConfig().system_prompt
    if system_prompt:
        prefix = _agent_access_prefix(access_mode, allow_tools, require_approval)
        base = prefix + system_prompt.strip() + "\n\n" + base
    return plan_prefix + base
def _build_agent_config(
    orc: Orchestrator,
    *,
    system_prompt: Optional[str],
    prompt_parts: Optional[List[Dict[str, Any]]],
    mode: Optional[str],
    access_mode: Optional[str],
    allow_tools: Optional[bool],
    require_approval: Optional[bool],
    workspace_roots: Optional[List[str]],
    tools: Optional[List[str]],
    capabilities: Optional[List[str]],
    reasoning: Optional[str],
    seed_messages: Optional[List[ChatMessage]],
    max_iterations: Optional[int] = None,
    allow_rules: Optional[List[str]] = None,
    deny_rules: Optional[List[str]] = None,
    ask_rules: Optional[List[str]] = None,
):
    """
    Build the AgentConfig shared by the multi-agent and single-agent graphs:
    system prompt, native completion_fn, and the tool executor.
    """
    from ..nodes.agent import AgentConfig
    model_fn = _build_agent_model_fn(orc)
    # Get the base_url from the synthesizer expert for protocol auto-detection
    ref_expert = orc.experts.get(orc.synthesizer_expert)
    syn_base_url = getattr(ref_expert, "base_url", "") if ref_expert else ""
    text_mode = _agent_protocol(syn_base_url) == "text"
    # In text mode the agent uses the free-text TOOL:/FINAL ANSWER: loop
    # (model_fn), which small models handle far better than the strict
    # JSON-schema envelope. We leave completion_fn unset so AgentNode picks
    # the text loop.
    completion_fn = None if text_mode else _build_agent_completion_fn(orc)
    agent_tools: List[object] = []
    executor = None
    # Build the executor whenever capabilities or an explicit tool allow-list
    # is declared, regardless of whether workspace roots are attached.  Tools
    # that need a root path will return a structured "no workspace" result so
    # the agent can explain the limitation to the user, rather than the entire
    # agent running with zero tools.
    if capabilities or tools:
        built = _build_agent_tools(
            workspace_roots, capabilities, tools, access_mode, require_approval,
            allow_rules=allow_rules, deny_rules=deny_rules, ask_rules=ask_rules,
        )
        agent_tools = built["tools"]
        executor = built["executor"]
    system_prompt = (
        _assemble_text_agent_system_prompt(prompt_parts, system_prompt, executor, agent_tools)
        if text_mode else
        _assemble_agent_system_prompt(
            prompt_parts, system_prompt, mode, access_mode, allow_tools, require_approval,
        )
    )
    return (
        AgentConfig(
            model_fn=model_fn,
            completion_fn=completion_fn,
            system_prompt=system_prompt,
            tools=agent_tools,
            executor=executor,
            capabilities=list(capabilities or []),
            workspace_roots=[str(r) for r in (workspace_roots or [])],
            reasoning_level=reasoning,
            seed_messages=[m.model_dump() for m in seed_messages] if seed_messages else [],
            approval_broker=_approvals(),
            # Honor the caller's iteration budget — previously ignored, so
            # every agent loop silently capped at the dataclass default (5).
            # The default here was later raised to 8, still nowhere near
            # enough for a real multi-file build (write_file once per file,
            # plus a run_command + fix round per file that needs one) —
            # confirmed empirically: an 8-iteration cap exhausted itself
            # after one file's write/run/debug cycle, well before a
            # multi-file app could be scaffolded. No caller in Anvira's UI
            # currently sends an explicit max_iterations, so this default is
            # what every real coding request gets.
            max_iterations=max_iterations if max_iterations is not None else 40,
        ),
        completion_fn,
    )
def _assemble_intent_gate_prompt(
    system_prompt: Optional[str],
    prompt_parts: Optional[List[Dict[str, Any]]],
) -> str:
    """
    Router prompt for the intent gate: the routing contract layered on the
    agent's persona ONLY.
    The instructions/fileOps prompt parts are deliberately excluded — they
    carry the tool-use directive ("call the appropriate tool directly...")
    that must never reach the router, or it would be biased toward declaring
    tool intent on every message, including greetings. Attachment/workspace
    metadata is added separately via ``attachments_ctx`` (see
    :func:`_assemble_attachments_ctx`) so the router is grounded without
    inheriting the tool directive.
    """
    from ..nodes.intent import INTENT_GATE_ROUTER_PROMPT
    persona = ""
    if system_prompt and system_prompt.strip():
        persona = system_prompt.strip()
    elif prompt_parts:
        for part in prompt_parts:
            if part.get("name") == "persona" and str(part.get("text", "")).strip():
                persona = str(part["text"]).strip()
                break
    if persona:
        return persona + "\n\n" + INTENT_GATE_ROUTER_PROMPT
    return INTENT_GATE_ROUTER_PROMPT
def _assemble_attachments_ctx(
    prompt_parts: Optional[List[Dict[str, Any]]],
    workspace_roots: Optional[List[str]],
    capabilities: Optional[List[str]],
) -> str:
    """
    Structured metadata about attached resources for the intent gate.
    Reuses the ``workspace`` prompt part (attachment names/paths/sizes/file
    and folder counts, extracted context, project index with tree and
    important files — built by the frontend from the real workspace model) and
    layers the workspace roots and available capabilities on top. Without this
    the router cannot resolve "this folder"/"these files" and falls back to
    language priors.
    """
    blocks: List[str] = []
    if prompt_parts:
        for part in prompt_parts:
            if part.get("name") == "workspace" and str(part.get("text", "")).strip():
                blocks.append(str(part["text"]).strip())
                break
    if workspace_roots:
        blocks.append("Workspace roots: " + ", ".join(str(r) for r in workspace_roots))
    if capabilities:
        blocks.append("Available capabilities: " + ", ".join(str(c) for c in capabilities))
    return "\n\n".join(blocks)
def _build_readonly_agent_config(agent_config: Any) -> Any:
    """
    Derive the READ-ONLY agent config from the full agent config.
    Keeps every field (system prompt, completion_fn, seed messages) but swaps
    the executor for one containing only SAFE (non-mutating) tools and removes
    the approval channel. Workspace inspection therefore executes read tools
    only and can never write files or request approval.
    """
    from dataclasses import replace
    from ..capabilities.base import SAFE, PermissionPolicy, ToolExecutor
    executor = getattr(agent_config, "executor", None)
    if executor is not None:
        read_tools = [t for t in executor.tools() if t.safety_level == SAFE]
    else:
        read_tools = [
            t for t in (getattr(agent_config, "tools", None) or [])
            if getattr(t, "safety_level", None) == SAFE
        ]
    if not read_tools:
        return None
    readonly_executor = ToolExecutor(
        read_tools, policy=PermissionPolicy(access_mode="full")
    )
    return replace(
        agent_config,
        executor=readonly_executor,
        tools=[],
        approval_broker=None,
    )
def _build_agent_runner_or_graph(config: "AgentGraphConfig"):
    """Engine switch for the single-agent graph.
    Returns the native ``Graph`` (validated, wrapped in ``GraphRuntime`` so
    checkpoints land in the server store) by default — this is the ONLY
    engine with the complexity-gate task-decomposition pipeline
    (task_planner/execution_loop/execution_observer) actually wired in;
    ``build_agent_runner``'s LangGraph implementation never had it ported
    over. Set ``ORCHA_AGENT_ENGINE=langgraph`` (or ``config.engine``) to get
    the genuine LangGraph StateGraph with its interrupt-based final-answer
    approval gate instead — the native engine's own per-tool
    ``PermissionPolicy``/``ApprovalBroker`` (capabilities/base.py) covers
    approval in the meantime, just shaped differently (per-tool ask/allow/
    deny rather than one gate at the end).
    """
    from ..builders import build_agent_runner
    from ..builders.agent import build_agent_graph
    engine = os.environ.get("ORCHA_AGENT_ENGINE") or config.engine or "native"
    if engine == "langgraph":
        return build_agent_runner(config, checkpointer=_get_langgraph_checkpointer())
    return build_agent_graph(config)
def _build_multi_agent_runner_or_graph(config: "MultiAgentGraphConfig"):
    """Engine switch for the multi-agent graph.
    Returns a ``MultiAgentLangGraphRunner`` (genuine LangGraph StateGraph
    with Send-based scatter/gather) by default; set
    ``ORCHA_AGENT_ENGINE=native`` to get the validated native ``Graph``
    instead — the call sites then wrap it in ``GraphRuntime``
    (checkpoints land in the server store).
    """
    from ..builders import build_multi_agent_runner
    from ..builders.multi_agent import build_multi_agent_graph
    engine = os.environ.get("ORCHA_AGENT_ENGINE") or "langgraph"
    if engine == "langgraph":
        return build_multi_agent_runner(
            config, checkpointer=_get_langgraph_checkpointer(),
        )
    return build_multi_agent_graph(config)
def _build_graph_by_name(
    name: str,
    orc: Orchestrator,
    system_prompt: Optional[str] = None,
    mode: Optional[str] = None,
    access_mode: Optional[str] = None,
    allow_tools: Optional[bool] = None,
    require_approval: Optional[bool] = None,
    workspace_roots: Optional[List[str]] = None,
    tools: Optional[List[str]] = None,
    capabilities: Optional[List[str]] = None,
    reasoning: Optional[str] = None,
    messages: Optional[List[ChatMessage]] = None,
    prompt_parts: Optional[List[Dict[str, Any]]] = None,
    max_iterations: Optional[int] = None,
    *,
    allow_rules: Optional[List[str]] = None,
    deny_rules: Optional[List[str]] = None,
    ask_rules: Optional[List[str]] = None,
):
    """Build a graph by name using the live orchestrator's config."""
    from ..builders.default import DefaultGraphConfig, build_default_graph
    from ..builders.research import ResearchGraphConfig, build_research_graph
    from ..builders.multi_agent import MultiAgentGraphConfig, build_multi_agent_graph
    from ..builders.agent import AgentGraphConfig, build_agent_graph
    common = dict(
        max_cost=orc.max_cost,
        max_latency_s=orc.max_latency_s,
        max_iterations=orc.max_iterations,
    )
    # Drives the compact planner/executor prompts and tighter context
    # windows for small local models. Read off the synthesizer expert
    # because that is the model that actually runs the agent loop.
    _ref_expert = orc.experts.get(orc.synthesizer_expert)
    is_small_model = _is_small_model(getattr(_ref_expert, "model", "") or "")
    # A tool surface exists when the agent declares capabilities/tools (or
    # tools are force-allowed). Workspace roots are no longer required here —
    # the executor builds with roots=[] when none are attached, and individual
    # tools that genuinely need a root (filesystem, git, terminal) return a
    # clear "no workspace attached" result rather than silently preventing
    # the entire agent from routing through the tool loop. When the native
    # completion_fn is available (an active local model), default and research
    # runs are routed through the single-agent graph so the agent actually
    # executes tools instead of describing them.
    has_tool_surface = bool(capabilities or tools or allow_tools)
    # Conversation memory: when the caller ships prior turns but exposes NO
    # tool surface, the legacy default/research graphs would silently drop
    # them — they only ever see packet.query, so every turn would answer in
    # a vacuum ("123 Whiskers" class of failures). Route those runs through
    # the single-agent path too: with no declared capabilities/tools the
    # executor stays empty, so it degrades to a plain chat loop that still
    # seeds history into every model call (and the intent gate answers
    # chat-only messages directly from that same history).
    wants_memory_chat = bool(messages) and not has_tool_surface
    build_agent_path = has_tool_surface or wants_memory_chat
    if name == "research":
        if build_agent_path:
            agent_config, completion_fn = _build_agent_config(
                orc,
                system_prompt=system_prompt,
                prompt_parts=prompt_parts,
                mode=mode,
                access_mode=access_mode,
                allow_tools=allow_tools,
                require_approval=require_approval,
                workspace_roots=workspace_roots,
                tools=tools,
                capabilities=capabilities,
                reasoning=reasoning,
                seed_messages=messages,
                max_iterations=max_iterations,
                allow_rules=allow_rules,
                deny_rules=deny_rules,
                ask_rules=ask_rules,
            )
            if completion_fn is not None:
                from ..nodes.intent import IntentGateConfig
                from ..nodes.planner import ComplexityGateConfig
                return _build_agent_runner_or_graph(
                    AgentGraphConfig(
                        agent_config=agent_config,
                        executor=agent_config.executor,
                        readonly_agent_config=_build_readonly_agent_config(agent_config),
                        capabilities=list(capabilities or []),
                        workspace_roots=list(workspace_roots or []),
                        reasoning_level=reasoning,
                        gate_config=IntentGateConfig(
                            completion_fn=completion_fn,
                            system_prompt=_assemble_intent_gate_prompt(
                                system_prompt, prompt_parts
                            ),
                            attachments_ctx=_assemble_attachments_ctx(
                                prompt_parts, workspace_roots, capabilities
                            ),
                            seed_messages=(
                                [m.model_dump() for m in messages] if messages else []
                            ),
                        ),
                        # Without this, EVERY tool-intent message — no matter
                        # how large or multi-step — runs through the plain
                        # single-shot agent node only; the decompose->plan->
                        # execute->verify loop below (task_planner,
                        # execution_loop/TaskExecutionLoop, execution_observer,
                        # replanner, final_verifier) is fully built and wired
                        # into build_agent_graph but was never reachable
                        # because this config was never constructed. A big,
                        # detailed multi-file build request is exactly what
                        # this gate exists to route into task decomposition
                        # instead of asking one model call to do everything.
                        complexity_gate_config=(
                            ComplexityGateConfig(
                                completion_fn=completion_fn,
                                seed_messages=(
                                    [m.model_dump() for m in messages] if messages else []
                                ),
                                **(
                                    {"heuristic_threshold": SMALL_MODEL_COMPLEXITY_THRESHOLD}
                                    if (is_small_model and ENABLE_SMALL_MODEL_DECOMPOSITION)
                                    else {}
                                ),
                            )
                            if has_tool_surface else None
                        ),
                        # Native is the real default now: it's the only
                        # engine with the complexity-gate task-decomposition
                        # pipeline wired in (see complexity_gate_config
                        # above). Set ORCHA_AGENT_ENGINE=langgraph to opt
                        # back into the LangGraph engine's interrupt-based
                        # approval gate instead of the native engine's
                        # per-tool ApprovalBroker.
                        engine=os.environ.get("ORCHA_AGENT_ENGINE") or "native",
                        require_approval=require_approval,
                        small_model=is_small_model,
                        **common,
                    )
                )
        return build_research_graph(
            ResearchGraphConfig(
                experts=dict(orc.experts),
                synthesizer_expert=orc.synthesizer_expert,
                **common,
            )
        )
    if name == "multi_agent":
        agent_config, completion_fn = _build_agent_config(
            orc,
            system_prompt=system_prompt,
            prompt_parts=prompt_parts,
            mode=mode,
            access_mode=access_mode,
            allow_tools=allow_tools,
            require_approval=require_approval,
            workspace_roots=workspace_roots,
            tools=tools,
            capabilities=capabilities,
            reasoning=reasoning,
            seed_messages=messages,
            max_iterations=max_iterations,
            allow_rules=allow_rules,
            deny_rules=deny_rules,
            ask_rules=ask_rules,
        )
        if agent_config.model_fn is None and completion_fn is None:
            raise OrchaError(
                "no_local_model",
                "Multi-agent mode needs an active local model to actually run agents on. "
                "Activate a model in Anvira first.",
                400,
            )
        return _build_multi_agent_runner_or_graph(
            MultiAgentGraphConfig(
                max_cost=orc.max_cost,
                max_latency_s=orc.max_latency_s,
                agent_config=agent_config,
                executor=agent_config.executor,
                capabilities=list(capabilities or []),
                reasoning_level=reasoning,
                model_fn=agent_config.model_fn,
                completion_fn=completion_fn,
            )
        )
    # default
    if build_agent_path:
        agent_config, completion_fn = _build_agent_config(
            orc,
            system_prompt=system_prompt,
            prompt_parts=prompt_parts,
            mode=mode,
            access_mode=access_mode,
            allow_tools=allow_tools,
            require_approval=require_approval,
            workspace_roots=workspace_roots,
            tools=tools,
            capabilities=capabilities,
            reasoning=reasoning,
            seed_messages=messages,
            max_iterations=max_iterations,
            )
        if completion_fn is not None or agent_config.model_fn is not None:
            from ..nodes.intent import IntentGateConfig
            from ..nodes.planner import ComplexityGateConfig
            return _build_agent_runner_or_graph(
                AgentGraphConfig(
                    agent_config=agent_config,
                    executor=agent_config.executor,
                    readonly_agent_config=_build_readonly_agent_config(agent_config),
                    capabilities=list(capabilities or []),
                    workspace_roots=list(workspace_roots or []),
                    reasoning_level=reasoning,
                    gate_config=IntentGateConfig(
                        completion_fn=completion_fn,
                        system_prompt=_assemble_intent_gate_prompt(
                            system_prompt, prompt_parts
                        ),
                        attachments_ctx=_assemble_attachments_ctx(
                            prompt_parts, workspace_roots, capabilities
                        ),
                        seed_messages=(
                            [m.model_dump() for m in messages] if messages else []
                        ),
                    ),
                    # See the matching comment in the "research" branch above:
                    # this is what actually turns on task decomposition
                    # (task_planner -> execution_loop/TaskExecutionLoop ->
                    # execution_observer -> verify) for big multi-step
                    # requests instead of forcing everything through one
                    # single-shot model call.
                    complexity_gate_config=(
                        ComplexityGateConfig(
                            completion_fn=completion_fn,
                            seed_messages=(
                                [m.model_dump() for m in messages] if messages else []
                            ),
                            **(
                                {"heuristic_threshold": SMALL_MODEL_COMPLEXITY_THRESHOLD}
                                if (is_small_model and ENABLE_SMALL_MODEL_DECOMPOSITION)
                                else {}
                            ),
                        )
                        if has_tool_surface else None
                    ),
                    # See the matching comment in the "research" branch
                    # above: native is the real default now.
                    engine=os.environ.get("ORCHA_AGENT_ENGINE") or "native",
                    require_approval=require_approval,
                    small_model=is_small_model,
                    **common,
                )
            )
    return build_default_graph(
        DefaultGraphConfig(
            experts=dict(orc.experts),
            synthesizer_expert=orc.synthesizer_expert,
            run_all_experts=orc.run_all_experts,
            **common,
        )
    )
# ── Graph-runtime diagnostics (Prompt 11: same store/viewer as agent runs) ────
def _graph_run_meta(req: "RunRequest", orc: Orchestrator) -> Dict[str, Any]:
    """Observability metadata for a graph-runtime diagnostic session."""
    return {
        "pipeline": "graph_runtime",
        "runtime_version": ORCHA_VERSION,
        "backend": getattr(orc, "_source", "unknown"),
        "model": orc.synthesizer_expert or "",
        "streaming": req.stream,
        "graph": req.graph,
        "workspace_roots": list(req.workspace_roots or []),
        "attached_files": list(req.workspace_roots or []),
        "query": req.query[:2000],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "orcha_version": ORCHA_VERSION,
        },
    }
def _record_request_received(diag: Any, req: "RunRequest") -> None:
    """Record what was received for this run — including whether an
    attachment (the frontend-built ``workspace`` prompt part) was present
    in the payload and what its resolved text looked like."""
    parts = list(req.prompt_parts or [])
    workspace = next(
        (p for p in parts if p.get("name") == "workspace"), None
    )
    workspace_text = str(workspace.get("text") or "") if workspace else ""
    diag.record(
        "request", "received",
        graph=req.graph, query=req.query[:2000], stream=req.stream,
        mode=req.mode, access_mode=req.access_mode,
        allow_tools=req.allow_tools, require_approval=req.require_approval,
        workspace_roots=list(req.workspace_roots or []),
        capabilities=list(req.capabilities or []),
        tools=list(req.tools or []),
        reasoning=req.reasoning,
        messages=len(req.messages or []),
        prompt_parts=[
            {"name": p.get("name"), "chars": len(str(p.get("text") or ""))}
            for p in parts
        ],
        attachments_present=workspace is not None,
        attachment_part="workspace" if workspace is not None else None,
        attachment_chars=len(workspace_text),
        attachment_preview=workspace_text[:1500],
    )
def _record_assembled_prompts(diag: Any, req: "RunRequest") -> None:
    """Record the prompts assembled from the request — the evidence of what
    the attachment context looked like going into the model calls."""
    agent_prompt = _assemble_agent_system_prompt(
        req.prompt_parts, req.system_prompt, req.mode, req.access_mode,
        req.allow_tools, req.require_approval,
    )
    attachments_ctx = _assemble_attachments_ctx(
        req.prompt_parts, req.workspace_roots, req.capabilities,
    )
    gate_prompt = _assemble_intent_gate_prompt(
        req.system_prompt, req.prompt_parts,
    )
    workspace = next(
        (p for p in (req.prompt_parts or []) if p.get("name") == "workspace"),
        None,
    )
    workspace_text = str(workspace.get("text") or "").strip() if workspace else ""
    diag.record(
        "prompt", "assembled",
        agent_system_prompt_chars=len(agent_prompt),
        agent_system_prompt_preview=agent_prompt[:2000],
        attachment_text_in_agent_prompt=(
            bool(workspace_text) and workspace_text in agent_prompt
        ),
        attachments_ctx_chars=len(attachments_ctx),
        attachments_ctx_preview=attachments_ctx[:2000],
        attachment_text_in_attachments_ctx=(
            bool(workspace_text) and workspace_text in attachments_ctx
        ),
        intent_gate_prompt_chars=len(gate_prompt),
        intent_gate_prompt_preview=gate_prompt[:2000],
    )
@app.post("/v1/run", response_model=RunResponse)
async def run_graph(req: RunRequest):
    """
    Execute an ORCHA3 graph for a query.
    By default runs synchronously and returns the final result. Set
    ``stream=true`` to run in the background and consume events via
    ``GET /v1/run/{run_id}/events``.
    """
    orc = _orc()
    store = _state.store or MemoryStore()
    run_id = str(uuid.uuid4())
    # Prompt 11: read-only diagnostics for this run — the SAME store the
    # agent-runs pipeline uses, tagged pipeline=graph_runtime so the viewer
    # can tell the two apart in one list.
    diag = get_diagnostics_store().start(
        run_id, run_id=run_id, meta=_graph_run_meta(req, orc),
    )
    diag.record("session", "start", session=run_id, outcome="running")
    _record_request_received(diag, req)
    for root in req.workspace_roots or []:
        diag.record_attachment(root)
    token = _RUN_DIAG.set(diag)
    bg_owns_diag = False
    try:
        try:
            built = _build_graph_by_name(
                req.graph,
                orc,
                req.system_prompt,
                req.mode,
                req.access_mode,
                req.allow_tools,
                req.require_approval,
                req.workspace_roots,
                req.tools,
                req.capabilities,
                req.reasoning,
                req.messages,
                req.prompt_parts,
                max_iterations=req.max_iterations,
                allow_rules=req.allow_rules,
                deny_rules=req.deny_rules,
                ask_rules=req.ask_rules,
            )
        except OrchaError as exc:
            diag.record_exception("graph", exc, message=str(exc))
            raise
        except Exception as exc:
            diag.record_exception("graph", exc, message=str(exc))
            raise OrchaError("graph_build_failed", str(exc), 400)
        _record_assembled_prompts(diag, req)
        # Agent graphs may arrive as a LangGraph runner (ORCHA_AGENT_ENGINE=
        # langgraph); everything else is a bare Graph wrapped in GraphRuntime
        # so checkpoints land in the server store.
        from ..builders import AgentLangGraphRunner, MultiAgentLangGraphRunner
        if isinstance(built, (AgentLangGraphRunner, MultiAgentLangGraphRunner)):
            runtime = built
            graph_name = built.graph_name
        else:
            runtime = GraphRuntime(built, store=store)
            graph_name = built.name
        if req.stream:
            if len(_state.active_runs) >= _MAX_CONCURRENT_BG_RUNS:
                raise OrchaError(
                    "too_many_runs",
                    f"Too many background runs are already in flight "
                    f"(max {_MAX_CONCURRENT_BG_RUNS}). Try again shortly.",
                    429,
                )
            # Launch in background; client consumes events via SSE. Failures are
            # recorded (not just logged) so get_run / the SSE stream can surface
            # them instead of reporting a failed run as "completed".
            async def _bg_run():
                try:
                    await runtime.run(req.query, run_id=run_id)
                    diag.outcome = "success"
                    diag.terminated_by = "completed"
                except ApprovalPending as exc:
                    # LangGraph final-answer gate: the run pauses, not fails.
                    # Keep the runner alive so the decision endpoint can resume
                    # the paused thread, and persist the record so the gate
                    # survives a server restart.
                    _state.pending_approvals[run_id] = {
                        "query": req.query,
                        "request": req.model_dump(mode="json"),
                        "runner": runtime,
                        "ts": time.time(),
                    }
                    _persist_pending_approvals()
                    diag.outcome = "pending_approval"
                    diag.error_summary = "awaiting final-answer approval"
                except asyncio.CancelledError:
                    diag.outcome = "failed"
                    diag.error_summary = "cancelled"
                    raise
                except Exception as exc:
                    # Keep the error typed (not a bare string) so the UI can
                    # tell a routing/config failure from a model timeout.
                    _state.run_errors[run_id] = f"{type(exc).__name__}: {exc}"
                    _log.warning("background_run_failed run=%s error=%s",
                                 run_id[:8], exc, exc_info=exc)
                    diag.outcome = "failed"
                    diag.error_summary = str(exc)
                    diag.record_exception("run", exc, message=str(exc))
                finally:
                    _state.active_runs.pop(run_id, None)
                    if run_id not in _state.pending_approvals:
                        # A finished run must not leave pending approvals
                        # behind: a stale entry could otherwise be decided later
                        # and leak a grant into a newer run's identical tool
                        # call. A paused run keeps its runner + broker state.
                        _state.run_runtimes.pop(run_id, None)
                        _approvals().clear_run(run_id)
                    diag.finish()
                    get_diagnostics_store().finalize(diag)
            _state.run_runtimes[run_id] = runtime
            _state.run_requests[run_id] = req.model_dump(mode="json")
            task = asyncio.create_task(_bg_run())
            _state.active_runs[run_id] = task
            bg_owns_diag = True
            return RunResponse(
                run_id=run_id, graph_name=graph_name, status="running",
            )
        # Synchronous: run to completion and return the result.
        _state.run_requests[run_id] = req.model_dump(mode="json")
        try:
            result = await runtime.run(req.query, run_id=run_id)
        except ApprovalPending as exc:
            # LangGraph final-answer gate: respond with the proposed answer
            # and keep the runner alive for the decision endpoint; persist
            # the record so the gate survives a server restart.
            _state.pending_approvals[run_id] = {
                "query": req.query,
                "request": req.model_dump(mode="json"),
                "runner": runtime,
                "ts": time.time(),
            }
            _persist_pending_approvals()
            _state.run_runtimes[run_id] = runtime
            diag.outcome = "pending_approval"
            diag.error_summary = "awaiting final-answer approval"
            return RunResponse(
                run_id=run_id, graph_name=graph_name,
                status="pending_approval", approval=dict(exc.approval_payload),
            )
        except asyncio.CancelledError:
            raise
        except GraphError as exc:
            diag.outcome = "failed"
            diag.error_summary = str(exc)
            diag.record_exception("run", exc, message=str(exc))
            raise OrchaError("graph_error", str(exc), 500)
        except Exception as exc:
            diag.outcome = "failed"
            diag.error_summary = str(exc)
            diag.record_exception("run", exc, message=str(exc))
            raise
        finally:
            # Same cleanup as the background path: no stale approvals for a run
            # that has already ended (a paused run keeps its broker state).
            if run_id not in _state.pending_approvals:
                _approvals().clear_run(run_id)
        diag.outcome = "success"
        diag.terminated_by = "completed"
        return _run_result_to_response(result, run_id)
    finally:
        _RUN_DIAG.reset(token)
        if not bg_owns_diag:
            diag.finish()
            get_diagnostics_store().finalize(diag)
@app.get("/v1/run/{run_id}", response_model=RunResponse)
async def get_run(run_id: str):
    """
    Get the latest state of a run by id.
    If the run is still in flight, returns the latest checkpoint. If it
    completed, returns the final result.
    """
    cancelled_message = _state.cancelled_runs.get(run_id)
    if cancelled_message:
        return RunResponse(
            run_id=run_id, graph_name="", status="cancelled",
            error=cancelled_message,
        )
    store = _state.store or MemoryStore()
    cp = await store.load_checkpoint(run_id)
    if cp is None:
        # Maybe still in flight with no checkpoint yet.
        if run_id in _state.active_runs:
            return RunResponse(
                run_id=run_id, graph_name="", status="running",
            )
        # LangGraph-engine runs checkpoint into the durable Sqlite
        # checkpointer, not the FileStore — read the thread directly so
        # get_run surfaces them (including paused approval threads and
        # completed runs whose runner has been dropped).
        error_message = _state.run_errors.get(run_id)
        if error_message:
            return RunResponse(
                run_id=run_id, graph_name="", status="failed", error=error_message,
            )
        thread = _get_langgraph_checkpointer().read_thread(run_id)
        if thread is not None:
            packet = thread["values"].get("packet")
            if isinstance(packet, OrchaPacket):
                result = RunResult(packet, graph_name="")
                if thread["failed"]:
                    return RunResponse(
                        run_id=run_id, graph_name="", status="failed",
                        error="Run failed before completing.",
                    )
                status = "pending_approval" if thread["paused"] else "completed"
                return _run_result_to_response(result, run_id, status=status)
        raise OrchaError("run_not_found", f"Run {run_id} not found", 404)
    # Reconstruct a RunResult from the checkpoint's packet.
    result = RunResult(cp.packet, graph_name="")
    status = "running" if run_id in _state.active_runs else "completed"
    error_message = None if run_id in _state.active_runs else _state.run_errors.get(run_id)
    if error_message:
        status = "failed"
    response = _run_result_to_response(result, run_id, status=status)
    response.error = error_message
    return response
@app.get("/v1/run/{run_id}/approvals")
async def list_run_approvals(run_id: str):
    """
    List pending interactive tool-approval requests for a run.
    Agent tool calls that require approval (approval/partial access mode)
    are registered here while the run waits for a human decision.
    """
    return {
        "run_id": run_id,
        "approvals": _approvals().list_pending(run_id=run_id),
    }
@app.post("/v1/run/{run_id}/approvals/{approval_id}")
async def decide_run_approval(run_id: str, approval_id: str, req: ApprovalDecisionRequest):
    """
    Approve or reject one pending tool-approval request for a run.
    An approved request unblocks the agent loop (it re-executes the tool
    call); a rejected one is relayed to the agent as a denial so it can
    adjust. Unknown or already-decided ids return 404.
    """
    decided = _approvals().decide(
        approval_id, approved=req.approved, run_id=run_id, update=req.update,
    )
    if not decided:
        raise OrchaError(
            "approval_not_found",
            f"Approval request {approval_id} for run {run_id} not found or already decided",
            404,
        )
    return {
        "id": approval_id,
        "run_id": run_id,
        "status": "approved" if req.approved else "rejected",
        "update_recorded": bool(req.update),
    }


@app.get("/v1/rules/updates")
def list_rule_updates():
    """
    Drain recorded permission updates ('don't ask again' answers). The
    client persists these into its rule store (session/project settings);
    each call consumes the queue.
    """
    return {"updates": _approvals().drain_updates()}


# ── MCP server management ─────────────────────────────────────────────────────

class McpServerAddRequest(BaseModel):
    """Body for POST /v1/mcp/servers."""
    name: str = Field(..., min_length=1, max_length=64)
    transport: str = Field(default="stdio", description="'stdio' or 'sse'.")
    command: Optional[str] = Field(default=None, description="stdio: executable to spawn.")
    args: List[str] = Field(default_factory=list, description="stdio: command arguments.")
    env: Dict[str, str] = Field(default_factory=dict, description="stdio: environment overrides.")
    url: Optional[str] = Field(default=None, description="sse: server endpoint URL.")
    headers: Dict[str, str] = Field(default_factory=dict, description="sse: HTTP headers.")
    timeout_s: float = Field(default=30.0, ge=1.0, le=300.0)
    enabled: bool = True


@app.get("/v1/mcp/servers")
def list_mcp_servers():
    """Every configured MCP server with its connection state and tools."""
    return {"servers": _mcp_manager().list_status()}


@app.post("/v1/mcp/servers", status_code=201)
async def add_mcp_server(req: McpServerAddRequest):
    """Register an MCP server (persisted) and try to connect immediately.
    A failed connection is NOT an error: the server lands in failed/
    needs_auth state with the reason in status."""
    from .mcp_manager import McpServerSettings

    settings = McpServerSettings(**req.model_dump())
    try:
        status_dict = await asyncio.to_thread(_mcp_manager().add, settings)
    except ValueError as exc:
        raise OrchaError("mcp_duplicate_name", str(exc), 409)
    except Exception as exc:
        raise OrchaError("mcp_add_failed", str(exc), 400)
    return status_dict


@app.delete("/v1/mcp/servers/{name}")
async def remove_mcp_server(name: str):
    if not await asyncio.to_thread(_mcp_manager().remove, name):
        raise OrchaError("mcp_not_found", f"No MCP server named {name!r}", 404)
    return {"removed": True, "name": name}


@app.post("/v1/mcp/servers/{name}/reconnect")
async def reconnect_mcp_server(name: str):
    """Drop the current connection and retry WITH exponential backoff."""
    try:
        status_dict = await asyncio.to_thread(_mcp_manager().reconnect, name)
    except KeyError:
        raise OrchaError("mcp_not_found", f"No MCP server named {name!r}", 404)
    return status_dict


@app.post("/v1/mcp/servers/{name}/refresh")
async def refresh_mcp_server(name: str):
    """Re-list a connected server's tools; picks up additions/removals."""
    try:
        status_dict = await asyncio.to_thread(_mcp_manager().refresh, name)
    except KeyError:
        raise OrchaError("mcp_not_found", f"No MCP server named {name!r}", 404)
    return status_dict


# ── Local SKILL.md packages ───────────────────────────────────────────────────

@app.get("/v1/skills")
def list_skills():
    """
    Every skill discovered in ORCHA_SKILLS_DIR directories: prompt skills
    (markdown-only packages) and script skills (lazy entrypoints), with the
    permission rules each one asks to run under.
    """
    return {"skills": _skills_manager().list_skills()}


@app.post("/v1/skills/reload")
async def reload_skills():
    """Rescan all configured skills directories."""
    return await asyncio.to_thread(_skills_manager().reload)
@app.post("/v1/run/{run_id}/approval", response_model=RunResponse)
async def decide_agent_approval(run_id: str, req: AgentApprovalDecisionRequest):
    """
    Decide a LangGraph final-answer approval for a paused run.
    ``POST /v1/run`` with the LangGraph agent engine
    (``ORCHA_AGENT_ENGINE=langgraph``) and ``require_approval=true`` pauses
    at the approval gate and returns ``status=pending_approval`` with the
    proposed answer. This endpoint resumes the paused thread with the
    human's decision: ``approved=true`` accepts the answer as-is,
    ``approved=false`` (with an optional ``note``) makes the agent answer
    with the refusal. Unknown or already-decided runs return 404.
    The gate survives server restarts: when the in-memory runner is gone,
    the runner is rebuilt from the persisted request and the thread is
    resumed from the durable Sqlite checkpointer.
    """
    from ..builders import AgentLangGraphRunner, MultiAgentLangGraphRunner
    pending = _state.pending_approvals.get(run_id)
    if pending is None:
        # Restart survival: the record was restored from disk.
        records = _read_pending_records()
        rec = records.get(run_id)
        if rec is not None:
            pending = {
                "query": rec.get("query", ""),
                "request": rec.get("request"),
                "runner": None,
            }
            _state.pending_approvals[run_id] = pending
    if pending is None:
        # Last resort: the LangGraph checkpointer may still have the thread
        # paused even if the pending record was lost (e.g. process restart
        # before persist, or persist silently failed).  If the thread IS
        # paused and we have a stored RunRequest, reconstruct the record.
        cp = _get_langgraph_checkpointer().read_thread(run_id)
        if cp is not None and cp.get("paused"):
            stored_req = _state.run_requests.get(run_id)
            if stored_req is not None:
                _log.warning(
                    "approval record missing but thread paused — "
                    "reconstructing from stored request run=%s",
                    run_id[:8],
                )
                pending = {
                    "query": stored_req.get("query", ""),
                    "request": stored_req,
                    "runner": None,
                }
                _state.pending_approvals[run_id] = pending
                _persist_pending_approvals()
    if pending is None:
        raise OrchaError(
            "approval_not_pending",
            f"Run {run_id} is not awaiting a final-answer approval",
            404,
        )
    runtime = _state.run_runtimes.get(run_id)
    if not isinstance(runtime, (AgentLangGraphRunner, MultiAgentLangGraphRunner)):
        # The runner did not survive (restart, eviction): rebuild it from
        # the persisted request so the durable thread can be resumed.
        _log.info(
            "rebuilding langgraph runner for decision run=%s", run_id[:8],
        )
        stored_req = RunRequest.model_validate(pending["request"])
        built = _build_graph_by_name(
            stored_req.graph,
            _orc(),
            system_prompt=stored_req.system_prompt,
            mode=stored_req.mode,
            access_mode=stored_req.access_mode,
            allow_tools=stored_req.allow_tools,
            require_approval=stored_req.require_approval,
            workspace_roots=stored_req.workspace_roots,
            tools=stored_req.tools,
            capabilities=stored_req.capabilities,
            reasoning=stored_req.reasoning,
            messages=stored_req.messages,
            prompt_parts=stored_req.prompt_parts,
            max_iterations=stored_req.max_iterations,
            allow_rules=getattr(stored_req, "allow_rules", None),
            deny_rules=getattr(stored_req, "deny_rules", None),
            ask_rules=getattr(stored_req, "ask_rules", None),
        )
        if not isinstance(built, (AgentLangGraphRunner, MultiAgentLangGraphRunner)):
            raise OrchaError(
                "approval_rebuild_failed",
                f"Run {run_id} was paused on a graph that no longer builds "
                f"as a LangGraph runner",
                500,
            )
        runtime = built
        _state.run_runtimes[run_id] = runtime
    try:
        result = await runtime.run(
            query=pending.get("query", ""),
            run_id=run_id,
            resume_from=object(),
            resume_value={"approved": req.approved, "note": req.note or ""},
        )
    except GraphError as exc:
        raise OrchaError("resume_failed", str(exc), 500)
    finally:
        _state.run_runtimes.pop(run_id, None)
        _state.pending_approvals.pop(run_id, None)
        _state.run_requests.pop(run_id, None)
        _approvals().clear_run(run_id)
        _drop_persisted_pending(run_id)
    return _run_result_to_response(
        result, run_id, status="approved" if req.approved else "rejected",
    )
@app.post("/v1/run/{run_id}/cancel")
async def cancel_run(run_id: str):
    """
    Cancel an in-flight background run.
    Stops the background task and records the cancellation so ``get_run``
    reports ``status="cancelled"`` (never a false failure) and the SSE
    stream emits a ``cancelled`` event before closing. Runs paused at the
    final-answer approval gate are dropped from the pending records so no
    later decision can resume them. Cancelling a run that already finished
    (or never existed) returns 409.
    """
    task = _state.active_runs.get(run_id)
    if task is None:
        pending = _state.pending_approvals.pop(run_id, None)
        if pending is not None:
            _state.run_runtimes.pop(run_id, None)
            _state.run_requests.pop(run_id, None)
            _approvals().clear_run(run_id)
            _drop_persisted_pending(run_id)
            _state.cancelled_runs[run_id] = "Run cancelled by the user."
            return {"run_id": run_id, "status": "cancelled"}
        if run_id in _state.cancelled_runs:
            return {"run_id": run_id, "status": "cancelled"}
        raise OrchaError(
            "run_not_active",
            f"Run {run_id} is not running and cannot be cancelled",
            409,
        )
    _state.cancelled_runs[run_id] = "Run cancelled by the user."
    task.cancel()
    # Give the background task a moment to run its cleanup (diag finalize,
    # active_runs pop); asyncio.wait never propagates the task's own
    # CancelledError. A slow non-cooperative section (e.g. a model call)
    # must not block this endpoint — the cancellation is already recorded.
    await asyncio.wait({task}, timeout=5)
    return {"run_id": run_id, "status": "cancelled"}
@app.get("/v1/run/{run_id}/events")
async def stream_run_events(run_id: str):
    """
    Server-Sent Events stream for a run.
    Emits one event per graph transition (node_start, node_end, checkpoint,
    etc.) until the run completes or the client disconnects.
    """
    store = _state.store or MemoryStore()
    orc = _orc()
    # If the run already completed, replay its events from history.
    cps = await store.list_checkpoints(run_id)
    if cps and run_id not in _state.active_runs:
        # Completed run: stream checkpoints as historical events.
        async def _history_stream():
            for cp in cps:
                event = {
                    "run_id": run_id,
                    "node": cp.node_id,
                    "kind": "checkpoint",
                    "ts": cp.ts,
                    "data": {"seq": cp.seq, "next_node": cp.next_node},
                }
                yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(0)
            cancelled_message = _state.cancelled_runs.get(run_id)
            if cancelled_message:
                yield (
                    f"data: {json.dumps({'run_id': run_id, 'kind': 'cancelled', 'message': cancelled_message})}\n\n"
                )
            yield f"data: {json.dumps({'run_id': run_id, 'kind': 'end'})}\n\n"
        return StreamingResponse(_history_stream(), media_type="text/event-stream")
    # Live run: subscribe to the runtime's live emitter when available and
    # drain the queue; checkpoints cover any events emitted before attach.
    queue: asyncio.Queue = asyncio.Queue()
    stop = asyncio.Event()
    unsub = None
    async def _sub(event: RunEvent):
        if stop.is_set():
            return
        await queue.put(event.to_dict())
    async def _attach():
        nonlocal unsub
        if unsub is not None:
            return
        rt = _state.run_runtimes.get(run_id)
        if rt is not None and rt.live_emitter is not None:
            unsub = rt.live_emitter.subscribe(_sub)
    seen_seqs = set()
    async def _poll_stream():
        try:
            while not stop.is_set():
                await _attach()
                while True:
                    try:
                        ev = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if ev.get("kind") == "checkpoint":
                        seq = (ev.get("data") or {}).get("seq")
                        if seq is not None:
                            seen_seqs.add(seq)
                    yield f"data: {json.dumps(ev)}\n\n"
                latest_cps = await store.list_checkpoints(run_id)
                for cp in latest_cps:
                    if cp.seq not in seen_seqs:
                        seen_seqs.add(cp.seq)
                        event = {
                            "run_id": run_id,
                            "node": cp.node_id,
                            "kind": "checkpoint",
                            "ts": cp.ts,
                            "data": {"seq": cp.seq, "next_node": cp.next_node},
                        }
                        yield f"data: {json.dumps(event)}\n\n"
                if run_id not in _state.active_runs:
                    cancelled_message = _state.cancelled_runs.get(run_id)
                    if cancelled_message:
                        yield (
                            f"data: {json.dumps({'run_id': run_id, 'kind': 'cancelled', 'message': cancelled_message})}\n\n"
                        )
                    error_message = _state.run_errors.get(run_id)
                    if error_message:
                        yield (
                            f"data: {json.dumps({'run_id': run_id, 'kind': 'error', 'message': error_message})}\n\n"
                        )
                    yield f"data: {json.dumps({'run_id': run_id, 'kind': 'end'})}\n\n"
                    break
                await asyncio.sleep(0.05)
        finally:
            if unsub is not None:
                unsub()
            stop.set()
    return StreamingResponse(_poll_stream(), media_type="text/event-stream")
@app.post("/v1/run/{run_id}/resume", response_model=RunResponse)
async def resume_run(run_id: str):
    """
    Resume an interrupted run from its latest checkpoint.

    The run continues from the node after the last completed checkpoint.
    On resume:
    1. The RunStateTracker is restored from the persisted snapshot in the packet
    2. A RUN_RESUMED event is emitted to the event stream
    3. Completed tasks are not re-executed
    4. The run continues from the correct task

    The tracker restoration happens inside _get_event_adapter() when the
    adapter detects a persisted _event_stream_snapshot in the packet payload.
    """
    store = _state.store or MemoryStore()
    orc = _orc()
    cp = await store.load_checkpoint(run_id)
    if cp is None:
        raise OrchaError("run_not_found", f"Run {run_id} has no checkpoint", 404)
    if cp.next_node == "__END__":
        # Already complete.
        result = RunResult(cp.packet, graph_name="")
        return _run_result_to_response(result, run_id, status="completed")

    # Rebuild the graph and resume.
    # The adapter inside the graph nodes will restore the tracker from
    # _event_stream_snapshot in the packet payload automatically, emitting
    # a RUN_RESUMED event and preserving all prior state.
    graph_name = cp.packet.payload.get("__graph__", "default")
    graph = _build_graph_by_name(graph_name, orc)
    runtime = GraphRuntime(graph, store=store)
    try:
        result = await runtime.run(
            query=cp.packet.query,
            packet=cp.packet,
            resume_from=store,
            run_id=run_id,
        )
    except GraphError as exc:
        raise OrchaError("resume_failed", str(exc), 500)

    return _run_result_to_response(result, run_id, status="resumed")
@app.post("/v1/run/{run_id}/replay", response_model=RunResponse)
async def replay_run(run_id: str):
    """
    Deterministically replay a run from its first checkpoint.
    The run re-executes from scratch using the original packet, re-emitting
    every event. Useful for debugging and benchmarking.
    """
    store = _state.store or MemoryStore()
    orc = _orc()
    cps = await store.list_checkpoints(run_id)
    if not cps:
        raise OrchaError("run_not_found", f"Run {run_id} has no checkpoints", 404)
    graph = _build_graph_by_name("default", orc)
    runtime = GraphRuntime(graph, store=store)
    try:
        result = await runtime.replay(run_id, store=store)
    except GraphError as exc:
        raise OrchaError("replay_failed", str(exc), 500)
    return _run_result_to_response(result, run_id, status="replayed")
@app.get("/v1/runs")
async def list_runs():
    """List all known run ids in the store."""
    store = _state.store or MemoryStore()
    run_ids = await store.list_runs()
    active = list(_state.active_runs.keys())
    return {
        "runs": run_ids,
        "active": active,
        "total": len(run_ids),
    }


@app.get("/v1/run/{run_id}/state")
async def get_run_state(run_id: str):
    """
    Get the full RunStateSnapshot for a run, including the execution event timeline.

    This allows the frontend to reconstruct the full timeline after reconnecting.
    Returns the RunStateSnapshot with:
    - run_id, status, objective, timestamps
    - plan state (steps, progress)
    - current task info
    - completed/failed task IDs
    - verification result
    - final response
    - all ExecutionEvents in order
    """
    # Check if run is active
    if run_id in _state.active_runs:
        # For active runs, try to get state from the runtime
        runtime = _state.run_runtimes.get(run_id)
        if runtime is not None:
            # Check if runtime has an event adapter
            live_emitter = getattr(runtime, "live_emitter", None)
            if live_emitter is not None:
                # Try to find the event adapter from the runtime's state
                # The adapter is stored in the packet payload during execution
                pass

    # Try to get state from checkpoint
    store = _state.store or MemoryStore()
    cp = await store.load_checkpoint(run_id)
    if cp is not None:
        packet = cp.packet
        snapshot = packet.payload.get("_event_stream_snapshot")
        if snapshot is not None:
            return snapshot

        # If no snapshot but we have a packet, construct a basic state
        plan_raw = packet.payload.get("execution_plan")
        if plan_raw is not None:
            from ..core.packets import ExecutionPlan, RunStateSnapshot
            plan = ExecutionPlan(**plan_raw) if isinstance(plan_raw, dict) else plan_raw
            return RunStateSnapshot(
                run_id=run_id,
                status="completed" if run_id not in _state.active_runs else "running",
                objective=plan.objective,
                created_at=cp.ts if hasattr(cp, "ts") else 0.0,
                updated_at=cp.ts if hasattr(cp, "ts") else 0.0,
                plan=plan,
            ).model_dump()

    # Check for errors
    error_message = _state.run_errors.get(run_id)
    if error_message:
        from ..core.packets import RunStateSnapshot
        return RunStateSnapshot(
            run_id=run_id,
            status="failed",
            error=error_message,
            created_at=0.0,
            updated_at=0.0,
        ).model_dump()

    # Check for cancelled
    cancelled_message = _state.cancelled_runs.get(run_id)
    if cancelled_message:
        from ..core.packets import RunStateSnapshot
        return RunStateSnapshot(
            run_id=run_id,
            status="cancelled",
            error=cancelled_message,
            created_at=0.0,
            updated_at=0.0,
        ).model_dump()

    # Try LangGraph thread
    thread = _get_langgraph_checkpointer().read_thread(run_id)
    if thread is not None:
        from ..core.packets import RunStateSnapshot
        status = "completed"
        if thread.get("failed"):
            status = "failed"
        elif thread.get("paused"):
            status = "pending_approval"
        return RunStateSnapshot(
            run_id=run_id,
            status=status,
            created_at=0.0,
            updated_at=0.0,
        ).model_dump()

    raise OrchaError("run_not_found", f"Run {run_id} not found", 404)


@app.get("/v1/run/{run_id}/execution-events")
async def stream_execution_events(run_id: str):
    """
    SSE stream of structured ExecutionEvents for a run.

    Events are frontend-safe, sequentially numbered, and include:
    - run_id, seq, kind, ts, summary
    - task_id, task_title, task_status
    - plan_progress, plan_step_count, plan_completed_count
    - verification_status, error, metadata

    For completed runs, replays all historical events.
    For live runs, streams events as they happen with cursor-based reconnect.

    Ends with a "run_end" event containing the final status.
    """
    from ..core.packets import RunStateSnapshot, ExecutionEvent

    # Check for completed run with snapshot in checkpoint
    store = _state.store or MemoryStore()
    cp = await store.load_checkpoint(run_id)
    if cp is not None and run_id not in _state.active_runs:
        packet = cp.packet
        snapshot_data = packet.payload.get("_event_stream_snapshot")
        if snapshot_data is not None:
            snapshot = RunStateSnapshot(**snapshot_data) if isinstance(snapshot_data, dict) else snapshot_data
            events = snapshot.events if hasattr(snapshot, "events") else []

            async def _replay_events():
                for event in events:
                    event_data = event.model_dump() if hasattr(event, "model_dump") else event
                    yield f"data: {json.dumps(event_data)}\n\n"
                    await asyncio.sleep(0)
                # Emit terminal event
                end_event = {
                    "run_id": run_id,
                    "kind": "run_end",
                    "status": snapshot.status if hasattr(snapshot, "status") else "completed",
                    "objective_satisfied": (
                        snapshot.verification_result.objective_satisfied
                        if hasattr(snapshot, "verification_result") and snapshot.verification_result is not None
                        else None
                    ),
                }
                yield f"data: {json.dumps(end_event)}\n\n"
            return StreamingResponse(_replay_events(), media_type="text/event-stream")

    # If no snapshot, try to get events from the live runtime
    if run_id in _state.active_runs:
        runtime = _state.run_runtimes.get(run_id)
        if runtime is not None:
            live_emitter = getattr(runtime, "live_emitter", None)
            if live_emitter is not None:
                # Subscribe to live events and stream them
                queue: asyncio.Queue = asyncio.Queue()
                stop = asyncio.Event()
                last_seq_sent = 0

                async def _sub(event: RunEvent):
                    nonlocal last_seq_sent
                    if stop.is_set():
                        return
                    # Every node transition's packet carries the same
                    # EventStreamAdapter instance task_executor.py feeds via
                    # emit_tool_started/emit_tool_completed/etc — real,
                    # frontend-safe structured events (task_started,
                    # tool_completed, ...), not the raw engine internals this
                    # loop otherwise sees. Prefer those when present so the
                    # UI gets the same vocabulary it already renders well
                    # instead of raw node_start/checkpoint/node_end/run_complete
                    # kinds it has no mapping for.
                    packet = getattr(event, "packet", None)
                    adapter = (
                        packet.payload.get("_event_adapter")
                        if packet is not None and hasattr(packet, "payload")
                        else None
                    )
                    if adapter is not None:
                        for structured in adapter.get_events():
                            if structured.seq <= last_seq_sent:
                                continue
                            last_seq_sent = structured.seq
                            event_dict = (
                                structured.model_dump()
                                if hasattr(structured, "model_dump")
                                else structured
                            )
                            event_dict.setdefault("run_id", run_id)
                            await queue.put(event_dict)
                        return
                    # No adapter on this packet (graph type that doesn't
                    # route through task_executor) — fall back to raw
                    # forwarding so the panel isn't left completely empty.
                    event_dict = {
                        "run_id": run_id,
                        "kind": event.kind,
                        "node": event.node,
                        "ts": event.ts,
                        "data": event.data if hasattr(event, "data") else {},
                    }
                    await queue.put(event_dict)

                unsub = live_emitter.subscribe(_sub)

                async def _live_stream():
                    try:
                        while not stop.is_set():
                            while True:
                                try:
                                    ev = queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    break
                                yield f"data: {json.dumps(ev)}\n\n"
                            if run_id not in _state.active_runs:
                                # Run completed, emit end event
                                end_event = {"run_id": run_id, "kind": "run_end", "status": "completed"}
                                yield f"data: {json.dumps(end_event)}\n\n"
                                break
                            await asyncio.sleep(0.05)
                    finally:
                        unsub()
                        stop.set()

                return StreamingResponse(_live_stream(), media_type="text/event-stream")

    # Fallback: no events available
    async def _empty_stream():
        end_event = {"run_id": run_id, "kind": "run_end", "status": "unknown"}
        yield f"data: {json.dumps(end_event)}\n\n"
    return StreamingResponse(_empty_stream(), media_type="text/event-stream")


# ── Deprecated un-versioned routes -> redirect to /v1 (back-compat) ────────────
_DEPRECATED = {
    "/health":      "/v1/health",
    "/experts":     "/v1/experts",
    "/performance": "/v1/performance",
}
# POST routes get a 307 so the method/body are preserved; GET get a 301.
for _old, _new in _DEPRECATED.items():
    @app.get(_old, include_in_schema=False)
    async def _redirect_get(request: Request, _new: str = _new) -> RedirectResponse:
        return RedirectResponse(url=_new, status_code=301)
@app.post("/query", include_in_schema=False)
async def _redirect_query(req: QueryRequest):
    return RedirectResponse(url="/v1/query", status_code=307)
@app.post("/reload", include_in_schema=False)
async def _redirect_reload():
    return RedirectResponse(url="/v1/reload", status_code=307)
# ── Agent Runtime endpoints (/v1/agent-runs …) ───────────────────────────────
from .agent_stream import router as agent_stream_router
app.include_router(agent_stream_router)
# ── Serve UI ─────────────────────────────────────────────────────────────────
_ui_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui"
)
if os.path.isdir(_ui_dir):
    app.mount("/", StaticFiles(directory=_ui_dir, html=True), name="ui")
