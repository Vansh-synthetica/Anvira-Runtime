"""
orcha.api.agent_stream
======================
The Agent Runtime HTTP surface for Anvira: create agent-run sessions, start
them, and stream their EventLogs over SSE with cursor-based reconnect
catch-up.

Endpoints
---------
- ``POST /v1/agent-runs``                      Create a session (returns run_id).
- ``POST /v1/agent-runs/{id}/start``           Run the session's query in the
                                               background; events land in the log.
- ``GET  /v1/agent-runs/{id}``                 Status of a session.
- ``GET  /v1/agent-runs/{id}/events?cursor=N`` SSE stream of events. ``cursor``
                                               is the last seq the client has;
                                               everything strictly after it is
                                               replayed, then the stream goes
                                               live. Ends with
                                               ``{"kind": "end", ...}``.

The wire format is the thin serialization from
:mod:`orcha.agent_runtime.stream` — the Prompt-1 event model (seq, kind, ts,
tokens, reasoning_tokens, payload) plus the Prompt-4 step-memory metadata —
so Anvira stays a pure renderer with zero agent-loop logic.
"""
from __future__ import annotations

import asyncio
import json
import logging
import platform
import sys
from typing import Any, List, Optional

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing_extensions import Literal

from .. import __version__ as ORCHA_VERSION
from ..agent_runtime.agent import AgentConfig, ToolCallingAgent
from ..agent_runtime.backend import ModelBackend, ModelResponse, TokenUsage
from ..agent_runtime.backends.openai_compat import (
    DEFAULT_OLLAMA_V1,
    OpenAICompatBackend,
    OpenAICompatBackendConfig,
)
from ..agent_runtime.conversation import Conversation, sanitize_final_answer
from ..agent_runtime.diagnostics import (
    diagnose_run, get_diagnostics_store, render_report,
)
from ..agent_runtime.stream import (
    AgentEventStream,
    AgentSession,
    AgentSessionRegistry,
    event_to_wire,
)
from ..agent_runtime.tools import ToolRegistry, build_default_tools
from ..agent_runtime.workspace import ToolWorkspace

logger = logging.getLogger("orcha.api.agent_stream")

router = APIRouter()
_registry = AgentSessionRegistry()


def get_agent_session_registry() -> AgentSessionRegistry:
    """Expose the shared registry (tests fabricate sessions through it)."""
    return _registry


class AgentBackendSpec(BaseModel):
    """
    Which model backend the session's agent talks to.

    ``type`` is REQUIRED whenever ``backend`` is supplied — a live request
    can never fall through to a default. ``"stub"`` is the test/demo double
    and must be requested explicitly (no production caller uses it).
    """
    type: Literal["stub", "openai_compat"]
    base_url: Optional[str] = Field(default=None, max_length=2000)
    model: Optional[str] = Field(default=None, max_length=500)


class AgentRunCreateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=20000)
    system_prompt: Optional[str] = Field(default=None, max_length=8000)
    workspace_roots: Optional[List[str]] = Field(
        default=None,
        description=(
            "Absolute directories the agent's workspace tools are allowed to "
            "read/write/edit. When empty, no file tools are exposed."
        ),
    )
    backend: Optional[AgentBackendSpec] = Field(
        default=None,
        description=(
            "Explicit model backend. When omitted, the server resolves the "
            "configured default (the primary local model Anvira activated); "
            "if none is configured the request fails with a typed error. "
            "It can NEVER fall through to the stub test double."
        ),
    )


class AgentRunResponse(BaseModel):
    run_id: str
    status: str = "created"          # created | started | finished
    last_seq: int = 0
    error: Optional[str] = None
    answer: Optional[str] = None     # final NL answer once the run finishes


class _StubBackend(ModelBackend):
    """Deterministic canned backend for demos/tests — never touches a model."""

    @property
    def model_name(self) -> str:
        return "stub"

    @property
    def supports_streaming_tool_calls(self) -> bool:
        return False

    @property
    def reports_reasoning_tokens(self) -> bool:
        return False

    def complete(
        self,
        messages: Any,
        tools: Optional[List[Any]] = None,
        config: Any = None,
    ) -> ModelResponse:
        return ModelResponse(
            content="[stub] canned reply",
            usage=TokenUsage(prompt_tokens=4, completion_tokens=7),
            finish_reason="stop",
        )


def _build_backend(spec: AgentBackendSpec) -> ModelBackend:
    """
    Build a backend from an EXPLICIT request spec. ``"stub"`` is honored
    only here — a client that explicitly asks for the test double gets it
    (tests/demos); anything else must name a real backend.
    """
    if spec.type == "openai_compat":
        return OpenAICompatBackend(
            OpenAICompatBackendConfig(
                base_url=spec.base_url or DEFAULT_OLLAMA_V1,
                model=spec.model or "llama3.2:3b",
            )
        )
    if spec.type == "stub":
        return _StubBackend()
    _raise_run_error(
        "backend_not_supported",
        f"Unsupported backend type {spec.type!r}. Supported: "
        f"'stub' (test double, explicit only), 'openai_compat'.",
        400,
    )


def _default_backend() -> ModelBackend:
    """
    The REAL configured backend for live requests with no explicit
    ``backend`` in the body: the primary local model Anvira activated
    (same ``orcha.settings.settings`` singleton the /v1/run orchestrator
    is built from), else the first concurrently-registered extra model.
    If nothing is configured, raise a typed OrchaError — never a stub.
    """
    from ..settings import settings  # lazy: same module the orchestrator reads

    if settings.local_server_model:
        return OpenAICompatBackend(
            OpenAICompatBackendConfig(
                base_url=settings.local_server_base_url,
                model=settings.local_server_model,
                api_key=settings.local_server_api_key or None,
            )
        )
    if settings.local_server_extra_models:
        extra = settings.local_server_extra_models[0]
        return OpenAICompatBackend(
            OpenAICompatBackendConfig(
                base_url=extra["base_url"],
                model=extra["model"],
                api_key=extra.get("api_key") or None,
            )
        )
    _raise_run_error(
        "no_model_configured",
        "No local model is configured. Activate a model in Anvira's Model "
        "Library first (or pass an explicit backend in the request).",
        400,
    )


def _backend_label(backend: ModelBackend) -> str:
    """The wire/observability label for a resolved backend."""
    if isinstance(backend, _StubBackend):
        return "stub"
    return "openai_compat"


def _raise_run_error(type_: str, message: str, status: int) -> None:
    # Lazy import: server.py owns the one shared OrchaError + envelope.
    from .server import OrchaError

    raise OrchaError(type_, message, status)


def _runtime_meta(req: AgentRunCreateRequest, backend: ModelBackend) -> dict:
    """Observability metadata for a diagnostic session (redacted at render)."""
    return {
        "pipeline": "agent_runtime",
        "runtime_version": ORCHA_VERSION,
        "backend": _backend_label(backend),
        "model": backend.model_name,
        "streaming": backend.supports_streaming_tool_calls,
        "workspace_roots": list(req.workspace_roots or []),
        "attached_files": list(req.workspace_roots or []),
        "conversation_id": None,  # filled after the session is created
        "query": req.query[:2000],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "orcha_version": ORCHA_VERSION,
        },
    }


async def _drive_session(session: AgentSession) -> None:
    """Background driver: run the conversation, then signal end-of-stream."""
    store = get_diagnostics_store()
    diag = session.diagnostics
    try:
        if diag is not None:
            result = await diagnose_run(diag, session.conversation)
        else:
            result = await session.conversation.run()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("agent run %s failed", session.session_id)
        if diag is not None:
            diag.outcome = "failed"
            diag.error_summary = str(exc)
            diag.record_exception("api", exc, message=str(exc))
            store.finalize(diag)
        session.notify_error(f"{type(exc).__name__}: {exc}")
    else:
        terminated_by = getattr(result, "terminated_by", "finish")
        answer = sanitize_final_answer(getattr(result, "answer", None))
        if terminated_by == "error" or answer is None:
            session.notify_error(
                "agent terminated without a final answer (see diagnostics)"
                if answer is None
                else "agent terminated with an error (see diagnostics)"
            )
        else:
            session.notify_finished()
        if diag is not None:
            store.finalize(diag)


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/v1/agent-runs", response_model=AgentRunResponse)
async def create_agent_run(req: AgentRunCreateRequest):
    """Create a session: agent + workspace over the chosen backend."""
    registry = ToolRegistry()
    if req.workspace_roots:
        for tool in build_default_tools(roots=req.workspace_roots):
            registry.register(tool)

    backend = (
        _build_backend(req.backend)
        if req.backend is not None
        else _default_backend()
    )
    agent = ToolCallingAgent(backend=backend, tools=registry)
    workspace = ToolWorkspace(registry)
    agent_config = None
    if req.system_prompt and req.system_prompt.strip():
        agent_config = AgentConfig(system_prompt=req.system_prompt.strip())
    convo = Conversation(
        agent,
        workspace,
        agent_config=agent_config,
        query=req.query,
    )
    session = _registry.create(convo, query=req.query)

    # Prompt 7: read-only diagnostics for this run.
    meta = _runtime_meta(req, backend)
    meta["conversation_id"] = session.session_id
    diag = get_diagnostics_store().start(
        session.session_id, run_id=session.session_id, meta=meta,
    )
    session.diagnostics = diag
    for root in req.workspace_roots or []:
        diag.record_attachment(root)

    return AgentRunResponse(run_id=session.session_id, status="created", last_seq=0)


@router.post("/v1/agent-runs/{session_id}/start", response_model=AgentRunResponse)
async def start_agent_run(session_id: str):
    """Submit the session's query and run the loop in the background."""
    session = _registry.get(session_id)
    if session is None:
        _raise_run_error("run_not_found", f"Unknown agent run {session_id}", 404)
    if session.started:
        _raise_run_error("already_started", "This agent run has already been started", 409)

    session.started = True
    session.conversation.submit_user_message(session.query)
    asyncio.get_running_loop().create_task(_drive_session(session))
    return AgentRunResponse(run_id=session_id, status="started", last_seq=session.last_seq)


@router.get("/v1/agent-runs/{session_id}", response_model=AgentRunResponse)
async def get_agent_run(session_id: str):
    session = _registry.get(session_id)
    if session is None:
        _raise_run_error("run_not_found", f"Unknown agent run {session_id}", 404)
    status = "finished" if session.finished else ("started" if session.started else "created")
    return AgentRunResponse(
        run_id=session_id,
        status=status,
        last_seq=session.last_seq,
        error=session.error,
        answer=session.conversation.answer if session.finished else None,
    )


@router.get("/v1/agent-runs/{session_id}/events")
async def stream_agent_run_events(
    session_id: str,
    cursor: int = Query(default=0, ge=0),
):
    """
    SSE stream of the session's events.

    Replays every event strictly after ``cursor`` (reconnect catch-up), then
    delivers live. Ends with ``{"kind": "end", "seq": ..., "error": ...}``.
    """
    session = _registry.get(session_id)
    if session is None:
        _raise_run_error("run_not_found", f"Unknown agent run {session_id}", 404)

    stream = AgentEventStream(session.log, cursor=cursor)
    session.attach(stream)

    async def _stream():
        try:
            while True:
                ev = await stream.next()
                if ev is None:
                    break
                wire = event_to_wire(ev, session.step_records())
                yield f"data: {json.dumps(wire)}\n\n"
            yield f"data: {json.dumps({'kind': 'end', 'seq': session.last_seq, 'error': session.error, 'answer': session.conversation.answer if session.finished else None})}\n\n"
        finally:
            session.detach(stream)

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ── Diagnostics (Prompt 7: read-only observability) ───────────────────────────

@router.get("/v1/agent-runs/{session_id}/diagnostics")
async def get_run_diagnostics(
    session_id: str,
    format: Literal["json", "text"] = Query(default="json"),
):
    """
    The full diagnostic session for one run.

    ``format=json`` returns the structured session (meta + entries);
    ``format=text`` returns the complete plain-text report — the
    developer-copyable artifact (secrets redacted server-side).
    """
    session = _registry.get(session_id)
    diag = getattr(session, "diagnostics", None) if session is not None else None
    if diag is None:
        # Graph-runtime (/v1/run) sessions live in the same store — the
        # viewer can open them through this same endpoint.
        diag = get_diagnostics_store().get(session_id)
    if diag is None:
        if session is None:
            _raise_run_error("run_not_found", f"Unknown agent run {session_id}", 404)
        _raise_run_error(
            "diagnostics_not_found",
            f"No diagnostics recorded for agent run {session_id}",
            404,
        )
    if format == "text":
        return PlainTextResponse(
            render_report(diag, session.log.events if session is not None else None),
            media_type="text/plain; charset=utf-8",
        )
    return {
        **diag.to_dict(),
        "events": (
            [event_to_wire(ev, None) for ev in session.log.events]
            if session is not None else []
        ),
    }


@router.get("/v1/diagnostics")
async def list_diagnostics():
    """Recent diagnostic runs, newest first (summaries only)."""
    return {"runs": get_diagnostics_store().list()}


class DiagnosticsClearResponse(BaseModel):
    runs_cleared: int
    note: str = (
        "Only stored diagnostics were removed; EventLogs, conversations, "
        "memory, workspaces, history, models and settings are untouched."
    )


@router.delete("/v1/diagnostics", response_model=DiagnosticsClearResponse)
async def clear_diagnostics():
    """Clear ONLY stored diagnostics."""
    cleared = get_diagnostics_store().clear()
    return DiagnosticsClearResponse(runs_cleared=cleared)
