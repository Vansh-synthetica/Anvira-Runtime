"""
orcha.agent_runtime.diagnostics
===============================
Developer-grade, READ-ONLY observability for agent runs.

This module deliberately changes NO runtime behavior:

- it never influences agent decisions, conversation state, replay,
  EventLog semantics, memory or budgeting;
- it extends the EXISTING ``orcha`` logging tree (see
  ``orcha.observability``) — a capture handler is attached to the
  ``orcha`` logger for the duration of a run, so every structured log
  line the runtime already emits (plus the ones added at previously
  silent stages) lands in one per-run DiagnosticSession;
- nothing here can raise into the runtime: capture, redaction and
  storage failures are swallowed (with a warning when possible).

One run → one DiagnosticSession (in-memory, capped) → optional rolling
disk JSONL archive → plain-text report for copy/paste.

Privacy: reports are redacted before they leave the process — API keys,
authorization headers, cookies, tokens, passwords and sensitive env
values become ``***`` while the surrounding context is preserved.
"""
from __future__ import annotations

import asyncio
import contextlib
import datetime as _dt
import json
import logging
import os
import platform
import re
import sys
import threading
import traceback as _traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

logger = logging.getLogger("orcha.agent_runtime.diagnostics")


def _runtime_version() -> str:
    """orcha.__version__ — imported lazily (orcha/__init__ imports this
    package while still executing)."""
    try:
        from .. import __version__  # noqa: PLC0415

        return __version__
    except Exception:
        return "unknown"

# The existing logger tree everything propagates through (see observability.py).
_LOGGER_NAME = "orcha"

_STANDARD_RECORD_ATTRS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
})


def diag_log(
    logger: logging.Logger,
    stage: str,
    event: str,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """
    One structured observability line through the EXISTING logger tree.
    The message is the event name; fields travel as ``extra`` (captured by
    DiagnosticSession handlers and JSON formatters). Never raises.
    """
    extra: Dict[str, Any] = {"stage": stage}
    for key, value in fields.items():
        if key in _STANDARD_RECORD_ATTRS or not key.isidentifier():
            key = f"diag_{key}"
        extra[key] = value
    try:
        logger.log(level, "%s.%s", stage, event, extra=extra)
    except Exception:  # observability must never break the runtime
        pass


def diag_log_exc(
    logger: logging.Logger,
    stage: str,
    event: str,
    exc: BaseException,
    **fields: Any,
) -> None:
    """Log an exception with its traceback intact (type, message, chain)."""
    extra: Dict[str, Any] = {
        "stage": stage,
        "exception_type": type(exc).__name__,
        "exception_message": getattr(exc, "message", None) or str(exc),
    }
    for key, value in fields.items():
        if key in _STANDARD_RECORD_ATTRS or not key.isidentifier():
            key = f"diag_{key}"
        extra[key] = value
    try:
        logger.error(
            "%s.%s", stage, event,
            exc_info=(type(exc), exc, exc.__traceback__),
            extra=extra,
        )
    except Exception:
        pass

_MAX_ENTRIES_PER_SESSION = 5_000
_MAX_SESSIONS_IN_MEMORY = 32
_DISK_MAX_BYTES = 10 * 1024 * 1024
_DISK_ROTATION_COUNT = 3

# ── Redaction ─────────────────────────────────────────────────────────────────

_SENSITIVE_KEY = re.compile(
    r"(api[_-]?key|authorization|auth|cookie|password|passwd|secret|"
    r"token|credential|session[_-]?id)", re.IGNORECASE,
)
_HEADER_LINE = re.compile(
    r"(authorization|set-cookie|cookie|x-api-key)\s*:\s*.+", re.IGNORECASE,
)
_URL_SECRET_PARAM = re.compile(
    r"((?:api[_-]?key|token|password|secret|auth)=)[^&\s\"']+", re.IGNORECASE,
)
_BEARER_OR_KEY = re.compile(
    r"(sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9]{20,}|xox[bap]-[A-Za-z0-9-]{8,}"
    r"|Bearer\s+[A-Za-z0-9._~+/=-]{8,})",
    re.IGNORECASE,
)
_MASK = "***"


def redact_value(key: str, value: Any) -> Any:
    """Mask a single (key, value) pair when the key or value is sensitive.
    ``***`` replaces the value; the key is preserved so context survives."""
    if _SENSITIVE_KEY.search(key):
        return _MASK
    if isinstance(value, str):
        if _HEADER_LINE.match(value.strip()):
            return _MASK
        value = _URL_SECRET_PARAM.sub(r"\1" + _MASK, value)
        value = _BEARER_OR_KEY.sub(_MASK, value)
    return value


def redact_payload(payload: Any) -> Any:
    """Recursively redact a JSON-safe structure in place of a copy."""
    if isinstance(payload, dict):
        return {
            k: redact_payload(redact_value(str(k), v)) if not _SENSITIVE_KEY.search(str(k))
            else _MASK
            for k, v in payload.items()
        }
    if isinstance(payload, (list, tuple)):
        return [redact_payload(v) for v in payload]
    if isinstance(payload, str):
        return redact_value("", payload)
    return payload


def redact_text(text: str) -> str:
    """Mask secrets that appear inside free text (URL params, keys, headers)."""
    text = _URL_SECRET_PARAM.sub(r"\1" + _MASK, text)
    text = _BEARER_OR_KEY.sub(_MASK, text)
    return _HEADER_LINE.sub(_MASK, text)


# ── The per-run session ───────────────────────────────────────────────────────

@dataclass
class DiagEntry:
    """One structured log line in a diagnostic session."""
    ts: float
    level: str
    stage: str
    event: str
    fields: Dict[str, Any] = field(default_factory=dict)
    traceback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": round(self.ts, 4),
            "level": self.level,
            "stage": self.stage,
            "event": self.event,
            "fields": self.fields,
            "traceback": self.traceback,
        }


class _CaptureHandler(logging.Handler):
    """Routes records from the existing ``orcha`` logger tree into a
    DiagnosticSession. Read-only: never propagates, never raises."""

    def __init__(self, session: "DiagnosticSession") -> None:
        super().__init__(level=logging.NOTSET)
        self._session = session

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - thin
        try:
            if not record.name.startswith(_LOGGER_NAME):
                return
            fields: Dict[str, Any] = {}
            standard = {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "taskName",
                "message", "asctime",
            }
            for key, value in record.__dict__.items():
                if key not in standard:
                    fields[key] = value
            tb = None
            if record.exc_info and record.exc_info[0] is not None:
                tb = "".join(_traceback.format_exception(*record.exc_info))
            stage = getattr(record, "stage", None) or record.name
            event = record.getMessage()
            if stage and event.startswith(stage + "."):
                event = event[len(stage) + 1:]  # console line embeds stage.event
            self._session._capture(
                stage=stage,
                event=event,
                level=record.levelname.lower(),
                fields=fields,
                traceback=tb,
            )
        except Exception:  # observability must never break the runtime
            pass


class DiagnosticSession:
    """
    One structured, read-only log for one agent run.

    - ``scope()`` (async context manager) installs the capture handler on
      the existing ``orcha`` logger tree for the duration of the run, then
      records start/finish with duration and outcome.
    - ``record`` / ``record_exception`` add direct entries at stages that
      do not already log.
    - ``to_dict`` / ``render_report`` produce the redacted, complete view.
    - ``outcome`` is ``"success"`` or ``"failed"`` (never influences the
      run itself).
    """

    def __init__(
        self,
        session_id: str,
        *,
        run_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        max_entries: int = _MAX_ENTRIES_PER_SESSION,
    ) -> None:
        self.session_id = session_id
        self.run_id = run_id or session_id
        self.meta: Dict[str, Any] = redact_payload(dict(meta or {}))
        self.created_at = _dt.datetime.now().isoformat(timespec="milliseconds")
        self.finished_at: Optional[str] = None
        self.duration_ms: Optional[float] = None
        self.outcome: str = "running"      # running | success | failed
        self.terminated_by: Optional[str] = None
        self.error_summary: Optional[str] = None
        self._max_entries = max(100, int(max_entries))
        self._entries: List[DiagEntry] = []
        self._truncated = False
        self._lock = threading.Lock()
        self._handler: Optional[_CaptureHandler] = None
        self._scope_depth = 0
        self._prev_logger_level = logging.NOTSET

    # ── capture ─────────────────────────────────────────────────────────

    def _capture(
        self, *, stage: str, event: str, level: str = "info",
        fields: Optional[Dict[str, Any]] = None, traceback: Optional[str] = None,
    ) -> None:
        try:
            import time as _time
            entry = DiagEntry(
                ts=_time.time(),
                level=level,
                stage=stage,
                event=event,
                fields=redact_payload(dict(fields or {})),
                traceback=redact_text(traceback) if traceback else None,
            )
            with self._lock:
                if len(self._entries) >= self._max_entries:
                    self._entries.pop(0)
                    self._truncated = True
                self._entries.append(entry)
        except Exception:
            pass

    def record(
        self, stage: str, event: str, level: str = "info", **fields: Any,
    ) -> None:
        """Add one structured entry directly (stages without a logger)."""
        self._capture(stage=stage, event=event, level=level, fields=fields)

    def record_exception(
        self,
        stage: str,
        exc: BaseException,
        *,
        api_status: Optional[int] = None,
        trace_id: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        """Record an exception WITHOUT collapsing it: type, message,
        full traceback, nested cause chain and API status survive. The
        frontend-safe message stays the caller's concern."""
        cause = exc
        chain: List[str] = []
        seen = 0
        while cause is not None and seen < 8:
            chain.append(
                f"{type(cause).__name__}: {getattr(cause, 'message', None) or str(cause)}"
            )
            cause = getattr(cause, "__cause__", None) or getattr(cause, "__context__", None)
            seen += 1
        tb = "".join(
            _traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        fields: Dict[str, Any] = {
            "exception_type": type(exc).__name__,
            "message": message or (getattr(exc, "message", None) or str(exc)),
            "cause_chain": chain,
        }
        if api_status is not None:
            fields["api_status"] = api_status
        if trace_id:
            fields["trace_id"] = trace_id
        self._capture(stage=stage, event="exception", level="error",
                      fields=fields, traceback=tb)

    def record_attachment(self, path: str) -> None:
        """Validate one attached file/dir; failures are recorded as
        ``attachment.failed`` entries (never raised)."""
        try:
            if os.path.isdir(path) or os.path.isfile(path):
                self.record("attachment", "attached", path=path)
            else:
                self.record(
                    "attachment", "failed", level="warning",
                    path=path, reason="path does not exist",
                )
        except Exception as exc:  # pragma: no cover
            self.record_exception("attachment", exc, message="attachment check failed")

    # ── scope ───────────────────────────────────────────────────────────

    @contextlib.asynccontextmanager
    async def scope(self) -> Iterator["DiagnosticSession"]:
        """Install capture for the duration of the run (nestable)."""
        if self._scope_depth == 0:
            # The capture handler needs records to flow through the
            # existing "orcha" logger tree: raise its level to INFO if the
            # host app has not configured it (default is WARNING), and
            # restore the previous level afterwards. This only changes
            # logging verbosity, never runtime behavior.
            orcha_logger = logging.getLogger(_LOGGER_NAME)
            prev_level = orcha_logger.level
            if prev_level == logging.NOTSET or prev_level > logging.INFO:
                orcha_logger.setLevel(logging.INFO)
            self._prev_logger_level = prev_level
            self._handler = _CaptureHandler(self)
            orcha_logger.addHandler(self._handler)
            self.record("session", "start", session=self.session_id,
                        outcome=self.outcome)
        self._scope_depth += 1
        try:
            yield self
        finally:
            self._scope_depth -= 1
            if self._scope_depth == 0:
                self._finish()
                orcha_logger = logging.getLogger(_LOGGER_NAME)
                if self._handler is not None:
                    orcha_logger.removeHandler(self._handler)
                    self._handler = None
                orcha_logger.setLevel(self._prev_logger_level)

    def finish(self) -> None:
        """Finalize this session manually (for run paths that are not wrapped
        in :meth:`scope`, e.g. the ``/v1/run`` graph runtime). Read-only."""
        self._finish()

    def _finish(self) -> None:
        try:
            self.finished_at = _dt.datetime.now().isoformat(timespec="milliseconds")
            self.duration_ms = self._duration_ms()
            if self.outcome == "running":
                self.outcome = "failed" if self._has_errors() else "success"
            self.record(
                "session", "finish", outcome=self.outcome,
                duration_ms=self.duration_ms, terminated_by=self.terminated_by,
                entries=len(self._entries), truncated=self._truncated,
            )
        finally:
            pass

    def _duration_ms(self) -> float:
        start = _dt.datetime.fromisoformat(self.created_at)
        end = _dt.datetime.fromisoformat(self.finished_at or _dt.datetime.now().isoformat())
        return round((end - start).total_seconds() * 1000, 2)

    def _has_errors(self) -> bool:
        return any(e.level in ("error", "critical") for e in self._entries)

    # ── views ───────────────────────────────────────────────────────────

    @property
    def entries(self) -> List[DiagEntry]:
        with self._lock:
            return list(self._entries)

    @property
    def truncated(self) -> bool:
        return self._truncated

    def errors(self) -> List[DiagEntry]:
        return [e for e in self.entries if e.level in ("error", "critical")]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "outcome": self.outcome,
            "terminated_by": self.terminated_by,
            "error_summary": self.error_summary,
            "truncated": self._truncated,
            "meta": redact_payload(self.meta),
            "entries": [e.to_dict() for e in self.entries],
        }


# ── Storage ───────────────────────────────────────────────────────────────────

class DiagnosticsStore:
    """
    In-memory ring of DiagnosticSessions (bounded — no unbounded growth),
    with an OPTIONAL rolling JSONL archive on disk (size-capped rotation).

    ``clear()`` removes ONLY stored diagnostics — nothing else.
    """

    def __init__(
        self,
        max_sessions: int = _MAX_SESSIONS_IN_MEMORY,
        disk_dir: Optional[str] = None,
    ) -> None:
        self._sessions: Dict[str, DiagnosticSession] = {}
        self._order: List[str] = []
        self._max_sessions = max(1, int(max_sessions))
        self._disk_dir = disk_dir
        self._lock = threading.Lock()

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(
        self,
        session_id: str,
        *,
        run_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticSession:
        session = DiagnosticSession(session_id, run_id=run_id, meta=meta)
        with self._lock:
            if session_id in self._sessions:
                self._order.remove(session_id)
            self._sessions[session_id] = session
            self._order.append(session_id)
            while len(self._order) > self._max_sessions:
                oldest = self._order.pop(0)
                self._sessions.pop(oldest, None)
        return session

    def get(self, session_id: str) -> Optional[DiagnosticSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def list(self) -> List[Dict[str, Any]]:
        """Summaries newest-first for the diagnostics UI."""
        with self._lock:
            sessions = [self._sessions[sid] for sid in reversed(self._order)]
        return [self._summary(s) for s in sessions]

    def finalize(self, session: DiagnosticSession) -> None:
        """Persist a finished session to the optional rolling disk log."""
        if not self._disk_dir:
            return
        try:
            os.makedirs(self._disk_dir, exist_ok=True)
            path = os.path.join(self._disk_dir, "diagnostics.jsonl")
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(session.to_dict(), default=str) + "\n")
            if os.path.getsize(path) > _DISK_MAX_BYTES:
                self._rotate(path)
        except Exception:  # disk logging is best-effort
            logger.warning("diagnostics disk write failed", exc_info=True)

    def _rotate(self, path: str) -> None:
        try:
            for i in range(_DISK_ROTATION_COUNT, 1, -1):
                prev = f"{path}.{i - 1}"
                cur = f"{path}.{i}"
                if os.path.exists(prev):
                    os.replace(prev, cur)
            os.replace(path, f"{path}.1")
        except Exception:
            pass

    def clear(self) -> int:
        """Remove ONLY stored diagnostics; return how many were removed."""
        with self._lock:
            n = len(self._sessions)
            self._sessions.clear()
            self._order.clear()
        return n

    @staticmethod
    def _summary(session: DiagnosticSession) -> Dict[str, Any]:
        meta = session.meta
        return {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "finished_at": session.finished_at,
            "duration_ms": session.duration_ms,
            "outcome": session.outcome,
            "pipeline": meta.get("pipeline"),
            "model": meta.get("model"),
            "backend": meta.get("backend"),
            "workspace": meta.get("workspace_roots") or [],
            "query": (meta.get("query") or "")[:120],
            "terminated_by": session.terminated_by,
            "error_summary": session.error_summary,
            "entries": len(session.entries),
            "truncated": session.truncated,
        }


_default_store: Optional[DiagnosticsStore] = None
_store_lock = threading.Lock()


def get_diagnostics_store(
    disk_dir: Optional[str] = None,
) -> DiagnosticsStore:
    """The shared store (tests may construct their own)."""
    global _default_store
    with _store_lock:
        if _default_store is None:
            _default_store = DiagnosticsStore(disk_dir=disk_dir)
    return _default_store


# ── The complete plain-text report ────────────────────────────────────────────

def _iso(ts: float) -> str:
    try:
        return _dt.datetime.fromtimestamp(ts).isoformat(timespec="milliseconds")
    except Exception:
        return str(ts)


def render_report(
    session: DiagnosticSession,
    events: Optional[Sequence[Any]] = None,
    *,
    max_trace_chars: int = 20_000,
) -> str:
    """
    The complete, redacted diagnostic report as plain text — designed for
    pasting into ChatGPT/Claude/GitHub issues/Discord: no markdown tables,
    no HTML, no control characters.
    """
    lines: List[str] = []
    add = lines.append
    meta = session.meta

    add("=" * 72)
    add("ORCHA AGENT DIAGNOSTIC LOG")
    add("=" * 72)
    add(f"session_id:       {session.session_id}")
    add(f"run_id:           {session.run_id}")
    add(f"created_at:       {session.created_at}")
    add(f"finished_at:      {session.finished_at or '(not finished)'}")
    add(f"duration_ms:      {session.duration_ms or '(running)'}")
    add(f"outcome:          {session.outcome}")
    add(f"terminated_by:    {session.terminated_by or '-'}")
    add(f"runtime_version:  {meta.get('runtime_version') or _runtime_version()}")
    add(f"backend:          {meta.get('backend') or '-'}")
    add(f"model:            {meta.get('model') or '-'}")
    add(f"temperature:      {meta.get('temperature') or 'default'}")
    add(f"streaming:        {meta.get('streaming') or 'default'}")
    add(f"conversation_id:  {meta.get('conversation_id') or session.run_id}")
    workspace = meta.get("workspace_roots") or []
    add(f"workspace:        {'; '.join(workspace) if workspace else '(none)'}")
    attached = meta.get("attached_files") or []
    add(f"attached_files:   {'; '.join(str(a) for a in attached) if attached else '(none)'}")
    env = meta.get("environment") or {}
    for key in sorted(env):
        add(f"env.{key}:         {env[key]}")

    add("")
    add("EVENT LOG")
    add("-" * 72)
    if not events:
        # None for graph-runtime (/v1/run) sessions, which have no event
        # log of their own — the diagnostics entries above already cover
        # them. An empty list means a real session with nothing recorded.
        add("(no events)")
    else:
        for ev in events:
            payload = getattr(ev, "payload", None)
            content = getattr(payload, "content", None)
            if isinstance(content, str) and len(content) > 200:
                content = content[:200] + "..."
            add(
                f"seq={ev.seq:<4} {ev.kind.value:<12} "
                f"type={getattr(payload, 'kind', '?'):<14} "
                f"tokens={ev.tokens} reasoning={ev.reasoning_tokens}"
                + (f"  {content}" if content else "")
            )

    add("")
    add("STAGES (chronological)")
    add("-" * 72)
    for entry in session.entries:
        fields = entry.fields
        suffix = ""
        if fields:
            parts = []
            for k, v in sorted(fields.items()):
                if isinstance(v, str) and len(v) > 300:
                    v = v[:300] + "..."
                parts.append(f"{k}={v}")
            suffix = "  " + " ".join(parts)
        add(f"[{_iso(entry.ts)}] {entry.level:<7} {entry.stage}.{entry.event}{suffix}")
        if entry.traceback:
            tb = entry.traceback[:max_trace_chars]
            add("  --- traceback ---")
            for tb_line in tb.rstrip().splitlines():
                add("  " + tb_line)
            add("  --- end traceback ---")

    errors = session.errors()
    add("")
    add("ERRORS")
    add("-" * 72)
    if not errors:
        add("(none)")
    for entry in errors:
        fields = entry.fields
        add(
            f"[{_iso(entry.ts)}] {entry.stage}: {entry.event}"
            + (f"  ({fields})" if fields else "")
        )

    add("")
    add("END OF DIAGNOSTIC LOG")
    add("=" * 72)
    return "\n".join(lines)


# ── Run wrapper (used by the API driver and tests) ───────────────────────────

async def diagnose_run(
    session: DiagnosticSession,
    conversation: Any,
) -> "Any":
    """
    Run a conversation inside the session's capture scope and derive the
    outcome. Mirrors the API background driver; read-only with respect to
    the conversation. Returns the ConversationResult.
    """
    async with session.scope():
        try:
            result = await conversation.run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            session.outcome = "failed"
            session.record_exception("run", exc, message=str(exc))
            session.error_summary = str(exc)
            raise
        else:
            session.terminated_by = getattr(result, "terminated_by", None)
            if session.terminated_by in ("error",):
                session.outcome = "failed"
                session.error_summary = "run terminated with an error action"
            else:
                session.outcome = "success"
            return result


__all__ = [
    "DiagEntry", "DiagnosticSession", "DiagnosticsStore",
    "get_diagnostics_store", "render_report", "diagnose_run",
    "diag_log", "diag_log_exc",
    "redact_payload", "redact_text", "redact_value",
]
