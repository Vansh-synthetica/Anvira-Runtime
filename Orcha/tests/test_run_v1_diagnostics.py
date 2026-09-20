"""
Prompt 11 tests: /v1/run (GraphRuntime) diagnostics parity with the
/v1/agent-runs pipeline — same store, same viewer, read-only.

Contract
--------
(a) Every POST /v1/run creates a diagnostic session in the SAME store the
    agent-runs pipeline uses; GET /v1/diagnostics lists both, tagged
    pipeline=graph_runtime vs agent_runtime.
(b) The session records: session start, request received (including whether
    an attachment/workspace prompt part was present in the payload and its
    resolved text), the assembled prompt/attachment context, each model call
    and its response, and a finish/error entry.
(c) The diagnostics detail endpoint resolves a graph-run session too (store
    fallback), so the existing viewer opens /v1/run runs unchanged.
(d) Instrumentation is read-only: run behavior/results are unchanged.
"""
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from orcha.api.server import app
from orcha.settings import settings

MARKER = "ATTACHMENT_MARKER_7f3a91c2"

ATTACHMENT_TEXT = (
    "Attached file: README.md\n"
    "Path: C:\\fabricated\\project\\README.md\n"
    "Size: 1.2 KB\n"
    f"Marker: {MARKER}\n"
    "Content preview: '# Fabricated project' — an example file attached by the user."
)


@pytest.fixture
def configured_model():
    """Configure a fabricated 'activated' local model BEFORE the server
    starts, so the orchestrator (and thus the agent completion_fn) sees it
    — the same sequence the real UI follows."""
    prev_model = settings.local_server_model
    prev_base = settings.local_server_base_url
    prev_extras = list(settings.local_server_extra_models)
    # The "123b" size pattern guarantees this fabricated model wins
    # synthesizer selection (pick_synthesizer = largest by param size) even
    # when a real Ollama daemon is running, so completion_fn is always built
    # from the unreachable local server and the test stays deterministic.
    settings.local_server_model = "Fabricated-Model-123b"
    settings.local_server_base_url = "http://127.0.0.1:1/v1"  # guaranteed unreachable
    settings.local_server_extra_models = []
    try:
        yield
    finally:
        settings.local_server_model = prev_model
        settings.local_server_base_url = prev_base
        settings.local_server_extra_models = prev_extras


@pytest.fixture
def no_configured_model():
    """Force the 'nothing configured' state before the server starts."""
    prev_model = settings.local_server_model
    prev_base = settings.local_server_base_url
    prev_extras = list(settings.local_server_extra_models)
    settings.local_server_model = ""
    settings.local_server_base_url = "http://127.0.0.1:8080/v1"
    settings.local_server_extra_models = []
    try:
        yield
    finally:
        settings.local_server_model = prev_model
        settings.local_server_base_url = prev_base
        settings.local_server_extra_models = prev_extras


@pytest.fixture
def client(configured_model):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def no_model_client(no_configured_model):
    with TestClient(app) as c:
        yield c


def _unique_query(tag: str) -> str:
    return f"{tag} {uuid.uuid4().hex[:8]}"


def _run_payload(query: str, *, stream: bool = False) -> dict:
    return {
        "query": query,
        "graph": "default",
        "capabilities": ["filesystem"],
        "workspace_roots": ["C:\\fabricated\\project"],
        "prompt_parts": [
            {"name": "persona", "text": "You are a careful assistant.", "priority": 9},
            {"name": "instructions", "text": "Call the appropriate tool directly.", "priority": 5},
            {"name": "workspace", "text": ATTACHMENT_TEXT, "priority": 3},
        ],
        "stream": stream,
    }


FOLDER_MARKER = "FOLDER_MARKER_a1b2c3"

# A realistic folder-attachment workspace part: metadata + several file
# sections with real-ish content (like the fixed Electron folder scan sends).
# Sized like the real post-fix payload (~3.5 KB) so it fits the agent-prompt
# budget and the test can assert "substantial" content made it through.
FOLDER_WORKSPACE_TEXT = (
    "Attachment: AICL\n"
    "Type: folder\n"
    "State: extracted\n"
    "Summary: 6 of 14 files indexed\n"
    "Lines: 1561\n"
    "Files: 14\n"
    "Folders: 3\n\n"
    "Extracted context:\n"
    f"--- bend/bridge.py ---\n{FOLDER_MARKER}\n"
    + "".join(
        f"--- {name} ---\n"
        f"# {name}\n"
        + "\n".join(f"def fn_{i}(x):\n    return x + {i}\n" for i in range(12))
        + "\n"
        for name in ("lang/encoder.py", "lang/grammar.py",
                     "lang/symbols.py", "packet.py", "registry.py")
    )
    + "--- lang/decoder.py ---\n"
)


def _find_summary(client, unique_query: str) -> dict:
    runs = client.get("/v1/diagnostics").json()["runs"]
    match = [r for r in runs if r["query"] == unique_query]
    assert match, "no /v1/run diagnostic session was recorded"
    return match[0]


def _diag_for(client, unique_query: str) -> dict:
    """The full structured diagnostics of the graph run, fetched through the
    SAME detail endpoint the diagnostics viewer uses."""
    summary = _find_summary(client, unique_query)
    assert summary["pipeline"] == "graph_runtime", summary
    body = client.get(
        f"/v1/agent-runs/{summary['session_id']}/diagnostics"
    ).json()
    assert body["session_id"] == summary["session_id"]
    return body


def _entries_by(diag: dict):
    by = {}
    for e in diag["entries"]:
        by.setdefault((e["stage"], e["event"]), []).append(e)
    return by


# ── (a)+(b)+(c) sync path: the attachment must be traceable end-to-end ────────

def test_sync_run_records_attachment_flow_to_model_call(client):
    """A fabricated /v1/run request WITH an attachment produces diagnostics
    that show where the attachment went: request received → assembled
    prompt/context → the system prompt of the model call that fired."""
    query = _unique_query("sync attachment marker")
    resp = client.post("/v1/run", json=_run_payload(query))
    # The fabricated model endpoint is unreachable, so the run fails fast
    # (typed 500); the diagnostics must exist regardless.
    assert resp.status_code == 500, resp.text
    envelope = resp.json()["error"]
    assert envelope["type"] == "graph_error"

    diag = _diag_for(client, query)
    by = _entries_by(diag)

    # (b) session start + request received, with the attachment captured.
    assert by[("session", "start")]
    received = by[("request", "received")][0]["fields"]
    assert received["attachments_present"] is True
    assert received["attachment_part"] == "workspace"
    assert received["attachment_chars"] > 0
    assert MARKER in received["attachment_preview"]

    # The assembled prompt/context retained the attachment text.
    assembled = by[("prompt", "assembled")][0]["fields"]
    assert assembled["attachment_text_in_attachments_ctx"] is True
    assert assembled["attachment_text_in_agent_prompt"] is True
    assert MARKER in assembled["attachments_ctx_preview"]

    # Every model call that fired carried the attachment in its system prompt.
    calls = by[("model", "call")]
    assert calls, "the agent graph must have attempted a model call"
    for call in calls:
        fields = call["fields"]
        assert fields["model"] == "Fabricated-Model-123b"
        assert MARKER in fields["system_preview"], (
            "the attachment text must reach the model call's system prompt"
        )

    # The failure was recorded in the session — never silently dropped.
    assert diag["outcome"] == "failed"
    assert diag["error_summary"]
    assert any(e["event"] == "exception" for e in diag["entries"])
    assert by[("session", "finish")]


def test_folder_attachment_payload_records_substantial_content(client):
    """A folder-attachment workspace part with real extracted content must
    be recorded with substantial chars and a content-bearing preview — the
    diagnostics-side regression for the folder-content bug: previously the
    folder workspace part was metadata-only (~900 chars) and the model
    received no file content at all."""
    query = _unique_query("folder marker")
    payload = _run_payload(query)
    payload["prompt_parts"] = [
        # Real frontend priorities (PROMPT_PRIORITY): workspace=60 survives
        # the agent-prompt budget, so folder content must reach the model.
        {"name": "persona", "text": "You are a careful assistant.", "priority": 100},
        {"name": "workspace", "text": FOLDER_WORKSPACE_TEXT, "priority": 60},
    ]
    resp = client.post("/v1/run", json=payload)
    # Fabricated model endpoint is unreachable → fails fast; diagnostics exist.
    assert resp.status_code == 500, resp.text

    diag = _diag_for(client, query)
    by = _entries_by(diag)
    received = by[("request", "received")][0]["fields"]
    assert received["attachments_present"] is True
    assert received["attachment_part"] == "workspace"
    # Substantial content, not a ~900-char metadata-only listing.
    assert received["attachment_chars"] > 2000, received["attachment_chars"]
    assert FOLDER_MARKER in received["attachment_preview"]

    assembled = by[("prompt", "assembled")][0]["fields"]
    assert assembled["attachment_text_in_attachments_ctx"] is True
    assert assembled["attachment_text_in_agent_prompt"] is True
    assert FOLDER_MARKER in assembled["attachments_ctx_preview"]

    calls = by[("model", "call")]
    assert calls, "the agent graph must have attempted a model call"
    assert FOLDER_MARKER in calls[0]["fields"]["system_preview"], (
        "folder file content must reach the model call's system prompt"
    )


# ── (a) stream path: the background task owns session finalization ───────────

def test_stream_run_records_background_failure(client):
    """stream=true runs in a background task; its diagnostics are finalized
    by the task, not by the returning request handler."""
    query = _unique_query("stream bg marker")
    resp = client.post("/v1/run", json=_run_payload(query, stream=True))
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]
    assert resp.json()["status"] == "running"

    deadline = time.time() + 15.0
    status = "running"
    while time.time() < deadline:
        status = client.get(f"/v1/run/{run_id}").json()["status"]
        if status != "running":
            break
        time.sleep(0.05)
    assert status in ("completed", "failed"), status

    diag = _diag_for(client, query)
    assert diag["meta"]["pipeline"] == "graph_runtime"
    by = _entries_by(diag)
    assert by[("request", "received")]
    assert by[("session", "start")]
    assert by[("session", "finish")]
    assert diag["outcome"] in ("success", "failed")
    if status == "failed":
        assert diag["outcome"] == "failed"
        assert diag["error_summary"]
        assert any(e["event"] == "exception" for e in diag["entries"])


# ── (a) agent-runs sessions keep their own pipeline tag ──────────────────────

def test_agent_runs_sessions_are_tagged_agent_runtime(client):
    """The agent-runs pipeline stays tagged pipeline=agent_runtime, so the
    one diagnostics list can tell the two pipelines apart."""
    r = client.post("/v1/agent-runs", json={
        "query": "hi", "backend": {"type": "stub"},
    })
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    body = client.get(f"/v1/agent-runs/{run_id}/diagnostics").json()
    assert body["meta"]["pipeline"] == "agent_runtime"


# ── (d) instrumentation is behavior-neutral without a local model ────────────

def test_run_without_local_model_still_succeeds_and_records(no_model_client):
    """With no local model the /v1/run behavior is unchanged (mock experts,
    completed run), while a diagnostics session is still recorded."""
    query = _unique_query("no model run")
    resp = no_model_client.post("/v1/run", json={
        "query": query, "graph": "default",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "completed"

    diag = _diag_for(no_model_client, query)
    assert diag["meta"]["pipeline"] == "graph_runtime"
    assert diag["outcome"] == "success"
    by = _entries_by(diag)
    assert by[("session", "start")]
    assert by[("request", "received")]
    assert by[("session", "finish")]
    received = by[("request", "received")][0]["fields"]
    assert received["attachments_present"] is False
    # No completion_fn exists without a local model → no agent model calls.
    assert not by.get(("model", "call")), "no local model, no model calls"
