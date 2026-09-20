"""
Prompt 11 regression: the first-message attachment race (confirmed cause).

Frontend behavior that caused it: the attach flow (picker -> per-path
inspection/extraction/indexing via Electron IPC) is asynchronous, but the
composer's send path was NOT gated on it. A message sent while the
attachment was still mid-extraction was composed with an EMPTY attachment
list -> prompt_parts WITHOUT the workspace part -> the run recorded
attachments_present=false / attachment_chars=0 and the model truthfully
reported "I don't see any files attached to this conversation". Only the
next message (sent after the attachment chip appeared) carried the
workspace part and succeeded.

This test pins the exact diagnostics differential between those two
moments, using the payloads the frontend composes:
  - turn 1 (the race): prompt_parts without the workspace part;
  - turn 2 (after extraction landed): the same conversation with the
    workspace part carrying the folder's extracted content.
"""
import uuid

import pytest
from fastapi.testclient import TestClient

from orcha.api.server import app
from orcha.settings import settings

RACE_MARKER = f"RACE_MARKER_{uuid.uuid4().hex[:8]}"

# What composePromptParts ships for a message sent while the attachment is
# still mid-extraction: persona/instructions/fileOps only — no workspace.
TURN1_PARTS = [
    {"name": "persona", "text": "You are a careful assistant.", "priority": 100},
    {"name": "instructions", "text": "Call the appropriate tool directly.", "priority": 90},
    {"name": "fileOps", "text": "File operations are available.", "priority": 85},
]

# Turn 2: what the frontend ships once the folder attachment has landed —
# the workspace part built from buildAttachmentContext, with extracted
# file content and the children listing.
TURN2_WORKSPACE_TEXT = (
    "Workspace context:\n"
    "Use the following attached files/folders as grounded context before answering.\n"
    "If a file is attached, inspect the extracted context before responding. Never ask the user to upload an attached item again.\n"
    "\n"
    "Attachment: graphs-parser\n"
    "Path: C:\\fabricated\\project\n"
    "Type: folder\n"
    "Size: 128 KB\n"
    "State: extracted\n"
    "Summary: 3 of 10 files indexed\n"
    "Lines: 640\n"
    "Files: 10\n"
    "Folders: 2\n"
    "\n"
    "Extracted context:\n"
    "--- parser/core.py ---\n"
    f"{RACE_MARKER}\n"
    "def parse(src):\n"
    "    return src.strip().splitlines()\n"
    "\n"
    "--- parser/token.py ---\n"
    "class Token:\n"
    "    pass\n"
    "\n"
    "Contents:\n"
    "- core.py (file, 4.5 KB)\n"
    "- token.py (file, 2.1 KB)\n"
    "- ...and 8 more items\n"
)


@pytest.fixture
def configured_model():
    """Fabricated 'activated' local model BEFORE the server starts, so the
    orchestrator (and thus the agent completion_fn) sees it — the sequence
    the real UI follows. The "123b" size pattern guarantees this model wins
    synthesizer selection even with a real Ollama daemon running."""
    prev_model = settings.local_server_model
    prev_base = settings.local_server_base_url
    prev_extras = list(settings.local_server_extra_models)
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
def client(configured_model):
    with TestClient(app) as c:
        yield c


def _turn(client, query: str, parts: list[dict]) -> None:
    """POST one /v1/run turn exactly as the frontend composes it. The
    fabricated model endpoint is unreachable, so the run fails fast (typed
    500); the diagnostics must exist regardless."""
    resp = client.post(
        "/v1/run",
        json={
            "query": query,
            "graph": "default",
            "capabilities": ["filesystem"],
            "workspace_roots": ["C:\\fabricated\\project"],
            "prompt_parts": parts,
        },
    )
    assert resp.status_code == 500, resp.text
    assert resp.json()["error"]["type"] == "graph_error"


def _find_summary(client, unique_query: str) -> dict:
    runs = client.get("/v1/diagnostics").json()["runs"]
    match = [r for r in runs if r["query"] == unique_query]
    assert match, "no /v1/run diagnostic session was recorded"
    return match[0]


def _diag_for(client, unique_query: str) -> dict:
    summary = _find_summary(client, unique_query)
    assert summary["pipeline"] == "graph_runtime", summary
    body = client.get(f"/v1/agent-runs/{summary['session_id']}/diagnostics").json()
    assert body["session_id"] == summary["session_id"]
    return body


def _entries_by(diag: dict):
    by = {}
    for e in diag["entries"]:
        by.setdefault((e["stage"], e["event"]), []).append(e)
    return by


def test_racy_turn1_ships_no_attachment_turn2_carries_it(client):
    """The regression: a message sent while the attachment is still
    mid-extraction provably carries NO attachment context (the model's
    denial is data-true), while the follow-up message — sent after
    processing completed — carries the full extracted content. This is the
    diagnostics differential that diagnosed the bug; blocking the send in
    that window removes the turn-1 gap entirely."""
    turn1 = f"race turn1 {uuid.uuid4().hex[:8]}"
    turn2 = f"race turn2 {uuid.uuid4().hex[:8]}"

    # Turn 1: the send that raced the still-in-flight attachment flow.
    _turn(client, turn1, TURN1_PARTS)
    # Turn 2: the next message in the same conversation, sent after the
    # attachment chip appeared.
    turn2_parts = TURN1_PARTS + [
        {"name": "workspace", "text": TURN2_WORKSPACE_TEXT, "priority": 60},
    ]
    _turn(client, turn2, turn2_parts)

    diag1 = _diag_for(client, turn1)
    diag2 = _diag_for(client, turn2)
    by1 = _entries_by(diag1)
    by2 = _entries_by(diag2)

    # ── Turn 1: request received with NO attachment context. ─────────────
    received1 = by1[("request", "received")][0]["fields"]
    assert received1["attachments_present"] is False
    assert received1["attachment_part"] is None
    assert received1["attachment_chars"] == 0
    assert received1["attachment_preview"] == ""

    # The model WAS asked, and its system prompt legitimately contained no
    # attachment — "I don't see any files attached" is data-true here.
    calls1 = by1[("model", "call")]
    assert calls1, "the agent graph must have attempted a model call for turn 1"
    for call in calls1:
        assert RACE_MARKER not in call["fields"]["system_preview"], (
            "turn-1 request had no workspace part, so the marker must not appear"
        )

    # ── Turn 2: the same conversation once the attachment is complete. ───
    received2 = by2[("request", "received")][0]["fields"]
    assert received2["attachments_present"] is True
    assert received2["attachment_part"] == "workspace"
    assert received2["attachment_chars"] > 0
    assert RACE_MARKER in received2["attachment_preview"]

    assembled2 = by2[("prompt", "assembled")][0]["fields"]
    assert assembled2["attachment_text_in_attachments_ctx"] is True
    assert assembled2["attachment_text_in_agent_prompt"] is True
    assert RACE_MARKER in assembled2["attachments_ctx_preview"]

    calls2 = by2[("model", "call")]
    assert calls2, "the agent graph must have attempted a model call for turn 2"
    for call in calls2:
        assert RACE_MARKER in call["fields"]["system_preview"], (
            "the attachment text must reach the model call's system prompt"
        )
