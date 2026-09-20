"""
Tests for Phase-5 background memory extraction (pure logic):

- parse_memories_payload: JSON robustness, type vocabulary, dedupe, caps
- MemoryExtractor decisions: cursor gating, mutual exclusion, transcript
  rendering bounds
- end-to-end run() with a stub backend: success advances cursor ONLY on
  success; failures leave it untouched
- MarkdownMemoryWriter file format + index upsert
"""
import json

import pytest

from orcha.agent_runtime.events import EventLog, ToolCallAction, UserMessageObservation
from orcha.agent_runtime.memory_extraction import (
    MarkdownMemoryWriter, MemoryExtractor, MemoryExtractionConfig,
    parse_memories_payload,
)


# ── Payload parsing ───────────────────────────────────────────────────────────

def test_parse_plain_json():
    raw = json.dumps({"memories": [
        {"type": "user", "name": "role", "title": "Role",
         "description": "d", "body": "b"},
    ]})
    out = parse_memories_payload(raw)
    assert len(out) == 1 and out[0]["name"] == "role" and out[0]["type"] == "user"


def test_parse_tolerates_fences_and_prose():
    raw = "Sure!\n```json\n{\"memories\": [{\"type\": \"project\", \"name\": \"stack\", \"body\": \"x\"}]}\n```\ndone"
    out = parse_memories_payload(raw)
    assert len(out) == 1


def test_parse_drops_bad_types_and_dupes_and_empty_bodies():
    raw = json.dumps({"memories": [
        {"type": "wizard", "name": "a", "body": "x"},        # bad type
        {"type": "user", "name": "dup", "body": "1"},
        {"type": "user", "name": "DUP!", "body": "2"},       # same slug after clean
        {"type": "reference", "name": "no_body"},            # empty body
    ]})
    out = parse_memories_payload(raw)
    assert [m["name"] for m in out] == ["dup"]


def test_parse_caps_count_and_body():
    items = [{"type": "project", "name": f"m{i}", "body": "y" * 100} for i in range(10)]
    out = parse_memories_payload(
        json.dumps({"memories": items}), max_memories=3, max_body_chars=50,
    )
    assert len(out) == 3
    assert len(out[0]["body"]) == 50


def test_parse_garbage_returns_empty():
    assert parse_memories_payload("not json at all") == []
    assert parse_memories_payload("") == []


# ── Extractor decisions ───────────────────────────────────────────────────────

def _log_with_user_message(text="please refactor the auth module"):
    log = EventLog()
    log.append(UserMessageObservation(content=text))
    return log


def test_has_new_work_respects_cursor():
    extractor = MemoryExtractor(backend=None)
    events = tuple(_log_with_user_message().events)
    assert extractor.has_new_work(events, cursor_seq=0)
    assert not extractor.has_new_work(events, cursor_seq=99)


def test_mutual_exclusion_detects_memory_writes():
    extractor = MemoryExtractor(backend=None, memory_dir_hint="memory")
    log = _log_with_user_message()
    log.append(ToolCallAction(
        name="write_file",
        arguments={"path": "C:/ws/project/memory/MEMORY.md", "content": "- note"},
    ))
    assert extractor.agent_wrote_memory(tuple(log.events)) is True


def test_mutual_exclusion_ignores_unrelated_writes():
    extractor = MemoryExtractor(backend=None, memory_dir_hint="memory")
    log = _log_with_user_message()
    log.append(ToolCallAction(name="write_file", arguments={"path": "src/app.py", "content": "x"}))
    assert extractor.agent_wrote_memory(tuple(log.events)) is False


def test_transcript_render_bounds():
    extractor = MemoryExtractor(
        backend=None, config=MemoryExtractionConfig(max_transcript_chars=200),
    )
    log = EventLog()
    for i in range(20):
        log.append(UserMessageObservation(content="word " * 30))
    text = extractor.render_transcript(tuple(log.events), cursor_seq=0)
    assert len(text) <= 210  # small slack for slicing
    assert text.count("USER:") >= 1


# ── run() flow with a stub backend ────────────────────────────────────────────

class _StubBackend:
    model_name = "stub"

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def complete(self, messages, tools=None, config=None):
        self.calls += 1

        class R:
            text = self.payload

        return R()


@pytest.mark.anyio
async def test_run_success_returns_memories_and_advances_cursor():
    backend = _StubBackend(json.dumps({"memories": [
        {"type": "feedback", "name": "be_concise", "title": "Be concise", "body": "Keep answers short."},
    ]}))
    extractor = MemoryExtractor(backend)
    events = tuple(_log_with_user_message("keep answers short please").events)
    result = await extractor.run(events, cursor_seq=0)
    assert result["skipped"] is None if "skipped" in result else True
    assert result["cursor"] == extractor.last_seq(events)
    assert result["memories"][0]["name"] == "be_concise"
    assert backend.calls == 1


@pytest.mark.anyio
async def test_run_failure_leaves_cursor_untouched():
    class Exploding:
        model_name = "x"

        async def complete(self, *a, **kw):
            raise RuntimeError("down")

    extractor = MemoryExtractor(Exploding())
    events = tuple(_log_with_user_message().events)
    result = await extractor.run(events, cursor_seq=0)
    assert result["cursor"] == 0
    assert str(result.get("skipped", "")).startswith("extraction_error")


@pytest.mark.anyio
async def test_run_skips_when_main_agent_already_saved():
    backend = _StubBackend("{}")
    extractor = MemoryExtractor(backend, memory_dir_hint="memory")
    log = _log_with_user_message()
    log.append(ToolCallAction(
        name="edit_file",
        arguments={"path": "memory/user_role.md", "old_string": "a", "new_string": "b"},
    ))
    result = await extractor.run(tuple(log.events), cursor_seq=0)
    assert result["skipped"] == "main_agent_already_wrote_memory"
    assert backend.calls == 0  # no wasted call


@pytest.mark.anyio
async def test_writer_receives_extracted_memories():
    backend = _StubBackend(json.dumps({"memories": [
        {"type": "user", "name": "pref", "title": "Pref", "body": "dark mode"},
    ]}))
    received = []
    extractor = MemoryExtractor(backend, writer=lambda ms: received.extend(ms))
    events = tuple(_log_with_user_message().events)
    await extractor.run(events, cursor_seq=0)
    assert len(received) == 1


# ── MarkdownMemoryWriter ──────────────────────────────────────────────────────

def test_writer_creates_files_and_index(tmp_path):
    writer = MarkdownMemoryWriter(tmp_path, user_id="local")
    writer([
        {"name": "pref_dark", "title": "Dark mode", "description": "UI preference",
         "type": "user", "body": "User prefers dark themes."},
    ])
    topic = tmp_path / "local" / "pref_dark.md"
    index = tmp_path / "local" / "MEMORY.md"
    assert topic.exists()
    body = topic.read_text(encoding="utf-8")
    assert "title: Dark mode" in body and "type: user" in body
    idx = index.read_text(encoding="utf-8")
    assert "[Dark mode](pref_dark.md)" in idx
