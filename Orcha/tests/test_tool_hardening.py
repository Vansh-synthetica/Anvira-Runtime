"""
Tests for Phase-1 tool-contract hardening:

- ToolSpec read-only / concurrency-safe metadata (explicit + derived)
- ToolExecutor.plan_parallel wave grouping
- ReadStateCache stale-write protection (never-read, partial view,
  changed-on-disk, content-compare fallback, fresh-chain edits)
- OutputStore oversized-output persistence via the executor
"""
import os

import pytest

from orcha.capabilities.base import (
    CapabilityContext, PermissionPolicy, ToolExecutor, spec,
)
from orcha.capabilities.filesystem import build_tools as build_fs_tools
from orcha.capabilities.outputstore import OutputStore
from orcha.capabilities.readstate import ReadStateCache
from orcha.capabilities.registry import CapabilityRegistry
from orcha.nodes.tool import ToolSpec


# ── Metadata ──────────────────────────────────────────────────────────────────

def _fs_executor(tmp_path, read_state=None):
    ctx = CapabilityContext(roots=[str(tmp_path)], read_state=read_state)
    return ToolExecutor(build_fs_tools(ctx))


def test_read_only_derived_from_permissions():
    assert ToolSpec(name="r", description="", permissions=["read"]).is_read_only()
    assert not ToolSpec(name="w", description="", permissions=["write"]).is_read_only()
    assert not ToolSpec(name="x", description="", permissions=["execute"]).is_read_only()


def test_concurrency_safe_defaults_to_read_only():
    ro = ToolSpec(name="r", description="")
    assert ro.is_concurrency_safe() is True
    unsafe = ToolSpec(name="w", description="", permissions=["write"])
    assert unsafe.is_concurrency_safe() is False


def test_explicit_overrides_win():
    forced = ToolSpec(name="q", description="", concurrency_safe=True, read_only=False)
    assert forced.is_read_only() is False
    assert forced.is_concurrency_safe() is True


def test_describe_exposes_contract_metadata(tmp_path):
    ex = _fs_executor(tmp_path)
    described = {d["name"]: d for d in ex.describe()}
    assert described["read_file"]["read_only"] is True
    assert described["read_file"]["concurrency_safe"] is True
    assert described["write_file"]["read_only"] is False


def test_plan_parallel_groups_safe_and_isolates_unsafe(tmp_path):
    ex = _fs_executor(tmp_path)
    waves = ex.plan_parallel([
        ("read_file", {"path": "a.txt"}),
        ("run_command", {"command": "echo hi"}),  # unsafe → alone
        ("list_directory", {}),
    ])
    assert len(waves) == 3
    assert [c[0] for c in waves[0]] == ["read_file"]
    assert waves[1][0][0] == "run_command"
    assert waves[2][0][0] == "list_directory"

    safe_only = ex.plan_parallel([
        ("read_file", {"path": "a.txt"}),
        ("exists", {"path": "b.txt"}),
    ])
    assert len(safe_only) == 1 and len(safe_only[0]) == 2


def test_spec_builder_passes_metadata():
    s = spec("t", "d", {}, lambda: None, permissions=["read"], read_only=False, concurrency_safe=True)
    assert s.is_read_only() is False
    assert s.is_concurrency_safe() is True


# ── Stale-write protection ────────────────────────────────────────────────────

def test_write_new_file_never_blocked(tmp_path):
    rs = ReadStateCache()
    ex = _fs_executor(tmp_path, rs)
    r = ex.invoke("write_file", path="new.txt", content="hi")
    assert r.ok, r.error


def test_edit_without_read_refused(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    ex = _fs_executor(tmp_path, ReadStateCache())
    r = ex.invoke("edit_file", path="a.txt", old_string="hello", new_string="bye")
    assert not r.ok
    assert r.error["code"] == "stale_write"
    assert "read_file" in r.error["message"]


def test_overwrite_existing_without_read_refused(tmp_path):
    (tmp_path / "a.txt").write_text("original")
    ex = _fs_executor(tmp_path, ReadStateCache())
    r = ex.invoke("write_file", path="a.txt", content="clobbered")
    assert not r.ok
    assert r.error["code"] == "stale_write"


def test_partial_view_blocks_edit_until_full_read(tmp_path):
    (tmp_path / "a.txt").write_text("l1\nl2\nl3\n")
    ex = _fs_executor(tmp_path, ReadStateCache())
    # Partial slice only.
    assert ex.invoke("read_file", path="a.txt", offset=0, limit=1).ok
    r = ex.invoke("edit_file", path="a.txt", old_string="l3", new_string="X")
    assert not r.ok and r.error["code"] == "stale_write"
    # Full read unlocks it.
    assert ex.invoke("read_file", path="a.txt").ok
    r = ex.invoke("edit_file", path="a.txt", old_string="l3", new_string="X")
    assert r.ok, r.error


def test_changed_on_disk_since_read_refused(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("v1")
    ex = _fs_executor(tmp_path, ReadStateCache())
    assert ex.invoke("read_file", path="a.txt").ok
    # External modification behind the runtime's back.
    os.utime(target, (0, 0))  # force a real stat change too
    target.write_text("v2 by someone else")
    r = ex.invoke("edit_file", path="a.txt", old_string="v1", new_string="nope")
    assert not r.ok
    assert r.error["code"] == "stale_write"
    assert "changed on disk" in r.error["message"]


def test_mtime_lie_content_compare_allows(tmp_path):
    """Windows mtimes can advance without content changing — must NOT refuse."""
    target = tmp_path / "a.txt"
    target.write_text("stable")
    ex = _fs_executor(tmp_path, ReadStateCache())
    assert ex.invoke("read_file", path="a.txt").ok
    os.utime(target, (0, 0))
    assert ex.invoke("write_file", path="a.txt", content="stable v2").ok


def test_fresh_chain_of_edits_needs_single_read(tmp_path):
    ex = _fs_executor(tmp_path, ReadStateCache())
    # create_file records the fresh state itself; one explicit read covers
    # everything that follows without re-reading.
    assert ex.invoke("create_file", path="chain.txt", content="one").ok
    assert ex.invoke("read_file", path="chain.txt").ok
    assert ex.invoke("append_file", path="chain.txt", content="-two").ok
    assert ex.invoke("edit_file", path="chain.txt", old_string="two", new_string="2").ok
    assert ex.invoke("delete_file", path="chain.txt").ok
    assert not (tmp_path / "chain.txt").exists()


def test_no_cache_keeps_legacy_behavior(tmp_path):
    (tmp_path / "a.txt").write_text("legacy")
    ex = _fs_executor(tmp_path, None)  # explicit no cache
    r = ex.invoke("edit_file", path="a.txt", old_string="legacy", new_string="new")
    assert r.ok, r.error


def test_delete_without_read_refused_but_after_read_allowed(tmp_path):
    (tmp_path / "gone.txt").write_text("data")
    ex = _fs_executor(tmp_path, ReadStateCache())
    r = ex.invoke("delete_file", path="gone.txt")
    assert not r.ok and r.error["code"] == "stale_write"
    assert ex.invoke("read_file", path="gone.txt").ok
    assert ex.invoke("delete_file", path="gone.txt").ok


# ── OutputStore ───────────────────────────────────────────────────────────────

def test_output_store_passthrough_small_results(tmp_path):
    store = OutputStore(str(tmp_path / "out"), threshold_chars=1000)
    value, saved = store.wrap_if_oversized("t", "small")
    assert value == "small" and saved is None


def test_output_store_persists_oversized_results(tmp_path):
    out_dir = tmp_path / "out"
    store = OutputStore(str(out_dir), threshold_chars=200, preview_chars=100)
    big = "x" * 500
    value, saved = store.wrap_if_oversized("search_text", big)
    assert saved is not None and os.path.isfile(saved)
    assert "[Output truncated" in value
    assert saved in value
    assert len(value) < 500 + 300  # preview bounded
    body = open(saved, encoding="utf-8").read()
    assert body == big


def test_executor_persists_oversized_tool_result(tmp_path):
    out_dir = tmp_path / "out"
    store = OutputStore(str(out_dir), threshold_chars=200, preview_chars=80)
    ctx = CapabilityContext(roots=[str(tmp_path)])
    ex = ToolExecutor(build_fs_tools(ctx), output_store=store)
    # A directory tree with many files exceeds 200 chars.
    for i in range(40):
        (tmp_path / f"f{i}.txt").write_text("content")
    r = ex.invoke("directory_tree", path=".", depth=1)
    assert r.ok
    assert "Full output saved to" in str(r.value)
    assert r.metadata.get("persisted_output")


def test_executor_small_results_untouched(tmp_path):
    ctx = CapabilityContext(roots=[str(tmp_path)])
    ex = ToolExecutor(build_fs_tools(ctx), output_store=OutputStore(str(tmp_path / "o")))
    assert ex.invoke("write_file", path="s.txt", content="tiny").ok
