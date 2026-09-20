"""
Unit tests for structured context assembly (orcha.context).
"""
from orcha.context import assemble_system_prompt, trim_history


# ── assemble_system_prompt ────────────────────────────────────────────────────

def test_assemble_orders_by_priority_descending():
    out = assemble_system_prompt([
        {"name": "low", "text": "LOW", "priority": 10},
        {"name": "high", "text": "HIGH", "priority": 100},
        {"name": "mid", "text": "MID", "priority": 50},
    ])
    assert out == "HIGH\n\nMID\n\nLOW"


def test_assemble_is_stable_for_equal_priority():
    parts = [
        {"name": "a", "text": "AAA", "priority": 5},
        {"name": "b", "text": "BBB", "priority": 5},
    ]
    assert assemble_system_prompt(parts, max_len=100) == "AAA\n\nBBB"


def test_assemble_skips_empty_and_missing_text():
    out = assemble_system_prompt([
        {"name": "a", "text": "A", "priority": 10},
        {"name": "b", "text": "", "priority": 50},
        {"name": "c", "priority": 60},
    ])
    assert out == "A"


def test_assemble_empty_inputs():
    assert assemble_system_prompt(None) == ""
    assert assemble_system_prompt([]) == ""
    assert assemble_system_prompt([{"name": "a", "text": "   "}]) == ""


def test_assemble_drops_low_priority_when_over_budget():
    out = assemble_system_prompt([
        {"name": "high", "text": "H" * 50, "priority": 100},
        {"name": "low", "text": "L" * 100, "priority": 10},
    ], max_len=60)
    assert out == "H" * 50
    assert "L" not in out


def test_assemble_truncates_first_part_that_overflows():
    out = assemble_system_prompt([
        {"name": "high", "text": "H" * 60, "priority": 100},
        {"name": "low", "text": "L" * 100, "priority": 10},
    ], max_len=130)
    assert out.startswith("H" * 60)
    assert "[context truncated to fit prompt budget]" in out
    assert len(out) <= 130
    assert "L" * 100 not in out  # low part truncated at tail
    assert out.count("L") == 27


# ── trim_history ──────────────────────────────────────────────────────────────

def test_trim_history_empty():
    assert trim_history(None) == []
    assert trim_history([]) == []


def test_trim_history_keeps_anchor_and_recent_tail():
    msgs = [{"role": "user", "content": f"message number {i}"} for i in range(50)]
    out = trim_history(msgs, max_len=16000, max_turns=10)
    ids = [int(m["content"].split()[-1]) for m in out]
    assert len(out) == 10
    assert ids[0] == 0        # oldest kept as anchor
    assert ids[-1] == 49      # newest kept
    assert 25 not in ids      # middle dropped
    # original order preserved
    assert ids == sorted(ids)


def test_trim_history_respects_char_budget():
    msgs = [{"role": "user", "content": "X" * 100} for _ in range(10)]
    out = trim_history(msgs, max_len=150)
    total = sum(len(m["content"]) for m in out)
    assert total <= 150
    assert out == [msgs[0]]  # only the anchor fits
