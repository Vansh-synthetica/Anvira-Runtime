"""Tests for orcha.core.packets — the typed message bus."""
import json

import pytest

from orcha.core.packets import OrchaPacket, PacketKind, BudgetState, _is_json_shaped


def test_is_json_shaped_accepts_plain_data():
    assert _is_json_shaped({"a": 1, "b": [1, 2, "x"], "c": None, "d": True})


def test_is_json_shaped_rejects_live_objects():
    class Fake:
        pass

    assert not _is_json_shaped(Fake())
    assert not _is_json_shaped({"a": Fake()})
    assert not _is_json_shaped([1, 2, Fake()])


def test_to_json_drops_non_serializable_payload_entries_instead_of_crashing():
    """
    Regression test: a node occasionally stashes a live, in-memory-only
    helper object in packet.payload to survive across node-to-node forks
    within one run (e.g. nodes/task_executor.py's EventStreamAdapter, or a
    bound method) — never meant to be persisted. Before this fix, the first
    checkpoint write of such a packet crashed the entire run with a
    PydanticSerializationError. Confirms the checkpoint JSON now drops only
    the offending keys and keeps the rest of the payload intact.
    """
    class Fake:
        pass

    p = OrchaPacket(kind=PacketKind.QUERY, query="hello")
    p.payload["live_object"] = Fake()
    p.payload["bound_method"] = p.age_s
    p.payload["plain_data"] = {"nested": [1, 2, {"ok": True}]}

    data = json.loads(p.to_json())
    assert "live_object" not in data["payload"]
    assert "bound_method" not in data["payload"]
    assert data["payload"]["plain_data"] == {"nested": [1, 2, {"ok": True}]}


def test_packet_basic_construction():
    p = OrchaPacket(kind=PacketKind.QUERY, query="hello")
    assert p.query == "hello"
    assert p.kind == PacketKind.QUERY
    assert p.payload == {}
    assert p.trace == []
    assert isinstance(p.budget, BudgetState)


def test_stamp_appends_trace_and_returns_self():
    p = OrchaPacket(kind=PacketKind.QUERY, query="hello")
    returned = p.stamp("decompose", 12.5, subtasks=3)

    assert returned is p  # chainable, mutates in place
    assert len(p.trace) == 1
    assert p.trace[0].stage == "decompose"
    assert p.trace[0].duration_ms == 12.5
    assert p.trace[0].data == {"subtasks": 3}


def test_fork_merges_payload_and_clones_budget_and_trace():
    p = OrchaPacket(kind=PacketKind.QUERY, query="hello", payload={"a": 1})
    p.stamp("step1", 1.0)

    child = p.fork(PacketKind.SUBTASKS, b=2)

    assert child.kind == PacketKind.SUBTASKS
    assert child.payload == {"a": 1, "b": 2}
    assert child.query == "hello"

    # trace is cloned, not shared
    assert len(child.trace) == 1
    child.stamp("step2", 2.0)
    assert len(child.trace) == 2
    assert len(p.trace) == 1  # original untouched

    # budget is a deep copy
    child.budget.cost_used += 0.5
    assert p.budget.cost_used == 0.0


def test_error_packet():
    p = OrchaPacket(kind=PacketKind.QUERY, query="hello")
    err = p.error_packet("boom", code=500)
    assert err.kind == PacketKind.ERROR
    assert err.payload["error"] == "boom"
    assert err.payload["code"] == 500


@pytest.mark.parametrize(
    "cost,latency,iters,max_cost,max_latency,max_iters,expected",
    [
        (0.0, 0.0, 0, 1.0, 60.0, 3, False),
        (1.0, 0.0, 0, 1.0, 60.0, 3, True),   # cost exhausted
        (0.0, 60.0, 0, 1.0, 60.0, 3, True),  # latency exhausted
        (0.0, 0.0, 3, 1.0, 60.0, 3, True),   # iterations exhausted
        (0.5, 30.0, 1, 1.0, 60.0, 3, False), # within budget
    ],
)
def test_budget_exhausted(cost, latency, iters, max_cost, max_latency, max_iters, expected):
    b = BudgetState(
        max_cost=max_cost, max_latency_s=max_latency, max_iterations=max_iters,
        cost_used=cost, latency_used_s=latency, iterations=iters,
    )
    assert b.exhausted is expected


def test_budget_remaining():
    b = BudgetState(max_cost=1.0, cost_used=0.3, max_latency_s=10.0, latency_used_s=4.0)
    assert b.remaining_cost == pytest.approx(0.7)
    assert b.remaining_latency_s == pytest.approx(6.0)

    # never negative
    b2 = BudgetState(max_cost=1.0, cost_used=1.5)
    assert b2.remaining_cost == 0.0
