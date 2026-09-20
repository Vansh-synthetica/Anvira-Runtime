"""
Tests for Phase-3 permission rules + atomic approval race:

- rule parsing & matching (tool glob, content glob, bare rules)
- PermissionRules.evaluate precedence (deny > ask > allow; source order)
- PermissionPolicy.evaluate pipeline (rules → trust → mode) incl. deny
  bypass-immunity and auto-mode degradation
- ToolExecutor integration: permission_denied vs approval_required
- ApprovalBroker: one-shot atomic decide race, update recording,
  run-scoped clearing of updates
"""
import asyncio

import pytest

from orcha.capabilities.base import (
    CapabilityContext, PermissionPolicy, ToolExecutor, spec,
)
from orcha.capabilities.filesystem import build_tools as build_fs_tools
from orcha.capabilities.rules import (
    ASK, ALLOW, DENY, PermissionRules, content_of, parse_rule,
)


# ── Rule parsing / matching ───────────────────────────────────────────────────

def test_parse_rule_shapes():
    r = parse_rule(ALLOW, "run_command(git *)")
    assert r.tool_pattern == "run_command" and r.content_pattern == "git *"
    bare = parse_rule(DENY, "write_file")
    assert bare.content_pattern is None
    star = parse_rule(ASK, "*")
    assert star.tool_pattern == "*"
    assert parse_rule(ALLOW, "bad kind!", "policy") is None or True  # never raises


def test_parse_rule_rejects_garbage():
    assert parse_rule(ALLOW, "") is None
    assert parse_rule("nope", "read_file") is None


def test_content_of_prefers_semantic_keys():
    assert content_of("edit_file", {"path": "a.py", "old_string": "x"}) == "a.py"
    assert content_of("run_command", {"command": "git status", "timeout_s": 5}) == "git status"


def test_rule_matching_tool_and_content():
    from orcha.capabilities.rules import Rule
    r = Rule(kind=ALLOW, source="session", raw="run_command(git *)",
             tool_pattern="run_command", content_pattern="git *")
    assert r.matches("run_command", {"command": "git push"})
    assert not r.matches("run_command", {"command": "rm -rf"})
    assert not r.matches("write_file", {"command": "git push"})


# ── PermissionRules evaluation ────────────────────────────────────────────────

def test_kind_precedence_deny_beats_allow():
    rules = PermissionRules()
    rules.add(ALLOW, "*", "user")
    rules.add(DENY, "edit_file(*.env*)", "project")
    resolved = rules.evaluate("edit_file", {"path": ".env.local"})
    assert resolved.decision == DENY
    assert rules.evaluate("read_file", {"path": ".env.local"}).decision == ALLOW


def test_ask_wins_over_allow():
    rules = PermissionRules()
    rules.add(ALLOW, "run_command(*)", "session")
    rules.add(ASK, "run_command(rm *)", "session")
    assert rules.evaluate("run_command", {"command": "rm x"}).decision == ASK


def test_source_priority_within_kind():
    rules = PermissionRules()
    rules.add(ALLOW, "read_file(tmp/*)", "session")
    rules.add(ALLOW, "read_file(*)", "policy")
    resolved = rules.evaluate("read_file", {"path": "tmp/x"})
    assert resolved.rule.source == "policy"


def test_to_dict_from_dict_roundtrip():
    rules = PermissionRules()
    rules.add(ALLOW, "run_command(git *)", "session")
    rules.add(DENY, "delete_file", "project")
    clone = PermissionRules.from_dict(rules.to_dict())
    assert clone.evaluate("run_command", {"command": "git log"}).decision == ALLOW
    assert clone.evaluate("delete_file", {"path": "x"}).decision == DENY


# ── Policy pipeline ───────────────────────────────────────────────────────────

def _executor(policy):
    ctx = CapabilityContext(roots=["."])
    return ToolExecutor(build_fs_tools(ctx), policy=policy)


def test_policy_deny_is_bypass_immune_even_in_action_mode():
    rules = PermissionRules()
    rules.add(DENY, "delete_file", "policy")
    policy = PermissionPolicy(access_mode="action", rules=rules)
    ex = _executor(policy)
    r = ex.invoke("delete_file", path="whatever.txt")
    assert not r.ok and r.error["code"] == "permission_denied"


def test_policy_allow_rule_auto_approves_in_approval_mode():
    rules = PermissionRules()
    rules.add(ALLOW, "run_command(git *)", "user")
    policy = PermissionPolicy(access_mode="approval", rules=rules)
    # git allowed by rule even though run_command needs approval…
    spec_run = spec("run_command", "", {}, lambda **kw: "ok",
                    permissions=["execute"], safety_level="dangerous")
    assert policy.evaluate(spec_run, {"command": "git status"}) == "allow"
    # …but other commands still ask.
    assert policy.evaluate(spec_run, {"command": "curl evil"}) == "ask"


def test_policy_trusted_capability_still_works():
    policy = PermissionPolicy(access_mode="approval", trusted_capabilities=["filesystem"])
    ex = _executor(policy)
    assert ex.invoke("write_file", path="t.txt", content="x").ok


def test_requires_approval_backcompat_view():
    policy = PermissionPolicy(access_mode="approval")
    spec_read = spec("read_file", "", {}, lambda **kw: "", permissions=["read"])
    spec_delete = spec("delete_file", "", {}, lambda **kw: "",
                       permissions=["write", "dangerous"])
    assert policy.requires_approval(spec_read, {}) is False
    assert policy.requires_approval(spec_delete, {}) is True


def test_explicit_ask_rule_forces_interactivity_on_safe_tools():
    """An ask RULE is absolute: even a read-only tool asks when a rule says so."""
    rules = PermissionRules()
    rules.add(ASK, "read_file(~/*)", "user")
    policy = PermissionPolicy(access_mode="action", rules=rules)
    ex = _executor(policy)
    r = ex.invoke("read_file", path="~/secrets.txt")
    assert r.error["code"] == "approval_required"


# ── Auto mode + circuit breaker ──────────────────────────────────────────────

def test_auto_mode_degrades_after_unanswered_waits():
    policy = PermissionPolicy(access_mode="auto")  # no broker → asks can't resolve
    spec_delete = spec("delete_file", "", {}, lambda **kw: "",
                       permissions=["write", "dangerous"])
    assert policy.evaluate(spec_delete, {}) == "ask"
    for _ in range(policy._MAX_UNANSWERED):
        policy.record_wait_outcome(decided=False)
    assert policy.interactive_exhausted is True
    assert policy.evaluate(spec_delete, {}) == "deny"
    # A human finally answering resets everything.
    policy.record_wait_outcome(decided=True)
    assert policy.evaluate(spec_delete, {}) == "ask"


# ── Executor integration ─────────────────────────────────────────────────────

def test_executor_reports_permission_denied_with_rule_context():
    rules = PermissionRules()
    rules.add(DENY, "write_file(secrets/*)", "policy")
    policy = PermissionPolicy(access_mode="action", rules=rules)
    ex = _executor(policy)
    r = ex.invoke("write_file", path="secrets/key.txt", content="x")
    assert r.error["code"] == "permission_denied"
    assert r.error["detail"]["rule"] == "write_file(secrets/*)"


def test_executor_approval_required_carries_ask_rule():
    rules = PermissionRules()
    rules.add(ASK, "write_file", "user")
    policy = PermissionPolicy(access_mode="approval", rules=rules)
    ex = _executor(policy)
    r = ex.invoke("write_file", path="a.txt", content="x")
    assert r.error["code"] == "approval_required"
    assert r.error["detail"]["ask_rule"] == "write_file"


# ── Broker: atomic race + updates ────────────────────────────────────────────

@pytest.mark.anyio
async def test_decide_is_one_shot_under_concurrent_race():
    from orcha.capabilities.base import ApprovalBroker
    broker = ApprovalBroker()
    entry = broker.request("run1", "call1", "write_file", {"path": "a.txt"})
    results = await asyncio.gather(*[
        asyncio.to_thread(broker.decide, entry["id"], approved=True, run_id="run1")
        for _ in range(5)
    ])
    assert sum(results) == 1  # exactly one winner


@pytest.mark.anyio
async def test_decision_update_recorded_and_drained():
    from orcha.capabilities.base import ApprovalBroker
    broker = ApprovalBroker()
    entry = broker.request("run1", "call1", "run_command", {"command": "git push"})
    ok = broker.decide(
        entry["id"], approved=True, run_id="run1",
        update={"kind": "add_rule", "rule": "run_command(git push)"},
    )
    assert ok
    updates = broker.drain_updates()
    assert len(updates) == 1
    assert updates[0]["update"]["rule"] == "run_command(git push)"
    assert updates[0]["tool"] == "run_command"
    assert broker.drain_updates() == []  # consumed


@pytest.mark.anyio
async def test_clear_run_also_clears_updates():
    from orcha.capabilities.base import ApprovalBroker
    broker = ApprovalBroker()
    entry = broker.request("run1", "call1", "write_file", {"path": "a.txt"})
    broker.decide(entry["id"], approved=True, run_id="run1", update={"kind": "add_rule"})
    broker.clear_run("run1")
    assert broker.drain_updates() == []
