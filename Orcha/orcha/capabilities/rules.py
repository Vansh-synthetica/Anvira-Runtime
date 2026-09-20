"""
orcha.capabilities.rules
========================
Rule-string permissions for the tool system — a compact, human-writable
language that answers "may the agent run THIS tool with THESE arguments?"
before any interactive approval is needed.

Rule strings
------------
    ToolName              matches every invocation of ToolName
    ToolName(pattern)     matches when the call's CONTENT matches pattern
    *                     matches every tool

Content matching uses glob syntax (``*``, ``?``, ``[...]``), case-insensitive.
The "content" of a call is extracted from its keyword arguments by a small
per-tool heuristic: ``path`` for file tools, ``command`` for terminal tools,
then common keys, then the first string argument — so rules read naturally:

    run_command(git *)            allow every git subcommand
    run_command(npm test*)        allow the project's tests
    edit_file(*.env*)             deny touching env files
    write_file                    ask before ANY write (bare rule)
    read_file(*)                  allow all reads

Kinds & precedence
------------------
Three kinds — **deny**, **ask**, **allow** — evaluated in strict order:
deny wins over ask, ask wins over allow, and only then do mode/trust
defaults apply. Within one kind, sources are checked by priority
(policy → user → project → session) but this only orders display; kind
always dominates.

"Yes, and don't ask again"
--------------------------
Approvals can carry a structured update (:meth:`PermissionRules.add` from
the API layer / broker suggestions), which lands as a session-scoped allow
rule — turning a repeated interactive approval into data instead of a
hardcoded branch.

Everything here is plain Python data: serializable via ``to_dict`` /
``from_dict``, safe to persist per project or per session.
"""
from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

ALLOW = "allow"
DENY = "deny"
ASK = "ask"

# Source priority for same-kind ordering (lower = checked first).
_SOURCE_PRIORITY = {"policy": 0, "user": 1, "project": 2, "session": 3}
_RULE_RE = re.compile(r"^\s*([A-Za-z0-9_*?\[\]-]+)\s*(?:\((.*)\))?\s*$", re.S)

Decision = Literal["allow", "deny", "ask"]


def content_of(tool: str, kwargs: Dict[str, Any]) -> str:
    """
    The string a rule pattern is matched against. Deterministic and cheap:
    known semantic keys first, then the first string value, then the JSON
    dump of the arguments.
    """
    for key in (
        "path", "command", "source", "target", "old_path", "new_path",
        "query", "url", "pattern", "name",
    ):
        if key in kwargs:
            value = kwargs.get(key)
            if isinstance(value, str):
                return value
    for value in kwargs.values():
        if isinstance(value, str):
            return value
    try:
        return json.dumps(kwargs, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return ""


@dataclass(frozen=True)
class Rule:
    """One parsed rule string plus its origin."""
    kind: str                     # allow | deny | ask
    source: str                   # policy | user | project | session
    raw: str                      # original text, e.g. "run_command(git *)"
    tool_pattern: str             # glob against the tool name
    content_pattern: Optional[str]  # glob against content_of(...); None = any

    @property
    def display(self) -> str:
        return self.raw

    def matches(self, tool: str, kwargs: Dict[str, Any]) -> bool:
        if not fnmatch.fnmatch(tool.lower(), self.tool_pattern.lower()):
            return False
        if self.content_pattern is None:
            return True
        return fnmatch.fnmatch(
            content_of(tool, kwargs).casefold(), self.content_pattern.casefold(),
        )


@dataclass
class ResolvedRule:
    """The winning rule (if any) for one evaluation, for UI explanations."""
    decision: Decision
    rule: Optional[Rule] = None


@dataclass
class PermissionRules:
    """
    Ordered rule sets by kind. Parsing failures are skipped silently at
    construction time (resilient-by-contract) — a malformed line must never
    take down an agent boot; it simply protects nothing.
    """
    deny: List[Rule] = field(default_factory=list)
    ask: List[Rule] = field(default_factory=list)
    allow: List[Rule] = field(default_factory=list)

    # ── Mutation ────────────────────────────────────────────────────────

    def add(
        self,
        kind: str,
        rule_str: str,
        source: str = "session",
    ) -> Optional[Rule]:
        """Parse and append one rule; returns it, or None when malformed."""
        parsed = parse_rule(kind, rule_str, source)
        if parsed is None:
            return None
        bucket = getattr(self, parsed.kind)
        bucket.append(parsed)
        _stable_insert(bucket, parsed)
        return parsed

    def remove(self, kind: str, raw: str) -> bool:
        bucket = getattr(self, kind, None)
        if bucket is None:
            return False
        before = len(bucket)
        setattr(self, kind, [r for r in bucket if r.raw != raw])
        return len(bucket) != before

    def clear(self, kind: Optional[str] = None, source: Optional[str] = None) -> None:
        kinds = [kind] if kind else ["allow", "deny", "ask"]
        for k in kinds:
            bucket = getattr(self, k)
            if source is None:
                bucket.clear()
            else:
                setattr(self, k, [r for r in bucket if r.source != source])

    # ── Evaluation ──────────────────────────────────────────────────────

    def evaluate(self, tool: str, kwargs: Dict[str, Any]) -> Optional[ResolvedRule]:
        """
        First matching rule by kind precedence (deny → ask → allow), then
        source priority within the kind. None when nothing matches.
        """
        for kind in ("deny", "ask", "allow"):
            for rule in getattr(self, kind):
                if rule.matches(tool, kwargs):
                    return ResolvedRule(decision=kind, rule=rule)  # type: ignore[arg-type]
        return None

    # ── Persistence ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            kind: [
                {"raw": r.raw, "source": r.source}
                for r in getattr(self, kind)
            ]
            for kind in ("allow", "deny", "ask")
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PermissionRules":
        rules = cls()
        if not data:
            return rules
        for kind in ("allow", "deny", "ask"):
            for item in data.get(kind) or []:
                if isinstance(item, str):
                    rules.add(kind, item)
                elif isinstance(item, dict) and item.get("raw"):
                    rules.add(kind, str(item["raw"]), str(item.get("source") or "session"))
        return rules


def parse_rule(kind: str, rule_str: str, source: str = "session") -> Optional[Rule]:
    """Parse ``Tool(pattern)`` / ``Tool`` / ``*`` into a Rule (or None)."""
    if kind not in (ALLOW, DENY, ASK):
        return None
    match = _RULE_RE.match(rule_str or "")
    if match is None:
        return None
    tool_pattern, content = match.group(1), match.group(2)
    if not tool_pattern:
        return None
    content_pattern: Optional[str] = None
    if content is not None:
        content = content.strip()
        # An empty pattern or bare "*" means "any arguments".
        content_pattern = None if content in ("", "*") else content
    return Rule(
        kind=kind,
        source=source if source in _SOURCE_PRIORITY else "session",
        raw=rule_str.strip(),
        tool_pattern=tool_pattern,
        content_pattern=content_pattern,
    )


def _stable_insert(bucket: List[Rule], rule: Rule) -> None:
    """Keep same-kind buckets ordered by source priority, newest last."""
    priority = _SOURCE_PRIORITY.get(rule.source, 3)
    for i, existing in enumerate(bucket):
        if _SOURCE_PRIORITY.get(existing.source, 3) > priority:
            bucket.insert(i, rule)
            return
    bucket.append(rule)


__all__ = [
    "ALLOW", "DENY", "ASK",
    "Rule", "ResolvedRule", "PermissionRules", "parse_rule", "content_of",
]
