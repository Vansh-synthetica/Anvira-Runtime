"""
orcha.nodes.intent
==================
Conversation-intent gate for the single-agent graph.

Decides what the user's latest message needs BEFORE the agent tool loop is
ever entered. The gate runs one completion with NO tool schemas and a
neutral router prompt (persona + routing contract + attached-resource
metadata). The router replies with a single intent code; for CHAT the
router's own reply IS the conversational answer.

Intents
-------
CHAT              → finalize directly (the router's answer is shown verbatim).
READ_WORKSPACE    → read-only agent: inspect the attached workspace/folder.
READ_ATTACHMENT   → read-only agent: read/summarize/review attached items.
SEARCH            → read-only agent: find things inside the workspace.
PROJECT_ANALYSIS  → read-only agent: analyze/understand the project/repo.
UNKNOWN           → read-only agent (harmless reads; never writes/approvals).
FILE_OPERATION    → full agent: create/edit/delete/rename/move, approval policy.
TOOL_REQUEST      → full agent: any other tool action.

The tool loop — and therefore the approval machinery — is only reachable
after the gate declares a tool intent, so a greeting can never produce an
approval request. Read intents enter a READ-ONLY agent whose executor
contains only SAFE (non-mutating) tools, so workspace inspection can never
trigger approvals or writes. Attachment awareness: the gate receives
structured metadata about attached folders/files, workspace roots, indexed
projects and capabilities (``attachments_ctx``) so "this folder"/"these
files" references are grounded and routed into the workspace pipeline
instead of being answered from language priors.

Fail-safe: any router output that is not an exact intent code is treated as
a CHAT answer, so garbled output can never trigger tools.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.packets import OrchaPacket, PacketKind
from ..graph.context import RunContext
from ..graph.node import Node

# Exact intent codes the router must emit (one per reply, nothing else).
# Anything else is treated as a CHAT answer (fail-safe).
INTENT_CHAT = "CHAT"
INTENT_READ_WORKSPACE = "READ_WORKSPACE"
INTENT_READ_ATTACHMENT = "READ_ATTACHMENT"
INTENT_SEARCH = "SEARCH"
INTENT_PROJECT_ANALYSIS = "PROJECT_ANALYSIS"
INTENT_UNKNOWN = "UNKNOWN"
INTENT_FILE_OPERATION = "FILE_OPERATION"
INTENT_TOOL_REQUEST = "TOOL_REQUEST"

# Router replies with an intent code; CHAT is implicit (any non-code output).
NON_CHAT_INTENTS = {
    INTENT_READ_WORKSPACE,
    INTENT_READ_ATTACHMENT,
    INTENT_SEARCH,
    INTENT_PROJECT_ANALYSIS,
    INTENT_UNKNOWN,
    INTENT_FILE_OPERATION,
    INTENT_TOOL_REQUEST,
}

INTENT_GATE_ROUTER_PROMPT = (
    "You are a conversation router for an assistant with an attached workspace. "
    "Decide what the user's latest message needs, then reply with a single line "
    "containing ONLY one of these intent codes:\n\n"
    "CHAT — greetings, small talk, jokes, opinions, thanks, or pure questions "
    "that do not require looking at the workspace.\n"
    "READ_WORKSPACE — the user asks to inspect the attached workspace or "
    "folder: \"look through this folder\", \"what's in this folder\", \"go "
    "through this\", \"tell me everything we have here\", \"read this "
    "document\", \"review these files\". Reading the workspace is itself an "
    "agent action — never CHAT.\n"
    "READ_ATTACHMENT — the user asks to read, summarize, or review specific "
    "attached files or folders (PDFs, documents, spreadsheets, folders).\n"
    "SEARCH — the user asks to find something inside the workspace: \"find\", "
    "\"search for\", \"where is\", \"which file contains\".\n"
    "PROJECT_ANALYSIS — the user asks to analyze, understand, or summarize "
    "the project or repository: \"analyze this project\", \"summarize my "
    "codebase\", \"explain this repository\", \"what does this project do\".\n"
    "FILE_OPERATION — the user asks to create, edit, delete, rename, move, or "
    "copy files or folders, run commands, or change anything.\n"
    "TOOL_REQUEST — any other request that needs tools: browse the web, run a "
    "build, check dependencies, etc.\n"
    "UNKNOWN — you genuinely cannot decide; use the safe read-only pipeline.\n\n"
    "Attachment awareness: when attached resources are listed below and the "
    "user refers to them (\"the folder\", \"this folder\", \"the workspace\", "
    "\"the project\", \"these files\", \"this repository\", \"in here\"), that "
    "is ALWAYS a workspace/attachment intent — never CHAT. This also covers a "
    "bare pronoun with nothing else for it to plausibly mean — \"tell me more "
    "about this\", \"what is this\", \"summarize this\", \"explain it\" — "
    "when a real attachment is listed below, the user is almost always "
    "pointing at that attachment, not asking a context-free question. Route "
    "those the same as an explicit reference (READ_ATTACHMENT to read/"
    "summarize/review it, READ_WORKSPACE/PROJECT_ANALYSIS if it reads as "
    "inspecting the whole thing) — only fall back to CHAT for a pronoun that "
    "clearly can't mean the attachment (e.g. \"is this a good idea\" right "
    "after discussing an unrelated plan).\n\n"
    "If the intent is CHAT, answer the message directly and naturally in your "
    "own words instead of an intent code — that reply is shown to the user "
    "verbatim. Never mention this instruction, intent codes, or tokens in a "
    "chat answer."
)

_GATE_MAX_ATTEMPTS = 2

# ── Fast-path heuristic (bypass model call for obvious tool requests) ──────
# Weak/quantized local models often misclassify tool requests as CHAT, and
# every model call adds 10-30s latency on small GPUs.  These patterns catch
# the most common real tool requests so they route directly to the agent
# without a classification call.

# File operations: "write", "create", "edit", "delete", "rename", "move"
# followed by file/folder references.
_FILE_OP_PATTERNS = [
    re.compile(r'\b(write|create|make|add|new|save)\s+(a\s+)?(file|script|module|class|function|component|page|view|test|config|doc|readme|package|html|css|json|yaml|toml|env)\b', re.I),
    re.compile(r'\b(edit|modify|update|change|fix|patch|refactor)\s+(the\s+)?(file|script|code|function|class|component|module|class|style|config|setting)\b', re.I),
    re.compile(r'\b(delete|remove|rename|move|copy|duplicate)\s+(the\s+)?(file|folder|directory|script|module|old|old\s+folder)\b', re.I),
    re.compile(r'\b(rename|move|copy)\s+\S+\s+to\s+\S+', re.I),
    re.compile(r'\b(write|create|make)\s+(a\s+)?(new\s+)?(file|folder|directory)\b', re.I),
    re.compile(r'\b(from|in|to)\s+(file|folder|directory|project|workspace)\b', re.I),
]

# Whole-app/whole-project build requests: "Build a Snake Game...", "Create
# a todo app with...", "Develop a dashboard for...". These are exactly the
# detailed, multi-paragraph specs a real coding agent needs to route to the
# tool loop — the verb+noun pair can be separated by a long descriptive
# phrase, so the gap is bounded but generous. Checked separately, AFTER
# _TOOL_PATTERNS below: an indefinite article ("a"/"an") is REQUIRED (not
# optional) specifically so this never shadows the deliberately distinct
# "build the app"/"run the project" phrasing _TOOL_PATTERNS already owns
# (a bare build/run/test command on an existing thing, not a new-app spec).
_APP_BUILD_PATTERNS = [
    re.compile(
        r'\b(build|create|make|develop|implement|write)\s+(a|an)\s+'
        r'[\w\s,.\'"-]{0,80}?'
        r'\b(app|application|game|website|site|dashboard|'
        r'api|service|bot|extension|plugin|cli\s+tool|script|component|library|'
        r'server|backend|frontend|webpage)\b',
        re.I,
    ),
]

# Web/internet lookups: SEARCH's own contract (see module docstring and
# INTENT_GATE_ROUTER_PROMPT's "SEARCH — ...find something INSIDE THE
# WORKSPACE") scopes it to local workspace search, routed to the READ-ONLY
# agent whose executor is filtered to SAFE tools only — which does not
# include web_search/web_fetch (tagged NETWORK/CAUTIOUS, deliberately, so
# a passive "read-only" pass can't reach out to the internet unsupervised).
# _SEARCH_PATTERNS below matches on the bare word "search", though, with no
# web/local distinction — "use the web_search tool to search for X" matched
# it and got routed to SEARCH, silently downgrading the run to a tool
# surface that can never contain web_search at all. Confirmed live: the
# agent then either called search_text against the local workspace instead
# (wrong data), or got "[No tool 'web_search' available]" when it tried
# the right tool anyway. Checked BEFORE _SEARCH_PATTERNS specifically so an
# explicit web/internet signal always wins the ambiguity, routing instead
# to TOOL_REQUEST (full agent, where NETWORK tools ARE available per the
# access mode) — matching what the router prompt itself already documents
# ("TOOL_REQUEST — any other request that needs tools: browse the web...").
_WEB_SEARCH_PATTERNS = [
    re.compile(r'\bweb_search\b|\bweb_fetch\b', re.I),
    re.compile(r'\b(search|look|find|check)\s+(the\s+)?(web|internet|online)\b', re.I),
    re.compile(r'\b(browse|fetch|scrape)\s+(the\s+)?(web|internet|a\s+website|a\s+url|a\s+page)\b', re.I),
    re.compile(r'\bhttps?://\S+', re.I),
    re.compile(r'\b(google|bing|duckduckgo)\s+(it|this|that|for)\b', re.I),
]

# Search/inspection: "find", "search", "list", "show", "read", "open"
_SEARCH_PATTERNS = [
    re.compile(r'\b(search|find|grep|locate|where\s+(is|are|can))\b.*\b(for|in|inside|through|across)\b', re.I),
    re.compile(r'\b(find|search)\s+(all\s+)?(files?|functions?|classes?|methods?|variables?|references?|usages?)\b', re.I),
    re.compile(r'\b(list|show|display|print)\s+(all\s+)?(files?|folders?|directories?|contents?|structure|tree)\b', re.I),
    re.compile(r'\b(read|open|show|display|cat|view|inspect)\s+(the\s+)?(file|script|code|content)\b', re.I),
]

# Project/code analysis
_PROJECT_PATTERNS = [
    re.compile(r'\b(analyze|analyse|review|audit|inspect|examine)\s+(this\s+)?(project|codebase|repo|repository|code|file|module)\b', re.I),
    re.compile(r'\b(summarize|summarise|explain|describe)\s+(this\s+)?(project|codebase|repo|code|file)\b', re.I),
]

# Build/run commands
_TOOL_PATTERNS = [
    re.compile(r'\b(run|execute|build|compile|test|lint|install|deploy)\s+(the\s+)?(project|app|server|build|test|script|command)\b', re.I),
    re.compile(r'\b(run|execute|build|compile|test|lint)\s+(this|it|the)\b', re.I),
]

# Code-file heuristic: any filename with a known code/document extension
# plus an explicit action verb signals a tool request regardless of the
# noun phrasing around it ("Create a Python file called greet.py …").
_KNOWN_FILE_EXT = (
    r'(?:py|js|mjs|cjs|ts|tsx|jsx|java|c|h|cpp|hpp|cs|go|rs|rb|php|swift|kt|'
    r'scala|md|markdown|json|ya?ml|toml|ini|cfg|conf|html?|css|scss|less|sql|'
    r'sh|bash|zsh|bat|cmd|ps1|txt|csv|tsv|xml|vue|svelte|dart|lua|r|m|pl|'
    r'ex|exs|clj|hs|elm|proto|gradle|env|log)'
)
_CODE_FILE_RE = re.compile(rf'\b[\w.-]+\.{_KNOWN_FILE_EXT}\b', re.I)
_ACTION_VERB_RE = re.compile(
    r'\b(create|make|write|add|generate|build|implement|fix|update|edit|'
    r'modify|refactor|change|delete|remove|rename|move|copy|save|append|'
    r'open|read|show|display|list|find|search|inspect|run|execute|start|'
    r'stop|restart|install|uninstall|compile|deploy|debug)\b',
    re.I,
)
_QUESTION_FORM_RE = re.compile(
    r'^\s*(how|what|why|when|where|who|which|can|could|should|would|is|are|'
    r'does|do|did|will|shall|may)\b[\s\S]{0,500}?\?\s*$',
    re.I,
)
_PATH_REF_RE = re.compile(
    r'\b(read|open|show|view|cat|edit|delete|remove|inspect)\s+'
    r'[\w.-]+[/\\][\w./\\-]+',
    re.I,
)


# A request that both LOOKS for something and CHANGES it ("find the bug, and fix it in stats.py") is a file operation. Without this the
# search pattern below matched first and the run got a read-only tool surface it could never edit with (measured on a 3B model).
_MODIFY_VERB_RE = re.compile(r'\b(fix|repair|debug|patch|update|edit|modify|change|refactor|rename|delete|remove|create|write|implement|add|rewrite)\b', re.I)
_CHANGE_TARGET_RE = re.compile(r'\b(bug|bugs|failing|fails?|error|errors|typo|tests?|function|class|method|variable|code)\b', re.I)
_NEGATED_CHANGE_RE = re.compile(r"\b(do not|don't|dont|never|without|no need to)\s+(?:\w+\s+){0,2}?(change|modify|edit|touch|delete|remove|update|alter)\b[^.;]*", re.I)


_NOTHING_CHANGE_RE = re.compile(r'\b(change|modify|edit|touch|alter|update)\s+(nothing|anything)\b', re.I)


def _fast_path_intent(query: str) -> Optional[str]:
    """Heuristic intent detection for obvious tool requests.

    Returns an intent code when the query clearly needs a tool, or None
    when the query is ambiguous and should go through the model classifier.
    This avoids the 10-30s latency of a model call on small GPUs.
    """
    text = (query or "").strip()
    if not text:
        return None
    # No upper length bound: a detailed, well-specified build/coding request
    # ("Build a Snake game with a React/TS frontend and Python backend...")
    # is exactly the kind of message that MUST route to the tool-using
    # agent, and such specs are typically long, not short. Bailing out here
    # previously forced every long request through a single classifier
    # model call instead — and a small local model faced with a long,
    # detailed spec would often just start answering/discussing it in prose
    # instead of emitting a bare intent code, which (per the fail-safe
    # below) got shown to the user AS the final answer, silently skipping
    # the tool loop entirely for exactly the requests that most needed it.
    # The patterns below are action-verb-anchored, not length-based, so
    # applying them to long text doesn't meaningfully increase false
    # positives on genuinely long conversational messages.
    #
    # Question-form messages ("How do I create a react component in
    # App.tsx?") are advice requests, not tool work. Checked FIRST — before
    # this, "create a react component" could satisfy the whole-app build
    # pattern below and misfire on what's actually a question.
    if _QUESTION_FORM_RE.match(text):
        return None
    for p in _FILE_OP_PATTERNS:
        if p.search(text):
            return INTENT_FILE_OPERATION
    unnegated = _NOTHING_CHANGE_RE.sub('', _NEGATED_CHANGE_RE.sub('', text))
    if _MODIFY_VERB_RE.search(unnegated) and (_CODE_FILE_RE.search(text) or _CHANGE_TARGET_RE.search(text)):
        return INTENT_FILE_OPERATION
    for p in _WEB_SEARCH_PATTERNS:
        if p.search(text):
            return INTENT_TOOL_REQUEST
    for p in _SEARCH_PATTERNS:
        if p.search(text):
            return INTENT_SEARCH
    for p in _PROJECT_PATTERNS:
        if p.search(text):
            return INTENT_PROJECT_ANALYSIS
    for p in _TOOL_PATTERNS:
        if p.search(text):
            return INTENT_TOOL_REQUEST
    for p in _APP_BUILD_PATTERNS:
        if p.search(text):
            return INTENT_FILE_OPERATION
    if _ACTION_VERB_RE.search(text) and _CODE_FILE_RE.search(text):
        return INTENT_FILE_OPERATION
    if _PATH_REF_RE.search(text):
        return INTENT_FILE_OPERATION
    return None


def parse_intent(content: str) -> Tuple[str, str]:
    """Map a router completion to (intent, chat_response).

    An exact non-CHAT intent code → that intent with no response text.
    A single intent code embedded in noise → that intent.
    Compound/malformed codes (e.g. "READ_WORKSPACE_READ_ATTACHMENT") →
    the first recognized code.
    Action-describing responses from weak models (e.g.
    "LISTING_ALL_FILES_IN_WORKSPACE_REQUESTED") → inferred intent.
    Anything else → CHAT with the content as the conversational answer.
    """
    text = (content or "").strip()
    if text in NON_CHAT_INTENTS:
        return text, ""
    # Weak/quantized models often emit compound or noisy output containing
    # one or more intent codes. Try to extract a single code before falling
    # back to CHAT — otherwise the raw router tokens reach the user verbatim.
    code = extract_intent_code(text)
    if code is not None:
        return code, ""
    # Weak models sometimes emit natural-language descriptions of what they
    # intend to do (e.g. "LISTING_ALL_FILES_IN_WORKSPACE_REQUESTED") instead
    # of a bare intent code. Map these to the closest intent so the agent
    # pipeline can route them correctly.  Only trigger for clearly
    # machine-generated patterns: ALLCAPS, underscore-delimited tokens, or
    # very short phrases (≤4 words) that are otherwise impossible to confuse
    # with real conversation.
    inferred = _infer_intent_from_description(text)
    if inferred is not None:
        return inferred, ""
    return INTENT_CHAT, text


# Heuristic keywords that map natural-language action descriptions to intents.
# Checked against lowercased, underscore-stripped text.
_ACTION_KEYWORDS: Dict[str, List[Tuple[str, ...]]] = {
    INTENT_FILE_OPERATION: [
        ("create", "make", "new", "write", "add"),
        ("edit", "modify", "update", "change", "rename"),
        ("delete", "remove", "erase", "drop"),
        ("move", "copy", "duplicate"),
    ],
    INTENT_READ_WORKSPACE: [
        ("list", "show", "browse", "explore", "inspect"),
        ("read", "open", "view", "examine"),
        ("files", "folders", "directory", "workspace", "project"),
    ],
    INTENT_READ_ATTACHMENT: [
        ("attach", "upload", "pdf", "document", "spreadsheet"),
    ],
    INTENT_SEARCH: [
        ("search", "find", "grep", "locate", "where"),
    ],
    INTENT_PROJECT_ANALYSIS: [
        ("analyze", "analyse", "summarize", "summarise", "explain"),
        ("understand", "overview", "review"),
    ],
    INTENT_TOOL_REQUEST: [
        ("run", "execute", "build", "test", "lint", "compile"),
        ("install", "deploy", "fetch", "download"),
    ],
}


def _infer_intent_from_description(text: str) -> Optional[str]:
    """Best-effort mapping of a natural-language action description to an
    intent code.  Returns ``None`` when the text is genuinely conversational.

    Only triggers for clearly machine-generated patterns:
    - ALLCAPS text (e.g. "LISTING_ALL_FILES")
    - Underscore-delimited tokens (e.g. "listing_all_files")
    - Very short phrases (≤4 words) that look like action labels
    """
    raw = (text or "").strip()
    if not raw:
        return None
    # Normalize: lowercase, collapse underscores/spaces, strip punctuation
    normalized = re.sub(r"[_\s]+", " ", raw.lower())
    normalized = re.sub(r"[^a-z0-9 ]", "", normalized).strip()
    if not normalized:
        return None
    # Only trigger for clearly machine-generated patterns:
    # 1. ALLCAPS text (e.g. "LISTING_ALL_FILES_IN_WORKSPACE_REQUESTED")
    # 2. Underscore-delimited tokens (e.g. "listing_all_files")
    # 3. Very short phrases (≤4 words) with no spaces but multiple words
    is_machine_generated = (
        "_" in raw
        or raw.upper() == raw and len(raw) > 3
        or (len(normalized.split()) <= 4 and "_" in raw)
    )
    if not is_machine_generated:
        return None
    # Same ordering fix as _WEB_SEARCH_PATTERNS in _fast_path_intent: a
    # web/internet signal must win over SEARCH's bare "search" keyword
    # BEFORE the generic scoring loop below, not just alongside it — a tie
    # (e.g. "SEARCHING_WEB_FOR_INFO" scoring 1 for both SEARCH's "search"
    # and TOOL_REQUEST's "web") would otherwise still resolve to SEARCH,
    # since it's scored first in _ACTION_KEYWORDS' dict order and the loop
    # only overwrites best_intent on a STRICTLY higher score.
    if any(kw in normalized for kw in ("web", "internet", "online", "browse")):
        return INTENT_TOOL_REQUEST
    # Score each intent by keyword hits.
    best_intent: Optional[str] = None
    best_score = 0
    for intent, groups in _ACTION_KEYWORDS.items():
        score = 0
        for group in groups:
            for kw in group:
                if kw in normalized:
                    score += 1
        if score > best_score:
            best_score = score
            best_intent = intent
    return best_intent


# Router replies on weak/quantized local models often wrap the code in extra
# tokens or emit several codes at once. This fixed re-prompt nudges the router
# back to a single clean code line for the retry attempt.
_GATE_REPROMPT = (
    "Your previous reply was not a valid intent code. Reply with exactly one "
    "line containing ONLY one of these codes: CHAT, READ_WORKSPACE, "
    "READ_ATTACHMENT, SEARCH, PROJECT_ANALYSIS, FILE_OPERATION, TOOL_REQUEST, "
    "UNKNOWN. No explanation, no quotes, no extra tokens."
)

# If a chat fallback would literally echo intent codes at the user, this
# neutral reply is shown instead — raw router tokens must never reach the UI.
_GATE_GARBLED_FALLBACK = (
    "I can help you look into that. Could you tell me a bit more about what "
    "you'd like to do with the attached workspace?"
)

_CODE_RE = re.compile(r"^\s*(CHAT|READ_WORKSPACE|READ_ATTACHMENT|SEARCH|PROJECT_ANALYSIS|FILE_OPERATION|TOOL_REQUEST|UNKNOWN)\s*$")


def extract_intent_code(text: str) -> Optional[str]:
    """Best-effort extraction of an intent code from a router completion.

    Accepts: an exact single code; a single code embedded in short noise
    (quotes, punctuation, leading/trailing words); or a JSON object with an
    ``intent`` field. Returns ``None`` when the reply is ambiguous (several
    different codes) or contains no code at all.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    if raw in NON_CHAT_INTENTS:
        return raw
    if _CODE_RE.match(raw):
        return raw.strip()
    stripped = raw.strip("\"'`[](){}<> \t\r\n")
    if stripped in NON_CHAT_INTENTS:
        return stripped
    try:
        if raw.startswith("{"):
            obj = json.loads(raw)
            intent = obj.get("intent") if isinstance(obj, dict) else None
            if isinstance(intent, str) and intent in NON_CHAT_INTENTS:
                return intent
    except Exception:
        pass
    found = [code for code in NON_CHAT_INTENTS if code in raw]
    if len(found) == 1:
        return found[0]
    # Compound intent codes (e.g. "READ_WORKSPACE_READ_ATTACHMENT"): weak
    # models sometimes concatenate two valid codes.  Pick the one that
    # appears first in the text so the result is deterministic.
    if len(found) > 1:
        return min(found, key=lambda c: raw.index(c))
    return None


def looks_like_garbled_codes(text: str) -> bool:
    """True when the router output is mostly intent tokens — never show it."""
    raw = (text or "").strip()
    if not raw:
        return False
    found = [code for code in NON_CHAT_INTENTS if code in raw]
    if not found:
        return False
    covered = sum(raw.count(code) * len(code) for code in found)
    return covered >= len(raw) * 0.4


def _router_leak(text: str) -> bool:
    """True when a router reply STARTS with an intent token — the model
    leaked the routing protocol instead of answering. Weak local models
    commonly reply \"READ_ATTACHMENT\\n\\nHere is the summary...\"; the
    leading code must be handled, never shown verbatim."""
    raw = (text or "").strip()
    if not raw:
        return False
    first_word = re.split(r"[\s\n,.;:'\"\[\]{}()<>]+", raw, maxsplit=1)[0]
    return first_word in NON_CHAT_INTENTS or first_word == INTENT_CHAT


@dataclass
class IntentGateConfig:
    """
    Configuration for the intent gate.

    Attributes
    ----------
    completion_fn    Async (messages, system, tools) -> dict completion, same
                     contract as AgentConfig.completion_fn. Called with
                     ``tools=None`` so the router never sees tool schemas.
    system_prompt    Router system prompt: routing contract (plus optional
                     persona). Must NOT contain the tool-use directive shipped
                     to the tool loop.
    attachments_ctx  Structured metadata about attached files/folders,
                     workspace roots, indexed projects, and available
                     capabilities. Rendered into the router prompt so
                     workspace references ("this folder", "these files") are
                     grounded instead of guessed from language priors.
    seed_messages    Prior conversation turns (role/content dicts) so the
                     router can judge intent with real session context.
    """

    completion_fn: Optional[Callable] = None
    system_prompt: str = INTENT_GATE_ROUTER_PROMPT
    attachments_ctx: str = ""
    seed_messages: List[Dict[str, Any]] = field(default_factory=list)


class IntentGateNode(Node):
    """
    Routes the run between a plain conversational answer, the read-only
    workspace pipeline, and the full tool loop.

    Writes ``intent_gate_intent`` (one of the intent codes) and, for CHAT,
    ``intent_gate_response`` (the router's own answer) into the packet
    payload. The graph's conditional edge routes read intents to the
    read-only agent, tool intents to the full agent, and CHAT to finalize.
    """

    name = "intent_gate"

    def __init__(
        self,
        name: str = "intent_gate",
        config: Optional[IntentGateConfig] = None,
        timeout_s: Optional[float] = 180.0,
        retries: int = 0,
    ) -> None:
        self.name = name
        self.timeout_s = timeout_s
        self.retries = retries
        self.config = config or IntentGateConfig()

    def _compose_prompt(self) -> str:
        prompt = self.config.system_prompt or INTENT_GATE_ROUTER_PROMPT
        if self.config.attachments_ctx and self.config.attachments_ctx.strip():
            prompt += "\n\nAttached resources:\n" + self.config.attachments_ctx.strip()
        return prompt

    def _conversational_prompt(self) -> str:
        """Persona-only prompt for the CHAT re-ask.

        The router contract instructs code-only output, so a model that
        obeys it literally replies with a bare "CHAT" code — which would
        otherwise surface as an empty answer. Stripping the routing
        contract leaves the persona alone, so the re-ask produces a real
        conversational reply.
        """
        prompt = (self.config.system_prompt or "").strip()
        marker = INTENT_GATE_ROUTER_PROMPT
        if prompt.endswith(marker):
            prompt = prompt[: -len(marker)].strip()
        return prompt

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        if self.config.completion_fn is None:
            return packet.fork(
                packet.kind, intent_gate_intent=INTENT_TOOL_REQUEST, intent_gate_response=""
            )

        latest = packet.payload.get("task", packet.query)

        # Fast-path heuristic: bypass the model classifier for obvious tool
        # requests (file ops, search, project analysis, build commands).
        # This avoids the 10-30s latency of a model call on small GPUs.
        fast = _fast_path_intent(latest)
        if fast is not None:
            return packet.fork(
                packet.kind, intent_gate_intent=fast, intent_gate_response=""
            )

        messages = list(self.config.seed_messages or [])
        # The history the frontend ships excludes the just-sent message, but
        # direct callers may already include it — never duplicate it.
        if not (
            messages
            and messages[-1].get("role") == "user"
            and messages[-1].get("content") == latest
        ):
            messages.append({"role": "user", "content": latest})

        # No tool schemas: the router must never see tool definitions, or it
        # would be tempted to call one for every message.
        intent: Optional[str] = None
        response = ""
        for attempt in range(_GATE_MAX_ATTEMPTS):
            message = await self.config.completion_fn(
                messages, self._compose_prompt(), None
            )
            content = message.get("content") or ""
            exact = parse_intent(content)
            if exact[0] != INTENT_CHAT or not _router_leak(content):
                # Clean code, or a genuine chat answer (no router tokens at
                # its head).
                intent, response = exact
                break
            # Router output starts with intent tokens (weak/quantized local
            # models commonly emit several codes or wrap one in noise).
            # Extract a single code when unambiguous; otherwise re-ask once
            # with a strict one-line instruction before falling back to a
            # safe answer.
            code = extract_intent_code(content)
            if code is not None:
                intent, response = code, ""
                break
            if attempt + 1 < _GATE_MAX_ATTEMPTS:
                messages.append({"role": "user", "content": _GATE_REPROMPT})
                continue
            intent, response = INTENT_CHAT, _GATE_GARBLED_FALLBACK
            break

        # A bare "CHAT" code must never become an empty answer: the model
        # followed the routing contract literally. Re-ask once with the
        # persona alone (no router contract) so the conversational reply is
        # real; if that still yields a code or empty text, surface the
        # neutral fallback instead of a blank response.
        if intent == INTENT_CHAT and not response.strip():
            message = await self.config.completion_fn(
                messages, self._conversational_prompt(), None
            )
            content = (message.get("content") or "").strip()
            if content and not extract_intent_code(content) and not looks_like_garbled_codes(content):
                response = content
            if not response.strip():
                response = _GATE_GARBLED_FALLBACK

        return packet.fork(
            packet.kind,
            intent_gate_intent=intent,
            intent_gate_response=response,
        )


__all__ = [
    "IntentGateNode",
    "IntentGateConfig",
    "parse_intent",
    "INTENT_GATE_ROUTER_PROMPT",
    "INTENT_CHAT",
    "INTENT_READ_WORKSPACE",
    "INTENT_READ_ATTACHMENT",
    "INTENT_SEARCH",
    "INTENT_PROJECT_ANALYSIS",
    "INTENT_UNKNOWN",
    "INTENT_FILE_OPERATION",
    "INTENT_TOOL_REQUEST",
    "NON_CHAT_INTENTS",
]
