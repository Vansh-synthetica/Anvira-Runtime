"""
orcha.experts.local_chat
=========================
Expert for ANY locally-hosted OpenAI-compatible chat endpoint.

Works out of the box with
--------------------------
llama.cpp server   ./llama-server -m model.gguf --port 8080
LM Studio          starts on localhost:1234 by default
vLLM               python -m vllm.entrypoints.openai.api_server ...
text-gen-webui     with the openai extension enabled
LocalAI            drop-in OpenAI replacement
Jan                local AI assistant with server mode
Kobold.cpp         with OpenAI-compat mode enabled

No cloud account, no API key, no internet connection required.
The api_key field defaults to "not-needed" — set it to anything
if the local server requires a token (some do for access control).
"""
from __future__ import annotations

import ast
import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional

from .base import BaseExpert, ExpertCapability, ExpertOutput


def _slugify(s: str) -> str:
    return (
        s.replace(":", "_").replace(".", "_")
         .replace("/", "_").replace("-", "_").replace(" ", "_")
    )


# ── Text-level tool-call recovery ────────────────────────────────────────────

_TOOL_CALL_COUNTER = {"n": 0}


def _balanced_json_objects(text: str, limit: int = 16) -> List[str]:
    """Extract balanced ``{...}`` substrings (nested braces, quoted strings
    with escapes supported)."""
    out: List[str] = []
    start = text.find("{")
    while start != -1 and len(out) < limit:
        depth = 0
        in_str = False
        esc = False
        quote = ""
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == quote:
                    in_str = False
            elif ch in ('"', "'"):
                in_str = True
                quote = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[start : i + 1])
                    break
        start = text.find("{", start + 1)
    return out


def _coerce_tool_arguments(raw: Any) -> Optional[str]:
    """Normalize an ``arguments`` payload to a JSON string (OpenAI shape)."""
    if isinstance(raw, str):
        return raw
    if raw is None:
        return "{}"
    try:
        return json.dumps(raw)
    except (TypeError, ValueError):
        return None


def _tool_call_from_obj(obj: Any) -> Optional[Dict[str, Any]]:
    """Build one OpenAI-shaped tool call from a parsed JSON object, or None."""
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool") or obj.get("function_name")
    if not name or not isinstance(name, str):
        return None
    # A genuine invocation always carries an arguments payload under one of
    # the known keys. Objects without one ("d = {\"name\": \"x\"}" in
    # example code) are data, not calls.
    has_args = any(k in obj for k in ("arguments", "parameters", "args"))
    if not has_args:
        return None
    args = _coerce_tool_arguments(
        obj.get("arguments", obj.get("parameters", obj.get("args")))
    )
    if args is None:
        return None
    _TOOL_CALL_COUNTER["n"] += 1
    return {
        "id": f"call_text_{_TOOL_CALL_COUNTER['n']}",
        "type": "function",
        "function": {"name": name.strip(), "arguments": args},
    }


def _tool_call_from_xml(match: "re.Match[str]") -> Optional[Dict[str, Any]]:
    """Build an OpenAI-shaped call from ``<function name=.. arguments=../>``."""
    name = match.group(1)
    raw_args = match.group(2).strip()
    if raw_args[:1] in ('"', "'"):
        raw_args = raw_args[1:-1]
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        try:
            args = ast.literal_eval(raw_args)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None
    args = _coerce_tool_arguments(args)
    if args is None:
        return None
    _TOOL_CALL_COUNTER["n"] += 1
    return {
        "id": f"call_text_{_TOOL_CALL_COUNTER['n']}",
        "type": "function",
        "function": {"name": name.strip(), "arguments": args},
    }


def extract_text_tool_calls(content: str) -> List[Dict[str, Any]]:
    """
    Recover tool calls from plain-text assistant output.

    Backends that do not parse the model's tool-call syntax server-side
    (older llama.cpp builds without --jinja, LM Studio profiles, models
    emitting fenced ```json blocks or Qwen-style <tool_call> tags) leave
    the invocation inside ``content``. This parser recognizes:

      1. ``<tool_call>{...}</tool_call>`` blocks (Qwen / Hermes format),
      2. fenced ```json ... ``` code blocks containing a tool-call object,
      3. bare ``{"name": ..., "arguments": {...}}`` objects,
      4. ``<invoke name="f"><parameter name="a">1</parameter></invoke>``
         (Anthropic/MiniMax XML form, optionally wrapped in a namespaced
         ``<ns:tool_call>`` tag — MiniMax emits ``<minimax:tool_call>``).

    Returns OpenAI-shaped tool calls; empty list when none are found.
    """
    if not content or not any(
        token in content for token in ("{", "<function", "[call", "<invoke")
    ):
        return []

    calls: List[Dict[str, Any]] = []

    # 0a. Hybrid JSON: models mix JSON structure with PYTHON triple-quoted
    #     string values ({"content": triple-quote...}), which no strict
    #     parser accepts. Locate fields positionally instead.
    for m in re.finditer(
        r'\{\s*"name"\s*:\s*"([\w.-]+)"\s*,\s*"arguments"\s*:\s*\{',
        content,
    ):
        name = m.group(1)
        seg_start = m.end()
        pm = re.match(r'\s*"path"\s*:\s*"((?:[^"\\]|\\.)*)"', content[seg_start:])
        path_val = None
        if pm:
            try:
                path_val = json.loads('"' + pm.group(1) + '"')
            except json.JSONDecodeError:
                path_val = pm.group(1)
            seg_start += pm.end()
        cm = re.match(r'\s*,\s*"content"\s*:\s*"""', content[seg_start:])
        if not cm:
            continue
        body_start = seg_start + cm.end()
        terms = list(re.compile(r'(?<!\\)"""').finditer(content, body_start))
        # The file body itself may contain unescaped triple-quote pairs
        # (docstrings); the TRUE terminator is the last one, immediately
        # followed by the closing brace(s) of the arguments object.
        tm = None
        for cand in reversed(terms):
            if re.match(r'\s*\}', content[cand.end():]):
                tm = cand
                break
        if tm is None:
            continue
        file_text = content[body_start:tm.start()]
        file_text = (
            file_text.replace('\\"', '"')
                     .replace("\\'", "'")
                     .replace('\\n', '\n')
                     .replace('\\t', '\t')
        )
        if file_text.startswith("\n"):
            file_text = file_text[1:]
        args_obj: Dict[str, Any] = {}
        if path_val:
            args_obj["path"] = path_val
        args_obj["content"] = file_text
        coerced = _coerce_tool_arguments(args_obj)
        if coerced is not None:
            _TOOL_CALL_COUNTER["n"] += 1
            calls.append({
                "id": f"call_text_{_TOOL_CALL_COUNTER['n']}",
                "type": "function",
                "function": {"name": name, "arguments": coerced},
            })
    if calls:
        return calls

    # 0b. XML function-call form: <function name="f" arguments='{...}'/>
    #    (emitted by some models when the server's tool template renders
    #    but the trained <tool_call> format never fires).
    for m in re.finditer(
        r"<function\s+name=\"([\w.-]+)\"\s+arguments\s*=\s*('[^']*'|\"[^\"]*\")\s*/?>",
        content,
    ):
        call = _tool_call_from_xml(m)
        if call is not None:
            calls.append(call)
    # 0c. Anthropic/MiniMax-style XML invoke form:
    #     <invoke name="f"><parameter name="a">1</parameter></invoke>
    #     Often wrapped in a namespaced tag (<minimax:tool_call>...</minimax:tool_call>)
    #     but the wrapper's name is irrelevant here — only <invoke>/<parameter> matter,
    #     so a bare <tool_call> wrapper (or no wrapper at all) works the same way.
    for m in re.finditer(r'<invoke\s+name="([\w.-]+)"\s*>(.*?)</invoke>', content, re.DOTALL):
        name = m.group(1)
        body = m.group(2)
        args: Dict[str, Any] = {}
        for pm in re.finditer(r'<parameter\s+name="([\w.-]+)"\s*>(.*?)</parameter>', body, re.DOTALL):
            args[pm.group(1)] = pm.group(2).strip()
        coerced = _coerce_tool_arguments(args)
        if coerced is not None and name:
            _TOOL_CALL_COUNTER["n"] += 1
            calls.append({
                "id": f"call_text_{_TOOL_CALL_COUNTER['n']}",
                "type": "function",
                "function": {"name": name, "arguments": coerced},
            })
    # 0b. Bracket-kwargs form: [call write_file(path='a.py', content='x')]
    #     (also [calls ...] / [invoke ...]). Payload parsed with Python's
    #     own AST so multi-line triple-quoted strings containing commas,
    #     parentheses and newlines survive intact.
    for m in re.finditer(
        r"\[(?:calls?|invokes?|tool[_ ]?call)\s+(\w+)\s*\((.*?)\)\s*\]",
        content,
        re.DOTALL,
    ):
        name, payload = m.group(1), m.group(2).strip()
        if not name:
            continue
        try:
            tree = ast.parse(f"_f({payload})")
            call_node = tree.body[0].value
            kwargs: Dict[str, Any] = {}
            for kw in call_node.keywords:
                if kw.arg is None:
                    continue  # **kwargs spread — not recoverable
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            coerced = _coerce_tool_arguments(kwargs)
            if coerced is not None:
                _TOOL_CALL_COUNTER["n"] += 1
                calls.append({
                    "id": f"call_text_{_TOOL_CALL_COUNTER['n']}",
                    "type": "function",
                    "function": {"name": name, "arguments": coerced},
                })
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            continue
    if calls:
        return calls

    candidates: List[str] = []
    # 1. Explicit tool_call tags.
    for match in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL):
        candidates.append(match)
    # 2. Fenced code blocks: scan for a balanced {...} object anywhere
    #    inside (models wrap calls in ```json AND ```plaintext fences,
    #    often with prose like "Final Answer:" before the object).
    for block in re.findall(r"```(?:[a-z]+)?\s*(.*?)```", content, re.DOTALL | re.IGNORECASE):
        candidates.extend(_balanced_json_objects(block))
    # 3. Bare object with "name" key anywhere in the text.
    for match in re.findall(r"\{[^{}]*\"name\"[^{}]*\}", content):
        candidates.append(match)
    # 4. Balanced-brace scan of the whole text as a last resort.
    if not candidates:
        candidates.extend(_balanced_json_objects(content, limit=32))

    calls: List[Dict[str, Any]] = []
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        obj = None
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            # Models frequently emit Python-dict literals (single quotes,
            # True/False/None). ast.literal_eval accepts those safely.
            try:
                obj = ast.literal_eval(candidate)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                continue
        call = _tool_call_from_obj(obj)
        if call is not None:
            calls.append(call)
    return calls


def strip_text_tool_calls(content: str) -> str:
    """Remove recognized tool-call payloads from content so the remaining
    prose can still be shown to the user."""
    if not content:
        return ""
    cleaned = re.sub(r"<tool_call>\s*\{.*?\}\s*</tool_call>", "", content, flags=re.DOTALL)
    # XML function form.
    cleaned = re.sub(
        r"<function\s+name=\"[\w.-]+\"\s+arguments\s*=\s*('[^']*'|\"[^\"]*\")\s*/?>",
        "",
        cleaned,
        flags=re.DOTALL,
    )
    # Anthropic/MiniMax XML invoke form — strip the namespaced wrapper (with
    # its contents) first, then any bare/unwrapped <invoke> left over.
    cleaned = re.sub(
        r"<[\w.:-]*tool_call>\s*<invoke.*?</invoke>\s*</[\w.:-]*tool_call>",
        "",
        cleaned,
        flags=re.DOTALL,
    )
    cleaned = re.sub(r'<invoke\s+name="[\w.-]+"\s*>.*?</invoke>', "", cleaned, flags=re.DOTALL)
    # Bracket-kwargs form.
    cleaned = re.sub(
        r"\[(?:calls?|invokes?|tool[_ ]?call)\s+\w+\s*\(.*?\)\s*\]",
        "",
        cleaned,
        flags=re.DOTALL,
    )
    # Drop every balanced {...} object that actually parsed as a tool call.
    for obj_str in _balanced_json_objects(cleaned):
        try:
            obj = json.loads(obj_str)
        except json.JSONDecodeError:
            try:
                obj = ast.literal_eval(obj_str)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                continue
        if _tool_call_from_obj(obj) is not None:
            cleaned = cleaned.replace(obj_str, "")
    # Collapse any code fences left empty by the removal above.
    cleaned = re.sub(r"```[a-zA-Z]*\s*```", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


# Hard ceiling for the length-truncation retry: generation never goes above
# this even if the model keeps hitting max_tokens.
_MAX_GENERATION_TOKENS = 8192
# Multiplier applied to max_tokens each time a completion is cut off by the
# length limit, so long answers are re-generated at a larger budget instead
# of being silently truncated mid-sentence.
_TRUNCATION_RETRIES = 2

# Cloud backends (BYOK: OpenRouter, Groq, ...) rate-limit, especially their
# free tiers — a burst of agent-loop calls (planner, executor, verifier,
# retries) routinely trips a 429 within a couple of seconds. Retrying
# immediately just repeats the same 429 until the caller gives up, which is
# exactly what made free-model agent runs fail outright even though the
# model itself was perfectly capable. Local llama-server never rate-limits,
# so this backoff is a no-op cost there — it only ever engages when the
# server actually returns 429/502/503/504.
#
# A free-tier 429 clears within a few seconds in the overwhelming majority
# of cases — the old schedule (5 retries doubling 2s->20s) summed to a ~50s
# worst-case stall on a SINGLE interactive question before giving up, which
# reads as "the app is frozen" long before it reads as "still retrying."
# This schedule still absorbs the same kind of burst (four attempts inside
# 15s) but gives up 3x faster once it's clear the limit isn't clearing, so
# a genuinely stuck call fails fast enough to retry/report instead of
# silently eating a minute of wall-clock time.
_RATE_LIMIT_RETRIES = 4
_RATE_LIMIT_BASE_DELAY_S = 1.0
_RATE_LIMIT_MAX_DELAY_S = 12.0
_RATE_LIMITED_STATUS_CODES = (429, 502, 503, 504)

# OpenRouter's free-tier routing intermittently fails to forward the
# Authorization header to whichever underlying provider it picked for that
# one request, coming back as a 401 for a key that is actually fine — the
# same request against the same key succeeds a moment later. A real bad/
# revoked key still fails, just one short retry later, so this is a small
# fixed cost for a real resilience win against a documented free-tier
# quirk. Deliberately NOT the exponential 429 schedule above: backing off
# doesn't make an actually-invalid key valid, so there is nothing to gain
# from waiting longer than it takes to rule out "just this one request."
_AUTH_RETRY_ATTEMPTS = 1
_AUTH_RETRY_DELAY_S = 1.5


class LocalChatExpert(BaseExpert):
    """
    Generic expert for any OpenAI-compatible local inference server.
    Talks to POST /v1/chat/completions.
    """

    version            = "1.1"
    cost_per_1k_tokens = 0.0   # local inference — no per-token cost
    supports_streaming = True

    def __init__(
        self,
        model:          str,
        base_url:       str           = "http://localhost:8080/v1",
        name:           Optional[str] = None,
        domain:         str           = "general",
        description:    Optional[str] = None,
        system_prompt:  Optional[str] = None,
        temperature:    float         = 0.7,
        top_p:          float         = 0.9,
        top_k:          int           = 40,
        repeat_penalty: float         = 1.15,
        max_tokens:     int           = 4096,
        stop:           Optional[List[str]] = None,
        api_key:        str           = "not-needed",
        timeout_s:      float         = 240.0,
        extra_body:     Optional[Dict[str, Any]] = None,
    ):
        """
        Parameters
        ----------
        model           Model identifier as the server knows it.
        base_url        Root of the OpenAI-compatible API (must end before /chat).
        name            Expert name in Orcha's registry. Auto-derived if None.
        domain          Routing domain (general, code, reasoning, …).
        description     Human-readable description used by the selector.
        system_prompt   Optional system message prepended to every call.
        temperature     Sampling temperature (0 = greedy, 1 = creative).
        top_p           Nucleus sampling threshold.
        top_k           Top-k sampling cutoff. Narrows the candidate pool
                        per token, which meaningfully reduces small models'
                        tendency to loop/repeat compared to top_p alone.
        repeat_penalty  Penalizes tokens already seen in the response so
                        far. This is the single most important knob for
                        preventing degenerate repetition loops ("the first
                        part. the first part. ...") that small models are
                        especially prone to without it. 1.0 = no penalty;
                        1.15 is a solid general-purpose default that most
                        llama.cpp-family servers also ship as their own
                        built-in default for exactly this reason.
        max_tokens      Maximum tokens in the generated response — also
                        acts as a hard backstop against runaway generation
                        even if repeat_penalty alone doesn't fully prevent
                        a loop. If a completion is cut off by this limit
                        (finish_reason == "length"), execute()/chat_completion()
                        automatically retry once with a larger budget so the
                        answer is not silently truncated.
        stop            Extra stop sequences. A couple of generic ones are
                        always included to catch a specific failure mode
                        seen with small instruct models that misinterpret
                        the chat template and start narrating meta-commentary
                        ("external", "let me know if...") instead of
                        answering — once that pattern starts, halting
                        early is better than letting it run to max_tokens.
        api_key         Bearer token. Most local servers ignore this.
        timeout_s       Per-call HTTP timeout.
        extra_body      Any additional JSON fields to include in the request
                        body (e.g. llama.cpp-specific options). Values here
                        override the named parameters above if both are set.
        """
        self.model          = model
        self.base_url       = base_url.rstrip("/")
        self.system_prompt  = system_prompt
        self.temperature    = temperature
        self.top_p          = top_p
        self.top_k          = top_k
        self.repeat_penalty = repeat_penalty
        self.max_tokens     = max_tokens
        self.stop           = list(stop) if stop else []
        self.api_key        = api_key
        self.timeout_s      = timeout_s
        self.extra_body     = extra_body or {}

        self.name        = name or f"local_{_slugify(model)}"
        self.domain      = domain
        self.description = description or f"Local model '{model}' at {base_url}"

        # Lazily-created per-expert httpx client: connections/TLS setup are
        # reused across model calls instead of rebuilt for every request.
        # The server process lives as long as the app, so the client is
        # intentionally never closed in production (aclose() exists for
        # tests/embedders that manage lifetimes).
        self._client_cache = None

        # Set by execute_stream() on the final SSE chunk. Used by the server
        # to detect truncation (finish_reason == "length") on the streaming
        # path, which cannot re-run mid-stream.
        self.last_finish_reason: str = "stop"

        # Model capability mapping: reasoning-focused model families (R1,
        # QwQ, o-series, ...) advertise deep native reasoning; everything
        # else gets orchestration-emulated depth from the reasoning level.
        from ..capabilities.reasoning import model_supports_native_reasoning
        self.reasoning_depth = "deep" if model_supports_native_reasoning(model) else "medium"

    def capabilities(self) -> ExpertCapability:
        return ExpertCapability(
            domain=self.domain,
            reasoning_depth=self.reasoning_depth,
        )

    def _build_request(
        self,
        query: str,
        stream: bool = False,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        msgs: List[Dict[str, str]] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        if messages:
            msgs.extend(messages)
        msgs.append({"role": "user", "content": query})

        return self._build_completion_request(msgs, stream=stream)

    def _build_completion_request(
        self,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Build a chat-completions body from an explicit message list."""
        full_messages = list(messages)
        if system:
            full_messages = [{"role": "system", "content": system}, *full_messages]

        body: Dict[str, Any] = {
            "model":          self.model,
            "messages":       full_messages,
            "temperature":    self.temperature,
            "top_p":          self.top_p,
            "max_tokens":     self.max_tokens,
            "stream":         stream,
        }
        # Ollama-specific parameters: only include for localhost endpoints
        # (Ollama, llama.cpp, LM Studio). Cloud providers (OpenRouter, Groq,
        # OpenAI, Together, etc.) reject unknown parameters with 400 errors.
        if self._is_local_endpoint():
            body["top_k"] = self.top_k
            body["repeat_penalty"] = self.repeat_penalty
        if self.stop:
            body["stop"] = self.stop
        if tools:
            body["tools"] = tools
        body.update(self.extra_body)
        body.update(overrides)
        return body

    async def _post_with_backoff(self, client, url: str, json_body: Dict[str, Any], headers: Dict[str, str]):
        """
        POSTs with exponential backoff on rate-limit (429) and transient
        server errors (502/503/504), honoring a numeric ``Retry-After``
        header when the server sends one. All other error statuses raise
        immediately via ``raise_for_status`` — this only exists to survive
        the bursty, short-lived 429s that free/rate-limited cloud tiers
        return under normal agent-loop traffic, not to mask real failures.
        """
        delay = _RATE_LIMIT_BASE_DELAY_S
        auth_retries_left = _AUTH_RETRY_ATTEMPTS
        resp = None
        for attempt in range(_RATE_LIMIT_RETRIES + 1):
            resp = await client.post(url, json=json_body, headers=headers)
            # Test doubles (see tests/test_local_chat.py's _FakeResponse) only
            # implement raise_for_status()/json() — a real httpx.Response
            # always has status_code, so treat one without it as a plain
            # success rather than requiring every test fixture to grow one.
            status_code = getattr(resp, "status_code", 200)
            if status_code == 401 and auth_retries_left > 0:
                auth_retries_left -= 1
                await asyncio.sleep(_AUTH_RETRY_DELAY_S)
                continue
            if status_code not in _RATE_LIMITED_STATUS_CODES:
                resp.raise_for_status()
                return resp
            # Detect daily/cap rate limits (X-RateLimit-Remaining=0 with a
            # far-future reset): retrying with backoff just wastes more of
            # the daily quota — surface the error immediately.
            resp_headers = getattr(resp, "headers", {}) or {}
            remaining = resp_headers.get("x-ratelimit-remaining")
            reset_ms = resp_headers.get("x-ratelimit-reset")
            if remaining == "0" and reset_ms:
                try:
                    reset_ts = int(reset_ms) / 1000.0
                    wait_s = max(0, reset_ts - time.time())
                except (ValueError, TypeError):
                    wait_s = 0
                # If the reset is more than 60s away, this is a daily cap —
                # don't burn more requests retrying.
                if wait_s > 60:
                    break
            if attempt >= _RATE_LIMIT_RETRIES:
                break
            retry_after = resp_headers.get("retry-after")
            try:
                wait_s = float(retry_after) if retry_after else delay
            except ValueError:
                wait_s = delay
            await asyncio.sleep(min(wait_s, _RATE_LIMIT_MAX_DELAY_S))
            delay = min(delay * 2, _RATE_LIMIT_MAX_DELAY_S)
        resp.raise_for_status()
        return resp

    async def _post_with_retry(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST /chat/completions and, when the model hit its token budget
        (finish_reason == "length") without producing tool calls, re-run at a
        larger max_tokens so long answers are not silently truncated. Stops
        growing at ``_MAX_GENERATION_TOKENS``.
        """
        attempt = dict(body)
        client = self._get_client()
        for _ in range(_TRUNCATION_RETRIES + 1):
            resp = await self._post_with_backoff(
                client,
                f"{self.base_url}/chat/completions",
                attempt,
                self._headers(),
            )
            data = resp.json()

            choices = data.get("choices") or []
            if not choices:
                return data
            choice = choices[0]
            if choice.get("finish_reason") != "length":
                return data
            if choice.get("message", {}).get("tool_calls"):
                return data

            current = int(attempt.get("max_tokens", 0) or self.max_tokens)
            if current >= _MAX_GENERATION_TOKENS:
                return data
            attempt = {
                **attempt,
                "max_tokens": min(_MAX_GENERATION_TOKENS, current * 3),
            }
        return data

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """
        Native OpenAI-style chat completion returning the full assistant
        message object (including ``tool_calls`` when the model invokes a
        tool). If the backend rejects the ``tools`` field (older servers),
        retries once without it so plain chat still works.

        The returned dict also carries ``finish_reason`` so callers (e.g. the
        AgentNode loop) can detect a length-truncated answer.
        """
        body = self._build_completion_request(
            messages, stream=False, tools=tools, system=system, **overrides
        )

        try:
            data = await self._post_with_retry(body)
        except Exception:
            if not tools:
                raise
            # Retry without tools — the backend may not support them.
            body_no_tools = self._build_completion_request(
                messages, stream=False, system=system, **overrides
            )
            data = await self._post_with_retry(body_no_tools)

        choices = data.get("choices") or []
        if not choices:
            return {"content": "", "role": "assistant", "finish_reason": "stop"}
        message = choices[0].get("message") or {"content": "", "role": "assistant"}
        message["finish_reason"] = choices[0].get("finish_reason", "stop")
        # Text-level recovery: when the backend left the tool invocation
        # inside content (no native parsing), extract it so the agent loop
        # can still execute the call. Only attempted when tools were part
        # of the request — a plain chat answer mentioning "name" must not
        # be mistaken for an invocation.
        if tools and not message.get("tool_calls"):
            content = message.get("content") or ""
            recovered = extract_text_tool_calls(content)
            if recovered:
                message["tool_calls"] = recovered
                # Keep any surrounding prose, but drop the raw JSON payload
                # the agent already consumed as a tool call.
                message["content"] = strip_text_tool_calls(content)
        return message

    def _is_local_endpoint(self) -> bool:
        """Detect whether base_url points to a local server (Ollama, llama.cpp,
        LM Studio, etc.) vs a cloud provider (OpenRouter, Groq, OpenAI, etc.).
        Local servers accept Ollama-specific params (top_k, repeat_penalty);
        cloud providers reject them with 400 errors."""
        url = (self.base_url or "").lower()
        return (
            "localhost" in url
            or "127.0.0.1" in url
            or "0.0.0.0" in url
            or ":8080" in url
            or ":11434" in url
            or ":8081" in url
            or ":8082" in url
            or ":8083" in url
        )

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

    def _get_client(self):
        """The expert's reused httpx client (created lazily on first use)."""
        if self._client_cache is None:
            import httpx  # soft dep
            self._client_cache = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client_cache

    async def aclose(self) -> None:
        """Release the cached client. Called by embedders/tests managing
        lifetimes; the server process never closes experts."""
        if self._client_cache is not None:
            await self._client_cache.aclose()
            self._client_cache = None

    async def prewarm_connection(self) -> None:
        """
        Opens (and pools, via the same cached client execute()/execute_stream()
        reuse) a connection to base_url ahead of the user's first real message.
        Without this, a fresh app session's very first chat pays a full cold
        DNS+TCP+TLS handshake to the BYOK endpoint on top of the model's own
        generation time — a fixed cost worth paying at startup instead, while
        nothing is waiting on it. Best-effort: failures here are silent since
        the real request will simply open the connection itself if this
        didn't get there first.
        """
        try:
            client = self._get_client()
            await client.get(
                f"{self.base_url}/models", headers=self._headers(), timeout=5.0
            )
        except Exception:
            pass

    async def execute(
        self,
        query: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> ExpertOutput:
        t0 = time.perf_counter()
        body = self._build_request(query, stream=False, messages=messages)
        data = await self._post_with_retry(body)

        choice         = data["choices"][0]
        answer         = (choice.get("message", {}).get("content") or "").strip()
        finish_reason  = choice.get("finish_reason", "stop")
        usage          = data.get("usage") or {}
        tokens         = usage.get("total_tokens", 0)
        latency_s      = time.perf_counter() - t0

        # Confidence proxy: natural stop > length-truncated > other
        if finish_reason == "stop":
            confidence = 0.80
        elif finish_reason == "length":
            confidence = 0.62  # answer may be truncated
        else:
            confidence = 0.65

        return ExpertOutput(
            answer=answer,
            confidence=confidence,
            tokens_used=tokens,
            latency_s=latency_s,
            finish_reason=finish_reason,
            model_version=data.get("model", self.model),
            metadata={
                "prompt_tokens":     usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            },
        )

    async def execute_stream(
        self,
        query: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Real token-by-token streaming via the server's SSE endpoint.

        Every backend named in this module's docstring (llama.cpp,
        LM Studio, vLLM, text-gen-webui, LocalAI, Jan, Kobold.cpp) speaks
        the same OpenAI-compatible chunk format:
            data: {"choices":[{"delta":{"content":"..."}}]}
            data: [DONE]
        so one parser here covers all of them.

        The final chunk's ``finish_reason`` is recorded on
        ``self.last_finish_reason`` so callers can detect a length-truncated
        answer (the streaming path cannot re-run mid-stream).
        """
        self.last_finish_reason = "stop"
        body = self._build_request(query, stream=True, messages=messages)

        client = self._get_client()
        # Retry only the connection attempt on 429/5xx/401 — nothing has been
        # yielded yet at that point, so it's always safe to redo. Once a
        # stream actually starts (status 2xx), it runs to completion; a mid-
        # stream drop is a different failure mode this doesn't attempt to fix.
        delay = _RATE_LIMIT_BASE_DELAY_S
        auth_retries_left = _AUTH_RETRY_ATTEMPTS
        for attempt in range(_RATE_LIMIT_RETRIES + 1):
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=body,
                headers=self._headers(),
            ) as resp:
                # See _post_with_backoff's comment: BYOK free-tier routing
                # (OpenRouter especially) occasionally 401s a request that
                # was never actually unauthorized — one quick retry, not
                # the exponential schedule, since backing off longer
                # doesn't help an auth failure either way.
                if resp.status_code == 401 and auth_retries_left > 0:
                    auth_retries_left -= 1
                    await resp.aclose()
                    await asyncio.sleep(_AUTH_RETRY_DELAY_S)
                    continue
                if resp.status_code in _RATE_LIMITED_STATUS_CODES and attempt < _RATE_LIMIT_RETRIES:
                    # Detect daily/cap rate limits — don't burn quota retrying.
                    remaining = resp.headers.get("x-ratelimit-remaining")
                    reset_ms = resp.headers.get("x-ratelimit-reset")
                    if remaining == "0" and reset_ms:
                        try:
                            reset_ts = int(reset_ms) / 1000.0
                            wait_s = max(0, reset_ts - time.time())
                        except (ValueError, TypeError):
                            wait_s = 0
                        if wait_s > 60:
                            break
                    retry_after = resp.headers.get("retry-after")
                    try:
                        wait_s = float(retry_after) if retry_after else delay
                    except ValueError:
                        wait_s = delay
                    await resp.aclose()
                    await asyncio.sleep(min(wait_s, _RATE_LIMIT_MAX_DELAY_S))
                    delay = min(delay * 2, _RATE_LIMIT_MAX_DELAY_S)
                    continue
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    finish = choices[0].get("finish_reason")
                    if finish:
                        self.last_finish_reason = finish
                    piece = delta.get("content")
                    if piece:
                        yield piece
                return

    async def healthcheck(self) -> bool:
        """
        Checks the /models endpoint. Returns True if it responds with 200.
        When the server enumerates its models, also require ``self.model``
        to be present — some servers (llama.cpp included) return 200 on
        /models even when the requested model failed to load, which would
        otherwise report 'ready' for a model that rejects every request.
        """
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                if resp.status_code != 200:
                    return False
                payload = resp.json()
                models = payload.get("data") or []
                if not models:
                    # Server doesn't enumerate models — 200 is the best signal.
                    return True
                requested = self.model.lower()
                for item in models:
                    model_id = str(item.get("id") or "").lower()
                    # Exact id, or a file-name suffix match for servers that
                    # return a different alias (e.g. a path).
                    if model_id == requested or model_id.endswith("/" + requested):
                        return True
                return False
        except Exception:
            return False

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "model":       self.model,
            "base_url":    self.base_url,
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
        })
        return base
