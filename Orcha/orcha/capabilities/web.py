"""
orcha.capabilities.web
=======================
Read-the-open-internet tools for local models: ``web_search`` (DuckDuckGo
HTML, falling back to Bing HTML — no API key required for either) and
``web_fetch`` (download a URL and extract its readable text, scripts/nav/
footer stripped out).

This is what gives a local model the same "look something up, then read
the page" ability Claude/Antigravity-style assistants get from their own
hosted web tools — Orcha's agent loop already knows how to call multiple
tools in sequence (search, then fetch a promising result), it just had
nothing that reached outside the workspace until now.

Both tools are tagged NETWORK/CAUTIOUS. PermissionPolicy already treats any
NETWORK-permission tool as needing approval outside action/full/auto access
modes (see ToolSpec.needs_approval in orcha/nodes/tool.py) — the exact same
mechanism run_command/git_push are gated by — so this reuses that gate
instead of inventing a second one.

Resilience, matching the same shape LocalChatExpert._post_with_backoff
already uses for the completion path (see orcha/experts/local_chat.py):
retry transient failures (timeouts, connection errors, 429/5xx) with
short exponential backoff before giving up, and — for search specifically
— fall back to a second, independent backend rather than depending on one
provider's markup never changing or never rate-limiting. Nothing here can
make a genuinely dead URL respond, or work with zero network connectivity;
what it CAN guarantee is that transient failures are retried, one search
provider's outage doesn't take down search entirely, and every failure
still comes back as a normal, structured ErrorObservation the agent can
read and react to — never an unhandled exception that kills the run (that
guarantee is Tool.run()'s, in tools.py, and applies to every tool here).
"""
from __future__ import annotations

import base64
import re
import time
from typing import Any, Callable, Dict, List
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from ..nodes.tool import ToolSpec
from .base import CAUTIOUS, NETWORK, CapabilityContext, ToolError, spec

CAPABILITY_NAME = "web"
CAPABILITY_LABEL = "Web"
CAPABILITY_DESCRIPTION = (
    "Search the open web and fetch page content — the only tools here that "
    "reach outside the local workspace."
)

_NET_PERMS = [NETWORK]
_TIMEOUT_S = 12
_USER_AGENT = "Mozilla/5.0 (compatible; AnviraAgent/1.0; +local-tool-use)"
_MAX_FETCH_CHARS = 12000
_MAX_SEARCH_RESULTS = 10

# ── Retry/backoff, mirroring LocalChatExpert's completion-path tuning ────────
_RETRIES = 2                    # up to 3 attempts total per request
_BASE_DELAY_S = 0.6
_MAX_DELAY_S = 4.0
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _request_with_retry(method: str, url: str, **kwargs: Any) -> requests.Response:
    """requests.get/post with exponential backoff on timeouts, connection
    errors, and rate-limit/transient-server statuses — honors a numeric
    Retry-After header when the server sends one. Anything else (4xx other
    than 429, a genuine DNS failure repeated across all attempts) raises
    after the last attempt so callers still see a real, specific error."""
    delay = _BASE_DELAY_S
    last_exc: Exception | None = None
    for attempt in range(_RETRIES + 1):
        try:
            resp = requests.request(method, url, timeout=_TIMEOUT_S, **kwargs)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= _RETRIES:
                raise
            time.sleep(min(delay, _MAX_DELAY_S))
            delay *= 2
            continue

        if resp.status_code not in _RETRYABLE_STATUS:
            resp.raise_for_status()
            return resp
        if attempt >= _RETRIES:
            resp.raise_for_status()
            return resp
        retry_after = resp.headers.get("retry-after")
        try:
            wait_s = float(retry_after) if retry_after else delay
        except ValueError:
            wait_s = delay
        time.sleep(min(wait_s, _MAX_DELAY_S))
        delay *= 2
    if last_exc:
        raise last_exc
    raise RuntimeError("unreachable")  # pragma: no cover — loop always returns or raises above


def _parse_duckduckgo(html: bytes, limit: int) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    results: List[Dict[str, str]] = []
    for row in soup.select(".result"):
        link = row.select_one(".result__a")
        if not link or not link.get("href"):
            continue
        title = link.get_text(strip=True)
        href = link["href"]
        if not title or not href:
            continue
        snippet_el = row.select_one(".result__snippet")
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
        results.append({"title": title, "url": href, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


def _decode_bing_redirect(href: str) -> str:
    """Bing wraps every result link in a bing.com/ck/a click-tracking
    redirect instead of the real URL — confirmed live: web_fetch on the
    raw href would hit Bing's redirect page, not the actual site. The
    destination is recoverable without a network round trip: it's the
    `u` query param, formatted as the literal prefix "a1" followed by
    unpadded base64url of the real URL (also confirmed live, decoding
    "a1aHR0cHM6Ly93d3cucHl0aG9uLm9yZy8" to "https://www.python.org/").
    Falls back to the original href for any link that isn't this shape
    (already a plain URL, or a redirect format that changes later)."""
    try:
        parsed = urlparse(href)
        if parsed.netloc.endswith("bing.com") and parsed.path == "/ck/a":
            encoded = parse_qs(parsed.query).get("u", [""])[0]
            if encoded.startswith("a1"):
                encoded = encoded[2:]
                padded = encoded + "=" * (-len(encoded) % 4)
                decoded = base64.urlsafe_b64decode(padded).decode("utf-8", errors="strict")
                if decoded.startswith("http://") or decoded.startswith("https://"):
                    return decoded
    except Exception:
        pass
    return href


def _parse_bing(html: bytes, limit: int) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    results: List[Dict[str, str]] = []
    for row in soup.select("li.b_algo"):
        link = row.select_one("h2 a")
        if not link or not link.get("href"):
            continue
        title = link.get_text(strip=True)
        href = _decode_bing_redirect(link["href"])
        if not title or not href:
            continue
        snippet_el = row.select_one(".b_caption p") or row.select_one(".b_caption")
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
        results.append({"title": title, "url": href, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


# Two independent providers, tried in order. Each is a (label, request,
# parser) triple — request raises on total failure, parser returns []
# when the markup didn't match (a layout change, not necessarily an
# error). A provider that returns a genuinely empty page for a real query
# is indistinguishable from "no results" at this layer, which is why the
# fallback keys off EITHER an exception OR a zero-result parse: a query
# that legitimately has no results will also come back empty from the
# second provider, at which point the empty result is trustworthy rather
# than assumed broken.
_SEARCH_BACKENDS: List[tuple[str, Callable[[str], requests.Response], Callable[[bytes, int], List[Dict[str, str]]]]] = [
    (
        "duckduckgo",
        lambda q: _request_with_retry(
            "POST", "https://html.duckduckgo.com/html/",
            data={"q": q}, headers={"User-Agent": _USER_AGENT},
        ),
        _parse_duckduckgo,
    ),
    (
        "bing",
        lambda q: _request_with_retry(
            "GET", "https://www.bing.com/search",
            params={"q": q}, headers={"User-Agent": _USER_AGENT},
        ),
        _parse_bing,
    ),
]


def _web_search():
    def tool(query: str, limit: int = 5) -> Dict[str, Any]:
        q = str(query or "").strip()
        if not q:
            raise ToolError("invalid_query", "query must not be empty")
        limit = max(1, min(int(limit or 5), _MAX_SEARCH_RESULTS))

        errors: List[str] = []
        for backend_name, do_request, parse in _SEARCH_BACKENDS:
            try:
                resp = do_request(q)
            except requests.RequestException as exc:
                errors.append(f"{backend_name}: {exc}")
                continue
            # Parse raw bytes, not resp.text: requests falls back to
            # ISO-8859-1 whenever a response doesn't declare a charset in
            # its Content-Type header, which silently mangles non-ASCII
            # characters. BeautifulSoup's own encoding sniffer (from the
            # bytes and any <meta charset>) gets this right.
            results = parse(resp.content, limit)
            if results:
                return {"query": q, "results": results, "count": len(results), "source": backend_name}
            errors.append(f"{backend_name}: 0 results parsed")

        # Every backend either errored or came back empty — a real,
        # structured failure the agent can see and react to (try a
        # different query, or tell the user), not a silent empty success
        # that looks the same as a legitimately zero-result query.
        raise ToolError(
            "search_unavailable",
            f"Web search failed on every backend for '{q}': " + "; ".join(errors),
        )
    return tool


_MAX_LINKS = 40
_SKIP_LINK_SCHEMES = ("mailto:", "javascript:", "tel:", "data:")


def _extract_links(soup: "BeautifulSoup", base_url: str) -> List[Dict[str, str]]:
    """Every distinct, followable link on the page, resolved to an absolute
    URL against `base_url` (so a relative href like "/docs/next" becomes a
    real URL web_fetch can be called on again) — this is what makes
    multi-hop crawling possible: the agent reads a page, sees its outbound
    links, and decides which (if any) to fetch next in a following tool
    call, the same composable way web_search + web_fetch already work
    together. Capped and deduped so a link-heavy page doesn't blow the
    token budget with navigation cruft repeated many times over."""
    seen: set = set()
    links: List[Dict[str, str]] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#"):
            continue
        if href.lower().startswith(_SKIP_LINK_SCHEMES):
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https"):
            continue
        # Drop the fragment (#section) for dedup purposes — the same page
        # linked from ten different anchors is one followable destination.
        absolute = absolute.split("#", 1)[0]
        if absolute in seen:
            continue
        seen.add(absolute)
        text = a.get_text(strip=True)[:120]
        links.append({"text": text or absolute, "url": absolute})
        if len(links) >= _MAX_LINKS:
            break
    return links


def _web_fetch():
    def tool(url: str, max_chars: int = 4000) -> Dict[str, Any]:
        target = str(url or "").strip()
        if not target:
            raise ToolError("invalid_url", "url must not be empty")
        parsed = urlparse(target)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ToolError("invalid_url", f"'{target}' is not a valid http(s) URL")

        try:
            resp = _request_with_retry(
                "GET", target,
                headers={"User-Agent": _USER_AGENT},
                allow_redirects=True,
            )
        except requests.RequestException as exc:
            raise ToolError("network_error", f"Failed to fetch {target}: {exc}")

        content_type = resp.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            raise ToolError(
                "unsupported_content_type",
                f"'{target}' returned '{content_type or 'unknown'}' content — not a readable page.",
            )

        # resp.url (not the requested `target`) is the base for both link
        # resolution and the returned `url` field — a redirect chain means
        # relative hrefs on the final page are relative to where it
        # actually landed, not where the request started.
        final_url = str(resp.url) if getattr(resp, "url", None) else target
        soup = BeautifulSoup(resp.content, "html.parser")
        links = _extract_links(soup, final_url)
        for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
            tag.decompose()
        title = soup.title.get_text(strip=True) if soup.title else ""
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))

        cap = max(500, min(int(max_chars or 4000), _MAX_FETCH_CHARS))
        truncated = len(text) > cap
        text = text[:cap]

        return {
            "url": final_url,
            "title": title,
            "text": text,
            "truncated": truncated,
            "links": links,
        }
    return tool


def build_tools(ctx: CapabilityContext) -> List[ToolSpec]:
    return [
        spec(
            "web_search",
            "Search the REAL INTERNET (DuckDuckGo, falling back to Bing if that "
            "fails) — not the local workspace — and get back result titles, "
            "URLs, and snippets. USE THIS WHEN: the request mentions 'the web', "
            "'online', 'the internet', or asks for current information / facts "
            "you don't already know / anything outside this project's own "
            "files. For anything about THIS project's own code, use search_text "
            "or grep instead — this tool never looks at local files. Does NOT "
            "return full page content — pair it with web_fetch on whichever "
            "result actually looks worth reading.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, e.g. 'FastAPI streaming response example'."},
                    "limit": {"type": "integer", "description": "Max results to return (1-10). Default 5."},
                },
                "required": ["query"],
            },
            _web_search(), permissions=_NET_PERMS, safety_level=CAUTIOUS, capability=CAPABILITY_NAME,
            result_format={"type": "object", "properties": {"results": {"type": "array"}, "count": {"type": "integer"}}},
        ),
        spec(
            "web_fetch",
            "Fetch a URL and extract its readable text (scripts, styles, nav and "
            "footer stripped out), PLUS every followable link on the page "
            "(resolved to full URLs). USE THIS WHEN: you have a specific URL — "
            "from web_search or given directly by the user — and need its actual "
            "content, not just a search snippet. USE THE RETURNED 'links' TO "
            "CRAWL: call web_fetch again on a link from this result to follow it "
            "to the next page — e.g. read a docs index, then fetch the specific "
            "page it links to, then a page THAT links to, as many hops as the "
            "task actually needs. There is no built-in depth limit, so decide "
            "when you have enough and stop rather than fetching indefinitely.",
            {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full http(s) URL to fetch, e.g. 'https://example.com/article'."},
                    "max_chars": {"type": "integer", "description": "Max characters of extracted text to return (500-12000). Default 4000."},
                },
                "required": ["url"],
            },
            _web_fetch(), permissions=_NET_PERMS, safety_level=CAUTIOUS, capability=CAPABILITY_NAME,
            result_format={"type": "object", "properties": {"title": {"type": "string"}, "text": {"type": "string"}, "links": {"type": "array"}}},
        ),
    ]
