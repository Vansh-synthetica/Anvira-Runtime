"""
Tests for LocalChatExpert response-completeness handling: the token-budget
retry (finish_reason == "length"), finish_reason surfacing, and history
plumbing. A fake httpx.AsyncClient is injected so no server is contacted.
"""
import asyncio

import httpx
import pytest

from orcha.experts import local_chat
from orcha.experts.local_chat import LocalChatExpert, _MAX_GENERATION_TOKENS


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.posts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json, headers):
        self.posts.append(json)
        data = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return _FakeResponse(data)


def _resp(content, finish_reason, tool_calls=None):
    message = {"content": content, "role": "assistant"}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": finish_reason}], "model": "fake"}


def _patch_client(monkeypatch, responses):
    client = _FakeClient(responses)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: client)
    return client


def test_chat_completion_retries_with_bigger_budget(monkeypatch):
    client = _patch_client(monkeypatch, [
        _resp("truncated...", "length"),
        _resp("full answer", "stop"),
    ])
    expert = LocalChatExpert(model="fake", base_url="http://x/v1")
    msg = asyncio.run(expert.chat_completion([{"role": "user", "content": "hi"}]))

    assert msg["content"] == "full answer"
    assert msg["finish_reason"] == "stop"
    assert len(client.posts) == 2
    assert client.posts[0]["max_tokens"] == 4096
    assert client.posts[1]["max_tokens"] == min(_MAX_GENERATION_TOKENS, 4096 * 3)


def test_chat_completion_stops_at_generation_ceiling(monkeypatch):
    client = _patch_client(monkeypatch, [_resp("still truncated", "length")])
    expert = LocalChatExpert(model="fake", base_url="http://x/v1", max_tokens=_MAX_GENERATION_TOKENS)
    msg = asyncio.run(expert.chat_completion([{"role": "user", "content": "hi"}]))

    assert msg["finish_reason"] == "length"  # surfaced, not silently swallowed
    assert len(client.posts) == 1


def test_chat_completion_does_not_retry_when_tool_calls_present(monkeypatch):
    client = _patch_client(monkeypatch, [
        _resp(None, "length", tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "write_file", "arguments": "{}"},
        }]),
    ])
    expert = LocalChatExpert(model="fake", base_url="http://x/v1")
    msg = asyncio.run(expert.chat_completion(
        [{"role": "user", "content": "hi"}], tools=[{"type": "function"}]
    ))

    assert msg["finish_reason"] == "length"
    assert msg["tool_calls"]
    assert len(client.posts) == 1  # never grew the budget on a tool call


def test_execute_passes_history_messages(monkeypatch):
    client = _patch_client(monkeypatch, [_resp("ok", "stop")])
    expert = LocalChatExpert(model="fake", base_url="http://x/v1")
    out = asyncio.run(expert.execute(
        "current", messages=[{"role": "assistant", "content": "prev"}]
    ))

    assert out.answer == "ok"
    posted = client.posts[0]["messages"]
    # Default expert has no system_prompt, so history leads the message list.
    assert posted[0] == {"role": "assistant", "content": "prev"}
    assert posted[-1] == {"role": "user", "content": "current"}


def test_execute_surfaces_finish_reason(monkeypatch):
    _patch_client(monkeypatch, [_resp("partial", "length")])
    expert = LocalChatExpert(model="fake", base_url="http://x/v1")
    out = asyncio.run(expert.execute("hi"))
    assert out.finish_reason == "length"


class _FakeStatusResponse:
    """Response double that actually carries a status_code, for exercising
    the 429/401 retry branches — the plain _FakeResponse above always
    looks like a 200 to _post_with_backoff's getattr(resp, "status_code",
    200) fallback."""

    def __init__(self, status_code, data=None):
        self.status_code = status_code
        self.headers = {}
        self._data = data or _resp("ok", "stop")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)  # type: ignore[arg-type]

    def json(self):
        return self._data


class _FakeStatusClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.posts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json, headers):
        self.posts.append(json)
        return self._responses.pop(0) if self._responses else self._responses[-1]


async def _instant_sleep(*_args, **_kwargs):
    return None


def _patch_status_client(monkeypatch, responses):
    client = _FakeStatusClient(responses)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: client)
    # The 401/429 retry paths sleep between attempts — real delays would
    # make this test slow for no reason, so collapse them to instant. Must
    # NOT reference asyncio.sleep from inside the replacement itself: once
    # patched, that name resolves to this same replacement, so calling it
    # again recurses forever instead of ever actually returning.
    monkeypatch.setattr(local_chat.asyncio, "sleep", _instant_sleep)
    return client


def test_execute_retries_once_on_transient_401(monkeypatch):
    # OpenRouter's free-tier routing occasionally 401s a request that was
    # never actually unauthorized — one quick retry should recover it.
    client = _patch_status_client(monkeypatch, [
        _FakeStatusResponse(401),
        _FakeStatusResponse(200, _resp("full answer", "stop")),
    ])
    expert = LocalChatExpert(model="fake", base_url="http://x/v1")
    out = asyncio.run(expert.execute("hi"))

    assert out.answer == "full answer"
    assert len(client.posts) == 2


def test_execute_raises_after_repeated_401(monkeypatch):
    # A genuinely bad/revoked key must still fail — just after the one
    # short retry, not be masked or retried forever.
    client = _patch_status_client(monkeypatch, [
        _FakeStatusResponse(401),
        _FakeStatusResponse(401),
    ])
    expert = LocalChatExpert(model="fake", base_url="http://x/v1")

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(expert.execute("hi"))
    assert len(client.posts) == 2


class _FakeGetResponse:
    def __init__(self, status_code, data=None):
        self.status_code = status_code
        self._data = data or {}

    def json(self):
        return self._data


class _FakeGetClient:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, headers):
        return self._response


def _patch_get_client(monkeypatch, response):
    client = _FakeGetClient(response)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: client)


def test_healthcheck_requires_listed_model(monkeypatch):
    _patch_get_client(monkeypatch, _FakeGetResponse(
        200, {"data": [{"id": "other-model"}]}
    ))
    expert = LocalChatExpert(model="my-model", base_url="http://x/v1")
    assert asyncio.run(expert.healthcheck()) is False


def test_healthcheck_accepts_listed_model(monkeypatch):
    _patch_get_client(monkeypatch, _FakeGetResponse(
        200, {"data": [{"id": "my-model"}]}
    ))
    expert = LocalChatExpert(model="my-model", base_url="http://x/v1")
    assert asyncio.run(expert.healthcheck()) is True


def test_healthcheck_accepts_path_alias(monkeypatch):
    _patch_get_client(monkeypatch, _FakeGetResponse(
        200, {"data": [{"id": "models/MyFolder/my-model.gguf"}]}
    ))
    expert = LocalChatExpert(model="my-model.gguf", base_url="http://x/v1")
    assert asyncio.run(expert.healthcheck()) is True


def test_healthcheck_tolerates_non_enumerating_server(monkeypatch):
    _patch_get_client(monkeypatch, _FakeGetResponse(200, {"data": []}))
    expert = LocalChatExpert(model="my-model", base_url="http://x/v1")
    assert asyncio.run(expert.healthcheck()) is True


def test_healthcheck_false_on_server_error(monkeypatch):
    _patch_get_client(monkeypatch, _FakeGetResponse(500))
    expert = LocalChatExpert(model="my-model", base_url="http://x/v1")
    assert asyncio.run(expert.healthcheck()) is False
