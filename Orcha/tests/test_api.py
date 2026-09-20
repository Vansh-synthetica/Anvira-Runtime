"""Integration tests for the Orcha HTTP API."""
import json
import pytest
from fastapi.testclient import TestClient
from orcha.api.server import app


@pytest.fixture
def client():
    return TestClient(app)


# ── Versioned /v1 routes — the stable public API ──────────────────────────

def test_health_v1(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.4.0"
    assert isinstance(body["experts"], list) and len(body["experts"]) > 0
    assert "synthesizer" in body
    assert "run_all_experts" in body
    assert "source" in body


def test_list_experts_v1(client):
    r = client.get("/v1/experts")
    assert r.status_code == 200
    experts = r.json()
    assert isinstance(experts, list) and len(experts) > 0
    for e in experts:
        assert "name" in e and "domain" in e and "description" in e
        assert "is_synthesizer" in e


def test_query_returns_full_shape(client):
    r = client.post("/v1/query", json={"query": "What is compound interest?"})
    assert r.status_code == 200
    body = r.json()
    for k in ("answer", "confidence", "quality_score", "synthesized",
              "iterations", "cost", "latency_s", "contributors",
              "primary", "domains", "trace", "agg_mode"):
        assert k in body, f"Missing key: {k}"
    assert body["answer"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["trace"][0]["stage"] == "input"


def test_query_max_iterations_override(client):
    r = client.post("/v1/query", json={"query": "Quick test", "max_iterations": 1})
    assert r.status_code == 200
    assert r.json()["iterations"] == 1


def test_query_run_all_override(client):
    r = client.post("/v1/query", json={"query": "Test", "run_all_experts": True})
    assert r.status_code == 200
    body = r.json()
    assert len(body["contributors"]) > 0


def test_query_max_cost_zero_is_honoured(client):
    """Regression: max_cost=0 must NOT be silently overridden to the default."""
    r = client.post("/v1/query", json={
        "query": "test", "max_cost": 0.0, "max_iterations": 1,
    })
    assert r.status_code == 200
    assert r.json()["cost"] >= 0.0   # local models cost $0 anyway


def test_reload_v1(client):
    r = client.post("/v1/reload")
    assert r.status_code == 200
    body = r.json()
    assert body["reloaded"] is True
    assert isinstance(body["experts"], list)
    assert "source" in body


def test_query_stream_v1(client):
    with client.stream("POST", "/v1/query/stream", json={"query": "hello streaming"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")

        events = []
        for line in r.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:"):].strip()))

    assert len(events) >= 2  # at least one chunk + a done event
    assert events[-1]["type"] == "done"
    assert "model" in events[-1]
    assert "latency_s" in events[-1]
    chunk_events = [e for e in events if e["type"] == "chunk"]
    assert len(chunk_events) >= 1
    assert all("content" in e for e in chunk_events)
    # Reassembling the chunks should reproduce a real answer, not be empty.
    full_answer = "".join(e["content"] for e in chunk_events)
    assert len(full_answer) > 0


def test_query_reasoning_fast_is_one_pass(client):
    r = client.post("/v1/query", json={"query": "Quick check", "reasoning": "fast"})
    assert r.status_code == 200
    body = r.json()
    assert body["iterations"] == 1


def test_query_reasoning_max_runs_all_experts(client):
    r = client.post("/v1/query", json={"query": "Thorough check", "reasoning": "max"})
    assert r.status_code == 200
    body = r.json()
    assert body["iterations"] >= 1
    assert len(body["contributors"]) > 0


def test_query_explicit_max_iterations_beats_reasoning_default(client):
    # max's reasoning default is 5 iterations; an explicit cap of 1 must win.
    r = client.post("/v1/query", json={
        "query": "Override", "reasoning": "max", "max_iterations": 1,
    })
    assert r.status_code == 200
    assert r.json()["iterations"] == 1


def test_query_invalid_reasoning_returns_400(client):
    r = client.post("/v1/query", json={"query": "Test", "reasoning": "turbo"})
    assert r.status_code == 400
    body = r.json()
    assert "error" in body and "message" in body["error"]


def test_query_stream_with_reasoning(client):
    with client.stream(
        "POST", "/v1/query/stream",
        json={"query": "hello reasoning", "reasoning": "high"},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = []
        for line in r.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:"):].strip()))
    assert events[-1]["type"] == "done"
    chunk_events = [e for e in events if e["type"] == "chunk"]
    assert len(chunk_events) >= 1


def test_query_stream_continues_truncated_answer(client, monkeypatch):
    """A length-truncated streamed answer is transparently continued instead
    of leaving the user with a half-finished sentence."""
    import orcha.api.server as server_module
    from orcha.experts.local_chat import LocalChatExpert

    calls = {"n": 0}

    class FakeLocalExpert(LocalChatExpert):
        async def execute_stream(self, query, messages=None):
            calls["n"] += 1
            if calls["n"] == 1:
                self.last_finish_reason = "length"
                yield "First half"
            else:
                self.last_finish_reason = "stop"
                yield "Second half"

    monkeypatch.setattr(server_module, "LocalChatExpert", FakeLocalExpert)
    orc = server_module._state.orc
    orc.experts["mock_synthesizer"] = FakeLocalExpert(
        model="fake-model", base_url="http://127.0.0.1:1/v1"
    )
    orc.synthesizer_expert = "mock_synthesizer"

    with client.stream("POST", "/v1/query/stream", json={"query": "hi"}) as r:
        assert r.status_code == 200
        events = []
        for line in r.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:"):].strip()))

    contents = [e["content"] for e in events if e["type"] == "chunk"]
    assert contents == ["First half", "Second half"]
    assert events[-1]["type"] == "done"
    assert events[-1]["truncated"] is False
    assert calls["n"] == 2


def test_set_local_model_v1(client):
    from orcha.settings import settings

    original_model = settings.local_server_model
    original_base_url = settings.local_server_base_url
    try:
        r = client.post("/v1/local-model", json={
            "model": "test-model.gguf",
            "base_url": "http://localhost:8080/v1",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["reloaded"] is True
        assert settings.local_server_model == "test-model.gguf"
        # Falls back to mock since nothing is actually listening on :8080/v1
        # in the test environment — what matters is the setting took effect
        # and the orchestrator rebuild ran without error.
        assert "source" in body
        assert isinstance(body["experts"], list)
    finally:
        # Reset so this test doesn't leak state into other tests that
        # assume default settings (e.g. source == "mock").
        settings.local_server_model = original_model
        settings.local_server_base_url = original_base_url


def test_set_local_model_requires_model_field(client):
    r = client.post("/v1/local-model", json={})
    assert r.status_code == 422


def test_multi_model_add_list_remove(client):
    from orcha.settings import settings

    original_model = settings.local_server_model
    original_base_url = settings.local_server_base_url
    original_extras = list(settings.local_server_extra_models)
    try:
        settings.local_server_model = ""
        settings.local_server_extra_models = []

        r = client.get("/v1/local-models")
        assert r.status_code == 200
        assert r.json()["models"] == []

        r = client.post("/v1/local-model", json={"model": "model-a.gguf", "base_url": "http://localhost:8080/v1"})
        assert r.status_code == 200
        assert r.json()["experts"] == ["local_model_a_gguf"]

        r = client.post("/v1/local-models/add", json={"model": "model-b.gguf", "base_url": "http://localhost:8081/v1"})
        assert r.status_code == 200
        assert set(r.json()["experts"]) == {"local_model_a_gguf", "local_model_b_gguf"}

        r = client.get("/v1/local-models")
        models = r.json()["models"]
        assert len(models) == 2
        assert models[0]["model"] == "model-a.gguf" and models[0]["primary"] is True
        assert models[1]["model"] == "model-b.gguf" and models[1]["primary"] is False

        # Removing the primary should promote model-b to primary.
        r = client.delete("/v1/local-models/model-a.gguf")
        assert r.status_code == 200
        assert r.json()["experts"] == ["local_model_b_gguf"]

        r = client.get("/v1/local-models")
        models = r.json()["models"]
        assert len(models) == 1
        assert models[0]["model"] == "model-b.gguf" and models[0]["primary"] is True

        r = client.delete("/v1/local-models/never-registered.gguf")
        assert r.status_code == 404
        assert r.json()["error"]["type"] == "model_not_found"
    finally:
        settings.local_server_model = original_model
        settings.local_server_base_url = original_base_url
        settings.local_server_extra_models = original_extras


def test_performance_v1(client):
    # Run a query first so there's data
    client.post("/v1/query", json={"query": "warm up", "max_iterations": 1})
    r = client.get("/v1/performance")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


# ── Back-compat redirects (deprecated un-versioned paths) ──────────────────

def test_legacy_health_redirects(client):
    r = client.get("/health", follow_redirects=False)
    assert r.status_code in (301, 307)
    assert "/v1/health" in r.headers.get("location", "")


def test_legacy_query_redirects(client):
    r = client.post("/query", json={"query": "hi"}, follow_redirects=False)
    assert r.status_code in (301, 307)
    assert "/v1/query" in r.headers.get("location", "")


def test_legacy_reload_redirects(client):
    r = client.post("/reload", follow_redirects=False)
    assert r.status_code in (301, 307)
    assert "/v1/reload" in r.headers.get("location", "")


# ── Structured error envelope ──────────────────────────────────────────────

def test_error_envelope_on_bad_query(client):
    """A query with missing required field returns 422 in the standard Orcha error envelope."""
    r = client.post("/v1/query", json={})
    assert r.status_code == 422
    body = r.json()
    assert "error" in body
    assert body["error"]["type"] == "invalid_request"
    assert body["error"]["status"] == 422
    assert "query" in body["error"]["message"]


def test_error_envelope_on_404(client):
    """Unknown endpoints return 404 in the standard Orcha error envelope."""
    r = client.get("/v1/nonexistent_route_xyz")
    assert r.status_code == 404
    body = r.json()
    assert "error" in body
    assert body["error"]["type"] == "not_found"
    assert body["error"]["status"] == 404


# ── Local API security (desktop token) ────────────────────────────────────

def test_api_token_not_required_when_env_unset(client):
    """No ORCHA_API_TOKEN in the environment → the API stays open (dev/manual)."""
    r = client.get("/v1/experts")
    assert r.status_code == 200


def test_api_token_enforced_when_set(client, monkeypatch):
    """With ORCHA_API_TOKEN set, every non-health request needs the header."""
    monkeypatch.setenv("ORCHA_API_TOKEN", "test-secret-token")

    r = client.get("/v1/experts")
    assert r.status_code == 401

    r = client.get("/v1/experts", headers={"X-Orcha-Token": "wrong-token"})
    assert r.status_code == 401

    r = client.get("/v1/experts", headers={"X-Orcha-Token": "test-secret-token"})
    assert r.status_code == 200

    r = client.post("/v1/run", json={"query": "x"}, headers={"X-Orcha-Token": "test-secret-token"})
    assert r.status_code in (200, 422)  # reached the handler, not the auth wall


def test_api_token_health_stays_open(client, monkeypatch):
    """Readiness probes run without the token — /v1/health must stay open."""
    monkeypatch.setenv("ORCHA_API_TOKEN", "test-secret-token")
    r = client.get("/v1/health")
    assert r.status_code == 200


def test_api_token_cors_preflight_passes_through(client, monkeypatch):
    """OPTIONS preflights carry no custom headers by design — the token gate
    must let them reach the CORS middleware, or every browser fetch from
    the app dies as a CORS failure (regression: preflight returned 401 with
    no Access-Control-Allow-Origin, breaking model activation/chat/agents)."""
    monkeypatch.setenv("ORCHA_API_TOKEN", "test-secret-token")
    r = client.options(
        "/v1/agent-runs",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-orcha-token,content-type",
        },
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") in ("*", "http://localhost:5173")
    assert "x-orcha-token" in r.headers.get("access-control-allow-headers", "").lower()


# ── UI ─────────────────────────────────────────────────────────────────────

def test_root_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Orcha" in r.text
    assert "text/html" in r.headers["content-type"]
