"""Runtime (real daemon): startup, health, version, auth, error shapes, and the no-model state.

File name sorts first on purpose: these tests need a runtime with NO model selected.
"""
import json
import os
import time
import urllib.request

import pytest

from anvira_client import AnviraError, AnviraRuntime, PermissionDenied
from anvira_client.discovery import probe, read_token
from anvira_client.http import Http
from anvira_runtime.cli.main import main as cli_main
from anvira_runtime.version import API_VERSION, RUNTIME_VERSION

pytestmark = pytest.mark.slow


def raw(runtime, method, path, body=None, token="owner", headers=None):
    """HTTP call returning (status, json) without SDK error handling."""
    tok = read_token("owner", runtime.env) if token == "owner" else token
    h = {"Accept": "application/json", **(headers or {})}
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    data = json.dumps(body).encode() if body is not None else None
    if data:
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(runtime.info.base_url + path, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


# ------------------------------------------------------------------- lifecycle
def test_runtime_is_running_and_discoverable(runtime):
    info = probe(runtime.env)
    assert info.running and info.ready and info.installed
    assert info.runtime_version == RUNTIME_VERSION and info.api_version == API_VERSION
    assert info.host == "127.0.0.1" and info.port and info.pid
    disc = json.loads((runtime.home / "state" / "runtime.json").read_text())
    assert disc["pid"] == info.pid and disc["port"] == info.port


def test_health_and_version_are_public_and_minimal(runtime):
    st, h = raw(runtime, "GET", "/health", token=None)
    assert st == 200 and h["status"] == "ok" and h["service"] == "anvira-runtime" and h["degraded"] == []
    st, v = raw(runtime, "GET", "/version", token=None)
    assert v["api_version"] == 1 and v["runtime_version"] == RUNTIME_VERSION and v["min_client_api"] == 1
    for cap in ("chat", "orcha.run", "memory.search", "models.select", "aicl.bus", "chat.stream"):
        assert cap in v["capabilities"]
    assert "orcha" not in json.dumps(v).lower().replace("orcha.run", "").replace("orcha.jobs", "").replace("orcha.cancel", "")  # no ORCHA internals


def test_status_shows_supervised_infrastructure(runtime):
    st, s = raw(runtime, "GET", "/v1/status")
    assert st == 200 and s["status"] == "ok"
    for name in ("orcha", "nomi"):
        svc = s["services"][name]
        assert svc["state"] == "running" and svc["pid"] and svc["port"] and svc["port"] not in (8420, 8000)   # private ports
    assert s["services"]["aicl"]["state"] == "running" and s["services"]["aicl"]["in_process"]
    assert s["model"]["active"] is None and s["jobs"] == {} or isinstance(s["jobs"], dict)
    assert s["paths"]["home"] == str(runtime.home)


def test_infrastructure_is_not_reachable_without_runtime_credentials(runtime):
    _, s = raw(runtime, "GET", "/v1/status")
    orcha_port, nomi_port = s["services"]["orcha"]["port"], s["services"]["nomi"]["port"]
    for port, path in ((orcha_port, "/v1/runs"), (nomi_port, "/api/v1/memory")):      # apps cannot bypass the runtime
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5)
            raise AssertionError("internal service answered without credentials")
        except urllib.error.HTTPError as e:
            assert e.code in (401, 403)


def test_services_listen_on_loopback_only(runtime):
    import socket
    _, s = raw(runtime, "GET", "/v1/status")
    ips = {a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)} - {"127.0.0.1"}
    for port in (runtime.info.port, s["services"]["orcha"]["port"], s["services"]["nomi"]["port"]):
        for ip in ips:
            with socket.socket() as sock:
                sock.settimeout(0.5)
                assert sock.connect_ex((ip, port)) != 0, f"port {port} reachable on non-loopback {ip}"


# ----------------------------------------------------------------------- auth
def test_authentication(runtime):
    assert raw(runtime, "GET", "/v1/status", token=None)[0] == 401
    st, body = raw(runtime, "GET", "/v1/status", token="anv_wrong")
    assert st == 401 and body["error"]["code"] == "unauthorized"
    assert raw(runtime, "GET", "/v1/models", token="Bearer x")[0] == 401


def test_app_registration_needs_the_registration_token(runtime):
    assert raw(runtime, "POST", "/v1/apps/register", {"app_id": "evil"}, token=None)[0] == 401
    assert raw(runtime, "POST", "/v1/apps/register", {"app_id": "evil"}, token=None, headers={"X-Anvira-Register-Token": "guess"})[0] == 401
    st, body = raw(runtime, "POST", "/v1/apps/register", {"app_id": "reg-test", "permissions": ["models.manage"]}, token=None,
                   headers={"X-Anvira-Register-Token": read_token("register", runtime.env)})
    assert st == 200 and body["token"].startswith("anv_") and body["app"]["requested"] == ["models.manage"]
    assert "models.manage" not in body["app"]["permissions"]


def test_permission_denied_is_structured_403(runtime):
    app = runtime.connect("perm-test")
    st, body = raw(runtime, "POST", "/v1/models/remove", {"id": "x"}, token=app._http.token)
    assert st == 403 and body["error"]["code"] == "permission_denied" and "anvira app grant perm-test models.manage" in body["error"]["message"]
    with pytest.raises(PermissionDenied):
        app.models.install("qwen3-4b")
    for path in ("/v1/apps", "/v1/runtime/logs", "/v1/diagnostics", "/v1/aicl/status"):
        assert raw(runtime, "GET", path, token=app._http.token)[0] == 403
    assert raw(runtime, "PUT", "/v1/runtime/config", {"key": "logging.level", "value": "DEBUG"}, token=app._http.token)[0] == 403


def test_local_network_guards(runtime):
    base = runtime.info.base_url
    req = urllib.request.Request(base + "/health", headers={"Origin": "http://evil.example"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 403 and json.loads(e.value.read())["error"]["code"] == "origin_not_allowed"
    req = urllib.request.Request(base + "/health", headers={"Host": "rebind.evil.example"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 421
    req = urllib.request.Request(base + "/health", headers={"Host": f"localhost:{runtime.info.port}"})
    assert urllib.request.urlopen(req, timeout=5).status == 200


def test_error_envelopes_are_json(runtime):
    st, b = raw(runtime, "GET", "/v1/nope")
    assert st == 404 and b["error"]["code"] == "not_found"
    st, b = raw(runtime, "DELETE", "/v1/status")
    assert st == 405 and b["error"]["code"] == "method_not_allowed"
    st, b = raw(runtime, "GET", "/v1/models/search")
    assert st == 400 and b["error"]["code"] == "invalid_request"
    st, b = raw(runtime, "POST", "/v1/models/select", {})
    assert st == 400 and "id" in b["error"]["message"]
    assert "Traceback" not in json.dumps(b)


# ---------------------------------------------------------------------- no model
def test_no_model_installed_state(runtime):
    runtime.owner.json("POST", "/v1/models/deselect")
    st, b = raw(runtime, "POST", "/v1/chat", {"messages": [{"role": "user", "content": "hi"}]})
    assert st == 409 and b["error"]["code"] == "no_model" and "anvira model list" in b["error"]["hint"]
    for path in ("/v1/task", "/v1/orcha/run", "/v1/agent/run"):
        st, b = raw(runtime, "POST", path, {"task": "do something"})
        assert st == 409 and b["error"]["code"] == "no_model", path


def test_model_selection_errors(runtime):
    st, b = raw(runtime, "POST", "/v1/models/select", {"id": "ghost-model"})
    assert st == 404 and b["error"]["code"] == "model_not_found" and "anvira model list" in b["error"]["hint"]
    st, b = raw(runtime, "POST", "/v1/models/select", {"id": "qwen3-8b"})
    assert st == 409 and b["error"]["code"] == "model_not_installed" and "anvira model install qwen3-8b" in b["error"]["hint"]
    assert raw(runtime, "GET", "/v1/models/active")[1]["active"] is None


def test_catalog_and_hardware_are_available_without_a_model(runtime):
    _, cat = raw(runtime, "GET", "/v1/models/catalog")
    ids = {m["id"] for m in cat["models"]}
    assert "qwen3-8b" in ids and all(not m["installed"] for m in cat["models"])
    assert all("compatibility" in m for m in cat["models"])
    _, hw = raw(runtime, "GET", "/v1/hardware")
    assert hw["cpu"]["logical_cores"] >= 1 and hw["ram"]["total_mib"] > 0 and "gpu" in hw and "storage" in hw
    _, rec = raw(runtime, "GET", "/v1/models/recommended")
    assert isinstance(rec["models"], list)
    _, comp = raw(runtime, "GET", "/v1/models/qwen3-8b/compatibility")
    assert comp["id"] == "qwen3-8b" and comp["mode"] in ("gpu", "partial-gpu", "cpu", "insufficient") and "hardware" in comp


def test_cli_reports_no_model(runtime, capsys):
    code = cli_main(["chat", "hello"])
    err = capsys.readouterr().err
    assert code == 6 and "No model is installed." in err and "anvira model list" in err
    code = cli_main(["model", "use", "ghost-model"])
    assert code == 6 and "was not found" in capsys.readouterr().err
    code = cli_main(["model", "list", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["active"] is None
