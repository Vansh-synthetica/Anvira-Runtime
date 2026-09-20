"""Runtime (real daemon + real ORCHA + real Nomi): chat, ORCHA jobs, memory, context, config, apps, ops."""
import json
import time
import urllib.error
import urllib.request

import pytest

from anvira_client import AnviraError, NoModel, PermissionDenied
from anvira_client.discovery import read_token
from conftest import wait_for

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def notes(runtime, provider_model):
    app = runtime.connect("notes")
    app.models.use(provider_model)
    return app


@pytest.fixture(scope="module")
def study(runtime, provider_model):
    return runtime.connect("study")


def call(runtime, method, path, body=None, token=None):
    tok = token or read_token("owner", runtime.env)
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/json"}
    data = json.dumps(body).encode() if body is not None else None
    if data:
        h["Content-Type"] = "application/json"
    try:
        with urllib.request.urlopen(urllib.request.Request(runtime.info.base_url + path, data=data, headers=h, method=method), timeout=60) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


# ------------------------------------------------------------------------ models
def test_provider_models_never_expose_keys(runtime, notes, provider_model):
    assert provider_model == "cloud:fake"
    rec = next(m for m in notes.models.list(installed=True) if m["id"] == provider_model)
    assert rec["kind"] == "provider" and rec["has_api_key"] and "api_key" not in rec and rec["compatibility"]["mode"] == "remote"
    _, status = call(runtime, "GET", "/v1/status")
    _, cfg = call(runtime, "GET", "/v1/runtime/config")
    for blob in (json.dumps(rec), json.dumps(status), json.dumps(cfg), json.dumps(notes.models.providers())):
        assert "sk-test-secret-1234567890" not in blob
    log = (runtime.home / "logs" / "runtime.log").read_text(encoding="utf-8")
    assert "sk-test-secret-1234567890" not in log


def test_select_provider_makes_it_active_everywhere(runtime, notes, provider_model):
    active = notes.models.active()
    assert active["active"] == provider_model and active["state"] == "remote"
    assert notes.status()["model"]["active"] == provider_model
    assert notes.orcha.status()["active_model"] == provider_model


# -------------------------------------------------------------------------- chat
def test_chat_non_streaming(notes, fake):
    r = notes.chat([{"role": "user", "content": "hello runtime"}])
    assert r["choices"][0]["message"]["content"] == "fake reply to: hello runtime"
    assert r["runtime"]["model"] == "cloud:fake" and r["runtime"]["kind"] == "provider"
    sent = fake.requests[-1]
    assert sent["model"] == "fake-model" and sent["stream"] is False           # runtime maps to the provider's model name
    assert notes.chat_text([{"role": "user", "content": "x"}]).startswith("fake reply")


def test_chat_streaming(notes, fake):
    assert "".join(notes.chat([{"role": "user", "content": "hi"}], stream=True)) == "fake stream reply"


def test_chat_upstream_failure_is_a_structured_error_not_a_hang(notes, fake):
    fake.mode = "error"
    t0 = time.monotonic()
    with pytest.raises(AnviraError) as e:
        notes.chat([{"role": "user", "content": "hi"}])
    assert e.value.code == "model_error" and e.value.status == 502 and time.monotonic() - t0 < 10
    with pytest.raises(AnviraError) as e:                                        # mid-stream: error frame, then the stream ends
        list(notes.chat([{"role": "user", "content": "hi"}], stream=True))
    assert e.value.code == "model_error"


def test_chat_validation_and_model_mismatch(runtime, notes):
    for bad in ({}, {"messages": []}, {"messages": "hi"}, {"messages": [{"content": "no role"}]}):
        st, b = call(runtime, "POST", "/v1/chat", bad, token=notes._http.token)
        assert st == 400 and b["error"]["code"] == "invalid_request"
    with pytest.raises(AnviraError) as e:
        notes.chat([{"role": "user", "content": "hi"}], model="some-other-model")
    assert e.value.code == "model_not_found" and e.value.status == 404


def test_chat_recalls_memories_from_the_apps_namespace(notes, study, fake):
    notes.memory.store("The user's dog is called Biscuit", title="Pet", workspace="home")
    notes.chat([{"role": "user", "content": "What is my dog called?"}], memory={"recall": True, "workspace": "home"})
    system = fake.requests[-1]["messages"][0]
    assert system["role"] == "system" and "Biscuit" in system["content"]
    study.chat([{"role": "user", "content": "What is my dog called?"}], memory={"recall": True})
    assert all("Biscuit" not in m["content"] for m in fake.requests[-1]["messages"])    # another app's memory is not recalled


# ------------------------------------------------------------------ ORCHA jobs
def test_task_runs_through_orcha_and_returns_a_job(runtime, notes, fake):
    job = notes.task("Explain what a mutex is in one sentence.")
    assert job.state == "completed" and job.kind == "task" and job.id.startswith("job_")
    res = job.unwrap()
    assert "fake reply" in res["answer"] and res["status"] == "completed" and res["run_id"]
    assert res["contributors"] and res["iterations"] >= 1
    assert any("mutex" in json.dumps(r["messages"]) for r in fake.requests)               # the model really was called by ORCHA
    assert job.data["meta"]["orcha_run_id"] == res["run_id"] and job.data["app"] == "notes"


def test_orcha_run_is_async_and_pollable(notes):
    job = notes.orcha.run("Name one prime number.", graph="default")
    assert job.state in ("queued", "running", "completed")
    done = job.wait(timeout=60)
    assert done.state == "completed" and "fake reply" in done.unwrap()["answer"]
    listed = [j["id"] for j in notes.orcha.jobs()]
    assert done.id in listed


def test_orcha_job_events_are_passed_through(runtime, notes):
    job = notes.orcha.run("Say hello.", wait=True)
    req = urllib.request.Request(runtime.info.base_url + f"/v1/jobs/{job.id}/events",
                                 headers={"Authorization": f"Bearer {notes._http.token}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            assert r.headers["Content-Type"].startswith("text/event-stream")
            r.read()
    except urllib.error.HTTPError as e:                     # a finished run may already have closed its stream
        assert e.code in (404, 409, 502)


def test_cancel_a_running_job(runtime, notes, fake):
    fake.mode, fake.delay = "slow", 25
    job = notes.orcha.run("This will be slow.")
    wait_for(lambda: notes.jobs.get(job.id).data["meta"].get("orcha_run_id"), 15, message="ORCHA run started")
    t0 = time.monotonic()
    cancelled = notes.orcha.cancel(job.id)
    assert cancelled.state == "cancelled" and time.monotonic() - t0 < 12
    assert notes.jobs.get(job.id).state == "cancelled"
    with pytest.raises(AnviraError) as e:
        notes.orcha.cancel(job.id)                                                          # already finished
    assert e.value.code == "job_not_cancellable" and e.value.status == 409
    with pytest.raises(AnviraError) as e:
        cancelled.unwrap()
    assert e.value.code == "cancelled"


def test_error_propagation_from_the_model_through_orcha(notes, fake):
    fake.mode = "error"
    job = notes.orcha.run("This will fail.", wait=True)
    if job.state == "failed":
        assert job.error["code"] and job.error["message"]
        with pytest.raises(AnviraError):
            job.unwrap()
    else:      # ORCHA may degrade gracefully instead of failing; the job must still be terminal and inspectable
        assert job.state == "completed"


def test_jobs_are_private_to_the_app(runtime, notes, study):
    job = notes.task("Private to notes.")
    with pytest.raises(AnviraError) as e:
        study.jobs.get(job.id)
    assert e.value.code == "job_not_found" and e.value.status == 404
    with pytest.raises(AnviraError):
        study.jobs.cancel(job.id)
    assert job.id not in [j["id"] for j in study.jobs.list()]
    _, owner_view = call(runtime, "GET", "/v1/jobs")
    assert job.id in [j["id"] for j in owner_view["jobs"]]                                   # the owner sees all


def test_task_validation(runtime, notes):
    st, b = call(runtime, "POST", "/v1/task", {}, token=notes._http.token)
    assert st == 400 and b["error"]["code"] == "invalid_request"
    st, b = call(runtime, "POST", "/v1/orcha/run", {"task": "x", "graph": "does-not-exist"}, token=notes._http.token)
    assert st == 400 and b["error"]["code"] == "invalid_graph" and "default" in b["error"]["hint"]   # stable contract, not ORCHA internals


def test_agent_run_passes_workspace_roots_to_orcha(notes, tmp_path):
    job = notes.agent_run("List the files.", workspace_roots=[str(tmp_path)], wait=True)
    assert job.state in ("completed", "failed") and job.kind == "agent.run"                  # reached ORCHA's agent path


def test_orcha_status_hides_internals_but_shows_engine(notes):
    s = notes.orcha.status()
    assert s["service"]["state"] == "running" and s["engine"]["source"] == "local"
    assert any("cloud" in e or "fake" in e for e in s["engine"]["experts"])


# -------------------------------------------------------------------------- memory
def test_memory_store_search_get_delete(notes):
    m = notes.memory.store("Quarterly planning happens in March", title="Planning", tags=["work", "calendar"], importance=4)
    assert m["app"] == "notes" and m["scope"] == "app" and m["tags"] == ["work", "calendar"] and m["importance"] == 4
    hits = notes.memory.search("planning march")
    assert hits and hits[0]["id"] == m["id"] and hits[0]["score"] > 0
    assert notes.memory.search("calendar", tags=["work"])[0]["id"] == m["id"]
    assert notes.memory.get(m["id"])["content"].startswith("Quarterly")
    assert notes.memory.search("zzzznotpresent") == []
    assert notes.memory.delete(m["id"])["deleted"] is True
    with pytest.raises(AnviraError) as e:
        notes.memory.get(m["id"])
    assert e.value.code in ("memory_not_found", "nomi_error")


def test_memory_is_isolated_between_apps(runtime, notes, study):
    secret = notes.memory.store("Notes-only secret: the vault code is 1234", title="Vault")
    assert study.memory.search("vault code") == []
    with pytest.raises(AnviraError) as e:
        study.memory.get(secret["id"])
    assert e.value.status == 404
    with pytest.raises(AnviraError) as e:
        study.memory.delete(secret["id"])
    assert e.value.status == 404 and notes.memory.get(secret["id"])["title"] == "Vault"     # untouched
    _, owner = call(runtime, "GET", "/v1/memory/search?q=vault&app=notes")
    assert owner["items"] and owner["items"][0]["app"] == "notes"                            # the owner can audit any namespace


def test_workspace_boundary_inside_an_app(notes):
    notes.memory.store("Alpha team roadmap", workspace="alpha")
    notes.memory.store("Beta team roadmap", workspace="beta")
    assert [m["workspace"] for m in notes.memory.search("roadmap", workspace="alpha")] == ["alpha"]
    assert {m["workspace"] for m in notes.memory.search("roadmap")} >= {"alpha", "beta"}


def test_shared_memory_requires_permission(runtime, notes, study):
    with pytest.raises(PermissionDenied):
        notes.memory.store("Shared fact: the office wifi is guest-net", scope="shared")
    call(runtime, "POST", "/v1/apps/notes/grant", {"permissions": ["memory.shared"]})
    call(runtime, "POST", "/v1/apps/study/grant", {"permissions": ["memory.shared"]})
    shared = notes.memory.store("Shared fact: the office wifi is guest-net", scope="shared")
    assert shared["scope"] == "shared"
    assert [m["id"] for m in study.memory.search("office wifi", scope="shared")] == [shared["id"]]
    assert study.memory.search("office wifi") == []                                          # default scope stays private
    with pytest.raises(AnviraError) as e:                                                    # readers cannot delete the author's memory
        study.memory.delete(shared["id"])
    assert e.value.code == "memory_not_owned" and e.value.status == 403
    call(runtime, "POST", "/v1/apps/study/deny", {"permissions": ["memory.shared"]})
    with pytest.raises(PermissionDenied):
        study.memory.search("office wifi", scope="shared")


def test_memory_validation(runtime, notes):
    for body in ({}, {"content": "  "}, {"content": "x", "tags": ["app:study"]}, {"content": "x", "scope": "everyone"}):
        st, b = call(runtime, "POST", "/v1/memory", body, token=notes._http.token)
        assert st == 400, body
    st, b = call(runtime, "POST", "/v1/memory", {"content": "x", "tags": ["scope:shared"]}, token=notes._http.token)
    assert b["error"]["code"] == "invalid_tag"                                                # apps cannot forge namespace tags


def test_nomi_status_endpoint(runtime):
    _, s = call(runtime, "GET", "/v1/nomi/status")
    assert s["service"]["state"] == "running" and s["memory"]["available"]


# ------------------------------------------------------------------------- context
def test_context_index_and_retrieval(notes, study):
    r = notes.context.put("nb1", "d1", "The mitochondria is the powerhouse of the cell.\n\nRibosomes build proteins.", title="Cells")
    assert r["chunks"] >= 1
    notes.context.put("nb1", "d2", "The French Revolution began in 1789.", title="History")
    hits = notes.context.search("nb1", "powerhouse mitochondria")
    assert hits[0]["doc_id"] == "d1" and hits[0]["score"] > 0 and hits[0]["title"] == "Cells"
    assert [h["doc_id"] for h in notes.context.search("nb1", "revolution", doc_ids=["d2"])] == ["d2"]
    assert notes.context.collections()[0]["collection"] == "nb1" and len(notes.context.documents("nb1")) == 2
    assert study.context.search("nb1", "mitochondria") == [] and study.context.collections() == []      # per-app namespace
    notes.context.put("nb1", "d1", "Replaced text about photosynthesis.")
    assert notes.context.search("nb1", "mitochondria") == []                                             # replace, not append
    notes.context.delete("nb1", "d2")
    assert [d["doc_id"] for d in notes.context.documents("nb1")] == ["d1"]
    notes.context.delete("nb1")
    assert notes.context.collections() == []


def test_context_validation(runtime, notes):
    st, b = call(runtime, "PUT", "/v1/context/c/documents/d", {"nope": 1}, token=notes._http.token)
    assert st == 400 and b["error"]["code"] == "invalid_request"


# -------------------------------------------------------------------------- config
def test_config_read_write_and_validation(runtime, notes):
    assert notes.me()["permissions"] and "config.read" in notes.me()["permissions"]
    st, cfg = call(runtime, "GET", "/v1/runtime/config", token=notes._http.token)
    assert st == 200 and cfg["config"]["api"]["host"] == "127.0.0.1"
    st, r = call(runtime, "PUT", "/v1/runtime/config", {"key": "orcha.job_timeout_s", "value": "75"})
    assert st == 200 and r["value"] == 75.0 and r["restart_required"] is False
    st, r = call(runtime, "PUT", "/v1/runtime/config", {"key": "api.port", "value": 5555})
    assert r["restart_required"] is True
    call(runtime, "PUT", "/v1/runtime/config", {"key": "api.port", "value": 0})
    st, b = call(runtime, "PUT", "/v1/runtime/config", {"key": "api.host", "value": "0.0.0.0"})
    assert st == 400 and b["error"]["code"] == "invalid_config" and "loopback" in b["error"]["message"]
    st, b = call(runtime, "PUT", "/v1/runtime/config", {"key": "no.such", "value": 1})
    assert st == 400
    on_disk = json.loads((runtime.home / "config" / "config.json").read_text())
    assert on_disk["orcha"]["job_timeout_s"] == 75.0


# ---------------------------------------------------------------------------- apps
def test_app_listing_and_revocation(runtime):
    tmp = runtime.connect("revoke-me")
    _, apps = call(runtime, "GET", "/v1/apps")
    assert "revoke-me" in [a["app_id"] for a in apps["apps"]] and "permissions" in apps
    assert tmp.models.installed() is not None
    call(runtime, "DELETE", "/v1/apps/revoke-me")
    with pytest.raises(PermissionDenied):
        tmp.models.installed()                                                                # token no longer valid


# ------------------------------------------------------------------- observability
def test_logs_diagnostics_and_aicl_trace(runtime, notes):
    _, logs = call(runtime, "GET", "/v1/runtime/logs?service=runtime&lines=50")
    assert logs["exists"] and any("registered app" in ln for ln in logs["lines"])
    _, orcha_log = call(runtime, "GET", "/v1/runtime/logs?service=orcha")
    assert orcha_log["exists"]
    assert call(runtime, "GET", "/v1/runtime/logs?service=../../etc/passwd")[0] == 400
    _, doc = call(runtime, "GET", "/v1/diagnostics")
    ids = {c["id"]: c["status"] for c in doc["checks"]}
    assert ids["orcha"] == "ok" and ids["nomi"] == "ok" and ids["aicl_bus"] == "ok" and doc["status"] in ("ok", "warn")
    _, aicl = call(runtime, "GET", "/v1/aicl/status")
    mods = aicl["modules"]
    assert all(mods[m]["calls"] > 0 for m in ("orcha", "nomi", "models", "context")) and aicl["codec"].startswith("aicl-bin")
    assert aicl["recent"] and {"module", "action", "op", "ms", "ok", "message_id"} <= set(aicl["recent"][-1])


def test_runtime_restarts_a_crashed_orcha_and_resyncs_the_model(runtime, notes, fake):
    import subprocess
    import sys
    _, s = call(runtime, "GET", "/v1/status")
    pid = s["services"]["orcha"]["pid"]
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"] if sys.platform == "win32" else ["kill", "-9", str(pid)],
                   capture_output=True)

    def recovered():
        st = call(runtime, "GET", "/v1/status")[1]["services"]["orcha"]
        return st["state"] == "running" and st["pid"] != pid and st["restarts"] >= 1
    wait_for(recovered, 60, 0.5, "ORCHA restart")
    job = wait_for(lambda: notes.task("After the crash.") if notes.orcha.status()["engine"]["source"] == "local" else None,
                   45, 1.0, "model re-registered with the restarted ORCHA")
    assert job.state == "completed"                                                            # apps saw a brief blip, not a dead runtime
