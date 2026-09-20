"""Runtime (real daemon): local model lifecycle with a fake Hugging Face and a fake llama-server."""
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from anvira_client import AnviraError, PermissionDenied
from anvira_client.discovery import read_token
from conftest import wait_for
from fakes import FAKE_GGUF

pytestmark = pytest.mark.slow


def call(rt, method, path, body=None, token=None):
    tok = token or read_token("owner", rt.env)
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/json"}
    data = json.dumps(body).encode() if body is not None else None
    if data:
        h["Content-Type"] = "application/json"
    try:
        with urllib.request.urlopen(urllib.request.Request(rt.info.base_url + path, data=data, headers=h, method=method), timeout=120) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def install(rt, model, **kw):
    st, b = call(rt, "POST", "/v1/models/install", {"model": model, "wait": True, **kw})
    assert st == 202, b
    return b["job"]


@pytest.fixture(scope="module")
def app(runtime):
    return runtime.connect("models-app")


def kill(pid):
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"] if sys.platform == "win32" else ["kill", "-9", str(pid)], capture_output=True)


# ---------------------------------------------------------------------- discovery
def test_remote_search_uses_the_configured_hub(app):
    found = app.models.search("tiny")
    assert {m["id"] for m in found} >= {"acme/tiny-GGUF"} and found[0]["installed"] is False


# ---------------------------------------------------------------------- install
def test_install_requires_permission_and_never_downloads_silently(runtime, app):
    with pytest.raises(PermissionDenied):
        app.models.install("acme/tiny-GGUF")
    assert not list((runtime.home / "models").glob("*.gguf"))                 # nothing was downloaded
    assert not any(m["id"].startswith("tiny") for m in call(runtime, "GET", "/v1/models/installed")[1]["models"])


def test_install_from_hub_registers_and_reports_progress(runtime):
    job = install(runtime, "acme/tiny-GGUF")
    assert job["state"] == "completed" and job["kind"] == "model.install"
    m = job["result"]["model"]
    assert m["id"] == "tiny-q4_k_m" and m["complete"] and m["location"] == "primary"
    assert Path(m["path"]).read_bytes() == FAKE_GGUF and Path(m["path"]).parent == runtime.home / "models"
    assert job["progress"]["downloaded"] == len(FAKE_GGUF) and job["progress"]["percent"] == 100.0
    assert not list((runtime.home / "models").glob("*.part"))
    _, inst = call(runtime, "GET", "/v1/models/installed")
    rec = next(x for x in inst["models"] if x["id"] == "tiny-q4_k_m")
    assert rec["source"]["repo"] == "acme/tiny-GGUF" and rec["compatibility"]["can_run"] is True


def test_install_checks_disk_space_before_downloading(runtime):
    job = install(runtime, "acme/huge-GGUF")
    assert job["state"] == "failed" and job["error"]["code"] == "insufficient_storage"
    assert "anvira model install" in job["error"]["hint"]
    assert not list((runtime.home / "models").glob("huge*"))


def test_install_unknown_and_invalid_targets(runtime):
    st, b = call(runtime, "POST", "/v1/models/install", {"model": "not-in-catalog"})
    assert st == 404 and b["error"]["code"] == "model_not_found"
    st, b = call(runtime, "POST", "/v1/models/install", {})
    assert st == 400
    st, b = call(runtime, "POST", "/v1/models/install", {"url": "http://x/evil.txt"})
    assert st in (400, 500) and "gguf" in json.dumps(b).lower()


def test_install_to_a_directory_of_the_users_choice(runtime, tmp_path):
    elsewhere = tmp_path / "Another Drive" / "LLM Files"
    job = install(runtime, "acme/tiny-GGUF", dir=str(elsewhere), url=None)
    assert job["state"] == "completed" and Path(job["result"]["installed_to"]) == elsewhere
    assert (elsewhere / "tiny-Q4_K_M.gguf").exists()
    (elsewhere / "tiny-Q4_K_M.gguf").unlink()
    call(runtime, "POST", "/v1/models/remove", {"id": "tiny-q4_k_m", "delete_file": False})
    install(runtime, "acme/tiny-GGUF")                                       # restore the primary copy for later tests


# ------------------------------------------------------------ select / run / switch
def test_select_starts_the_backend_and_serves_chat(runtime, app, fake):
    st = app.models.use("tiny-q4_k_m")
    assert st["active"] == "tiny-q4_k_m" and st["state"] == "running"
    backend = st["backend"]
    assert backend["kind"] == "model" and backend["port"] and backend["pid"]
    info = st["backend_info"]
    assert info["gpu"] is False and info["binary"].endswith(("llama-server.cmd", "llama-server"))
    assert app.chat_text([{"role": "user", "content": "hello local"}]) == "local reply to: hello local"
    assert "fake reply" not in app.chat_text([{"role": "user", "content": "x"}])          # really the local backend, not the provider


def test_orcha_uses_the_active_local_model(app):
    job = app.task("Say something.")
    assert job.state == "completed" and "local reply" in job.unwrap()["answer"]
    eng = app.orcha.status()["engine"]
    assert eng["source"] == "local" and any("tiny" in e for e in eng["experts"])


def test_switching_models_stops_the_previous_backend(runtime, app, provider_model):
    before = call(runtime, "GET", "/v1/models/active")[1]["backend"]["pid"]
    app.models.use(provider_model)
    st = app.status()
    assert st["model"]["active"] == provider_model and st["model"]["state"] == "remote"
    wait_for(lambda: not _alive(before), 10, message="local backend stopped after switching to a provider")
    assert "fake reply" in app.chat_text([{"role": "user", "content": "x"}])
    app.models.use("tiny-q4_k_m")


def _alive(pid):
    from anvira_runtime.process.supervisor import pid_alive
    return pid_alive(pid)


def test_a_model_crash_returns_a_structured_error_and_recovers(runtime, app):
    st = call(runtime, "GET", "/v1/models/active")[1]
    pid = st["backend"]["pid"]
    kill(pid)
    seen, t0 = set(), time.monotonic()
    ok_again = False
    while time.monotonic() - t0 < 40:
        try:
            r = app.chat_text([{"role": "user", "content": "still there?"}])
            if r.startswith("local reply"):
                ok_again = True
                break
        except AnviraError as e:
            seen.add(e.code)
            assert e.code in ("model_crashed", "model_not_running", "model_loading", "model_start_failed", "model_error")
            assert e.status in (502, 503)
        time.sleep(0.4)
    assert ok_again, f"never recovered; errors seen: {seen}"
    wait_for(lambda: call(runtime, "GET", "/v1/models/active")[1]["backend"]["state"] == "running", 20, message="backend running")
    new = call(runtime, "GET", "/v1/models/active")[1]["backend"]
    assert new["pid"] != pid and new["state"] == "running"           # a fresh backend process took over
    assert time.monotonic() - t0 < 40                                                       # bounded, never hangs


def test_status_and_doctor_see_the_model(runtime):
    st = call(runtime, "GET", "/v1/status")[1]
    assert st["model"]["active"] == "tiny-q4_k_m" and st["model"]["state"] == "running"
    doc = call(runtime, "GET", "/v1/diagnostics")[1]
    models = next(c for c in doc["checks"] if c["id"] == "models")
    assert models["status"] == "ok" and "tiny-q4_k_m" in models["message"]


def test_compatibility_uses_detected_hardware(runtime):
    c = call(runtime, "GET", "/v1/models/tiny-q4_k_m/compatibility")[1]
    assert c["can_run"] is True and c["mode"] in ("cpu", "gpu", "partial-gpu") and c["hardware"]["ram"]["total_mib"] > 0


# ------------------------------------------------- locations: any folder, linking, moving
def test_extra_folders_are_scanned_and_protect_their_files(runtime, tmp_path):
    other = tmp_path / "LM Studio models"
    other.mkdir()
    f = other / "external-Q8_0.gguf"
    f.write_bytes(FAKE_GGUF)
    st, b = call(runtime, "POST", "/v1/models/storage/dirs", {"path": str(other)})
    assert st == 200 and b["models_found"] == 1
    m = next(x for x in call(runtime, "GET", "/v1/models/installed")[1]["models"] if x["id"] == "external-q8_0")
    assert m["location"] == "extra" and m["directory"] == str(other)
    st, b = call(runtime, "POST", "/v1/models/remove", {"id": "external-q8_0", "delete_file": True})
    assert st == 409 and b["error"]["code"] == "confirmation_required" and f.exists()            # never deletes outside its own folder unasked
    st, b = call(runtime, "POST", "/v1/models/remove", {"id": "external-q8_0", "delete_file": True, "confirm": True})
    assert st == 200 and not f.exists()
    call(runtime, "DELETE", f"/v1/models/storage/dirs?path={urllib.request.quote(str(other))}")


def test_link_a_model_in_place_from_anywhere(runtime, tmp_path):
    far = tmp_path / "somewhere" / "deep"
    far.mkdir(parents=True)
    f = far / "linked-model-Q4_K_M.gguf"
    f.write_bytes(FAKE_GGUF)
    st, m = call(runtime, "POST", "/v1/models/link", {"path": str(f)})
    assert st == 200 and m["location"] == "linked" and Path(m["path"]) == f.resolve()
    assert not (runtime.home / "models" / f.name).exists()                                        # not copied
    st, b = call(runtime, "POST", "/v1/models/remove", {"id": m["id"]})
    assert st == 200 and b["deleted_files"] == [] and f.exists()                                  # unlink only, file untouched
    assert call(runtime, "POST", "/v1/models/link", {"path": str(tmp_path / "missing.gguf")})[0] == 400


def test_move_the_models_folder_and_keep_working(runtime, app, tmp_path):
    target = tmp_path / "New Models Home"
    call(runtime, "POST", "/v1/models/deselect")
    st, res = call(runtime, "PUT", "/v1/models/storage", {"path": str(target), "move": True})
    assert st == 200 and res["failed"] == [] and any(m["target"].endswith("tiny-Q4_K_M.gguf") for m in res["moved"])
    assert (target / "tiny-Q4_K_M.gguf").exists() and not (runtime.home / "models" / "tiny-Q4_K_M.gguf").exists()
    m = call(runtime, "GET", "/v1/models/tiny-q4_k_m")[1]
    assert Path(m["path"]).parent == target                                                       # registry followed the file
    assert json.loads((runtime.home / "config" / "config.json").read_text())["models"]["models_dir"] == str(target)
    assert app.models.use("tiny-q4_k_m")["state"] == "running"
    (tmp_path / "blocker").write_text("file, not folder")
    st, b = call(runtime, "PUT", "/v1/models/storage", {"path": str(tmp_path / "blocker" / "sub")})
    assert st == 400 and b["error"]["code"] == "storage_unwritable"


def test_remove_the_active_model_stops_it(runtime, app):
    pid = call(runtime, "GET", "/v1/models/active")[1]["backend"]["pid"]
    st, b = call(runtime, "POST", "/v1/models/remove", {"id": "tiny-q4_k_m"})
    assert st == 200 and len(b["deleted_files"]) == 1
    wait_for(lambda: not _alive(pid), 10, message="backend stopped")
    assert call(runtime, "GET", "/v1/models/active")[1]["active"] is None
    st, b = call(runtime, "POST", "/v1/chat", {"messages": [{"role": "user", "content": "x"}]}, token=app._http.token)
    assert st == 409 and b["error"]["code"] == "no_model"
    assert call(runtime, "POST", "/v1/models/remove", {"id": "tiny-q4_k_m"})[0] == 404
