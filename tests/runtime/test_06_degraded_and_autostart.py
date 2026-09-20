"""Failure behaviour: a service that cannot start, a missing inference backend, and an app auto-starting a stopped runtime."""
import json
import socket

import pytest

from anvira_client import AnviraError, AnviraRuntime, bootstrap
from anvira_client.discovery import probe
from conftest import start_isolated, stop_isolated, wait_for
from fakes import FAKE_GGUF

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def degraded(tmp_path_factory, fake_llm):
    """Nomi's port is taken by a foreign program; llama-server does not exist anywhere."""
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(5)
    tmp = tmp_path_factory.mktemp("deg")
    h = start_isolated(tmp, fake_llm, tmp / "no-such" / "llama-server",
                       extra_config={"services": {"nomi": {"port": blocker.getsockname()[1]}}})
    h.blocker = blocker
    yield h
    stop_isolated(h)
    blocker.close()


def test_runtime_stays_up_and_reports_a_degraded_service(degraded):
    info = probe(degraded.env)
    assert info.running and info.ready
    st = degraded.owner.json("GET", "/v1/status")
    assert st["status"] == "degraded" and st["degraded"] == ["nomi"]
    assert st["services"]["nomi"]["state"] == "failed" and "in use by another program" in st["services"]["nomi"]["last_error"]
    assert st["services"]["orcha"]["state"] == "running"                                   # the rest of the runtime is unaffected
    assert degraded.owner.json("GET", "/health")["status"] == "degraded"


def test_memory_fails_gracefully_when_nomi_is_down(degraded):
    app = degraded.connect("deg-app")
    with pytest.raises(AnviraError) as e:
        app.memory.store("x")
    assert e.value.code == "nomi_unavailable" and e.value.status == 503 and "anvira doctor" in e.value.hint
    with pytest.raises(AnviraError) as e:
        app.memory.search("x")
    assert e.value.code == "nomi_unavailable"
    app.context.put("c", "d", "still works without memory")                                # non-Nomi features keep working
    assert app.context.search("c", "works")


def test_chat_survives_memory_recall_failure_with_a_warning(degraded, provider_model=None):
    owner = degraded.owner
    rec = owner.json("POST", "/v1/providers", {"label": "Fake", "base_url": degraded.fake.base + "/v1", "model": "fake-model"})
    owner.json("POST", "/v1/models/select", {"id": rec["id"]})
    app = degraded.connect("deg-app")
    r = app.chat([{"role": "user", "content": "hi"}], memory={"recall": True})
    assert r["choices"][0]["message"]["content"].startswith("fake reply")
    assert any("memory recall skipped" in w for w in r["runtime"]["warnings"])


def test_doctor_pinpoints_the_problem(degraded):
    doc = degraded.owner.json("GET", "/v1/diagnostics")
    by = {c["id"]: c for c in doc["checks"]}
    assert doc["status"] == "fail" and by["nomi"]["status"] == "fail" and "anvira runtime logs --service nomi" in by["nomi"]["fix"]
    assert by["orcha"]["status"] == "ok"
    log = degraded.owner.json("GET", "/v1/runtime/logs", params={"service": "runtime"})["lines"]
    assert any("nomi" in ln and "in use" in ln for ln in log)


def test_missing_inference_backend_is_a_structured_error(degraded):
    (degraded.home / "models").mkdir(exist_ok=True)
    (degraded.home / "models" / "local-Q4_K_M.gguf").write_bytes(FAKE_GGUF)
    app = degraded.connect("deg-app")
    try:
        app.models.use("local-q4_k_m")
        pytest.skip("this machine has a real llama-server on a searched path")
    except AnviraError as e:
        assert e.code == "backend_missing" and e.status == 424
        assert "llama-server" in e.message and "models.llama_server_path" in e.hint
    assert app.models.active()["active"] == degraded.owner.json("GET", "/v1/models/active")["active"]   # failed switch changed nothing


def test_stopped_runtime_is_started_by_the_first_app_that_needs_it(tmp_path_factory, fake_llm, fake_llama):
    h = start_isolated(tmp_path_factory.mktemp("auto"), fake_llm, fake_llama)
    try:
        assert bootstrap.stop_runtime(h.env)
        assert not probe(h.env).running and not (h.home / "state" / "runtime.json").exists()   # clean shutdown removed the discovery file
        seen = []
        app = AnviraRuntime.connect("anvira-notes", env=h.env, on_status=seen.append)
        assert "Starting runtime..." in seen and app.health()["status"] == "ok"
        app2 = AnviraRuntime.connect("anvira-study", env=h.env, on_status=seen.append)      # already running: no second start
        assert app2.info.pid == app.info.pid and seen.count("Starting runtime...") == 1
    finally:
        stop_isolated(h)


def test_stale_discovery_file_after_a_crash_is_recovered(tmp_path_factory, fake_llm, fake_llama):
    import subprocess, sys
    h = start_isolated(tmp_path_factory.mktemp("crash"), fake_llm, fake_llama)
    try:
        pid = probe(h.env).pid
        children = json.loads((h.home / "state" / "children.json").read_text())
        subprocess.run(["taskkill", "/PID", str(pid), "/F"] if sys.platform == "win32" else ["kill", "-9", str(pid)], capture_output=True)
        wait_for(lambda: probe(h.env).stale_discovery, 15, message="stale discovery file noticed")
        assert not probe(h.env).running
        app = AnviraRuntime.connect("anvira-notes", env=h.env)                              # recovers by starting a fresh runtime
        assert app.info.pid != pid and app.health()["status"] == "ok"
        # orphaned ORCHA/Nomi children of the dead runtime were reaped, not leaked
        from anvira_runtime.process.supervisor import pid_alive
        wait_for(lambda: not any(pid_alive(c["pid"]) for c in children.values() if c.get("pid")), 20, message="orphans reaped")
    finally:
        stop_isolated(h)
