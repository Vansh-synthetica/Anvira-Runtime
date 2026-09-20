"""On-demand lifecycle: the runtime is active only while an application is open."""
import json
import subprocess
import sys
import time

import pytest

from anvira_client import AnviraError, AnviraRuntime, bootstrap
from anvira_client.discovery import probe, read_token
from anvira_client.http import Http
from anvira_runtime.cli.main import main as cli
from anvira_runtime.core.lifecycle import Lifecycle
from anvira_runtime.process.supervisor import pid_alive
from conftest import start_isolated, stop_isolated, wait_for


# ----------------------------------------------------------------- unit (fake clock)
def make(auto_stop=True, busy=0, grace=10.0, ttl=20.0):
    state = {"busy": busy}
    lc = Lifecycle(auto_stop=auto_stop, idle_grace_s=grace, lease_ttl_s=ttl, busy=lambda: state["busy"])
    return lc, state


def test_idle_countdown_starts_only_when_nothing_holds_the_runtime():
    lc, _ = make()
    t0 = lc._idle_since
    assert lc.tick(t0 + 5) is False and lc.tick(t0 + 10.5) is True                  # nobody ever opened it: stops after the grace
    lc, _ = make()
    lease = lc.acquire("notes")
    beat = lc.leases[lease["id"]]["last"]
    assert lc.tick(beat + 5) is False and lc.status()["shutdown_in_s"] is None      # an open app: no countdown at all
    lc.release(lease["id"])
    assert lc.tick(beat + 6) is False                                               # countdown just started
    assert lc.tick(beat + 6 + 10.1) is True


def test_leases_expire_without_release_and_work_blocks_shutdown():
    lc, st = make(ttl=20.0)
    lc.acquire("crashy")
    last = next(iter(lc.leases.values()))["last"]
    assert lc.tick(last + 19) is False                                              # still within ttl
    assert lc.tick(last + 21) is False and not lc.leases                            # expired -> idle clock starts now
    assert lc.tick(last + 21 + 9) is False and lc.tick(last + 21 + 10.5) is True
    lc, st = make(busy=1)
    assert lc.tick(lc.started + 1000) is False and lc.status()["busy"] == 1         # a running job/stream/download blocks it
    st["busy"] = 0
    assert lc.tick(lc.started + 1000) is False                                      # countdown restarts when work ends
    assert lc.tick(lc.started + 1011) is True


def test_persistent_mode_never_shuts_down_but_still_tracks_leases():
    lc, _ = make(auto_stop=False)
    assert lc.tick(lc.started + 10**6) is False                                     # idle for ages, still never stops
    lc.acquire("notes")
    assert lc.status()["mode"] == "persistent" and lc.status()["leases"][0]["app"] == "notes"


def test_lease_ownership_and_ttl_clamping():
    lc, _ = make()
    a = lc.acquire("a", ttl_s=1)
    assert a["ttl_s"] == 5.0 and lc.acquire("b", ttl_s=10**6)["ttl_s"] == 600.0
    assert lc.heartbeat(a["id"], "b") is None and lc.release(a["id"], "b") is False  # cannot touch another app's lease
    assert lc.heartbeat(a["id"], "a")["app"] == "a" and lc.release(a["id"], "a") is True and lc.release(a["id"]) is False


# ------------------------------------------------------------------------ real runtime
def config(ttl=6.0):
    return {"lifecycle": {"lease_ttl_s": ttl}}


@pytest.fixture()
def home(tmp_path, fake_llm, fake_llama):
    """An isolated home with a short lease TTL and NO runtime running (an app has not been opened yet)."""
    h = start_isolated(tmp_path, fake_llm, fake_llama, extra_config=config(), start=False)
    (h.home / "install.json").write_text(json.dumps({"version": "1.0.0", "entry": sys.executable}))   # runtime installed, not running
    yield h
    stop_isolated(h)


def wait_stopped(h, timeout=45):
    wait_for(lambda: not probe(h.env).running, timeout, 0.5, "runtime to stop by itself")


def pids_of(h):
    st = Http(probe(h.env).base_url, read_token("owner", h.env), 30).json("GET", "/v1/status")
    return st["pid"], [st["services"][n]["pid"] for n in ("orcha", "nomi")]


@pytest.mark.slow
def test_runtime_starts_when_an_app_opens_and_stops_after_it_closes(home):
    assert not probe(home.env).running                                              # nothing running at login
    seen = []
    app = AnviraRuntime.connect("anvira-notes", env=home.env, idle_grace_s=4, on_status=seen.append)
    assert "Starting runtime..." in seen
    lc = app.lifecycle()
    assert lc["mode"] == "on-demand" and lc["auto_stop"] and [x["app"] for x in lc["leases"]] == ["anvira-notes"]
    daemon, children = pids_of(home)
    time.sleep(11)                                                                   # far longer than grace 4 s + ttl 6 s
    assert probe(home.env).running and app.health()["status"] == "ok"               # heartbeats keep it up while the app is open
    assert app.lifecycle()["shutdown_in_s"] is None
    app.close()
    wait_stopped(home)
    assert not pid_alive(daemon) and not any(pid_alive(p) for p in children)        # ORCHA and Nomi went with it
    assert not (home.home / "state" / "runtime.json").exists()
    log = (home.home / "logs" / "runtime.log").read_text(encoding="utf-8")
    assert "shutting down" in log and "runtime stopped" in log


@pytest.mark.slow
def test_runtime_stays_up_until_the_last_app_closes(home):
    a = AnviraRuntime.connect("anvira-notes", env=home.env, idle_grace_s=3)
    b = AnviraRuntime.connect("anvira-study", env=home.env, idle_grace_s=3)
    assert {x["app"] for x in b.lifecycle()["leases"]} == {"anvira-notes", "anvira-study"}
    a.close()
    time.sleep(8)
    assert probe(home.env).running and [x["app"] for x in b.lifecycle()["leases"]] == ["anvira-study"]   # Study still needs it
    b.close()
    wait_stopped(home)


@pytest.mark.slow
def test_a_crashed_app_does_not_keep_the_runtime_alive(home):
    app = AnviraRuntime.connect("anvira-dev", env=home.env, idle_grace_s=3)
    app._closed.set()                                                                # heartbeat thread dies, lease is never released
    t0 = time.monotonic()
    wait_stopped(home, 40)
    assert time.monotonic() - t0 < 30                                                # ~ ttl(6) + grace(3), not "forever"
    assert "expired without release" in (home.home / "logs" / "runtime.log").read_text(encoding="utf-8")


@pytest.mark.slow
def test_running_work_keeps_the_runtime_alive_after_the_app_closes(home, fake_llm):
    app = AnviraRuntime.connect("anvira-notes", env=home.env, idle_grace_s=3)
    owner = Http(probe(home.env).base_url, read_token("owner", home.env), 60)
    rec = owner.json("POST", "/v1/providers", {"label": "Fake", "base_url": fake_llm.base + "/v1", "model": "fake-model"})
    owner.json("POST", "/v1/models/select", {"id": rec["id"]})
    fake_llm.mode, fake_llm.delay = "slow", 12
    try:
        job = app.orcha.run("A long task.")
        wait_for(lambda: app.jobs.get(job.id).data["meta"].get("orcha_run_id"), 20, message="ORCHA run started")
        app.close()                                                                  # the user closed the app mid-task
        time.sleep(9)                                                                # grace (3) + ttl (6) have both passed
        assert probe(home.env).running                                               # ...but the job is still going
        assert owner.json("GET", "/v1/lifecycle")["busy"] >= 1
        wait_for(lambda: owner.json("GET", f"/v1/jobs/{job.id}")["job"]["state"] in ("completed", "failed"), 60, message="job end")
        wait_stopped(home, 40)                                                       # done + grace elapsed -> stops
    finally:
        fake_llm.mode, fake_llm.delay = "ok", 0.0


@pytest.mark.slow
def test_persistent_runtime_never_stops_by_itself(tmp_path, fake_llm, fake_llama):
    h = start_isolated(tmp_path, fake_llm, fake_llama, extra_config=config(), auto_stop=False)
    try:
        app = AnviraRuntime.connect("anvira-notes", env=h.env)
        assert app.lifecycle()["mode"] == "persistent"
        app.close()
        time.sleep(10)                                                               # no leases, well beyond any grace
        assert probe(h.env).running
        cli(["status"])
    finally:
        stop_isolated(h)


@pytest.mark.slow
def test_an_open_app_survives_the_runtime_being_stopped_underneath_it(home):
    app = AnviraRuntime.connect("anvira-notes", env=home.env, idle_grace_s=4)
    first = probe(home.env).pid
    subprocess.run(["taskkill", "/PID", str(first), "/T", "/F"] if sys.platform == "win32" else ["kill", "-9", str(first)], capture_output=True)
    wait_for(lambda: not pid_alive(first), 15, message="runtime killed")
    assert app.health()["status"] == "ok"                                            # transparently started again
    assert probe(home.env).pid != first
    assert [x["app"] for x in app.lifecycle()["leases"]] == ["anvira-notes"]         # and the app is registered as open again
    app.close()
    wait_stopped(home)


@pytest.mark.slow
def test_open_close_open_cycles(home):
    for _ in range(2):
        app = AnviraRuntime.connect("anvira-study", env=home.env, idle_grace_s=3)
        assert app.status()["lifecycle"]["mode"] == "on-demand"
        app.close()
        wait_stopped(home)


@pytest.mark.slow
def test_cli_modes(home, capsys):
    import os
    old = {k: os.environ.get(k) for k in home.env}
    os.environ.update(home.env)
    try:
        assert cli(["runtime", "start", "--auto-stop", "--idle-grace", "3"]) == 0
        capsys.readouterr()
        assert cli(["status"]) == 0
        assert "on-demand" in capsys.readouterr().out
        wait_stopped(home)                                                           # nobody opened it: gone after the grace
        assert cli(["runtime", "start"]) == 0                                        # a hand-started runtime is persistent
        capsys.readouterr()
        cli(["status"])
        assert "persistent" in capsys.readouterr().out
        bootstrap.stop_runtime(home.env)
        # an action command starts it on demand and releases its lease when done
        assert cli(["model", "list", "--installed"]) == 0
        capsys.readouterr()
        lc = Http(probe(home.env).base_url, read_token("owner", home.env), 30).json("GET", "/v1/lifecycle")
        assert lc["mode"] == "on-demand" and lc["leases"] == [] and lc["idle_grace_s"] == 120.0
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


@pytest.mark.slow
def test_lease_api_is_scoped_to_the_calling_app(home):
    a = AnviraRuntime.connect("app-a", env=home.env, idle_grace_s=30, keep_alive=False)
    b = AnviraRuntime.connect("app-b", env=home.env, idle_grace_s=30, keep_alive=False)
    lease = a._http.json("POST", "/v1/leases", {"ttl_s": 30})["lease"]
    with pytest.raises(AnviraError) as e:
        b._http.json("POST", f"/v1/leases/{lease['id']}/heartbeat")
    assert e.value.code == "lease_not_found" and e.value.status == 404
    assert b._http.json("DELETE", f"/v1/leases/{lease['id']}")["released"] is False        # cannot release another app's lease
    assert a._http.json("POST", f"/v1/leases/{lease['id']}/heartbeat")["lease"]["app"] == "app-a"
    assert a._http.json("DELETE", f"/v1/leases/{lease['id']}")["released"] is True
