"""CLI, terminal UI, and cross-app model discovery/announcing (real daemon)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from anvira_client import AnviraError, bootstrap
from anvira_runtime.cli import tui
from anvira_runtime.cli.main import build_parser, main as cli
from conftest import wait_for
from fakes import FAKE_GGUF

pytestmark = pytest.mark.slow


def run_cli(capsys, *argv):
    code = cli(list(argv))
    cap = capsys.readouterr()
    return code, cap.out, cap.err


@pytest.fixture(scope="module")
def model_ready(runtime, provider_model):
    """Make the fake provider the active model through the CLI itself."""
    runtime.env  # runtime fixture sets os.environ for in-process CLI calls
    return provider_model


# ------------------------------------------------------------------ help / usage
def test_help_lists_every_command_group():
    out = subprocess.run([sys.executable, "-m", "anvira_runtime", "--help"], capture_output=True, text=True,
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"}).stdout
    for word in ("status", "doctor", "version", "runtime", "model", "orcha", "nomi", "chat", "config", "app", "aicl", "ui",
                 "hardware", "job", "--json", "exit codes", "examples"):
        assert word in out, word


def test_subcommand_help_and_invalid_usage(capsys):
    for args in (["runtime", "--help"], ["model", "--help"], ["orcha", "run", "--help"], ["nomi", "--help"], ["app", "--help"]):
        with pytest.raises(SystemExit) as e:
            cli(args)
        assert e.value.code == 0
        assert "usage: anvira" in capsys.readouterr().out
    for bad in (["bogus"], ["model"], ["model", "use"], ["orcha", "run"], ["config", "get", "nope.key"], ["--nope"]):
        with pytest.raises(SystemExit) as e:
            cli(bad)
        assert e.value.code == 2, bad
    capsys.readouterr()


def test_version_flag_and_command(capsys, runtime):
    with pytest.raises(SystemExit):
        cli(["--version"])
    assert "anvira 1.0.0" in capsys.readouterr().out
    code, out, _ = run_cli(capsys, "version", "--json")
    d = json.loads(out)
    assert code == 0 and d["cli"] == "1.0.0" and d["runtime"]["running"] and d["runtime"]["api_version"] == 1


# ---------------------------------------------------------------- runtime missing
def test_missing_runtime_messages_and_exit_codes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANVIRA_RUNTIME_HOME", str(tmp_path / "nothing-here"))
    monkeypatch.setattr(bootstrap, "runtime_python", lambda env=None: None)
    code, out, _ = run_cli(capsys, "status")
    assert code == 3 and "Runtime unavailable" in out and "anvira runtime install" in out
    code, out, _ = run_cli(capsys, "status", "--json")
    assert code == 3 and json.loads(out)["installed"] is False
    code, _, err = run_cli(capsys, "model", "list")
    assert code == 3 and "Runtime unavailable" in err
    code, _, err = run_cli(capsys, "runtime", "start")
    assert code == 3
    code, out, _ = run_cli(capsys, "model", "list", "--json")
    assert code == 3 and json.loads(out)["error"]["code"] == "runtime_not_installed"


def test_installed_but_stopped_runtime(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANVIRA_RUNTIME_HOME", str(tmp_path / "stopped"))
    code, out, _ = run_cli(capsys, "status")
    assert code == 4 and "not running" in out and "anvira runtime start" in out
    code, out, _ = run_cli(capsys, "runtime", "stop")                                      # stopping a stopped runtime is fine
    assert code == 0 and "was not running" in out


def test_doctor_works_without_a_running_runtime(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANVIRA_RUNTIME_HOME", str(tmp_path / "doc"))
    code, out, _ = run_cli(capsys, "doctor", "--json")
    d = json.loads(out)
    ids = {c["id"]: c for c in d["checks"]}
    assert code in (0, 8) and {"python", "directories", "config", "storage", "orcha_install", "nomi_install", "aicl", "hardware"} <= set(ids)
    assert ids["runtime_process"]["status"] == "warn" and "anvira runtime start" in ids["runtime_process"]["fix"]
    assert ids["orcha_deps"]["status"] == "ok" and ids["nomi_deps"]["status"] == "ok"


def test_doctor_detects_broken_config_and_missing_service(tmp_path, monkeypatch, capsys):
    home = tmp_path / "broken"
    (home / "config").mkdir(parents=True)
    (home / "config" / "config.json").write_text("{ nope", encoding="utf-8")
    monkeypatch.setenv("ANVIRA_RUNTIME_HOME", str(home))
    monkeypatch.setenv("ANVIRA_ORCHA_DIR", str(tmp_path / "missing"))
    code, out, _ = run_cli(capsys, "doctor", "--json")
    d = json.loads(out)
    cfg = next(c for c in d["checks"] if c["id"] == "config")
    assert cfg["status"] == "fail" and "Broken configuration" in cfg["message"]
    assert d["status"] == "fail" and code == 8


def test_doctor_detects_port_conflict(tmp_path, monkeypatch, capsys):
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    home = tmp_path / "conflict"
    (home / "config").mkdir(parents=True)
    (home / "config" / "config.json").write_text(json.dumps({"api": {"port": port}}), encoding="utf-8")
    monkeypatch.setenv("ANVIRA_RUNTIME_HOME", str(home))
    _, out, _ = run_cli(capsys, "doctor", "--json")
    p = next(c for c in json.loads(out)["checks"] if c["id"] == "port")
    s.close()
    assert p["status"] == "fail" and "in use" in p["message"] and "api.port 0" in p["fix"]


# ---------------------------------------------------------------- live CLI usage
def test_status_and_read_commands(runtime, capsys, model_ready):
    code, out, _ = run_cli(capsys, "status")
    assert code == 0 and "ORCHA" in out and "Nomi" in out and "AICL" in out and "running" in out
    code, out, _ = run_cli(capsys, "status", "--json")
    assert json.loads(out)["services"]["orcha"]["state"] == "running"
    for args in (["hardware"], ["orcha", "status"], ["nomi", "status"], ["aicl", "status"], ["aicl", "trace"], ["runtime", "info"],
                 ["config", "list"], ["config", "path"], ["app", "list"], ["model", "recommend"], ["model", "status"], ["model", "dir"],
                 ["model", "discover"], ["model", "provider", "list"]):
        code, out, err = run_cli(capsys, *args)
        assert code == 0, (args, err)
        assert out.strip(), args


def test_model_commands_and_json(runtime, capsys, model_ready):
    code, out, _ = run_cli(capsys, "model", "provider", "list", "--json")
    assert json.loads(out)["providers"][0]["id"] == "cloud:fake"
    assert "sk-test-secret" not in out
    code, _, _ = run_cli(capsys, "model", "use", "cloud:fake")
    assert code == 0
    code, out, _ = run_cli(capsys, "model", "info", "cloud:fake", "--json")
    assert json.loads(out)["active"] is True
    code, out, err = run_cli(capsys, "model", "compat", "qwen3-8b")
    assert "needs about" in out
    code, _, err = run_cli(capsys, "model", "info", "nope")
    assert code == 6 and "not found" in err.lower()
    code, _, err = run_cli(capsys, "model", "install", "qwen3-4b")                        # explicit consent required, non-interactive
    assert code == 1 and "confirmation" in err.lower() and "--yes" in err


def test_provider_key_from_environment_is_stored_without_echo(runtime, capsys, monkeypatch):
    monkeypatch.setenv("MY_KEY", "sk-env-secret-abcdefghijklmn")
    code, out, _ = run_cli(capsys, "model", "provider", "add", "--base-url", runtime.fake.base + "/v1", "--model", "m2",
                           "--label", "EnvKey", "--api-key-env", "MY_KEY")
    assert code == 0 and "key stored" in out and "sk-env-secret" not in out
    assert run_cli(capsys, "model", "provider", "remove", "cloud:envkey")[0] == 0


def test_orcha_and_job_commands(runtime, capsys, model_ready, fake):
    code, out, _ = run_cli(capsys, "orcha", "run", "What is a semaphore?")
    assert code == 0 and "fake reply" in out and "completed" in out
    code, out, _ = run_cli(capsys, "orcha", "run", "Again?", "--json")
    job = json.loads(out)["job"]
    assert job["state"] == "completed"
    code, out, _ = run_cli(capsys, "job", "get", job["id"])
    assert code == 0 and job["id"] in out
    code, out, _ = run_cli(capsys, "orcha", "jobs")
    assert job["id"] in out
    code, out, _ = run_cli(capsys, "orcha", "run", "later", "--no-wait", "--json")
    jid = json.loads(out)["job"]["id"]
    code, out, _ = run_cli(capsys, "orcha", "cancel", jid)
    assert code in (0, 1)
    code, _, err = run_cli(capsys, "job", "get", "job_doesnotexist")
    assert code == 6 and "not found" in err.lower()


def test_orcha_failure_exit_code(runtime, capsys, model_ready, fake):
    fake.mode = "error"
    code, _, err = run_cli(capsys, "chat", "hi", "--no-stream")
    assert code == 1 and "model" in err.lower()


def test_chat_command_streams(runtime, capsys, model_ready, fake):
    code, out, _ = run_cli(capsys, "chat", "hello there")
    assert code == 0 and out.strip() == "fake stream reply"
    code, out, _ = run_cli(capsys, "chat", "hello", "--no-stream")
    assert out.strip() == "fake reply to: hello"


def test_nomi_commands(runtime, capsys):
    code, out, _ = run_cli(capsys, "nomi", "store", "CLI stored memory about zebras", "--title", "Zebra", "--tag", "animals")
    assert code == 0 and "Stored" in out
    code, out, _ = run_cli(capsys, "nomi", "search", "zebras", "--json")
    item = json.loads(out)["items"][0]
    assert item["title"] == "Zebra" and item["app"] == "owner"
    code, out, _ = run_cli(capsys, "nomi", "inspect", item["id"])
    assert code == 0 and "zebras" in out
    assert run_cli(capsys, "nomi", "delete", item["id"])[0] == 0
    assert "No matching" in run_cli(capsys, "nomi", "search", "zebras")[1]


def test_config_and_app_commands(runtime, capsys):
    assert run_cli(capsys, "config", "set", "orcha.job_timeout_s", "61")[0] == 0
    code, out, _ = run_cli(capsys, "config", "get", "orcha.job_timeout_s")
    assert out.strip() == "61.0"
    code, _, err = run_cli(capsys, "config", "set", "api.host", "0.0.0.0")
    assert code != 0 and "loopback" in err
    code, _, err = run_cli(capsys, "app", "grant", "cli-app", "make.coffee")
    assert code == 1 and "Unknown permission" in err
    runtime.connect("cli-app")
    code, out, _ = run_cli(capsys, "app", "grant", "cli-app", "models.manage")
    assert code == 0 and "models.manage" in out
    code, out, _ = run_cli(capsys, "app", "show", "cli-app", "--json")
    assert "models.manage" in json.loads(out)["permissions"]
    assert run_cli(capsys, "app", "revoke", "cli-app", "--yes")[0] == 0


def test_logs_command_redacts(runtime, capsys):
    code, out, _ = run_cli(capsys, "runtime", "logs", "-n", "2000")
    assert code == 0 and " INFO " in out and "sk-test-secret" not in out       # the log is long by now: only its tail is shown
    code, out, _ = run_cli(capsys, "runtime", "logs", "-s", "orcha", "--json")
    assert json.loads(out)["exists"] is True
    code, out, err = run_cli(capsys, "runtime", "logs", "-s", "../etc")            # a clean error and exit code, never a traceback
    assert code != 0 and "Unknown log" in (out + err)


# ------------------------------------------------------------------- terminal UI
def test_tui_frame_shows_runtime_state(runtime, capsys, model_ready):
    code, out, _ = run_cli(capsys, "ui", "--once")
    assert code == 0
    for word in ("Anvira Runtime", "ORCHA", "Nomi", "AICL", "Jobs", "Activity", "Output", "anvira>"):
        assert word in out, word


def test_tui_runs_orcha_and_other_commands(runtime, capsys, model_ready, fake):
    code, out, _ = run_cli(capsys, "ui", "--exec", "help;;models;;run explain a deadlock;;jobs;;remember tui memory about otters;;memory otters;;logs orcha 2;;bogus")
    assert code == 0
    assert "commands:" in out and "cloud:fake" in out and "completed" in out and "fake reply" in out
    assert "otters" in out and "unknown command 'bogus'" in out and "job_" in out
    code, out, _ = run_cli(capsys, "ui", "--exec", "chat hi there;;use cloud:fake;;use ghost;;cancel job_zzz")
    assert "fake stream reply" in out and "active model: cloud:fake" in out and "not found" in out.lower()


def test_tui_survives_a_dead_runtime(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANVIRA_RUNTIME_HOME", str(tmp_path / "dead"))
    code, out, _ = run_cli(capsys, "ui", "--once")
    assert code == 4 and "not connected" in out and "anvira runtime start" in out


def test_tui_rendering_is_bounded_and_ansi_safe():
    d = tui.Dashboard(lambda: None, color=True)
    d.uni = False
    d.status = {"status": "ok", "runtime_version": "1.0.0", "uptime_s": 5, "apps": 1, "paths": {},
                "model": {"active": "m" * 90, "state": "running", "installed_count": 1, "models_dir": "/very/long/" * 12, "backend_info": {}},
                "services": {"orcha": {"state": "failed", "last_error": "x" * 300}, "nomi": {"state": "running", "port": 1}, "aicl": {"state": "running"}}}
    d.jobs = [{"id": "job_abc", "kind": "orcha.run", "state": "running", "created_at": 0}]
    for w, h in ((60, 18), (80, 24), (120, 40), (200, 60)):
        rows = d.render(w, h, "typed text")
        assert len(rows) <= h and all(tui.visible_len(r) <= w for r in rows), (w, h)
    assert tui.fit("\033[31mred\033[0m text", 5).count("\033[31m") == 1 and tui.visible_len(tui.fit("abc", 10)) == 10
    assert tui.summarize_event({"node": "plan", "kind": "checkpoint", "data": {"next_node": "select"}}) == "plan checkpoint -> select"
    assert tui.summarize_event({"node": "execute", "kind": "node_end", "data": {"duration_ms": 89.87}}) == "execute node_end 90ms"


# ---------------------------------------- models added via ANY Anvira app are found
def _appdata(runtime):
    return Path(runtime.env["APPDATA"])


def test_models_from_other_anvira_apps_are_discovered_and_usable(runtime, provider_model, capsys):
    ad = _appdata(runtime)
    custom = ad.parent / "Where Anvira Keeps Models"                       # like a user-chosen models folder
    custom.mkdir()
    (custom / "from-anvira-Q4_K_M.gguf").write_bytes(FAKE_GGUF)
    (ad / "Anvira").mkdir(exist_ok=True)
    (ad / "Anvira" / "model-storage.json").write_text(json.dumps({"version": 1, "modelsDir": str(custom)}))
    notes_models = ad / "Anvira Notes" / "models"
    notes_models.mkdir(parents=True)
    (notes_models / "from-notes-Q8_0.gguf").write_bytes(FAKE_GGUF)

    inst = {m["id"]: m for m in runtime.owner.json("GET", "/v1/models/installed")["models"]}
    assert inst["from-anvira-q4_k_m"]["location"] == "app" and inst["from-anvira-q4_k_m"]["origin"] == "Anvira"
    assert inst["from-notes-q8_0"]["origin"] == "Anvira Notes"
    disc = runtime.owner.json("GET", "/v1/models/discovered")
    assert {x["app"] for x in disc["locations"]} >= {"Anvira", "Anvira Notes"}
    code = cli(["model", "discover"])
    assert code == 0 and "Anvira Notes" in capsys.readouterr().out

    study = runtime.connect("study-disc")
    assert "from-anvira-q4_k_m" in [m["id"] for m in study.models.installed()]           # another app sees it, no download
    # ...and any app can use it whenever it is called: naming the model switches the backend to it
    reply = study.chat([{"role": "user", "content": "hello"}], model="from-anvira-q4_k_m")
    assert reply["choices"][0]["message"]["content"] == "local reply to: hello" and reply["runtime"]["kind"] == "local"
    assert study.models.active()["active"] == "from-anvira-q4_k_m"
    job = study.orcha.run("Summarise.", model="from-notes-q8_0", wait=True)
    assert job.state == "completed" and "local reply" in job.unwrap()["answer"]
    assert study.models.active()["active"] == "from-notes-q8_0"
    with pytest.raises(AnviraError) as e:
        study.chat([{"role": "user", "content": "x"}], model="never-heard-of-it")
    assert e.value.code == "model_not_found"
    study.models.use(provider_model)


def test_discovered_files_are_not_deleted_without_confirmation(runtime):
    f = _appdata(runtime).parent / "Where Anvira Keeps Models" / "from-anvira-Q4_K_M.gguf"
    with pytest.raises(AnviraError) as e:
        runtime.owner.json("POST", "/v1/models/remove", {"id": "from-anvira-q4_k_m", "delete_file": True})
    assert e.value.code == "confirmation_required" and f.exists()


def test_an_app_can_announce_where_its_model_lives(runtime):
    somewhere = _appdata(runtime).parent / "Study Downloads" / "deep"
    somewhere.mkdir(parents=True)
    f = somewhere / "study-downloaded-Q4_K_M.gguf"
    f.write_bytes(FAKE_GGUF)
    study, notes = runtime.connect("study-ann"), runtime.connect("notes-ann")
    res = study.models.register(str(f))
    assert res["kind"] == "file" and res["model"]["id"] == "study-downloaded-q4_k_m" and res["model"]["source"]["app"] == "study-ann"
    assert not list((runtime.home / "models").glob("study-downloaded*"))                # nothing copied
    assert "study-downloaded-q4_k_m" in [m["id"] for m in notes.models.installed()]     # visible to every app
    folder = _appdata(runtime).parent / "Dev Models"
    folder.mkdir()
    (folder / "dev-Q4_K_M.gguf").write_bytes(FAKE_GGUF)
    res = notes.models.register(str(folder))
    assert res["kind"] == "directory" and res["models_found"] >= 1
    assert "dev-q4_k_m" in [m["id"] for m in study.models.installed()]
    with pytest.raises(AnviraError) as e:
        study.models.register(str(somewhere / "missing.gguf"))
    assert e.value.code == "invalid_model_file"


def test_interactive_dashboard_loop_renders_and_runs_commands(runtime, monkeypatch, capsys, model_ready):
    """Drive the real interactive loop with scripted key presses (msvcrt/termios replaced by a fake key source)."""
    keys = list("help") + ["enter"] + list("models") + ["enter"] + ["up", "esc"] + list("quit") + ["enter"]

    class FakeKeys:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def poll(self, timeout):
            time.sleep(0.05)
            return keys.pop(0) if keys else "quit"

    import time
    monkeypatch.setattr(tui, "Keys", FakeKeys)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    code = tui.run_ui(lambda: runtime.owner)
    out = capsys.readouterr().out
    assert code == 0
    assert "\033[?1049h" in out and "\033[?1049l" in out                 # entered and left the alternate screen
    assert "Anvira Runtime" in out and "commands:" in out and "cloud:fake" in out
    assert "anvira>" in out


def test_bare_anvira_in_a_terminal_opens_the_dashboard(runtime, monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    called = {}
    monkeypatch.setattr(tui, "run_ui", lambda factory, once=False, exec_cmd=None: called.setdefault("ui", True) and 0)
    assert cli([]) == 0 and called.get("ui")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)                # piped: keep printing help (scripts)
    called.clear()
    assert cli([]) == 0 and not called and "usage: anvira" in capsys.readouterr().out
