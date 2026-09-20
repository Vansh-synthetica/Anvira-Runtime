"""The real installer: consent -> venv + services -> start -> connect -> shared by later apps -> update."""
import json
import os
import sys
from pathlib import Path

import pytest

from anvira_client import AnviraRuntime, bootstrap
from anvira_client.discovery import probe
from anvira_client.paths import runtime_dirs
from conftest import REPO

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def installed_env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("install")
    appdata = tmp / "appdata"
    appdata.mkdir()
    env = {"ANVIRA_RUNTIME_HOME": str(tmp / "AnviraRuntime"), "APPDATA": str(appdata), "LOCALAPPDATA": str(appdata),
           "ANVIRA_INSTALL_SYSTEM_SITE": "1"}       # reuse already-present heavy deps; pip still fetches anything missing
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    yield env
    bootstrap.stop_runtime(env, timeout=30)
    for k, v in old.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_platform_detection():
    info = bootstrap.platform_info()
    assert info["os"] in ("windows", "macos", "linux") and info["arch"] and info["python"].count(".") >= 1


def test_bundle_source_is_found_or_reported(tmp_path):
    from anvira_client import AnviraError
    assert bootstrap.find_source(str(REPO)) == REPO
    with pytest.raises(AnviraError) as e:
        bootstrap._materialize(Path("/definitely/not/a/bundle"), tmp_path, print)
    assert e.value.code == "bad_install_source"


def test_first_app_installs_only_after_the_user_agrees(installed_env):
    home = Path(installed_env["ANVIRA_RUNTIME_HOME"])
    assert not probe(installed_env).installed
    prompts, status = [], []

    def ask_user(info):
        prompts.append(info.as_dict())
        return True                                                             # the user clicked [Install]

    app = AnviraRuntime.connect("anvira-notes", env=installed_env, install=ask_user, source=str(REPO), on_status=status.append)
    assert len(prompts) == 1 and prompts[0]["installed"] is False
    assert any("Installing" in s for s in status) and "Starting runtime..." in status
    rec = json.loads((home / "install.json").read_text())
    assert rec["version"] == "1.0.0" and Path(rec["entry"]).exists() and rec["os"] and rec["arch"]
    assert app.health()["status"] == "ok" and app.info.runtime_version == "1.0.0"

    # installed layout: private venv, infrastructure copied, no tests/vendor bloat, no models downloaded
    for name in ("orcha", "nomi", "aicl"):
        assert (home / "services" / name).is_dir()
    assert (home / "services" / "aicl" / "aicl" / "bin" / "codec_api.py").exists()
    assert not (home / "services" / "orcha" / "vendor").exists() and not (home / "services" / "orcha" / "tests").exists()
    assert not list((home / "models").glob("*.gguf")) and not any(Path(p).suffix == ".gguf" for p in home.rglob("*") if p.is_file())
    for d in ("state", "config", "logs", "data"):
        assert (home / d).is_dir()

    st = app.status()
    assert st["status"] == "ok" and st["services"]["orcha"]["source"]["path"].startswith(str(home / "services"))
    assert st["services"]["nomi"]["source"]["path"].startswith(str(home / "services"))       # running the INSTALLED copies
    assert Path(sys.prefix) != home / "venv"
    assert app.models.catalog() and app.models.installed() == []                            # models are a separate step


def test_later_apps_share_the_installed_runtime(installed_env):
    calls = []
    study = AnviraRuntime.connect("anvira-study", env=installed_env, install=lambda i: calls.append(i) or True, source=str(REPO))
    dev = AnviraRuntime.connect("anvira-dev", env=installed_env, install=lambda i: calls.append(i) or True, source=str(REPO))
    assert calls == []                                                                       # no second install
    assert study.info.pid == dev.info.pid and probe(installed_env).running
    venvs = [p for p in Path(installed_env["ANVIRA_RUNTIME_HOME"]).iterdir() if p.name.startswith("venv")]
    assert len(venvs) == 1


def test_install_is_idempotent_and_update_replaces_an_older_runtime(installed_env):
    home = Path(installed_env["ANVIRA_RUNTIME_HOME"])
    same = bootstrap.install_runtime(source=str(REPO), env=installed_env)
    assert same["version"] == "1.0.0"                                                        # nothing to do
    marker = home / "install.json"
    rec = json.loads(marker.read_text())
    rec["version"] = "0.9.0"                                                                 # pretend an older runtime is installed
    marker.write_text(json.dumps(rec))
    said = []
    old_pid = probe(installed_env).pid
    updated = bootstrap.install_runtime(source=str(REPO), env=installed_env, on_status=said.append)
    assert updated["version"] == "1.0.0" and any("Stopping the running runtime" in s for s in said)
    now = probe(installed_env)
    # stopped for the update. Apps that are still open may bring it back on the NEW files afterwards - never before, never the old process
    assert not now.running or now.pid != old_pid
    assert not bootstrap.release.update_in_progress(installed_env)                          # the update lock is always released
    app = AnviraRuntime.connect("anvira-notes", env=installed_env)                            # existing app token still works
    assert app.info.runtime_version == "1.0.0" and app.health()["status"] == "ok"


def test_incompatible_runtime_is_reported_to_the_app(installed_env):
    from anvira_client import IncompatibleRuntime
    with pytest.raises(IncompatibleRuntime) as e:
        AnviraRuntime.connect("anvira-notes", env=installed_env, require_api=2)
    assert "API v2" in str(e.value) and "anvira runtime update" in str(e.value)               # runtime older than the app needs
    with pytest.raises(IncompatibleRuntime) as e:
        AnviraRuntime.connect("anvira-notes", env=installed_env, require_api=0)             # runtime newer than the app knows
    assert "Update this application" in str(e.value)
    with pytest.raises(IncompatibleRuntime) as e:
        AnviraRuntime.connect("anvira-notes", env=installed_env, min_version="9.0.0")
    assert "runtime update" in e.value.hint
    assert AnviraRuntime.connect("anvira-notes", env=installed_env, min_version="1.0.0").health()


def test_cli_install_needs_consent(tmp_path, monkeypatch, capsys):
    from anvira_runtime.cli.main import main
    monkeypatch.setenv("ANVIRA_RUNTIME_HOME", str(tmp_path / "consent"))
    monkeypatch.setattr(bootstrap, "runtime_python", lambda env=None: None)
    code = main(["runtime", "install", "--source", str(REPO)])                               # non-interactive, no --yes
    err = capsys.readouterr().err
    assert code == 1 and "confirmation" in err.lower() and "--yes" in err
    assert not (tmp_path / "consent" / "install.json").exists()
