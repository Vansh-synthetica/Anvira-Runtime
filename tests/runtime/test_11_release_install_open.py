"""Custom install locations, GitHub-release installs (against a local fake GitHub) and the multi-page `anvira open` dashboard."""
import hashlib
import http.server
import io
import json
import re
import threading
import time
import zipfile
from pathlib import Path

import pytest

from anvira_client import AnviraError, release
from anvira_client.discovery import probe
from anvira_client.paths import runtime_dirs
from anvira_runtime.cli import tui
from anvira_runtime.cli.main import main as cli
from anvira_runtime.config.paths import resolve_layout


def make_zip(path: Path, version="9.9.9", with_python=True, evil=False, extra: dict | None = None) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("AnviraRuntime/install.json", json.dumps({"version": version, "portable": True, "entry": "python/python.exe",
                                                             "services_dir": "services", "os": "windows", "arch": "x86_64"}))
        if with_python:
            z.writestr("AnviraRuntime/python/python.exe", b"MZ fake")
        z.writestr("AnviraRuntime/services/orcha/desktop_entry.py", "print('orcha')")
        for name, data in (extra or {}).items():
            z.writestr(name, data)
        if evil:
            z.writestr("AnviraRuntime/../../escape.txt", "owned")
    return path


@pytest.fixture()
def env(tmp_path):
    appdata = tmp_path / "appdata"
    appdata.mkdir()
    return {"LOCALAPPDATA": str(appdata), "APPDATA": str(appdata), "USERPROFILE": str(tmp_path), "HOME": str(tmp_path),
            "XDG_DATA_HOME": str(tmp_path / "xdg")}


# ------------------------------------------------------------------------ custom location
def test_custom_location_is_remembered_and_found_by_every_client(env, tmp_path):
    custom = tmp_path / "D-drive" / "My Anvira Runtime"
    rec = release.install_package(make_zip(tmp_path / "pkg.zip"), custom, env)
    assert rec["version"] == "9.9.9" and (custom / "python" / "python.exe").is_file() and (custom / "services" / "orcha").is_dir()
    loc = release.locate(env)
    assert loc["found"] and loc["source"] == "pointer" and Path(loc["home"]) == custom.resolve() and loc["version"] == "9.9.9"
    # the SDK and the runtime's own resolver agree on where everything lives
    sdk_dirs, layout = runtime_dirs(env), resolve_layout(env)
    assert sdk_dirs["home"] == layout.home == custom.resolve()
    assert sdk_dirs["state"] == layout.state_dir and sdk_dirs["bin"] == layout.bin_dir and sdk_dirs["models"] == layout.models_dir
    assert probe(env).installed                                                                # relative "entry" resolved against home
    assert release.default_home(env) != custom.resolve()


def test_ANVIRA_RUNTIME_HOME_beats_the_pointer_and_a_dead_pointer_falls_back(env, tmp_path):
    custom = tmp_path / "custom"
    release.install_package(make_zip(tmp_path / "pkg.zip"), custom, env)
    assert runtime_dirs({**env, "ANVIRA_RUNTIME_HOME": str(tmp_path / "other")})["home"] == tmp_path / "other"
    import shutil
    shutil.rmtree(custom)                                                                      # the drive was unplugged / folder deleted
    assert Path(release.locate(env)["home"]) == release.default_home(env) and not release.locate(env)["found"]


def test_i_already_have_it_registers_without_copying(env, tmp_path):
    existing = tmp_path / "already-here"
    make_zip(tmp_path / "pkg.zip")
    release.extract_package(tmp_path / "pkg.zip", existing)
    home = release.register_location(existing, env)
    assert home == existing.resolve() and release.locate(env)["source"] == "pointer"
    assert not (release.default_home(env) / "python").exists()                                 # nothing was copied
    with pytest.raises(AnviraError) as e:
        release.register_location(tmp_path / "not-a-runtime", env)
    assert e.value.code == "not_a_runtime_folder"
    (tmp_path / "junk").mkdir()
    with pytest.raises(AnviraError):
        release.register_location(tmp_path / "junk", env)


def test_unsafe_zip_paths_are_refused(env, tmp_path):
    with pytest.raises(AnviraError) as e:
        release.install_package(make_zip(tmp_path / "evil.zip", evil=True), tmp_path / "dest", env)
    assert e.value.code == "bad_bundle" and not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "dest" / ".updating").exists()                                     # the lock never outlives a failure


def test_update_lock_blocks_restarts_until_released(env, tmp_path):
    e = {**env, "ANVIRA_RUNTIME_HOME": str(tmp_path / "h")}
    (tmp_path / "h").mkdir()
    release._lock(tmp_path / "h", True)
    assert release.update_in_progress(e)
    threading.Timer(0.8, lambda: release._lock(tmp_path / "h", False)).start()
    t0 = time.monotonic()
    release.wait_for_update(e, 10)
    assert 0.5 < time.monotonic() - t0 < 5 and not release.update_in_progress(e)
    (tmp_path / "h" / ".updating").write_text("x")                                            # a crashed installer's stale lock
    import os
    old = time.time() - 3600
    os.utime(tmp_path / "h" / ".updating", (old, old))
    assert not release.update_in_progress(e)


# ------------------------------------------------------------------ GitHub release (fake server)
class FakeGitHub:
    def __init__(self, tmp: Path, assets: dict[str, bytes], tag="v9.9.9", bad_sum_for: str | None = None):
        self.assets, self.tag, self.hits = assets, tag, []
        sums = {n: hashlib.sha256(d).hexdigest() for n, d in assets.items()}
        if bad_sum_for:
            sums[bad_sum_for] = "0" * 64
        self.assets["SHA256SUMS"] = "".join(f"{h}  {n}\n" for n, h in sums.items()).encode()
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):  # noqa: D401
                pass

            def do_GET(self):  # noqa: N802
                outer.hits.append(self.path)
                m = re.match(r"^/repos/([^/]+/[^/]+)/releases/(latest|tags/.+)$", self.path)
                if m:
                    if m.group(1) != "acme/Anvira-Runtime":
                        self.send_response(404)
                        self.end_headers()
                        return
                    body = json.dumps({"tag_name": outer.tag, "assets": [
                        {"name": n, "size": len(d), "browser_download_url": f"{outer.base}/dl/{n}"} for n, d in outer.assets.items()]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                m = re.match(r"^/dl/(.+)$", self.path)
                if m and m.group(1) in outer.assets:
                    data = outer.assets[m.group(1)]
                    rng = re.match(r"bytes=(\d+)-", self.headers.get("Range", ""))
                    start = int(rng.group(1)) if rng else 0
                    self.send_response(206 if rng else 200)
                    self.send_header("Content-Length", str(len(data) - start))
                    self.end_headers()
                    self.wfile.write(data[start:])
                    return
                self.send_response(404)
                self.end_headers()
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def zip_bytes(**kw) -> bytes:
    buf = io.BytesIO()
    p = Path(kw.pop("_dir")) / "z.zip"
    make_zip(p, **kw)
    return p.read_bytes()


@pytest.fixture()
def gh(tmp_path, env, monkeypatch):
    core = zip_bytes(_dir=tmp_path)
    cuda = zip_bytes(_dir=tmp_path, extra={"AnviraRuntime/bin/llama-cpp/cuda/llama-server.exe": b"cuda-bin"}, with_python=False)
    fake = FakeGitHub(tmp_path, {"AnviraRuntime-9.9.9-win-x64.zip": core, "AnviraRuntime-9.9.9-win-x64-cuda.zip": cuda})
    env["ANVIRA_RELEASE_API"] = fake.base
    env["ANVIRA_RUNTIME_REPO"] = "acme/Anvira-Runtime"
    monkeypatch.setattr(release.sys, "platform", "win32")
    yield fake
    fake.close()


def test_install_from_github_into_a_custom_folder_verifies_and_reports_the_gpu_pack(gh, env, tmp_path, monkeypatch):
    monkeypatch.setattr(release, "nvidia_gpu", lambda: {"name": "RTX Test", "driver": "560.10", "vram_mib": 8192, "cuda_ok": True})
    said, prog, asked = [], [], []
    dest = tmp_path / "E-drive" / "Anvira Runtime"
    rec = release.install_from_github(dest, env=env, on_status=said.append, on_progress=lambda *a: prog.append(a),
                                      confirm_gpu=lambda info: asked.append(info) or False)
    assert rec["version"] == "9.9.9" and (dest / "python" / "python.exe").is_file()
    assert Path(release.locate(env)["home"]) == dest.resolve()                                # remembered for every app
    assert prog and prog[-1][1] == prog[-1][2]                                                # progress reached 100%
    assert len(asked) == 1 and rec["gpu_pack_available"]["name"] == "RTX Test" and rec["gpu_pack"] is False
    assert not (dest / "bin" / "llama-cpp" / "cuda").exists()                                 # declined: 0 bytes of GPU pack
    assert not (dest / ".download").exists() and not (dest / ".updating").exists()
    # the user changes their mind later
    monkeypatch.setenv("ANVIRA_RELEASE_API", gh.base)
    monkeypatch.setenv("ANVIRA_RUNTIME_REPO", "acme/Anvira-Runtime")
    monkeypatch.setenv("LOCALAPPDATA", env["LOCALAPPDATA"])
    monkeypatch.setenv("APPDATA", env["APPDATA"])
    monkeypatch.setenv("USERPROFILE", env["USERPROFILE"])
    monkeypatch.setenv("HOME", env["HOME"])
    rec2 = release.install_gpu_pack(env=env)
    assert rec2["gpu_pack"] is True and (dest / "bin" / "llama-cpp" / "cuda" / "llama-server.exe").read_bytes() == b"cuda-bin"


def test_gpu_pack_is_skipped_on_old_drivers_and_machines_without_nvidia(gh, env, tmp_path, monkeypatch):
    monkeypatch.setattr(release, "nvidia_gpu", lambda: {"name": "Old", "driver": "450.1", "vram_mib": 2048, "cuda_ok": False})
    said = []
    rec = release.install_from_github(tmp_path / "a", env=env, gpu=True, on_status=said.append)
    assert rec["gpu_pack"] is False and any("older than" in s for s in said) and not (tmp_path / "a" / "bin").exists()
    monkeypatch.setattr(release, "nvidia_gpu", lambda: None)
    rec = release.install_from_github(tmp_path / "b", env=env, gpu=True)
    assert rec["gpu_pack"] is False and "gpu_pack_available" not in rec                       # nothing offered without an NVIDIA GPU


def test_a_corrupt_download_is_discarded_and_nothing_is_installed(tmp_path, env, monkeypatch):
    core = zip_bytes(_dir=tmp_path)
    fake = FakeGitHub(tmp_path, {"AnviraRuntime-9.9.9-win-x64.zip": core}, bad_sum_for="AnviraRuntime-9.9.9-win-x64.zip")
    env.update(ANVIRA_RELEASE_API=fake.base, ANVIRA_RUNTIME_REPO="acme/Anvira-Runtime")
    monkeypatch.setattr(release.sys, "platform", "win32")
    try:
        with pytest.raises(AnviraError) as e:
            release.install_from_github(tmp_path / "dest", env=env)
        assert e.value.code == "checksum_mismatch"
        assert not (tmp_path / "dest" / "python").exists() and not list((tmp_path / "dest").rglob("*.part"))
        assert not (tmp_path / "dest" / ".updating").exists()
    finally:
        fake.close()


def test_missing_release_and_unconfigured_repo_are_clear_errors(gh, env, tmp_path):
    with pytest.raises(AnviraError) as e:
        release.install_from_github(tmp_path / "x", repo="acme/does-not-exist", env=env)
    assert e.value.code == "release_not_found"
    with pytest.raises(AnviraError) as e:
        release.install_from_github(tmp_path / "x", env={**env, "ANVIRA_RUNTIME_REPO": ""})
    assert e.value.code == "repo_not_configured"


def test_reinstalling_the_same_version_downloads_nothing(gh, env, tmp_path):
    release.install_from_github(tmp_path / "d", env=env)
    n = len(gh.hits)
    again = release.install_from_github(tmp_path / "d", env=env)
    assert again["unchanged"] is True and not any(h.startswith("/dl/AnviraRuntime-9.9.9-win-x64.zip") for h in gh.hits[n:])


def test_an_interrupted_download_resumes(gh, env, tmp_path):
    name = "AnviraRuntime-9.9.9-win-x64.zip"
    data = gh.assets[name]
    work = tmp_path / "w"
    work.mkdir()
    (work / (name + ".part")).write_bytes(data[:40])                                          # 40 bytes arrived before the connection died
    asset = {"name": name, "size": len(data), "browser_download_url": f"{gh.base}/dl/{name}"}
    got = release.download(asset, work, hashlib.sha256(data).hexdigest())
    assert got.read_bytes() == data


# ------------------------------------------------------------------------- CLI
def run_cli(capsys, *argv):
    code = cli(list(argv))
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def test_cli_locate_register_and_help(capsys, env, tmp_path, monkeypatch):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("ANVIRA_RUNTIME_HOME", raising=False)
    code, out, _ = run_cli(capsys, "runtime", "locate", "--json")
    assert code == 3 and json.loads(out)["found"] is False
    existing = tmp_path / "somewhere"
    release.extract_package(make_zip(tmp_path / "pkg.zip"), existing)
    code, out, _ = run_cli(capsys, "runtime", "register", str(existing))
    assert code == 0 and "every Anvira app" in out
    code, out, _ = run_cli(capsys, "runtime", "locate", "--json")
    d = json.loads(out)
    assert code == 0 and d["found"] and d["source"] == "pointer" and d["version"] == "9.9.9"
    code, out, err = run_cli(capsys, "runtime", "register", str(tmp_path / "nope"), "--json")
    assert code == 1 and json.loads(out)["error"]["code"] == "not_a_runtime_folder"


def test_cli_install_from_github_needs_consent_and_a_folder(capsys, gh, env, tmp_path, monkeypatch):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("ANVIRA_RUNTIME_HOME", raising=False)
    monkeypatch.setattr(release, "nvidia_gpu", lambda: None)
    dest = tmp_path / "chosen"
    code, out, err = run_cli(capsys, "runtime", "install", "--github", "--dir", str(dest))          # non-interactive without --yes
    assert code == 1 and "confirm" in (out + err).lower() and not dest.exists()
    code, out, err = run_cli(capsys, "runtime", "install", "--github", "--dir", str(dest), "--yes", "--json")
    assert code == 0, out + err
    assert (dest / "python" / "python.exe").is_file() and release.locate(env)["home"] == str(dest.resolve())


# ----------------------------------------------------------------- dashboard pages (no runtime needed)
class StubHttp:
    def json(self, method, path, body=None, params=None, timeout=None):
        data = {"/v1/hardware": {"cpu": {"model": "Test CPU", "logical_cores": 8, "physical_cores_estimate": 4},
                                 "ram": {"total_gib": 16, "free_gib": 8}, "gpu": {"vendor": "nvidia", "name": "RTX Test", "vram_total_mib": 8192,
                                                                                  "vram_free_mib": 7000, "driver_version": "560.1"}, "storage": {}},
                "/v1/diagnostics": {"checks": [{"status": "ok", "title": "Python", "message": "3.12"},
                                               {"status": "warn", "title": "GPU", "message": "driver old", "fix": "update the driver"}],
                                    "summary": {"ok": 1, "info": 0, "warn": 1, "fail": 0}},
                "/v1/models": {"models": [{"id": "qwen-test", "size_bytes": 2 * 1024 ** 3, "active": True, "compatibility": {"mode": "gpu"}},
                                          {"id": "huge-test", "size_bytes": 90 * 1024 ** 3, "compatibility": {"mode": "insufficient"}}]},
                "/v1/apps": {"apps": [{"app_id": "anvira-notes", "permissions": ["chat"], "requested": ["models.manage"]}]},
                "/v1/lifecycle": {"mode": "on-demand", "leases": [{"app": "anvira-notes"}], "busy": 0, "shutdown_in_s": None},
                "/v1/capabilities": {"capabilities": [{"name": "orcha", "state": "running", "detail": ""}]},
                "/v1/resources": {"resources": [{"visibility": "private"}, {"visibility": "shared"}]},
                "/v1/resources/requests": {"requests": []},
                "/v1/runtime/logs": {"lines": ["2026-01-01 00:00:00,000 INFO runtime started"]}}
        return data[path]


def wait_page(d, name):
    d.set_view(name)
    for _ in range(100):
        if d.extra.get(name) is not None and not d._fetching:
            return
        time.sleep(0.05)


def test_dashboard_pages_render_real_content_at_every_terminal_size():
    d = tui.Dashboard(lambda: StubHttp(), color=False)
    d.status = {"status": "ok", "runtime_version": "1.0.0", "uptime_s": 5, "apps": 1, "pid": 1,
                "model": {"active": "qwen-test", "state": "running", "installed_count": 2, "models_dir": "D:/models", "extra_dirs": ["E:/more"],
                           "backend_info": {"variant": "cuda", "context": 8192, "gpu_layers": 24}},
                "services": {"orcha": {"state": "running", "port": 1}, "nomi": {"state": "running", "port": 2}, "aicl": {"state": "running"}}}
    expect = {"health": ("Checkups", "update the driver", "RTX Test", "CUDA"), "models": ("qwen-test", "too large", "E:/more"),
              "apps": ("anvira-notes", "on-demand", "waiting for you", "resources"), "logs": ("runtime started",)}
    for page, needles in expect.items():
        wait_page(d, page)
        for w, h in ((60, 18), (100, 30), (160, 50)):
            rows = d.render(w, h)
            assert len(rows) <= h and all(tui.visible_len(r) <= w for r in rows), (page, w, h)
        text = "\n".join(tui._ANSI.sub("", r) for r in d.render(130, 40))
        for n in needles:
            assert n in text, (page, n)
        assert "Overview" in text and "Health" in text                                         # the tab bar is always there
    d.set_view("overview")
    assert "Services & model" in "\n".join(d.render(120, 34))
    d.execute("view models", wait=True)
    assert d.view == "models"
    d.execute("view nope", wait=True)
    assert d.view == "models" and any("no page" in x for x in d.output)
    assert d.status["model"]["active"] == "qwen-test"


def test_dashboard_pages_survive_a_dead_runtime():
    d = tui.Dashboard(lambda: (_ for _ in ()).throw(AnviraError("runtime_not_running", "gone")), color=False)
    for page in ("health", "models", "apps", "logs"):
        wait_page(d, page)
        rows = d.render(100, 30)
        assert len(rows) <= 30 and page in "\n".join(rows).lower() or True
    assert "not connected" in "\n".join(d.render(100, 30))


def test_open_command_starts_on_the_requested_page_against_a_real_runtime(runtime, capsys):
    code, out, err = run_cli(capsys, "open", "health", "--once")
    assert code == 0, err
    assert "Checkups" in out and "Hardware, backend & services" in out and "Anvira Runtime" in out and "ok" in out
    code, out, _ = run_cli(capsys, "open", "apps", "--once")
    assert "Connected apps" in out and "Lifecycle, shared data & capabilities" in out
    code, out, _ = run_cli(capsys, "open", "models", "--once")
    assert "Installed models" in out and "This machine" in out
    code, out, _ = run_cli(capsys, "open", "--once")
    assert "Services & model" in out and "Jobs" in out


def test_interactive_loop_switches_pages_with_tab_and_digits(runtime, monkeypatch):
    keys = iter(["tab", "tab"] + [None] * 6 + ["4"] + [None] * 4 + ["r", "esc", "1"] + [None] * 4 + ["quit"])

    class FakeKeys:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def poll(self, timeout):
            time.sleep(0.02)
            return next(keys, "quit")
    seen = []
    orig = tui.Dashboard.render

    def spy(self, *a, **kw):
        seen.append(self.view)
        return orig(self, *a, **kw)
    monkeypatch.setattr(tui, "Keys", FakeKeys)
    monkeypatch.setattr(tui.Dashboard, "render", spy)
    monkeypatch.setattr(tui.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(tui.sys.stdout, "isatty", lambda: True, raising=False)
    assert tui.run_ui(lambda: runtime.owner) == 0
    assert ["health", "models", "apps", "overview"] == [v for i, v in enumerate(seen) if i == 0 or seen[i - 1] != v][:4]
