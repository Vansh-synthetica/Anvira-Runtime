"""Runtime installer / starter (stdlib only).

Install layout (one per OS user, shared by every Anvira app)::

    <runtime home>/
        install.json          marker: version, interpreter, source, installed_at
        venv/                 the runtime's private Python environment
        services/orcha|nomi|aicl/   the infrastructure source trees
        state/ config/ data/ logs/ models/ bin/   (created by the runtime on first start)

``install_runtime`` is idempotent and NEVER downloads or installs a model;
model installation is a separate, explicit operation.
It must only be called after the user agreed (see ``AnviraRuntime.connect``).
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable

from . import release
from .discovery import RuntimeInfo, pid_alive, probe, read_json, read_token
from .errors import AnviraError, RuntimeNotInstalled, RuntimeStartFailed
from .paths import runtime_dirs

SERVICE_DIRS = {"Orcha": "orcha", "nomi": "nomi", "AICL": "aicl"}
_SKIP_DIRS = {".git", ".venv", ".build-venv", "venv", "__pycache__", ".pytest_cache", "build", "dist", "vendor",
              "benchmarks", "node_modules", "target", "tests", "docs", "docker", "core-cpp", "native-cpp", "_legacy_v0",
              ".mypy_cache", ".ruff_cache"}
_SKIP_SUFFIXES = (".pyc", ".pyo", ".spec", ".log")
Status = Callable[[str], None]


def platform_info() -> dict[str, str]:
    return {"os": {"win32": "windows", "darwin": "macos"}.get(sys.platform, sys.platform),
            "arch": platform.machine().lower(), "python": platform.python_version()}


def _ignore(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n in _SKIP_DIRS or n.endswith(_SKIP_SUFFIXES) or n.endswith(".egg-info")}


def find_source(source: str | None, env: dict[str, str] | None = None) -> str | Path:
    """Resolve the runtime bundle: explicit arg, ``ANVIRA_RUNTIME_SOURCE``, or the enclosing repo checkout."""
    src = source or (env or os.environ).get("ANVIRA_RUNTIME_SOURCE")
    if src:
        return src if src.startswith(("http://", "https://")) else Path(src).expanduser()
    repo = Path(__file__).resolve().parents[3]
    if (repo / "runtime" / "pyproject.toml").is_file():
        return repo
    raise RuntimeNotInstalled(
        "no_install_source", "No Anvira Runtime bundle was provided.",
        hint="Pass source=<bundle.zip | URL | repo dir> or set ANVIRA_RUNTIME_SOURCE.")


def _materialize(source: str | Path, work: Path, say: Status) -> Path:
    """Return a directory containing runtime/, sdk/python/, Orcha/, nomi/, AICL/."""
    if isinstance(source, str):  # URL — the caller obtained user consent to download this
        say(f"Downloading runtime bundle from {source} ...")
        dest = work / "bundle.zip"
        with urllib.request.urlopen(source, timeout=60) as resp, open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        source = dest
    if source.is_file() and source.suffix.lower() == ".zip":
        say("Unpacking runtime bundle...")
        out = work / "bundle"
        with zipfile.ZipFile(source) as z:
            z.extractall(out)
        roots = [p for p in out.iterdir() if p.is_dir()]
        return roots[0] if len(roots) == 1 and not (out / "runtime").exists() else out
    if source.is_dir():
        return source
    raise RuntimeNotInstalled("bad_install_source", f"Install source not found or unsupported: {source}")


def bundle_version(root: Path) -> str:
    m = re.search(r'RUNTIME_VERSION\s*=\s*"([^"]+)"', (root / "runtime" / "anvira_runtime" / "version.py").read_text(encoding="utf-8"))
    return m.group(1) if m else "0.0.0"


def _pip(venv_python: Path, args: list[str], say: Status) -> None:
    proc = subprocess.run([str(venv_python), "-m", "pip", "install", "--disable-pip-version-check", *args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise AnviraError("install_failed", "pip failed while installing the runtime.",
                          hint=(proc.stderr or proc.stdout)[-600:])


def install_runtime(source: str | None = None, env: dict[str, str] | None = None, on_status: Status | None = None,
                    force: bool = False, python: str | None = None) -> dict[str, Any]:
    """Install or update the shared runtime. Returns the ``install.json`` record."""
    say = on_status or (lambda _m: None)
    dirs = runtime_dirs(env)
    home = dirs["home"]
    plat = platform_info()
    if isinstance(source, (str, Path)) and not str(source).startswith(("http://", "https://"))             and Path(source).is_file() and release.is_portable_zip(Path(source)):
        return release.install_package(source, None, env, say, stop=lambda: stop_runtime(env) if probe(env).running else None)
    say(f"Detected {plat['os']} / {plat['arch']} (Python {plat['python']})")
    existing = read_json(home / "install.json")
    src = find_source(source, env)
    work = Path(tempfile.mkdtemp(prefix="anvira-install-"))
    try:
        root = _materialize(src, work, say)
        new_version = bundle_version(root)
        entry = Path(existing.get("entry", "")) if existing.get("entry") else None
        if existing and not force and entry and entry.exists() and _semver(existing.get("version", "0")) >= _semver(new_version):
            say(f"Anvira Runtime {existing.get('version')} is already installed; nothing to do.")
            return existing
        # An update must not run over a live runtime, and no open app may revive it halfway through.
        home.mkdir(parents=True, exist_ok=True)
        release._lock(home, True)
        live = probe(env)
        if live.running:
            say("Stopping the running runtime for the update...")
            stop_runtime(env)
        home.mkdir(parents=True, exist_ok=True)
        services = home / "services"
        say("Installing ORCHA, Nomi and AICL...")
        staging = home / f".services-new-{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging)
        for src_name, dst_name in SERVICE_DIRS.items():
            if (root / src_name).is_dir():
                shutil.copytree(root / src_name, staging / dst_name, ignore=_ignore)
        missing = [n for n in SERVICE_DIRS.values() if not (staging / n).is_dir()]
        if missing:
            shutil.rmtree(staging, ignore_errors=True)
            raise AnviraError("bad_bundle", f"The bundle is missing: {', '.join(missing)}")
        if services.exists():
            shutil.rmtree(services)
        staging.rename(services)

        venv = home / "venv"
        if not venv.exists():
            say("Creating the runtime's Python environment...")
            args = [python or sys.executable, "-m", "venv", str(venv)]
            if (env or os.environ).get("ANVIRA_INSTALL_SYSTEM_SITE") == "1":
                args.append("--system-site-packages")
            proc = subprocess.run(args, capture_output=True, text=True)
            if proc.returncode != 0:
                raise AnviraError("install_failed", "Could not create a virtual environment.", hint=proc.stderr[-400:])
        vpy = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        say("Installing runtime packages (this needs an internet connection the first time)...")
        pip_args = ["--no-build-isolation"] if (env or os.environ).get("ANVIRA_INSTALL_OFFLINE_BUILD") == "1" else []
        _pip(vpy, [*pip_args, f"{root / 'runtime'}[all]", str(root / "sdk" / "python")], say)

        record = {"version": new_version, "installed_at": time.time(), "entry": str(vpy), "os": plat["os"],
                  "arch": plat["arch"], "services_dir": str(services),
                  "source": str(src) if isinstance(src, str) else str(src)}
        (home / "install.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        say(f"Anvira Runtime {new_version} installed at {home}")
        return record
    finally:
        release._lock(home, False)
        shutil.rmtree(work, ignore_errors=True)


def _semver(v: str) -> tuple[int, ...]:
    parts = [int(p) if p.isdigit() else 0 for p in v.lstrip("vV").split("-")[0].split(".")[:3]]
    return tuple(parts + [0] * (3 - len(parts)))


def runtime_python(env: dict[str, str] | None = None) -> str | None:
    """Interpreter that runs the runtime: the install marker's, else the current one if it has the runtime."""
    home = runtime_dirs(env)["home"]
    entry = read_json(home / "install.json").get("entry")
    if entry:
        p = Path(entry) if Path(entry).is_absolute() else home / entry       # a portable install records it relative
        if p.exists():
            return str(p)
    try:
        import importlib.util
        if importlib.util.find_spec("anvira_runtime"):
            return sys.executable
    except (ImportError, ValueError):
        pass
    return None


def start_runtime(env: dict[str, str] | None = None, on_status: Status | None = None, timeout: float = 90.0,
                  extra_args: list[str] | None = None, auto_stop: bool = False,
                  idle_grace_s: float | None = None) -> RuntimeInfo:
    """Start the runtime daemon (detached) and wait until it answers health checks.

    ``auto_stop=True`` starts it *on demand*: it shuts itself down ``idle_grace_s`` after the last app closed
    (apps use this). The default is a persistent runtime (``anvira runtime start``).
    """
    say = on_status or (lambda _m: None)
    release.wait_for_update(env, 240.0, say)
    extra_args = [*(extra_args or []), *(["--auto-stop"] if auto_stop else []),
                  *(["--idle-grace", str(idle_grace_s)] if idle_grace_s is not None else [])]
    info = probe(env)
    if info.running and info.ready:
        return info
    if info.running:  # another process is already starting it
        return wait_ready(env, timeout, say)
    py = runtime_python(env)
    if not py:
        raise RuntimeNotInstalled("runtime_not_installed", "Anvira Runtime is not installed.",
                                  hint="Install it first (see INSTALLATION.md).")
    dirs = runtime_dirs(env)
    dirs["logs"].mkdir(parents=True, exist_ok=True)
    child_env = {**os.environ, **(env or {}), "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        # NOT DETACHED_PROCESS: a console-less process makes every child (the venv python launcher, ORCHA, Nomi...)
        # open its own visible console window. CREATE_NO_WINDOW gives the daemon a hidden console its children inherit.
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    boot_log = dirs["logs"] / "daemon-boot.log"
    with open(boot_log, "ab") as log:
        proc = subprocess.Popen([py, "-m", "anvira_runtime", "daemon", *(extra_args or [])], env=child_env,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **kwargs)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = boot_log.read_text(encoding="utf-8", errors="replace")[-800:] if boot_log.exists() else ""
            raise RuntimeStartFailed("runtime_start_failed", "Runtime failed to start.",
                                     hint="Run `anvira doctor` for diagnostics.", details={"log_tail": tail})
        info = probe(env)
        if info.running and info.ready:
            say("Runtime is up.")
            return info
        time.sleep(0.3)
    raise RuntimeStartFailed("runtime_start_failed", f"Runtime did not become healthy within {timeout:.0f}s.",
                             hint="Run `anvira doctor` for diagnostics.")


def wait_ready(env: dict[str, str] | None, timeout: float, say: Status) -> RuntimeInfo:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = probe(env)
        if info.running and info.ready:
            return info
        time.sleep(0.3)
    raise RuntimeStartFailed("runtime_start_failed", f"Runtime did not become ready within {timeout:.0f}s.",
                             hint="Run `anvira doctor` for diagnostics.")


def stop_runtime(env: dict[str, str] | None = None, timeout: float = 25.0) -> bool:
    """Ask the runtime to shut down gracefully; force-kill as a last resort. True if it is stopped."""
    info = probe(env)
    if not info.running and not (info.pid and pid_alive(info.pid)):
        return True
    token = read_token("owner", env)
    if info.running and token:
        try:
            req = urllib.request.Request(f"{info.base_url}/v1/runtime/stop", method="POST",
                                         headers={"Authorization": f"Bearer {token}"})
            urllib.request.urlopen(req, timeout=5).read()
        except Exception:  # noqa: BLE001 - fall through to waiting/killing
            pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_alive(info.pid):
            return True
        time.sleep(0.2)
    if info.pid and pid_alive(info.pid):
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(info.pid), "/T", "/F"], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            import signal
            os.kill(info.pid, signal.SIGKILL)
        time.sleep(0.5)
    return not pid_alive(info.pid)
