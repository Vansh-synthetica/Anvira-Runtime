"""``anvira doctor``: find what is wrong and say how to fix it.

Works with the runtime stopped (offline checks on files, dependencies, ports,
hardware) and adds live checks (services, models, versions) when a status
snapshot from a running runtime is supplied.
"""
from __future__ import annotations

import importlib.util
import os
import stat
import sys
from typing import Any

from anvira_client.discovery import probe

from ..config.paths import RuntimeLayout
from ..config.settings import RuntimeConfig
from ..core.bus import native_available
from ..hardware import detect_hardware, disk_info
from ..models import backend as llama
from ..models.compat import assess
from ..process import infra
from ..process.supervisor import port_in_use
from ..version import API_VERSION, RUNTIME_VERSION

ORDER = {"ok": 0, "info": 0, "warn": 1, "fail": 2}


def _c(cid: str, title: str, status: str, message: str, fix: str | None = None, **data: Any) -> dict[str, Any]:
    out = {"id": cid, "title": title, "status": status, "message": message}
    if fix:
        out["fix"] = fix
    if data:
        out["data"] = data
    return out


def run_checks(layout: RuntimeLayout, config: RuntimeConfig, status: dict[str, Any] | None = None,
               env: dict[str, str] | None = None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    add = checks.append

    # -- environment ------------------------------------------------------------
    py_ok = sys.version_info >= (3, 10)
    add(_c("python", "Python interpreter", "ok" if py_ok else "fail",
           f"Python {sys.version.split()[0]} at {sys.executable}",
           None if py_ok else "Anvira Runtime needs Python 3.10 or newer."))

    # -- installation / directories -------------------------------------------------
    marker = layout.install_file
    info = probe(env)
    if info.installed or marker.exists():
        add(_c("install", "Runtime installation", "ok",
               f"Installed (marker: {info.install.get('version', 'unknown')}) at {layout.home}"))
    else:
        add(_c("install", "Runtime installation", "warn",
               "No install marker found; running from a source checkout or not installed.",
               "Install with `anvira runtime install` (see INSTALLATION.md)."))
    bad_dirs = []
    for d in layout.dirs():
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe_file = d / f".doctor-{os.getpid()}"
            probe_file.write_text("ok")
            probe_file.unlink()
        except OSError as exc:
            bad_dirs.append(f"{d} ({exc})")
    add(_c("directories", "Runtime directories", "ok" if not bad_dirs else "fail",
           "All runtime directories are writable." if not bad_dirs else "Cannot write to: " + "; ".join(bad_dirs),
           None if not bad_dirs else "Fix permissions or set ANVIRA_RUNTIME_HOME to a writable location."))

    # -- configuration ---------------------------------------------------------------
    add(_c("config", "Configuration", "ok" if not config.load_error else "fail",
           f"Loaded {layout.config_file}" if not config.load_error else f"Broken configuration: {config.load_error}",
           None if not config.load_error else f"Fix or delete {layout.config_file}; defaults are being used meanwhile."))

    # -- storage ----------------------------------------------------------------------
    models_dir = config.models_dir()
    disk = disk_info(models_dir)
    free = disk.get("free_gib", 0)
    add(_c("storage", "Model storage", "fail" if free < 1 else "warn" if free < 8 else "ok",
           f"{free} GiB free at {models_dir}",
           "Free up disk space or move models: `anvira model dir <path> --move`." if free < 8 else None,
           free_gib=free))

    # -- dependencies -------------------------------------------------------------------
    for name, mods in (("orcha", infra.ORCHA_MODULES), ("nomi", infra.NOMI_MODULES)):
        loc = infra.locate_service(name, layout, config)
        if loc.mode == "missing":
            add(_c(f"{name}_install", f"{name.upper() if name == 'orcha' else name.capitalize()} installation", "fail",
                   loc.detail))
            continue
        add(_c(f"{name}_install", f"{name.upper() if name == 'orcha' else name.capitalize()} installation", "ok",
               f"{loc.mode}: {loc.path}"))
        if loc.mode == "source" and infra.python_for_services(config) == sys.executable:
            missing = infra.missing_modules(mods)
            add(_c(f"{name}_deps", f"{name.upper() if name == 'orcha' else name.capitalize()} Python dependencies",
                   "fail" if missing else "ok",
                   f"Missing: {', '.join(missing)}" if missing else "All required packages are importable.",
                   f"pip install \"anvira-runtime[{name}]\"" if missing else None))
    aicl_root = infra.locate_aicl(layout)
    if aicl_root is None:
        add(_c("aicl", "AICL", "fail", "The new AICL package (aicl/bin/codec_api.py) was not found.",
               "Set ANVIRA_AICL_DIR to the folder that contains the `aicl` package."))
    else:
        add(_c("aicl", "AICL", "ok",
               f"Found at {aicl_root}; native Rust core: {'available' if native_available() else 'not built (in-process Python codec is used)'}"))

    # -- ports / process ------------------------------------------------------------------
    port = config.get("api.port")
    if info.running:
        add(_c("runtime_process", "Runtime process", "ok",
               f"Running: {info.runtime_version} (API v{info.api_version}) on port {info.port}, pid {info.pid}"))
        if info.api_version != API_VERSION:
            add(_c("version", "Version compatibility", "fail",
                   f"Running runtime speaks API v{info.api_version}, this CLI expects v{API_VERSION}.",
                   "Run `anvira runtime update` (or use the matching CLI)."))
        else:
            add(_c("version", "Version compatibility", "ok",
                   f"Runtime {info.runtime_version} / API v{info.api_version} (CLI {RUNTIME_VERSION})"))
    else:
        if info.stale_discovery:
            add(_c("runtime_process", "Runtime process", "warn", info.error or "Stale discovery file (runtime died).",
                   "Run `anvira runtime start` (the stale file is replaced)."))
        else:
            add(_c("runtime_process", "Runtime process", "warn", "The runtime is not running.",
                   "Start it with `anvira runtime start`."))
        if port and port_in_use(port):
            add(_c("port", "API port", "fail", f"Port {port} is in use by another program.",
                   "Free the port, or `anvira config set api.port 0` to pick a free one automatically."))
        else:
            add(_c("port", "API port", "ok", f"Port {port or 'auto'} is available."))

    # -- hardware -------------------------------------------------------------------------------
    hw = detect_hardware(models_dir, use_cache=False)
    gpu = hw["gpu"]
    add(_c("hardware", "Hardware", "info",
           f"{hw['cpu']['model']} | {hw['ram']['total_gib']} GiB RAM | "
           + (f"{gpu['name']} ({round((gpu['vram_total_mib'] or 0) / 1024, 1)} GiB VRAM, {gpu['backend']})" if gpu["name"] else "no GPU")
           , data=hw))
    binary, variant, searched = llama.find_llama_server(layout, config, prefer_cuda=gpu["backend"] == "cuda")

    # -- live checks (need a running runtime) --------------------------------------------------------
    if status:
        for name in ("orcha", "nomi"):
            svc = status["services"].get(name, {})
            state = svc.get("state")
            label = name.upper() if name == "orcha" else name.capitalize()
            if state == "running":
                add(_c(name, label, "ok", f"Running on port {svc.get('port')} (restarts: {svc.get('restarts', 0)})"))
            elif state == "disabled":
                add(_c(name, label, "info", "Disabled in configuration."))
            else:
                add(_c(name, label, "fail", f"{label} is {state}: {svc.get('last_error') or 'unavailable'}",
                       f"See `anvira runtime logs --service {name}`; then `anvira runtime restart`."))
        aicl = status["services"].get("aicl", {})
        add(_c("aicl_bus", "AICL bus", "ok" if aicl.get("state") == "running" else "fail",
               "In-process bus running." if aicl.get("state") == "running" else f"Unavailable: {aicl.get('last_error')}"))
        model = status.get("model", {})
        installed = model.get("installed_count", 0)
        if not installed:
            add(_c("models", "Models", "warn", "No model is installed.",
                   "Use `anvira model list` and `anvira model install <id>`."))
        elif not model.get("active"):
            add(_c("models", "Models", "warn", f"{installed} model(s) installed but none is selected.",
                   "Select one with `anvira model use <id>`."))
        else:
            st = model.get("state")
            rec = model.get("model") or {}
            level = "ok" if st in ("running", "remote") else ("warn" if st == "loading" else "fail")
            add(_c("models", "Active model", level, f"{model['active']}: {st}",
                   None if level == "ok" else f"See `anvira runtime logs --service model:{model['active']}`."))
            if rec.get("kind") == "local":
                a = assess({**rec}, hw)
                if a["can_run"] is False:
                    add(_c("model_hardware", "Model vs hardware", "warn", "; ".join(a["reasons"]),
                           "Choose a smaller model: `anvira model recommend`."))
                if binary is None and st not in ("running",):
                    add(_c("backend", "Inference backend", "fail", "llama-server was not found.",
                           "Install llama.cpp's llama-server and run `anvira config set models.llama_server_path <path>`."))
    else:
        if binary is None:
            add(_c("backend", "Inference backend", "warn",
                   "llama-server (llama.cpp) was not found; local models cannot be started until it is.",
                   "Install llama-server and `anvira config set models.llama_server_path <path>` (see INSTALLATION.md).",
                   searched=searched[:8]))
        else:
            add(_c("backend", "Inference backend", "ok", f"llama-server ({variant}) at {binary}"))

    # -- secrets file permissions (POSIX) -----------------------------------------------------------------
    if os.name == "posix" and layout.secrets_file.exists():
        mode = stat.S_IMODE(layout.secrets_file.stat().st_mode)
        if mode & 0o077:
            add(_c("secrets", "Secret file permissions", "warn", f"{layout.secrets_file} is mode {oct(mode)}.",
                   f"chmod 600 {layout.secrets_file}"))

    worst = max((ORDER[c["status"]] for c in checks), default=0)
    return {"status": ["ok", "warn", "fail"][worst], "runtime_version": RUNTIME_VERSION, "checks": checks,
            "summary": {s: sum(1 for c in checks if c["status"] == s) for s in ("ok", "info", "warn", "fail")}}
