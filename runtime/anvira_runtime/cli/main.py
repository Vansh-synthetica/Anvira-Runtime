"""``anvira`` - terminal control for Anvira Runtime.

Exit codes: 0 ok | 1 failure (or degraded status) | 2 usage | 3 runtime not installed |
4 runtime not running/failed to start | 5 permission denied | 6 not found / no model |
8 `doctor` found failures.  Add ``--json`` to any command for machine-readable output.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from anvira_client import bootstrap, release
from anvira_client.discovery import RuntimeInfo, probe, read_token
from anvira_client.errors import (AnviraError, InstallDeclined, PermissionDenied, RuntimeNotInstalled,
                                  RuntimeNotRunning, RuntimeStartFailed)
from anvira_client.http import Http

from ..config.paths import RuntimeLayout, resolve_layout
from ..config.settings import ConfigError, RuntimeConfig, flat_keys
from ..diagnostics.doctor import run_checks
from ..core.errors import RuntimeApiError
from ..diagnostics.logs import log_path
from ..security.secrets import PERMISSIONS, redact
from ..version import API_VERSION, RUNTIME_VERSION
from .output import Printer, human_duration, human_size

OK, FAIL, USAGE, NOT_INSTALLED, NOT_RUNNING, DENIED, NOT_FOUND, DOCTOR_FAIL = 0, 1, 2, 3, 4, 5, 6, 8

DESCRIPTION = """\
Anvira Runtime - the shared local AI runtime (ORCHA + Nomi + AICL) behind the Anvira apps.
Manage the runtime, models, orchestration jobs and memory from the terminal."""

EPILOG = """\
examples:
  anvira status                     show runtime, services and the active model
  anvira doctor                     diagnose problems and print fixes
  anvira runtime start              start the runtime (installs nothing)
  anvira model list                 installed + available models
  anvira model install qwen3-8b     download a model (asks first)
  anvira model use qwen3-8b         make it the active model
  anvira model dir D:\\AI\\models    put models wherever you like
  anvira orcha run "summarise ..."  run an orchestration job
  anvira nomi search "preferences"  search memory

exit codes: 0 ok, 1 failure/degraded, 2 usage, 3 not installed, 4 not running,
            5 permission denied, 6 not found/no model, 8 doctor failures.
Every command accepts --json. Run `anvira <command> --help` for details."""


class Ctx:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out = Printer(json_mode=getattr(args, "json", False), quiet=getattr(args, "quiet", False))
        self.layout: RuntimeLayout = resolve_layout()
        self._config: RuntimeConfig | None = None
        self._lease: dict[str, Any] | None = None
        self._lease_http: Http | None = None

    @property
    def config(self) -> RuntimeConfig:
        if self._config is None:
            self._config = RuntimeConfig.load(self.layout)
        return self._config

    def http(self, start: bool = True) -> Http:
        """Authenticated client for the local runtime (auto-starts it for action commands)."""
        info = probe()
        if info.running and not info.ready:
            self.out.progress("Waiting for the runtime to finish starting...")
            info = bootstrap.wait_ready(None, 120.0, self.out.progress)
        if not info.running:
            if not start:
                raise RuntimeNotRunning("runtime_not_running", "Anvira Runtime is not running.",
                                        hint="Start it with `anvira runtime start`.")
            if not info.installed and bootstrap.runtime_python() is None:
                raise RuntimeNotInstalled("runtime_not_installed", "Anvira Runtime is not installed.",
                                          hint="Install it with `anvira runtime install` (see INSTALLATION.md).")
            self.out.progress("Starting runtime...")
            # started for this command: on-demand, so it does not linger after you stop using it
            info = bootstrap.start_runtime(on_status=None, auto_stop=True, idle_grace_s=120.0)
        token = read_token("owner")
        if not token:
            raise PermissionDenied("no_owner_token", "Cannot read the runtime's owner token.",
                                   hint="Run the CLI as the same OS user that runs the runtime.")
        http = Http(info.base_url or "", token, timeout=60.0)
        if start and self._lease is None:       # hold a lease while this command runs
            try:
                self._lease = http.json("POST", "/v1/leases", {"ttl_s": 60}, timeout=10)["lease"]
                self._lease_http = http
            except AnviraError:
                pass
        return http

    def release(self) -> None:
        if self._lease is not None and self._lease_http is not None:
            try:
                self._lease_http.json("DELETE", f"/v1/leases/{self._lease['id']}", timeout=5)
            except AnviraError:
                pass
            self._lease = None


# ----------------------------------------------------------------------------- helpers
def _exit_for(exc: AnviraError) -> int:
    if isinstance(exc, (RuntimeNotInstalled,)):
        return NOT_INSTALLED
    if isinstance(exc, (RuntimeNotRunning, RuntimeStartFailed)):
        return NOT_RUNNING
    if isinstance(exc, PermissionDenied):
        return DENIED
    if exc.code.endswith("_not_found") or exc.code in ("no_model", "model_not_installed", "model_missing", "not_found"):
        return NOT_FOUND
    return FAIL


def _confirm(ctx: Ctx, question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if ctx.out.json_mode or not sys.stdin.isatty():
        raise AnviraError("confirmation_required", f"{question} - confirmation needed.",
                          hint="Re-run with --yes to confirm non-interactively.")
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def _state_mark(out: Printer, state: str | None) -> str:
    if state in ("running", "ready", "ok", "remote"):
        return out.green(out.sym["up"]) + f" {state}"
    if state in ("starting", "loading", "degraded", "stopped", "none", "disabled"):
        return out.yellow(out.sym["up"]) + f" {state}"
    return out.red(out.sym["up"]) + f" {state}"


def _uptime(v: float | None) -> str:
    return human_duration(v)


# ------------------------------------------------------------------------------ status
def cmd_status(ctx: Ctx) -> int:
    out, info = ctx.out, probe()
    if not info.running:
        installed = info.installed or bootstrap.runtime_python() is not None
        data = {"installed": installed, "running": False, "home": str(ctx.layout.home), "stale_discovery": info.stale_discovery}
        code = NOT_RUNNING if installed else NOT_INSTALLED

        def render(_: Any) -> None:
            if installed:
                out.line(f"{out.yellow(out.sym['up'])} Anvira Runtime is not running.")
                out.line("  Start it with: anvira runtime start")
            else:
                out.line("Runtime unavailable: Anvira Runtime is not installed.")
                out.line("  Install it with: anvira runtime install")
        out.emit(data, render)
        return code
    st = ctx.http(start=False).json("GET", "/v1/status")

    def render(s: dict[str, Any]) -> None:
        head = f"Anvira Runtime {s['runtime_version']} (API v{s['api_version']})"
        out.line(f"{out.bold(head)}  {_state_mark(out, 'running' if s['status'] == 'ok' else s['status'])}"
                 f"   pid {s['pid']}   up {_uptime(s['uptime_s'])}")
        out.line("")
        out.line(out.bold("Services"))
        for name, label in (("orcha", "ORCHA"), ("nomi", "Nomi"), ("aicl", "AICL")):
            svc = s["services"].get(name, {})
            extra = []
            if svc.get("port"):
                extra.append(f"port {svc['port']}")
            if name == "aicl":
                extra.append("in-process bus" + (" + native core" if svc.get("native_core") else ""))
            if svc.get("restarts"):
                extra.append(f"{svc['restarts']} restarts")
            if svc.get("last_error") and svc.get("state") != "running":
                extra.append(out.red(str(svc["last_error"]).splitlines()[0][:80]))
            out.line(f"  {label:<6} {_state_mark(out, svc.get('state'))}  {' | '.join(extra)}")
        out.line("")
        m = s["model"]
        if m.get("active"):
            det = m.get("backend_info") or {}
            extra = f"  (ctx {det['context']}, GPU layers {det['gpu_layers']})" if det.get("context") else ""
            out.line(f"{out.bold('Model')}  {m['active']}  {_state_mark(out, m['state'])}{extra}")
        else:
            out.line(f"{out.bold('Model')}  none selected - `anvira model list` / `anvira model use <id>`")
        jobs = s.get("jobs", {})
        out.line(f"{out.bold('Jobs')}   " + (", ".join(f"{v} {k}" for k, v in sorted(jobs.items())) or "none"))
        lc = s.get("lifecycle") or {}
        if lc:
            mode = "on-demand (stops by itself when no app is open)" if lc["auto_stop"] else "persistent (runs until `anvira runtime stop`)"
            who = ", ".join(sorted({x["app"] for x in lc["leases"]})) or "none"
            tail = f"; stopping in {lc['shutdown_in_s']:.0f}s" if lc.get("shutdown_in_s") is not None else ""
            out.line(f"{out.bold('Mode')}   {mode}; apps open: {who}{tail}")
        out.line(f"{out.bold('Apps')}   {s['apps']} registered")
        out.line(f"{out.bold('Home')}   {s['paths']['home']}")
        if s["status"] != "ok":
            out.line("")
            out.line(out.yellow(f"Degraded: {', '.join(s['degraded'])}. Run `anvira doctor`."))
    out.emit(st, render)
    return OK if st["status"] == "ok" else FAIL


def cmd_version(ctx: Ctx) -> int:
    info = probe()
    data = {"cli": RUNTIME_VERSION, "api_version": API_VERSION,
            "runtime": {"running": info.running, "version": info.runtime_version, "api_version": info.api_version,
                        "capabilities": info.capabilities}}

    def render(d: dict[str, Any]) -> None:
        ctx.out.line(f"anvira {d['cli']} (runtime API v{d['api_version']})")
        r = d["runtime"]
        ctx.out.line(f"running runtime: {r['version']} (API v{r['api_version']})" if r["running"] else "running runtime: none")
    ctx.out.emit(data, render)
    return OK


def cmd_doctor(ctx: Ctx) -> int:
    out, info = ctx.out, probe()
    status = None
    if info.running and info.ready:
        try:
            status = ctx.http(start=False).json("GET", "/v1/status")
        except AnviraError:
            status = None
    report = run_checks(ctx.layout, ctx.config, status)

    def render(r: dict[str, Any]) -> None:
        out.line(out.bold(f"Anvira doctor - runtime {r['runtime_version']}"))
        out.line("")
        for c in r["checks"]:
            out.line(f"{out.status_mark(c['status'])} {c['title']}: {c['message'].splitlines()[0]}")
            if c.get("fix") and c["status"] in ("warn", "fail"):
                out.line(out.dim(f"    fix: {c['fix']}"))
        s = r["summary"]
        out.line("")
        out.line(f"{s['ok'] + s['info']} ok, {s['warn']} warnings, {s['fail']} failures")
    out.emit(report, render)
    return DOCTOR_FAIL if report["status"] == "fail" else OK


# ------------------------------------------------------------------------------ runtime
def cmd_runtime_start(ctx: Ctx) -> int:
    out, a = ctx.out, ctx.args
    info = probe()
    if info.running and info.ready:
        out.emit({"started": False, "already_running": True, "pid": info.pid, "port": info.port},
                 lambda d: out.line(f"Runtime already running (pid {d['pid']}, port {d['port']})."))
        return OK
    if a.foreground:
        from ..daemon import run_daemon
        return run_daemon(console_log=True, port_override=a.port)
    if not info.installed and bootstrap.runtime_python() is None:
        raise RuntimeNotInstalled("runtime_not_installed", "Anvira Runtime is not installed.",
                                  hint="Install it with `anvira runtime install`.")
    out.progress("Starting runtime...")
    try:
        info = bootstrap.start_runtime(extra_args=["--port", str(a.port)] if a.port is not None else None,
                                       auto_stop=a.auto_stop, idle_grace_s=a.idle_grace)
    except RuntimeStartFailed as exc:
        exc.hint = "Runtime failed to start. Run `anvira doctor` for diagnostics."
        raise
    st = Http(info.base_url or "", read_token("owner"), 30).json("GET", "/v1/status")

    def render(s: dict[str, Any]) -> None:
        out.line(f"Runtime started (pid {s['pid']}, port {info.port}).")
        if s["status"] != "ok":
            out.line(out.yellow(f"Degraded: {', '.join(s['degraded'])} - run `anvira doctor`."))
    out.emit({"started": True, **st}, lambda d: render(st))
    return OK if st["status"] == "ok" else FAIL


def cmd_runtime_stop(ctx: Ctx) -> int:
    was = probe()
    running = was.running or bool(was.pid)
    stopped = bootstrap.stop_runtime()
    ctx.out.emit({"stopped": stopped, "was_running": running},
                 lambda d: ctx.out.line("Runtime stopped." if d["was_running"] and d["stopped"]
                                        else "Runtime was not running." if not d["was_running"]
                                        else "Runtime did not stop; try again or kill the process."))
    return OK if stopped else FAIL


def cmd_runtime_restart(ctx: Ctx) -> int:
    ctx.out.progress("Stopping runtime...")
    bootstrap.stop_runtime()
    return cmd_runtime_start(ctx)


def cmd_runtime_logs(ctx: Ctx) -> int:
    a, out = ctx.args, ctx.out
    try:
        path = log_path(ctx.layout, a.service)
    except RuntimeApiError as exc:          # an unknown service name is a usage error, not a traceback
        raise AnviraError(exc.code, exc.message, exc.status, exc.hint) from None
    if not path.exists():
        out.emit({"service": a.service, "path": str(path), "exists": False, "lines": []},
                 lambda d: out.line(f"No log yet at {path}"))
        return OK
    from ..process.supervisor import tail_file
    lines = redact(tail_file(path, a.lines)).splitlines()
    if out.json_mode:
        out.emit({"service": a.service, "path": str(path), "exists": True, "lines": lines})
        return OK
    for ln in lines:
        print(ln)
    if a.follow:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(0, os.SEEK_END)
                while True:
                    chunk = fh.readline()
                    if chunk:
                        print(redact(chunk.rstrip("\n")), flush=True)
                    else:
                        time.sleep(0.4)
        except KeyboardInterrupt:
            return OK
    return OK


def cmd_runtime_info(ctx: Ctx) -> int:
    info = probe()
    data = {"paths": ctx.layout.as_dict(), "install": info.install, "running": info.running,
            "models_dir": str(ctx.config.models_dir()), "config_file": str(ctx.layout.config_file)}
    ctx.out.emit(data, lambda d: (ctx.out.kv(d["paths"].items()), ctx.out.line(""),
                                  ctx.out.kv([("install", json.dumps(d["install"]) if d["install"] else "(no install marker)"),
                                              ("models (downloads)", d["models_dir"])])))
    return OK


def _portable_root() -> Path | None:
    """The folder this CLI runs from when it is the self-contained package (python/ + install.json next to anvira.cmd)."""
    home = os.environ.get("ANVIRA_RUNTIME_HOME")
    if home and (Path(home) / "install.json").is_file() and (Path(home) / "python").is_dir():
        return Path(home)
    return None


def _autoregister_portable() -> None:
    """Unzipped it somewhere and ran `anvira`? Remember the folder so every app finds this runtime (no-op if already known)."""
    root = _portable_root()
    if root is None:
        return
    try:
        env = {k: v for k, v in os.environ.items() if k != "ANVIRA_RUNTIME_HOME"}
        if release.locate(env)["home"] != str(root.resolve()) and release.default_home(env).resolve() != root.resolve():
            release.register_location(root, env)
    except Exception:  # noqa: BLE001 - registering is a convenience; never block a command on it
        pass


def _user_path_edit(folder: str, add: bool) -> bool:
    """Add/remove ``folder`` in the *user* PATH (Windows registry). Returns True if something changed."""
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ | winreg.KEY_WRITE) as key:
        try:
            cur, kind = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            cur, kind = "", winreg.REG_EXPAND_SZ
        parts = [x for x in cur.split(";") if x]
        norm = [os.path.normcase(os.path.normpath(x)) for x in parts]
        me = os.path.normcase(os.path.normpath(folder))
        if add and me not in norm:
            parts.append(folder)
        elif not add and me in norm:
            parts = [x for x, n in zip(parts, norm) if n != me]
        else:
            return False
        winreg.SetValueEx(key, "Path", 0, kind, ";".join(parts))
    try:                                            # tell running programs (new terminals pick it up)
        import ctypes
        ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "Environment", 0x2, 3000, ctypes.byref(ctypes.c_ulong()))
    except Exception:  # noqa: BLE001
        pass
    return True


def cmd_path(ctx: Ctx) -> int:
    a, out = ctx.args, ctx.out
    folder = str(_portable_root() or Path(__file__).resolve().parents[3])
    if os.name != "nt":
        out.line(f"Add this to your shell profile:  export PATH=\"{folder}:$PATH\"")
        return OK
    if not _confirm(ctx, f"{'Add' if a.action == 'add' else 'Remove'} {folder} {'to' if a.action == 'add' else 'from'} your user PATH "
                    f"so you can type `anvira` in any new terminal?", a.yes):
        out.line("Cancelled; nothing was changed.")
        return FAIL
    changed = _user_path_edit(folder, a.action == "add")
    out.emit({"folder": folder, "changed": changed},
             lambda d: out.line(("Done. Open a NEW terminal and type:  anvira open" if a.action == "add" else "Removed from PATH.")
                                if d["changed"] else "Nothing to change - already " + ("on your PATH." if a.action == "add" else "absent.")))
    return OK


def _in_checkout() -> bool:
    return (Path(__file__).resolve().parents[3] / "runtime" / "pyproject.toml").is_file()


def _ask(ctx: Ctx, question: str, default: str = "") -> str:
    if ctx.out.json_mode or not sys.stdin.isatty():
        return default
    ans = input(f"{question}" + (f" [{default}]" if default else "") + " ").strip()
    return ans or default


def _progress_bar(ctx: Ctx):
    last = {"t": 0.0}

    def cb(name: str, done: int, total: int) -> None:
        now = time.monotonic()
        if now - last["t"] < 0.4 and done < total:
            return
        last["t"] = now
        pct = f" {done * 100 // total}%" if total else ""
        ctx.out.progress(f"  {name}  {done / 1e6:.0f} / {total / 1e6:.0f} MB{pct}")
    return cb


def cmd_runtime_locate(ctx: Ctx) -> int:
    loc = release.locate()
    ctx.out.emit(loc, lambda d: (ctx.out.kv([("found", "yes" if d["found"] else "no"), ("location", d["home"]), ("how", d["source"]),
                                             ("version", d["version"] or "-"), ("running", "yes" if d["running"] else "no"),
                                             ("GPU pack", "yes" if d["gpu_pack"] else "no"), ("default location", d["default_home"])]),
                                 None if d["found"] else ctx.out.line("Not installed. Run `anvira runtime install`.")))
    return OK if loc["found"] else NOT_INSTALLED


def cmd_runtime_register(ctx: Ctx) -> int:
    home = release.register_location(ctx.args.path)
    ctx.out.emit({"home": str(home)}, lambda d: ctx.out.line(f"Remembered {d['home']} - every Anvira app will use the runtime there."))
    return OK


def cmd_runtime_install_gpu(ctx: Ctx) -> int:
    a, out = ctx.args, ctx.out
    card = release.nvidia_gpu()
    if not card:
        raise AnviraError("no_nvidia_gpu", "No NVIDIA GPU was found; the GPU pack would not be used.")
    if not _confirm(ctx, f"Download the NVIDIA GPU pack for {card['name']} into {ctx.layout.home}?", a.yes):
        raise InstallDeclined("install_declined", "Cancelled; nothing was changed.")
    bootstrap.stop_runtime() if probe().running else None
    rec = release.install_gpu_pack(repo=a.repo, on_status=out.progress, on_progress=_progress_bar(ctx))
    out.emit(rec, lambda r: out.line("GPU pack installed. Models will use your GPU the next time they load."))
    return OK


def _install_from_github(ctx: Ctx) -> int:
    a, out = ctx.args, ctx.out
    loc = release.locate()
    if loc["found"] and not a.force:
        out.emit(loc, lambda d: out.line(f"Anvira Runtime {d['version']} is already installed at {d['home']}. Use --force to reinstall."))
        return OK
    dest = a.dir
    if not dest and not a.yes and not out.json_mode and sys.stdin.isatty():
        out.line(out.bold("Anvira Runtime is not installed on this PC."))
        out.line("  Already have it somewhere?  Type its folder.  Otherwise press Enter to download it.")
        existing = _ask(ctx, "Existing folder (Enter = download):")
        if existing:
            home = release.register_location(existing)
            out.emit({"home": str(home)}, lambda d: out.line(f"Using the runtime at {d['home']}."))
            return OK
        dest = _ask(ctx, "Install into", str(release.default_home())) or None
    if not _confirm(ctx, f"Download Anvira Runtime from GitHub into {dest or release.default_home()}? (no AI models are downloaded)", a.yes):
        raise InstallDeclined("install_declined", "Cancelled; nothing was changed.")

    def confirm_gpu(card: dict[str, Any]) -> bool:
        if a.gpu == "no" or not card.get("cuda_ok"):
            return False
        if a.gpu == "yes" or a.yes:
            return True
        return _confirm(ctx, f"NVIDIA GPU found ({card['name']}). Download the GPU acceleration pack ({card['size'] / 1e9:.1f} GB)?", False)
    was_running = probe().running
    rec = release.install_from_github(dest, repo=a.repo, gpu={"yes": True, "no": False}.get(a.gpu), confirm_gpu=confirm_gpu, force=a.force,
                                      on_status=out.progress, on_progress=_progress_bar(ctx),
                                      stop=lambda: bootstrap.stop_runtime() if probe().running else None)
    if was_running or a.start:
        bootstrap.start_runtime()
    out.emit(rec, lambda r: (out.line(f"Anvira Runtime {r.get('version')} is ready."),
                             out.line("  Open the dashboard:  anvira open") if not (was_running or a.start) else None,
                             out.line(out.yellow("  A GPU pack is available for your NVIDIA card:  anvira runtime install-gpu"))
                             if r.get("gpu_pack_available") and not r.get("gpu_pack") else None))
    return OK


def cmd_runtime_install(ctx: Ctx, update: bool = False) -> int:
    a, out = ctx.args, ctx.out
    if a.github or (a.source is None and not _in_checkout()):
        return _install_from_github(ctx)
    info = probe()
    if info.installed and not update and not a.force:
        out.emit({"installed": True, "version": info.install.get("version")},
                 lambda d: out.line(f"Anvira Runtime {d['version']} is already installed (shared by all Anvira apps). "
                                    f"Use `anvira runtime update` to upgrade."))
        return OK
    verb = "Update" if update else "Install"
    if not _confirm(ctx, f"{verb} Anvira Runtime into {ctx.layout.home}? (no models are downloaded)", a.yes):
        raise InstallDeclined("install_declined", "Cancelled; nothing was changed.")
    was_running = info.running
    rec = bootstrap.install_runtime(source=a.source, force=a.force or update, on_status=out.progress)
    if was_running or a.start:
        out.progress("Starting runtime...")
        bootstrap.start_runtime()
    out.emit(rec, lambda r: out.line(f"Anvira Runtime {r['version']} ready at {ctx.layout.home}. "
                                     f"Start it with `anvira runtime start`." if not (was_running or a.start)
                                     else f"Anvira Runtime {r['version']} installed and running."))
    return OK


# ------------------------------------------------------------------------------- models
def _model_row(out: Printer, m: dict[str, Any]) -> list[Any]:
    comp = (m.get("compatibility") or {})
    if m.get("active"):
        state = out.green("active")
    elif m.get("installed"):
        state = "installed"
    else:
        state = out.dim("available")
    fit = {"gpu": "GPU", "partial-gpu": "GPU+CPU", "cpu": "CPU", "remote": "remote", "insufficient": out.red("too large"),
           "unknown": "?"}.get(comp.get("mode"), "?")
    return [m["id"], m.get("name", ""), "cloud" if m.get("kind") == "provider" else "local",
            human_size(m.get("size_bytes")), state, fit]


def cmd_model_list(ctx: Ctx) -> int:
    a, out = ctx.args, ctx.out
    data = ctx.http().json("GET", "/v1/models", params={"installed": "true" if a.installed else None})
    models = data["models"]

    def render(d: dict[str, Any]) -> None:
        if not any(m.get("installed") for m in d["models"]):
            out.line(out.yellow("No model is installed.") + " Use `anvira model install <id>` (see the list below).")
            out.line("")
        out.table(["ID", "NAME", "TYPE", "SIZE", "STATE", "FITS"], [_model_row(out, m) for m in d["models"]])
    out.emit(data, render)
    return OK


def cmd_model_search(ctx: Ctx) -> int:
    a, out = ctx.args, ctx.out
    data = ctx.http().json("GET", "/v1/models/search", params={"q": a.query, "limit": a.limit})
    out.emit(data, lambda d: out.table(["REPO", "PARAMS", "DOWNLOADS", "GATED"], [
        [m["id"], f"{m['params_b']}B" if m.get("params_b") else "-", m.get("downloads") or "-", "yes" if m.get("gated") else ""]
        for m in d["models"]]))
    if not out.json_mode and data["models"]:
        out.line("")
        out.line("Install one with: anvira model install <owner/repo> [--file <name.gguf>]")
    return OK


def cmd_model_info(ctx: Ctx) -> int:
    out = ctx.out
    m = ctx.http().json("GET", f"/v1/models/{ctx.args.id}")

    def render(m: dict[str, Any]) -> None:
        out.kv([("id", m["id"]), ("name", m.get("name")), ("type", m.get("kind")), ("installed", m.get("installed")),
                ("active", m.get("active")), ("size", human_size(m.get("size_bytes"))),
                ("quantization", m.get("quantization")), ("path", m.get("path")), ("location", m.get("location")),
                ("source", json.dumps(m.get("source")) if m.get("source") else None),
                ("fits", (m.get("compatibility") or {}).get("mode"))])
        for r in (m.get("compatibility") or {}).get("reasons", []):
            out.line(f"  - {r}")
    out.emit(m, render)
    return OK


def cmd_model_compat(ctx: Ctx) -> int:
    out = ctx.out
    d = ctx.http().json("GET", f"/v1/models/{ctx.args.id}/compatibility")

    def render(d: dict[str, Any]) -> None:
        verdict = out.green("can run") if d["can_run"] else out.red("cannot run") if d["can_run"] is False else "unknown"
        out.line(f"{d['id']}: {verdict} ({d['mode']}), needs about {d.get('needs_mib')} MiB")
        for r in d["reasons"]:
            out.line(f"  - {r}")
        hw = d["hardware"]
        out.line(out.dim(f"  hardware: {hw['ram']['total_gib']} GiB RAM, GPU {hw['gpu']['name'] or 'none'} "
                         f"({hw['gpu']['backend']})"))
    out.emit(d, render)
    return OK if d["can_run"] is not False else FAIL


def cmd_model_recommend(ctx: Ctx) -> int:
    out = ctx.out
    d = ctx.http().json("GET", "/v1/models/recommended", params={"limit": ctx.args.limit})
    out.emit(d, lambda d: out.table(["ID", "NAME", "SIZE", "RUNS ON", "INSTALLED"], [
        [m["id"], m["name"], human_size(m.get("size_bytes")), m["compatibility"]["mode"], "yes" if m.get("installed") else "no"]
        for m in d["models"]]) if d["models"] else out.line("No catalog model fits this machine's memory."))
    return OK


def cmd_model_status(ctx: Ctx) -> int:
    out = ctx.out
    http = ctx.http()
    st = http.json("GET", "/v1/models/active")
    hw = http.json("GET", "/v1/hardware")

    def render(s: dict[str, Any]) -> None:
        if not s["active"]:
            out.line(out.yellow("No model is selected.") + (" No model is installed." if not s["installed_count"] else ""))
            out.line("Use `anvira model list` or `anvira model install ...`")
        else:
            out.line(f"Active model: {out.bold(s['active'])}  {_state_mark(out, s['state'])}")
            info = s.get("backend_info") or {}
            if info:
                out.kv([("backend", info.get("binary")), ("variant", info.get("variant")),
                        ("context", info.get("context")), ("gpu layers", info.get("gpu_layers")),
                        ("port", info.get("port"))], 2)
            b = s.get("backend") or {}
            if b.get("last_error"):
                out.line(out.red(f"  last error: {str(b['last_error']).splitlines()[0]}"))
            if b.get("log_tail"):
                out.line(out.dim("  log tail:\n    " + b["log_tail"].replace("\n", "\n    ")))
        out.line("")
        g = hw["gpu"]
        out.kv([("models dir", s["models_dir"]), ("extra dirs", ", ".join(s["extra_dirs"]) or "-"),
                ("hardware", f"{hw['cpu']['model']} | {hw['ram']['total_gib']} GiB RAM | "
                             f"{g['name'] or 'no GPU'} ({g['backend']})")])
    out.emit({**st, "hardware": hw}, lambda _: render(st))
    return OK


def cmd_model_use(ctx: Ctx) -> int:
    out = ctx.out
    out.progress(f"Selecting {ctx.args.id} (loading the model can take a while)...")
    st = ctx.http().json("POST", "/v1/models/select", {"id": ctx.args.id, "wait_s": ctx.args.wait})
    out.emit(st, lambda s: out.line(f"Active model: {s['active']}  {_state_mark(out, s['state'])}"
                                    + ("\nStill loading in the background; check `anvira model status`." if s["state"] == "loading" else "")))
    return OK


def cmd_model_unuse(ctx: Ctx) -> int:
    st = ctx.http().json("POST", "/v1/models/deselect")
    ctx.out.emit(st, lambda s: ctx.out.line("No active model."))
    return OK


def cmd_model_install(ctx: Ctx) -> int:
    a, out = ctx.args, ctx.out
    http = ctx.http()
    size, dest = None, a.dir or http.json("GET", "/v1/models/storage")["models_dir"]
    if not a.url:
        try:
            size = http.json("GET", f"/v1/models/{a.model}").get("size_bytes")
        except AnviraError:
            size = None
    what = f"~{human_size(size)}" if size else "an unknown size"
    if not _confirm(ctx, f"Download {a.model or a.url} ({what}) to {dest}?", a.yes):
        out.line("Cancelled; nothing was downloaded.")
        return OK
    body = {k: v for k, v in {"model": a.model, "file": a.file, "url": a.url, "dir": a.dir}.items() if v}
    job = http.json("POST", "/v1/models/install", body)["job"]
    if a.no_wait:
        out.emit({"job": job}, lambda d: out.line(f"Install started: {d['job']['id']} (track with `anvira job get {d['job']['id']}`)"))
        return OK
    try:
        while job["state"] in ("queued", "running"):
            p = job.get("progress") or {}
            if p.get("total") and not out.json_mode:
                sys.stderr.write(f"\r  {p.get('percent', 0):5.1f}%  {human_size(p['downloaded'])} / {human_size(p['total'])}"
                                 f"  {human_size(p.get('speed_bps'))}/s   ")
                sys.stderr.flush()
            time.sleep(0.5)
            job = http.json("GET", f"/v1/jobs/{job['id']}")["job"]
    except KeyboardInterrupt:
        http.json("POST", f"/v1/jobs/{job['id']}/cancel")
        out.err("\nCancelled. The partial download is kept and resumes if you run the install again.")
        return 130
    if not out.json_mode:
        sys.stderr.write("\n")
    if job["state"] != "completed":
        err = job.get("error") or {}
        raise AnviraError(err.get("code", job["state"]), err.get("message", "Install failed"), hint=err.get("hint"))
    out.emit({"job": job}, lambda d: out.line(f"Installed {d['job']['result']['model']['id']} at {d['job']['result']['model']['path']}\n"
                                              f"Use it with: anvira model use {d['job']['result']['model']['id']}"))
    return OK


def cmd_model_remove(ctx: Ctx) -> int:
    a, out = ctx.args, ctx.out
    m = ctx.http().json("GET", f"/v1/models/{a.id}")
    if a.delete_file and not _confirm(ctx, f"Delete the model file {m.get('path')}?", a.yes):
        out.line("Cancelled.")
        return OK
    res = ctx.http().json("POST", "/v1/models/remove", {"id": a.id, "delete_file": True if a.delete_file else None,
                                                        "confirm": bool(a.delete_file)})
    out.emit(res, lambda r: out.line(f"Removed {r['id']}" + (f" (deleted {len(r['deleted_files'])} file(s))" if r["deleted_files"]
                                                             else " (file left in place)") + (f"\n{r['note']}" if r.get("note") else "")))
    return OK


def cmd_model_dir(ctx: Ctx) -> int:
    a, out = ctx.args, ctx.out
    http = ctx.http()
    if not a.path:
        st = http.json("GET", "/v1/models/storage")
        out.emit(st, lambda s: (out.kv([("download dir", s["models_dir"]), ("extra dirs", ", ".join(s["extra_dirs"]) or "-"),
                                        ("free space", f"{s['disk'].get('free_gib')} GiB"),
                                        ("linked files", len(s["linked"]))])))
        return OK
    res = http.json("PUT", "/v1/models/storage", {"path": a.path, "move": a.move})
    out.emit(res, lambda r: (out.line(f"New models are stored in {r['models_dir']}"),
                             [out.line(f"  moved {m['target']}") for m in r["moved"]],
                             [out.line(out.red(f"  could not move {f['source']}: {f['error']}")) for f in r["failed"]]))
    return FAIL if res["failed"] else OK


def cmd_model_dirs(ctx: Ctx) -> int:
    a, out = ctx.args, ctx.out
    http = ctx.http()
    if a.action == "add":
        res = http.json("POST", "/v1/models/storage/dirs", {"path": a.path})
    elif a.action == "remove":
        res = http.json("DELETE", "/v1/models/storage/dirs", params={"path": a.path})
    else:
        res = {"extra_dirs": http.json("GET", "/v1/models/storage")["extra_dirs"]}
    out.emit(res, lambda r: out.line("\n".join(r["extra_dirs"]) or "(no extra model folders)"))
    return OK


def cmd_model_add(ctx: Ctx) -> int:
    a = ctx.args
    path = str(Path(a.file).resolve())
    if Path(path).is_dir():
        res = ctx.http().json("POST", "/v1/models/register", {"path": path})
        ctx.out.emit(res, lambda r: ctx.out.line(f"Scanning {r['path']} ({r['models_found']} model(s) found in extra folders); nothing was copied."))
        return OK
    if a.id:
        res = ctx.http().json("POST", "/v1/models/link", {"path": path, "id": a.id})
    else:
        res = ctx.http().json("POST", "/v1/models/register", {"path": path})["model"]
    ctx.out.emit(res, lambda m: ctx.out.line(f"Linked {m['id']} in place ({human_size(m['size_bytes'])}); the file was not copied."))
    return OK


def cmd_model_discover(ctx: Ctx) -> int:
    out = ctx.out
    d = ctx.http().json("GET", "/v1/models/discovered")

    def render(d: dict[str, Any]) -> None:
        if not d["enabled"]:
            return out.line("App discovery is off (anvira config set models.discover_apps true).")
        if not d["locations"]:
            return out.line("No Anvira app model folders found. Apps can announce theirs; you can also run "
                            "`anvira model add <file-or-folder>`.")
        out.table(["APP", "FOLDER", "MODELS"], [[x["app"], x["dir"], len(x["models"])] for x in d["locations"]])
    out.emit(d, render)
    return OK


def cmd_ui(ctx: Ctx) -> int:
    from .tui import run_ui
    a = ctx.args
    once = a.once or a.exec is not None

    def factory() -> Http:
        return ctx.http(start=not once)
    page = getattr(a, "page", None)
    return run_ui(factory, once=a.once, exec_cmd=a.exec, **({"view": page} if page else {}))


def cmd_model_provider(ctx: Ctx) -> int:
    a, out = ctx.args, ctx.out
    http = ctx.http()
    if a.action == "add":
        key = a.api_key or (os.environ.get(a.api_key_env, "") if a.api_key_env else "")
        if a.api_key_stdin:
            key = getpass.getpass("API key: ") if sys.stdin.isatty() else sys.stdin.readline().strip()
        rec = http.json("POST", "/v1/providers", {"base_url": a.base_url, "model": a.model, "label": a.label,
                                                  "api_key": key})
        out.emit(rec, lambda r: out.line(f"Added {r['id']} ({'key stored' if r['has_api_key'] else 'no key'}). "
                                         f"Use it: anvira model use {r['id']}"))
    elif a.action == "remove":
        res = http.json("DELETE", f"/v1/providers/{a.id.removeprefix('cloud:')}")
        out.emit(res, lambda r: out.line(f"Removed {r['id']}"))
    else:
        res = http.json("GET", "/v1/providers")
        out.emit(res, lambda r: out.table(["ID", "MODEL", "BASE URL", "KEY"], [
            [p["id"], p["model"], p["base_url"], p.get("key_hint") or ("set" if p["has_api_key"] else "none")] for p in r["providers"]]))
    return OK


def cmd_hardware(ctx: Ctx) -> int:
    out = ctx.out
    hw = ctx.http().json("GET", "/v1/hardware", params={"refresh": "true"})

    def render(h: dict[str, Any]) -> None:
        g = h["gpu"]
        out.kv([("platform", f"{h['platform']['os']} {h['platform']['arch']} ({h['platform']['release']})"),
                ("cpu", f"{h['cpu']['model']} ({h['cpu']['logical_cores']} logical cores)"),
                ("ram", f"{h['ram']['total_gib']} GiB total, {h['ram']['free_gib']} GiB free"),
                ("gpu", f"{g['name']} - {round((g['vram_total_mib'] or 0) / 1024, 1)} GiB VRAM" if g["name"] else "none detected"),
                ("cuda", g.get("cuda_version") or "-"), ("accelerators", ", ".join(h["acceleration_backends"])),
                ("models disk", f"{h['storage'].get('models', {}).get('free_gib', '?')} GiB free")])
    out.emit(hw, render)
    return OK


# ------------------------------------------------------------------------------- orcha / jobs
def _render_job(out: Printer, job: dict[str, Any]) -> None:
    mark = {"completed": out.green(out.sym["ok"]), "failed": out.red(out.sym["fail"]),
            "cancelled": out.yellow(out.sym["warn"])}.get(job["state"], out.dim(out.sym["info"]))
    out.line(f"{mark} {job['id']}  {job['kind']}  {job['state']}")
    res = job.get("result") or {}
    if job["state"] == "completed" and res.get("answer") is not None:
        out.line("")
        out.line(res["answer"])
        out.line("")
        out.line(out.dim(f"confidence {res.get('confidence')} | iterations {res.get('iterations')} | "
                         f"contributors {', '.join(res.get('contributors') or []) or '-'} | {res.get('latency_s')}s"))
    elif job["state"] == "completed" and job["kind"] == "model.install":
        out.line(f"installed {res['model']['id']}")
    if job.get("error"):
        out.line(out.red(f"{job['error'].get('code')}: {job['error'].get('message')}"))
        if job["error"].get("hint"):
            out.line(out.dim(f"hint: {job['error']['hint']}"))


def cmd_orcha_status(ctx: Ctx) -> int:
    out = ctx.out
    d = ctx.http().json("GET", "/v1/orcha/status")

    def render(d: dict[str, Any]) -> None:
        svc = d["service"]
        out.line(f"ORCHA  {_state_mark(out, svc.get('state'))}   port {svc.get('port')}   restarts {svc.get('restarts', 0)}")
        eng = d.get("engine") or {}
        if eng:
            out.kv([("experts", ", ".join(eng.get("experts") or []) or "-"), ("source", eng.get("source")),
                    ("synthesizer", eng.get("synthesizer")), ("active model", d.get("active_model") or "none")], 2)
        if d.get("engine_error"):
            out.line(out.red(f"  {d['engine_error']['message']}"))
        out.line(f"  jobs: " + (", ".join(f"{v} {k}" for k, v in d["jobs"].items()) or "none"))
    out.emit(d, render)
    return OK if d["service"].get("state") == "running" else FAIL


def cmd_orcha_run(ctx: Ctx) -> int:
    a, out = ctx.args, ctx.out
    body: dict[str, Any] = {"task": a.task, "graph": a.graph}
    if a.reasoning:
        body["reasoning"] = a.reasoning
    if a.workspace:
        body["workspace_roots"] = [str(Path(p).resolve()) for p in a.workspace]
        body["allow_tools"] = True
    http = ctx.http()
    job = http.json("POST", "/v1/orcha/run", body, timeout=60)["job"]
    if a.no_wait:
        out.emit({"job": job}, lambda d: out.line(f"Started {d['job']['id']}  (anvira orcha jobs / anvira job get {d['job']['id']})"))
        return OK
    try:
        while job["state"] in ("queued", "running"):
            time.sleep(0.5)
            job = http.json("GET", f"/v1/jobs/{job['id']}")["job"]
    except KeyboardInterrupt:
        http.json("POST", f"/v1/jobs/{job['id']}/cancel")
        out.err("\nCancelled.")
        return 130
    out.emit({"job": job}, lambda d: _render_job(out, d["job"]))
    return OK if job["state"] == "completed" else FAIL


def cmd_orcha_jobs(ctx: Ctx) -> int:
    a, out = ctx.args, ctx.out
    d = ctx.http().json("GET", "/v1/jobs", params={"state": a.state, "limit": a.limit})
    out.emit(d, lambda d: out.table(["JOB", "KIND", "APP", "STATE", "AGE"], [
        [j["id"], j["kind"], j["app"] or "owner", j["state"], human_duration(time.time() - j["created_at"])]
        for j in d["jobs"]]) if d["jobs"] else out.line("No jobs."))
    return OK


def cmd_job_get(ctx: Ctx) -> int:
    job = ctx.http().json("GET", f"/v1/jobs/{ctx.args.id}")["job"]
    ctx.out.emit({"job": job}, lambda d: _render_job(ctx.out, d["job"]))
    return OK


def cmd_orcha_cancel(ctx: Ctx) -> int:
    job = ctx.http().json("POST", f"/v1/jobs/{ctx.args.id}/cancel")["job"]
    ctx.out.emit({"job": job}, lambda d: ctx.out.line(f"{d['job']['id']}: {d['job']['state']}"))
    return OK


# ------------------------------------------------------------------------------------ nomi
def _render_mem(out: Printer, m: dict[str, Any]) -> None:
    score = f"  score {m['score']}" if m.get("score") is not None else ""
    out.line(f"{out.bold(m['title'])}  {out.dim(m['id'])}{score}")
    out.line(f"  {m['content'][:300]}")
    out.line(out.dim(f"  app={m.get('app')} scope={m.get('scope')} type={m.get('type')} tags={','.join(m.get('tags') or []) or '-'}"))


def cmd_nomi_status(ctx: Ctx) -> int:
    out = ctx.out
    d = ctx.http().json("GET", "/v1/nomi/status")

    def render(d: dict[str, Any]) -> None:
        svc = d["service"]
        out.line(f"Nomi  {_state_mark(out, svc.get('state'))}   port {svc.get('port')}   restarts {svc.get('restarts', 0)}")
        if d.get("memory"):
            out.line(f"  memory API reachable ({d['memory'].get('service')}, {d['memory'].get('environment')})")
        if d.get("memory_error"):
            out.line(out.red(f"  {d['memory_error']['message']}"))
    out.emit(d, render)
    return OK if d["service"].get("state") == "running" else FAIL


def cmd_nomi_search(ctx: Ctx) -> int:
    a, out = ctx.args, ctx.out
    d = ctx.http().json("GET", "/v1/memory/search", params={"q": a.query, "limit": a.limit, "scope": a.scope, "app": a.app})
    out.emit(d, lambda d: ([_render_mem(out, m) for m in d["items"]] if d["items"] else out.line("No matching memories.")))
    return OK


def cmd_nomi_inspect(ctx: Ctx) -> int:
    m = ctx.http().json("GET", f"/v1/memory/{ctx.args.id}")
    ctx.out.emit(m, lambda m: (_render_mem(ctx.out, m), ctx.out.line(ctx.out.dim(f"  created {m.get('created_at')}  importance {m.get('importance')}"))))
    return OK


def cmd_nomi_store(ctx: Ctx) -> int:
    a = ctx.args
    m = ctx.http().json("POST", "/v1/memory", {"content": a.text, "title": a.title, "tags": a.tag or None,
                                               "scope": a.scope, "app": a.app, "type": a.type})
    ctx.out.emit(m, lambda m: ctx.out.line(f"Stored {m['id']} (app={m['app']}, scope={m['scope']})"))
    return OK


def cmd_nomi_delete(ctx: Ctx) -> int:
    r = ctx.http().json("DELETE", f"/v1/memory/{ctx.args.id}")
    ctx.out.emit(r, lambda r: ctx.out.line(f"Deleted {r['id']}"))
    return OK


# ------------------------------------------------------------------- shared context / workspaces
def _render_res(out: Printer, r: dict[str, Any]) -> None:
    vis = {"private": out.dim("private"), "shared": out.yellow("shared"), "global": out.red("global")}.get(r["visibility"], r["visibility"])
    out.line(f"{out.bold(r['title'])}  {out.dim(r['id'])}  [{r['type']}]  owner={r['owner']}  {vis}"
             + (f"  ws={r['workspace']}" if r.get("workspace") else ""))
    for g in r.get("grants") or []:
        out.line(out.dim(f"    {'everyone' if g['app'] == '*' else g['app']}: {g['access']}"))


def cmd_context(ctx: Ctx) -> int:
    a, out, act = ctx.args, ctx.out, ctx.args.action
    http = ctx.http()
    if act == "list":
        d = http.json("GET", "/v1/resources", params={"type": a.type, "workspace": a.workspace})
        out.emit(d, lambda d: [_render_res(out, r) for r in d["resources"]] if d["resources"] else out.line("No resources yet."))
    elif act == "add":
        body: dict[str, Any] = {"type": a.type, "title": a.title, "owner": a.owner, "workspace": a.workspace}
        if a.file:
            body.update(kind="file", path=str(Path(a.file).expanduser().resolve()))
        elif a.text is not None:
            body.update(kind="text", text=a.text)
        else:
            body.update(kind="context", collection=a.collection)
        r = http.json("POST", "/v1/resources", {k: v for k, v in body.items() if v is not None})
        out.emit(r, lambda r: (out.line(f"Created {r['id']} ({r['ref']}); private to {r['owner']} until you share it."),))
    elif act == "inspect":
        r = http.json("GET", f"/v1/resources/{a.id}")
        out.emit(r, lambda r: (_render_res(out, r), out.line(out.dim(f"    ref {r['ref']}   content {r.get('content')}"))))
    elif act == "read":
        r = http.json("GET", f"/v1/resources/{a.id}/read", params={"doc": a.doc})
        out.emit(r, lambda r: out.line(r["text"]) if "text" in r else [out.line(f"{d['doc_id']}  {d.get('title') or ''}  ({d['chars']} chars)") for d in r["documents"]])
    elif act == "search":
        d = http.json("POST", "/v1/resources/search", {"query": a.query, "limit": a.limit, "workspace": a.workspace})
        out.emit(d, lambda d: [(out.line(f"{out.bold(i['resource_title'])} {out.dim(i['resource'] + '/' + i['doc_id'])}  score {i['score']}"),
                                out.line(f"  {i['text'][:240]}")) for i in d["items"]] or out.line("Nothing matched."))
    elif act == "share":
        if a.everyone:
            if not _confirm(ctx, f"Make {a.id} readable by EVERY app on this machine?", a.yes):
                out.line("Cancelled.")
                return FAIL
            r = http.json("POST", f"/v1/resources/{a.id}/share", {"global": True})
        else:
            if not a.apps:
                raise AnviraError("invalid_request", "Name the apps to share with, or use --global.")
            r = http.json("POST", f"/v1/resources/{a.id}/share", {"with": a.apps, "access": "write" if a.write else "read"})
        out.emit(r, lambda r: _render_res(out, r))
    elif act == "revoke":
        r = http.json("POST", f"/v1/resources/{a.id}/revoke", {"with": a.apps or None})
        out.emit(r, lambda r: (out.line("Revoked."), _render_res(out, r)))
    elif act == "delete":
        if not _confirm(ctx, f"Delete resource {a.id}{' and its indexed documents' if a.purge else ''}?", a.yes):
            out.line("Cancelled.")
            return FAIL
        r = http.json("DELETE", f"/v1/resources/{a.id}", params={"purge": "true" if a.purge else None})
        out.emit(r, lambda r: out.line(f"Deleted {r['id']}."))
    elif act == "requests":
        d = http.json("GET", "/v1/resources/requests", params={"state": "all" if a.all else "pending"})
        out.emit(d, lambda d: [out.line(f"{q['id']}  {q['requester']} wants {q['access']} on {q['resource']}"
                                        f" ({q.get('title') or '?'})  [{q['state']}]  {out.dim(q.get('reason') or '')}")
                               for q in d["requests"]] or out.line("No access requests."))
    elif act in ("approve", "deny"):
        r = http.json("POST", f"/v1/resources/requests/{a.id}/{act}")
        out.emit(r, lambda r: out.line(f"Request {r['id']} {r['state']} ({r['requester']} -> {r['resource']})."))
    elif act == "audit":
        d = http.json("GET", "/v1/resources/audit", params={"limit": a.limit, "resource": a.resource})
        out.emit(d, lambda d: [out.line(f"{time.strftime('%H:%M:%S', time.localtime(e['ts']))}  {e['actor']:<14} {e['action']:<8} "
                                        f"{e['resource'] or '-':<16} {out.dim(e['detail'])}") for e in d["events"]] or out.line("No events."))
    return OK


def cmd_workspace(ctx: Ctx) -> int:
    a, out = ctx.args, ctx.out
    http = ctx.http()
    if a.action == "list":
        d = http.json("GET", "/v1/workspaces")
        out.emit(d, lambda d: [out.line(f"{w['name']}  {out.dim(w['id'])}  {w['resources']} resources") for w in d["workspaces"]]
                 or out.line("No workspaces."))
    elif a.action == "create":
        r = http.json("POST", "/v1/workspaces", {"name": a.name})
        out.emit(r, lambda r: out.line(f"Workspace '{r['name']}' ({r['id']})."))
    else:
        r = http.json("POST", f"/v1/workspaces/{a.name}/share", {"with": a.apps, "access": "write" if a.write else "read"})
        out.emit(r, lambda r: out.line(f"Shared {len(r['shared'])} resource(s) in '{r['workspace']}' with {', '.join(r['with'])}."))
    return OK


def cmd_capability(ctx: Ctx) -> int:
    d = ctx.http().json("GET", "/v1/capabilities")
    out = ctx.out
    out.emit(d, lambda d: ([out.line(f"{_state_mark(out, c['state']):<22} {c['name']:<10} {out.dim(str(c['detail']))}") for c in d["capabilities"]],
                           out.line(out.dim(d["note"]))))
    return OK


# --------------------------------------------------------------------------------------- chat
def cmd_chat(ctx: Ctx) -> int:
    a, out = ctx.args, ctx.out
    messages = ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": a.prompt}]
    body: dict[str, Any] = {"messages": messages, "stream": not (a.no_stream or out.json_mode)}
    if a.memory:
        body["memory"] = {"recall": True}
    http = ctx.http()
    if not body["stream"]:
        res = http.json("POST", "/v1/chat", body, timeout=660)
        out.emit(res, lambda r: out.line(r["choices"][0]["message"]["content"]))
        return OK
    for event, data in http.sse("POST", "/v1/chat", body):
        if event == "error":
            raise AnviraError.from_response(502, json.loads(data))
        if event != "message" or data.strip() == "[DONE]":
            continue
        try:
            delta = json.loads(data)["choices"][0]["delta"].get("content")
        except (ValueError, KeyError, IndexError):
            continue
        if delta:
            sys.stdout.write(delta)
            sys.stdout.flush()
    print()
    return OK


# ------------------------------------------------------------------------------- config / apps / aicl
def cmd_config(ctx: Ctx) -> int:
    a, out = ctx.args, ctx.out
    info = probe()
    if a.action == "path":
        out.emit({"path": str(ctx.layout.config_file)}, lambda d: out.line(d["path"]))
        return OK
    live = info.running and info.ready
    if a.action in ("list", "get"):
        cfg = ctx.http(start=False).json("GET", "/v1/runtime/config")["config"] if live else ctx.config.as_dict()
        flat = RuntimeConfig(ctx.layout, cfg).flat()
        if a.action == "get":
            if a.key not in flat:
                raise AnviraError("config_key_unknown", f"Unknown config key '{a.key}'.", hint="Run `anvira config list`.")
            out.emit({a.key: flat[a.key]}, lambda d: out.line(json.dumps(d[a.key]) if not isinstance(d[a.key], str) else d[a.key]))
        else:
            out.emit(flat, lambda d: out.kv([(k, json.dumps(v)) for k, v in sorted(d.items())]))
        return OK
    # set
    if live:
        res = ctx.http(start=False).json("PUT", "/v1/runtime/config", {"key": a.key, "value": a.value})
    else:
        try:
            val = ctx.config.set(a.key, a.value)
        except ConfigError as exc:
            raise AnviraError("invalid_config", str(exc)) from None
        ctx.config.save()
        res = {"key": a.key, "value": val, "restart_required": False}
    out.emit(res, lambda r: out.line(f"{r['key']} = {json.dumps(r['value'])}" + ("  (restart the runtime to apply)" if r.get("restart_required") and live else "")))
    return OK


def cmd_app(ctx: Ctx) -> int:
    a, out = ctx.args, ctx.out
    http = ctx.http()
    if a.action == "list":
        d = http.json("GET", "/v1/apps")

        def render(d: dict[str, Any]) -> None:
            if not d["apps"]:
                out.line("No applications registered yet. Apps register themselves when they first connect.")
                return
            out.table(["APP", "NAME", "PERMISSIONS", "PENDING", "LAST SEEN"], [
                [x["app_id"], x["name"], len(x["permissions"]), ",".join(x.get("requested") or []) or "-",
                 human_duration(time.time() - x["last_seen"]) + " ago" if x.get("last_seen") else "never"] for x in d["apps"]])
        out.emit(d, render)
    elif a.action == "show":
        d = next((x for x in http.json("GET", "/v1/apps")["apps"] if x["app_id"] == a.app_id), None)
        if d is None:
            raise AnviraError("app_not_found", f"App '{a.app_id}' is not registered.")
        out.emit(d, lambda x: out.kv([("app", x["app_id"]), ("name", x["name"]), ("permissions", ", ".join(x["permissions"])),
                                      ("requested", ", ".join(x.get("requested") or []) or "-")]))
    elif a.action in ("grant", "deny"):
        bad = [p for p in a.permissions if p not in PERMISSIONS]
        if bad:
            raise AnviraError("unknown_permission", f"Unknown permission(s): {', '.join(bad)}",
                              hint="Valid: " + ", ".join(sorted(PERMISSIONS)))
        d = http.json("POST", f"/v1/apps/{a.app_id}/{a.action}", {"permissions": a.permissions})
        out.emit(d, lambda x: out.line(f"{x['app_id']}: {', '.join(x['permissions'])}"))
    elif a.action == "revoke":
        if not _confirm(ctx, f"Revoke '{a.app_id}'? It must register again to reconnect.", a.yes):
            return OK
        d = http.json("DELETE", f"/v1/apps/{a.app_id}")
        out.emit(d, lambda x: out.line(f"Revoked {x['revoked']}"))
    return OK


def cmd_aicl(ctx: Ctx) -> int:
    a, out = ctx.args, ctx.out
    d = ctx.http().json("GET", "/v1/aicl/status")
    if not d.get("available"):
        raise AnviraError("aicl_unavailable", f"AICL bus unavailable: {d.get('error')}", hint="Run `anvira doctor`.")

    def render(d: dict[str, Any]) -> None:
        if a.action == "trace":
            out.table(["TIME", "MODULE", "ACTION", "OP", "APP", "BYTES", "MS", "OK"], [
                [time.strftime("%H:%M:%S", time.localtime(r["ts"])), r["module"], r["action"], r["op"], r["app"] or "-",
                 f"{r['request_bytes']}/{r['response_bytes']}", r["ms"], "yes" if r["ok"] else out.red("no")] for r in d["recent"]])
            return
        out.line(f"AICL {d['aicl_version'] or ''}  codec: {d['codec']}  native core: {'yes' if d['native_core'] else 'no (pure-Python codec)'}")
        out.table(["MODULE", "CALLS", "ERRORS", "AVG MS", "CAPABILITIES"], [
            [n, m["calls"], m["errors"], m["avg_ms"], ",".join(m["capabilities"])] for n, m in d["modules"].items()])
    out.emit(d, render)
    return OK


# -------------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="machine-readable JSON output")
    common.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS, help="suppress progress text")

    p = argparse.ArgumentParser(prog="anvira", description=DESCRIPTION, epilog=EPILOG,
                                formatter_class=argparse.RawDescriptionHelpFormatter, parents=[common])
    p.set_defaults(json=False, quiet=False)
    p.add_argument("-V", "--version", action="version", version=f"anvira {RUNTIME_VERSION} (runtime API v{API_VERSION})")
    sub = p.add_subparsers(dest="cmd", metavar="<command>", title="commands")

    def add(parent, name, func, help_, **kw):
        sp = parent.add_parser(name, help=help_, description=help_, parents=[common], **kw)
        sp.set_defaults(func=func)
        return sp

    add(sub, "status", cmd_status, "show runtime, service and model status")
    add(sub, "doctor", cmd_doctor, "diagnose problems and suggest fixes")
    add(sub, "version", cmd_version, "show CLI, runtime and API versions")
    add(sub, "hardware", cmd_hardware, "show detected CPU/RAM/GPU/storage")
    s = add(sub, "ui", cmd_ui, "live terminal dashboard: watch the runtime and run ORCHA jobs")
    s.add_argument("--once", action="store_true", help="print one frame and exit")
    s.add_argument("--exec", metavar="CMD", help="run dashboard command(s) headlessly (separate several with ';;')")
    s.add_argument("--page", choices=["overview", "health", "models", "apps", "logs"], help="start on this page")
    s = add(sub, "open", cmd_ui, "open the live dashboard (same as `ui`); e.g. `anvira open health`")
    s.add_argument("page", nargs="?", choices=["overview", "health", "models", "apps", "logs"], help="page to start on")
    s.add_argument("--once", action="store_true", help="print one frame and exit")
    s.add_argument("--exec", metavar="CMD", help=argparse.SUPPRESS)

    rt = sub.add_parser("runtime", help="manage the runtime process and installation", parents=[common])
    rts = rt.add_subparsers(dest="sub", metavar="<action>", required=True)
    s = add(rts, "start", cmd_runtime_start, "start the runtime (detached)")
    s.add_argument("--foreground", action="store_true", help="run in this terminal with console logging")
    s.add_argument("--port", type=int, help="API port (0 = choose a free one)")
    s.add_argument("--auto-stop", action="store_true",
                   help="on-demand: stop by itself when no app is open (default: stay running until `anvira runtime stop`)")
    s.add_argument("--idle-grace", type=float, metavar="SECONDS", help="with --auto-stop: idle time before stopping (default 30)")
    add(rts, "stop", cmd_runtime_stop, "stop the runtime gracefully")
    s = add(rts, "restart", cmd_runtime_restart, "stop and start the runtime")
    s.add_argument("--foreground", action="store_true", default=False, help=argparse.SUPPRESS)
    s.add_argument("--port", type=int, help=argparse.SUPPRESS)
    s.add_argument("--auto-stop", action="store_true", default=False, help=argparse.SUPPRESS)
    s.add_argument("--idle-grace", type=float, default=None, help=argparse.SUPPRESS)
    add(rts, "status", cmd_status, "same as `anvira status`")
    s = add(rts, "logs", cmd_runtime_logs, "show runtime/service logs")
    s.add_argument("-s", "--service", default="runtime", help="runtime | orcha | nomi | model:<id> (default: runtime)")
    s.add_argument("-n", "--lines", type=int, default=80)
    s.add_argument("-f", "--follow", action="store_true")
    add(rts, "info", cmd_runtime_info, "show install paths and configuration locations")
    add(rts, "locate", cmd_runtime_locate, "where is the runtime installed, and how was it found")
    s = add(rts, "register", cmd_runtime_register, "use a runtime you already have in another folder (nothing is copied)")
    s.add_argument("path", help="the folder that contains install.json / anvira.cmd")
    s = add(rts, "install-gpu", cmd_runtime_install_gpu, "add the NVIDIA GPU acceleration pack to this install")
    s.add_argument("--repo", help="GitHub repository (owner/name)")
    s.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    for name, upd in (("install", False), ("update", True)):
        s = add(rts, name, lambda c, u=upd: cmd_runtime_install(c, u),
                "update to the bundled runtime version" if upd else "install the shared runtime (asks first; downloads no models)")
        s.add_argument("--github", action="store_true", help="download the latest release from GitHub (the default outside a source checkout)")
        s.add_argument("--repo", help="GitHub repository, owner/name (default: ANVIRA_RUNTIME_REPO)")
        s.add_argument("--dir", help="install into this folder (any drive); the location is remembered for every app")
        s.add_argument("--gpu", choices=["auto", "yes", "no"], default="auto", help="NVIDIA GPU pack: ask (auto), always, or never")
        s.add_argument("--source", help="runtime bundle: .zip, URL or source checkout (default: ANVIRA_RUNTIME_SOURCE / this checkout)")
        s.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
        s.add_argument("--force", action="store_true", help="reinstall even if up to date")
        s.add_argument("--start", action="store_true", help="start the runtime afterwards")

    md = sub.add_parser("model", help="discover, install, select and place models", parents=[common])
    mds = md.add_subparsers(dest="sub", metavar="<action>", required=True)
    s = add(mds, "list", cmd_model_list, "list installed and available models")
    s.add_argument("--installed", action="store_true", help="only installed models")
    s = add(mds, "search", cmd_model_search, "search Hugging Face for GGUF models")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=15)
    s = add(mds, "info", cmd_model_info, "show details of one model")
    s.add_argument("id")
    s = add(mds, "compat", cmd_model_compat, "can this model run on this machine?")
    s.add_argument("id")
    s = add(mds, "recommend", cmd_model_recommend, "catalog models that fit this machine")
    s.add_argument("--limit", type=int, default=5)
    add(mds, "status", cmd_model_status, "active model, backend and hardware")
    s = add(mds, "install", cmd_model_install, "download a model (catalog id, owner/repo or --url)")
    s.add_argument("model", nargs="?", help="catalog id (e.g. qwen3-8b) or Hugging Face repo owner/name")
    s.add_argument("--file", help="specific .gguf file in the repo")
    s.add_argument("--url", help="direct URL to a .gguf file")
    s.add_argument("--dir", help="download into this folder instead of the models folder")
    s.add_argument("-y", "--yes", action="store_true", help="skip the download confirmation")
    s.add_argument("--no-wait", action="store_true", help="return immediately; track with `anvira job get`")
    s = add(mds, "remove", cmd_model_remove, "remove a model (unlink; --delete-file to delete the file)")
    s.add_argument("id")
    s.add_argument("--delete-file", action="store_true")
    s.add_argument("-y", "--yes", action="store_true")
    s = add(mds, "use", cmd_model_use, "make a model the active model (starts it)")
    s.add_argument("id")
    s.add_argument("--wait", type=float, default=120.0, help="seconds to wait for loading (default 120)")
    add(mds, "unuse", cmd_model_unuse, "stop and clear the active model")
    s = add(mds, "dir", cmd_model_dir, "show or set where new models are stored (any folder)")
    s.add_argument("path", nargs="?")
    s.add_argument("--move", action="store_true", help="also move existing models there")
    s = add(mds, "dirs", cmd_model_dirs, "extra folders scanned for existing models")
    s.add_argument("action", choices=["list", "add", "remove"])
    s.add_argument("path", nargs="?")
    add(mds, "discover", cmd_model_discover, "model folders found from installed Anvira apps")
    s = add(mds, "add", cmd_model_add, "register a .gguf file or a models folder from anywhere, in place (no copy)")
    s.add_argument("file", metavar="file-or-folder")
    s.add_argument("--id")
    s = add(mds, "provider", cmd_model_provider, "cloud / remote OpenAI-compatible models")
    s.add_argument("action", choices=["list", "add", "remove"])
    s.add_argument("id", nargs="?", help="provider id (for remove)")
    s.add_argument("--base-url")
    s.add_argument("--model")
    s.add_argument("--label")
    s.add_argument("--api-key", help="(visible in shell history; prefer --api-key-env or --api-key-stdin)")
    s.add_argument("--api-key-env", help="read the key from this environment variable")
    s.add_argument("--api-key-stdin", action="store_true", help="prompt for / read the key from stdin")

    oc = sub.add_parser("orcha", help="orchestration engine: status and jobs", parents=[common])
    ocs = oc.add_subparsers(dest="sub", metavar="<action>", required=True)
    add(ocs, "status", cmd_orcha_status, "ORCHA service and engine status")
    s = add(ocs, "run", cmd_orcha_run, "run an orchestration job")
    s.add_argument("task")
    s.add_argument("--graph", default="default", choices=["default", "research", "multi_agent"])
    s.add_argument("--reasoning", choices=["fast", "light", "medium", "high", "max"])
    s.add_argument("--workspace", action="append", help="allow agent tools in this folder (repeatable)")
    s.add_argument("--no-wait", action="store_true")
    s = add(ocs, "jobs", cmd_orcha_jobs, "list jobs")
    s.add_argument("--state")
    s.add_argument("--limit", type=int, default=20)
    s = add(ocs, "cancel", cmd_orcha_cancel, "cancel a running job")
    s.add_argument("id")

    jb = sub.add_parser("job", help="inspect a job by id", parents=[common])
    jbs = jb.add_subparsers(dest="sub", metavar="<action>", required=True)
    s = add(jbs, "get", cmd_job_get, "show a job and its result")
    s.add_argument("id")

    nm = sub.add_parser("nomi", help="memory subsystem: status, search, inspect", parents=[common])
    nms = nm.add_subparsers(dest="sub", metavar="<action>", required=True)
    add(nms, "status", cmd_nomi_status, "Nomi service status")
    s = add(nms, "search", cmd_nomi_search, "search memory")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--scope", choices=["app", "shared", "all"])
    s.add_argument("--app", help="search this app's namespace (default: owner)")
    s = add(nms, "inspect", cmd_nomi_inspect, "show one memory")
    s.add_argument("id")
    s = add(nms, "store", cmd_nomi_store, "store a memory")
    s.add_argument("text")
    s.add_argument("--title")
    s.add_argument("--tag", action="append")
    s.add_argument("--scope", choices=["app", "shared"])
    s.add_argument("--app")
    s.add_argument("--type", default="note")
    s = add(nms, "delete", cmd_nomi_delete, "delete a memory")
    s.add_argument("id")

    cx = sub.add_parser("context", help="shared data: list, search, share and audit resources", parents=[common])
    cxs = cx.add_subparsers(dest="action", metavar="<action>", required=True)
    s = add(cxs, "list", cmd_context, "resources you can see")
    s.add_argument("--type")
    s.add_argument("--workspace")
    s = add(cxs, "add", cmd_context, "register a resource (private until shared)")
    s.add_argument("title")
    s.add_argument("--type", default="document")
    src = s.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="point at a text file (read on demand, never copied)")
    src.add_argument("--text", help="short inline text")
    src.add_argument("--collection", help="a collection in the owner's context index")
    s.add_argument("--owner", default="user", help="owning app id (default: user)")
    s.add_argument("--workspace")
    s = add(cxs, "inspect", cmd_context, "one resource, its content pointer and grants")
    s.add_argument("id")
    s = add(cxs, "read", cmd_context, "read a resource (or one document of it)")
    s.add_argument("id")
    s.add_argument("--doc")
    s = add(cxs, "search", cmd_context, "search across everything authorised")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=8)
    s.add_argument("--workspace")
    s = add(cxs, "share", cmd_context, "let other apps use a resource")
    s.add_argument("id")
    s.add_argument("apps", nargs="*", metavar="app")
    s.add_argument("--write", action="store_true", help="allow writes (default: read)")
    s.add_argument("--global", dest="everyone", action="store_true", help="every app may read it (asks first)")
    s.add_argument("-y", "--yes", action="store_true")
    s = add(cxs, "revoke", cmd_context, "withdraw sharing (all apps, or the named ones)")
    s.add_argument("id")
    s.add_argument("apps", nargs="*", metavar="app")
    s = add(cxs, "delete", cmd_context, "delete a resource record")
    s.add_argument("id")
    s.add_argument("--purge", action="store_true", help="also delete its indexed documents")
    s.add_argument("-y", "--yes", action="store_true")
    s = add(cxs, "requests", cmd_context, "access requests from apps")
    s.add_argument("--all", action="store_true")
    for name in ("approve", "deny"):
        s = add(cxs, name, cmd_context, f"{name} an access request")
        s.add_argument("id")
    s = add(cxs, "audit", cmd_context, "who shared, read or wrote what")
    s.add_argument("--limit", type=int, default=30)
    s.add_argument("--resource")

    ws = sub.add_parser("workspace", help="group resources by project", parents=[common])
    wss = ws.add_subparsers(dest="action", metavar="<action>", required=True)
    add(wss, "list", cmd_workspace, "workspaces")
    s = add(wss, "create", cmd_workspace, "create a workspace")
    s.add_argument("name")
    s = add(wss, "share", cmd_workspace, "share every resource you own in a workspace")
    s.add_argument("name")
    s.add_argument("apps", nargs="+", metavar="app")
    s.add_argument("--write", action="store_true")

    pa = sub.add_parser("path", help="put `anvira` on your PATH (asks first) so it works from any terminal", parents=[common])
    pas = pa.add_subparsers(dest="action", metavar="<action>", required=True)
    for name in ("add", "remove"):
        s = add(pas, name, cmd_path, f"{name} this folder {'to' if name == 'add' else 'from'} your user PATH")
        s.add_argument("-y", "--yes", action="store_true")

    cp = sub.add_parser("capability", help="what the runtime can do, and what is active", parents=[common])
    cps = cp.add_subparsers(dest="action", metavar="<action>", required=True)
    add(cps, "list", cmd_capability, "capabilities and their current state")

    s = add(sub, "chat", cmd_chat, "send one prompt to the active model")
    s.add_argument("prompt")
    s.add_argument("--system")
    s.add_argument("--memory", action="store_true", help="recall relevant memories first")
    s.add_argument("--no-stream", action="store_true")

    cf = sub.add_parser("config", help="read and change runtime configuration", parents=[common])
    cfs = cf.add_subparsers(dest="action", metavar="<action>", required=True)
    add(cfs, "list", cmd_config, "list every setting")
    s = add(cfs, "get", cmd_config, "show one setting")
    s.add_argument("key", choices=flat_keys(), metavar="key")
    s = add(cfs, "set", cmd_config, "change a setting")
    s.add_argument("key", metavar="key")
    s.add_argument("value")
    add(cfs, "path", cmd_config, "path of the config file")

    ap = sub.add_parser("app", help="applications connected to the runtime", parents=[common])
    aps = ap.add_subparsers(dest="action", metavar="<action>", required=True)
    add(aps, "list", cmd_app, "registered applications")
    s = add(aps, "show", cmd_app, "one application")
    s.add_argument("app_id")
    for name in ("grant", "deny"):
        s = add(aps, name, cmd_app, f"{name} permissions to an application")
        s.add_argument("app_id")
        s.add_argument("permissions", nargs="+", metavar="permission")
    s = add(aps, "revoke", cmd_app, "remove an application (it must re-register)")
    s.add_argument("app_id")
    s.add_argument("-y", "--yes", action="store_true")

    ai = sub.add_parser("aicl", help="the internal AICL module bus", parents=[common])
    ais = ai.add_subparsers(dest="action", metavar="<action>", required=True)
    add(ais, "status", cmd_aicl, "bus statistics per module")
    add(ais, "trace", cmd_aicl, "most recent AICL packets")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    typed = sys.argv[1:] if argv is None else argv
    if not typed and sys.stdin.isatty() and sys.stdout.isatty():
        argv = ["ui"]                       # a bare `anvira` in a terminal opens the live dashboard, not a wall of help
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return USAGE if typed else OK
    _autoregister_portable()
    ctx = Ctx(args)
    try:
        try:
            return args.func(ctx)
        finally:
            ctx.release()
    except AnviraError as exc:
        code = _exit_for(exc)
        if ctx.out.json_mode:
            print(json.dumps({"error": {"code": exc.code, "message": exc.message, "hint": exc.hint, "exit_code": code,
                                        "details": exc.details}}, indent=2))
        else:
            label = "Runtime unavailable" if isinstance(exc, RuntimeNotInstalled) else "Error"
            ctx.out.err(f"{label}: {exc.message}")
            if isinstance(exc, RuntimeStartFailed):
                ctx.out.err("Runtime failed to start.\nRun `anvira doctor` for diagnostics.")
                tail = (exc.details or {}).get("log_tail")
                if tail:
                    ctx.out.err(ctx.out.dim(redact(tail)))
            elif exc.hint:
                ctx.out.err(f"  {exc.hint}")
            if exc.code == "no_model" and not (exc.details or {}).get("installed"):
                ctx.out.err("Get a model first: `anvira model dirs add <folder with .gguf files>` or `anvira model install <id>`.")
        return code
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return OK


if __name__ == "__main__":
    sys.exit(main())
