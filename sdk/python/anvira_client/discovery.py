"""Detect whether the runtime is installed / running, without importing it."""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import runtime_dirs


@dataclass
class RuntimeInfo:
    installed: bool = False
    running: bool = False
    ready: bool = False
    home: Path | None = None
    state_dir: Path | None = None
    install: dict[str, Any] = field(default_factory=dict)      # contents of install.json
    host: str = "127.0.0.1"
    port: int | None = None
    pid: int | None = None
    runtime_version: str | None = None
    api_version: int | None = None
    capabilities: list[str] = field(default_factory=list)
    stale_discovery: bool = False
    error: str | None = None

    @property
    def base_url(self) -> str | None:
        return f"http://{self.host}:{self.port}" if self.port else None

    def as_dict(self) -> dict[str, Any]:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items()}


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def http_json(url: str, timeout: float = 2.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def pid_alive(pid: int | None) -> bool:
    """Liveness without signalling (``os.kill(pid, 0)`` would kill on Windows)."""
    if not pid:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        k32.OpenProcess.restype = wintypes.HANDLE
        h = k32.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        try:
            code = wintypes.DWORD()
            return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def probe(env: dict[str, str] | None = None, timeout: float = 2.0) -> RuntimeInfo:
    """Inspect the shared install marker and discovery file, then ask the live runtime.

    Never raises; problems are reported in ``error``.
    """
    dirs = runtime_dirs(env)
    info = RuntimeInfo(home=dirs["home"], state_dir=dirs["state"])
    info.install = read_json(dirs["home"] / "install.json")
    entry = info.install.get("entry")
    entry_path = (Path(entry) if Path(entry).is_absolute() else dirs["home"] / entry) if entry else None   # portable: relative
    info.installed = bool(info.install) and (not entry_path or entry_path.exists())
    if info.install and entry_path and not entry_path.exists():
        info.error = f"Install marker points to a missing interpreter: {entry}"
    if (dirs["state"] / "owner.token").exists() and not info.error:
        info.installed = True          # a runtime has run here before (pip/source install without a marker)
    disc = read_json(dirs["state"] / "runtime.json")
    if disc:
        info.host, info.port, info.pid = disc.get("host", "127.0.0.1"), disc.get("port"), disc.get("pid")
        info.installed = True  # a discovery file means a runtime is/was here
        if not pid_alive(info.pid):
            info.stale_discovery, info.port = True, None
    if info.port:
        ver = http_json(f"http://{info.host}:{info.port}/version", timeout)
        health = http_json(f"http://{info.host}:{info.port}/health", timeout)
        if ver and ver.get("service") == "anvira-runtime" and health:
            info.running = True
            info.ready = health.get("status") in ("ok", "degraded")
            info.runtime_version, info.api_version = ver.get("runtime_version"), ver.get("api_version")
            info.capabilities = ver.get("capabilities", [])
        else:
            info.error = f"Nothing answering as Anvira Runtime on {info.host}:{info.port} (stale or port conflict)."
            info.stale_discovery = True
    return info


def read_token(kind: str, env: dict[str, str] | None = None) -> str | None:
    """``owner`` or ``register`` token from the state dir, if readable."""
    p = runtime_dirs(env)["state"] / f"{kind}.token"
    try:
        return p.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None
