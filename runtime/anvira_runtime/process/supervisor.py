"""Lightweight local process supervisor.

Replaces the per-service Electron modules (``orchaProcess.cjs``,
``nomiProcess.cjs``, ``runtimeTracker.cjs``) with one implementation that
keeps their good ideas: adopt a live compatible service instead of spawning a
duplicate, record child pids so a crashed parent doesn't leak processes, wait
for a real health probe (not just "process started"), and surface the log
tail when startup fails.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from ..security.secrets import read_json, redact, write_private

# --------------------------------------------------------------------- helpers


def pid_alive(pid: int | None) -> bool:
    """Cross-platform liveness check WITHOUT signalling the process.

    (On Windows ``os.kill(pid, 0)`` would call TerminateProcess.)
    """
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        k32.OpenProcess.restype = wintypes.HANDLE
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_tree(pid: int, force: bool = False) -> None:
    if not pid_alive(pid):
        return
    if sys.platform == "win32":
        args = ["taskkill", "/PID", str(pid), "/T"] + (["/F"] if force else [])
        subprocess.run(args, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW, check=False)
    else:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(pid), signal.SIGKILL if force else signal.SIGTERM)


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def tail_file(path: Path, lines: int = 40) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 64_000))
            data = fh.read().decode("utf-8", errors="replace")
        return "\n".join(data.splitlines()[-lines:])
    except OSError:
        return ""


# ------------------------------------------------------------------- data types
@dataclass
class ServiceSpec:
    name: str
    argv: list[str]
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    log_path: Path | None = None
    port: int | None = None
    health_url: str | None = None
    health_headers: dict[str, str] = field(default_factory=dict)
    ready_timeout_s: float = 60.0
    restart: bool = True
    kind: str = "service"                       # service | model
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceState:
    spec: ServiceSpec
    state: str = "stopped"   # stopped | starting | running | degraded | failed
    proc: subprocess.Popen | None = None
    pid: int | None = None
    started_at: float | None = None
    restarts: list[float] = field(default_factory=list)
    total_restarts: int = 0
    last_error: str | None = None
    last_health: dict[str, Any] | None = None
    consecutive_failures: int = 0
    expected_stop: bool = False
    adopted: bool = False

    def public(self) -> dict[str, Any]:
        return {
            "name": self.spec.name, "kind": self.spec.kind, "state": self.state, "pid": self.pid,
            "port": self.spec.port, "restarts": self.total_restarts, "last_error": self.last_error,
            "started_at": self.started_at,
            "uptime_s": round(time.time() - self.started_at, 1) if self.started_at and self.state == "running" else None,
            "health": self.last_health, "log": str(self.spec.log_path) if self.spec.log_path else None,
            **self.spec.meta,
        }


class Supervisor:
    def __init__(self, children_file: Path, *, health_interval_s: float = 2.0, restart_limit: int = 5,
                 restart_window_s: float = 300.0, log: Callable[[str, str], None] | None = None):
        self.children_file = children_file
        self.services: dict[str, ServiceState] = {}
        self.health_interval_s = health_interval_s
        self.restart_limit = restart_limit
        self.restart_window_s = restart_window_s
        self._log = log or (lambda level, msg: None)
        self._monitor: asyncio.Task | None = None
        self._client = httpx.AsyncClient(timeout=3.0)
        self._locks: dict[str, asyncio.Lock] = {}

    # -------------------------------------------------------------- persistence
    def _children(self) -> dict[str, dict[str, Any]]:
        return read_json(self.children_file, {})

    def _record_child(self, st: ServiceState) -> None:
        data = self._children()
        data[st.spec.name] = {"pid": st.pid, "port": st.spec.port, "health_url": st.spec.health_url}
        write_private(self.children_file, json.dumps(data, indent=2))

    def _forget_child(self, name: str) -> None:
        data = self._children()
        if data.pop(name, None) is not None:
            write_private(self.children_file, json.dumps(data, indent=2))

    async def reap_stale(self) -> list[str]:
        """Terminate children left behind by a previous runtime that died.

        A recorded pid is only killed if its recorded health URL still answers
        (so a recycled pid belonging to an unrelated program is left alone).
        """
        reaped = []
        for name, rec in self._children().items():
            pid, url = rec.get("pid"), rec.get("health_url")
            if pid_alive(pid) and url:
                try:
                    r = await self._client.get(url)
                    alive = r.status_code < 500
                except httpx.HTTPError:
                    alive = False
                if alive:
                    kill_tree(pid, force=True)
                    reaped.append(name)
                    self._log("WARNING", f"reaped stale child '{name}' pid={pid}")
        write_private(self.children_file, "{}")
        return reaped

    # ------------------------------------------------------------------ control
    def add(self, spec: ServiceSpec) -> ServiceState:
        st = self.services.get(spec.name)
        if st is None:
            st = self.services[spec.name] = ServiceState(spec)
        else:
            st.spec = spec
        self._locks.setdefault(spec.name, asyncio.Lock())
        return st

    async def _probe(self, st: ServiceState) -> dict[str, Any]:
        spec = st.spec
        if not spec.health_url:
            return {"ok": st.proc is not None and st.proc.poll() is None}
        t0 = time.monotonic()
        try:
            r = await self._client.get(spec.health_url, headers=spec.health_headers)
            return {"ok": r.status_code == 200, "status": r.status_code,
                    "latency_ms": round((time.monotonic() - t0) * 1000, 1), "checked_at": time.time()}
        except httpx.HTTPError as exc:
            return {"ok": False, "error": type(exc).__name__, "checked_at": time.time()}

    async def start(self, name: str) -> ServiceState:
        async with self._locks[name]:
            return await self._start_locked(name)

    async def _start_locked(self, name: str) -> ServiceState:
        st = self.services[name]
        spec = st.spec
        if st.state == "running" and st.proc and st.proc.poll() is None:
            return st
        st.state, st.expected_stop, st.last_error = "starting", False, None

        # Adopt a healthy compatible service already listening (never spawn a duplicate).
        if spec.port and spec.health_url and port_in_use(spec.port):
            probe = await self._probe(st)
            if probe["ok"]:
                st.state, st.adopted, st.last_health = "running", True, probe
                st.started_at = st.started_at or time.time()
                self._log("INFO", f"adopted existing '{name}' on port {spec.port}")
                return st
            st.state, st.last_error = "failed", f"Port {spec.port} is in use by another program"
            raise RuntimeError(st.last_error)

        env = {**os.environ, **spec.env}
        log_fh = open(spec.log_path, "ab") if spec.log_path else subprocess.DEVNULL
        if spec.log_path:
            spec.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} starting {name}: "  # type: ignore[union-attr]
                         f"{redact(' '.join(spec.argv))}\n".encode())
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        try:
            st.proc = subprocess.Popen(spec.argv, cwd=spec.cwd, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log_fh, stderr=subprocess.STDOUT, **kwargs)
        except OSError as exc:
            st.state, st.last_error = "failed", f"could not launch: {exc}"
            raise RuntimeError(st.last_error) from exc
        finally:
            if spec.log_path:
                log_fh.close()  # type: ignore[union-attr]
        st.pid, st.adopted, st.started_at = st.proc.pid, False, time.time()
        self._record_child(st)

        deadline = time.monotonic() + spec.ready_timeout_s
        while time.monotonic() < deadline:
            if st.proc.poll() is not None:
                st.state = "failed"
                st.last_error = (f"exited during startup (code {st.proc.returncode}). Log tail:\n"
                                 f"{tail_file(spec.log_path, 15) if spec.log_path else ''}")
                self._forget_child(name)
                raise RuntimeError(st.last_error)
            probe = await self._probe(st)
            if probe["ok"]:
                st.state, st.last_health, st.consecutive_failures = "running", probe, 0
                self._log("INFO", f"'{name}' ready pid={st.pid} port={spec.port}")
                return st
            await asyncio.sleep(0.4)
        await self._stop_locked(name)
        st.state = "failed"
        st.last_error = f"did not become healthy within {spec.ready_timeout_s:.0f}s"
        raise RuntimeError(st.last_error)

    async def stop(self, name: str, timeout: float | None = None) -> None:
        async with self._locks[name]:
            await self._stop_locked(name, timeout)

    async def _stop_locked(self, name: str, timeout: float | None = None) -> None:
        # Windows console services (uvicorn, llama-server) cannot be asked to exit politely
        # without a console attached, so give the polite request only a moment there.
        if timeout is None:
            timeout = 2.0 if sys.platform == "win32" else 8.0
        st = self.services.get(name)
        if not st:
            return
        st.expected_stop = True
        if st.adopted:
            st.state, st.adopted, st.pid = "stopped", False, None
            return
        pid = st.pid
        if pid and pid_alive(pid):
            kill_tree(pid, force=False)
            deadline = time.monotonic() + timeout
            while pid_alive(pid) and time.monotonic() < deadline:
                await asyncio.sleep(0.1)
            if pid_alive(pid):
                kill_tree(pid, force=True)
                for _ in range(30):
                    if not pid_alive(pid):
                        break
                    await asyncio.sleep(0.1)
        if st.proc:
            with contextlib.suppress(Exception):
                st.proc.wait(timeout=1)
        st.state, st.pid, st.proc = "stopped", None, None
        self._forget_child(name)

    async def restart(self, name: str) -> ServiceState:
        async with self._locks[name]:
            await self._stop_locked(name)
            return await self._start_locked(name)

    async def stop_all(self) -> None:
        if self._monitor:
            self._monitor.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._monitor
            self._monitor = None
        await asyncio.gather(*(self.stop(name) for name in list(self.services)), return_exceptions=True)
        await self._client.aclose()

    def remove(self, name: str) -> None:
        self.services.pop(name, None)

    # ------------------------------------------------------------------ monitor
    def start_monitor(self) -> None:
        if self._monitor is None:
            self._monitor = asyncio.create_task(self._monitor_loop())

    async def _monitor_loop(self) -> None:
        while True:
            await asyncio.sleep(self.health_interval_s)
            for name, st in list(self.services.items()):
                if st.state not in ("running", "degraded") or st.expected_stop:
                    continue
                try:
                    await self._check(name, st)
                except Exception as exc:  # never let the monitor die
                    self._log("ERROR", f"monitor error for '{name}': {exc}")

    async def _check(self, name: str, st: ServiceState) -> None:
        dead = (not st.adopted) and st.proc is not None and st.proc.poll() is not None
        probe = await self._probe(st)
        st.last_health = probe
        if probe["ok"] and not dead:
            st.consecutive_failures = 0
            st.state = "running"
            return
        st.consecutive_failures += 1
        if not dead and st.consecutive_failures < 3:
            st.state = "degraded"
            return
        code = st.proc.returncode if st.proc else None
        st.last_error = (f"process exited (code {code})" if dead else "health checks failing")
        tail = tail_file(st.spec.log_path, 12) if st.spec.log_path else ""
        self._log("ERROR", f"'{name}' {st.last_error}. Log tail:\n{tail}")
        if not st.spec.restart:
            st.state = "failed"
            return
        now = time.time()
        st.restarts = [t for t in st.restarts if now - t < self.restart_window_s]
        if len(st.restarts) >= self.restart_limit:
            st.state = "failed"
            st.last_error += f"; gave up after {self.restart_limit} restarts in {self.restart_window_s:.0f}s"
            return
        st.restarts.append(now)
        st.total_restarts += 1
        backoff = min(10.0, 2 ** (len(st.restarts) - 1))
        self._log("WARNING", f"restarting '{name}' in {backoff:.0f}s (attempt {len(st.restarts)})")
        await asyncio.sleep(backoff)
        try:
            await self.restart(name)
        except RuntimeError as exc:
            st.state, st.last_error = "failed" if len(st.restarts) >= self.restart_limit else "degraded", str(exc)

    def status(self) -> dict[str, dict[str, Any]]:
        return {n: s.public() for n, s in self.services.items()}
