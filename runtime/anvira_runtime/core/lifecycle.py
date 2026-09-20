"""On-demand lifecycle: the runtime is active only while an application is open.

* Nothing starts at login. The first app that needs the runtime starts it (the SDK does this on ``connect()``).
* Every open app holds a **lease** and renews it with a heartbeat (default TTL 45 s, renewed every ~15 s).
  Closing the app releases the lease; a crashed app simply stops renewing and its lease expires.
* When the runtime was started in ``auto_stop`` mode and no lease is held, nothing is running (no ORCHA run, model
  load, download or open stream) and that has been true for ``idle_grace_s``, the runtime shuts itself down cleanly:
  ORCHA, Nomi and any llama-server are stopped and the discovery file is removed. The next app that opens starts it
  again.
* A runtime started by hand (``anvira runtime start``) is *persistent*: leases are still tracked and shown, but it never
  stops on its own.
"""
from __future__ import annotations

import asyncio
import secrets
import time
from typing import Any, Callable


class Lifecycle:
    def __init__(self, *, auto_stop: bool, idle_grace_s: float, lease_ttl_s: float,
                 busy: Callable[[], int], on_idle: Callable[[], None] | None = None,
                 log: Callable[[str, str], None] | None = None):
        self.auto_stop, self.idle_grace_s, self.lease_ttl_s = auto_stop, float(idle_grace_s), float(lease_ttl_s)
        self.busy, self.on_idle = busy, on_idle
        self._log = log or (lambda level, msg: None)
        self.leases: dict[str, dict[str, Any]] = {}
        self.started = time.monotonic()
        self._idle_since: float | None = self.started
        self._task: asyncio.Task | None = None
        self.fired = False

    # ------------------------------------------------------------------ leases
    def acquire(self, app: str, ttl_s: float | None = None) -> dict[str, Any]:
        ttl = max(5.0, min(600.0, float(ttl_s or self.lease_ttl_s)))
        lid = "lease_" + secrets.token_hex(5)
        now = time.monotonic()
        self.leases[lid] = {"id": lid, "app": app, "ttl_s": ttl, "created": now, "last": now}
        self._idle_since = None
        self._log("DEBUG", f"lease acquired by {app} ({lid}, ttl {ttl:.0f}s); {len(self.leases)} open")
        return self._public(self.leases[lid], now)

    def heartbeat(self, lid: str, app: str | None = None) -> dict[str, Any] | None:
        lease = self.leases.get(lid)
        if lease is None or (app is not None and lease["app"] != app):
            return None
        lease["last"] = time.monotonic()
        return self._public(lease, lease["last"])

    def release(self, lid: str, app: str | None = None) -> bool:
        lease = self.leases.get(lid)
        if lease is None or (app is not None and lease["app"] != app):
            return False
        del self.leases[lid]
        self._log("DEBUG", f"lease released by {lease['app']} ({lid}); {len(self.leases)} open")
        return True

    def _prune(self, now: float) -> None:
        for lid in [k for k, v in self.leases.items() if now - v["last"] > v["ttl_s"]]:
            lease = self.leases.pop(lid)
            self._log("WARNING", f"lease of {lease['app']} expired without release ({lid}); the app probably closed or crashed")

    @staticmethod
    def _public(lease: dict[str, Any], now: float) -> dict[str, Any]:
        return {"id": lease["id"], "app": lease["app"], "ttl_s": lease["ttl_s"], "age_s": round(now - lease["created"], 1),
                "expires_in_s": round(lease["ttl_s"] - (now - lease["last"]), 1)}

    # ---------------------------------------------------------------- watcher
    def tick(self, now: float | None = None) -> bool:
        """Advance the idle clock. Returns True when the runtime should shut down now."""
        now = time.monotonic() if now is None else now
        self._prune(now)
        if self.leases or self.busy() > 0:
            self._idle_since = None
            return False
        if self._idle_since is None:
            self._idle_since = now
        return self.auto_stop and now - self._idle_since >= self.idle_grace_s

    async def _watch(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            if self.tick() and not self.fired:
                self.fired = True
                self._log("INFO", f"no app is using the runtime for {self.idle_grace_s:.0f}s; shutting down (it starts again when an app opens)")
                if self.on_idle:
                    self.on_idle()
                return

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._watch())

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        idle_for = None if self._idle_since is None else round(now - self._idle_since, 1)
        return {"mode": "on-demand" if self.auto_stop else "persistent", "auto_stop": self.auto_stop,
                "idle_grace_s": self.idle_grace_s, "lease_ttl_s": self.lease_ttl_s,
                "leases": [self._public(v, now) for v in self.leases.values()], "busy": self.busy(),
                "idle_for_s": idle_for,
                "shutdown_in_s": (max(0.0, round(self.idle_grace_s - idle_for, 1)) if self.auto_stop and idle_for is not None else None)}
