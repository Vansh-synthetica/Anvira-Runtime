"""High-level Anvira Runtime client (Python).

    runtime = AnviraRuntime.connect("anvira-notes", install=ask_user)
    runtime.models.use("qwen3-8b")
    reply = runtime.chat_text([{"role": "user", "content": "hi"}])
    job = runtime.orcha.run("Summarise my notes", wait=True)
    runtime.memory.store("Prefers concise answers")

Applications never see ORCHA, Nomi or AICL — only these capability methods.
"""
from __future__ import annotations

import atexit
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from . import bootstrap
from .discovery import RuntimeInfo, probe, read_token
from .errors import (AnviraError, IncompatibleRuntime, InstallDeclined, RuntimeNotInstalled,
                     RuntimeNotRunning)
from .http import Http
from .paths import runtime_dirs

SDK_API_VERSION = 1   # the runtime API major this client speaks


def _semver(v: str) -> tuple[int, ...]:
    core = v.lstrip("vV").split("-", 1)[0].split("+", 1)[0]
    parts = [int(p) if p.isdigit() else 0 for p in core.split(".")[:3]]
    return tuple(parts + [0] * (3 - len(parts)))


class Job:
    """A unit of long-running runtime work (ORCHA run, model download, ...)."""

    TERMINAL = ("completed", "failed", "cancelled", "interrupted")

    def __init__(self, client: "AnviraRuntime", data: dict[str, Any]):
        self._client, self.data = client, data

    id = property(lambda self: self.data["id"])
    state = property(lambda self: self.data["state"])
    result = property(lambda self: self.data.get("result"))
    error = property(lambda self: self.data.get("error"))
    progress = property(lambda self: self.data.get("progress") or {})
    done = property(lambda self: self.data["state"] in Job.TERMINAL)

    def refresh(self) -> "Job":
        self.data = self._client._http.json("GET", f"/v1/jobs/{self.id}")["job"]
        return self

    def wait(self, timeout: float = 600.0, poll: float = 0.5) -> "Job":
        deadline = time.monotonic() + timeout
        while not self.done:
            if time.monotonic() > deadline:
                raise AnviraError("timeout", f"Job {self.id} still {self.state} after {timeout:.0f}s.")
            time.sleep(poll)
            self.refresh()
        return self

    def cancel(self) -> "Job":
        self.data = self._client._http.json("POST", f"/v1/jobs/{self.id}/cancel")["job"]
        return self

    def unwrap(self) -> Any:
        """Result of a completed job; raises :class:`AnviraError` if it failed or was cancelled."""
        if not self.done:
            self.wait()
        if self.state != "completed":
            err = self.error or {}
            raise AnviraError(err.get("code", self.state), err.get("message", f"Job {self.state}"),
                              hint=err.get("hint"), details=err.get("details"))
        return self.result

    def events(self) -> Iterator[dict[str, Any]]:
        """ORCHA event frames for this job (advanced; raw ORCHA event payloads)."""
        for _event, data in self._client._http.sse("GET", f"/v1/jobs/{self.id}/events"):
            try:
                yield json.loads(data)
            except ValueError:
                yield {"raw": data}

    def __repr__(self) -> str:
        return f"<Job {self.id} {self.kind} {self.state}>"

    kind = property(lambda self: self.data.get("kind"))


class _Models:
    def __init__(self, c: "AnviraRuntime"):
        self._c = c

    def list(self, installed: bool = False) -> list[dict[str, Any]]:
        return self._c._http.json("GET", "/v1/models", params={"installed": "true" if installed else None})["models"]

    def installed(self) -> list[dict[str, Any]]:
        return self._c._http.json("GET", "/v1/models/installed")["models"]

    def catalog(self, q: str | None = None) -> list[dict[str, Any]]:
        return self._c._http.json("GET", "/v1/models/catalog", params={"q": q})["models"]

    def search(self, q: str, limit: int = 20) -> list[dict[str, Any]]:
        return self._c._http.json("GET", "/v1/models/search", params={"q": q, "limit": limit})["models"]

    def recommended(self, limit: int = 5) -> list[dict[str, Any]]:
        return self._c._http.json("GET", "/v1/models/recommended", params={"limit": limit})["models"]

    def active(self) -> dict[str, Any]:
        return self._c._http.json("GET", "/v1/models/active")

    def get(self, model_id: str) -> dict[str, Any]:
        return self._c._http.json("GET", f"/v1/models/{model_id}")

    def compatibility(self, model_id: str) -> dict[str, Any]:
        return self._c._http.json("GET", f"/v1/models/{model_id}/compatibility")

    def use(self, model_id: str, wait_s: float = 120.0) -> dict[str, Any]:
        return self._c._http.json("POST", "/v1/models/select", {"id": model_id, "wait_s": wait_s},
                                  timeout=wait_s + 40)

    def install(self, model: str | None = None, *, url: str | None = None, file: str | None = None,
                dir: str | None = None, wait: bool = False) -> Job:
        """Start a model download. Needs the ``models.manage`` permission (user-granted)."""
        body = {k: v for k, v in {"model": model, "url": url, "file": file, "dir": dir}.items() if v}
        job = Job(self._c, self._c._http.json("POST", "/v1/models/install", body)["job"])
        return job.wait(3600) if wait else job

    def remove(self, model_id: str, *, delete_file: bool | None = None, confirm: bool = False) -> dict[str, Any]:
        return self._c._http.json("POST", "/v1/models/remove",
                                  {"id": model_id, "delete_file": delete_file, "confirm": confirm})

    def register(self, path: str) -> dict[str, Any]:
        """Tell the runtime where a model (file) or a models folder already lives. Nothing is copied or downloaded;
        the model becomes usable by every Anvira app."""
        return self._c._http.json("POST", "/v1/models/register", {"path": path})

    def discovered(self) -> dict[str, Any]:
        """Model folders the runtime found from installed Anvira apps."""
        return self._c._http.json("GET", "/v1/models/discovered")

    def link(self, path: str, model_id: str | None = None) -> dict[str, Any]:
        return self._c._http.json("POST", "/v1/models/link", {"path": path, "id": model_id})

    def hardware(self, refresh: bool = False) -> dict[str, Any]:
        return self._c._http.json("GET", "/v1/hardware", params={"refresh": "true" if refresh else None})

    def storage(self) -> dict[str, Any]:
        return self._c._http.json("GET", "/v1/models/storage")

    def providers(self) -> list[dict[str, Any]]:
        return self._c._http.json("GET", "/v1/providers")["providers"]

    def add_provider(self, *, base_url: str, model: str, api_key: str = "", label: str | None = None) -> dict[str, Any]:
        return self._c._http.json("POST", "/v1/providers", {"base_url": base_url, "model": model,
                                                            "api_key": api_key, "label": label})


class _Orcha:
    def __init__(self, c: "AnviraRuntime"):
        self._c = c

    def run(self, task: str, *, graph: str = "default", wait: bool = False, timeout: float = 900.0, **options: Any) -> Job:
        """Run an orchestration job. ``options`` are ORCHA run options (reasoning, workspace_roots, ...)."""
        body = {"task": task, "graph": graph, **options}
        job = Job(self._c, self._c._http.json("POST", "/v1/orcha/run", body)["job"])
        return job.wait(timeout) if wait else job

    def status(self) -> dict[str, Any]:
        return self._c._http.json("GET", "/v1/orcha/status")

    def jobs(self, state: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        jobs = self._c._http.json("GET", "/v1/jobs", params={"state": state, "limit": 200})["jobs"]
        return [j for j in jobs if j["kind"] in ("orcha.run", "task", "agent.run")][:limit]   # every ORCHA-backed job kind

    def cancel(self, job_id: str) -> Job:
        return Job(self._c, self._c._http.json("POST", f"/v1/jobs/{job_id}/cancel")["job"])


class _Jobs:
    def __init__(self, c: "AnviraRuntime"):
        self._c = c

    def get(self, job_id: str) -> Job:
        return Job(self._c, self._c._http.json("GET", f"/v1/jobs/{job_id}")["job"])

    def list(self, state: str | None = None, kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return self._c._http.json("GET", "/v1/jobs", params={"state": state, "kind": kind, "limit": limit})["jobs"]

    def cancel(self, job_id: str) -> Job:
        return Job(self._c, self._c._http.json("POST", f"/v1/jobs/{job_id}/cancel")["job"])


class _Memory:
    def __init__(self, c: "AnviraRuntime"):
        self._c = c

    def store(self, content: str, *, title: str | None = None, type: str = "note", tags: list[str] | None = None,
              importance: int = 3, scope: str | None = None, workspace: str | None = None,
              extra: dict[str, Any] | None = None) -> dict[str, Any]:
        body = {k: v for k, v in dict(content=content, title=title, type=type, tags=tags, importance=importance,
                                      scope=scope, workspace=workspace, extra=extra).items() if v is not None}
        return self._c._http.json("POST", "/v1/memory", body)

    def search(self, query: str, *, limit: int = 10, scope: str | None = None, workspace: str | None = None,
               tags: list[str] | None = None, type: str | None = None) -> list[dict[str, Any]]:
        return self._c._http.json("GET", "/v1/memory/search", params={
            "q": query, "limit": limit, "scope": scope, "workspace": workspace, "tag": tags, "type": type})["items"]

    def list(self, limit: int = 20, **filters: Any) -> list[dict[str, Any]]:
        return self._c._http.json("GET", "/v1/memory", params={"limit": limit, **filters})["items"]

    def get(self, memory_id: str) -> dict[str, Any]:
        return self._c._http.json("GET", f"/v1/memory/{memory_id}")

    def delete(self, memory_id: str) -> dict[str, Any]:
        return self._c._http.json("DELETE", f"/v1/memory/{memory_id}")


class _Context:
    """Per-app document index (chunk + BM25 retrieval) for grounding."""

    def __init__(self, c: "AnviraRuntime"):
        self._c = c

    def put(self, collection: str, doc_id: str, text: str, *, title: str | None = None,
            metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._c._http.json("PUT", f"/v1/context/{collection}/documents/{doc_id}",
                                  {"text": text, "title": title, "metadata": metadata})

    def search(self, collection: str, query: str, *, limit: int = 8, doc_ids: list[str] | None = None) -> list[dict[str, Any]]:
        return self._c._http.json("POST", f"/v1/context/{collection}/search",
                                  {"query": query, "limit": limit, "doc_ids": doc_ids})["items"]

    def collections(self) -> list[dict[str, Any]]:
        return self._c._http.json("GET", "/v1/context")["collections"]

    def documents(self, collection: str) -> list[dict[str, Any]]:
        return self._c._http.json("GET", f"/v1/context/{collection}/documents")["documents"]

    def delete(self, collection: str, doc_id: str | None = None) -> dict[str, Any]:
        path = f"/v1/context/{collection}" + (f"/documents/{doc_id}" if doc_id else "")
        return self._c._http.json("DELETE", path)


class _Resources:
    """Shared resources: private by default, shared with named apps on the owner's say-so, referenced (never copied).

    An app creates a resource pointing at content it already holds, shares it with other apps, and other apps read or
    search it *through the runtime*. Global visibility is the user's decision only (``anvira context share --global``).
    """

    def __init__(self, c: "AnviraRuntime"):
        self._c = c

    def create(self, title: str, *, type: str = "document", collection: str | None = None, doc_ids: list[str] | None = None,
               text: str | None = None, path: str | None = None, workspace: str | None = None,
               metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        kind = "context" if collection else "text" if text is not None else "file"
        body = dict(type=type, title=title, kind=kind, collection=collection, doc_ids=doc_ids, text=text, path=path,
                    workspace=workspace, metadata=metadata)
        return self._c._http.json("POST", "/v1/resources", {k: v for k, v in body.items() if v is not None})

    def list(self, *, type: str | None = None, workspace: str | None = None, owned: bool = False) -> list[dict[str, Any]]:
        return self._c._http.json("GET", "/v1/resources", params={"type": type, "workspace": workspace,
                                                                    "owned": "true" if owned else None})["resources"]

    def get(self, resource_id: str) -> dict[str, Any]:
        return self._c._http.json("GET", f"/v1/resources/{resource_id}")

    def update(self, resource_id: str, *, title: str | None = None, metadata: dict[str, Any] | None = None,
               workspace: str | None = None, text: str | None = None) -> dict[str, Any]:
        """Owner: rename / re-tag / move to a workspace. Writers may only replace inline ``text`` of a kind=text resource."""
        body = dict(title=title, metadata=metadata, workspace=workspace, text=text)
        return self._c._http.json("PATCH", f"/v1/resources/{resource_id}", {k: v for k, v in body.items() if v is not None})

    def resolve(self, ref: str) -> dict[str, Any]:
        """Look up a ``runtime://res_...`` reference (only if this app is allowed to see it)."""
        return self._c._http.json("GET", "/v1/resources/resolve", params={"ref": ref})

    def read(self, resource_id: str, doc: str | None = None) -> dict[str, Any]:
        return self._c._http.json("GET", f"/v1/resources/{resource_id}/read", params={"doc": doc})

    def write(self, resource_id: str, doc_id: str, text: str, *, title: str | None = None) -> dict[str, Any]:
        return self._c._http.json("PUT", f"/v1/resources/{resource_id}/documents/{doc_id}", {"text": text, "title": title})

    def search(self, query: str, *, resources: list[str] | None = None, workspace: str | None = None,
               types: list[str] | None = None, limit: int = 8) -> dict[str, Any]:
        body = dict(query=query, resources=resources, workspace=workspace, types=types, limit=limit)
        return self._c._http.json("POST", "/v1/resources/search", {k: v for k, v in body.items() if v is not None})

    def share(self, resource_id: str, apps: list[str], *, access: str = "read") -> dict[str, Any]:
        return self._c._http.json("POST", f"/v1/resources/{resource_id}/share", {"with": apps, "access": access})

    def revoke(self, resource_id: str, apps: list[str] | None = None) -> dict[str, Any]:
        return self._c._http.json("POST", f"/v1/resources/{resource_id}/revoke", {"with": apps})

    def permissions(self, resource_id: str) -> dict[str, Any]:
        return self._c._http.json("GET", f"/v1/resources/{resource_id}/permissions")

    def request_access(self, resource_id: str, *, access: str = "read", reason: str = "") -> dict[str, Any]:
        return self._c._http.json("POST", f"/v1/resources/{resource_id}/request", {"access": access, "reason": reason})

    def requests(self, state: str | None = "pending") -> list[dict[str, Any]]:
        return self._c._http.json("GET", "/v1/resources/requests", params={"state": state or "all"})["requests"]

    def decide(self, request_id: str, approve: bool) -> dict[str, Any]:
        return self._c._http.json("POST", f"/v1/resources/requests/{request_id}/{'approve' if approve else 'deny'}")

    def delete(self, resource_id: str, *, purge: bool = False) -> dict[str, Any]:
        return self._c._http.json("DELETE", f"/v1/resources/{resource_id}", params={"purge": "true" if purge else None})

    def audit(self, limit: int = 50, resource: str | None = None) -> list[dict[str, Any]]:
        return self._c._http.json("GET", "/v1/resources/audit", params={"limit": limit, "resource": resource})["events"]

    def workspaces(self) -> list[dict[str, Any]]:
        return self._c._http.json("GET", "/v1/workspaces")["workspaces"]

    def create_workspace(self, name: str) -> dict[str, Any]:
        return self._c._http.json("POST", "/v1/workspaces", {"name": name})

    def share_workspace(self, name: str, apps: list[str], *, access: str = "read") -> dict[str, Any]:
        return self._c._http.json("POST", f"/v1/workspaces/{name}/share", {"with": apps, "access": access})


class AnviraRuntime:
    """A connected, authenticated handle to the shared Anvira Runtime."""

    def __init__(self, http: Http, info: RuntimeInfo, app_id: str, env: dict[str, str] | None = None):
        self._http, self.info, self.app_id, self._env = http, info, app_id, env
        self.models, self.orcha, self.jobs = _Models(self), _Orcha(self), _Jobs(self)
        self.memory, self.context = _Memory(self), _Context(self)
        self.resources = _Resources(self)
        self._lease: dict[str, Any] | None = None
        self._lease_pid: int | None = None
        self._closed = threading.Event()
        self._recover_lock = threading.Lock()
        self._auto_start, self._idle_grace_s, self._on_status = True, None, None

    # ------------------------------------------------------------------ lifecycle
    def _start_lease(self) -> None:
        """Tell the runtime this app is open, and keep telling it (heartbeat) until :meth:`close`."""
        self._acquire_lease()
        self._lease_pid = self.info.pid

        def beat() -> None:
            while not self._closed.wait(max(1.0, (self._lease or {}).get("ttl_s", 45.0) / 3)):
                lid = self._lease["id"]
                try:
                    self._http.json("POST", f"/v1/leases/{lid}/heartbeat", timeout=10)
                except AnviraError as exc:
                    if exc.code == "lease_not_found" and self._lease["id"] == lid:   # not already replaced by a recovery
                        try:
                            self._acquire_lease()
                        except AnviraError:
                            pass
                except Exception:  # noqa: BLE001 - a heartbeat must never crash the host app
                    pass
        threading.Thread(target=beat, name=f"anvira-lease-{self.app_id}", daemon=True).start()
        atexit.register(self.close)

    def _acquire_lease(self) -> None:
        out = self._http.json("POST", "/v1/leases", {}, timeout=10)
        self._lease = {**out["lease"], "mode": out.get("mode")}

    def _recover(self) -> str | None:
        """The runtime is gone (an on-demand runtime stopped while this app stayed open): start it again."""
        if self._closed.is_set() or not self._auto_start:
            return None
        with self._recover_lock:
            info = probe(self._env)
            if not (info.running and info.ready):
                (self._on_status or (lambda _m: None))("Starting runtime...")
                info = bootstrap.start_runtime(env=self._env, on_status=self._on_status, auto_stop=True,
                                               idle_grace_s=self._idle_grace_s)
            self.info = info
            base = info.base_url
            if base and self._lease is not None and info.pid != self._lease_pid:   # a NEW runtime process: register presence there
                self._lease_pid = info.pid
                try:
                    out = Http(base, self._http.token, 10).json("POST", "/v1/leases", {})
                    self._lease = {**out["lease"], "mode": out.get("mode")}
                except AnviraError:
                    pass
            return base

    def close(self) -> None:
        """This app is closing: release the lease so an on-demand runtime can shut down when nobody else needs it."""
        if self._closed.is_set():
            return
        self._closed.set()
        lease = self._lease
        if lease:
            try:
                Http(self._http.base_url, self._http.token, 5).json("DELETE", f"/v1/leases/{lease['id']}", timeout=5)
            except Exception:  # noqa: BLE001 - runtime already gone: nothing to release
                pass

    def __enter__(self) -> "AnviraRuntime":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def lifecycle(self) -> dict[str, Any]:
        """Mode (on-demand / persistent), open leases and the idle countdown."""
        return self._http.json("GET", "/v1/lifecycle")

    # ---------------------------------------------------------------- detection
    @staticmethod
    def detect(env: dict[str, str] | None = None) -> RuntimeInfo:
        """Is the runtime installed / running? Never raises, never installs anything."""
        return probe(env)

    @classmethod
    def connect(cls, app_id: str, *, name: str | None = None, permissions: list[str] | None = None,
                require_api: int = SDK_API_VERSION, min_version: str | None = None, auto_start: bool = True,
                install: Callable[[RuntimeInfo], bool] | None = None, source: str | None = None,
                env: dict[str, str] | None = None, timeout: float = 30.0,
                on_status: Callable[[str], None] | None = None, keep_alive: bool = True,
                idle_grace_s: float | None = None) -> "AnviraRuntime":
        """Detect -> (install with consent) -> start on demand -> authenticate -> verify compatibility -> hold a lease.

        The runtime is active only while an app is open: ``connect()`` starts it if needed (in on-demand mode), keeps
        a heartbeat lease while this handle is open, and :meth:`close` releases it so the runtime can stop by itself
        when no other app needs it. ``keep_alive=False`` skips the lease (short-lived scripts).

        ``install`` is called with the detection result when the runtime is missing; return
        ``True`` only after the *user* agreed. If omitted, a missing runtime raises
        :class:`RuntimeNotInstalled` and nothing is ever installed silently.
        """
        say = on_status or (lambda _m: None)
        info = probe(env)
        if not info.running:
            if not info.installed:
                if install is None:
                    raise RuntimeNotInstalled("runtime_not_installed", "Anvira Runtime is required but not installed.",
                                              hint="Ask the user, then install it (see INSTALLATION.md) or pass install=... to connect().")
                if not install(info):
                    raise InstallDeclined("install_declined", "Anvira Runtime is required. Installation was declined.")
                say("Installing Anvira Runtime...")
                bootstrap.install_runtime(source=source, env=env, on_status=say)
                info = probe(env)
            if not info.running:
                if not auto_start:
                    raise RuntimeNotRunning("runtime_not_running", "Anvira Runtime is installed but not running.",
                                            hint="Run `anvira runtime start`.")
                say("Starting runtime...")
                info = bootstrap.start_runtime(env=env, on_status=say, auto_stop=True, idle_grace_s=idle_grace_s)
        if info.running and not info.ready:
            say("Waiting for the runtime to finish starting...")
            info = bootstrap.wait_ready(env, 120.0, say)
        if info.api_version != require_api:
            raise IncompatibleRuntime(
                "incompatible_runtime",
                f"This app needs Runtime API v{require_api} but the installed runtime speaks v{info.api_version} "
                f"(runtime {info.runtime_version}).",
                hint="Update the runtime: `anvira runtime update`." if (info.api_version or 0) < require_api
                else "Update this application to a version that supports the newer runtime API.")
        if min_version and _semver(info.runtime_version or "0") < _semver(min_version):
            raise IncompatibleRuntime("incompatible_runtime",
                                      f"This app needs Anvira Runtime >= {min_version}; found {info.runtime_version}.",
                                      hint="Update the runtime: `anvira runtime update`.")
        token = cls._app_token(info, app_id, name, permissions, env, timeout)
        rt = cls(Http(info.base_url or "", token, timeout), info, app_id, env)
        rt._auto_start, rt._idle_grace_s, rt._on_status = auto_start, idle_grace_s, on_status
        rt._http.recover = rt._recover
        if keep_alive:
            rt._start_lease()
        return rt

    @staticmethod
    def _app_token(info: RuntimeInfo, app_id: str, name: str | None, permissions: list[str] | None,
                   env: dict[str, str] | None, timeout: float) -> str:
        dirs = runtime_dirs(env)
        token_file = dirs["state"] / "app-tokens" / f"{app_id}.token"
        if token_file.exists():
            return token_file.read_text(encoding="utf-8").strip()
        reg = read_token("register", env)
        if not reg:
            raise AnviraError("no_registration_token", "Cannot register with the runtime (registration token unreadable).",
                              hint="Run the app as the same OS user that installed the runtime.")
        out = Http(info.base_url or "", None, timeout).json(
            "POST", "/v1/apps/register", {"app_id": app_id, "name": name, "permissions": permissions or []},
            headers={"X-Anvira-Register-Token": reg})
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(out["token"], encoding="utf-8")
        try:
            os.chmod(token_file, 0o600)
        except OSError:
            pass
        return out["token"]

    # ------------------------------------------------------------------ basics
    def health(self) -> dict[str, Any]:
        return self._http.json("GET", "/health")

    def version(self) -> dict[str, Any]:
        return self._http.json("GET", "/version")

    def status(self) -> dict[str, Any]:
        return self._http.json("GET", "/v1/status")

    def me(self) -> dict[str, Any]:
        """This app's identity and granted permissions."""
        return self._http.json("GET", "/v1/apps/me")

    def has_capability(self, name: str) -> bool:
        return name in (self.info.capabilities or self.version().get("capabilities", []))

    # -------------------------------------------------------------------- chat
    def chat(self, messages: list[dict[str, Any]], *, stream: bool = False, memory: dict[str, Any] | None = None,
             **options: Any):
        """Chat with the active model. With ``stream=True`` returns an iterator of text deltas."""
        body = {"messages": messages, "stream": stream, **({"memory": memory} if memory else {}), **options}
        if not stream:
            return self._http.json("POST", "/v1/chat", body, timeout=660)
        return self._stream_chat(body)

    def _stream_chat(self, body: dict[str, Any]) -> Iterator[str]:
        for event, data in self._http.sse("POST", "/v1/chat", body):
            if event == "error":
                raise AnviraError.from_response(502, json.loads(data))
            if event != "message" or data.strip() == "[DONE]":
                continue
            try:
                delta = json.loads(data)["choices"][0]["delta"].get("content")
            except (ValueError, KeyError, IndexError):
                continue
            if delta:
                yield delta

    def chat_text(self, messages: list[dict[str, Any]], **options: Any) -> str:
        return self.chat(messages, **options)["choices"][0]["message"]["content"]

    # ---------------------------------------------------------------- tasks
    def task(self, task: str, *, wait: bool = True, **options: Any) -> Job:
        job = Job(self, self._http.json("POST", "/v1/task", {"task": task, "wait": wait, **options},
                                        timeout=(options.get("timeout_s") or 900) + 30)["job"])
        return job

    def agent_run(self, task: str, *, workspace_roots: list[str] | None = None, wait: bool = False, **options: Any) -> Job:
        body = {"task": task, "workspace_roots": workspace_roots, **options}
        job = Job(self, self._http.json("POST", "/v1/agent/run", {k: v for k, v in body.items() if v is not None})["job"])
        return job.wait() if wait else job

    def __repr__(self) -> str:
        return f"<AnviraRuntime app={self.app_id} runtime={self.info.runtime_version} api=v{self.info.api_version}>"
