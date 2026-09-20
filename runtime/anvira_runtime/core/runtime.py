"""RuntimeCore: the composition root.

Owns the supervisor, the AICL bus, the model manager, the memory/context
services and the job registry, and exposes capability-level operations. The
HTTP API is a thin layer over this class; the CLI and SDK never bypass it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import httpx

from ..config.paths import RuntimeLayout
from ..config.settings import RuntimeConfig
from ..models.manager import ModelManager
from ..process import infra
from ..process.supervisor import Supervisor, free_port
from ..security.secrets import AppRegistry, Principal, SecretStore, redact
from ..version import API_VERSION, CAPABILITIES, RUNTIME_VERSION
from .bus import AICLBus, BusUnavailable
from .clients import NomiClient, OrchaClient
from .context import ContextIndex
from .errors import RuntimeApiError, no_model, service_unavailable
from .jobs import Job, JobRegistry
from .lifecycle import Lifecycle
from .memory import MemoryService
from .resources import ResourceService

GRAPHS = ("default", "research", "multi_agent")
_NOTHING_RELEVANT = ("The user's own notes and documents contain NOTHING relevant to this question. "
                     "Do not answer it from general knowledge. Reply only: I could not find that in your notes.")
# What an agent gets when the app does not choose: read/write/search files inside its workspace roots. Anything that can run
# programs needs the user's explicit `orcha.exec` grant for that app.
DEFAULT_AGENT_CAPABILITIES = ("filesystem", "workspace", "search")
EXEC_CAPABILITIES = ("terminal", "git")
ORCHA_RUN_FIELDS = (
    "graph", "mode", "access_mode", "allow_tools", "require_approval", "system_prompt", "max_cost",
    "max_iterations", "workspace_roots", "tools", "allow_rules", "deny_rules", "ask_rules", "capabilities",
    "reasoning", "messages", "prompt_parts")


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


def setup_logging(layout: RuntimeLayout, level: str = "INFO", console: bool = False) -> logging.Logger:
    logger = logging.getLogger("anvira.runtime")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    layout.logs_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(layout.runtime_log, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    fh.addFilter(_RedactFilter())
    logger.addHandler(fh)
    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        ch.addFilter(_RedactFilter())
        logger.addHandler(ch)
    logger.propagate = False
    return logger


class RuntimeCore:
    def __init__(self, layout: RuntimeLayout, config: RuntimeConfig, *, console_log: bool = False,
                 orcha_transport: httpx.AsyncBaseTransport | None = None,
                 auto_stop: bool | None = None, idle_grace_s: float | None = None):
        self.layout, self.config = layout.ensure(), config
        self.logger = setup_logging(layout, config.get("logging.level"), console_log)
        self.started_at = time.time()
        self.ready = False
        self.secrets = SecretStore(layout)
        self.apps = AppRegistry(layout, self.secrets)
        self.supervisor = Supervisor(
            layout.children_file, health_interval_s=config.get("supervisor.health_interval_s"),
            restart_limit=config.get("supervisor.restart_limit"),
            restart_window_s=config.get("supervisor.restart_window_s"), log=self.log)
        self.jobs = JobRegistry(layout.state_dir / "jobs.json")
        self.ports: dict[str, int | None] = {"orcha": None, "nomi": None}
        self.orcha = OrchaClient(lambda: self._service_base("orcha"), self.secrets.get("orcha_token"),
                                 transport=orcha_transport)
        self.nomi = NomiClient(lambda: self._service_base("nomi"), self.secrets.get("nomi_password"))
        self.memory = MemoryService(self.nomi, config.get("memory.default_scope"))
        self.context = ContextIndex(layout.data_dir / "context" / "context.db")      # opens on first use
        self.resources = ResourceService(layout.data_dir / "context" / "resources.db", self.context)
        self.models = ModelManager(layout, config, self.supervisor, on_active_changed=self._sync_orcha_model,
                                   log=self.log)
        self.bus: AICLBus | None = None
        self.bus_error: str | None = None
        self.startup_errors: dict[str, str] = {}
        self._reconcile: asyncio.Task | None = None
        self._autostart: asyncio.Task | None = None
        self._synced_signature: str | None = None
        self._chat_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=None))
        self.located: dict[str, infra.Located] = {}
        self.active_streams = 0
        self.lifecycle = Lifecycle(
            auto_stop=config.get("lifecycle.auto_stop") if auto_stop is None else auto_stop,
            idle_grace_s=config.get("lifecycle.idle_grace_s") if idle_grace_s is None else idle_grace_s,
            lease_ttl_s=config.get("lifecycle.lease_ttl_s"), busy=self.busy_count, log=self.log)

    def busy_count(self) -> int:
        """Work that must finish before an on-demand runtime may stop: ORCHA runs, downloads, model loads, open streams."""
        jobs = self.jobs.counts()
        loading = 1 if self.models._loading_id else 0
        return jobs.get("running", 0) + jobs.get("queued", 0) + self.active_streams + loading

    # ------------------------------------------------------------------ logging
    def log(self, level: str, message: str) -> None:
        self.logger.log(getattr(logging, level.upper(), logging.INFO), message)

    def _service_base(self, name: str) -> str | None:
        st = self.supervisor.services.get(name)
        if st is None or st.state not in ("running", "degraded") or not self.ports.get(name):
            return None
        return f"http://127.0.0.1:{self.ports[name]}"

    # ------------------------------------------------------------------- start
    async def start(self) -> None:
        self.log("INFO", f"Anvira Runtime {RUNTIME_VERSION} starting (api v{API_VERSION}) home={self.layout.home}")
        if self.config.load_error:
            self.log("ERROR", f"configuration problem, using defaults: {self.config.load_error}")
        await self.supervisor.reap_stale()
        self._register_bus()
        await asyncio.gather(self._start_service("nomi"), self._start_service("orcha"))
        self.supervisor.start_monitor()
        self.lifecycle.start()
        self._reconcile = asyncio.create_task(self._reconcile_loop())
        if self.config.get("models.autostart_active") and self.models.store.active_id():
            self._autostart = asyncio.create_task(self._autostart_model())
        self.ready = True
        self.log("INFO", "runtime started" + (f" (degraded: {sorted(self.startup_errors)})" if self.startup_errors else ""))

    async def _start_service(self, name: str) -> None:
        if not self.config.get(f"services.{name}.enabled"):
            return
        loc = infra.locate_service(name, self.layout, self.config)
        self.located[name] = loc
        if loc.mode == "missing":
            self.startup_errors[name] = loc.detail
            self.log("ERROR", f"{name}: {loc.detail}")
            return
        if loc.mode == "source":
            missing = infra.missing_modules(infra.ORCHA_MODULES if name == "orcha" else infra.NOMI_MODULES)
            if missing and infra.python_for_services(self.config) == sys.executable:
                self.startup_errors[name] = f"missing Python packages: {', '.join(missing)}"
                self.log("ERROR", f"{name}: {self.startup_errors[name]}")
                return
        port = infra.resolve_port(self.config, name)
        self.ports[name] = port
        builder = infra.build_orcha_spec if name == "orcha" else infra.build_nomi_spec
        spec = builder(self.layout, self.config, self.secrets, port, loc)
        self.supervisor.add(spec)
        try:
            await self.supervisor.start(name)
        except RuntimeError as exc:
            self.startup_errors[name] = str(exc)
            self.log("ERROR", f"{name} failed to start: {exc}")

    async def _autostart_model(self) -> None:
        try:
            await self.models.ensure_ready(wait_s=900)
            await self._sync_orcha_model(self.models.active_record())
        except Exception as exc:  # noqa: BLE001 - must not kill the runtime
            self.log("ERROR", f"active model failed to autostart: {exc}")

    async def stop(self) -> None:
        self.log("INFO", "runtime stopping")
        self.lifecycle.stop()
        for t in (self._reconcile, self._autostart):
            if t:
                t.cancel()
        await self.jobs.shutdown()
        await self.supervisor.stop_all()
        for closer in (self.orcha.aclose(), self.nomi.aclose(), self.models.aclose(), self._chat_client.aclose()):
            try:
                await closer
            except Exception:  # noqa: BLE001
                pass
        self.resources.close()
        self.context.close()
        self.log("INFO", "runtime stopped")

    # --------------------------------------------------------------- AICL bus
    def _register_bus(self) -> None:
        try:
            self.bus = AICLBus(infra.locate_aicl(self.layout))
        except BusUnavailable as exc:
            self.bus, self.bus_error = None, str(exc)
            self.log("ERROR", f"AICL bus unavailable: {exc}")
            return
        bus = self.bus
        p = self._principal_from

        async def orcha_h(action: str, args: dict, ctx) -> Any:
            if action == "start_run":
                return await self.orcha.start_run(args["body"])
            if action == "get_run":
                return await self.orcha.get_run(args["run_id"])
            if action == "cancel_run":
                return await self.orcha.cancel_run(args["run_id"])
            if action == "health":
                return await self.orcha.health()
            if action == "runs":
                return await self.orcha.list_runs()
            if action == "sync_model":
                return await self._do_sync_orcha_model()
            raise RuntimeApiError("unknown_action", f"orcha.{action}", 400)

        async def nomi_h(action: str, args: dict, ctx) -> Any:
            pr = p(args.pop("_principal"))
            m = self.memory
            if action == "store":
                return await m.store(pr, **args)
            if action == "search":
                return await m.search(pr, **args)
            if action == "get":
                return await m.get(pr, args["id"])
            if action == "delete":
                return await m.delete(pr, args["id"])
            if action == "status":
                return await m.status()
            raise RuntimeApiError("unknown_action", f"nomi.{action}", 400)

        async def models_h(action: str, args: dict, ctx) -> Any:
            if action == "target":
                await self.models.ensure_ready()
                return self.models.inference_target()
            if action == "select":
                return await self.models.select(args["id"], args.get("wait_s", 120.0))
            if action == "status":
                return self.models.status()
            raise RuntimeApiError("unknown_action", f"models.{action}", 400)

        async def context_h(action: str, args: dict, ctx) -> Any:
            app = args.pop("app")
            c = self.context
            if action == "put":
                return await c.put(app, **args)
            if action == "search":
                return await c.search(app, **args)
            if action == "delete":
                return await c.delete(app, **args)
            if action == "collections":
                return await c.collections(app)
            if action == "documents":
                return await c.documents(app, args["collection"])
            raise RuntimeApiError("unknown_action", f"context.{action}", 400)

        bus.register("orcha", orcha_h, ["orcha.run", "orcha.cancel", "orcha.status"])
        bus.register("nomi", nomi_h, ["memory.store", "memory.search", "memory.delete"])
        bus.register("models", models_h, ["models.select", "models.target"])
        bus.register("context", context_h, ["context.index", "context.search"])

    @staticmethod
    def _principal_from(d: dict[str, Any]) -> Principal:
        return Principal(d["kind"], d.get("app_id"), frozenset(d.get("permissions") or ()))

    @staticmethod
    def principal_payload(pr: Principal) -> dict[str, Any]:
        return {"kind": pr.kind, "app_id": pr.app_id, "permissions": sorted(pr.permissions)}

    async def call(self, module: str, action: str, payload: dict[str, Any] | None = None, *,
                   pr: Principal | None = None, timeout: float = 30.0, op: int | None = None) -> Any:
        """Dispatch over the AICL bus (falls back to a clear error if AICL is unavailable)."""
        if self.bus is None:
            raise RuntimeApiError("aicl_unavailable", f"The AICL bus is unavailable: {self.bus_error}", 503,
                                  hint="Run `anvira doctor`.")
        return await self.bus.call(module, action, payload, op=op, app_id=(pr.app_id if pr else None),
                                   timeout=timeout)

    # ------------------------------------------------------------- ORCHA sync
    async def _sync_orcha_model(self, _record: dict[str, Any] | None) -> None:
        if self._service_base("orcha") is None:
            return
        try:
            await self.call("orcha", "sync_model", timeout=120)
        except RuntimeApiError as exc:
            self.log("WARNING", f"could not sync active model to ORCHA: {exc.message}")

    async def _do_sync_orcha_model(self) -> dict[str, Any]:
        """Point ORCHA at the active model (or clear it) — idempotent."""
        rec = self.models.active_record()
        current = await self.orcha.local_models()
        if rec is None:
            for m in current:
                await self.orcha.remove_model(m["model"])
            self._synced_signature = None
            return {"synced": None}
        try:
            target = self.models.inference_target()
        except RuntimeApiError:
            return {"synced": None, "pending": True}
        sig = f"{target['model']}|{target['base_url']}"
        if sig == self._synced_signature and any(m["model"] == target["model"] for m in current):
            return {"synced": target["model"], "changed": False}
        for m in current:
            if m["model"] != target["model"]:
                await self.orcha.remove_model(m["model"])
        await self.orcha.set_primary_model(target["model"], target["base_url"], target["api_key"] or None,
                                           label=rec.get("name"))
        self._synced_signature = sig
        return {"synced": target["model"], "changed": True}

    async def _reconcile_loop(self) -> None:
        while True:
            await asyncio.sleep(5)
            try:
                if self._service_base("orcha") and self.models.active_record():
                    await self.call("orcha", "sync_model", timeout=60)
            except Exception as exc:  # noqa: BLE001
                self.log("DEBUG", f"reconcile: {exc}")

    # ------------------------------------------------------- on-demand model use
    async def ensure_model(self, requested: str | None, pr: Principal) -> None:
        """If a request names an installed model that is not the active one, make it active first.

        This is how a model added through ANY app (Anvira, Notes, Study, Dev) is used "whenever it is
        called": the runtime knows where it is, switches the single inference backend to it, and serves
        the request. Only one local model runs at a time; selection is global by design.
        """
        if not requested or requested == self.models.store.active_id():
            return
        pr.require("models.select")
        installed = {m["id"] for m in self.models.store.scan()} | {m["id"] for m in self.models.providers.list()}
        if requested not in installed:
            raise RuntimeApiError("model_not_found", f"Model '{requested}' is not installed.", 404,
                                  hint="Run `anvira model list`, or install it: `anvira model install <id>`.")
        await self.call("models", "select", {"id": requested, "wait_s": 300}, pr=pr, timeout=330)

    # --------------------------------------------------------------- ORCHA jobs
    @staticmethod
    def _agent_capabilities(pr: Principal, body: dict[str, Any]) -> None:
        """Give an agent run real tools (ORCHA's small-model gates only work with a tool executor), and keep command
        execution behind the user's per-app ``orcha.exec`` grant."""
        wanted = list(body.get("capabilities") or [])
        if not wanted:
            wanted = list(DEFAULT_AGENT_CAPABILITIES)
            if pr.has("orcha.exec"):
                wanted.append("terminal")
        RuntimeCore._guard_exec(pr, wanted)
        body["capabilities"] = wanted

    @staticmethod
    def _guard_exec(pr: Principal, capabilities: list[str]) -> None:
        risky = [c for c in capabilities if c in EXEC_CAPABILITIES]
        if risky and not pr.has("orcha.exec"):
            raise RuntimeApiError(
                "permission_denied", f"'{pr.app_id or 'caller'}' may not give an agent the {'/'.join(risky)} capability "
                "(it can run programs on this computer).", 403,
                hint=f"The user can allow it: anvira app grant {pr.app_id or '<app-id>'} orcha.exec")

    def _require_model_for_orcha(self) -> None:
        if self.models.active_record() is None and not self.config.get("orcha.allow_mock"):
            raise no_model(self.models._installed_ids())

    async def submit_orcha(self, pr: Principal, kind: str, request: dict[str, Any]) -> Job:
        pr.require("orcha.run")
        if self._service_base("orcha") is None:
            raise service_unavailable("orcha", self.startup_errors.get("orcha", ""))
        await self.ensure_model(request.get("model"), pr)
        self._require_model_for_orcha()
        query = (request.get("task") or request.get("query") or "").strip()
        if not query:
            raise RuntimeApiError("invalid_request", "'task' (or 'query') is required.", 400)
        if request.get("graph") not in (None, *GRAPHS):
            raise RuntimeApiError("invalid_graph", f"Unknown graph '{request.get('graph')}'.", 400,
                                  hint="graph must be one of: " + ", ".join(GRAPHS))
        body: dict[str, Any] = {"query": query, **{k: request[k] for k in ORCHA_RUN_FIELDS if request.get(k) is not None}}
        if kind == "agent.run":
            body.setdefault("allow_tools", True)
            self._agent_capabilities(pr, body)
        else:
            self._guard_exec(pr, body.get("capabilities") or [])
        if request.get("context"):                   # authorised shared context, resolved on demand, becomes a prompt part
            ctx_req = request["context"]
            rec = self.models.active_record()
            if rec is not None and rec["kind"] == "provider":        # local models are on this machine by definition
                private = self.models.providers.get_private(rec["provider_id"]) or {}
                if self.is_remote(private) and not request.get("allow_remote_context"):
                    raise self._remote_context_error(private)
            snippets = await self.resources.context_for(pr, query=str(ctx_req.get("query") or query),
                                                        resources=ctx_req.get("resources"), limit=int(ctx_req.get("limit", 5)))
            if snippets:
                text = "Context from the user's own data:\n" + "\n---\n".join(f"[{s['resource_title']}] {s['text']}" for s in snippets)
                body["prompt_parts"] = [*(body.get("prompt_parts") or []), {"name": "shared-context", "text": text, "priority": 40}]
            elif ctx_req.get("strict"):
                body["prompt_parts"] = [*(body.get("prompt_parts") or []), {"name": "shared-context", "text": _NOTHING_RELEVANT, "priority": 40}]
        job = self.jobs.create(kind, pr.app_id, {"graph": body.get("graph", "default")})
        timeout = float(request.get("timeout_s") or self.config.get("orcha.job_timeout_s"))

        async def work(j: Job) -> dict[str, Any]:
            started = await self.call("orcha", "start_run", {"body": body}, pr=pr, op=self.bus.ops.OP_EXECUTE if self.bus else None)
            run_id = started["run_id"]
            j.meta["orcha_run_id"] = run_id

            async def cancel_hook() -> None:
                try:
                    await self.call("orcha", "cancel_run", {"run_id": run_id}, pr=pr, op=self.bus.ops.OP_CANCEL if self.bus else None)
                except RuntimeApiError:
                    pass
            self.jobs.set_cancel_hook(j.id, cancel_hook)
            deadline = time.monotonic() + timeout
            delay, missing = 0.3, 0
            while True:
                if time.monotonic() > deadline:
                    await cancel_hook()
                    raise RuntimeApiError("timeout", f"The job exceeded {timeout:.0f}s and was cancelled.", 504)
                try:
                    run = await self.call("orcha", "get_run", {"run_id": run_id}, pr=pr)
                    missing = 0
                except RuntimeApiError as exc:
                    if exc.status == 404 and j.progress.get("seen_running"):
                        missing += 1
                        if missing > 5:
                            raise RuntimeApiError("run_lost", "ORCHA lost track of this run (it may have restarted).", 502) from exc
                        await asyncio.sleep(delay)
                        continue
                    raise
                status = run.get("status")
                if status == "running":
                    j.progress = {**j.progress, "seen_running": True, "steps": len(run.get("agent_steps") or []),
                                  "iterations": run.get("iterations")}
                elif status == "completed":
                    return self._summarize_run(run)
                elif status == "failed":
                    raise RuntimeApiError("orcha_run_failed", run.get("error") or "The run failed.", 500,
                                          details={"run_id": run_id})
                elif status == "cancelled":
                    raise asyncio.CancelledError()
                elif status == "pending_approval":
                    j.progress = {**j.progress, "approval": run.get("approval")}
                    return {**self._summarize_run(run), "status": "pending_approval"}
                await asyncio.sleep(delay)
                delay = min(1.0, delay * 1.3)

        return self.jobs.submit(job, work)

    @staticmethod
    def _summarize_run(run: dict[str, Any]) -> dict[str, Any]:
        return {"run_id": run.get("run_id"), "status": run.get("status"), "answer": run.get("answer", ""),
                "confidence": run.get("confidence"), "synthesized": run.get("synthesized"),
                "contributors": run.get("contributors") or [], "iterations": run.get("iterations"),
                "latency_s": run.get("latency_s"), "graph": run.get("graph_name"),
                "agent_steps": run.get("agent_steps") or [], "agent_tool_calls": run.get("agent_tool_calls") or [],
                "agent_completed": run.get("agent_completed")}

    async def wait_job(self, job: Job, timeout: float) -> Job:
        deadline = time.monotonic() + timeout
        while job.state in ("queued", "running") and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        return job

    # -------------------------------------------------------------------- chat
    async def chat(self, pr: Principal, body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Prepare a chat request. Returns (target, upstream_body) after model readiness and memory recall."""
        pr.require("chat")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) and "role" in m for m in messages):
            raise RuntimeApiError("invalid_request", "'messages' must be a non-empty list of {role, content}.", 400)
        await self.ensure_model(body.get("model"), pr)
        target = await self.call("models", "target", pr=pr, timeout=320)
        requested = body.get("model")
        if requested and requested not in (target["id"], target["model"]):
            raise RuntimeApiError("model_not_found", f"Model '{requested}' is not available.", 404,
                                  hint="Run `anvira model list`.")
        used: list[dict[str, Any]] = []
        ctx_used: list[str] = []
        warnings: list[str] = []
        recall = body.get("memory") or {}
        ctx_req = body.get("context") or None
        if (recall.get("recall") or ctx_req) and self.is_remote(target) and not body.get("allow_remote_context"):
            raise self._remote_context_error(target)
        if ctx_req:
            last = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            snippets = await self.resources.context_for(
                pr, query=str(ctx_req.get("query") or (last or {}).get("content", "")),
                resources=ctx_req.get("resources"), limit=int(ctx_req.get("limit", 5)))
            if snippets:
                block = "Use this context from the user's own data when it helps (cite the source titles):\n" + "\n---\n".join(
                    f"[{s['resource_title']}] {s['text']}" for s in snippets)
                messages = [{"role": "system", "content": block}, *messages]
                ctx_used = sorted({s["resource"] for s in snippets})
            elif ctx_req.get("strict"):        # nothing relevant: say so, so a small model does not improvise from general knowledge
                messages = [{"role": "system", "content": _NOTHING_RELEVANT}, *messages]
        if recall.get("recall"):
            last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            found_mem: list[dict[str, Any]] = []
            try:
                found = await self.call("nomi", "search", {
                    "_principal": self.principal_payload(pr), "query": str(last_user.get("content", "")) if last_user else "",
                    "limit": int(recall.get("limit", 5)), "scope": recall.get("scope"),
                    "workspace": recall.get("workspace")}, pr=pr)
                found_mem = found["items"]
            except RuntimeApiError as exc:
                warnings.append(f"memory recall skipped: {exc.message}")
            used.extend(found_mem)
            if found_mem:
                ctx = "Relevant things you remember about the user:\n" + "\n".join(
                    f"- {m['title']}: {m['content']}" for m in found_mem)
                messages = [{"role": "system", "content": ctx}, *messages]
        upstream = {k: v for k, v in body.items() if k in (
            "temperature", "top_p", "max_tokens", "stop", "seed", "presence_penalty", "frequency_penalty",
            "response_format", "tools", "tool_choice")}
        upstream.update({"model": target["model"], "messages": messages, "stream": bool(body.get("stream"))})
        target = {**target, "memories_used": [m["id"] for m in used], "context_used": ctx_used, "warnings": warnings}
        return target, upstream

    @staticmethod
    def is_remote(target: dict[str, Any]) -> bool:
        """True when the model endpoint is not on this machine (a cloud/remote provider)."""
        from urllib.parse import urlparse
        host = (urlparse(target.get("base_url", "")).hostname or "").lower()
        return host not in ("127.0.0.1", "localhost", "::1")

    @staticmethod
    def _remote_context_error(target: dict[str, Any]) -> RuntimeApiError:
        return RuntimeApiError(
            "remote_context_not_allowed",
            "This request would send local memory/context to a remote model provider.", 403,
            hint="Local data is never sent to a cloud model silently. Use a local model, or pass allow_remote_context=true "
                 "to confirm you (the user) accept sending it.")

    def _upstream_headers(self, target: dict[str, Any]) -> dict[str, str]:
        return {"Authorization": f"Bearer {target['api_key']}"} if target.get("api_key") else {}

    def _upstream_error(self, target: dict[str, Any], exc: Exception) -> RuntimeApiError:
        if isinstance(exc, httpx.ConnectError):
            code = "model_crashed" if target["kind"] == "local" else "provider_unreachable"
            return RuntimeApiError(code, f"The model backend for '{target['id']}' is not reachable.", 502,
                                   hint="Run `anvira model status` and `anvira runtime logs`.")
        if isinstance(exc, httpx.TimeoutException):
            return RuntimeApiError("model_timeout", "The model did not respond in time.", 504)
        return RuntimeApiError("model_error", f"{type(exc).__name__}: {exc}", 502)

    async def chat_complete(self, target: dict[str, Any], upstream: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._chat_client.post(f"{target['base_url']}/chat/completions", json=upstream,
                                                headers=self._upstream_headers(target), timeout=httpx.Timeout(30.0, read=600.0))
        except httpx.HTTPError as exc:
            raise self._upstream_error(target, exc) from exc
        if resp.status_code >= 400:
            raise RuntimeApiError("model_error", f"The model returned HTTP {resp.status_code}: {resp.text[:300]}", 502)
        data = resp.json()
        data["runtime"] = {"model": target["id"], "kind": target["kind"], "memories_used": target["memories_used"],
                           "context_used": target["context_used"], "warnings": target["warnings"]}
        return data

    async def chat_stream(self, target: dict[str, Any], upstream: dict[str, Any]) -> AsyncIterator[bytes]:
        """Yield OpenAI-style SSE bytes; upstream failures become a structured ``event: error`` frame."""
        meta = json.dumps({"runtime": {"model": target["id"], "kind": target["kind"],
                                       "memories_used": target["memories_used"], "context_used": target["context_used"],
                                       "warnings": target["warnings"]}})
        yield f"event: runtime\ndata: {meta}\n\n".encode()
        self.active_streams += 1
        try:
            async with self._chat_client.stream("POST", f"{target['base_url']}/chat/completions", json=upstream,
                                                headers=self._upstream_headers(target),
                                                timeout=httpx.Timeout(30.0, read=600.0)) as resp:
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode(errors="replace")[:300]
                    raise RuntimeApiError("model_error", f"The model returned HTTP {resp.status_code}: {text}", 502)
                async for chunk in resp.aiter_raw():
                    yield chunk
        except (httpx.HTTPError, RuntimeApiError) as exc:
            err = exc if isinstance(exc, RuntimeApiError) else self._upstream_error(target, exc)
            yield f"event: error\ndata: {json.dumps(err.to_dict())}\n\n".encode()
        finally:
            self.active_streams -= 1

    # ------------------------------------------------------------------ status
    def service_status(self) -> dict[str, Any]:
        sv = self.supervisor.status()
        out: dict[str, Any] = {}
        for name in ("orcha", "nomi"):
            if not self.config.get(f"services.{name}.enabled"):
                out[name] = {"state": "disabled"}
            elif name in sv:
                out[name] = sv[name]
            else:
                out[name] = {"state": "unavailable", "last_error": self.startup_errors.get(name)}
            loc = self.located.get(name)
            if loc:
                out[name]["source"] = {"mode": loc.mode, "path": str(loc.path) if loc.path else None}
        if self.bus:
            from .bus import native_available
            out["aicl"] = {"state": "running", "in_process": True, "version": self.bus.version,
                           "native_core": native_available(), "modules": list(self.bus.modules())}
        else:
            out["aicl"] = {"state": "unavailable", "last_error": self.bus_error}
        return out

    def health(self) -> dict[str, Any]:
        svc = self.service_status()
        bad = sorted(n for n, s in svc.items() if s.get("state") not in ("running", "disabled"))
        status = "starting" if not self.ready else ("ok" if not bad else "degraded")
        return {"status": status, "degraded": bad if self.ready else [],
                "runtime_version": RUNTIME_VERSION, "api_version": API_VERSION}

    def status(self) -> dict[str, Any]:
        return {
            **self.health(), "pid": os.getpid(), "started_at": self.started_at,
            "uptime_s": round(time.time() - self.started_at, 1),
            "services": self.service_status(), "model": self.models.status(),
            "jobs": self.jobs.counts(), "paths": self.layout.as_dict(), "capabilities": list(CAPABILITIES),
            "lifecycle": self.lifecycle.status(),
            "apps": len(self.apps.list()),
        }
