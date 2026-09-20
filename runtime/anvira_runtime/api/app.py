"""Runtime HTTP API (loopback only), versioned under ``/v1``.

Public (no token): ``GET /health`` and ``GET /version`` — they reveal no user
data and are what applications use to detect the runtime. Everything else
requires ``Authorization: Bearer <token>`` (an app token, or the owner token
used by the CLI) and checks the caller's granted permissions.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Callable

from fastapi import Body, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..config.settings import ConfigError, flat_keys
from ..core.errors import RuntimeApiError
from ..core.runtime import RuntimeCore
from ..diagnostics.doctor import run_checks
from ..diagnostics.logs import read_log
from ..process.supervisor import tail_file  # noqa: F401  (re-exported for tests)
from ..security.secrets import PERMISSIONS, AuthError, Principal
from ..version import API_REVISION, API_VERSION, CAPABILITIES, RUNTIME_VERSION, SERVICE_NAME

PUBLIC_PATHS = {"/health", "/version"}


def _err(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, "status": status, **extra}}, status_code=status)


def create_app(core: RuntimeCore, *, shutdown: Callable[[], None] | None = None) -> FastAPI:
    app = FastAPI(title="Anvira Runtime API", version=RUNTIME_VERSION, docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.state.core = core
    allowed_origins = list(core.config.get("api.allowed_origins"))
    if allowed_origins:
        app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["*"],
                           allow_headers=["Authorization", "Content-Type", "X-Anvira-Register-Token"])

    # ------------------------------------------------------------ guards / errors
    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]").lower()
        if host not in ("127.0.0.1", "localhost", "::1"):
            return _err(421, "invalid_host", "Requests must address the runtime as 127.0.0.1 or localhost.")
        origin = request.headers.get("origin")
        if origin and origin not in allowed_origins:
            return _err(403, "origin_not_allowed",
                        "Browser requests are not allowed. Call the runtime from your app's backend/main process, "
                        "or add the origin with `anvira config set api.allowed_origins`.")
        started = time.perf_counter()
        response = await call_next(request)
        if request.url.path not in PUBLIC_PATHS:
            core.log("DEBUG", f"{request.method} {request.url.path} -> {response.status_code} "
                              f"{(time.perf_counter() - started) * 1000:.0f}ms")
        return response

    @app.exception_handler(RuntimeApiError)
    async def _api_error(_: Request, exc: RuntimeApiError):
        if exc.status >= 500:
            core.log("ERROR", f"api error {exc.code}: {exc.message}")
        return JSONResponse(exc.to_dict(), status_code=exc.status)

    @app.exception_handler(AuthError)
    async def _auth_error(_: Request, exc: AuthError):
        headers = {"WWW-Authenticate": "Bearer"} if exc.status == 401 else None
        return JSONResponse({"error": {"code": exc.code, "message": exc.message, "status": exc.status}},
                            status_code=exc.status, headers=headers)

    @app.exception_handler(ConfigError)
    async def _config_error(_: Request, exc: ConfigError):
        return _err(400, "invalid_config", str(exc))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        return _err(400, "invalid_request", f"{loc}: {first.get('msg', 'invalid request')}".strip(": "))

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _err(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        core.log("ERROR", f"unhandled error on {request.method} {request.url.path}: {type(exc).__name__}: {exc}")
        return _err(500, "internal_error", "An unexpected error occurred. See `anvira runtime logs`.")

    def principal(request: Request) -> Principal:
        header = request.headers.get("authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else request.headers.get("x-anvira-token", "")
        if not token:
            raise AuthError("unauthorized", "Missing token. Send 'Authorization: Bearer <token>'.", 401)
        pr = core.apps.authenticate(token)
        if pr is None:
            raise AuthError("unauthorized", "Invalid or revoked token.", 401)
        return pr

    def need(request: Request, permission: str) -> Principal:
        pr = principal(request)
        pr.require(permission)
        return pr

    def as_app(pr: Principal, requested: str | None) -> str:
        """The app namespace to act in: an app is always itself; the owner may name any."""
        return (requested or "owner") if pr.kind == "owner" else (pr.app_id or "unknown")

    # ------------------------------------------------------------------ public
    @app.get("/health")
    async def health():
        return {**core.health(), "service": SERVICE_NAME}

    @app.get("/version")
    async def version():
        return {"service": SERVICE_NAME, "runtime_version": RUNTIME_VERSION, "api_version": API_VERSION,
                "api_revision": API_REVISION, "min_client_api": 1, "capabilities": list(CAPABILITIES)}

    # ------------------------------------------------------------------ status
    @app.get("/v1/status")
    async def status(request: Request):
        need(request, "orcha.read")
        return core.status()

    @app.get("/v1/runtime/config")
    async def get_config(request: Request):
        need(request, "config.read")
        return {"config": core.config.as_dict(), "models_dir": str(core.models.store.primary_dir),
                "keys": flat_keys(), "load_error": core.config.load_error}

    @app.put("/v1/runtime/config")
    async def set_config(request: Request, body: dict = Body(...)):
        need(request, "config.write")
        key = body.get("key")
        if not isinstance(key, str) or "value" not in body:
            raise RuntimeApiError("invalid_request", "Body must be {\"key\": ..., \"value\": ...}.", 400)
        if key == "models.models_dir":
            res = await core.models.set_models_dir(str(body["value"]), move=bool(body.get("move")))
            return {"key": key, "value": res["models_dir"], "restart_required": False, **res}
        value = core.config.set(key, body["value"])
        core.config.save()
        core.models.store.set_dirs(core.config.models_dir(), core.config.extra_model_dirs())
        return {"key": key, "value": value,
                "restart_required": key.startswith(("api.", "services.", "supervisor.", "logging."))}

    @app.get("/v1/runtime/logs")
    async def logs(request: Request, service: str = "runtime", lines: int = Query(100, ge=1, le=2000)):
        need(request, "runtime.admin")
        return read_log(core.layout, service, lines)

    @app.post("/v1/runtime/services/{name}/restart")
    async def restart_service(request: Request, name: str):
        need(request, "runtime.admin")
        if name not in ("orcha", "nomi"):
            raise RuntimeApiError("service_not_found", f"Unknown service '{name}'.", 404)
        if name not in core.supervisor.services:
            await core._start_service(name)
        else:
            await core.supervisor.restart(name)
            if name == "nomi":
                core.nomi.reset()
        return core.service_status()[name]

    @app.post("/v1/runtime/stop")
    async def stop(request: Request):
        need(request, "runtime.admin")
        if shutdown:
            asyncio.get_running_loop().call_later(0.2, shutdown)
        return {"stopping": True}

    @app.get("/v1/diagnostics")
    async def diagnostics(request: Request):
        need(request, "runtime.admin")
        return await asyncio.to_thread(run_checks, core.layout, core.config, core.status())

    # ------------------------------------------------------------------- apps
    @app.post("/v1/apps/register")
    async def register_app(request: Request, body: dict = Body(...)):
        supplied = request.headers.get("x-anvira-register-token", "")
        if not core.secrets.matches("register_token", supplied):
            raise AuthError("unauthorized", "Missing or invalid registration token.", 401)
        rec, token = core.apps.register(str(body.get("app_id", "")), body.get("name"),
                                        body.get("permissions") or [])
        core.log("INFO", f"registered app '{rec['app_id']}' requested={rec['requested']}")
        return {"app": rec, "token": token,
                "note": "Store this token; it is shown once. Permissions listed under 'requested' need user approval: "
                        "`anvira app grant <app_id> <permission>`."}

    @app.get("/v1/apps/me")
    async def me(request: Request):
        pr = principal(request)
        rec = core.apps.get(pr.app_id) if pr.app_id else None
        return {"kind": pr.kind, "app_id": pr.app_id, "permissions": sorted(pr.permissions),
                "app": rec}

    @app.get("/v1/apps")
    async def list_apps(request: Request):
        need(request, "apps.admin")
        return {"apps": core.apps.list(), "permissions": PERMISSIONS}

    @app.post("/v1/apps/{app_id}/grant")
    async def grant(request: Request, app_id: str, body: dict = Body(...)):
        need(request, "apps.admin")
        return core.apps.grant(app_id, body.get("permissions") or [])

    @app.post("/v1/apps/{app_id}/deny")
    async def deny(request: Request, app_id: str, body: dict = Body(...)):
        need(request, "apps.admin")
        return core.apps.deny(app_id, body.get("permissions") or [])

    @app.delete("/v1/apps/{app_id}")
    async def revoke(request: Request, app_id: str):
        need(request, "apps.admin")
        core.apps.revoke(app_id)
        return {"revoked": app_id}

    # ---------------------------------------------------------------- lifecycle
    core.lifecycle.on_idle = shutdown

    @app.post("/v1/leases", status_code=201)
    async def lease_acquire(request: Request, body: dict = Body(default={})):
        """An open app holds a lease and renews it; the runtime stays up while any lease (or any work) exists."""
        pr = principal(request)
        lease = core.lifecycle.acquire(pr.app_id or "cli", body.get("ttl_s"))
        return {"lease": lease, "heartbeat_every_s": round(lease["ttl_s"] / 3, 1), "mode": core.lifecycle.status()["mode"]}

    @app.post("/v1/leases/{lease_id}/heartbeat")
    async def lease_heartbeat(request: Request, lease_id: str):
        pr = principal(request)
        lease = core.lifecycle.heartbeat(lease_id, pr.app_id or "cli")
        if lease is None:
            raise RuntimeApiError("lease_not_found", "This lease expired or was released; acquire a new one.", 404)
        return {"lease": lease}

    @app.delete("/v1/leases/{lease_id}")
    async def lease_release(request: Request, lease_id: str):
        pr = principal(request)
        return {"released": core.lifecycle.release(lease_id, pr.app_id or "cli")}

    @app.get("/v1/lifecycle")
    async def lifecycle_status(request: Request):
        need(request, "orcha.read")
        return core.lifecycle.status()

    # ------------------------------------------------------------------ hardware
    @app.get("/v1/hardware")
    async def hardware(request: Request, refresh: bool = False):
        need(request, "models.read")
        return core.models.hardware(refresh)

    # ------------------------------------------------------------------- models
    m = core.models

    @app.get("/v1/models")
    async def list_models(request: Request, installed: bool | None = None):
        need(request, "models.read")
        items = m.installed() if installed else m.installed() + [c for c in m.catalog() if not c["installed"]]
        return {"active": m.store.active_id(), "models": items}

    @app.get("/v1/models/installed")
    async def models_installed(request: Request):
        need(request, "models.read")
        return {"active": m.store.active_id(), "models": m.installed()}

    @app.get("/v1/models/catalog")
    async def models_catalog(request: Request, q: str | None = None):
        need(request, "models.read")
        return {"models": m.catalog(q)}

    @app.get("/v1/models/search")
    async def models_search(request: Request, q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=50)):
        need(request, "models.read")
        return {"query": q, "models": await m.search_remote(q, limit)}

    @app.get("/v1/models/recommended")
    async def models_recommended(request: Request, limit: int = Query(5, ge=1, le=20)):
        need(request, "models.read")
        return {"hardware": m.hardware(), "models": m.recommended(limit)}

    @app.get("/v1/models/active")
    async def models_active(request: Request):
        need(request, "models.read")
        return m.status()

    @app.get("/v1/models/storage")
    async def models_storage(request: Request):
        need(request, "models.read")
        return m.storage()

    @app.put("/v1/models/storage")
    async def models_set_storage(request: Request, body: dict = Body(...)):
        need(request, "models.manage")
        if not body.get("path"):
            raise RuntimeApiError("invalid_request", "'path' is required.", 400)
        return await m.set_models_dir(str(body["path"]), bool(body.get("move")))

    @app.post("/v1/models/storage/dirs")
    async def models_add_dir(request: Request, body: dict = Body(...)):
        need(request, "models.manage")
        return m.add_extra_dir(str(body.get("path", "")))

    @app.delete("/v1/models/storage/dirs")
    async def models_del_dir(request: Request, path: str):
        need(request, "models.manage")
        return m.remove_extra_dir(path)

    @app.get("/v1/models/discovered")
    async def models_discovered(request: Request):
        need(request, "models.read")
        return {"enabled": core.config.get("models.discover_apps"), "locations": m.discovered()}

    @app.post("/v1/models/register")
    async def models_register(request: Request, body: dict = Body(...)):
        """An app announces a model file (or folder) it already downloaded. No copy, no download."""
        pr = need(request, "models.register")
        if not body.get("path"):
            raise RuntimeApiError("invalid_request", "'path' (a .gguf file or a folder) is required.", 400)
        res = m.announce(str(body["path"]), pr.app_id)
        core.log("INFO", f"model location announced by {pr.app_id or 'owner'}: {body['path']}")
        return res

    @app.post("/v1/models/link")
    async def models_link(request: Request, body: dict = Body(...)):
        need(request, "models.manage")
        return m.link(str(body.get("path", "")), body.get("id"), body.get("name"))

    @app.post("/v1/models/install", status_code=202)
    async def models_install(request: Request, body: dict = Body(...)):
        pr = need(request, "models.manage")
        target = body.get("model") or ""
        if not target and not body.get("url"):
            raise RuntimeApiError("invalid_request", "'model' (catalog id or Hugging Face repo) or 'url' is required.", 400)
        plan = await m.resolve_install(target, body.get("file"), body.get("url"))
        dest = Path(body["dir"]).expanduser() if body.get("dir") else None
        job = core.jobs.create("model.install", pr.app_id, {"model": plan["id"], "size_bytes": plan.get("size_bytes")})
        core.jobs.set_cancel_hook(job.id, m.cancel_hook(job.id))
        core.jobs.submit(job, m.install_work(plan, dest))
        if body.get("wait"):
            await core.wait_job(job, float(body.get("timeout_s", 3600)))
        return {"job": job.public()}

    @app.post("/v1/models/remove")
    async def models_remove(request: Request, body: dict = Body(...)):
        need(request, "models.manage")
        return await m.remove(str(body.get("id", "")), body.get("delete_file"), bool(body.get("confirm")))

    @app.post("/v1/models/select")
    async def models_select(request: Request, body: dict = Body(...)):
        pr = need(request, "models.select")
        if not body.get("id"):
            raise RuntimeApiError("invalid_request", "'id' is required.", 400)
        return await core.call("models", "select", {"id": body["id"], "wait_s": float(body.get("wait_s", 120))},
                               pr=pr, timeout=float(body.get("wait_s", 120)) + 30)

    @app.post("/v1/models/deselect")
    async def models_deselect(request: Request):
        need(request, "models.select")
        return await m.deselect()

    @app.get("/v1/models/{model_id:path}/compatibility")
    async def models_compat(request: Request, model_id: str):
        need(request, "models.read")
        return m.compatibility(model_id)

    @app.get("/v1/models/{model_id:path}")
    async def models_get(request: Request, model_id: str):
        need(request, "models.read")
        return m.get(model_id)

    # ---------------------------------------------------------------- providers
    @app.get("/v1/providers")
    async def providers_list(request: Request):
        need(request, "models.read")
        return {"providers": m.providers.list()}

    @app.post("/v1/providers", status_code=201)
    async def providers_add(request: Request, body: dict = Body(...)):
        need(request, "models.manage")
        try:
            rec = m.providers.add(label=str(body.get("label") or body.get("model") or "provider"),
                                  base_url=str(body.get("base_url", "")), model=str(body.get("model", "")),
                                  api_key=str(body.get("api_key") or ""), provider=str(body.get("provider") or "openai-compatible"))
        except ValueError as exc:
            raise RuntimeApiError("invalid_request", str(exc), 400) from exc
        core.log("INFO", f"provider added: {rec['id']} ({rec['base_url']}) key={'set' if rec['has_api_key'] else 'none'}")
        return rec

    @app.delete("/v1/providers/{provider_id}")
    async def providers_remove(request: Request, provider_id: str):
        need(request, "models.manage")
        return await m.remove(f"cloud:{provider_id}")

    # --------------------------------------------------------------------- chat
    @app.post("/v1/chat")
    async def chat(request: Request, body: dict = Body(...)):
        pr = need(request, "chat")
        target, upstream = await core.chat(pr, body)
        if upstream["stream"]:
            return StreamingResponse(core.chat_stream(target, upstream), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        return await core.chat_complete(target, upstream)

    # ------------------------------------------------------------------- orcha
    async def _submit(request: Request, kind: str, body: dict) -> dict:
        pr = need(request, "orcha.run")
        job = await core.submit_orcha(pr, kind, body)
        if body.get("wait", kind == "task"):
            await core.wait_job(job, float(body.get("wait_timeout_s", core.config.get("orcha.job_timeout_s"))) + 5)
        return {"job": job.public()}

    @app.post("/v1/orcha/run", status_code=202)
    async def orcha_run(request: Request, body: dict = Body(...)):
        return await _submit(request, "orcha.run", body)

    @app.post("/v1/task", status_code=202)
    async def task(request: Request, body: dict = Body(...)):
        return await _submit(request, "task", body)

    @app.post("/v1/agent/run", status_code=202)
    async def agent_run(request: Request, body: dict = Body(...)):
        return await _submit(request, "agent.run", body)

    @app.get("/v1/orcha/status")
    async def orcha_status(request: Request):
        need(request, "orcha.read")
        svc = core.service_status()["orcha"]
        info: dict[str, Any] = {"service": svc}
        if svc.get("state") == "running":
            try:
                h = await core.call("orcha", "health")
                info["engine"] = {"experts": h.get("experts"), "source": h.get("source"),
                                  "synthesizer": h.get("synthesizer"), "run_all_experts": h.get("run_all_experts")}
            except RuntimeApiError as exc:
                info["engine_error"] = exc.to_dict()["error"]
        info["active_model"] = m.store.active_id()
        info["jobs"] = core.jobs.counts()
        return info

    # --------------------------------------------------------------------- jobs
    @app.get("/v1/jobs")
    async def jobs_list(request: Request, state: str | None = None, kind: str | None = None,
                        limit: int = Query(50, ge=1, le=200)):
        pr = principal(request)
        jobs = core.jobs.list(pr.app_id, pr.kind == "owner", state, kind, limit)
        return {"jobs": [j.public(include_result=False) for j in jobs]}

    @app.get("/v1/jobs/{job_id}")
    async def jobs_get(request: Request, job_id: str):
        pr = principal(request)
        return {"job": core.jobs.get(job_id, pr.app_id, pr.kind == "owner").public()}

    @app.post("/v1/jobs/{job_id}/cancel")
    async def jobs_cancel(request: Request, job_id: str):
        pr = principal(request)
        job = core.jobs.get(job_id, pr.app_id, pr.kind == "owner")
        return {"job": (await core.jobs.cancel(job.id)).public()}

    @app.get("/v1/jobs/{job_id}/events")
    async def jobs_events(request: Request, job_id: str):
        pr = principal(request)
        job = core.jobs.get(job_id, pr.app_id, pr.kind == "owner")
        run_id = job.meta.get("orcha_run_id")
        if not run_id:
            raise RuntimeApiError("no_events", "This job has no event stream (yet).", 409,
                                  hint="Only ORCHA jobs stream events; poll GET /v1/jobs/{id} for others.")
        async def counted():
            core.active_streams += 1
            try:
                async for chunk in core.orcha.events(run_id):
                    yield chunk
            finally:
                core.active_streams -= 1
        return StreamingResponse(counted(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    # ------------------------------------------------------------------- memory
    def _mem_payload(pr: Principal, **kw: Any) -> dict[str, Any]:
        return {"_principal": core.principal_payload(pr), **kw}

    @app.post("/v1/memory", status_code=201)
    async def memory_store(request: Request, body: dict = Body(...)):
        pr = need(request, "memory.write")
        if not isinstance(body.get("content"), str) or not body["content"].strip():
            raise RuntimeApiError("invalid_request", "'content' (non-empty string) is required.", 400)
        if body.get("tags") is not None and not (isinstance(body["tags"], list) and all(isinstance(t, str) for t in body["tags"])):
            raise RuntimeApiError("invalid_request", "'tags' must be a list of strings.", 400)
        allowed = ("content", "title", "type", "tags", "importance", "scope", "workspace", "extra")
        return await core.call("nomi", "store", _mem_payload(pr, app=body.get("app") if pr.kind == "owner" else None,
                                                             **{k: body[k] for k in allowed if k in body}),
                               pr=pr, op=core.bus.ops.OP_MEMORY_WRITE if core.bus else None)

    @app.get("/v1/memory/search")
    async def memory_search(request: Request, q: str = "", limit: int = Query(10, ge=1, le=100),
                            scope: str | None = None, workspace: str | None = None, tag: list[str] | None = Query(None),
                            type: str | None = None, app_id: str | None = Query(None, alias="app")):
        pr = need(request, "memory.read")
        return await core.call("nomi", "search", _mem_payload(
            pr, query=q, limit=limit, scope=scope, workspace=workspace, tags=tag, type=type,
            app=app_id if pr.kind == "owner" else None), pr=pr, op=core.bus.ops.OP_MEMORY_READ if core.bus else None)

    @app.get("/v1/memory")
    async def memory_list(request: Request, limit: int = Query(20, ge=1, le=100), scope: str | None = None,
                          workspace: str | None = None, type: str | None = None,
                          app_id: str | None = Query(None, alias="app")):
        pr = need(request, "memory.read")
        return await core.call("nomi", "search", _mem_payload(
            pr, query="", limit=limit, scope=scope, workspace=workspace, type=type,
            app=app_id if pr.kind == "owner" else None), pr=pr, op=core.bus.ops.OP_MEMORY_READ if core.bus else None)

    @app.get("/v1/memory/{memory_id}")
    async def memory_get(request: Request, memory_id: str):
        pr = need(request, "memory.read")
        return await core.call("nomi", "get", _mem_payload(pr, id=memory_id), pr=pr)

    @app.delete("/v1/memory/{memory_id}")
    async def memory_delete(request: Request, memory_id: str):
        pr = need(request, "memory.write")
        return await core.call("nomi", "delete", _mem_payload(pr, id=memory_id), pr=pr,
                               op=core.bus.ops.OP_MEMORY_DELETE if core.bus else None)

    @app.get("/v1/nomi/status")
    async def nomi_status(request: Request):
        need(request, "orcha.read")
        svc = core.service_status()["nomi"]
        info: dict[str, Any] = {"service": svc}
        if svc.get("state") == "running":
            try:
                info["memory"] = await core.call("nomi", "status", _mem_payload(principal(request)))
            except RuntimeApiError as exc:
                info["memory_error"] = exc.to_dict()["error"]
        return info

    # ------------------------------------------------------------------ context
    @app.get("/v1/context")
    async def context_collections(request: Request):
        pr = need(request, "context.read")
        return {"collections": await core.call("context", "collections", {"app": as_app(pr, request.query_params.get("app"))}, pr=pr)}

    @app.put("/v1/context/{collection}/documents/{doc_id}")
    async def context_put(request: Request, collection: str, doc_id: str, body: dict = Body(...)):
        pr = need(request, "context.write")
        if not isinstance(body.get("text"), str):
            raise RuntimeApiError("invalid_request", "'text' (string) is required.", 400)
        return await core.call("context", "put", {"app": as_app(pr, body.get("app")), "collection": collection,
                                                  "doc_id": doc_id, "text": body["text"], "title": body.get("title"),
                                                  "metadata": body.get("metadata")}, pr=pr, op=core.bus.ops.OP_INDEX_UPSERT if core.bus else None)

    @app.get("/v1/context/{collection}/documents")
    async def context_docs(request: Request, collection: str):
        pr = need(request, "context.read")
        return {"documents": await core.call("context", "documents", {"app": as_app(pr, request.query_params.get("app")),
                                                                       "collection": collection}, pr=pr)}

    @app.delete("/v1/context/{collection}/documents/{doc_id}")
    async def context_del_doc(request: Request, collection: str, doc_id: str):
        pr = need(request, "context.write")
        return await core.call("context", "delete", {"app": as_app(pr, request.query_params.get("app")),
                                                     "collection": collection, "doc_id": doc_id}, pr=pr)

    @app.delete("/v1/context/{collection}")
    async def context_drop(request: Request, collection: str):
        pr = need(request, "context.write")
        return await core.call("context", "delete", {"app": as_app(pr, request.query_params.get("app")),
                                                     "collection": collection}, pr=pr)

    @app.post("/v1/context/{collection}/search")
    async def context_search(request: Request, collection: str, body: dict = Body(...)):
        pr = need(request, "context.read")
        return await core.call("context", "search", {"app": as_app(pr, body.get("app")), "collection": collection,
                                                     "query": str(body.get("query", "")), "limit": int(body.get("limit", 8)),
                                                     "doc_ids": body.get("doc_ids")}, pr=pr,
                               op=core.bus.ops.OP_INDEX_QUERY if core.bus else None)

    # ------------------------------------------- shared resources (explicit, user-controlled)
    R = core.resources

    @app.post("/v1/resources", status_code=201)
    async def resource_create(request: Request, body: dict = Body(...)):
        pr = need(request, "context.write")
        keys = ("type", "title", "kind", "collection", "doc_ids", "text", "path", "workspace", "metadata")
        return R.create(pr, owner=body.get("owner"), **{k: body[k] for k in keys if k in body})

    @app.get("/v1/resources")
    async def resource_list(request: Request, type: str | None = None, workspace: str | None = None, owned: bool = False):
        pr = need(request, "context.read")
        return {"resources": R.list(pr, type=type, workspace=workspace, owned=owned)}

    @app.post("/v1/resources/search")
    async def resource_search(request: Request, body: dict = Body(...)):
        pr = need(request, "context.read")
        return await R.search(pr, str(body.get("query", "")), resources=body.get("resources"), workspace=body.get("workspace"),
                              types=body.get("types"), limit=int(body.get("limit", 8)))

    @app.get("/v1/resources/resolve")
    async def resource_resolve(request: Request, ref: str):
        return R.resolve(need(request, "context.read"), ref)

    @app.get("/v1/resources/requests")
    async def resource_requests(request: Request, state: str | None = "pending"):
        return {"requests": R.list_requests(need(request, "context.read"), state if state != "all" else None)}

    @app.post("/v1/resources/requests/{qid}/approve")
    async def resource_approve(request: Request, qid: str):
        return R.decide(need(request, "context.share"), qid, True)

    @app.post("/v1/resources/requests/{qid}/deny")
    async def resource_deny(request: Request, qid: str):
        return R.decide(need(request, "context.share"), qid, False)

    @app.get("/v1/resources/audit")
    async def resource_audit(request: Request, limit: int = Query(50, ge=1, le=500), resource: str | None = None):
        return {"events": R.audit(need(request, "context.read"), limit, resource)}

    @app.get("/v1/resources/{rid}")
    async def resource_get(request: Request, rid: str):
        return R.get(need(request, "context.read"), rid)

    @app.patch("/v1/resources/{rid}")
    async def resource_update(request: Request, rid: str, body: dict = Body(...)):
        pr = need(request, "context.write")
        return R.update(pr, rid, **{k: body[k] for k in ("title", "metadata", "workspace", "text") if k in body})

    @app.delete("/v1/resources/{rid}")
    async def resource_delete(request: Request, rid: str, purge: bool = False):
        return await R.delete(need(request, "context.write"), rid, purge)

    @app.get("/v1/resources/{rid}/read")
    async def resource_read(request: Request, rid: str, doc: str | None = None):
        return await R.read(need(request, "context.read"), rid, doc)

    @app.put("/v1/resources/{rid}/documents/{doc_id}")
    async def resource_write(request: Request, rid: str, doc_id: str, body: dict = Body(...)):
        pr = need(request, "context.write")
        if not isinstance(body.get("text"), str):
            raise RuntimeApiError("invalid_request", "'text' (string) is required.", 400)
        return await R.write_document(pr, rid, doc_id, body["text"], body.get("title"))

    @app.post("/v1/resources/{rid}/share")
    async def resource_share(request: Request, rid: str, body: dict = Body(...)):
        pr = need(request, "context.share")
        targets = body.get("with") or ([] if not body.get("global") else ["*"])
        if body.get("global"):
            targets = [*targets, "*"]
        if not targets:
            raise RuntimeApiError("invalid_request", "'with' (list of app ids) or 'global': true is required.", 400)
        return R.share(pr, rid, targets, body.get("access", "read"))

    @app.post("/v1/resources/{rid}/revoke")
    async def resource_revoke(request: Request, rid: str, body: dict = Body(default={})):
        return R.revoke(need(request, "context.share"), rid, body.get("with"))

    @app.get("/v1/resources/{rid}/permissions")
    async def resource_permissions(request: Request, rid: str):
        return R.permissions(need(request, "context.read"), rid)

    @app.post("/v1/resources/{rid}/request")
    async def resource_request(request: Request, rid: str, body: dict = Body(default={})):
        return R.request_access(need(request, "context.read"), rid, body.get("access", "read"), str(body.get("reason", "")))

    @app.get("/v1/workspaces")
    async def workspaces(request: Request):
        return {"workspaces": R.workspaces(need(request, "context.read"))}

    @app.post("/v1/workspaces", status_code=201)
    async def workspace_create(request: Request, body: dict = Body(...)):
        return R.create_workspace(need(request, "context.write"), str(body.get("name", "")))

    @app.post("/v1/workspaces/{name}/share")
    async def workspace_share(request: Request, name: str, body: dict = Body(...)):
        return R.share_workspace(need(request, "context.share"), name, body.get("with") or [], body.get("access", "read"))

    @app.get("/v1/capabilities")
    async def capabilities(request: Request):
        """What the runtime provides and what is currently active. Subsystems are only *used* when an app asks."""
        need(request, "orcha.read")
        svc = core.service_status()
        st = lambda s: s.get("state", "unavailable")  # noqa: E731
        m = core.models.status()
        return {"capabilities": [
            {"name": "models", "state": "active" if m["active"] else "idle", "detail": m["active"] or "no active model"},
            {"name": "chat", "state": "active" if m["state"] in ("running", "remote") else "idle", "detail": "needs an active model"},
            {"name": "orcha", "state": st(svc["orcha"]), "detail": "orchestration jobs, agents"},
            {"name": "agents", "state": st(svc["orcha"]), "detail": "workspace-scoped agent runs (ORCHA)"},
            {"name": "memory", "state": st(svc["nomi"]), "detail": "Nomi; app-private or shared"},
            {"name": "context", "state": "active" if core.context.opened else "idle",
             "detail": "opens on first use" if not core.context.opened else "document index open"},
            {"name": "resources", "state": "active" if core.resources.opened else "idle", "detail": core.resources.stats()},
            {"name": "aicl", "state": st(svc["aicl"]), "detail": "in-process bus"},
        ], "note": "Notes, Study and Code experiences live in the applications, not in the runtime."}

    # --------------------------------------------------------------------- aicl
    @app.get("/v1/aicl/status")
    async def aicl_status(request: Request):
        need(request, "runtime.admin")
        if core.bus is None:
            return {"available": False, "error": core.bus_error}
        return {"available": True, **core.bus.stats(), "recent": core.bus.recent(20)}

    return app
