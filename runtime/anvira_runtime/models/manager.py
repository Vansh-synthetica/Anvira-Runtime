"""Model lifecycle: discovery, install, select, remove, storage locations.

The runtime owns this so applications (and the CLI) share one model library,
one download of each file, and one running inference backend.
"""
from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..config.paths import RuntimeLayout
from ..config.settings import RuntimeConfig
from ..core.errors import RuntimeApiError, no_model
from ..core.jobs import Job
from ..hardware import detect_hardware, disk_info
from ..process.supervisor import Supervisor, tail_file
from . import backend as llama
from .catalog import HuggingFaceClient, load_catalog, pick_preferred_file
from .compat import assess, estimate_params_b, recommend
from .discovery import discover_app_model_dirs
from .download import DownloadCancelled, DownloadError, download_file
from .store import (SPLIT_RE, ModelStore, ProviderStore, move_model_files, parse_gguf_filename,
                    slugify, validate_storage_dir)

ActiveChanged = Callable[[dict[str, Any] | None], Awaitable[None]]


class ModelManager:
    def __init__(self, layout: RuntimeLayout, config: RuntimeConfig, supervisor: Supervisor, *,
                 hf: HuggingFaceClient | None = None, on_active_changed: ActiveChanged | None = None,
                 log: Callable[[str, str], None] | None = None):
        self.layout, self.config, self.supervisor = layout, config, supervisor
        self.store = ModelStore(layout, config.models_dir(), config.extra_model_dirs(),
                                discover=lambda: discover_app_model_dirs() if config.get("models.discover_apps") else {})
        self.providers = ProviderStore(layout)
        self.hf = hf or HuggingFaceClient(config.get("models.huggingface_endpoint"))
        self.on_active_changed = on_active_changed
        self._log = log or (lambda level, msg: None)
        self._catalog = load_catalog()
        self._load_task: asyncio.Task | None = None
        self._loading_id: str | None = None
        self._cancel_events: dict[str, asyncio.Event] = {}
        self.last_backend_info: dict[str, Any] = {}

    # ------------------------------------------------------------------ hardware
    def hardware(self, refresh: bool = False) -> dict[str, Any]:
        return detect_hardware(self.store.primary_dir, use_cache=not refresh)

    # --------------------------------------------------------------- discovery
    def _decorate(self, m: dict[str, Any], hw: dict[str, Any], active: str | None) -> dict[str, Any]:
        out = {**m, "active": m["id"] == active}
        if m.get("kind") == "local" and m.get("installed"):
            out["running"] = self._service_state(m["id"]) == "running"
        out["compatibility"] = assess(m, hw)
        return out

    def installed(self) -> list[dict[str, Any]]:
        hw, active = self.hardware(), self.store.active_id()
        models = self.store.scan() + self.providers.list()
        return [self._decorate(m, hw, active) for m in models]

    def catalog(self, query: str | None = None) -> list[dict[str, Any]]:
        """Curated catalog merged with install state (installed models the catalog doesn't know come from ``installed``)."""
        hw, active = self.hardware(), self.store.active_id()
        installed = {m["id"]: m for m in self.store.scan()}
        by_file = {m["file"].lower(): m for m in installed.values()}
        out = []
        for c in self._catalog:
            hit = installed.get(c["id"]) or by_file.get(c["source"]["file"].lower())
            merged = {**c, **({"installed": True, "path": hit["path"], "id": hit["id"], "complete": hit["complete"]} if hit
                              else {"installed": False})}
            if query and query.lower() not in f"{c['id']} {c['name']} {' '.join(c['tags'])}".lower():
                continue
            out.append(self._decorate(merged, hw, active))
        return out

    async def search_remote(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        try:
            results = await self.hf.search(query, limit)
        except Exception as exc:  # noqa: BLE001 - network is optional (local-first)
            raise RuntimeApiError("search_unavailable", f"Model search failed: {type(exc).__name__}: {exc}", 502,
                                  hint="Check your internet connection; installed models still work offline.") from exc
        return results

    def recommended(self, limit: int = 5) -> list[dict[str, Any]]:
        hw = self.hardware()
        return recommend(self.catalog(), hw, limit)

    def get(self, model_id: str) -> dict[str, Any]:
        for m in self.installed():
            if m["id"] == model_id:
                return m
        for m in self.catalog():
            if m["id"] == model_id:
                return m
        raise RuntimeApiError("model_not_found", f"Model '{model_id}' was not found.", 404,
                              hint="Run `anvira model list` to see available and installed models.")

    def compatibility(self, model_id: str) -> dict[str, Any]:
        m = self.get(model_id)
        return {"id": model_id, "hardware": self.hardware(), **m["compatibility"]}

    # ------------------------------------------------------------------- state
    def _service_name(self, model_id: str) -> str:
        return f"model:{model_id}"

    def _service_state(self, model_id: str) -> str:
        st = self.supervisor.services.get(self._service_name(model_id))
        return st.state if st else "stopped"

    def active_record(self) -> dict[str, Any] | None:
        aid = self.store.active_id()
        if not aid:
            return None
        for m in self.store.scan() + self.providers.list():
            if m["id"] == aid:
                return m
        return None

    def _installed_ids(self) -> list[str]:
        try:
            return [m["id"] for m in self.store.scan()] + [p["id"] for p in self.providers.list()]
        except Exception:  # noqa: BLE001 - only used to make an error message more helpful
            return []

    def status(self) -> dict[str, Any]:
        rec = self.active_record()
        aid = self.store.active_id()
        state, detail = "none", {}
        if aid and rec is None:
            state = "missing"
            detail = {"message": f"The active model '{aid}' is no longer installed."}
        elif rec and rec["kind"] == "provider":
            state = "remote"
        elif rec:
            svc = self.supervisor.services.get(self._service_name(rec["id"]))
            state = ("loading" if self._loading_id == rec["id"] else (svc.state if svc else "stopped"))
            if svc:
                detail = svc.public()
                if svc.state in ("failed", "degraded") and svc.spec.log_path:
                    detail["log_tail"] = tail_file(svc.spec.log_path, 10)
        return {"active": aid, "state": state, "model": rec, "backend": detail or None,
                "backend_info": self.last_backend_info or None,
                "models_dir": str(self.store.primary_dir), "extra_dirs": [str(d) for d in self.store.extra_dirs],
                "installed_count": len(self.store.scan()) + len(self.providers.list())}

    # ------------------------------------------------------------------ select
    async def select(self, model_id: str, wait_s: float = 120.0) -> dict[str, Any]:
        rec = next((m for m in self.store.scan() + self.providers.list() if m["id"] == model_id), None)
        if rec is None:
            known = any(c["id"] == model_id for c in self._catalog)
            raise RuntimeApiError(
                "model_not_installed" if known else "model_not_found",
                f"Model '{model_id}' is not installed." if known else f"Model '{model_id}' was not found.", 404 if not known else 409,
                hint=(f"Install it first: `anvira model install {model_id}`." if known
                      else "Run `anvira model list` to see installed and available models."))
        if rec["kind"] == "local" and not rec.get("complete", True):
            raise RuntimeApiError("model_incomplete", f"'{model_id}' is an incomplete split model (missing shards).", 409,
                                  hint="Re-download or add the missing shard files.")
        previous = self.store.active_id()
        self.store.set_active(model_id)
        try:
            if rec["kind"] == "local":
                await self._stop_other_models(keep=model_id)
                await self._ensure_started(rec, wait_s)
            else:
                await self._stop_other_models(keep=None)
        except Exception:
            self.store.set_active(previous)
            raise
        if self.on_active_changed:
            await self.on_active_changed(rec)
        return self.status()

    async def deselect(self) -> dict[str, Any]:
        self.store.set_active(None)
        await self._stop_other_models(keep=None)
        if self.on_active_changed:
            await self.on_active_changed(None)
        return self.status()

    async def _stop_other_models(self, keep: str | None) -> None:
        for name in [n for n in list(self.supervisor.services) if n.startswith("model:")]:
            if keep is None or name != self._service_name(keep):
                await self.supervisor.stop(name)
                self.supervisor.remove(name)

    async def _ensure_started(self, rec: dict[str, Any], wait_s: float) -> None:
        name = self._service_name(rec["id"])
        st = self.supervisor.services.get(name)
        if st and st.state == "running":
            return
        if self._load_task is None or self._load_task.done() or self._loading_id != rec["id"]:
            self._loading_id = rec["id"]
            self._load_task = asyncio.create_task(self._load(rec))
        try:
            await asyncio.wait_for(asyncio.shield(self._load_task), timeout=wait_s)
        except asyncio.TimeoutError:
            self._log("INFO", f"model '{rec['id']}' still loading after {wait_s:.0f}s; continuing in background")

    async def _load(self, rec: dict[str, Any]) -> None:
        try:
            spec, info = llama.build_launch(self.layout, self.config, rec, self.hardware(), self.layout.logs_dir)
            self.last_backend_info = info
            if spec is None:
                raise RuntimeApiError(
                    "backend_missing", "The llama.cpp inference backend (llama-server) was not found.", 424,
                    hint="Install llama-server and run `anvira config set models.llama_server_path <path>`; "
                         "see INSTALLATION.md. Searched: " + "; ".join(info.get("searched", [])[:6]),
                    details={"searched": info.get("searched")})
            self.supervisor.add(spec)
            try:
                await self.supervisor.start(spec.name)
            except RuntimeError as exc:
                raise RuntimeApiError("model_start_failed", f"Model '{rec['id']}' failed to start: {exc}", 500,
                                      hint="See `anvira runtime logs --service " + spec.name + "`.") from exc
        finally:
            self._loading_id = None

    async def ensure_ready(self, wait_s: float = 300.0) -> None:
        """Make the active model servable (start it lazily if needed)."""
        rec = self.active_record()
        if rec is None:
            raise no_model(self._installed_ids())
        if rec["kind"] == "provider":
            return
        st = self.supervisor.services.get(self._service_name(rec["id"]))
        if st and st.state == "running":
            return
        if st and st.state == "failed":
            raise RuntimeApiError("model_crashed", f"Model '{rec['id']}' stopped unexpectedly and could not be restarted.", 502,
                                  hint="Run `anvira doctor` and `anvira runtime logs --service model:" + rec["id"] + "`.",
                                  details={"log_tail": tail_file(st.spec.log_path, 12) if st.spec.log_path else ""})
        if self._loading_id == rec["id"] and self._load_task and not self._load_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._load_task), timeout=wait_s)
            except asyncio.TimeoutError:
                raise RuntimeApiError("model_loading", f"Model '{rec['id']}' is still loading.", 503,
                                      hint="Try again shortly; check `anvira model status`.") from None
            return
        await self._ensure_started(rec, wait_s)
        st = self.supervisor.services.get(self._service_name(rec["id"]))
        if not st or st.state != "running":
            raise RuntimeApiError("model_loading", f"Model '{rec['id']}' is not ready yet.", 503,
                                  hint="Check `anvira model status`.")

    def inference_target(self) -> dict[str, Any]:
        """Where chat requests for the active model go (raises structured errors)."""
        rec = self.active_record()
        if rec is None:
            if self.store.active_id():
                raise RuntimeApiError("model_missing", "The active model is no longer installed.", 409,
                                      hint="Select another with `anvira model use <id>`.")
            raise no_model(self._installed_ids())
        if rec["kind"] == "provider":
            private = self.providers.get_private(rec["provider_id"]) or {}
            return {"kind": "provider", "base_url": private["base_url"], "api_key": private.get("api_key") or "",
                    "model": private["model"], "id": rec["id"]}
        st = self.supervisor.services.get(self._service_name(rec["id"]))
        if st is None or st.state == "stopped":
            raise RuntimeApiError("model_not_running", f"Model '{rec['id']}' is not running.", 503,
                                  hint="It will start on the next request, or run `anvira model use " + rec["id"] + "`.")
        if st.state == "failed":
            raise RuntimeApiError("model_crashed", f"Model '{rec['id']}' crashed and could not be restarted.", 502,
                                  hint="Run `anvira runtime logs --service model:" + rec["id"] + "`.",
                                  details={"log_tail": tail_file(st.spec.log_path, 12) if st.spec.log_path else ""})
        if st.state != "running":
            raise RuntimeApiError("model_loading", f"Model '{rec['id']}' is {st.state}.", 503)
        return {"kind": "local", "base_url": f"http://127.0.0.1:{st.spec.port}/v1", "api_key": "not-needed",
                "model": rec["id"], "id": rec["id"]}

    # ----------------------------------------------------------------- install
    async def resolve_install(self, model: str, file: str | None = None, url: str | None = None) -> dict[str, Any]:
        if url:
            fname = Path(url.split("?")[0]).name
            try:
                ModelStore.check_filename(fname)
            except ValueError as exc:
                raise RuntimeApiError("invalid_request", f"{exc} The URL must point to a .gguf file.", 400) from exc
            return {"id": slugify(Path(fname).stem), "name": Path(fname).stem, "files": [{"filename": fname, "url": url}],
                    "source": {"type": "url", "url": url}, "size_bytes": None,
                    "quantization": parse_gguf_filename(fname)["quantization"], "params_b": estimate_params_b(fname)}
        cat = next((c for c in self._catalog if c["id"] == model), None)
        if cat:
            repo, fname = cat["source"]["repo"], file or cat["source"]["file"]
            meta = {"id": cat["id"], "name": cat["name"], "size_bytes": cat["size_bytes"],
                    "quantization": cat["quantization"], "params_b": cat["params_b"]}
        elif "/" in model:
            repo = model[3:] if model.startswith("hf:") else model
            fname = file
            meta = {"id": None, "name": repo.split("/")[-1], "size_bytes": None, "quantization": None,
                    "params_b": estimate_params_b(repo)}
        else:
            raise RuntimeApiError("model_not_found", f"'{model}' is not in the catalog.", 404,
                                  hint="Use a catalog id (`anvira model list`), a Hugging Face repo 'owner/name', or --url.")
        files = None
        if fname is None or SPLIT_RE.match(fname):
            try:
                files = await self.hf.files(repo)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeApiError("search_unavailable", f"Could not list files of '{repo}': {exc}", 502) from exc
        if fname is None:
            pick = pick_preferred_file(files or [])
            if pick is None:
                raise RuntimeApiError("no_gguf", f"'{repo}' has no downloadable .gguf file.", 404)
            fname, meta["size_bytes"] = pick["name"], pick.get("size_bytes")
            meta["quantization"] = parse_gguf_filename(Path(fname).name)["quantization"]
        shard_files = [fname]
        m = SPLIT_RE.match(Path(fname).name)
        if m and files:  # a split GGUF: every shard is required
            shard_files = sorted(f["name"] for f in files if (sm := SPLIT_RE.match(Path(f["name"]).name))
                                 and sm["base"] == m["base"] and sm["total"] == m["total"])
            meta["size_bytes"] = sum(f.get("size_bytes") or 0 for f in files if f["name"] in shard_files) or meta["size_bytes"]
        mid = meta["id"] or slugify(re.sub(r"-\d{5}-of-\d{5}", "", Path(fname).stem))
        return {**meta, "id": mid, "files": [{"filename": Path(f).name, "url": self.hf.download_url(repo, f)} for f in shard_files],
                "source": {"type": "huggingface", "repo": repo, "file": fname}}

    def install_work(self, plan: dict[str, Any], target_dir: Path | None = None
                     ) -> Callable[[Job], Awaitable[dict[str, Any]]]:
        """Return the coroutine that performs the install for a job."""
        async def work(job: Job) -> dict[str, Any]:
            if not self.config.get("models.allow_downloads"):
                raise RuntimeApiError("downloads_disabled", "Model downloads are disabled in configuration.", 403,
                                      hint="anvira config set models.allow_downloads true")
            dest_dir = target_dir or self.store.primary_dir
            ok, err = validate_storage_dir(dest_dir)
            if not ok:
                raise RuntimeApiError("storage_unwritable", err or "Cannot write to the models directory.", 500)
            need = plan.get("size_bytes")
            free = disk_info(dest_dir).get("free_bytes", 0)
            if need and free < need * 1.05:
                raise RuntimeApiError("insufficient_storage",
                                      f"Not enough free disk space in {dest_dir}: {free / 1024**3:.1f} GiB free, "
                                      f"about {need / 1024**3:.1f} GiB needed.", 507,
                                      hint="Free space, or pick another location with `anvira model install ... --dir <path>`.")
            cancel = asyncio.Event()
            self._cancel_events[job.id] = cancel

            async def _cancel() -> None:
                cancel.set()
            job.meta["_cancel"] = True
            job.progress = {"downloaded": 0, "total": need, "percent": 0.0, "speed_bps": 0.0}
            files = plan["files"]
            done_before = 0
            first_path: Path | None = None
            try:
                for i, f in enumerate(files):
                    dest = ModelStore.check_filename(f["filename"])
                    dest_path = dest_dir / dest
                    first_path = first_path or dest_path

                    def on_progress(done: int, total: int | None, speed: float, base=done_before) -> None:
                        tot = need if (len(files) > 1 and need) else (total or need)
                        d = base + done
                        job.progress = {"downloaded": d, "total": tot, "speed_bps": round(speed),
                                        "percent": round(d * 100 / tot, 1) if tot else None,
                                        "file": dest, "file_index": i + 1, "file_count": len(files)}
                    await download_file(f["url"], dest_path, on_progress=on_progress, cancel=cancel)
                    done_before += dest_path.stat().st_size
            except DownloadCancelled:
                raise asyncio.CancelledError() from None
            except DownloadError as exc:
                raise RuntimeApiError("download_failed", str(exc), 502,
                                      hint="Run the install again; the partial download resumes.") from exc
            except Exception as exc:  # noqa: BLE001
                raise RuntimeApiError("download_failed", f"{type(exc).__name__}: {exc}", 502) from exc
            finally:
                self._cancel_events.pop(job.id, None)
            assert first_path is not None
            self.store.register(plan["id"], first_path, name=plan.get("name"), source=plan.get("source"),
                                quantization=plan.get("quantization"), params_b=plan.get("params_b"))
            rec = self.store.get(plan["id"])
            return {"model": rec, "installed_to": str(first_path.parent)}
        return work

    def cancel_hook(self, job_id: str) -> Callable[[], Awaitable[None]]:
        async def hook() -> None:
            ev = self._cancel_events.get(job_id)
            if ev:
                ev.set()
        return hook

    # ------------------------------------------------- remove / link / locations
    async def remove(self, model_id: str, delete_file: bool | None = None, confirm: bool = False) -> dict[str, Any]:
        if model_id.startswith("cloud:"):
            if not self.providers.remove(model_id[6:]):
                raise RuntimeApiError("model_not_found", f"Provider '{model_id}' was not found.", 404)
            if self.store.active_id() == model_id:
                await self.deselect()
            return {"id": model_id, "deleted_files": [], "location": "provider"}
        rec = self.store.get(model_id)
        if rec is None:
            raise RuntimeApiError("model_not_found", f"Model '{model_id}' is not installed.", 404)
        if delete_file is None:
            delete_file = rec["location"] == "primary"
        if delete_file and rec["location"] != "primary" and not confirm:
            raise RuntimeApiError(
                "confirmation_required",
                f"'{model_id}' is stored outside the runtime's download folder ({rec['directory']}). Deleting its "
                f"file needs explicit confirmation.", 409,
                hint="Re-run with --delete-file --yes (CLI) or confirm=true (API), or omit --delete-file to just unlink it.")
        if self.store.active_id() == model_id:
            await self.deselect()
        if self._service_state(model_id) != "stopped":
            await self.supervisor.stop(self._service_name(model_id))
            self.supervisor.remove(self._service_name(model_id))
        result = self.store.remove(model_id, delete_file=delete_file)
        if not delete_file and rec["location"] == "extra":
            result["note"] = "The file is in a scanned folder, so it will reappear; remove the folder from models.extra_dirs to hide it."
        return result

    def link(self, path: str, model_id: str | None = None, name: str | None = None,
             announced_by: str | None = None) -> dict[str, Any]:
        try:
            return self.store.link(path, model_id=model_id, name=name,
                                   source={"type": "announced", "app": announced_by} if announced_by else None)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeApiError("invalid_model_file", str(exc), 400) from exc

    def announce(self, path: str, app_id: str | None) -> dict[str, Any]:
        """An application tells the runtime where a model it downloaded lives.

        A file is linked in place; a folder is added to the scanned folders. Nothing is copied,
        moved or downloaded, and the model becomes available to every app.
        """
        p = Path(path).expanduser()
        if p.is_dir():
            res = self.add_extra_dir(str(p))
            return {"kind": "directory", "path": str(p), **res}
        return {"kind": "file", "model": self.link(str(p), announced_by=app_id)}

    def discovered(self) -> list[dict[str, Any]]:
        scan = self.store.scan()
        return [{"app": label, "dir": str(d), "models": sorted(m["id"] for m in scan if m["location"] == "app"
                                                                and m.get("origin") == label
                                                                and Path(m["directory"]) == d)}
                for d, label in self.store.discovered().items()]

    def _persist_dirs(self) -> None:
        self.config.save()
        self.store.set_dirs(self.config.models_dir(), self.config.extra_model_dirs())

    async def set_models_dir(self, path: str, move: bool = False) -> dict[str, Any]:
        """Change where NEW models are downloaded (any local path); optionally move existing ones."""
        target = Path(path).expanduser()
        ok, err = validate_storage_dir(target)
        if not ok:
            raise RuntimeApiError("storage_unwritable", err or "Invalid directory.", 400)
        old = self.store.primary_dir
        result: dict[str, Any] = {"moved": [], "failed": []}
        if move and target.resolve() != old.resolve():
            files: list[Path] = []
            for m in self.store.scan():
                if m["location"] == "primary":
                    base = Path(m["path"])
                    sm = SPLIT_RE.match(base.name)
                    files += ([p for p in base.parent.glob("*.gguf") if (x := SPLIT_RE.match(p.name))
                               and x["base"] == sm["base"] and x["total"] == sm["total"]] if sm else [base])  # type: ignore[index]
            running = {Path(str(s.spec.argv[2])).resolve() for n, s in self.supervisor.services.items()
                       if n.startswith("model:") and s.state in ("running", "starting") and len(s.spec.argv) > 2}
            result = move_model_files(files, target, is_running=lambda p: p.resolve() in running)
        self.config.set("models.models_dir", str(target))
        self._persist_dirs()
        self.store.repoint({m["source"]: m["target"] for m in result["moved"]})
        return {"models_dir": str(target), "previous": str(old), **result}

    def add_extra_dir(self, path: str) -> dict[str, Any]:
        d = Path(path).expanduser()
        if not d.is_dir():
            raise RuntimeApiError("invalid_directory", f"'{d}' is not a directory.", 400)
        dirs = self.config.get("models.extra_dirs")
        if str(d) not in dirs:
            self.config.set("models.extra_dirs", [*dirs, str(d)])
            self._persist_dirs()
        return {"extra_dirs": [str(x) for x in self.store.extra_dirs],
                "models_found": len([m for m in self.store.scan() if m["location"] == "extra"])}

    def remove_extra_dir(self, path: str) -> dict[str, Any]:
        dirs = [x for x in self.config.get("models.extra_dirs") if Path(x) != Path(path)]
        self.config.set("models.extra_dirs", dirs)
        self._persist_dirs()
        return {"extra_dirs": [str(x) for x in self.store.extra_dirs]}

    def storage(self) -> dict[str, Any]:
        return {"models_dir": str(self.store.primary_dir), "extra_dirs": [str(d) for d in self.store.extra_dirs],
                "disk": disk_info(self.store.primary_dir),
                "linked": [m["path"] for m in self.store.scan() if m["location"] == "linked"]}

    async def aclose(self) -> None:
        await self.hf.aclose()
