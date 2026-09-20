"""Runtime job registry: one uniform view over long-running work.

ORCHA runs, model downloads and model loads all appear as jobs with the same
lifecycle (``queued -> running -> completed | failed | cancelled``), so an
application needs one polling/cancel mechanism regardless of what is running
underneath. Applications only ever see their own jobs; the owner sees all.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..security.secrets import read_json, write_private
from .errors import RuntimeApiError

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}
MAX_KEPT = 200


@dataclass
class Job:
    id: str
    kind: str
    app_id: str | None
    state: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def public(self, include_result: bool = True) -> dict[str, Any]:
        d = {"id": self.id, "kind": self.kind, "app": self.app_id, "state": self.state,
             "created_at": self.created_at, "started_at": self.started_at, "finished_at": self.finished_at,
             "progress": self.progress, "error": self.error, "meta": self.meta}
        if include_result:
            d["result"] = self.result
        return d


class JobRegistry:
    def __init__(self, persist_file: Path | None = None):
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancel_hooks: dict[str, Callable[[], Awaitable[None]]] = {}
        self._persist_file = persist_file
        if persist_file:
            for raw in read_json(persist_file, {}).get("jobs", []):
                job = Job(**raw)
                if job.state not in TERMINAL:
                    job.state, job.finished_at = "interrupted", time.time()
                    job.error = {"code": "runtime_restarted", "message": "The runtime restarted while this job was running."}
                self._jobs[job.id] = job

    # -- persistence -----------------------------------------------------------
    def _persist(self) -> None:
        if not self._persist_file:
            return
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at)[-MAX_KEPT:]
        self._jobs = {j.id: j for j in jobs}
        payload = [{**j.__dict__} for j in jobs]
        try:
            write_private(self._persist_file, json.dumps({"jobs": payload}, default=str))
        except (OSError, TypeError):
            pass

    # -- lifecycle ---------------------------------------------------------------
    def create(self, kind: str, app_id: str | None, meta: dict[str, Any] | None = None) -> Job:
        job = Job(id="job_" + secrets.token_hex(6), kind=kind, app_id=app_id, meta=meta or {})
        self._jobs[job.id] = job
        self._persist()
        return job

    def set_cancel_hook(self, job_id: str, hook: Callable[[], Awaitable[None]]) -> None:
        self._cancel_hooks[job_id] = hook

    def update(self, job: Job, **fields: Any) -> None:
        for k, v in fields.items():
            setattr(job, k, v)

    def finish(self, job: Job, state: str, *, result: Any = None, error: dict[str, Any] | None = None) -> None:
        if job.state in TERMINAL:
            return
        job.state, job.finished_at = state, time.time()
        if result is not None:
            job.result = result
        if error is not None:
            job.error = error
        self._cancel_hooks.pop(job.id, None)
        self._persist()

    def submit(self, job: Job, work: Callable[[Job], Awaitable[Any]]) -> Job:
        """Run ``work(job)`` in the background, recording the outcome on ``job``."""
        async def runner() -> None:
            job.state, job.started_at = "running", time.time()
            try:
                result = await work(job)
                self.finish(job, "completed", result=result)
            except asyncio.CancelledError:
                self.finish(job, "cancelled", error={"code": "cancelled", "message": "Cancelled."})
            except RuntimeApiError as exc:
                self.finish(job, "failed", error=exc.to_dict()["error"])
            except Exception as exc:  # noqa: BLE001 - jobs must never crash the runtime
                self.finish(job, "failed", error={"code": "job_failed", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                self._tasks.pop(job.id, None)

        self._tasks[job.id] = asyncio.create_task(runner())
        return job

    async def cancel(self, job_id: str) -> Job:
        job = self._jobs[job_id]
        if job.state in TERMINAL:
            raise RuntimeApiError("job_not_cancellable", f"Job {job_id} already {job.state}.", 409)
        hook = self._cancel_hooks.get(job_id)
        if hook:
            try:
                await hook()
            except Exception:  # noqa: BLE001 - best effort; still cancel locally
                pass
        task = self._tasks.get(job_id)
        if task:
            task.cancel()
            await asyncio.wait({task}, timeout=5)
        self.finish(job, "cancelled", error={"code": "cancelled", "message": "Cancelled by the caller."})
        return job

    # -- queries ---------------------------------------------------------------
    def get(self, job_id: str, app_id: str | None = None, owner: bool = False) -> Job:
        job = self._jobs.get(job_id)
        if job is None or (not owner and job.app_id != app_id):
            raise RuntimeApiError("job_not_found", f"Job '{job_id}' was not found.", 404)
        return job

    def list(self, app_id: str | None = None, owner: bool = False, state: str | None = None,
             kind: str | None = None, limit: int = 50) -> list[Job]:
        jobs = [j for j in self._jobs.values()
                if (owner or j.app_id == app_id) and (not state or j.state == state) and (not kind or j.kind == kind)]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)[:limit]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for j in self._jobs.values():
            out[j.state] = out.get(j.state, 0) + 1
        return out

    async def shutdown(self) -> None:
        for job_id, task in list(self._tasks.items()):
            task.cancel()
        if self._tasks:
            await asyncio.wait(list(self._tasks.values()), timeout=5)
        for job in self._jobs.values():
            if job.state not in TERMINAL:
                self.finish(job, "interrupted", error={"code": "runtime_stopped", "message": "The runtime is stopping."})
