"""Resumable model file downloader (replaces Electron ``download-model-file``)."""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Callable

import httpx

ProgressFn = Callable[[int, int | None, float], None]  # downloaded, total, bytes/sec


class DownloadCancelled(Exception):
    pass


class DownloadError(Exception):
    pass


async def download_file(
    url: str, dest: Path, *, on_progress: ProgressFn | None = None,
    cancel: asyncio.Event | None = None, transport: httpx.AsyncBaseTransport | None = None,
    headers: dict[str, str] | None = None, chunk_size: int = 1 << 20,
) -> Path:
    """Download ``url`` to ``dest`` via ``dest.part`` with HTTP Range resume.

    The partial file is kept on cancel/failure so a retry continues where it
    stopped; it is renamed to ``dest`` only after the byte count matches the
    server-declared length.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req_headers = {"User-Agent": "anvira-runtime", **(headers or {})}
    if have:
        req_headers["Range"] = f"bytes={have}-"

    async with httpx.AsyncClient(follow_redirects=True, transport=transport,
                                 timeout=httpx.Timeout(30.0, read=60.0)) as client:
        async with client.stream("GET", url, headers=req_headers) as resp:
            if resp.status_code == 416 and have:  # already complete on the server side
                os.replace(part, dest)
                return dest
            if resp.status_code not in (200, 206):
                raise DownloadError(f"Server returned HTTP {resp.status_code} for {url}")
            mode = "ab"
            if resp.status_code == 200 and have:  # server ignored Range: restart
                mode, have = "wb", 0
            length = resp.headers.get("content-length")
            total = (int(length) + have) if length is not None and length.isdigit() else None
            done = have
            started, last_emit, start_bytes = time.monotonic(), 0.0, done
            with open(part, mode) as fh:
                async for chunk in resp.aiter_bytes(chunk_size):
                    if cancel is not None and cancel.is_set():
                        raise DownloadCancelled()
                    fh.write(chunk)
                    done += len(chunk)
                    now = time.monotonic()
                    if on_progress and now - last_emit >= 0.25:
                        speed = (done - start_bytes) / max(now - started, 1e-6)
                        on_progress(done, total, speed)
                        last_emit = now
    final = part.stat().st_size
    if total is not None and final != total:
        raise DownloadError(f"Download incomplete: got {final} of {total} bytes (kept for resume).")
    os.replace(part, dest)
    if on_progress:
        on_progress(final, total or final, 0.0)
    return dest
