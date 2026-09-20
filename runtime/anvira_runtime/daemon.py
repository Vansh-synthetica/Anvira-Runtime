"""The runtime daemon: API server + supervisor in one process.

Started by ``anvira runtime start`` (detached) or ``python -m anvira_runtime daemon``.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import uvicorn
from anvira_client.discovery import probe

from .api.app import create_app
from .config.paths import resolve_layout
from .config.settings import RuntimeConfig
from .core.runtime import RuntimeCore
from .process.supervisor import free_port, port_in_use
from .security.secrets import write_private
from .version import API_VERSION, RUNTIME_VERSION


def run_daemon(console_log: bool = False, port_override: int | None = None, auto_stop: bool | None = None,
               idle_grace_s: float | None = None) -> int:
    layout = resolve_layout().ensure()
    config = RuntimeConfig.load(layout)

    existing = probe()
    if existing.running:
        print(f"Anvira Runtime is already running (pid {existing.pid}, port {existing.port}).", file=sys.stderr)
        return 0

    host = config.get("api.host")
    port = port_override if port_override is not None else config.get("api.port")
    if port == 0:
        port = free_port(host if host != "localhost" else "127.0.0.1")
    elif port_in_use(port, "127.0.0.1"):
        print(f"Port {port} is in use by another program. Free it, or run "
              f"`anvira config set api.port 0` to choose a free port automatically.", file=sys.stderr)
        return 2

    core = RuntimeCore(layout, config, console_log=console_log, auto_stop=auto_stop, idle_grace_s=idle_grace_s)
    server: uvicorn.Server | None = None

    def request_shutdown() -> None:
        if server is not None:
            server.should_exit = True

    app = create_app(core, shutdown=request_shutdown)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False,
                                           lifespan="off", timeout_keep_alive=30))

    async def main() -> None:
        serve_task = asyncio.create_task(server.serve())
        while not server.started and not serve_task.done():
            await asyncio.sleep(0.05)
        if serve_task.done():
            await serve_task
            return
        write_private(layout.discovery_file, json.dumps({
            "pid": os.getpid(), "host": host if host != "localhost" else "127.0.0.1", "port": port,
            "api_version": API_VERSION, "runtime_version": RUNTIME_VERSION, "started_at": core.started_at}))
        try:
            await core.start()
            await serve_task
        finally:
            try:
                await core.stop()
            finally:
                try:
                    current = json.loads(layout.discovery_file.read_text(encoding="utf-8"))
                    if current.get("pid") == os.getpid():
                        layout.discovery_file.unlink()
                except (OSError, ValueError):
                    pass

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    return 0
