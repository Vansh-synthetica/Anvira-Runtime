"""Read runtime and service logs (redacted)."""
from __future__ import annotations

from typing import Any

from ..config.paths import RuntimeLayout
from ..core.errors import RuntimeApiError
from ..process.supervisor import tail_file
from ..security.secrets import redact


def log_path(layout: RuntimeLayout, service: str):
    if service in ("runtime", "orcha", "nomi", "daemon-boot"):
        return layout.logs_dir / ("runtime.log" if service == "runtime" else f"{service}.log")
    if service.startswith("model:"):
        safe = service[6:].replace("/", "_").replace("\\", "_")
        return layout.logs_dir / f"model-{safe}.log"
    raise RuntimeApiError("unknown_log", f"Unknown log '{service}'. Use runtime, orcha, nomi or model:<id>.", 400)


def read_log(layout: RuntimeLayout, service: str, lines: int = 100) -> dict[str, Any]:
    path = log_path(layout, service)
    if not path.exists():
        return {"service": service, "path": str(path), "exists": False, "lines": []}
    return {"service": service, "path": str(path), "exists": True,
            "lines": redact(tail_file(path, lines)).splitlines()}
