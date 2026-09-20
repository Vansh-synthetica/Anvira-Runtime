"""Locate ORCHA / Nomi / AICL and build their launch specs.

ORCHA and Nomi are launched exactly the way the Anvira desktop app launches
them today — via their own ``desktop_entry.py`` (or a frozen
``*-server`` executable when one is installed) — so their code is reused
unmodified. What changes is *who* launches them (the runtime supervisor),
*where* they listen (private loopback ports chosen by the runtime, never the
well-known 8420/8000 the legacy Anvira app uses), and *who holds their
credentials* (the runtime; applications never see them).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from ..config.paths import RuntimeLayout
from ..config.settings import RuntimeConfig
from ..security.secrets import SecretStore
from .supervisor import ServiceSpec, free_port

REPO_ROOT = Path(__file__).resolve().parents[3]  # <repo>/runtime/anvira_runtime/process/infra.py

NOMI_EMAIL = "runtime-owner@anvira-runtime.dev"
NOMI_USERNAME = "anvira-runtime"

# Modules each service needs to import, used by `anvira doctor`.
ORCHA_MODULES = ("fastapi", "uvicorn", "pydantic", "httpx", "langgraph", "yaml")
NOMI_MODULES = ("fastapi", "uvicorn", "sqlalchemy", "aiosqlite", "alembic", "jose", "bcrypt",
                "pydantic_settings", "email_validator")


@dataclass
class Located:
    name: str
    mode: str                 # "source" | "executable" | "missing"
    path: Path | None
    detail: str = ""


def _exe_name(base: str) -> str:
    return base + (".exe" if sys.platform == "win32" else "")


def locate_service(name: str, layout: RuntimeLayout, config: RuntimeConfig) -> Located:
    """Find where ``orcha`` or ``nomi`` lives (source tree or frozen executable)."""
    import os
    marker_pkg = {"orcha": "orcha", "nomi": "app"}[name]
    src_name = {"orcha": "Orcha", "nomi": "nomi"}[name]

    exe = config.get(f"services.{name}.executable")
    if exe and Path(exe).is_file():
        return Located(name, "executable", Path(exe))
    bundled = layout.home / "services" / f"{name}-server" / _exe_name(f"{name}-server")
    if bundled.is_file():
        return Located(name, "executable", bundled)

    candidates = [config.get(f"services.{name}.source_dir"), os.environ.get(f"ANVIRA_{name.upper()}_DIR"),
                  REPO_ROOT / src_name, layout.home / "services" / name]
    for cand in candidates:
        if not cand:
            continue
        p = Path(cand)
        if (p / "desktop_entry.py").is_file() and (p / marker_pkg).is_dir():
            return Located(name, "source", p)
    return Located(name, "missing", None,
                   f"{src_name} not found. Set `anvira config set services.{name}.source_dir <path>` "
                   f"or install a bundled {name}-server.")


def locate_aicl(layout: RuntimeLayout) -> Path | None:
    """The *new* AICL package root (the directory that contains the ``aicl`` package)."""
    import os
    for cand in (os.environ.get("ANVIRA_AICL_DIR"), REPO_ROOT / "AICL", layout.home / "services" / "aicl"):
        if cand and (Path(cand) / "aicl" / "bin" / "codec_api.py").is_file():
            return Path(cand)
    return None


def python_for_services(config: RuntimeConfig) -> str:
    return config.get("services.python") or sys.executable


def missing_modules(modules: tuple[str, ...]) -> list[str]:
    """Modules not importable by the *runtime's* interpreter (best effort for the service one)."""
    return [m for m in modules if importlib.util.find_spec(m) is None]


def resolve_port(config: RuntimeConfig, name: str, remembered: int | None = None) -> int:
    configured = config.get(f"services.{name}.port")
    return configured or remembered or free_port()


def build_orcha_spec(layout: RuntimeLayout, config: RuntimeConfig, secrets: SecretStore,
                     port: int, loc: Located) -> ServiceSpec:
    state_dir = layout.data_dir / "orcha"
    state_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "ORCHA_PORT": str(port), "ORCHA_API_TOKEN": secrets.get("orcha_token"),
        "ORCHA_STATE_DIR": str(state_dir),
        "ORCHA_TOOL_OUTPUT_DIR": str(state_dir / "tool-outputs"),
        "ORCHA_TRANSCRIPT_DIR": str(state_dir / "transcripts"),
        "ANVIRA_RUNTIME_SERVICE": "orcha", "ANVIRA_RUNTIME_PORT": str(port),
        "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
    }
    if loc.mode == "executable":
        argv, cwd = [str(loc.path)], str(state_dir)
    else:
        argv, cwd = [python_for_services(config), str(loc.path / "desktop_entry.py")], str(state_dir)
        env["PYTHONPATH"] = str(loc.path)
    # Agents run commands like `python app.py`. If the user has no Python on PATH, fall back to the runtime's own (appended, so a
    # Python the user installed themselves always wins).
    env["PATH"] = os.environ.get("PATH", "") + os.pathsep + str(Path(python_for_services(config)).parent)
    return ServiceSpec(
        name="orcha", argv=argv, cwd=cwd, env=env, log_path=layout.service_log("orcha"), port=port,
        health_url=f"http://127.0.0.1:{port}/v1/health",
        ready_timeout_s=config.get("supervisor.start_timeout_s"), meta={"mode": loc.mode})


def build_nomi_spec(layout: RuntimeLayout, config: RuntimeConfig, secrets: SecretStore,
                    port: int, loc: Located) -> ServiceSpec:
    data = layout.data_dir / "nomi"
    data.mkdir(parents=True, exist_ok=True)
    # Nomi's desktop_entry persists SECRET_KEY under NOMI_DATA_DIR only when not in the env;
    # we always supply the runtime-owned key so the JWT signing key lives in runtime secrets.
    env = {
        "NOMI_DATA_DIR": str(data), "NOMI_PORT": str(port), "SECRET_KEY": secrets.get("nomi_secret_key"),
        "MEMORY_MD_DIR": str(data / "memory_md"), "LOG_LEVEL": "INFO",
        "AUTH_RATE_LIMIT_MAX_ATTEMPTS": "200", "ANVIRA_RUNTIME_SERVICE": "nomi",
        "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
    }
    if loc.mode == "executable":
        argv = [str(loc.path)]
    else:
        argv = [python_for_services(config), str(loc.path / "desktop_entry.py")]
        env["PYTHONPATH"] = str(loc.path)
    # cwd is the runtime data dir (not the Nomi source tree) so a stray developer
    # ``.env`` next to the source can never override runtime-owned settings.
    return ServiceSpec(
        name="nomi", argv=argv, cwd=str(data), env=env, log_path=layout.service_log("nomi"), port=port,
        health_url=f"http://127.0.0.1:{port}/health",
        ready_timeout_s=config.get("supervisor.start_timeout_s"), meta={"mode": loc.mode})
