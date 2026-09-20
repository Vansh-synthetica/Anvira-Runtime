"""Platform-specific runtime directory layout.

One runtime home per OS user, shared by every Anvira application. Runtime
state lives here; application workspace data (notes, study sets, projects)
does NOT — each app keeps that in its own location.

Override the whole tree with ``ANVIRA_RUNTIME_HOME`` (used by tests and by
portable installs). The models directory can additionally be redirected with
``ANVIRA_MODELS_DIR`` or the ``models.models_dir`` config key so that models
can live on a bigger drive, or be shared with an existing Anvira install.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

APP_DIR_NAME_WIN_MAC = "AnviraRuntime"
APP_DIR_NAME_POSIX = "anvira-runtime"


@dataclass(frozen=True)
class RuntimeLayout:
    """Every directory the runtime reads or writes."""

    home: Path        # install root: bin/, services/, runtime install marker
    config_dir: Path  # config.json
    data_dir: Path    # nomi database, orcha checkpoints (runtime-owned data)
    models_dir: Path  # GGUF files + registry.json
    logs_dir: Path
    cache_dir: Path
    state_dir: Path   # runtime.json (discovery), secrets, pids, apps registry
    bin_dir: Path     # inference backends (llama-server)

    # -- well-known files ------------------------------------------------
    @property
    def config_file(self) -> Path:
        return self.config_dir / "config.json"

    @property
    def discovery_file(self) -> Path:
        return self.state_dir / "runtime.json"

    @property
    def secrets_file(self) -> Path:
        return self.state_dir / "secrets.json"

    @property
    def apps_file(self) -> Path:
        return self.state_dir / "apps.json"

    @property
    def providers_file(self) -> Path:
        return self.state_dir / "providers.json"

    @property
    def model_state_file(self) -> Path:
        return self.state_dir / "models.json"

    @property
    def children_file(self) -> Path:
        return self.state_dir / "children.json"

    @property
    def install_file(self) -> Path:
        return self.home / "install.json"

    @property
    def app_tokens_dir(self) -> Path:
        return self.state_dir / "app-tokens"

    @property
    def runtime_log(self) -> Path:
        return self.logs_dir / "runtime.log"

    def service_log(self, name: str) -> Path:
        return self.logs_dir / f"{name}.log"

    def dirs(self) -> list[Path]:
        return [
            self.home, self.config_dir, self.data_dir, self.models_dir,
            self.logs_dir, self.cache_dir, self.state_dir, self.bin_dir,
            self.app_tokens_dir,
        ]

    def ensure(self) -> "RuntimeLayout":
        for d in self.dirs():
            d.mkdir(parents=True, exist_ok=True)
        return self

    def as_dict(self) -> dict[str, str]:
        return {
            "home": str(self.home), "config": str(self.config_dir),
            "data": str(self.data_dir), "models": str(self.models_dir),
            "logs": str(self.logs_dir), "cache": str(self.cache_dir),
            "state": str(self.state_dir), "bin": str(self.bin_dir),
        }


def pointer_target(default_home: Path) -> Path | None:
    """Where a *custom-location* install lives: ``<default home>/location.json`` -> ``{"home": "<path>"}``.

    Lets a user put Anvira Runtime on any drive and still have every app find it. Ignored when
    ``ANVIRA_RUNTIME_HOME`` is set, and when the target is gone (falls back to the default location).
    """
    try:
        import json
        home = json.loads((default_home / "location.json").read_text(encoding="utf-8")).get("home")
    except (OSError, ValueError, AttributeError):
        return None
    return Path(home) if home and (Path(home) / "install.json").is_file() else None


def _rooted(root: Path) -> "RuntimeLayout":
    return RuntimeLayout(home=root, config_dir=root / "config", data_dir=root / "data", models_dir=root / "models",
                         logs_dir=root / "logs", cache_dir=root / "cache", state_dir=root / "state", bin_dir=root / "bin")


def _home_from_env(env: dict[str, str]) -> Path:
    return Path(env.get("USERPROFILE") or env.get("HOME") or Path.home())


def resolve_layout(env: dict[str, str] | None = None, platform: str | None = None) -> RuntimeLayout:
    """Compute the layout for the current (or given) platform.

    ``platform`` accepts ``win32`` / ``darwin`` / ``linux`` so the per-OS
    rules can be unit-tested from any host.
    """
    env = dict(os.environ if env is None else env)
    plat = platform or sys.platform

    override = env.get("ANVIRA_RUNTIME_HOME")
    if override:
        root = Path(override).expanduser()
        layout = RuntimeLayout(
            home=root, config_dir=root / "config", data_dir=root / "data",
            models_dir=root / "models", logs_dir=root / "logs",
            cache_dir=root / "cache", state_dir=root / "state", bin_dir=root / "bin",
        )
    elif plat.startswith("win"):
        local = Path(env.get("LOCALAPPDATA") or (_home_from_env(env) / "AppData" / "Local"))
        root = local / APP_DIR_NAME_WIN_MAC
        layout = RuntimeLayout(
            home=root, config_dir=root / "config", data_dir=root / "data",
            models_dir=root / "models", logs_dir=root / "logs",
            cache_dir=root / "cache", state_dir=root / "state", bin_dir=root / "bin",
        )
    elif plat == "darwin":
        home = _home_from_env(env)
        root = home / "Library" / "Application Support" / APP_DIR_NAME_WIN_MAC
        layout = RuntimeLayout(
            home=root, config_dir=root / "config", data_dir=root / "data",
            models_dir=root / "models",
            logs_dir=home / "Library" / "Logs" / APP_DIR_NAME_WIN_MAC,
            cache_dir=home / "Library" / "Caches" / APP_DIR_NAME_WIN_MAC,
            state_dir=root / "state", bin_dir=root / "bin",
        )
    else:  # linux / other POSIX — XDG base directories
        home = _home_from_env(env)
        config_root = Path(env.get("XDG_CONFIG_HOME") or home / ".config")
        data_root = Path(env.get("XDG_DATA_HOME") or home / ".local" / "share")
        state_root = Path(env.get("XDG_STATE_HOME") or home / ".local" / "state")
        cache_root = Path(env.get("XDG_CACHE_HOME") or home / ".cache")
        d = data_root / APP_DIR_NAME_POSIX
        layout = RuntimeLayout(
            home=d, config_dir=config_root / APP_DIR_NAME_POSIX, data_dir=d / "data",
            models_dir=d / "models", logs_dir=state_root / APP_DIR_NAME_POSIX / "logs",
            cache_dir=cache_root / APP_DIR_NAME_POSIX,
            state_dir=state_root / APP_DIR_NAME_POSIX, bin_dir=d / "bin",
        )

    if not override:
        custom = pointer_target(layout.home)
        if custom:
            layout = _rooted(custom)
    models_override = env.get("ANVIRA_MODELS_DIR")
    if models_override:
        layout = RuntimeLayout(**{**layout.__dict__, "models_dir": Path(models_override).expanduser()})
    return layout
