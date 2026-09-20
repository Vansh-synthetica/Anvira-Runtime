"""Runtime configuration: one JSON file, dotted keys, validated types.

Secrets (tokens, provider API keys) are NOT stored here — see
``security.secrets``. This file is safe to print, share and back up.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .paths import RuntimeLayout

DEFAULT_API_PORT = 47615

DEFAULTS: dict[str, Any] = {
    "api": {
        "host": "127.0.0.1",          # must be a loopback address
        "port": DEFAULT_API_PORT,     # 0 = pick a free port at start
        "allowed_origins": [],        # browser Origins allowed to call the API
    },
    "services": {
        "orcha": {"enabled": True, "port": 0, "source_dir": None, "executable": None},
        "nomi": {"enabled": True, "port": 0, "source_dir": None, "executable": None},
        "python": None,               # interpreter used to run ORCHA/Nomi (default: runtime's)
    },
    "supervisor": {
        "health_interval_s": 2.0,
        "restart_limit": 5,           # restarts allowed within restart_window_s
        "restart_window_s": 300.0,
        "start_timeout_s": 60.0,
    },
    "models": {
        "models_dir": None,           # where NEW downloads go; None = <runtime home>/models. Any path.
        "extra_dirs": [],             # more folders to scan for .gguf files (e.g. an existing Anvira/LM Studio folder)
        "discover_apps": True,        # find models already downloaded by Anvira / Notes / Study / Dev
        "autostart_active": True,     # start the active model with the runtime
        "context_size": None,         # None = pick automatically
        "gpu": "auto",                # auto | off | on
        "llama_server_path": None,    # explicit llama-server binary
        "huggingface_endpoint": "https://huggingface.co",
        "allow_downloads": True,
    },
    "orcha": {
        "allow_mock": False,          # allow ORCHA's built-in mock experts when no model is active
        "job_timeout_s": 900.0,
    },
    "memory": {
        "default_scope": "app",       # app | shared
    },
    "lifecycle": {
        "auto_stop": False,           # normally set by whoever starts the daemon (apps: on-demand; `anvira runtime start`: persistent)
        "idle_grace_s": 30.0,         # shut down this long after the last app closed and nothing is running
        "lease_ttl_s": 45.0,          # an app's lease expires this long after its last heartbeat
    },
    "logging": {"level": "INFO"},
}

_ENUMS = {
    "models.gpu": {"auto", "off", "on"},
    "memory.default_scope": {"app", "shared"},
    "logging.level": {"DEBUG", "INFO", "WARNING", "ERROR"},
}
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}
# Keys whose default is None but which accept a string when set.
_NULLABLE_STR = {
    "services.orcha.source_dir", "services.orcha.executable",
    "services.nomi.source_dir", "services.nomi.executable", "services.python",
    "models.models_dir", "models.llama_server_path",
}
_NULLABLE_INT = {"models.context_size"}


class ConfigError(ValueError):
    """Invalid configuration key or value."""


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


_FLAT_DEFAULTS = _flatten(DEFAULTS)


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    for k, v in over.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _merge(base[k], v)
        elif k in base:
            base[k] = v
    return base


def coerce(key: str, raw: Any) -> Any:
    """Validate/convert ``raw`` (often a CLI string) for ``key``."""
    if key not in _FLAT_DEFAULTS:
        raise ConfigError(f"Unknown config key '{key}'. Run `anvira config list` to see valid keys.")
    default = _FLAT_DEFAULTS[key]
    value: Any = raw
    if key in _NULLABLE_STR:
        if raw is None or (isinstance(raw, str) and raw.strip().lower() in ("", "null", "none")):
            return None
        return str(raw)
    if key in _NULLABLE_INT:
        if raw is None or (isinstance(raw, str) and raw.strip().lower() in ("", "null", "none", "auto")):
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise ConfigError(f"'{key}' expects an integer or null") from None
    if isinstance(default, bool):
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str) and raw.lower() in ("true", "1", "yes", "on"):
            return True
        if isinstance(raw, str) and raw.lower() in ("false", "0", "no", "off"):
            return False
        raise ConfigError(f"'{key}' expects true or false")
    if isinstance(default, int):
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ConfigError(f"'{key}' expects an integer") from None
        if value < 0:
            raise ConfigError(f"'{key}' must be >= 0")
    elif isinstance(default, float):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ConfigError(f"'{key}' expects a number") from None
        if value < 0:
            raise ConfigError(f"'{key}' must be >= 0")
    elif isinstance(default, list):
        if isinstance(raw, str):
            raw = [p.strip() for p in raw.split(",") if p.strip()]
        if not isinstance(raw, list):
            raise ConfigError(f"'{key}' expects a list")
        value = [str(x) for x in raw]
    elif isinstance(default, str):
        value = str(raw)
    if key in _ENUMS and value not in _ENUMS[key]:
        raise ConfigError(f"'{key}' must be one of: {', '.join(sorted(_ENUMS[key]))}")
    if key == "api.host" and value not in _LOOPBACK:
        raise ConfigError(
            "api.host must be a loopback address (127.0.0.1, ::1 or localhost); "
            "the runtime never listens on external interfaces."
        )
    if key == "api.port" and not (0 <= value <= 65535):
        raise ConfigError("api.port must be between 0 and 65535")
    return value


class RuntimeConfig:
    """Loaded configuration plus load diagnostics."""

    def __init__(self, layout: RuntimeLayout, data: dict[str, Any] | None = None,
                 load_error: str | None = None):
        self.layout = layout
        self.data: dict[str, Any] = data if data is not None else copy.deepcopy(DEFAULTS)
        self.load_error = load_error

    # -- persistence -------------------------------------------------------
    @classmethod
    def load(cls, layout: RuntimeLayout) -> "RuntimeConfig":
        data = copy.deepcopy(DEFAULTS)
        error = None
        path = layout.config_file
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    raise ValueError("top-level value must be an object")
                _merge(data, loaded)
                # Re-validate every persisted value so a hand-edited bad value
                # is reported rather than silently trusted.
                for key, val in _flatten(data).items():
                    try:
                        coerce(key, val)
                    except ConfigError as exc:
                        raise ValueError(str(exc)) from None
            except (OSError, ValueError) as exc:
                error = f"{path}: {exc}"
                data = copy.deepcopy(DEFAULTS)
        return cls(layout, data, error)

    def save(self) -> None:
        self.layout.config_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.layout.config_dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, indent=2, sort_keys=True)
            os.replace(tmp, self.layout.config_file)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self.load_error = None

    # -- access ------------------------------------------------------------
    def get(self, key: str) -> Any:
        if key not in _FLAT_DEFAULTS:
            raise ConfigError(f"Unknown config key '{key}'.")
        node: Any = self.data
        for part in key.split("."):
            node = node[part]
        return copy.deepcopy(node)

    def set(self, key: str, raw: Any) -> Any:
        value = coerce(key, raw)
        node = self.data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
        return value

    def flat(self) -> dict[str, Any]:
        return _flatten(self.data)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    # -- derived -----------------------------------------------------------
    def models_dir(self) -> Path:
        if os.environ.get("ANVIRA_MODELS_DIR"):
            return self.layout.models_dir
        custom = self.get("models.models_dir")
        return Path(custom).expanduser() if custom else self.layout.models_dir

    def extra_model_dirs(self) -> list[Path]:
        return [Path(p).expanduser() for p in self.get("models.extra_dirs")]


def flat_keys() -> list[str]:
    return sorted(_FLAT_DEFAULTS)
