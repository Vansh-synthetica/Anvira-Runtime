"""Where the shared runtime lives (mirrors ``anvira_runtime.config.paths``).

Kept dependency-free and duplicated on purpose: a client must be able to find
(or notice the absence of) the runtime *before* it is installed. A parity test
(``tests/runtime/test_paths.py``) guarantees both copies agree on every OS.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

WIN_MAC_NAME = "AnviraRuntime"
POSIX_NAME = "anvira-runtime"


def pointer_target(default_home: Path) -> Path | None:
    """A custom-location install: ``<default home>/location.json`` -> ``{"home": "<path>"}`` (see anvira_runtime.config.paths)."""
    try:
        home = json.loads((default_home / "location.json").read_text(encoding="utf-8")).get("home")
    except (OSError, ValueError, AttributeError):
        return None
    return Path(home) if home and (Path(home) / "install.json").is_file() else None


def runtime_dirs(env: dict[str, str] | None = None, platform: str | None = None,
                 ignore_pointer: bool = False) -> dict[str, Path]:
    """Return ``{"home", "state", "config", "logs", "models", "bin"}`` for the current user."""
    env = dict(os.environ if env is None else env)
    plat = platform or sys.platform
    override = env.get("ANVIRA_RUNTIME_HOME")
    home_dir = Path(env.get("USERPROFILE") or env.get("HOME") or Path.home())
    if override:
        root = Path(override).expanduser()
        d = {"home": root, "state": root / "state", "config": root / "config", "logs": root / "logs",
             "models": root / "models", "bin": root / "bin"}
    elif plat.startswith("win"):
        root = Path(env.get("LOCALAPPDATA") or (home_dir / "AppData" / "Local")) / WIN_MAC_NAME
        d = {"home": root, "state": root / "state", "config": root / "config", "logs": root / "logs",
             "models": root / "models", "bin": root / "bin"}
    elif plat == "darwin":
        root = home_dir / "Library" / "Application Support" / WIN_MAC_NAME
        d = {"home": root, "state": root / "state", "config": root / "config",
             "logs": home_dir / "Library" / "Logs" / WIN_MAC_NAME, "models": root / "models", "bin": root / "bin"}
    else:
        data = Path(env.get("XDG_DATA_HOME") or home_dir / ".local" / "share") / POSIX_NAME
        state = Path(env.get("XDG_STATE_HOME") or home_dir / ".local" / "state") / POSIX_NAME
        d = {"home": data, "state": state, "config": Path(env.get("XDG_CONFIG_HOME") or home_dir / ".config") / POSIX_NAME,
             "logs": state / "logs", "models": data / "models", "bin": data / "bin"}
    if not override and not ignore_pointer:
        custom = pointer_target(d["home"])
        if custom:
            root = custom
            d = {"home": root, "state": root / "state", "config": root / "config", "logs": root / "logs",
                 "models": root / "models", "bin": root / "bin"}
    if env.get("ANVIRA_MODELS_DIR"):
        d["models"] = Path(env["ANVIRA_MODELS_DIR"]).expanduser()
    return d
