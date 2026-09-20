"""Find models that Anvira applications already downloaded.

Whoever downloaded a model — the main Anvira app, Anvira Notes, Anvira Study or
Anvira Dev — the runtime learns where it is, so every app can use it without a
second download. Each Electron app keeps its user data under a per-user folder
and (like the current Anvira) records a custom models folder in
``model-storage.json``; both that folder and the default ``<userData>/models``
are picked up. Discovery is read-only: it never moves or modifies app data.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

APP_NAMES = ("Anvira", "anvira", "Anvira Notes", "Anvira Study", "Anvira Dev")


def user_data_roots(env: dict[str, str] | None = None, platform: str | None = None) -> list[Path]:
    env = dict(os.environ if env is None else env)
    plat = platform or sys.platform
    home = Path(env.get("USERPROFILE") or env.get("HOME") or Path.home())
    if plat.startswith("win"):
        return [Path(env.get("APPDATA") or home / "AppData" / "Roaming")]
    if plat == "darwin":
        return [home / "Library" / "Application Support"]
    return [Path(env.get("XDG_CONFIG_HOME") or home / ".config")]


def discover_app_model_dirs(env: dict[str, str] | None = None, platform: str | None = None) -> dict[Path, str]:
    """``{models_dir: app_label}`` for every Anvira app data folder found on this machine."""
    found: dict[Path, str] = {}
    seen: set[str] = set()
    for root in user_data_roots(env, platform):
        for name in APP_NAMES:
            data = root / name
            if not data.is_dir():
                continue
            candidates: list[Path] = []
            try:
                cfg = json.loads((data / "model-storage.json").read_text(encoding="utf-8"))
                if isinstance(cfg, dict) and cfg.get("modelsDir"):
                    candidates.append(Path(str(cfg["modelsDir"])).expanduser())
            except (OSError, ValueError):
                pass
            candidates.append(data / "models")
            for c in candidates:
                key = os.path.normcase(str(c))
                if key not in seen and c.is_dir():
                    seen.add(key)
                    found[c] = name.title() if name == "anvira" else name
    return found
