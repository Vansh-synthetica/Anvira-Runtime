"""
Orcha desktop entry point.
---------------------------
Used only for the bundled one-click desktop launch (via Electron ->
PyInstaller executable). Running `uvicorn orcha.api.server:app --port 8420`
manually (as documented for developers) remains fully supported and
unchanged — this file exists purely so Electron has a single frozen
executable to spawn, with no dependency on a system Python or manual
terminal command.

Orcha has no database, so unlike Nomi's desktop_entry.py there's no
migration step here — just environment defaults and a server start.
"""
import os


def main() -> None:
    # Orcha reads LOCAL_SERVER_MODEL / LOCAL_SERVER_BASE_URL / RUN_ALL_EXPERTS
    # etc. from the environment at import time (orcha/settings.py) — Anvira
    # sets these at runtime via the POST /v1/local-model endpoint instead,
    # so no desktop-specific defaults are required here beyond the port.
    port = int(os.environ.get("ORCHA_PORT", "8420"))

    import uvicorn

    from orcha.api.server import app

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
