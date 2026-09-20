"""
Nomi desktop entry point.
--------------------------
Used only for the bundled one-click desktop launch (via Electron ->
PyInstaller executable). The Docker Compose path (docker-compose.yml)
remains the recommended flow for developers who want a real Postgres
instance; this script is the SQLite-backed equivalent for end users who
just want to run the app.

What it does, in order:
  1. Reads NOMI_DATA_DIR from the environment (set by Electron to a
     writable per-user directory) and derives a SQLITE database path
     and a persisted SECRET_KEY from it.
  2. Runs Alembic migrations programmatically against that database
     (idempotent — safe to run on every launch; Alembic no-ops once
     the schema is current).
  3. Starts uvicorn serving the existing FastAPI app unchanged.

This file intentionally does NOT duplicate any application logic —
app.main.app is used as-is. Only startup/environment wiring lives here.
"""
import os
import secrets
import sys
from pathlib import Path


def _ensure_env_defaults() -> None:
    """
    Populates the environment with desktop-appropriate defaults BEFORE
    app.config.settings is imported anywhere (Pydantic settings read
    the environment at import time), so this must run first.
    """
    data_dir = Path(os.environ.get("NOMI_DATA_DIR", Path.home() / ".anvira" / "nomi"))
    data_dir.mkdir(parents=True, exist_ok=True)

    db_path = data_dir / "nomi.db"
    os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")

    # SECRET_KEY must be stable across restarts (it signs JWTs — regenerating
    # it on every launch would invalidate every existing session). Persist
    # it to a file in the same data directory on first run.
    secret_file = data_dir / ".secret_key"
    if "SECRET_KEY" not in os.environ:
        if secret_file.exists():
            os.environ["SECRET_KEY"] = secret_file.read_text().strip()
        else:
            generated = secrets.token_hex(32)  # 64 hex chars, well over the min length
            secret_file.write_text(generated)
            os.environ["SECRET_KEY"] = generated

    os.environ.setdefault("ENVIRONMENT", "development")
    os.environ.setdefault("DEBUG", "false")
    os.environ.setdefault(
        "BACKEND_CORS_ORIGINS",
        '["http://localhost:5173","http://localhost:5174","app://.","file://"]',
    )
    # Redis is declared as a dependency but not actually used anywhere in
    # the application code (verified) — no desktop-specific handling needed.


def _run_migrations() -> None:
    """Programmatic equivalent of `alembic upgrade head`."""
    from alembic import command
    from alembic.config import Config

    if getattr(sys, "frozen", False):
        # Running as a PyInstaller-frozen executable: migrations are
        # bundled as data files relative to the executable, not the
        # source tree.
        base_dir = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        base_dir = Path(__file__).resolve().parent

    alembic_cfg = Config()
    alembic_cfg.set_main_option("script_location", str(base_dir / "migrations"))
    # settings.database_url (read inside env.py) already reflects the
    # DATABASE_URL we set in _ensure_env_defaults() above.
    #
    # Migrations 0002 and 0003 check column/index existence directly
    # against the live schema before adding anything, rather than
    # trusting alembic_version to be perfectly in sync with it — so this
    # is safe to run even against a dev SQLite file whose schema drifted
    # ahead of its recorded version (e.g. from an interrupted previous
    # run, or surviving across a code update during testing).
    command.upgrade(alembic_cfg, "head")


def main() -> None:
    _ensure_env_defaults()
    _run_migrations()

    import uvicorn

    from app.main import app

    port = int(os.environ.get("NOMI_PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
