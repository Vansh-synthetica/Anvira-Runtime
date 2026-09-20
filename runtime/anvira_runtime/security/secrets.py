"""File-backed secret storage, app registry and log redaction.

Threat model (be honest about it): the runtime protects against
  * remote hosts and other machines (loopback bind only),
  * web pages in the user's browser (token required, Origin/Host checks),
  * an application using capabilities it was never granted (per-app tokens
    carry explicit permissions).
It does NOT defend against malicious code running as the same OS user: such
code can read the token files. Files are created owner-only (0600) on POSIX;
on Windows they inherit the per-user profile ACL.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..config.paths import RuntimeLayout

# ---------------------------------------------------------------- permissions
PERMISSIONS: dict[str, str] = {
    "models.read": "List models, hardware and the active model",
    "models.select": "Change the active model",
    "models.register": "Tell the runtime where a model file/folder already lives (no download, no copy)",
    "models.manage": "Install, remove and configure models/providers (downloads, disk use)",
    "chat": "Send chat completions to the active model",
    "orcha.run": "Start ORCHA jobs (tasks, agents) and read/cancel own jobs",
    "orcha.read": "Read ORCHA status",
    "orcha.exec": "Let ORCHA agents RUN COMMANDS (terminal/git) on this computer inside a workspace - the user grants this per app",
    "memory.read": "Search memory in the app's own namespace",
    "memory.write": "Store memory in the app's own namespace",
    "memory.shared": "Read/write the shared (cross-app) memory namespace",
    "context.read": "Search the app's own document index",
    "context.write": "Create/update/delete the app's own shared resources and index documents",
    "context.share": "Share the app's OWN resources with other apps (grant/revoke, approve access requests)",
    "config.read": "Read runtime configuration",
    "config.write": "Change runtime configuration",
    "runtime.admin": "Start/stop/restart the runtime, read logs and diagnostics",
    "apps.admin": "Register, grant and revoke applications",
}

DEFAULT_APP_PERMISSIONS = (
    "models.read", "models.select", "models.register", "chat", "orcha.run", "orcha.read",
    "memory.read", "memory.write", "context.read", "context.write", "context.share", "config.read",
)

APP_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,47}$")


class AuthError(Exception):
    """Authentication or authorization failure."""

    def __init__(self, code: str, message: str, status: int = 401):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass
class Principal:
    kind: str                      # "owner" | "app"
    app_id: str | None = None
    permissions: frozenset[str] = field(default_factory=frozenset)

    def has(self, permission: str) -> bool:
        return self.kind == "owner" or permission in self.permissions

    def require(self, permission: str) -> None:
        if not self.has(permission):
            who = self.app_id or "caller"
            raise AuthError(
                "permission_denied",
                f"'{who}' does not have permission '{permission}'. "
                f"Grant it with: anvira app grant {self.app_id or '<app-id>'} {permission}",
                403,
            )

    @property
    def namespace(self) -> str:
        return f"app:{self.app_id}" if self.kind == "app" else "owner"


# ------------------------------------------------------------------ redaction
_REDACTIONS = [
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(x-(?:anvira|orcha)-token['\":= ]+)[A-Za-z0-9._~+/=-]{8,}"), r"\1[REDACTED]"),
    (re.compile(r"(?i)((?:api[_-]?key|token|secret|password)['\"]?\s*[:=]\s*['\"]?)[^\s'\",}]{6,}"), r"\1[REDACTED]"),
    (re.compile(r"\b(sk|gsk|hf|ghp|xox[bp])[-_][A-Za-z0-9_-]{16,}"), "[REDACTED]"),
]


def redact(text: str) -> str:
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


def redact_obj(obj: Any) -> Any:
    """Recursively mask values under secret-looking keys."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if re.search(r"(?i)(api[_-]?key|token|secret|password|authorization)", str(k)):
                out[k] = "[REDACTED]" if v else v
            else:
                out[k] = redact_obj(v)
        return out
    if isinstance(obj, list):
        return [redact_obj(x) for x in obj]
    if isinstance(obj, str):
        return redact(obj)
    return obj


# --------------------------------------------------------------- atomic files
def write_private(path: Path, data: str) -> None:
    """Atomically write ``data`` to ``path`` with owner-only permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------- secrets
class SecretStore:
    """Runtime-internal secrets: owner/registration tokens and service creds.

    Never returned by the API and never logged.
    """

    def __init__(self, layout: RuntimeLayout):
        self.layout = layout
        self._lock = threading.Lock()
        self._data: dict[str, str] = read_json(layout.secrets_file, {})
        changed = False
        for key, nbytes in (
            ("owner_token", 32), ("register_token", 32), ("orcha_token", 32),
            ("nomi_secret_key", 32), ("nomi_password", 24),
        ):
            if not self._data.get(key):
                self._data[key] = secrets.token_hex(nbytes)
                changed = True
        if changed:
            self._flush()
        self._publish_token_files()

    def _flush(self) -> None:
        write_private(self.layout.secrets_file, json.dumps(self._data, indent=2))
        self._publish_token_files()

    def _publish_token_files(self) -> None:
        """Clients read only the token they need, never the whole secrets file:
        ``owner.token`` (CLI / full control) and ``register.token`` (lets an
        app register itself with default permissions)."""
        write_private(self.layout.state_dir / "owner.token", self._data["owner_token"])
        write_private(self.layout.state_dir / "register.token", self._data["register_token"])

    def get(self, key: str) -> str:
        return self._data[key]

    def rotate(self, key: str) -> str:
        with self._lock:
            self._data[key] = secrets.token_hex(32)
            self._flush()
            return self._data[key]

    def matches(self, key: str, provided: str) -> bool:
        return bool(provided) and hmac.compare_digest(_hash(provided), _hash(self._data[key]))


# ----------------------------------------------------------------- app registry
class AppRegistry:
    """Registered applications and their granted permissions."""

    def __init__(self, layout: RuntimeLayout, secrets_store: SecretStore):
        self.layout = layout
        self.secrets = secrets_store
        self._lock = threading.Lock()
        self._apps: dict[str, dict[str, Any]] = read_json(layout.apps_file, {})

    def _flush(self) -> None:
        write_private(self.layout.apps_file, json.dumps(self._apps, indent=2, sort_keys=True))

    # -- registration --------------------------------------------------------
    def register(self, app_id: str, name: str | None = None,
                 requested: Iterable[str] = (), *, allow_rotate: bool = False) -> tuple[dict[str, Any], str]:
        """Register (or re-register) an app. Returns (public_record, plaintext_token).

        Permissions beyond ``DEFAULT_APP_PERMISSIONS`` are recorded as
        *requested* but not granted; the user grants them explicitly.
        """
        if not APP_ID_RE.match(app_id or ""):
            raise AuthError("invalid_app_id",
                            "app_id must be 2-48 chars: lowercase letters, digits, '.', '_' or '-'.", 400)
        requested = [p for p in requested if p in PERMISSIONS]
        with self._lock:
            existing = self._apps.get(app_id)
            if existing and not allow_rotate:
                raise AuthError(
                    "app_exists",
                    f"App '{app_id}' is already registered. Use its saved token, or have the "
                    f"user run `anvira app revoke {app_id}` and register again.", 409)
            token = "anv_" + secrets.token_urlsafe(32)
            record = {
                "app_id": app_id,
                "name": name or (existing or {}).get("name") or app_id,
                "token_hash": _hash(token),
                "permissions": list((existing or {}).get("permissions") or DEFAULT_APP_PERMISSIONS),
                "requested": sorted(set(requested) - set(DEFAULT_APP_PERMISSIONS)),
                "created_at": (existing or {}).get("created_at") or time.time(),
                "last_seen": None,
            }
            self._apps[app_id] = record
            self._flush()
        return self.public(record), token

    def authenticate(self, token: str) -> Principal | None:
        if not token:
            return None
        if self.secrets.matches("owner_token", token):
            return Principal("owner", None, frozenset(PERMISSIONS))
        digest = _hash(token)
        for rec in self._apps.values():
            if hmac.compare_digest(rec["token_hash"], digest):
                rec["last_seen"] = time.time()
                return Principal("app", rec["app_id"], frozenset(rec["permissions"]))
        return None

    # -- management ----------------------------------------------------------
    @staticmethod
    def public(rec: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in rec.items() if k != "token_hash"}

    def list(self) -> list[dict[str, Any]]:
        return [self.public(r) for r in sorted(self._apps.values(), key=lambda r: r["app_id"])]

    def get(self, app_id: str) -> dict[str, Any] | None:
        rec = self._apps.get(app_id)
        return self.public(rec) if rec else None

    def grant(self, app_id: str, permissions: Iterable[str]) -> dict[str, Any]:
        perms = list(permissions)
        bad = [p for p in perms if p not in PERMISSIONS]
        if bad:
            raise AuthError("unknown_permission", f"Unknown permission(s): {', '.join(bad)}", 400)
        with self._lock:
            rec = self._require(app_id)
            rec["permissions"] = sorted(set(rec["permissions"]) | set(perms))
            rec["requested"] = sorted(set(rec.get("requested", [])) - set(perms))
            self._flush()
            return self.public(rec)

    def deny(self, app_id: str, permissions: Iterable[str]) -> dict[str, Any]:
        with self._lock:
            rec = self._require(app_id)
            rec["permissions"] = sorted(set(rec["permissions"]) - set(permissions))
            self._flush()
            return self.public(rec)

    def revoke(self, app_id: str) -> None:
        with self._lock:
            self._require(app_id)
            del self._apps[app_id]
            self._flush()

    def _require(self, app_id: str) -> dict[str, Any]:
        rec = self._apps.get(app_id)
        if not rec:
            raise AuthError("app_not_found", f"App '{app_id}' is not registered.", 404)
        return rec
