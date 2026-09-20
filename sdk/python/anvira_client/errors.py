"""Client-side errors. Every runtime failure maps to :class:`AnviraError` with a stable ``code``."""
from __future__ import annotations

from typing import Any


class AnviraError(Exception):
    def __init__(self, code: str, message: str, status: int | None = None, hint: str | None = None,
                 details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code, self.message, self.status, self.hint, self.details = code, message, status, hint, details or {}

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}" + (f"\nHint: {self.hint}" if self.hint else "")

    @classmethod
    def from_response(cls, status: int, body: Any) -> "AnviraError":
        err = (body or {}).get("error") if isinstance(body, dict) else None
        if isinstance(err, dict):
            code = err.get("code", "error")
            klass = _BY_CODE.get(code, AnviraError)
            return klass(code, err.get("message", "Request failed"), status, err.get("hint"), err.get("details"))
        return cls("http_error", f"HTTP {status}", status)


class RuntimeNotInstalled(AnviraError):
    """Anvira Runtime is not installed. Ask the user before installing."""


class RuntimeNotRunning(AnviraError):
    """The runtime is installed but not running (and auto-start was off or failed)."""


class RuntimeStartFailed(AnviraError):
    """The runtime could not be started. Run `anvira doctor`."""


class IncompatibleRuntime(AnviraError):
    """The installed runtime does not satisfy the app's version requirement."""


class PermissionDenied(AnviraError):
    """The app lacks a permission the user must grant."""


class NoModel(AnviraError):
    """No model is installed or selected."""


class InstallDeclined(AnviraError):
    """The user declined to install the runtime."""


_BY_CODE = {
    "permission_denied": PermissionDenied,
    "no_model": NoModel,
    "unauthorized": PermissionDenied,
}
