"""Structured runtime errors.

Every failure that reaches an application is one of these, serialised as
``{"error": {"code", "message", "hint", "status", "details"}}`` so callers can
branch on ``code`` instead of parsing text.
"""
from __future__ import annotations

from typing import Any


class RuntimeApiError(Exception):
    def __init__(self, code: str, message: str, status: int = 500,
                 hint: str | None = None, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.hint = hint
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        err: dict[str, Any] = {"code": self.code, "message": self.message, "status": self.status}
        if self.hint:
            err["hint"] = self.hint
        if self.details:
            err["details"] = self.details
        return {"error": err}


def no_model(installed: list[str] | None = None) -> RuntimeApiError:
    """Nothing to run. Say which of the two situations it is: nothing installed, or installed but none selected."""
    if installed:
        pick = installed[0]
        return RuntimeApiError(
            "no_model", f"No model is selected ({len(installed)} installed).", 409,
            hint=f"Pick one: `anvira model use {pick}`  (see them all with `anvira model list --installed`).",
            details={"installed": installed[:20]})
    return RuntimeApiError(
        "no_model", "No model is installed.", 409,
        hint="See `anvira model list`. Add a folder that already has .gguf files (`anvira model dirs add <folder>`) or download one (`anvira model install <id>`).",
        details={"installed": []})


def not_found(kind: str, ident: str) -> RuntimeApiError:
    return RuntimeApiError(f"{kind}_not_found", f"{kind.capitalize()} '{ident}' was not found.", 404)


def service_unavailable(name: str, detail: str = "") -> RuntimeApiError:
    return RuntimeApiError(
        f"{name}_unavailable", f"The {name.upper() if name == 'orcha' else name.capitalize()} "
        f"subsystem is not available.{(' ' + detail) if detail else ''}", 503,
        hint="Run `anvira doctor` for diagnostics and `anvira runtime logs --service " + name + "`.")
