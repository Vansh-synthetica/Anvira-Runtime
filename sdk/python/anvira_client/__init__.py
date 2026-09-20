"""Anvira Runtime client for Python applications (stdlib only)."""
from .client import SDK_API_VERSION, AnviraRuntime, Job
from .discovery import RuntimeInfo, probe
from .errors import (AnviraError, IncompatibleRuntime, InstallDeclined, NoModel, PermissionDenied,
                     RuntimeNotInstalled, RuntimeNotRunning, RuntimeStartFailed)

__version__ = "1.0.0"
__all__ = ["AnviraRuntime", "Job", "RuntimeInfo", "probe", "SDK_API_VERSION", "AnviraError", "IncompatibleRuntime",
           "InstallDeclined", "NoModel", "PermissionDenied", "RuntimeNotInstalled", "RuntimeNotRunning",
           "RuntimeStartFailed"]
