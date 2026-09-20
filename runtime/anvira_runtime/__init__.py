"""Anvira Runtime — shared local intelligence runtime for Anvira applications.

Composes ORCHA (orchestration), Nomi (memory/identity/permissions) and AICL
(inter-module protocol) behind one local, token-authenticated API.
"""
from .version import API_VERSION, RUNTIME_VERSION

__version__ = RUNTIME_VERSION
__all__ = ["API_VERSION", "RUNTIME_VERSION", "__version__"]
