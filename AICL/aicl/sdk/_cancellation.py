"""
Cancellation support for AICL SDK operations.

Provides a CancellationToken that can be passed to async operations
to request early termination.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional


class CancellationToken:
    """Thread-safe cancellation token.

    Usage::

        token = CancellationToken()

        # In a task:
        while not token.is_cancelled:
            process_next()

        # From another thread:
        token.cancel()
    """

    __slots__ = ("_cancelled", "_event", "_callbacks", "_lock")

    def __init__(self):
        self._cancelled = False
        self._event = threading.Event()
        self._callbacks: list[Callable[[], Any]] = []
        self._lock = threading.Lock()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        """Request cancellation. Safe to call multiple times."""
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            self._event.set()
            for cb in self._callbacks:
                try:
                    cb()
                except Exception:
                    pass  # Don't let callback errors propagate

    def on_cancelled(self, callback: Callable[[], Any]) -> None:
        """Register a callback to be invoked when cancellation is requested."""
        with self._lock:
            if self._cancelled:
                try:
                    callback()
                except Exception:
                    pass
            else:
                self._callbacks.append(callback)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until cancelled or timeout. Returns True if cancelled."""
        return self._event.wait(timeout=timeout)

    def check(self) -> None:
        """Raise CancelledError if cancelled. Use as a checkpoint."""
        if self._cancelled:
            from .errors import CancelledError
            raise CancelledError()

    def __repr__(self) -> str:
        state = "cancelled" if self._cancelled else "active"
        return f"<CancellationToken {state}>"


class _NullCancellationToken:
    """Sentinel for when no cancellation token is provided."""

    @property
    def is_cancelled(self) -> bool:
        return False

    def cancel(self) -> None:
        pass

    def on_cancelled(self, callback: Callable[[], Any]) -> None:
        pass

    def wait(self, timeout: Optional[float] = None) -> bool:
        return False

    def check(self) -> None:
        pass


NULL_TOKEN = _NullCancellationToken()
