"""AICL module bus.

This is where AICL becomes infrastructure. Every call from the API layer to a
runtime subsystem (ORCHA, Nomi, models, context) is framed as a genuine
AICL-BIN packet from the *new* ``aicl`` package: encoded with a CRC32
trailer, decoded and validated, dispatched by opcode/target, and answered with
a correlated ``OP_RETURN`` (or ``OP_ERROR`` carrying ``ErrorInfo``). Each hop
is stamped into the packet's trace and recorded for diagnostics.

What AICL does NOT do here, honestly: it is the in-process message format and
dispatch contract between modules. Nomi and ORCHA are still separate HTTP
services underneath, so the module handlers for them speak HTTP to those
services. When AICL's native (Rust) transport is built, this bus can move
those hops onto it without touching applications; ``native_available()``
reports whether that core is present.

Applications never build these packets — they use the runtime API/SDK.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..security.secrets import AuthError
from .errors import RuntimeApiError

Handler = Callable[[str, dict[str, Any], "CallContext"], Awaitable[Any]]


class BusUnavailable(RuntimeError):
    """The AICL package could not be imported."""


class CallContext:
    __slots__ = ("origin", "app_id", "deadline_ms", "message_id", "op")

    def __init__(self, origin: str, app_id: str | None, deadline_ms: int, message_id: str, op: int):
        self.origin, self.app_id, self.deadline_ms, self.message_id, self.op = (
            origin, app_id, deadline_ms, message_id, op)


def load_aicl(root: Path | None):
    """Import the new ``aicl`` package, adding ``root`` (dir containing ``aicl/``) to sys.path."""
    try:
        import aicl  # noqa: F401
    except ImportError:
        if root is None:
            raise BusUnavailable("The AICL package was not found (set ANVIRA_AICL_DIR or install localhousellm-aicl).")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            import aicl  # noqa: F401,F811
        except ImportError as exc:
            raise BusUnavailable(f"AICL at {root} could not be imported: {exc}") from exc
    import aicl as mod
    return mod


def native_available() -> bool:
    try:
        from aicl.sdk._ffi import get_lib
        get_lib()
        return True
    except Exception:  # noqa: BLE001 - OSError / NativeError / ImportError all mean "no"
        return False


class AICLBus:
    def __init__(self, root: Path | None):
        self.root = root
        self._aicl = load_aicl(root)
        from aicl.bin import constants as C, ops
        from aicl.bin.codec_api import decode, encode
        from aicl.bin.codec_packet import Packet
        from aicl.bin.identity import IDENTITY_MODULE, IDENTITY_RUNTIME
        from aicl.bin.symbol_types import S_STRING
        from aicl.bin.types import ErrorInfo, Identity, Symbol, TraceEntry
        self._C, self.ops = C, ops
        self._encode, self._decode, self._Packet = encode, decode, Packet
        self._Identity, self._Symbol, self._ErrorInfo, self._TraceEntry = Identity, Symbol, ErrorInfo, TraceEntry
        self._S_STRING, self._ID_MODULE, self._ID_RUNTIME = S_STRING, IDENTITY_MODULE, IDENTITY_RUNTIME
        self._handlers: dict[str, Handler] = {}
        self._caps: dict[str, list[str]] = {}
        self._stats: dict[str, dict[str, float]] = {}
        self._recent: deque[dict[str, Any]] = deque(maxlen=50)
        self.version = getattr(self._aicl, "__version__", None) or _pkg_version()

    # -- registration ------------------------------------------------------------
    def register(self, module: str, handler: Handler, capabilities: list[str] | None = None) -> None:
        self._handlers[module] = handler
        self._caps[module] = capabilities or []
        self._stats.setdefault(module, {"calls": 0, "errors": 0, "bytes_in": 0, "bytes_out": 0, "total_ms": 0.0})

    def modules(self) -> dict[str, list[str]]:
        return dict(self._caps)

    # -- dispatch ----------------------------------------------------------------------
    async def call(self, module: str, action: str, payload: dict[str, Any] | None = None, *,
                   op: int | None = None, origin: str = "runtime-api", app_id: str | None = None,
                   timeout: float = 30.0) -> Any:
        """Send ``action`` to ``module`` as an AICL packet and return the decoded result."""
        op = self.ops.OP_CALL if op is None else op
        handler = self._handlers.get(module)
        if handler is None:
            raise RuntimeApiError("module_unavailable", f"AICL module '{module}' is not registered.", 503)
        C, S = self._C, self._S_STRING
        t0 = time.perf_counter()
        body = json.dumps(payload or {}, separators=(",", ":"), default=str)
        req = self._Packet(
            flags=C.FLAG_REQUEST | C.FLAG_TRACED, operation=op, intent=action, targets=[module],
            origin=self._Identity(self._ID_MODULE, origin[:60]),
            symbols=[self._Symbol(S, action), self._Symbol(S, body)],
            deadline_ms=int(timeout * 1000), metadata={"app": app_id or "owner"},
            trace=[self._TraceEntry(self._ID_RUNTIME, "runtime-bus", f"dispatch:{module}", int(time.time() * 1000), 0)])
        wire = self._encode(req, checksum=True)
        view = self._decode(wire)            # validates magic/version/lengths/CRC32
        symbols = view.symbols
        ctx = CallContext(origin, app_id, view.deadline_ms, bytes(view.message_id).hex(), op)
        st = self._stats[module]
        st["calls"] += 1
        st["bytes_in"] += len(wire)
        ok, resp_len, err_msg = True, 0, None
        try:
            result = await asyncio.wait_for(
                handler(symbols[0].value, json.loads(symbols[1].value), ctx), timeout=timeout)
            resp = self._Packet(flags=C.FLAG_RESPONSE | C.FLAG_TRACED, operation=self.ops.OP_RETURN,
                                correlation_id=bytes(req.message_id), targets=[origin],
                                origin=self._Identity(self._ID_MODULE, module[:60]), intent=action,
                                symbols=[self._Symbol(S, json.dumps(result, default=str))])
            rwire = self._encode(resp, checksum=True)
            rview = self._decode(rwire)
            resp_len = len(rwire)
            return json.loads(rview.symbols[0].value)
        except asyncio.TimeoutError:
            ok, err_msg = False, f"'{module}.{action}' timed out after {timeout:.0f}s"
            self._error_packet(module, req, 408, err_msg)
            raise RuntimeApiError("timeout", err_msg, 504) from None
        except RuntimeApiError as exc:
            ok, err_msg = False, exc.message
            self._error_packet(module, req, exc.status, exc.message, exc.code)
            raise
        except AuthError as exc:  # a module refused the caller: keep it a 4xx with its own code
            ok, err_msg = False, exc.message
            self._error_packet(module, req, exc.status, exc.message, exc.code)
            raise RuntimeApiError(exc.code, exc.message, exc.status) from exc
        except Exception as exc:  # noqa: BLE001 - surface as a structured error, never hang
            ok, err_msg = False, f"{type(exc).__name__}: {exc}"
            self._error_packet(module, req, 500, err_msg)
            raise RuntimeApiError("module_error", f"AICL module '{module}' failed: {err_msg}", 500) from exc
        finally:
            dur = (time.perf_counter() - t0) * 1000
            st["bytes_out"] += resp_len
            st["total_ms"] += dur
            if not ok:
                st["errors"] += 1
            self._recent.append({"ts": time.time(), "module": module, "action": action, "op": self.ops.OPERATION_NAMES.get(op, str(op)),
                                 "app": app_id, "message_id": ctx.message_id, "request_bytes": len(wire),
                                 "response_bytes": resp_len, "ok": ok, "error": err_msg, "ms": round(dur, 2)})

    def _error_packet(self, module: str, req: Any, status: int, message: str, code: str = "") -> None:
        """Build (and self-check) the ``OP_ERROR`` packet an AICL peer would receive."""
        C = self._C
        err = self._Packet(flags=C.FLAG_RESPONSE | C.FLAG_ERROR, operation=self.ops.OP_ERROR,
                           correlation_id=bytes(req.message_id),
                           error_info=self._ErrorInfo(error_code=status, severity=2,
                                                      message=message[:250], details=code))
        self._decode(self._encode(err, checksum=True))

    # -- diagnostics ---------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        mods = {}
        for name, s in self._stats.items():
            calls = int(s["calls"])
            mods[name] = {"calls": calls, "errors": int(s["errors"]), "bytes_in": int(s["bytes_in"]),
                          "bytes_out": int(s["bytes_out"]), "avg_ms": round(s["total_ms"] / calls, 2) if calls else 0.0,
                          "capabilities": self._caps.get(name, [])}
        return {"aicl_version": self.version, "codec": "aicl-bin (crc32 trailer)", "native_core": native_available(),
                "modules": mods}

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return list(self._recent)[-limit:]


def _pkg_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("localhousellm-aicl")
    except Exception:  # noqa: BLE001
        return None
