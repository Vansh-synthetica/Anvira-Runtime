"""Tiny stdlib HTTP layer: JSON in/out, SSE streaming, structured errors."""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterator

from .errors import AnviraError, RuntimeNotRunning


class Http:
    def __init__(self, base_url: str, token: str | None = None, timeout: float = 30.0, recover=None):
        self.base_url, self.token, self.timeout = base_url.rstrip("/"), token, timeout
        # ``recover()`` returns a new base URL after (re)starting the runtime, or None. Used so an app that is
        # still open keeps working when an on-demand runtime stopped in the meantime.
        self.recover = recover

    def _request(self, method: str, path: str, body: Any = None, params: dict[str, Any] | None = None,
                 headers: dict[str, str] | None = None, timeout: float | None = None):
        try:
            return self._request_once(method, path, body, params, headers, timeout)
        except RuntimeNotRunning:
            if self.recover is None:
                raise
            new_base = self.recover()
            if not new_base:
                raise
            self.base_url = new_base.rstrip("/")
            return self._request_once(method, path, body, params, headers, timeout)

    def _request_once(self, method: str, path: str, body: Any = None, params: dict[str, Any] | None = None,
                      headers: dict[str, str] | None = None, timeout: float | None = None):
        url = self.base_url + path
        if params:
            clean = [(k, str(v)) for k, vs in params.items() if vs is not None
                     for v in (vs if isinstance(vs, (list, tuple)) else [vs])]
            if clean:
                url += "?" + urllib.parse.urlencode(clean)
        hdrs = {"Accept": "application/json", **(headers or {})}
        if self.token:
            hdrs["Authorization"] = f"Bearer {self.token}"
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            return urllib.request.urlopen(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8"))
            except (ValueError, OSError):
                payload = None
            raise AnviraError.from_response(exc.code, payload) from None
        except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, (socket.timeout, TimeoutError)) or isinstance(exc, (socket.timeout, TimeoutError)):
                raise AnviraError("timeout", f"The runtime did not answer within {timeout or self.timeout:.0f}s.") from None
            raise RuntimeNotRunning("runtime_unreachable", f"Cannot reach Anvira Runtime at {self.base_url}: {reason}",
                                    hint="Start it with `anvira runtime start`.") from None

    def json(self, method: str, path: str, body: Any = None, params: dict[str, Any] | None = None,
             headers: dict[str, str] | None = None, timeout: float | None = None) -> Any:
        with self._request(method, path, body, params, headers, timeout) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8")) if raw else {}

    def sse(self, method: str, path: str, body: Any = None, params: dict[str, Any] | None = None,
            timeout: float | None = None) -> Iterator[tuple[str, str]]:
        """Yield ``(event, data)`` pairs from a text/event-stream response."""
        resp = self._request(method, path, body, params, {"Accept": "text/event-stream"}, timeout or 600)
        with resp:
            event, data = "message", []
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    if data:
                        yield event, "\n".join(data)
                    event, data = "message", []
                elif line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip())
            if data:
                yield event, "\n".join(data)
