import time
from collections import defaultdict
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.config.settings import settings
from app.exceptions.errors import TooManyRequestsError

_AUTH_PREFIX = "/api/v1/auth/"
_attempts: dict[str, list[float]] = defaultdict(list)


class AuthRateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory sliding-window rate limiter for authentication endpoints."""

    def _client_key(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"

    def _prune(self, key: str, now: float) -> None:
        window = settings.AUTH_RATE_LIMIT_WINDOW_SECONDS
        _attempts[key] = [ts for ts in _attempts[key] if now - ts < window]

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if (
            settings.is_testing
            or request.method != "POST"
            or not request.url.path.startswith(_AUTH_PREFIX)
            or request.url.path.endswith("/logout")
        ):
            return await call_next(request)

        key = self._client_key(request)
        now = time.monotonic()
        self._prune(key, now)
        if len(_attempts[key]) >= settings.AUTH_RATE_LIMIT_MAX_ATTEMPTS:
            raise TooManyRequestsError("Too many authentication attempts. Try again later.")
        _attempts[key].append(now)
        return await call_next(request)
