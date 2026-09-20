class NomiError(Exception):
    """Base domain exception."""

    status_code = 400
    code = "nomi_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(NomiError):
    status_code = 404
    code = "not_found"


class ConflictError(NomiError):
    status_code = 409
    code = "conflict"


class UnauthorizedError(NomiError):
    status_code = 401
    code = "unauthorized"


class ForbiddenError(NomiError):
    status_code = 403
    code = "forbidden"


class TooManyRequestsError(NomiError):
    status_code = 429
    code = "too_many_requests"

