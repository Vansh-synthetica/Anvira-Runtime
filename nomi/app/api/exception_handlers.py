import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.context import get_request_id
from app.exceptions.errors import NomiError
from app.schemas.common import APIError, APIValidationError, ValidationErrorDetail

logger = logging.getLogger("nomi.errors")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(NomiError)
    async def nomi_error_handler(_: Request, exc: NomiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=APIError(
                code=exc.code,
                message=exc.message,
                request_id=get_request_id(),
            ).model_dump(exclude_none=True),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            ValidationErrorDetail(
                loc=[str(part) for part in error.get("loc", ())],
                message=str(error.get("msg", "Invalid value")),
            )
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=APIValidationError(
                code="validation_error",
                message="Request validation failed",
                request_id=get_request_id(),
                details=details,
            ).model_dump(exclude_none=True),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled error method=%s path=%s request_id=%s",
            request.method,
            request.url.path,
            get_request_id(),
        )
        return JSONResponse(
            status_code=500,
            content=APIError(
                code="internal_error",
                message="An unexpected error occurred",
                request_id=get_request_id(),
            ).model_dump(exclude_none=True),
        )
