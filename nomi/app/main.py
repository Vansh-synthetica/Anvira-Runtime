import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.exception_handlers import register_exception_handlers
from app.api.v1.router import api_router
from app.config.settings import settings
from app.config.validation import validate_settings
from app.db.session import close_engine, engine
from app.logging.config import configure_logging
from app.middleware.rate_limit import AuthRateLimitMiddleware
from app.middleware.request_id import REQUEST_ID_HEADER, RequestIdMiddleware
from app.middleware.request_logging import RequestLoggingMiddleware


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging(settings.LOG_LEVEL, settings.ENVIRONMENT)
    validate_settings(settings)
    logging.getLogger("nomi").info(
        "Nomi backend starting environment=%s debug=%s",
        settings.ENVIRONMENT,
        settings.DEBUG,
    )
    yield
    logging.getLogger("nomi").info("Nomi backend shutting down")
    await close_engine()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description="Identity, memory, and permission layer for AI tools.",
        version="0.2.0",
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(AuthRateLimitMiddleware)
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[REQUEST_ID_HEADER],
    )
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)
    register_exception_handlers(app)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": settings.APP_NAME,
            "environment": settings.ENVIRONMENT,
        }

    @app.get("/ready", tags=["health"])
    async def ready() -> dict[str, str]:
        try:
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "service": settings.APP_NAME,
                    "error": str(exc),
                },
            ) from exc

        return {"status": "ready", "service": settings.APP_NAME}

    return app


app = create_app()