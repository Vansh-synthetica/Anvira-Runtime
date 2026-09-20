import logging

from app.config.settings import Settings

logger = logging.getLogger("nomi")

_INSECURE_SECRET_KEYS = frozenset(
    {
        "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7",
        "changeme",
        "change-me",
        "test-secret-key-with-at-least-thirty-two-characters",
    }
)


def validate_settings(settings: Settings) -> None:
    if settings.ENVIRONMENT != "production":
        return

    if settings.DEBUG:
        raise RuntimeError("DEBUG must be false in production")

    if settings.SECRET_KEY in _INSECURE_SECRET_KEYS:
        raise RuntimeError("SECRET_KEY is not safe for production use")

    logger.info(
        "Production settings validated environment=%s cors_origins=%d",
        settings.ENVIRONMENT,
        len(settings.BACKEND_CORS_ORIGINS),
    )
