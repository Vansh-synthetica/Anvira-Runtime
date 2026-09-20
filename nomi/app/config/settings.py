from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    APP_NAME: str = "Nomi"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: Literal["development", "testing", "production"] = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    DATABASE_URL: str | None = None
    POSTGRES_USER: str = "nomi"
    POSTGRES_PASSWORD: str = "nomi"
    POSTGRES_DB: str = "nomi"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    REDIS_URL: str | None = None
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    SECRET_KEY: str = Field(..., min_length=32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    AUTH_RATE_LIMIT_MAX_ATTEMPTS: int = 20
    AUTH_RATE_LIMIT_WINDOW_SECONDS: int = 60
    MAX_FAILED_LOGIN_ATTEMPTS: int = 5
    ACCOUNT_LOCKOUT_MINUTES: int = 15

    BACKEND_CORS_ORIGINS: list[AnyHttpUrl | str] = []

    MEMORY_MD_DIR: str | None = None

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str] | str:
        if isinstance(value, str) and value and not value.startswith("["):
            return [origin.strip() for origin in value.split(",")]
        return value

    @property
    def database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def redis_url(self) -> str:
        if self.REDIS_URL:
            return self.REDIS_URL
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    @property
    def memory_md_dir(self) -> Path:
        """Root directory for the file-backed markdown memory layer.

        Resolution order: explicit ``MEMORY_MD_DIR`` → the desktop data dir
        (``NOMI_DATA_DIR``, set by the launcher) → the default
        ``~/.anvira/nomi`` home.
        """
        import os

        base = (
            self.MEMORY_MD_DIR
            or os.environ.get("NOMI_DATA_DIR")
            or str(Path.home() / ".anvira" / "nomi")
        )
        return Path(base) / "memory-md"

    @property
    def is_testing(self) -> bool:
        return self.ENVIRONMENT == "testing"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
