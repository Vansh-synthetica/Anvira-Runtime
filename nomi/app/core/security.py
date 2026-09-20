from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import bcrypt
from jose import JWTError, jwt

from app.config.settings import settings
from app.core.enums import TokenType

_BCRYPT_ROUNDS = 12


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except ValueError:
        return False


def create_token(
    subject: UUID | str,
    token_type: TokenType,
    expires_delta: timedelta | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    now = datetime.now(UTC)
    expire = now + (
        expires_delta
        or timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
            if token_type == TokenType.ACCESS
            else settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60
        )
    )
    payload: dict[str, Any] = {
        "sub": str(subject),
        "type": token_type.value,
        "iat": int(now.timestamp()),
        "exp": expire,
        "jti": str(uuid4()),
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str, expected_type: TokenType) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError as exc:
        raise ValueError("Invalid token") from exc
    if payload.get("type") != expected_type.value:
        raise ValueError("Invalid token type")
    return payload


def token_version_matches(payload: dict[str, Any], user_token_version: int) -> bool:
    token_version = payload.get("tv")
    if token_version is None:
        return False
    return int(token_version) == user_token_version
