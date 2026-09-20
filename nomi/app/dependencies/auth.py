from uuid import UUID

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.core.enums import TokenType
from app.core.security import decode_token, token_version_matches
from app.db.session import get_db_session
from app.exceptions.errors import ForbiddenError, UnauthorizedError
from app.models.user import User
from app.repositories.users import UserRepository

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_PREFIX}/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_db_session),
) -> User:
    try:
        payload = decode_token(token, TokenType.ACCESS)
        user_id = UUID(payload["sub"])
    except (ValueError, KeyError) as exc:
        raise UnauthorizedError("Invalid access token") from exc

    user = await UserRepository(session).get_active(user_id)
    if not user or not token_version_matches(payload, user.token_version):
        raise UnauthorizedError("Invalid access token")
    return user


async def require_admin(user: User = Depends(get_current_user)) -> User:
    if not user.can_administer():
        raise ForbiddenError("Admin access required")
    return user
