from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.core.enums import TokenType, UserRole
from app.core.security import (
    create_token,
    decode_token,
    hash_password,
    token_version_matches,
    verify_password,
)
from app.exceptions.errors import ConflictError, UnauthorizedError
from app.models.user import User
from app.repositories.users import UserRepository
from app.schemas.auth import AuthResponse, TokenPair
from app.schemas.user import UserCreate
from app.services.audit import AuditContext, AuditService


class AuthService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.users = UserRepository(session)
        self.audit = AuditService(session)

    async def register(self, payload: UserCreate, context: AuditContext | None = None) -> AuthResponse:
        email = payload.email.lower()
        username = payload.username.strip()
        if await self.users.get_by_email(email):
            raise ConflictError("Email is already registered")
        if await self.users.get_by_username(username):
            raise ConflictError("Username is already taken")

        user = await self.users.add(
            User(
                email=email,
                username=username,
                hashed_password=hash_password(payload.password),
                role=UserRole.USER,
            )
        )
        await self.audit.record(
            action_type="user.registration",
            user_id=user.id,
            actor_user_id=user.id,
            resource_type="user",
            resource_id=str(user.id),
            context=context,
        )
        tokens = self._issue_tokens(user)
        return AuthResponse(user=user, **tokens.model_dump())

    async def login(self, email: str, password: str, context: AuditContext | None = None) -> AuthResponse:
        normalized_email = email.lower()
        user = await self.users.get_by_email(normalized_email)
        now = datetime.now(UTC)

        if user and user.locked_until and user.locked_until > now:
            await self.audit.record(
                action_type="auth.locked",
                user_id=user.id,
                extra={"email": normalized_email},
                context=context,
            )
            raise UnauthorizedError("Invalid credentials")

        if not user or not verify_password(password, user.hashed_password) or not user.is_active:
            await self._record_failed_login(user, normalized_email, context)
            raise UnauthorizedError("Invalid credentials")

        user.failed_login_count = 0
        user.locked_until = None
        user.last_login_at = now
        self.session.add(user)
        await self.session.flush()
        await self.session.refresh(user)
        await self.audit.record(
            action_type="auth.login",
            user_id=user.id,
            actor_user_id=user.id,
            resource_type="user",
            resource_id=str(user.id),
            context=context,
        )
        tokens = self._issue_tokens(user)
        return AuthResponse(user=user, **tokens.model_dump())

    async def refresh(self, refresh_token: str, context: AuditContext | None = None) -> TokenPair:
        try:
            payload = decode_token(refresh_token, TokenType.REFRESH)
            user_id = UUID(payload["sub"])
        except (ValueError, KeyError) as exc:
            raise UnauthorizedError("Invalid refresh token") from exc

        user = await self.users.get_active(user_id)
        if not user or not token_version_matches(payload, user.token_version):
            raise UnauthorizedError("Invalid refresh token")

        await self.audit.record(
            action_type="auth.refresh",
            user_id=user.id,
            actor_user_id=user.id,
            resource_type="user",
            resource_id=str(user.id),
            context=context,
        )
        return self._issue_tokens(user)

    async def logout(self, user: User, context: AuditContext | None = None) -> None:
        user.bump_token_version()
        self.session.add(user)
        await self.session.flush()
        await self.audit.record(
            action_type="auth.logout",
            user_id=user.id,
            actor_user_id=user.id,
            resource_type="user",
            resource_id=str(user.id),
            context=context,
        )

    async def _record_failed_login(
        self,
        user: User | None,
        email: str,
        context: AuditContext | None,
    ) -> None:
        extra: dict[str, str] = {"email": email}
        if user and user.is_active:
            user.failed_login_count += 1
            if user.failed_login_count >= settings.MAX_FAILED_LOGIN_ATTEMPTS:
                user.locked_until = datetime.now(UTC) + timedelta(
                    minutes=settings.ACCOUNT_LOCKOUT_MINUTES
                )
                extra["locked"] = "true"
                await self.audit.record(
                    action_type="auth.suspicious",
                    user_id=user.id,
                    extra=extra,
                    context=context,
                )
            self.session.add(user)
            await self.session.flush()

        await self.audit.record(
            action_type="auth.failure",
            user_id=user.id if user else None,
            extra=extra,
            context=context,
        )

    def _issue_tokens(self, user: User) -> TokenPair:
        claims = {"role": user.role.value, "tv": user.token_version}
        return TokenPair(
            access_token=create_token(user.id, TokenType.ACCESS, extra_claims=claims),
            refresh_token=create_token(user.id, TokenType.REFRESH, extra_claims={"tv": user.token_version}),
        )
