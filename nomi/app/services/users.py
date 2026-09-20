from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions.errors import ConflictError
from app.models.user import User
from app.repositories.users import UserRepository
from app.schemas.user import UserUpdate
from app.services.audit import AuditContext, AuditService


class UserService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.users = UserRepository(session)
        self.audit = AuditService(session)

    async def update_me(self, user: User, payload: UserUpdate, context: AuditContext | None = None) -> User:
        values = payload.model_dump(exclude_unset=True)
        if "email" in values and values["email"]:
            values["email"] = values["email"].lower()
            existing = await self.users.get_by_email(values["email"])
            if existing and existing.id != user.id:
                raise ConflictError("Email is already registered")
        if "username" in values and values["username"]:
            values["username"] = values["username"].strip()
            existing = await self.users.get_by_username(values["username"])
            if existing and existing.id != user.id:
                raise ConflictError("Username is already taken")
        await self.users.apply_updates(user, values)
        await self.audit.record(
            action_type="user.update",
            user_id=user.id,
            actor_user_id=user.id,
            resource_type="user",
            resource_id=str(user.id),
            context=context,
        )
        return user

    async def deactivate_me(self, user: User, context: AuditContext | None = None) -> None:
        user.is_active = False
        user.bump_token_version()
        self.session.add(user)
        await self.session.flush()
        await self.audit.record(
            action_type="user.deactivate",
            user_id=user.id,
            actor_user_id=user.id,
            resource_type="user",
            resource_id=str(user.id),
            context=context,
        )
