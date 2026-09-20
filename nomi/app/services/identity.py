from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions.errors import ConflictError, ForbiddenError, NotFoundError
from app.models.identity_profile import IdentityProfile
from app.models.user import User
from app.repositories.identity_profiles import IdentityRepository
from app.schemas.identity import IdentityCreate, IdentityUpdate
from app.services.audit import AuditContext, AuditService


class IdentityService:
    def __init__(self, session: AsyncSession) -> None:
        self.profiles = IdentityRepository(session)
        self.audit = AuditService(session)

    async def create(self, user: User, payload: IdentityCreate, context: AuditContext | None = None) -> IdentityProfile:
        if await self.profiles.get_by_user_id(user.id):
            raise ConflictError("Identity profile already exists")
        values = payload.model_dump()
        if values.get("avatar_url") is not None:
            values["avatar_url"] = str(values["avatar_url"])
        profile = await self.profiles.add(IdentityProfile(user_id=user.id, **values))
        await self.audit.record(
            action_type="identity.create",
            user_id=user.id,
            actor_user_id=user.id,
            resource_type="identity_profile",
            resource_id=str(profile.id),
            context=context,
        )
        return profile

    async def get_me(self, user: User) -> IdentityProfile:
        profile = await self.profiles.get_by_user_id(user.id)
        if not profile:
            raise NotFoundError("Identity profile not found")
        return profile

    async def get_by_id(self, user: User, profile_id: UUID) -> IdentityProfile:
        profile = await self.profiles.get(profile_id)
        if not profile:
            raise NotFoundError("Identity profile not found")
        if profile.user_id != user.id and not user.can_administer():
            raise ForbiddenError("You do not have access to this identity profile")
        return profile

    async def update_me(
        self,
        user: User,
        payload: IdentityUpdate,
        context: AuditContext | None = None,
    ) -> IdentityProfile:
        profile = await self.get_me(user)
        values = payload.model_dump(exclude_unset=True)
        if values.get("avatar_url") is not None:
            values["avatar_url"] = str(values["avatar_url"])
        await self.profiles.apply_updates(profile, values)
        await self.audit.record(
            action_type="identity.update",
            user_id=user.id,
            actor_user_id=user.id,
            resource_type="identity_profile",
            resource_id=str(profile.id),
            context=context,
        )
        return profile

    async def delete_me(self, user: User, context: AuditContext | None = None) -> None:
        profile = await self.get_me(user)
        await self.profiles.delete(profile)
        await self.audit.record(
            action_type="identity.delete",
            user_id=user.id,
            actor_user_id=user.id,
            resource_type="identity_profile",
            resource_id=str(profile.id),
            context=context,
        )
