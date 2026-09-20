from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.identity_profile import IdentityProfile
from app.repositories.base import Repository


class IdentityRepository(Repository[IdentityProfile]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, IdentityProfile)

    async def get_by_user_id(self, user_id: UUID) -> IdentityProfile | None:
        result = await self.session.execute(
            select(IdentityProfile).where(IdentityProfile.user_id == user_id)
        )
        return result.scalar_one_or_none()

