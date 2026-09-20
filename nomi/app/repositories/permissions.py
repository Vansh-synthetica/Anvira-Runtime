from uuid import UUID

from sqlalchemy import Select, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import PermissionTargetType
from app.models.permission import Permission
from app.repositories.base import Repository


class PermissionRepository(Repository[Permission]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Permission)

    async def get_owned(self, user_id: UUID, permission_id: UUID) -> Permission | None:
        result = await self.session.execute(
            select(Permission).where(Permission.id == permission_id, Permission.user_id == user_id)
        )
        return result.scalar_one_or_none()

    def list_statement(self, user_id: UUID) -> Select[tuple[Permission]]:
        return (
            select(Permission)
            .where(Permission.user_id == user_id, Permission.revoked_at.is_(None))
            .order_by(Permission.created_at.desc())
        )

    async def find_active_grants(
        self,
        user_id: UUID,
        *,
        application_name: str,
        application_id: str | None,
        target_type: PermissionTargetType,
        target_id: str | None,
    ) -> list[Permission]:
        statement = select(Permission).where(
            Permission.user_id == user_id,
            Permission.application_name == application_name,
            Permission.is_active.is_(True),
            Permission.revoked_at.is_(None),
            or_(
                Permission.target_type == target_type,
                Permission.target_type == PermissionTargetType.WORKSPACE,
            ),
        )
        if application_id:
            statement = statement.where(
                or_(Permission.application_id.is_(None), Permission.application_id == application_id)
            )
        if target_id:
            statement = statement.where(
                or_(Permission.target_id.is_(None), Permission.target_id == target_id)
            )

        result = await self.session.execute(statement)
        return list(result.scalars().all())
