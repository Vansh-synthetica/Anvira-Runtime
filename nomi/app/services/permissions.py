from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions.errors import NotFoundError
from app.models.permission import Permission
from app.models.user import User
from app.repositories.permissions import PermissionRepository
from app.schemas.common import Page
from app.schemas.permission import PermissionCreate, PermissionRead, PermissionUpdate
from app.services.audit import AuditContext, AuditService


class PermissionService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.permissions = PermissionRepository(session)
        self.audit = AuditService(session)

    async def create(
        self,
        user: User,
        payload: PermissionCreate,
        context: AuditContext | None = None,
    ) -> Permission:
        permission = await self.permissions.add(Permission(user_id=user.id, **payload.model_dump()))
        await self.audit.record(
            action_type="permission.create",
            user_id=user.id,
            actor_user_id=user.id,
            resource_type="permission",
            resource_id=str(permission.id),
            context=context,
        )
        return permission

    async def list(self, user: User, limit: int, offset: int) -> Page[PermissionRead]:
        statement = self.permissions.list_statement(user.id)
        total = await self.permissions.count(statement)
        items = await self.permissions.list(statement, limit, offset)
        return Page[PermissionRead](
            items=[PermissionRead.model_validate(item) for item in items],
            total=total,
            limit=limit,
            offset=offset,
        )

    async def get(self, user: User, permission_id: UUID) -> Permission:
        permission = await self.permissions.get_owned(user.id, permission_id)
        if not permission or permission.revoked_at is not None:
            raise NotFoundError("Permission not found")
        return permission

    async def update(
        self,
        user: User,
        permission_id: UUID,
        payload: PermissionUpdate,
        context: AuditContext | None = None,
    ) -> Permission:
        permission = await self.get(user, permission_id)
        values = payload.model_dump(exclude_unset=True)
        await self.permissions.apply_updates(permission, values)
        await self.audit.record(
            action_type="permission.update",
            user_id=user.id,
            actor_user_id=user.id,
            resource_type="permission",
            resource_id=str(permission.id),
            context=context,
        )
        return permission

    async def revoke(self, user: User, permission_id: UUID, context: AuditContext | None = None) -> None:
        permission = await self.get(user, permission_id)
        permission.is_active = False
        permission.revoked_at = datetime.now(UTC)
        self.session.add(permission)
        await self.session.flush()
        await self.audit.record(
            action_type="permission.revoke",
            user_id=user.id,
            actor_user_id=user.id,
            resource_type="permission",
            resource_id=str(permission.id),
            context=context,
        )
