from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions.errors import ForbiddenError
from app.models.user import User
from app.repositories.audit_logs import AuditLogRepository
from app.schemas.audit import AuditLogRead
from app.schemas.common import Page


class AuditQueryService:
    def __init__(self, session: AsyncSession) -> None:
        self.audit_logs = AuditLogRepository(session)

    async def list_me(self, user: User, limit: int, offset: int) -> Page[AuditLogRead]:
        statement = self.audit_logs.list_for_user_statement(user.id)
        total = await self.audit_logs.count(statement)
        items = await self.audit_logs.list(statement, limit, offset)
        return Page[AuditLogRead](
            items=[AuditLogRead.model_validate(item) for item in items],
            total=total,
            limit=limit,
            offset=offset,
        )

    async def list_admin(self, user: User, limit: int, offset: int) -> Page[AuditLogRead]:
        if not user.can_administer():
            raise ForbiddenError("Admin access required")
        statement = self.audit_logs.list_all_statement()
        total = await self.audit_logs.count(statement)
        items = await self.audit_logs.list(statement, limit, offset)
        return Page[AuditLogRead](
            items=[AuditLogRead.model_validate(item) for item in items],
            total=total,
            limit=limit,
            offset=offset,
        )

