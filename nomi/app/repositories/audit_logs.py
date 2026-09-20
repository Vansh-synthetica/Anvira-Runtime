from uuid import UUID

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit_log import AuditLog
from app.repositories.base import Repository


class AuditLogRepository(Repository[AuditLog]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, AuditLog)

    def list_for_user_statement(self, user_id: UUID) -> Select[tuple[AuditLog]]:
        return select(AuditLog).where(AuditLog.user_id == user_id).order_by(AuditLog.created_at.desc())

    def list_all_statement(self) -> Select[tuple[AuditLog]]:
        return select(AuditLog).order_by(AuditLog.created_at.desc())
