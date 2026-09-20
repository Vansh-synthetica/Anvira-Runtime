from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.context import get_request_id
from app.models.audit_log import AuditLog
from app.repositories.audit_logs import AuditLogRepository


@dataclass(frozen=True)
class AuditContext:
    ip_address: str | None = None
    user_agent: str | None = None


class AuditService:
    def __init__(self, session: AsyncSession) -> None:
        self.repository = AuditLogRepository(session)

    async def record(
        self,
        *,
        action_type: str,
        user_id: UUID | None = None,
        actor_user_id: UUID | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        extra: dict[str, Any] | None = None,
        context: AuditContext | None = None,
    ) -> AuditLog:
        return await self.repository.add(
            AuditLog(
                user_id=user_id,
                actor_user_id=actor_user_id,
                action_type=action_type,
                resource_type=resource_type,
                resource_id=resource_id,
                extra=extra or {},
                ip_address=context.ip_address if context else None,
                user_agent=context.user_agent if context else None,
                request_id=get_request_id(),
            )
        )

