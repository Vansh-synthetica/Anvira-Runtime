from typing import Protocol
from uuid import UUID

from app.core.enums import PermissionLevel, PermissionTargetType
from app.services.audit import AuditContext


class PermissionEvaluatorPort(Protocol):
    async def check_access(
        self,
        owner_user_id: UUID,
        *,
        application_name: str,
        application_id: str | None,
        target_type: PermissionTargetType,
        target_id: str | None,
        required_level: PermissionLevel,
        actor_user_id: UUID | None = None,
        context: AuditContext | None = None,
    ) -> bool: ...
