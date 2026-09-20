from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import PermissionLevel, PermissionTargetType
from app.models.permission import Permission
from app.repositories.permissions import PermissionRepository
from app.services.audit import AuditContext, AuditService

_LEVEL_RANK = {
    PermissionLevel.READ: 1,
    PermissionLevel.WRITE: 2,
    PermissionLevel.ADMIN: 3,
    PermissionLevel.DENY: 0,
}


class PermissionEvaluator:
    """Evaluates stored permission grants for apps and future connectors."""

    def __init__(self, session: AsyncSession) -> None:
        self.permissions = PermissionRepository(session)
        self.audit = AuditService(session)

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
    ) -> bool:
        grants = await self.permissions.find_active_grants(
            owner_user_id,
            application_name=application_name,
            application_id=application_id,
            target_type=target_type,
            target_id=target_id,
        )
        allowed = self._evaluate(grants, required_level)
        if not allowed:
            await self.audit.record(
                action_type="permission.denied",
                user_id=owner_user_id,
                actor_user_id=actor_user_id,
                resource_type=target_type.value,
                resource_id=target_id,
                extra={
                    "application_name": application_name,
                    "application_id": application_id,
                    "required_level": required_level.value,
                },
                context=context,
            )
        return allowed

    def _evaluate(self, grants: list[Permission], required_level: PermissionLevel) -> bool:
        if not grants:
            return False

        now = datetime.now(UTC)
        applicable: list[Permission] = []
        for grant in grants:
            if not grant.is_active:
                continue
            if grant.expires_at and grant.expires_at <= now:
                continue
            if grant.revoked_at is not None:
                continue
            applicable.append(grant)

        if not applicable:
            return False

        if any(grant.permission_level == PermissionLevel.DENY for grant in applicable):
            return False

        required_rank = _LEVEL_RANK[required_level]
        best_rank = max(_LEVEL_RANK[grant.permission_level] for grant in applicable)
        return best_rank >= required_rank
