from datetime import datetime
from typing import Any
from uuid import UUID

from app.schemas.common import ORMModel


class AuditLogRead(ORMModel):
    id: UUID
    user_id: UUID | None
    actor_user_id: UUID | None
    action_type: str
    resource_type: str | None
    resource_id: str | None
    extra: dict[str, Any]
    ip_address: str | None
    user_agent: str | None
    request_id: str | None = None
    created_at: datetime

