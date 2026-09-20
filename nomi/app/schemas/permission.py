from datetime import datetime
from uuid import UUID

from pydantic import Field

from app.core.enums import PermissionLevel, PermissionTargetType
from app.schemas.common import ORMModel


class PermissionBase(ORMModel):
    application_name: str = Field(min_length=1, max_length=160)
    application_id: str | None = Field(default=None, max_length=160)
    target_type: PermissionTargetType
    target_id: str | None = Field(default=None, max_length=160)
    permission_level: PermissionLevel
    scope: str | None = Field(default=None, max_length=240)
    expires_at: datetime | None = None
    is_active: bool = True


class PermissionCreate(PermissionBase):
    pass


class PermissionUpdate(ORMModel):
    application_name: str | None = Field(default=None, min_length=1, max_length=160)
    application_id: str | None = Field(default=None, max_length=160)
    target_type: PermissionTargetType | None = None
    target_id: str | None = Field(default=None, max_length=160)
    permission_level: PermissionLevel | None = None
    scope: str | None = Field(default=None, max_length=240)
    expires_at: datetime | None = None
    is_active: bool | None = None


class PermissionRead(PermissionBase):
    id: UUID
    user_id: UUID
    revoked_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

