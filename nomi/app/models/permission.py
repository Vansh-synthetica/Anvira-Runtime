import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from app.core.enums import PermissionLevel, PermissionTargetType
from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import str_enum_column

if TYPE_CHECKING:
    from app.models.user import User


class Permission(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "permissions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    application_name: Mapped[str] = mapped_column(String(160), index=True, nullable=False)
    application_id: Mapped[str | None] = mapped_column(String(160), index=True)
    target_type: Mapped[PermissionTargetType] = mapped_column(
        str_enum_column(PermissionTargetType),
        index=True,
        nullable=False,
    )
    target_id: Mapped[str | None] = mapped_column(String(160), index=True)
    permission_level: Mapped[PermissionLevel] = mapped_column(
        str_enum_column(PermissionLevel), nullable=False
    )
    scope: Mapped[str | None] = mapped_column(String(240), index=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True, nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    user: Mapped["User"] = relationship(back_populates="permissions")
