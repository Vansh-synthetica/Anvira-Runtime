import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import UserRole
from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import str_enum_column

if TYPE_CHECKING:
    from app.models.identity_profile import IdentityProfile
    from app.models.memory import Memory
    from app.models.permission import Permission


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        str_enum_column(UserRole), default=UserRole.USER, nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    token_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    failed_login_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    identity_profile: Mapped["IdentityProfile | None"] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    memories: Mapped[list["Memory"]] = relationship(back_populates="user")
    permissions: Mapped[list["Permission"]] = relationship(back_populates="user")

    def can_administer(self) -> bool:
        return self.role == UserRole.ADMIN and self.is_active

    def bump_token_version(self) -> None:
        self.token_version += 1

    @property
    def user_id(self) -> uuid.UUID:
        return self.id
