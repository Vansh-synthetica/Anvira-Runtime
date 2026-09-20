import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.user import User


class IdentityProfile(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "identity_profiles"

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=False,
    )
    display_name: Mapped[str | None] = mapped_column(String(120))
    pronouns: Mapped[str | None] = mapped_column(String(64))
    timezone: Mapped[str | None] = mapped_column(String(64))
    locale: Mapped[str | None] = mapped_column(String(32))
    preferred_language: Mapped[str | None] = mapped_column(String(64))
    writing_style: Mapped[str | None] = mapped_column(Text)
    tone: Mapped[str | None] = mapped_column(String(120))
    preferred_output_format: Mapped[str | None] = mapped_column(String(120))
    occupation: Mapped[str | None] = mapped_column(String(160))
    bio: Mapped[str | None] = mapped_column(Text)
    interests: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    goals: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    preferences: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    constraints: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    avatar_url: Mapped[str | None] = mapped_column(String(2048))
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    user: Mapped["User"] = relationship(back_populates="identity_profile")
