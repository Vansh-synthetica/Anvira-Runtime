import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from app.core.enums import MemoryType, MemoryVisibility
from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.types import str_enum_column

if TYPE_CHECKING:
    from app.models.user import User


class Memory(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "memories"

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(240), index=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    memory_type: Mapped[MemoryType] = mapped_column(
        str_enum_column(MemoryType), index=True, nullable=False
    )
    category: Mapped[str | None] = mapped_column(String(120), index=True)
    importance: Mapped[int] = mapped_column(default=3, nullable=False)
    confidence: Mapped[float] = mapped_column(default=1.0, nullable=False)
    visibility: Mapped[MemoryVisibility] = mapped_column(
        str_enum_column(MemoryVisibility),
        default=MemoryVisibility.PRIVATE,
        nullable=False,
    )
    source: Mapped[str | None] = mapped_column(String(240))
    tags: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, index=True, nullable=False)

    user: Mapped["User"] = relationship(back_populates="memories")
