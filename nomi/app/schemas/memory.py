from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from app.core.enums import MemoryType, MemoryVisibility
from app.schemas.common import ORMModel


class MemoryBase(ORMModel):
    title: str = Field(min_length=1, max_length=240)
    content: str = Field(min_length=1)
    memory_type: MemoryType = MemoryType.NOTE
    category: str | None = Field(default=None, max_length=120)
    importance: int = Field(default=3, ge=1, le=5)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    visibility: MemoryVisibility = MemoryVisibility.PRIVATE
    source: str | None = Field(default=None, max_length=240)
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
    expires_at: datetime | None = None


class MemoryCreate(MemoryBase):
    pass


class MemoryUpdate(ORMModel):
    title: str | None = Field(default=None, min_length=1, max_length=240)
    content: str | None = Field(default=None, min_length=1)
    memory_type: MemoryType | None = None
    category: str | None = Field(default=None, max_length=120)
    importance: int | None = Field(default=None, ge=1, le=5)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    visibility: MemoryVisibility | None = None
    source: str | None = Field(default=None, max_length=240)
    tags: list[str] | None = None
    extra: dict[str, Any] | None = None
    expires_at: datetime | None = None


class MemoryRead(MemoryBase):
    id: UUID
    user_id: UUID
    deleted_at: datetime | None = None
    is_archived: bool
    created_at: datetime
    updated_at: datetime

