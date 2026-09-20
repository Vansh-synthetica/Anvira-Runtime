from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from app.schemas.common import ORMModel


class IdentityBase(ORMModel):
    display_name: str | None = Field(default=None, max_length=120)
    pronouns: str | None = Field(default=None, max_length=64)
    timezone: str | None = Field(default=None, max_length=64)
    locale: str | None = Field(default=None, max_length=32)
    preferred_language: str | None = Field(default=None, max_length=64)
    writing_style: str | None = None
    tone: str | None = Field(default=None, max_length=120)
    preferred_output_format: str | None = Field(default=None, max_length=120)
    occupation: str | None = Field(default=None, max_length=160)
    bio: str | None = None
    interests: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    avatar_url: str | None = Field(default=None, max_length=2048)
    extra: dict[str, Any] = Field(default_factory=dict)


class IdentityCreate(IdentityBase):
    pass


class IdentityUpdate(ORMModel):
    display_name: str | None = Field(default=None, max_length=120)
    pronouns: str | None = Field(default=None, max_length=64)
    timezone: str | None = Field(default=None, max_length=64)
    locale: str | None = Field(default=None, max_length=32)
    preferred_language: str | None = Field(default=None, max_length=64)
    writing_style: str | None = None
    tone: str | None = Field(default=None, max_length=120)
    preferred_output_format: str | None = Field(default=None, max_length=120)
    occupation: str | None = Field(default=None, max_length=160)
    bio: str | None = None
    interests: list[str] | None = None
    goals: list[str] | None = None
    preferences: dict[str, Any] | None = None
    constraints: list[str] | None = None
    avatar_url: str | None = Field(default=None, max_length=2048)
    extra: dict[str, Any] | None = None


class IdentityRead(IdentityBase):
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime
