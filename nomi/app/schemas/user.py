from datetime import datetime
from uuid import UUID

from pydantic import EmailStr, Field

from app.core.enums import UserRole
from app.schemas.common import ORMModel


class UserBase(ORMModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=64)


class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=128)


class UserUpdate(ORMModel):
    username: str | None = Field(default=None, min_length=3, max_length=64)
    email: EmailStr | None = None


class UserRead(UserBase):
    id: UUID
    role: UserRole
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime
    last_login_at: datetime | None = None

