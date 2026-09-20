from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class APIError(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class ValidationErrorDetail(BaseModel):
    loc: list[str]
    message: str


class APIValidationError(APIError):
    details: list[ValidationErrorDetail] = Field(default_factory=list)


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)


class Message(BaseModel):
    message: str


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

