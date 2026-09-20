"""Pydantic schemas for the markdown memory layer."""

from typing import Literal

from pydantic import BaseModel, Field

MemoryMdType = Literal["user", "feedback", "project", "reference"]


class MemoryMdIndex(BaseModel):
    content: str
    truncated: bool


class MemoryMdFileSummary(BaseModel):
    name: str
    title: str
    description: str
    type: str
    updated: str
    size: int = 0


class MemoryMdWrite(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    title: str = Field(..., min_length=1, max_length=120)
    description: str = Field(default="", max_length=300)
    type: MemoryMdType = "project"
    body: str = Field(..., min_length=1)


class MemoryMdFile(BaseModel):
    name: str
    title: str
    description: str = ""
    type: str = ""
    updated: str = ""
    body: str


class RecallResponse(BaseModel):
    query: str
    candidates: list[MemoryMdFileSummary]
    bodies: dict[str, str]
    index: str
    index_truncated: bool


class CursorRead(BaseModel):
    last_message_id: str | None = None


class CursorUpdate(BaseModel):
    last_message_id: str | None = None
