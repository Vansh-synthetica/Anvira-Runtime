from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Select, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import MemoryType
from app.models.memory import Memory
from app.repositories.base import Repository


class MemoryRepository(Repository[Memory]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Memory)

    async def get_owned(self, user_id: UUID, memory_id: UUID, include_deleted: bool = False) -> Memory | None:
        statement = select(Memory).where(Memory.id == memory_id, Memory.user_id == user_id)
        if not include_deleted:
            statement = statement.where(Memory.deleted_at.is_(None))
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    def filter_statement(
        self,
        user_id: UUID,
        *,
        query: str | None = None,
        memory_type: MemoryType | None = None,
        category: str | None = None,
        tag: str | None = None,
        include_archived: bool = False,
        include_deleted: bool = False,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        sort: str = "-created_at",
    ) -> Select[tuple[Memory]]:
        statement = select(Memory).where(Memory.user_id == user_id)
        if not include_deleted:
            statement = statement.where(Memory.deleted_at.is_(None))
        if not include_archived:
            statement = statement.where(Memory.is_archived.is_(False))
        if query:
            pattern = f"%{query}%"
            statement = statement.where(or_(Memory.title.ilike(pattern), Memory.content.ilike(pattern)))
        if memory_type:
            statement = statement.where(Memory.memory_type == memory_type)
        if category:
            statement = statement.where(Memory.category == category)
        if tag:
            dialect_name = self.session.get_bind().dialect.name
            if dialect_name == "sqlite":
                statement = statement.where(Memory.tags.like(f'%"{tag}"%'))
            else:
                statement = statement.where(Memory.tags.contains([tag]))
        if created_after:
            statement = statement.where(Memory.created_at >= created_after)
        if created_before:
            statement = statement.where(Memory.created_at <= created_before)

        sort_map: dict[str, Any] = {
            "created_at": Memory.created_at.asc(),
            "-created_at": Memory.created_at.desc(),
            "updated_at": Memory.updated_at.asc(),
            "-updated_at": Memory.updated_at.desc(),
            "importance": Memory.importance.asc(),
            "-importance": Memory.importance.desc(),
            "title": Memory.title.asc(),
            "-title": Memory.title.desc(),
        }
        return statement.order_by(sort_map.get(sort, Memory.created_at.desc()))
