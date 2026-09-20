from typing import Any, Generic, TypeVar
from uuid import UUID

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class Repository(Generic[ModelT]):
    """Small async repository base for SQLAlchemy models."""

    def __init__(self, session: AsyncSession, model: type[ModelT]) -> None:
        self.session = session
        self.model = model

    async def get(self, item_id: UUID) -> ModelT | None:
        return await self.session.get(self.model, item_id)

    async def add(self, obj: ModelT) -> ModelT:
        self.session.add(obj)
        await self.session.flush()
        await self.session.refresh(obj)
        return obj

    async def delete(self, obj: ModelT) -> None:
        await self.session.delete(obj)
        await self.session.flush()

    async def count(self, statement: Select[tuple[ModelT]]) -> int:
        count_statement = select(func.count()).select_from(statement.subquery())
        return int(await self.session.scalar(count_statement) or 0)

    async def list(self, statement: Select[tuple[ModelT]], limit: int, offset: int) -> list[ModelT]:
        result = await self.session.execute(statement.limit(limit).offset(offset))
        return list(result.scalars().all())

    async def apply_updates(self, obj: ModelT, values: dict[str, Any]) -> ModelT:
        for field, value in values.items():
            setattr(obj, field, value)
        self.session.add(obj)
        await self.session.flush()
        await self.session.refresh(obj)
        return obj
