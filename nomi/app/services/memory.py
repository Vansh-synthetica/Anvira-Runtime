from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import MemoryType
from app.exceptions.errors import NotFoundError
from app.models.memory import Memory
from app.models.user import User
from app.repositories.memories import MemoryRepository
from app.schemas.common import Page
from app.schemas.memory import MemoryCreate, MemoryRead, MemoryUpdate
from app.services.audit import AuditContext, AuditService


class MemoryService:
    def __init__(self, session: AsyncSession) -> None:
        self.memories = MemoryRepository(session)
        self.audit = AuditService(session)

    async def create(self, user: User, payload: MemoryCreate, context: AuditContext | None = None) -> Memory:
        memory = await self.memories.add(Memory(user_id=user.id, **payload.model_dump()))
        await self.audit.record(
            action_type="memory.create",
            user_id=user.id,
            actor_user_id=user.id,
            resource_type="memory",
            resource_id=str(memory.id),
            context=context,
        )
        return memory

    async def list(
        self,
        user: User,
        *,
        limit: int,
        offset: int,
        query: str | None = None,
        memory_type: MemoryType | None = None,
        category: str | None = None,
        tag: str | None = None,
        include_archived: bool = False,
        include_deleted: bool = False,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        sort: str = "-created_at",
    ) -> Page[MemoryRead]:
        statement = self.memories.filter_statement(
            user.id,
            query=query,
            memory_type=memory_type,
            category=category,
            tag=tag,
            include_archived=include_archived,
            include_deleted=include_deleted,
            created_after=created_after,
            created_before=created_before,
            sort=sort,
        )
        total = await self.memories.count(statement)
        items = await self.memories.list(statement, limit, offset)
        return Page[MemoryRead](
            items=[MemoryRead.model_validate(item) for item in items],
            total=total,
            limit=limit,
            offset=offset,
        )

    async def get(self, user: User, memory_id: UUID, include_deleted: bool = False) -> Memory:
        memory = await self.memories.get_owned(user.id, memory_id, include_deleted=include_deleted)
        if not memory:
            raise NotFoundError("Memory not found")
        return memory

    async def update(
        self,
        user: User,
        memory_id: UUID,
        payload: MemoryUpdate,
        context: AuditContext | None = None,
    ) -> Memory:
        memory = await self.get(user, memory_id)
        values = payload.model_dump(exclude_unset=True)
        await self.memories.apply_updates(memory, values)
        await self.audit.record(
            action_type="memory.update",
            user_id=user.id,
            actor_user_id=user.id,
            resource_type="memory",
            resource_id=str(memory.id),
            context=context,
        )
        return memory

    async def soft_delete(self, user: User, memory_id: UUID, context: AuditContext | None = None) -> None:
        memory = await self.get(user, memory_id)
        memory.deleted_at = datetime.now(UTC)
        await self.audit.record(
            action_type="memory.delete",
            user_id=user.id,
            actor_user_id=user.id,
            resource_type="memory",
            resource_id=str(memory.id),
            context=context,
        )

    async def archive(self, user: User, memory_id: UUID, context: AuditContext | None = None) -> Memory:
        memory = await self.get(user, memory_id)
        memory.is_archived = True
        self.memories.session.add(memory)
        await self.memories.session.flush()
        await self.memories.session.refresh(memory)
        await self.audit.record(
            action_type="memory.archive",
            user_id=user.id,
            actor_user_id=user.id,
            resource_type="memory",
            resource_id=str(memory.id),
            context=context,
        )
        return memory

    async def unarchive(self, user: User, memory_id: UUID, context: AuditContext | None = None) -> Memory:
        memory = await self.get(user, memory_id)
        memory.is_archived = False
        self.memories.session.add(memory)
        await self.memories.session.flush()
        await self.memories.session.refresh(memory)
        await self.audit.record(
            action_type="memory.unarchive",
            user_id=user.id,
            actor_user_id=user.id,
            resource_type="memory",
            resource_id=str(memory.id),
            context=context,
        )
        return memory
