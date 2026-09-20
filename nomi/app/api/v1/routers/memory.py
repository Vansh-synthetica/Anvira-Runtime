from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import MemoryType
from app.db.session import get_db_session
from app.dependencies.audit import get_audit_context
from app.dependencies.auth import get_current_user
from app.models.memory import Memory
from app.models.user import User
from app.schemas.common import Message, Page
from app.schemas.memory import MemoryCreate, MemoryRead, MemoryUpdate
from app.services.audit import AuditContext
from app.services.memory import MemoryService

router = APIRouter()


@router.post("", response_model=MemoryRead, status_code=status.HTTP_201_CREATED)
async def create_memory(
    payload: MemoryCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    audit_context: AuditContext = Depends(get_audit_context),
) -> Memory:
    return await MemoryService(session).create(user, payload, audit_context)


@router.get("", response_model=Page[MemoryRead])
async def list_memories(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
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
    return await MemoryService(session).list(
        user,
        limit=limit,
        offset=offset,
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


@router.get("/{memory_id}", response_model=MemoryRead)
async def get_memory(
    memory_id: UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> Memory:
    return await MemoryService(session).get(user, memory_id)


@router.patch("/{memory_id}", response_model=MemoryRead)
async def update_memory(
    memory_id: UUID,
    payload: MemoryUpdate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    audit_context: AuditContext = Depends(get_audit_context),
) -> Memory:
    return await MemoryService(session).update(user, memory_id, payload, audit_context)


@router.delete("/{memory_id}", response_model=Message)
async def delete_memory(
    memory_id: UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    audit_context: AuditContext = Depends(get_audit_context),
) -> Message:
    await MemoryService(session).soft_delete(user, memory_id, audit_context)
    return Message(message="Memory deleted")


@router.post("/{memory_id}/archive", response_model=MemoryRead)
async def archive_memory(
    memory_id: UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    audit_context: AuditContext = Depends(get_audit_context),
) -> Memory:
    return await MemoryService(session).archive(user, memory_id, audit_context)


@router.post("/{memory_id}/unarchive", response_model=MemoryRead)
async def unarchive_memory(
    memory_id: UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    audit_context: AuditContext = Depends(get_audit_context),
) -> Memory:
    return await MemoryService(session).unarchive(user, memory_id, audit_context)

