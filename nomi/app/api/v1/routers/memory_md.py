"""HTTP surface for the file-backed markdown memory layer.

Complements (never replaces) the SQL memory domain: topic files + a capped
MEMORY.md index that hosts inject into system prompts, plus the
background-extraction cursor. Per-user isolation is enforced by keying the
store on the authenticated user's id; every mutation is audited for parity
with the SQL memory router.
"""

from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status

from app.config.settings import settings
from app.dependencies.audit import get_audit_context
from app.dependencies.auth import get_current_user
from app.memory_md.store import MemoryMdError, MarkdownMemoryStore
from app.models.user import User
from app.schemas.memory_md import (
    CursorRead,
    CursorUpdate,
    MemoryMdFile,
    MemoryMdFileSummary,
    MemoryMdIndex,
    MemoryMdWrite,
    RecallResponse,
)
from app.services.audit import AuditContext, AuditService
from app.db.session import get_db_session
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


def _store(user: User) -> MarkdownMemoryStore:
    return MarkdownMemoryStore(settings.memory_md_dir, str(user.id))


def _http_error(exc: MemoryMdError) -> Any:
    from fastapi import HTTPException

    code = {
        "not_found": status.HTTP_404_NOT_FOUND,
        "invalid_name": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "invalid_type": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "too_large": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    }.get(exc.code, status.HTTP_400_BAD_REQUEST)
    raise HTTPException(status_code=code, detail=exc.message)


@router.get("/index", response_model=MemoryMdIndex)
async def read_index(user: User = Depends(get_current_user)) -> MemoryMdIndex:
    content, truncated = _store(user).read_index()
    return MemoryMdIndex(content=content, truncated=truncated)


@router.get("/files", response_model=list[MemoryMdFileSummary])
async def list_files(user: User = Depends(get_current_user)) -> list[MemoryMdFileSummary]:
    return [MemoryMdFileSummary(**item) for item in _store(user).list_files()]


@router.get("/recall", response_model=RecallResponse)
async def recall(
    query: str = Query(..., min_length=1, max_length=500),
    limit: int = Query(default=10, ge=1, le=50),
    user: User = Depends(get_current_user),
) -> RecallResponse:
    """Deterministic candidate pre-filter; upstream LLM selection optional."""
    store = _store(user)
    candidates = store.recall_manifest(query, limit=limit)
    bodies: dict[str, str] = {}
    index_content, truncated = store.read_index()
    for item in candidates:
        try:
            bodies[item["name"]] = store.read_file(item["name"])["body"]
        except MemoryMdError:  # pragma: no cover — raced delete
            continue
    return RecallResponse(
        query=query,
        candidates=[MemoryMdFileSummary(**item) for item in candidates],
        bodies=bodies,
        index=index_content,
        index_truncated=truncated,
    )


@router.get("/cursor", response_model=CursorRead)
async def get_cursor(user: User = Depends(get_current_user)) -> CursorRead:
    return CursorRead(**_store(user).get_cursor())


@router.put("/cursor", response_model=CursorRead)
async def set_cursor(
    payload: CursorUpdate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    audit_context: AuditContext = Depends(get_audit_context),
) -> CursorRead:
    store = _store(user)
    store.set_cursor(payload.last_message_id)
    await AuditService(session).record(
        action_type="memory_md.cursor_update",
        user_id=user.id,
        actor_user_id=user.id,
        resource_type="memory_md",
        resource_id=str(store.dir.name),
        context=audit_context,
    )
    return CursorRead(last_message_id=payload.last_message_id)


@router.post("/files", response_model=MemoryMdFile, status_code=status.HTTP_201_CREATED)
async def write_file(
    payload: MemoryMdWrite,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    audit_context: AuditContext = Depends(get_audit_context),
) -> MemoryMdFile:
    store = _store(user)
    try:
        created = store.write_file(
            payload.name,
            title=payload.title,
            description=payload.description,
            type_=payload.type,
            body=payload.body,
        )
    except MemoryMdError as exc:
        _http_error(exc)
    await AuditService(session).record(
        action_type="memory_md.write",
        user_id=user.id,
        actor_user_id=user.id,
        resource_type="memory_md",
        resource_id=str(created["name"]),
        context=audit_context,
    )
    return MemoryMdFile(**{**created, "description": payload.description, "body": payload.body})


@router.get("/files/{name}", response_model=MemoryMdFile)
async def read_file(name: str, user: User = Depends(get_current_user)) -> MemoryMdFile:
    try:
        data = _store(user).read_file(name)
    except MemoryMdError as exc:
        _http_error(exc)
    return MemoryMdFile(**data)


@router.delete("/files/{name}", response_model=dict[str, bool])
async def delete_file(
    name: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    audit_context: AuditContext = Depends(get_audit_context),
) -> dict[str, bool]:
    try:
        deleted = _store(user).delete_file(name)
    except MemoryMdError as exc:
        _http_error(exc)
    if deleted:
        await AuditService(session).record(
            action_type="memory_md.delete",
            user_id=user.id,
            actor_user_id=user.id,
            resource_type="memory_md",
            resource_id=name,
            context=audit_context,
        )
    return {"deleted": deleted}


__all__ = ["router"]
