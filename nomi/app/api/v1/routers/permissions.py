from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.dependencies.audit import get_audit_context
from app.dependencies.auth import get_current_user
from app.models.permission import Permission
from app.models.user import User
from app.schemas.common import Message, Page
from app.schemas.permission import PermissionCreate, PermissionRead, PermissionUpdate
from app.services.audit import AuditContext
from app.services.permissions import PermissionService

router = APIRouter()


@router.post("", response_model=PermissionRead, status_code=status.HTTP_201_CREATED)
async def create_permission(
    payload: PermissionCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    audit_context: AuditContext = Depends(get_audit_context),
) -> Permission:
    return await PermissionService(session).create(user, payload, audit_context)


@router.get("", response_model=Page[PermissionRead])
async def list_permissions(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> Page[PermissionRead]:
    return await PermissionService(session).list(user, limit, offset)


@router.get("/{permission_id}", response_model=PermissionRead)
async def get_permission(
    permission_id: UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> Permission:
    return await PermissionService(session).get(user, permission_id)


@router.patch("/{permission_id}", response_model=PermissionRead)
async def update_permission(
    permission_id: UUID,
    payload: PermissionUpdate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    audit_context: AuditContext = Depends(get_audit_context),
) -> Permission:
    return await PermissionService(session).update(user, permission_id, payload, audit_context)


@router.delete("/{permission_id}", response_model=Message)
async def delete_permission(
    permission_id: UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    audit_context: AuditContext = Depends(get_audit_context),
) -> Message:
    await PermissionService(session).revoke(user, permission_id, audit_context)
    return Message(message="Permission deleted")

