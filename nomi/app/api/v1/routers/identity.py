from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.dependencies.audit import get_audit_context
from app.dependencies.auth import get_current_user
from app.models.identity_profile import IdentityProfile
from app.models.user import User
from app.schemas.common import Message
from app.schemas.identity import IdentityCreate, IdentityRead, IdentityUpdate
from app.services.audit import AuditContext
from app.services.identity import IdentityService

router = APIRouter()


@router.post("", response_model=IdentityRead, status_code=status.HTTP_201_CREATED)
async def create_identity(
    payload: IdentityCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    audit_context: AuditContext = Depends(get_audit_context),
) -> IdentityProfile:
    return await IdentityService(session).create(user, payload, audit_context)


@router.get("/me", response_model=IdentityRead)
async def get_my_identity(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> IdentityProfile:
    return await IdentityService(session).get_me(user)


@router.get("/{profile_id}", response_model=IdentityRead)
async def get_identity(
    profile_id: UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> IdentityProfile:
    return await IdentityService(session).get_by_id(user, profile_id)


@router.patch("/me", response_model=IdentityRead)
async def update_my_identity(
    payload: IdentityUpdate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    audit_context: AuditContext = Depends(get_audit_context),
) -> IdentityProfile:
    return await IdentityService(session).update_me(user, payload, audit_context)


@router.delete("/me", response_model=Message)
async def delete_my_identity(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    audit_context: AuditContext = Depends(get_audit_context),
) -> Message:
    await IdentityService(session).delete_me(user, audit_context)
    return Message(message="Identity profile deleted")

