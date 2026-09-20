from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.dependencies.audit import get_audit_context
from app.dependencies.auth import get_current_user, require_admin
from app.exceptions.errors import NotFoundError
from app.models.user import User
from app.repositories.users import UserRepository
from app.schemas.common import Message
from app.schemas.user import UserRead, UserUpdate
from app.services.audit import AuditContext
from app.services.users import UserService

router = APIRouter()


@router.get("/me", response_model=UserRead)
async def get_me(user: User = Depends(get_current_user)) -> User:
    return user


@router.patch("/me", response_model=UserRead)
async def update_me(
    payload: UserUpdate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    audit_context: AuditContext = Depends(get_audit_context),
) -> User:
    return await UserService(session).update_me(user, payload, audit_context)


@router.delete("/me", response_model=Message)
async def delete_me(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    audit_context: AuditContext = Depends(get_audit_context),
) -> Message:
    await UserService(session).deactivate_me(user, audit_context)
    return Message(message="Account deactivated")


@router.get("/{user_id}", response_model=UserRead)
async def get_user(
    user_id: UUID,
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> User:
    user = await UserRepository(session).get(user_id)
    if not user:
        raise NotFoundError("User not found")
    return user

