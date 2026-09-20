from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.dependencies.audit import get_audit_context
from app.dependencies.auth import get_current_user
from app.models.user import User
from app.schemas.auth import AuthResponse, LoginRequest, RefreshRequest, RegisterRequest, TokenPair
from app.schemas.common import Message
from app.schemas.user import UserRead
from app.services.audit import AuditContext
from app.services.auth import AuthService

router = APIRouter()


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(
    payload: RegisterRequest,
    session: AsyncSession = Depends(get_db_session),
    audit_context: AuditContext = Depends(get_audit_context),
) -> AuthResponse:
    return await AuthService(session).register(payload, audit_context)


@router.post("/login", response_model=AuthResponse)
async def login(
    payload: LoginRequest,
    session: AsyncSession = Depends(get_db_session),
    audit_context: AuditContext = Depends(get_audit_context),
) -> AuthResponse:
    return await AuthService(session).login(payload.email, payload.password, audit_context)


@router.post("/refresh", response_model=TokenPair)
async def refresh(
    payload: RefreshRequest,
    session: AsyncSession = Depends(get_db_session),
    audit_context: AuditContext = Depends(get_audit_context),
) -> TokenPair:
    return await AuthService(session).refresh(payload.refresh_token, audit_context)


@router.post("/logout", response_model=Message)
async def logout(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    audit_context: AuditContext = Depends(get_audit_context),
) -> Message:
    await AuthService(session).logout(user, audit_context)
    return Message(message="Logged out")


@router.get("/me", response_model=UserRead)
async def me(user: User = Depends(get_current_user)) -> User:
    return user

