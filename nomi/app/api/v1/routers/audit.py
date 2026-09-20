from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.dependencies.auth import get_current_user, require_admin
from app.models.user import User
from app.schemas.audit import AuditLogRead
from app.schemas.common import Page
from app.services.audit_queries import AuditQueryService

router = APIRouter()


@router.get("/me", response_model=Page[AuditLogRead])
async def get_my_audit_logs(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> Page[AuditLogRead]:
    return await AuditQueryService(session).list_me(user, limit, offset)


@router.get("/admin", response_model=Page[AuditLogRead])
async def get_admin_audit_logs(
    user: User = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> Page[AuditLogRead]:
    return await AuditQueryService(session).list_admin(user, limit, offset)

