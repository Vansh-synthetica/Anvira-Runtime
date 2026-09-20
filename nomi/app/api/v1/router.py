from fastapi import APIRouter

from app.api.v1.routers import audit, auth, identity, memory, memory_md, permissions, users

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(identity.router, prefix="/identity", tags=["identity"])
api_router.include_router(memory.router, prefix="/memory", tags=["memory"])
api_router.include_router(memory_md.router, prefix="/memory/md", tags=["memory-md"])
api_router.include_router(permissions.router, prefix="/permissions", tags=["permissions"])
api_router.include_router(audit.router, prefix="/audit", tags=["audit"])

