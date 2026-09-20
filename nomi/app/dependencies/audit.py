from fastapi import Request

from app.services.audit import AuditContext


def get_audit_context(request: Request) -> AuditContext:
    forwarded = request.headers.get("x-forwarded-for")
    ip_address = forwarded.split(",")[0].strip() if forwarded else None
    if not ip_address and request.client:
        ip_address = request.client.host
    return AuditContext(
        ip_address=ip_address,
        user_agent=request.headers.get("user-agent"),
    )

