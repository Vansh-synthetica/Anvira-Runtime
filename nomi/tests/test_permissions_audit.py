import pytest
from httpx import AsyncClient

from tests.conftest import auth_headers


@pytest.mark.asyncio
async def test_permission_crud_and_audit_log(client: AsyncClient) -> None:
    headers = await auth_headers(client)
    created = await client.post(
        "/api/v1/permissions",
        json={
            "application_name": "Cursor",
            "application_id": "cursor-local",
            "target_type": "memory",
            "target_id": "writing",
            "permission_level": "read",
            "scope": "memory:read",
        },
        headers=headers,
    )
    assert created.status_code == 201, created.text
    permission_id = created.json()["id"]

    listed = await client.get("/api/v1/permissions", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["total"] == 1

    patched = await client.patch(
        f"/api/v1/permissions/{permission_id}",
        json={"permission_level": "deny", "is_active": False},
        headers=headers,
    )
    assert patched.status_code == 200
    assert patched.json()["permission_level"] == "deny"

    audit = await client.get("/api/v1/audit/me", headers=headers)
    assert audit.status_code == 200
    action_types = {item["action_type"] for item in audit.json()["items"]}
    assert "user.registration" in action_types
    assert "permission.update" in action_types

    deleted = await client.delete(f"/api/v1/permissions/{permission_id}", headers=headers)
    assert deleted.status_code == 200


@pytest.mark.asyncio
async def test_audit_admin_endpoint_requires_admin(client: AsyncClient) -> None:
    headers = await auth_headers(client)
    response = await client.get("/api/v1/audit/admin", headers=headers)
    assert response.status_code == 403

