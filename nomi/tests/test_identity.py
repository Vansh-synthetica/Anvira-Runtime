import pytest
from httpx import AsyncClient

from tests.conftest import auth_headers


@pytest.mark.asyncio
async def test_identity_crud(client: AsyncClient) -> None:
    headers = await auth_headers(client)
    created = await client.post(
        "/api/v1/identity",
        json={"display_name": "Nomi Owner", "interests": ["local AI"], "preferences": {"format": "bullets"}},
        headers=headers,
    )
    assert created.status_code == 201, created.text

    fetched = await client.get("/api/v1/identity/me", headers=headers)
    assert fetched.status_code == 200
    assert fetched.json()["display_name"] == "Nomi Owner"

    patched = await client.patch("/api/v1/identity/me", json={"tone": "direct"}, headers=headers)
    assert patched.status_code == 200
    assert patched.json()["tone"] == "direct"

    deleted = await client.delete("/api/v1/identity/me", headers=headers)
    assert deleted.status_code == 200


@pytest.mark.asyncio
async def test_identity_is_owner_scoped(client: AsyncClient) -> None:
    owner_headers = await auth_headers(client, "owner@example.com")
    other_headers = await auth_headers(client, "other@example.com")
    created = await client.post("/api/v1/identity", json={"display_name": "Owner"}, headers=owner_headers)

    denied = await client.get(f"/api/v1/identity/{created.json()['id']}", headers=other_headers)
    assert denied.status_code == 403

