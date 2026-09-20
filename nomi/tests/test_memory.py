import pytest
from httpx import AsyncClient

from tests.conftest import auth_headers


@pytest.mark.asyncio
async def test_memory_crud_filter_archive_and_soft_delete(client: AsyncClient) -> None:
    headers = await auth_headers(client)
    created = await client.post(
        "/api/v1/memory",
        json={
            "title": "Output preferences",
            "content": "Prefers concise backend explanations.",
            "memory_type": "preference",
            "category": "writing",
            "tags": ["style", "backend"],
        },
        headers=headers,
    )
    assert created.status_code == 201, created.text
    memory_id = created.json()["id"]

    listed = await client.get("/api/v1/memory?query=concise&tag=style", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["total"] == 1

    patched = await client.patch(
        f"/api/v1/memory/{memory_id}",
        json={"importance": 5},
        headers=headers,
    )
    assert patched.status_code == 200
    assert patched.json()["importance"] == 5

    archived = await client.post(f"/api/v1/memory/{memory_id}/archive", headers=headers)
    assert archived.status_code == 200
    assert archived.json()["is_archived"] is True

    hidden = await client.get("/api/v1/memory", headers=headers)
    assert hidden.json()["total"] == 0

    visible = await client.get("/api/v1/memory?include_archived=true", headers=headers)
    assert visible.json()["total"] == 1

    deleted = await client.delete(f"/api/v1/memory/{memory_id}", headers=headers)
    assert deleted.status_code == 200

    missing = await client.get(f"/api/v1/memory/{memory_id}", headers=headers)
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_memory_owner_isolation(client: AsyncClient) -> None:
    owner_headers = await auth_headers(client, "owner@example.com")
    other_headers = await auth_headers(client, "other@example.com")
    created = await client.post(
        "/api/v1/memory",
        json={"title": "Private", "content": "Only owner can read"},
        headers=owner_headers,
    )
    denied = await client.get(f"/api/v1/memory/{created.json()['id']}", headers=other_headers)
    assert denied.status_code == 404

