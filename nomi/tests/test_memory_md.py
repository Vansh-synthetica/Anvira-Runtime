"""End-to-end tests for the markdown memory layer (/api/v1/memory/md/*)."""

from collections.abc import AsyncIterator

import pytest
from httpx import AsyncClient

from app.config.settings import settings
from tests.conftest import auth_headers


@pytest.fixture(autouse=True)
def memory_md_dir(tmp_path, monkeypatch) -> AsyncIterator[None]:
    """Point the md layer at a per-test directory (conftest supplies client)."""
    monkeypatch.setattr(settings, "MEMORY_MD_DIR", str(tmp_path / "md"))
    yield


@pytest.mark.asyncio
async def _write(client: AsyncClient, headers: dict, **overrides) -> dict:
    payload = {
        "name": "user_role",
        "title": "User role",
        "description": "Works on backend services",
        "type": "user",
        "body": "The user is a backend engineer who prefers Python.",
        **overrides,
    }
    created = await client.post("/api/v1/memory/md/files", json=payload, headers=headers)
    assert created.status_code == 201, created.text
    return payload


@pytest.mark.asyncio
async def test_write_read_index_and_list(client: AsyncClient) -> None:
    headers = await auth_headers(client)
    await _write(client, headers)

    listed = await client.get("/api/v1/memory/md/files", headers=headers)
    assert listed.status_code == 200
    items = listed.json()
    assert len(items) == 1
    assert items[0]["name"] == "user_role"
    assert items[0]["type"] == "user"

    index = await client.get("/api/v1/memory/md/index", headers=headers)
    assert index.status_code == 200
    body = index.json()
    assert not body["truncated"]
    assert "[User role](user_role.md)" in body["content"]
    assert "backend services" in body["content"]

    single = await client.get("/api/v1/memory/md/files/user_role", headers=headers)
    assert single.status_code == 200
    data = single.json()
    assert data["title"] == "User role"
    assert "backend engineer" in data["body"]


@pytest.mark.asyncio
async def test_index_upsert_and_delete_removes_pointer(client: AsyncClient) -> None:
    headers = await auth_headers(client)
    await _write(client, headers)
    # Overwrite with a new description → index line replaced, not duplicated.
    await _write(client, headers, description="Now leads the platform team")

    index = (await client.get("/api/v1/memory/md/index", headers=headers)).json()
    assert index["content"].count("user_role.md") == 1
    assert "platform team" in index["content"]

    deleted = await client.delete("/api/v1/memory/md/files/user_role", headers=headers)
    assert deleted.status_code == 200 and deleted.json()["deleted"] is True
    index_after = (await client.get("/api/v1/memory/md/index", headers=headers)).json()
    assert "user_role.md" not in index_after["content"]

    missing = await client.get("/api/v1/memory/md/files/user_role", headers=headers)
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_owner_isolation_between_users(client: AsyncClient) -> None:
    headers_a = await auth_headers(client)
    await _write(client, headers_a)

    # Register + login a second user; confirm total isolation.
    reg = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "second@example.com",
            "username": "second",
            "password": "correct-password",
        },
    )
    assert reg.status_code == 201, reg.text
    token_b = reg.json()["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}

    theirs = await client.get("/api/v1/memory/md/files", headers=headers_b)
    assert theirs.status_code == 200
    assert theirs.json() == []
    gone = await client.get("/api/v1/memory/md/files/user_role", headers=headers_b)
    assert gone.status_code == 404


@pytest.mark.asyncio
async def test_invalid_type_rejected(client: AsyncClient) -> None:
    headers = await auth_headers(client)
    bad = await client.post(
        "/api/v1/memory/md/files",
        json={"name": "x", "title": "X", "description": "", "type": "secret", "body": "b"},
        headers=headers,
    )
    assert bad.status_code == 422


@pytest.mark.asyncio
async def test_recall_ranks_matching_files_first(client: AsyncClient) -> None:
    headers = await auth_headers(client)
    await _write(
        client, headers,
        name="python_style", title="Python style",
        description="Formatting preferences for Python code",
        body="Use ruff. Line length 100.", type="reference",
    )
    await _write(
        client, headers,
        name="garden", title="Garden project",
        description="Home garden layout", body="Tomatoes front.", type="project",
    )

    recall = await client.get(
        "/api/v1/memory/md/recall", params={"query": "python formatting"}, headers=headers,
    )
    assert recall.status_code == 200
    data = recall.json()
    names = [c["name"] for c in data["candidates"]]
    assert names[0] == "python_style"
    assert "ruff" in data["bodies"]["python_style"]


@pytest.mark.asyncio
async def test_cursor_roundtrip(client: AsyncClient) -> None:
    headers = await auth_headers(client)
    empty = await client.get("/api/v1/memory/md/cursor", headers=headers)
    assert empty.status_code == 200 and empty.json()["last_message_id"] is None

    put = await client.put(
        "/api/v1/memory/md/cursor", json={"last_message_id": "msg-42"}, headers=headers,
    )
    assert put.status_code == 200
    got = await client.get("/api/v1/memory/md/cursor", headers=headers)
    assert got.json()["last_message_id"] == "msg-42"


@pytest.mark.asyncio
async def test_requires_auth(client: AsyncClient) -> None:
    anon = await client.get("/api/v1/memory/md/index")
    assert anon.status_code in (401, 403)


# conftest helpers re-exported for readability of the tests above.
from tests.conftest import auth_headers  # noqa: E402
