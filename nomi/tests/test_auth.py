import pytest
from httpx import AsyncClient

from tests.conftest import register_user


@pytest.mark.asyncio
async def test_register_login_refresh_and_me(client: AsyncClient) -> None:
    registered = await register_user(client)
    assert registered["user"]["email"] == "owner@example.com"
    assert "hashed_password" not in registered["user"]

    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "owner@example.com", "password": "correct-password"},
    )
    assert login.status_code == 200
    token = login.json()["access_token"]

    me = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["username"] == "owner"

    refresh = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": login.json()["refresh_token"]},
    )
    assert refresh.status_code == 200
    assert refresh.json()["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_rejects_invalid_password(client: AsyncClient) -> None:
    await register_user(client)
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "owner@example.com", "password": "wrong-password"},
    )
    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"

