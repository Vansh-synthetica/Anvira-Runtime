import os
from collections.abc import AsyncGenerator

os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite://")
os.environ.setdefault("SECRET_KEY", "test-secret-key-with-at-least-thirty-two-characters")
os.environ.setdefault("BACKEND_CORS_ORIGINS", "[]")

import asyncio
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.db.base import Base
from app.db.session import engine
from app.main import app

# Use SelectorEventLoopPolicy on Windows to avoid aiosqlite deadlocks on Python 3.13+
if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

@pytest_asyncio.fixture(autouse=True)
async def reset_database() -> AsyncGenerator[None, None]:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client


async def register_user(
    client: AsyncClient,
    *,
    email: str = "owner@example.com",
    username: str = "owner",
    password: str = "correct-password",
) -> dict[str, object]:
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "username": username, "password": password},
    )
    assert response.status_code == 201, response.text
    return response.json()


async def auth_headers(client: AsyncClient, email: str = "owner@example.com") -> dict[str, str]:
    data = await register_user(client, email=email, username=email.split("@")[0])
    return {"Authorization": f"Bearer {data['access_token']}"}

