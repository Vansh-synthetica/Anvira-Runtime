import logging
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config.settings import settings

logger = logging.getLogger("nomi.database")

engine_kwargs: dict[str, object] = {
    "echo": settings.DEBUG,
    "pool_pre_ping": True,
    "pool_size": 10,
    "max_overflow": 20,
    "pool_recycle": 3600,
}

if settings.database_url.startswith("sqlite"):
    engine_kwargs = {
        "echo": settings.DEBUG,
        "connect_args": {"check_same_thread": False},
        "poolclass": StaticPool,
    }

engine = create_async_engine(settings.database_url, **engine_kwargs)
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def close_engine() -> None:
    logger.info("Disposing database engine connection pool")
    await engine.dispose()
