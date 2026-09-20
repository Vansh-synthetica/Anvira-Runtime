from typing import Protocol
from uuid import UUID

from app.schemas.memory import MemoryRead


class MemoryRetrieverPort(Protocol):
    """Future-facing interface for semantic/vector retrieval without coupling storage."""

    async def retrieve(
        self,
        user_id: UUID,
        query: str,
        *,
        limit: int = 20,
    ) -> list[MemoryRead]: ...
