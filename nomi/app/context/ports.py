from typing import Any, Protocol
from uuid import UUID


class ContextAssemblerPort(Protocol):
    """Future-facing interface for permission-aware context assembly."""

    async def build_context(
        self,
        user_id: UUID,
        *,
        application_name: str,
        application_id: str | None = None,
        query: str | None = None,
        max_memories: int = 20,
    ) -> dict[str, Any]: ...
