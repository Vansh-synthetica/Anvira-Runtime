# Nomi Backend Architecture

Nomi is organized around clean boundaries:

- `models`: SQLAlchemy persistence models.
- `schemas`: Pydantic request and response contracts.
- `repositories`: database access and query composition.
- `services`: application use cases and business rules.
- `api`: versioned FastAPI routers.
- `dependencies`: auth, session, and request context injection.

Future systems such as semantic search, encrypted memory storage, sync, SDKs, and AI
tool connectors should enter through new adapters and services instead of route-level
business logic.

