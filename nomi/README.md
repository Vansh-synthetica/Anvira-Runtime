# Nomi

Nomi is the identity, memory, and permission layer for AI in the LocalHouseLLM
ecosystem. It is not a chatbot or an LLM. It is a portable backend foundation for
user-owned AI context across tools.

## What Is Included

- FastAPI REST API under `/api/v1`
- JWT registration, login, refresh, logout hook, and current user endpoints
- Owner-scoped identity profiles
- Structured memory CRUD with filters, archive, and soft delete
- Permission records for future apps/connectors
- Append-only audit logging for sensitive actions
- SQLAlchemy 2 async models, repositories, and services
- Alembic migrations for PostgreSQL
- Docker Compose with PostgreSQL and Redis
- Pytest coverage for core flows

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
alembic upgrade head
uvicorn app.main:app --reload
```

API docs are available at `http://localhost:8000/docs`.

## Docker

```powershell
docker compose up --build
```

The API starts on `http://localhost:8000`.

## Useful Commands

```powershell
pytest
ruff check .
ruff format .
mypy app
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```

## Architecture

Routers validate HTTP input and delegate to services. Services own application
rules. Repositories own persistence queries. Models are SQLAlchemy ORM types and
schemas are Pydantic API contracts.

See [docs/architecture.md](docs/architecture.md).

## API Examples

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","username":"you","password":"correct-password"}'
```

```bash
curl -X POST http://localhost:8000/api/v1/memory \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"title":"Writing style","content":"Prefers concise answers.","memory_type":"preference","tags":["style"]}'
```

## Roadmap Extension Points

- Semantic and vector search
- Graph-backed relationships
- Encrypted memory fields
- Sync and offline conflict resolution
- AI tool connectors
- Context assembly and ranking
- SDKs, CLI, dashboard, plugin ecosystem, and protocol specs

## Contributing

Keep business logic out of routers, database access out of services, and external
systems behind adapters. Add tests for new behavior and update migrations when
models change.

## License

MIT license placeholder. Replace this section with the final project license text
before public release.

