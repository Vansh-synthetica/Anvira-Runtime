# Runtime API (v1)

Base URL: `http://127.0.0.1:<port>` — read the port from `<runtime home>/state/runtime.json` (or use an SDK, which does).
All bodies are JSON. Auth: `Authorization: Bearer <token>` (an app token, or the owner token the CLI uses).
Browser `Origin` headers and non-loopback `Host` headers are rejected (403 / 421).

Public (no token): `GET /health`, `GET /version`.

## Errors

Every failure is `{"error": {"code", "message", "status", "hint"?, "details"?}}`. Branch on `code`, never on text.

| status | codes (non-exhaustive) |
|---|---|
| 400 | `invalid_request`, `invalid_config`, `invalid_graph`, `invalid_tag`, `invalid_scope`, `invalid_model_file`, `storage_unwritable`, `unknown_permission` |
| 401 / 403 / 421 | `unauthorized`, `permission_denied` (message names the `anvira app grant …` fix), `origin_not_allowed`, `invalid_host`, `memory_not_owned`, `access_denied`, `user_approval_required`, `remote_context_not_allowed` |
| 404 | `model_not_found`, `job_not_found`, `memory_not_found`, `app_not_found`, `resource_not_found`, `request_not_found`, `lease_not_found`, `not_found` |
| 409 | `no_model`, `model_not_installed`, `model_incomplete`, `job_not_cancellable`, `confirmation_required`, `app_exists`, `request_pending`, `request_decided`, `already_allowed` |
| 410 | `resource_unavailable` (the file behind a resource is gone) |
| 424 / 5xx | `backend_missing` (no llama-server), `model_start_failed`, `model_crashed`, `model_loading`, `model_error`, `nomi_unavailable`, `orcha_unavailable`, `timeout`, `insufficient_storage` (job error), `internal_error` (never a traceback) |

## Detection & status

| | |
|---|---|
| `GET /health` | `{status: starting\|ok\|degraded, degraded: [...], runtime_version, api_version}` |
| `GET /version` | `{runtime_version, api_version, api_revision, min_client_api, capabilities[]}` |
| `GET /v1/status` | services (ORCHA/Nomi/AICL), active model + backend, job counts, paths, capabilities |
| `GET /v1/apps/me` | caller identity and granted permissions |

## Models & hardware  (`models.read` unless noted)

| | |
|---|---|
| `GET /v1/hardware[?refresh=true]` | CPU, RAM, GPU/VRAM/CUDA, backends, storage, OS/arch |
| `GET /v1/models[?installed=true]` · `/installed` · `/catalog[?q=]` · `/recommended` · `/search?q=` · `/active` · `/discovered` · `/storage` | discovery. Each record has `compatibility {can_run, mode: gpu\|partial-gpu\|cpu\|insufficient\|remote, needs_mib, reasons[]}` |
| `GET /v1/models/{id}` · `/{id}/compatibility` | one model; fit against detected hardware |
| `POST /v1/models/select` `{id, wait_s?}` (`models.select`) | make active; starts llama-server for local models |
| `POST /v1/models/deselect` (`models.select`) | stop and clear |
| `POST /v1/models/register` `{path}` (`models.register`) | announce a `.gguf` file or a models folder; no copy/download |
| `POST /v1/models/install` `{model\|url, file?, dir?, wait?}` → 202 job (`models.manage`) | catalog id, `owner/repo`, or URL; disk check first; resumable |
| `POST /v1/models/remove` `{id, delete_file?, confirm?}` (`models.manage`) | outside-folder deletes need `confirm` |
| `PUT /v1/models/storage` `{path, move?}` · `POST/DELETE /v1/models/storage/dirs` · `POST /v1/models/link` (`models.manage`) | where models live |
| `GET/POST /v1/providers` · `DELETE /v1/providers/{id}` | cloud/remote OpenAI-compatible connections; keys are write-only |

## Chat, tasks, agents, jobs

| | |
|---|---|
| `POST /v1/chat` `{messages, stream?, model?, memory?: {recall, limit, scope, workspace}, temperature, max_tokens, …}` (`chat`) | OpenAI-shaped response plus `runtime {model, kind, memories_used, context_used, warnings}`. `stream:true` → SSE (`event: runtime` first; failures arrive as `event: error` with the structured error). `model` may name any installed model (switches the backend). |
| `POST /v1/task` `{task, model?, wait?=true, …}` · `POST /v1/orcha/run` `{task, graph: default\|research\|multi_agent, reasoning?, …}` · `POST /v1/agent/run` `{task, workspace_roots[], …}` (`orcha.run`) | each returns `{job}`; `wait:true` blocks |
| `GET /v1/jobs[?state&kind&limit]` · `GET /v1/jobs/{id}` · `POST /v1/jobs/{id}/cancel` | uniform for ORCHA runs and model installs; states `queued running completed failed cancelled interrupted` |
| `GET /v1/jobs/{id}/events` | SSE passthrough of ORCHA's event frames (`node`, `kind`, `data`) |
| `GET /v1/orcha/status` (`orcha.read`) | service + engine (experts, synthesizer) + active model |

A completed ORCHA job's `result`: `{run_id, status, answer, confidence, synthesized, contributors[], iterations, latency_s, graph, agent_steps[], agent_tool_calls[]}`.
Passthrough run options: `mode, access_mode, allow_tools, require_approval, system_prompt, max_cost, max_iterations, workspace_roots, tools, allow_rules, deny_rules, ask_rules, capabilities, reasoning, messages, prompt_parts, timeout_s`.

## Memory (Nomi)  — `memory.read` / `memory.write` / `memory.shared`

| | |
|---|---|
| `POST /v1/memory` `{content, title?, type?, tags?, importance?, scope?: app\|shared, workspace?, extra?}` | store |
| `GET /v1/memory/search?q=&limit=&scope=app\|shared\|all&workspace=&tag=&type=` | word-based BM25 over the caller's namespace (default: app-private) |
| `GET /v1/memory[?limit&scope&workspace&type]` · `GET/DELETE /v1/memory/{id}` | list · get · delete (own or shared-authored only) |
| `GET /v1/nomi/status` | service + reachability |

## Context index  — `context.read` / `context.write`

`PUT /v1/context/{collection}/documents/{doc}` `{text, title?, metadata?}` · `POST /v1/context/{collection}/search` `{query, limit?, doc_ids?}` ·
`GET /v1/context` · `GET /v1/context/{collection}/documents` · `DELETE /v1/context/{collection}[/documents/{doc}]`.
Per-app namespaces; chunking and BM25 identical to Anvira's `bm25.ts`.

## Shared resources, workspaces, capabilities  — `context.read` / `context.write` / `context.share`

Private by default, shared by named app, global only by the user, referenced not copied, audited. Full model in
[SHARED_CONTEXT.md](SHARED_CONTEXT.md). Routes: `/v1/resources` (create, list, get, patch, delete, `read`, `documents/{doc}`,
`search`, `resolve`, `share`, `revoke`, `permissions`, `request`, `requests`, `audit`), `/v1/workspaces`, `/v1/capabilities`.
`POST /v1/chat` and `/v1/orcha/run` accept `context: {query?, resources?, limit?}`; a remote model needs `allow_remote_context: true`.

## Lifecycle (on-demand runtime)

`POST /v1/leases` · `POST /v1/leases/{id}/heartbeat` · `DELETE /v1/leases/{id}` · `GET /v1/lifecycle` — see [LIFECYCLE.md](LIFECYCLE.md).

## Configuration, apps, operations

| | |
|---|---|
| `GET /v1/runtime/config` (`config.read`) · `PUT /v1/runtime/config` `{key, value}` (`config.write`) | validated; response says `restart_required` |
| `POST /v1/apps/register` (header `X-Anvira-Register-Token`) `{app_id, name?, permissions?}` | returns the app token **once**; extra permissions are recorded as *requested*, not granted |
| `GET /v1/apps` · `POST /v1/apps/{id}/grant\|deny` `{permissions}` · `DELETE /v1/apps/{id}` (`apps.admin`) | user-side management |
| `GET /v1/runtime/logs?service=runtime\|orcha\|nomi\|model:<id>&lines=` · `GET /v1/diagnostics` · `GET /v1/aicl/status` · `POST /v1/runtime/services/{orcha\|nomi}/restart` · `POST /v1/runtime/stop` (`runtime.admin`) | operations |

## Permissions

Default for a new app: `models.read models.select models.register chat orcha.run orcha.read memory.read memory.write context.read context.write context.share config.read`.
Needs the user: `orcha.exec` (lets ORCHA agents run commands: `terminal`/`git` capabilities; without it `agent.run` still gets file/search tools inside its workspace roots), `models.manage` (downloads/removal/providers/storage), `memory.shared`, `config.write`, `runtime.admin`, `apps.admin`.

## Stability

Everything under `/v1` is the stable contract. Internals (ORCHA/Nomi endpoints, AICL packets, ports) are not part of it and
are not reachable by apps. Additive changes bump `api_revision`; a breaking change would introduce `/v2` alongside `/v1`
(see [VERSIONING.md](VERSIONING.md)).
