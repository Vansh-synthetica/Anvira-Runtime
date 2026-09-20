# Anvira Runtime — Architecture

Anvira Runtime is one local service per OS user that composes the existing LocalHouseLLM infrastructure —
**ORCHA** (orchestration), **Nomi** (memory/identity), **AICL** (inter-module protocol) and a **llama.cpp**
model backend — behind a single token-authenticated loopback API, a CLI (`anvira`) and client SDKs.
Applications (Anvira, Anvira Notes, Anvira Study, Anvira Dev, …) are *clients*: they never start, configure or
talk to ORCHA, Nomi or AICL directly.

```
 Anvira        Anvira Notes        Anvira Study        Anvira Dev        anvira CLI / anvira ui
    └──────────────┴──────────────────┴──────────────────┴───────────────────────┘
                     SDKs (sdk/typescript, sdk/python)           HTTP 127.0.0.1 /v1  (Bearer token)
 ┌─────────────────────────────────── ANVIRA RUNTIME (one process) ───────────────────────────────────┐
 │  api/         auth, permissions, Host/Origin guards, structured errors                               │
 │  core/        RuntimeCore ─ jobs ─ memory (namespaces) ─ context index ─ chat ─ model→ORCHA sync      │
 │  core/bus     AICL bus: every module call is an aicl.bin packet (CRC32, opcode, trace)               │
 │  models/      catalog, hardware-aware compatibility, install/select/remove, storage locations,       │
 │               app-folder discovery, llama-server launcher                                            │
 │  process/     Supervisor: start, health, restart+backoff, adopt, stale-child reaping                 │
 │  hardware/ security/ config/ diagnostics/ (doctor, logs)                                             │
 └───────────┬───────────────────────┬────────────────────────────────┬────────────────────────────────┘
             │ private port + token  │ private port + JWT             │ private port
        ORCHA (unmodified)      Nomi (unmodified)              llama-server (per active local model)
        Orcha/desktop_entry.py  nomi/desktop_entry.py          or a cloud/remote OpenAI-compatible provider
```

## What was reused, wrapped, and created

| Existing | Treatment |
|---|---|
| `Orcha/` (FastAPI, graphs, agents, MCP, skills) | **Unmodified.** Launched via its own `desktop_entry.py`. The runtime holds its `ORCHA_API_TOKEN`, keeps it pointed at the active model through ORCHA's existing `/v1/local-model` API, and exposes runs as *jobs*. ORCHA event frames are passed through unchanged. |
| `nomi/` (FastAPI, SQLAlchemy, JWT) | **Unmodified.** Launched via `desktop_entry.py` (SQLite + migrations). The runtime owns one Nomi account and enforces application boundaries with reserved tags (`app:<id>`, `ws:<id>`, `scope:shared`). |
| `AICL/aicl` (new binary protocol) | **Unmodified.** The runtime is its first real consumer: `core/bus.py` frames every inter-module call as an `aicl.bin` `Packet` (opcodes `OP_CALL/OP_MEMORY_*/OP_INDEX_*/OP_EXECUTE/OP_CANCEL`, CRC32 trailer, correlation ids, `OP_ERROR`+`ErrorInfo`). |
| Anvira Electron code (`llamaServer.cjs`, `hardwareDetection.cjs`, `modelStorage.cjs`, `runtimeTracker.cjs`, `huggingface.ts`, `bm25.ts`, `orchaProcess.cjs`…) | **Ported** into Python with the same heuristics (GPU-layer fill 82%/90%, 4096 ctx floor, llama flags, GGUF header parsing, `nvidia-smi` quoting, adopt-live-service, pid tracking). `bm25.ts` ranking is parity-tested against Node. |
| New | supervisor, API, jobs, security, config/paths, model manager, doctor, CLI, TUI, installer, SDKs. |

Only one repo change outside `runtime/` and `sdk/`: the untracked *old* AICL files that had been copied over the new AICL
repo root (`__init__.py`, `router.py`, `packet.py`, …) were moved to `AICL/_legacy_v0/`. They broke AICL's own pytest
collection and nothing imports them.

## Process model

`anvira runtime start` spawns **one detached daemon** (`python -m anvira_runtime daemon`). It listens first (so
`/health` answers `starting`), writes `state/runtime.json` (pid, port, versions), then the supervisor starts:

* **Nomi** and **ORCHA** — own processes on *private ephemeral loopback ports* (never 8000/8420, so the legacy
  Anvira app's servers are untouched). ORCHA requires `X-Orcha-Token`; Nomi requires a JWT. Neither is reachable without
  runtime-held credentials.
* **AICL** — a library, not a process: the in-process bus. `anvira aicl status` reports whether the native Rust core is
  built (it currently is not; the pure-Python codec is used).
* **llama-server** — one process for the active *local* model (`model:<id>`), started on `models use`, on first use, or at
  runtime start (`models.autostart_active`). Cloud/remote providers need no process.

Supervisor behaviour: health probe every `supervisor.health_interval_s`; 3 failed probes or a dead process ⇒ restart with
exponential backoff (max `restart_limit` per `restart_window_s`, then `failed`); adopts a healthy service already on the
port instead of spawning a duplicate; records child pids in `state/children.json` and, after a runtime crash, reaps
orphans **only if their recorded health URL still answers** (a recycled pid is never killed). On Windows, liveness uses
`OpenProcess` because `os.kill(pid, 0)` would terminate the process. Shutdown stops services in parallel (~4 s).

The model→ORCHA link is reconciled every 5 s, so if ORCHA restarts it is re-pointed at the active model automatically.

## Two data planes

*Control plane* (status, models, memory, context, jobs) goes through the AICL bus. *Data plane* (chat token streams,
ORCHA SSE events) goes straight from the API to the model/ORCHA over loopback HTTP — streaming does not fit request/response
packets, and the bus would only add latency there.

## Shared installation and data model

```
<runtime home>   Windows %LOCALAPPDATA%\AnviraRuntime | macOS ~/Library/Application Support/AnviraRuntime | Linux ~/.local/share/anvira-runtime
  install.json  venv/  services/{orcha,nomi,aicl}/      the installation (shared by every app)
  config/config.json                                      runtime configuration (no secrets)
  state/   runtime.json owner.token register.token secrets.json apps.json app-tokens/ model-registry.json models.json jobs.json children.json
  data/    nomi/ (SQLite)  orcha/  context/context.db
  logs/    runtime.log orcha.log nomi.log model-<id>.log
  models/  default download location — configurable to ANY path
```

Override everything with `ANVIRA_RUNTIME_HOME`. Application workspace data (notebooks, study sets, projects, chats) is **not**
stored here; each app keeps its own data. Runtime state and app data never mix.

### Models: one library, anywhere on disk

The runtime keeps a single model library so nothing is downloaded twice:

1. `models.models_dir` — where *new* downloads go. Any local path (other drive, spaces, external disk), validated writable;
   `anvira model dir <path> --move` moves existing files (rename or copy+verify+unlink; never overwrites).
2. `models.extra_dirs` — extra folders scanned for `.gguf` (LM Studio, an old folder…).
3. **App discovery** (`models.discover_apps`, on by default) — the runtime reads `model-storage.json` and `models/` in the
   user-data folders of Anvira, Anvira Notes, Anvira Study and Anvira Dev, so a model downloaded through *any* of them is
   already known. Read-only; never moves or modifies app data.
4. **Announce** — an app can call `models.register(path)` (file or folder) to say where a model lives; it is linked in place.
5. Linked files at any path (`anvira model add <file>`), in place, never copied.

Whatever the source, the model has one id (`qwen2.5-coder-7b-instruct-q4_k_m`) and any request may name it (`model` in
`chat`/`task`/`agent`): the runtime switches the single inference backend to it and serves the request. Files outside the
download folder are never deleted without explicit confirmation.

### Application boundaries

* Memory: private to the storing app by default; `scope: "shared"` needs the `memory.shared` permission; a reader of a shared
  memory cannot delete another app's contribution; apps cannot forge `app:`/`scope:`/`ws:` tags; optional `workspace`
  boundary inside an app.
* Context index (chunk + BM25 retrieval — the product-neutral half of what Notes/Study did privately): per-app collections.
* Jobs: an app sees and cancels only its own jobs. The owner (CLI) sees all.

## Security model

Loopback only (`api.host` can only be `127.0.0.1`/`::1`/`localhost`); `Host` header must be loopback (DNS-rebinding);
requests with a browser `Origin` are rejected unless allow-listed; every route except `/health` and `/version` needs a
token; per-app tokens carry explicit permissions (`anvira app grant`), only SHA-256 hashes are stored; provider API keys
never leave `state/providers.json` and are never returned or logged (log redaction covers bearer tokens, `sk-…`, `hf_…`,
`api_key=…`). **Limit:** any process running as the same OS user can read the token files — this protects against the network,
web pages and over-reaching apps, not against malicious same-user code.

## What is deliberately not here

No database for the runtime itself (JSON files + one small SQLite for the context index), no broker, no containers, no
per-app SDKs beyond the two the products need (TypeScript for the Electron apps, stdlib Python for tooling/CLI).
