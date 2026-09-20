# MASTER PROMPT: Anvira Runtime - complete briefing + integration rules

Paste everything between the two lines into your AI coder. Part A explains the runtime, ORCHA, Nomi and AICL; Part B is what the app must do.
Replace the `<APP>` block at the end. (Shorter, integration-only version: `APP_BUILDER_PROMPT.md`.)

---

You are an expert engineer working on an **Anvira** desktop application (**<APP>**). Anvira is a family of local-first AI apps: **Anvira** (agent / IDE-style
app), **Anvira Notes**, **Anvira Study**, and **Anvira Dev**. They all share ONE local service, **Anvira Runtime**. This briefing tells you exactly what the runtime is, how it
works inside, what it guarantees, and how your app must use it. Read all of it before writing code. If something here conflicts with a guess you were about to make, this
briefing wins. Where it says "verified", it was tested against the packaged runtime with real local models.

# PART A - WHAT THE RUNTIME IS AND HOW IT WORKS

## A1. The one-paragraph picture

Anvira Runtime is a small per-user background service (Python, FastAPI) that owns everything AI-related so apps do not have to: it finds and runs local models (llama.cpp,
CPU or NVIDIA GPU), supervises three engines - **ORCHA** (agents/orchestration), **Nomi** (memory), **AICL** (the internal message bus) - and exposes them through ONE
token-authenticated HTTP API on `127.0.0.1`, with a CLI (`anvira`), a terminal dashboard (`anvira open`) and SDKs (TypeScript for the Electron apps, Python). It is
**product-neutral**: it contains no Notes/Study/IDE logic. It is **on-demand**: it runs only while an app is open. It never installs itself or downloads a model without asking.

```
 your app (Electron)                       Anvira Runtime (one process per OS user)
 +-------------------+   SDK / HTTP        +------------------------------------------------+
 | UI + product logic| ------------------> | API (127.0.0.1:47615, bearer token)            |
 +-------------------+  Authorization:     |  models | chat | jobs | memory | context |     |
                        Bearer <app token> |  resources | apps | leases | capabilities     |
                                           |----------------+-------------------------------|
                                           | supervisor     |  AICL bus (in-process packets)|
                                           |  |  |  |       |    orcha.*  nomi.*  models.*  |
                                           |  v  v  v       |                               |
                                           | ORCHA  Nomi  llama-server (one per loaded model)|
                                           |  (private loopback ports, private tokens)      |
                                           +------------------------------------------------+
```
Apps NEVER talk to ORCHA, Nomi, llama-server or AICL directly and never see their ports or credentials. The runtime holds those.

## A2. Process model and lifecycle (why it is "active only while an app is open")

- The API answers immediately after start; `/health` says `starting` until ORCHA and Nomi are up, then `ok` (or `degraded` if one is down - chat and other parts still work).
- `state/runtime.json` is the discovery file (host, port, pid). SDKs read it; a stale file (dead pid) is ignored.
- **Leases.** When an app connects, the SDK starts the runtime if needed (in *on-demand* mode) and holds a **lease**, renewed by a heartbeat (default ttl 45 s, renewed about every
  15 s). When the last lease is gone, no job/stream/download/model-load is running, and an idle grace (30 s; the CLI uses 120 s) has passed, the runtime stops its children and itself.
  A crashed app's lease simply expires. `anvira runtime start` starts a **persistent** runtime instead (stays until `runtime stop`).
- If the runtime stops underneath an open app, the SDK restarts it, re-acquires the lease and retries. During an install/update a **lock file** (`.updating`) holds every restart
  back until the files are replaced.
- The **supervisor** health-probes ORCHA/Nomi/model servers, restarts crashed ones with backoff, adopts an already-healthy service, reaps stale children, and stops everything in parallel.
- All child processes are launched **hidden** (no console windows). One llama-server runs per loaded model on a private port; switching models stops the old one.

## A3. Security model (assume a hostile web page and a curious sibling app)

- Loopback only (`api.host` can never be non-loopback); requests with a browser `Origin` or non-loopback `Host` are refused.
- Tokens: an **owner token** (the user/CLI, full rights), a **register token** (lets a new app register), and one **app token** per app (returned once at registration, stored hashed).
  Token files live in the runtime's `state/` folder, readable by the same OS user only. (A malicious process running as the same user could read them - documented limit.)
- **Permissions** per app. Default grant: `models.read models.select models.register chat orcha.run orcha.read memory.read memory.write context.read context.write context.share config.read`.
  Needs the user (`anvira app grant <app> <perm>`): `orcha.exec` (agents may RUN COMMANDS), `models.manage` (downloads, provider keys, storage), `memory.shared`, `config.write`, `runtime.admin`, `apps.admin`.
  An app may *request* extra permissions at registration; they are recorded as requested, never auto-granted.
- Secrets (cloud API keys) are write-only through the API and redacted from logs. Local data is never sent to a remote/cloud model unless the request carries `allow_remote_context: true`
  (a decision your UI must put to the user) - otherwise `remote_context_not_allowed` (403).

## A4. Models (llama.cpp) and hardware

- **Discovery, not duplication.** Models can live ANYWHERE. `models.register(path)` (file or folder) records the location without copying; folders left by other Anvira apps
  (`model-storage.json`) are discovered automatically; extra folders via `anvira model dirs add`. `anvira model install <id>` downloads only when the user asks (resumable, disk-checked, needs `models.manage`).
- **Compatibility** is computed from real hardware facts (VRAM/RAM/disk): each model has `compatibility.mode` = `gpu | partial-gpu | cpu | insufficient | remote`.
- **Backend.** The release package bundles `llama-server` for CPU; the optional GPU pack adds the CUDA backend and NVIDIA runtime DLLs (driver 525+). The runtime picks CUDA on NVIDIA machines, otherwise CPU.
  GPU-layer count and context size come from a calibrated heuristic (fills about 82-90 % of VRAM; context 4096 up to 16384 by free VRAM). Flags: `--jinja`, `-np 1`, flash-attention and q8 KV cache on CUDA.
- **Cloud providers** (OpenAI-compatible) can be added by the user (`models.manage`); a provider counts as an installed "remote" model.
- Selecting a model (`models.use(id)`) loads it; chat/ORCHA calls lazily start it if needed and may name a `model` to switch.

## A5. ORCHA - the agent/orchestration engine

ORCHA is a graph-based orchestrator built for **edge models (1.5B-8B)**: it assumes the model is weak and compensates in code. Runs are jobs (`orcha.run`, `task`, `agent.run`) with
states `queued running completed failed cancelled interrupted`; progress streams as SSE frames (`node`, `kind`, `data`) at `/v1/jobs/{id}/events`.

**Graphs:** `default` (an intent gate then chat or a tool-using agent), `research`, `multi_agent`. Unknown graph names are rejected (`invalid_graph`).

**Pipeline for a tool-using run:** intent gate -> (complexity gate) -> agent node -> tools. The **intent gate** classifies the message (CHAT / READ_* / SEARCH / PROJECT_ANALYSIS / FILE_OPERATION /
TOOL_REQUEST) with regex fast-paths before any model call; read intents get a **read-only** tool surface. Hardened here: "look AND change" requests (e.g. "read stats.py, find the bug and fix it") are file operations; negations ("do not change", "change nothing") stay read-only.

**Tools ("capabilities"):** `filesystem` (read/write/edit/create/delete/move/copy/list/tree/search, `apply_patch`), `workspace`, `search`, `web`, `terminal` (`run_command` ...), `git`, `diagnostics`,
`code_intelligence`. Tools are **root-guarded** to the run's `workspace_roots` and policy-checked (access modes: approval / partial / action / full; allow/deny/ask rules).
Through the runtime, `agent.run` defaults to `filesystem + workspace + search`; **`terminal`/`git` are added only for apps the user granted `orcha.exec`** (otherwise `permission_denied` with the exact grant command).

**Edge-model tolerance (this is the point of ORCHA - preserve it, never remove it):**
- *Tool-name aliases* (`find_and_replace`, `save_file`, `modify_file`, `patch` ... resolve to the real tool) and *argument aliases* (`old_str/oldString` -> `old_string`, `patch`/`diff` -> `patch_text`, ...), applied before "no such tool".
- *Text tool-call recovery*: calls emitted as `[calls f(a=1)]`, `<tool_call>{...}</tool_call>`, fenced JSON, XML `<invoke>` are parsed. A strict JSON **envelope** (llama.cpp json_schema) can force a tool call so the model cannot answer in prose.
- *Editing*: `edit_file` uses a strategy ladder (exact -> unicode-normalised -> trailing-whitespace tolerant -> line-number-prefix stripped), preserves CRLF, reports remaining occurrences, hints the closest line on failure; `apply_patch` applies multi-file `*** Begin Patch` envelopes atomically with rollback.
- *Deterministic gates* (no extra model calls): verification gate (code written but never run -> run it), fix gate (command failed -> fix, not narrate), **nothing-changed gate** (task asked for a change but only reads happened), **rename gate** (old name still exists -> `file:line` list, max 3 nudges), forced tool call after every nudge, empty `run_command` gets the project's tests or the file just written, validation errors are never mistaken for failed commands, loop guard for identical repeated calls.
- *Long-horizon planning*: a complexity gate can route big requests to decompose -> plan -> execute -> verify. Upstream keeps small-model decomposition OFF by default (`ORCHA_SMALL_MODEL_DECOMPOSITION=1` enables it) because it over-decomposes on 3B models.
- ***Project scaffold*** (added): a request that **names two or more files** ("Create app.py, templates/index.html, static/style.css and static/game.js ...") is built one file per step: plain-text generation, written through the normal tools, verified mechanically (stub/placeholder/syntax/required features/consistent ids/every button wired/**executed in a fake browser** when Node exists), repaired with tiny targeted requests (missing function only) or SEARCH/REPLACE patches, best attempt kept, honest report `Built N of N files` / `Partly built ... PROBLEMS`.
- Verified with real models (RTX 3050 4 GB): 7B coder finished every workflow; 3B built the Snake project cleanly in 2 of 3 runs and did multi-file rename, but not every bug-fix / create-from-spec; 1.5B is only reliable for Notes/Study grounding. Recommend 7B+ for agent work.

A finished job's `result`: `{run_id, status, answer, confidence, contributors, iterations, latency_s, graph, agent_steps[], agent_tool_calls[], agent_completed}`. Show `agent_tool_calls` - never trust the prose alone.

## A6. Nomi - memory

Nomi is a separate memory service (its own database) that the runtime supervises and namespaces. **Per-app private by default**: memories are tagged with reserved tags
(`app:<id>`, `ws:<workspace>`, `scope:shared`) so one app cannot read another's. `scope:'shared'` needs `memory.shared`. Search is **word-based BM25** (a port of Anvira's `bm25.ts`, parity-tested)
- it matches shared words, not meaning, so store keyword-rich text and phrase recall queries with the same words. `store`, `search`, `list`, `get`, `delete` (only the author or the owner can delete a shared one).
In chat, `memory: {recall: true}` injects the top matches into the prompt and returns `memories_used` ids.

## A7. Context and shared resources (the data layer between apps)

- **Context index**: per-app documents chunked and BM25-ranked (`context.put/search`), lazily opened (nothing loads until used).
- **Resources** are small records that POINT at content (`context` collection, `file`, or short `text`) - references (`runtime://res_...`), never copies. **Private by default**; the owner (or the user)
  shares with named apps as `read` or `write`; **global (every app, read-only) is the user's decision only**. Apps can `requestAccess`; the owner/user approves or denies. Everything (create, share, revoke, read, search, write, request, approve) is audited. An app with no access gets `resource_not_found` - it cannot even learn a private resource exists.
- Chat/ORCHA accept `context: {query?, resources?, limit?, strict?}`: authorised snippets are retrieved on demand (BM25 over the pooled chunks) and injected; `strict:true` tells the model when nothing relevant exists (verified to stop small models inventing answers). The response reports `context_used`.
- "Connected" never means "loaded": the runtime reads data only when asked.

## A8. AICL - the internal bus

AICL is a compact binary protocol used INSIDE the runtime between its modules (`orcha.*`, `nomi.*`, `models.*`, ...). Every module call is an `aicl.bin` **packet** with a CRC32 trailer and an opcode
(`OP_CALL`, `OP_MEMORY_*`, `OP_INDEX_*`, `OP_EXECUTE`, `OP_CANCEL`, `OP_RETURN`, `OP_ERROR`); the bus is in-process (the native Rust core is not built; the Python codec is used) and keeps per-module stats and a
recent-packet trace (`anvira aicl status|trace`). Apps never use AICL; it is observable (dashboard Activity pane) and is the reason failures are structured errors rather than exceptions.

## A9. Errors, jobs and streaming contracts

- Every failure: `{"error": {"code","message","status","hint"?,"details"?}}`. **Branch on `code`, show `message` + `hint`.** Common: `runtime_not_installed`, `install_declined`, `permission_denied`, `no_model`
  (`details.installed` says whether any model exists), `model_loading`, `model_crashed`, `resource_not_found`, `access_denied`, `user_approval_required`, `remote_context_not_allowed`, `invalid_graph`, `checksum_mismatch`, `incompatible_runtime`.
- Chat is OpenAI-shaped plus `runtime {model, kind, memories_used, context_used, warnings}`; `stream:true` sends SSE (`event: runtime` first, failures as `event: error`).
- Jobs are private to the app that created them; ORCHA runs and model installs share one lifecycle; `cancel` works on running jobs.

## A10. CLI and dashboard (what the user can do without your app)

`anvira open` (live dashboard: Overview / Health / Models / Apps & data / Logs; Tab or 1-5), `status`, `doctor` (with a fix for every problem), `hardware`, `runtime start|stop|restart|logs|info|locate|register|install|install-gpu|update`,
`model list|search|info|compat|recommend|install|use|unuse|remove|dir|dirs|add|discover|provider`, `orcha run|jobs|cancel|status`, `job get`, `nomi status|search|inspect|store|delete`, `chat`, `context ...` (resources, share, requests, approve, audit),
`workspace`, `capability list`, `config`, `app list|show|grant|deny|revoke`, `aicl status|trace`, `path add|remove`. Every command accepts `--json`. Exit codes: 0 ok, 1 fail, 2 usage, 3 not installed, 4 not running, 5 permission, 6 not found/no model, 8 doctor failures.

## A11. Where things live and how a user gets the runtime

One folder (any drive): `python/` (own interpreter), `services/{orcha,nomi,aicl}/`, `bin/llama-cpp/{cpu,cuda}/`, `state/` (tokens, discovery), `config/`, `data/` (Nomi DB, context/resource DBs), `logs/`, `models/`. Default `%LOCALAPPDATA%\AnviraRuntime`;
a custom location is remembered by a pointer file so every app finds it. Release assets on GitHub: a ~45 MB core zip and an optional ~650 MB NVIDIA GPU pack, checksummed (`SHA256SUMS`). Source layout: `runtime/` (daemon, API, CLI, TUI, model layer, supervisor), `sdk/{python,typescript}`, `Orcha/`, `nomi/`, `AICL/`, `docs/`, `scripts/`, `tests/`, `examples/`.

# PART B - HOW YOUR APP MUST USE IT

## B1. First launch: find the runtime, or ask the user - never install silently

Call `AnviraRuntime.connect({ appId, name, install, onStatus })`. If the runtime is missing, the SDK calls your `install(info)`
callback. **You must show the user a dialog** (do not auto-answer it):

> **Anvira Runtime is needed** - it runs your local AI models and is shared by all Anvira apps.
> ( ) **I already have it** - [Browse for the folder that contains `install.json`]
> ( ) **Download it** from GitHub - install into: [ default folder ] [Choose folder...]   (about 45 MB)
> [ ] Enable NVIDIA GPU acceleration (extra ~650 MB) - shown only if `nvidiaGpu()` finds a card
> [Cancel]

and resolve the callback with the user's choice:

```ts
install: async (info) => {
  const choice = await showRuntimeDialog(info)           // YOUR UI. Never return true without the user having chosen.
  if (choice.cancel) return false                        // -> connect() throws InstallDeclined; show "Anvira Runtime is required"
  if (choice.haveIt) return { action: 'use-existing', path: choice.folder }      // registers it; nothing is downloaded
  return { action: 'download', dest: choice.folder,      // any folder / any drive; remembered for every Anvira app
           gpu: choice.gpu, onProgress: (name, done, total) => updateProgressBar(done / total) }
}
```

Before showing the dialog you may call `locate()` (`{ found, home, source, version, running }`) and `nvidiaGpu()` (`{ name, driver, cudaOk }`).
If `locate().found` is true, skip the dialog. If the user picks a folder with `registerLocation(folder)` it throws
`not_a_runtime_folder` when it is not a runtime - show that message and let them pick again.

The download comes from the GitHub release of the runtime repository (checksummed, resumable, never a model). Set the repo once:
`ANVIRA_RUNTIME_REPO=<owner>/Anvira-Runtime` or edit `DEFAULT_REPO`. After the first install every other Anvira app finds the
same runtime automatically (a pointer file records the custom location), so **the second app must not ask again**.

## B2. Lifecycle: the runtime is on only while an app is open

`connect()` starts the runtime on demand and holds a heartbeat lease. Call `await runtime.close()` when the app quits (also on
crash-safe paths: `before-quit`). Do not call `runtime stop`. Long work (agent runs, downloads, streams) keeps it alive by itself.
If the runtime is updated while your app is open the SDK waits and restarts it; you do not handle that.

## B3. Models: reuse, never download silently

- List: `runtime.models.installed()`. Show `compatibility.mode` (`gpu`, `partial-gpu`, `cpu`, `insufficient`) as a badge.
- The user already has `.gguf` files somewhere? `runtime.models.register(pathToFileOrFolder)` - no copy, every app can then use them.
  `runtime.models.discovered()` lists model folders left by other Anvira apps; offer them.
- Select: `runtime.models.use(id)`. Downloads need the `models.manage` permission the user grants
  (`anvira app grant <appId> models.manage`); never start a multi-GB download without an explicit click showing the size.
- No model selected -> the error code is `no_model` (its `details.installed` says whether any exist). Show the model picker.

## B4. Chat, streaming, memory, context

```ts
for await (const delta of runtime.chat(messages, { stream: true, memory: { recall: true }, context: { resources: [rid], strict: true } })) ...
```
- `memory.recall` pulls this app's own memories (word-based BM25, so keep memory text keyword-rich).
- `context.resources` injects authorised snippets from shared resources (section 5) **on demand**. `strict: true` tells the model
  "nothing relevant was found" instead of letting a small model improvise - use it for anything grounded in the user's notes.
- The response's `runtime.context_used` / `memories_used` tell you what was used: show them as citations.
- Local data is **never** sent to a cloud model silently: with a cloud provider active, `context`/`memory` fail with
  `remote_context_not_allowed` unless you pass `allow_remote_context: true` - only after an explicit user prompt.

## B4b. Agent work (Anvira - the agent/IDE app, and Anvira Dev)

`runtime.agentRun(task, { workspaceRoots: [dir] })` gives the agent file and search tools **inside those folders only**. Running commands
(tests, builds) needs the user's grant: `anvira app grant <appId> orcha.exec`; without it the request is refused with
`permission_denied` and a hint - explain that to the user and let them approve (a one-time, per-app choice). Watch progress with
`job.events()`, cancel with `job.cancel()`. The result is `job.result.answer` (plain text) plus `agent_steps` / `agent_tool_calls`
(what it actually did): show those, do not just trust the prose.

How to get the best out of small local models (1.5B-8B) - the runtime already compensates for their weaknesses, so ask in the shape it handles:

- **Name the files.** "Create app.py, templates/index.html, static/style.css and static/game.js ... Flask ..." is built **file by file**, each file verified
  (syntax, required features, ids consistent across files, and - with Node installed - executed in a fake browser). The answer says
  `Built 4 of 4 files` or `Partly built ...` with the unresolved problems named. Surface that honestly; offer "retry" or "use a bigger model".
- **One concrete change per run** ("rename X to Y everywhere", "make the failing test pass"). The runtime refuses to accept "done" when nothing changed or a rename is unfinished.
- Small models fail differently from big ones: on this project's real tests a 7B finished every task, a 3B built the Snake project cleanly in 2 of 3 runs and
  did multi-file renames but not every bug fix or create-from-spec task. Let the user pick the model and show `compatibility.mode`; recommend 7B+ for agent work, 3B is fine for Notes/Study grounding.
- Never claim success from the answer text alone; check the files or the run result.

Anvira (the IDE-style app) should own: the workspace picker, the file tree/diff view, the approval UI for `orcha.exec`, and the job event panel. The runtime owns the
model, the agent loop and the safety gates.

## B5. Sharing data between apps (Notes <-> Study <-> Anvira)

- Keep the document store in your app. Index text with `runtime.context.put(collection, docId, text)`, then register a reference:
  `const res = await runtime.resources.create('Biology 101', { type: 'notebook', collection })`. It is **private by default**.
- Another app asks: `resources.requestAccess(id, { reason })`. The owner app (or the user) decides: `resources.requests()` then
  `resources.decide(requestId, true)`. Show pending requests in your UI. Sharing is `resources.share(id, ['anvira-study'], { access: 'read' })`; revoke with
  `resources.revoke(id)`. **Only the user can make something global** (`anvira context share <id> --global`); never try.
- Read what you were given with `resources.search(query)`, `resources.read(id, docId)`, or `context: { resources: [...] }` in chat.
  References look like `runtime://res_...`; store the reference, not a copy.
- Connecting to the runtime loads nothing. Fetch data only when the user asks for it.

## B6. Errors: branch on `error.code`, show `error.message` and `error.hint`

`runtime_not_installed`/`install_declined` (dialog), `permission_denied` (tell the user the exact `anvira app grant` command from the
hint), `no_model`, `model_loading`/`model_crashed` (retry), `resource_not_found` (it does not exist **or** was not shared with you),
`remote_context_not_allowed`, `checksum_mismatch`/`download_failed` (offer retry), `incompatible_runtime` (update prompt).

## B7. What NOT to do

No `child_process` for llama-server/ORCHA/Nomi; no direct HTTP to their ports; no reading the runtime's token files yourself;
no bundling models; no silent installs/downloads; no copying another app's data; no storing API keys for cloud providers in the
app (the user adds them once via the runtime: `anvira model provider add`); no "connected = load everything".

## B8. Verify before you say it works

Run the app against a real runtime with a real small model (the repo's `examples/real_workflows.py` shows the calls). Check: first
launch with no runtime shows the dialog; "I already have it" and "Download" both work; the second app does not ask; closing the
last app lets the runtime stop (`anvira status` shows the countdown); a private resource is invisible to another app until shared.

<APP>
Name / appId:            (e.g. Anvira Notes / `anvira-notes`)
What it owns:            (e.g. notebooks, pages, sources, citations)
What it needs from the runtime: (e.g. grounded chat over notebooks, flashcard generation, sharing notebooks with Study)
Extra permissions to request: (e.g. `models.manage`, `orcha.exec`)
</APP>

---

### Per-product notes to append

- **Anvira** - the agent/ORCHA-focused app: chat + `agentRun` with workspace roots, ORCHA event panel (`job.events()`), model picker, provider (cloud) keys (`models.manage`), memory recall. Keep workspaces, agents marketplace, attachments in the app.
- **Anvira Notes** - notebooks stay in the app; index every source with `context.put`, expose the notebook as a `resources.create(..., { collection })`, ground every answer with `context: { resources, strict: true }` and show `context_used` as citations. Offer "Share with Study" -> `resources.share`.
- **Anvira Study** - flashcards/FSRS/plans stay in the app; get material via `requestAccess` on a Notes resource (or a shared one), generate cards with `chat` + `context`, ask for JSON only and validate it. Use `memory` (workspace = course) for study state.
