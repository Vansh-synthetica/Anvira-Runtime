# Integrating an application

An app needs **one dependency** (the SDK) and never sees ORCHA, Nomi, AICL, ports or model processes.

| Question | Answer |
|---|---|
| How do I detect the runtime? | `AnviraRuntime.detect()` → `{installed, running, ready, runtimeVersion, apiVersion}`. Never throws, never installs. |
| How do I connect? | `AnviraRuntime.connect({appId, install})`. Detects → (asks you to install) → starts if stopped → registers the app once → checks compatibility. |
| How do I check health? | `runtime.health()` / `runtime.status()` |
| How do I list models? | `runtime.models.installed()` / `.catalog()` / `.recommended()` / `.hardware()` |
| How do I run a task? | `runtime.task("…")`, `runtime.chat([...])` (streaming: `{stream: true}`) |
| How do I use ORCHA? | `runtime.orcha.run("…", {graph, wait})` → `Job`; `runtime.agentRun("…", {workspaceRoots})` |
| How do I use memory? | `runtime.memory.store(...)` / `.search(...)`; chat can recall: `chat(msgs, {memory: {recall: true}})` |
| Runtime not installed? | `connect()` throws `RuntimeNotInstalled` unless you pass `install: async (info) => userAgreed`. Nothing installs silently. |

## TypeScript (Electron main process / Node ≥ 18)

Use it in the **main process**, not the renderer: the runtime rejects browser `Origin`s and the app token must stay out of web
content — expose your own IPC methods to the renderer.

```ts
import { AnviraRuntime, RuntimeNotInstalled, InstallDeclined } from '@anvira/runtime-client'

const runtime = await AnviraRuntime.connect({
  appId: 'anvira-notes', name: 'Anvira Notes',
  install: async () => await dialogAsk('Anvira Runtime is required. Install it now?'),   // true only if the user clicked [Install]
  source: bundlePath,                       // the runtime bundle shipped/downloaded with the app (used only if they agree)
  onStatus: (m) => splash.text(m),          // "Installing Anvira Runtime..." / "Starting runtime..."
})

const models = await runtime.models.installed()           // shared with every other Anvira app
await runtime.models.use(models[0].id)                    // or let the user pick from runtime.models.recommended()

const hits = await runtime.context.search('biology', 'how is carbon fixed?')          // grounding
const reply = await runtime.chatText([{ role: 'system', content: hits.map(h => h.text).join('\n') },
                                      { role: 'user', content: 'How is carbon fixed?' }])
for await (const delta of runtime.chat(msgs, { stream: true })) ui.append(delta)

const job = await runtime.orcha.run('Summarise my notes', { wait: true })
console.log((await job.unwrap()).answer)
await runtime.memory.store('Prefers concise answers'); const m = await runtime.memory.search('concise')
```

Python (stdlib only): `from anvira_client import AnviraRuntime` — identical shape (`runtime.models.use(...)`, `runtime.chat_text(...)`,
`runtime.orcha.run(..., wait=True)`, `runtime.memory.search(...)`).

## Errors

`AnviraError{code, message, status, hint}` with subclasses `RuntimeNotInstalled`, `RuntimeNotRunning`, `RuntimeStartFailed`,
`IncompatibleRuntime`, `InstallDeclined`, `PermissionDenied`. Common `code`s: `no_model` (offer the model picker), `permission_denied`
(hint names the grant command), `model_crashed`/`model_loading` (retry), `nomi_unavailable` (memory is down; chat/ORCHA still work). A crashed
model yields a structured error within seconds, not a hang.

## Permissions

Registration grants the safe defaults (models read/select/register, chat, ORCHA, own-namespace memory/context, config read).
Anything heavier is *requested* (`connect({permissions: ['models.manage']})`) and the user approves with `anvira app grant <app> <perm>`.
`models.install()` therefore fails with `permission_denied` until they do — a app cannot start a multi-GB download by itself.

## Per product

* **Anvira** — chat/agents/workspaces: `chat` + `orcha.run/agentRun` (`workspaceRoots`, `reasoning`) + `jobs.events()` for the activity panel
  (ORCHA event frames unchanged) + `memory` with `recall`. Replace `services/orcha.ts` and `services/nomi.ts` with the SDK; the Model
  Library/Installed Models/Downloads pages become a thin picker over `runtime.models.*`; BYOK connections → `models.addProvider`
  (needs `models.manage`, which the user grants: `anvira app grant anvira models.manage`).
* **Anvira Notes** — keep the notebook store and editor in the app. Index each source with `context.put(notebookId, sourceId, text)` and
  ground answers with `context.search` (replaces the in-renderer `buildNotebookContext`/`bm25.ts`); generation via `chat`/`task`.
* **Anvira Study** — keep flashcards/FSRS/plans in the app. Grounding via `context`, generation via `chat`, study-state hints via `memory` with a `workspace` per course.
* **Anvira Dev** — `agentRun` with `workspaceRoots`, `jobs.events()`, `orcha.cancel`, `models.use` for a coding model; keep workspace indexing/diff/file-ops in the app.

## Sharing models across apps

Downloaded a model in Anvira? Notes, Study and Dev see it already (the runtime reads each app's `model-storage.json`/`models/`).
Downloaded it somewhere else, or with another tool? Call `runtime.models.register(pathToFileOrFolder)` — the runtime remembers where it is,
never copies it, and any request can use it (`chat(msgs, {model: id})`, `orcha.run(task, {model: id})`).

## Open and close: the runtime lives only while an app does

`connect()` starts the runtime if needed (on-demand mode) and holds a lease, renewing it in the background; call `close()` when the
app quits. Nothing else is required: a crashed app's lease expires on its own, running work keeps the runtime alive, and if the
runtime stops underneath a live app the SDK restarts it and retries. See [LIFECYCLE.md](LIFECYCLE.md).

## Sharing data between apps

Notes indexes a notebook (`context.put`), registers it (`resources.create(title, {collection})`) and — when the user agrees —
`resources.share(id, ['anvira-study'])`. Study then `resources.search(...)`es or passes `context: {query}` to `chat`/`orcha.run`.
Private is the default, global is the user's decision, and nothing is copied. See [SHARED_CONTEXT.md](SHARED_CONTEXT.md).
Connecting to the runtime does not load any of it.

## Testing your integration

Point `ANVIRA_RUNTIME_HOME` at a temp folder, start a runtime, and register a fake OpenAI-compatible server with
`models.addProvider` — see `tests/runtime/conftest.py` and `test_05_app_integration.py` for the pattern.
