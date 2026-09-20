# MASTER PROMPT - make Anvira, Anvira Notes and Anvira Study use Anvira Runtime

Paste everything between the two lines into your coding assistant when it starts work on an Anvira application (the same prompt for all three; only the
`<APP>` block at the end changes). Real behaviour it describes was verified against the packaged runtime; see `docs/REAL_WORLD_RESULTS.md`. It is written so the assistant cannot "helpfully" reimplement what the runtime already does.

---

You are building **<APP>**, a desktop application that is one of several Anvira products (Anvira, Anvira Notes, Anvira Study,
Anvira Dev). All of them share ONE local service, **Anvira Runtime**. The runtime already provides the AI plumbing: model
management (llama.cpp, CPU + NVIDIA GPU), hardware detection, chat, the ORCHA agent/orchestration engine, Nomi memory, a
document/context index and cross-app sharing. **Do not rebuild any of it, do not spawn llama-server/ORCHA/Nomi yourself, and do
not talk to their ports.** Use only the runtime's SDK (`@anvira/runtime-client`, TypeScript, in `sdk/typescript`). The application
keeps only what is specific to the product (its UI, its documents, its own logic).

## 1. First launch: find the runtime, or ask the user - never install silently

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

## 2. Lifecycle: the runtime is on only while an app is open

`connect()` starts the runtime on demand and holds a heartbeat lease. Call `await runtime.close()` when the app quits (also on
crash-safe paths: `before-quit`). Do not call `runtime stop`. Long work (agent runs, downloads, streams) keeps it alive by itself.
If the runtime is updated while your app is open the SDK waits and restarts it; you do not handle that.

## 3. Models: reuse, never download silently

- List: `runtime.models.installed()`. Show `compatibility.mode` (`gpu`, `partial-gpu`, `cpu`, `insufficient`) as a badge.
- The user already has `.gguf` files somewhere? `runtime.models.register(pathToFileOrFolder)` - no copy, every app can then use them.
  `runtime.models.discovered()` lists model folders left by other Anvira apps; offer them.
- Select: `runtime.models.use(id)`. Downloads need the `models.manage` permission the user grants
  (`anvira app grant <appId> models.manage`); never start a multi-GB download without an explicit click showing the size.
- No model selected -> the error code is `no_model` (its `details.installed` says whether any exist). Show the model picker.

## 4. Chat, streaming, memory, context

```ts
for await (const delta of runtime.chat(messages, { stream: true, memory: { recall: true }, context: { resources: [rid], strict: true } })) ...
```
- `memory.recall` pulls this app's own memories (word-based BM25, so keep memory text keyword-rich).
- `context.resources` injects authorised snippets from shared resources (section 5) **on demand**. `strict: true` tells the model
  "nothing relevant was found" instead of letting a small model improvise - use it for anything grounded in the user's notes.
- The response's `runtime.context_used` / `memories_used` tell you what was used: show them as citations.
- Local data is **never** sent to a cloud model silently: with a cloud provider active, `context`/`memory` fail with
  `remote_context_not_allowed` unless you pass `allow_remote_context: true` - only after an explicit user prompt.

## 4b. Agent work (Anvira - the agent/IDE app, and Anvira Dev)

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

## 5. Sharing data between apps (Notes <-> Study <-> Anvira)

- Keep the document store in your app. Index text with `runtime.context.put(collection, docId, text)`, then register a reference:
  `const res = await runtime.resources.create('Biology 101', { type: 'notebook', collection })`. It is **private by default**.
- Another app asks: `resources.requestAccess(id, { reason })`. The owner app (or the user) decides: `resources.requests()` then
  `resources.decide(requestId, true)`. Show pending requests in your UI. Sharing is `resources.share(id, ['anvira-study'], { access: 'read' })`; revoke with
  `resources.revoke(id)`. **Only the user can make something global** (`anvira context share <id> --global`); never try.
- Read what you were given with `resources.search(query)`, `resources.read(id, docId)`, or `context: { resources: [...] }` in chat.
  References look like `runtime://res_...`; store the reference, not a copy.
- Connecting to the runtime loads nothing. Fetch data only when the user asks for it.

## 6. Errors: branch on `error.code`, show `error.message` and `error.hint`

`runtime_not_installed`/`install_declined` (dialog), `permission_denied` (tell the user the exact `anvira app grant` command from the
hint), `no_model`, `model_loading`/`model_crashed` (retry), `resource_not_found` (it does not exist **or** was not shared with you),
`remote_context_not_allowed`, `checksum_mismatch`/`download_failed` (offer retry), `incompatible_runtime` (update prompt).

## 7. What NOT to do

No `child_process` for llama-server/ORCHA/Nomi; no direct HTTP to their ports; no reading the runtime's token files yourself;
no bundling models; no silent installs/downloads; no copying another app's data; no storing API keys for cloud providers in the
app (the user adds them once via the runtime: `anvira model provider add`); no "connected = load everything".

## 8. Verify before you say it works

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
