# `src` → Anvira Runtime migration map

Result of inspecting the existing Anvira `src` / `electron` tree. **No `src` file was moved or rewritten**: that tree has no
`package.json`, `node_modules` or test runner here, so a refactor could not be verified. The runtime, SDKs and this map are the
deliverable; the frontend cutover is a follow-up (see "What is not done").

## Where user data lives today (Electron `userData` = `%APPDATA%\Anvira`)

| Data | Location | Coupled to |
|---|---|---|
| Notebooks, pages, sources, study sets/plans, citations, progress | `anvira.db` (SQLite + sqlite-vec, encrypted); key in `db.key.enc` via Electron `safeStorage`; JSON fallback in `notes/` | Electron main |
| Model files | `models/`, relocatable through `model-storage.json` | Electron main |
| Model registry | renderer `localStorage` (`anvira_installed_models`) | Renderer |
| Cloud (BYOK) keys | `byok-connections.json` (`safeStorage`-encrypted) | Electron main |
| Chat sessions, projects, index, settings, agents | renderer `localStorage` (`anvira-settings`, `anvira-agents`, workspace store) | Renderer |
| Nomi DB | `nomi-data/`, plus a Nomi token in `localStorage` | Electron main + renderer |

## Mapping

| Current | What it does | Action | Target |
|---|---|---|---|
| `electron/llamaServer.cjs`, `hardwareDetection.cjs`, `modelStorage.cjs`, `main.cjs` download/inference handlers | llama-server lifecycle, GPU-layer heuristics, hardware detection, downloads, models dir | **Move** (logic ported) | Runtime `models/`, `hardware/` — done |
| `services/huggingface.ts`, `services/modelManager.ts`, `utils/modelSize.ts`, `pages/ModelLibrary`, `InstalledModels`, `Downloads` | HF search, installed-model registry, size tiers, UIs | **Extract logic, keep UI thin** | Runtime model catalog/registry; the pages become a picker over the SDK |
| `electron/byokStorage.cjs`, `components/models/CloudConnections.tsx` | Cloud provider keys | **Move storage, keep UI** | Runtime provider store. `safeStorage` keys cannot be decrypted outside Electron → **re-enter once** |
| `electron/orchaProcess.cjs`, `nomiProcess.cjs`, `runtimeTracker.cjs`, `runtimeStartupLog.cjs` | spawn, pid tracking, health probes | **Move** | Runtime supervisor (reuse-if-alive, stale-pid cleanup) — done |
| `services/orcha.ts`, `services/nomi.ts`, `OrchaActivityPanel`, `AgentEventStream`, `useAgentEventStream` | Direct ORCHA/Nomi clients + event UIs | **Refactor** | SDK. Job SSE passes ORCHA's event frames through unchanged, so the activity UIs keep working |
| `pages/Chat.tsx`, `Agents`, `AgentActivity`, `AgentDiagnostics`, `SessionSidebar`, `workspaceStore`, `agentStore`, `data/agents.ts`, `utils/prompt.ts`, `fileOps`, `diff`, `attachmentGate`, `workspaceContext` | Workspace, agents, chat, attachments | **Keep in Anvira** (workspace/agent logic reusable in Dev) | Anvira; Anvira Dev takes `fileOps`, `diff`, workspace index, `AgentActivity`, diagnostics |
| `electron/workspaceIndex.cjs`, `fileExtraction.cjs`, `utils/bm25.ts`, `notesContext.ts` | Project indexing, PDF/Office/OCR extraction, BM25 | **Extract** BM25 + chunking | Runtime `context` service (BM25 ported, parity-tested against Node). Node-only extraction stays in a shared app-side package |
| `pages/Notes.tsx`, `components/notes/*`, `notesStore`, `notesStorage*.cjs`, `db.cjs`, `markdownToTiptap`, `pageLinks/Tags/Templates/Sizes`, `transcriptToNotes`, `citations`, `mindMapGeneration` | Editor, notebook persistence | **Reuse** | Anvira Notes. Encrypted DB layer → shared Node package (not a runtime concern) |
| `pages/Study.tsx`, `components/study/*`, `spacedRepetition` (ts-fsrs), `studyGeneration`, `studyPlan`, `gradeCalc/Extraction`, `textMatch`, `ankiExport`, `jsonExtraction`, `workloadEstimate`, `achievements` | Flashcards, quizzes, FSRS, plans | **Reuse** | Anvira Study (`jsonExtraction` shared) |
| `services/auth.ts`, `supabase.ts`, `Login/Signup/Forgot/Profile`, `plans.ts`, `usageStore`, `PlansComparison`, `UpgradeModal` | Accounts + plan gating | **Keep per app / shared auth package** | Not runtime (product/billing) |
| `pochi*`, `Layout`, `Markdown*`, `CodeBlock`, `index.css` | Mascot, shell, UI kit | **Keep / shared UI** | Anvira + a shared UI package |
| `settingsStore` (`inferenceProfile`, `contextWindow`, `gpuAcceleration`, `modelStorageDir`) | Model knobs mixed with UI prefs | **Refactor** | Model knobs → runtime config; UI prefs stay |
| `agentMarketplace.ts`, `McpServersSection`, `SkillsCard` | Agents, MCP servers, skills (ORCHA-backed) | **Wrap** | Runtime `orcha` passthrough |

## Decisions this drove

1. The runtime models directory **adopts** `%APPDATA%\Anvira\models` (and the other Anvira apps' folders) via `model-storage.json` discovery — no duplicate downloads.
2. A small shared **context/resource layer** exists in the runtime; Notes-/Study-specific logic stays out.
3. Old AICL leftovers were quarantined to `AICL/_legacy_v0/`; the AICL suite passes in place (93/93).

## What is not done

* The `src/` cutover (replacing `services/orcha.ts`, `services/nomi.ts`, model pages with the SDK) — needs the app's own build and test setup.
* The Notes/Study/Dev feature code in `src/` is untouched and stays in the apps; the runtime never contains it.
* Existing localStorage model registries and BYOK keys are not migrated automatically (keys cannot be decrypted outside Electron).
