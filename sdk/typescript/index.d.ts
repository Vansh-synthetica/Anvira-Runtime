// Type definitions for @anvira/runtime-client

export const SDK_API_VERSION: 1

export class AnviraError extends Error {
  code: string
  status?: number
  hint?: string
  details: Record<string, unknown>
  constructor(code: string, message: string, opts?: { status?: number; hint?: string; details?: Record<string, unknown> })
  static fromResponse(status: number, body: unknown): AnviraError
}
export class RuntimeNotInstalled extends AnviraError {}
export class RuntimeNotRunning extends AnviraError {}
export class RuntimeStartFailed extends AnviraError {}
export class IncompatibleRuntime extends AnviraError {}
export class InstallDeclined extends AnviraError {}
export class PermissionDenied extends AnviraError {}

export interface RuntimeDirs { home: string; state: string; config: string; logs: string; models: string; bin: string }
export function runtimeDirs(env?: Record<string, string | undefined>, platform?: string, opts?: { ignorePointer?: boolean }): RuntimeDirs

export interface RuntimeInfo {
  installed: boolean
  running: boolean
  ready: boolean
  home: string
  install: Record<string, unknown>
  host: string
  port: number | null
  pid: number | null
  runtimeVersion: string | null
  apiVersion: number | null
  capabilities: string[]
  staleDiscovery: boolean
  error: string | null
}
export function detect(env?: Record<string, string | undefined>): Promise<RuntimeInfo>
export function startRuntime(opts?: { env?: Record<string, string | undefined>; timeoutMs?: number; onStatus?: (m: string) => void }): Promise<RuntimeInfo>
export function installRuntime(opts: { source: string; env?: Record<string, string | undefined>; force?: boolean; onStatus?: (m: string) => void }): Promise<Record<string, unknown>>

export type JobState = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted'

export interface OrchaResult {
  run_id: string
  status: string
  answer: string
  confidence: number | null
  synthesized: boolean | null
  contributors: string[]
  iterations: number | null
  latency_s: number | null
  graph: string | null
  agent_steps: unknown[]
  agent_tool_calls: unknown[]
  agent_completed: boolean | null
}

export class Job<R = unknown> {
  readonly id: string
  readonly kind: string
  readonly state: JobState
  readonly result: R | null
  readonly error: { code: string; message: string; hint?: string; details?: unknown } | null
  readonly progress: Record<string, unknown>
  readonly done: boolean
  data: Record<string, unknown>
  refresh(): Promise<this>
  wait(opts?: { timeoutMs?: number; pollMs?: number }): Promise<this>
  cancel(): Promise<this>
  /** Result of a completed job; throws AnviraError if it failed or was cancelled. */
  unwrap(): Promise<R>
  events(): AsyncGenerator<Record<string, unknown>>
}

export interface ModelRecord {
  id: string
  name: string
  kind: 'local' | 'provider'
  installed: boolean
  active?: boolean
  size_bytes?: number
  path?: string
  compatibility?: { can_run: boolean | null; mode: 'gpu' | 'partial-gpu' | 'cpu' | 'insufficient' | 'remote' | 'unknown'; needs_mib: number | null; reasons: string[] }
  [k: string]: unknown
}

export interface Resource {
  id: string
  ref: string
  owner: string
  type: string
  kind: 'context' | 'text' | 'file'
  title: string
  workspace: string | null
  metadata: Record<string, unknown>
  visibility: 'private' | 'shared' | 'global'
  /** This caller's access: read, write or admin (owner). */
  access: 'read' | 'write' | 'admin'
  /** Only for the owner: where the content lives and who else has access. */
  content?: Record<string, unknown>
  grants?: { app: string; access: string }[]
}

export interface MemoryItem {
  id: string
  title: string
  content: string
  type: string
  tags: string[]
  scope: 'app' | 'shared'
  app: string | null
  workspace: string | null
  score?: number | null
  [k: string]: unknown
}

export interface ChatMessage { role: 'system' | 'user' | 'assistant' | 'tool'; content: string }
export interface ChatOptions {
  stream?: boolean
  temperature?: number
  max_tokens?: number
  /** Recall relevant memories from this app's namespace into the prompt. */
  memory?: { recall: boolean; limit?: number; scope?: 'app' | 'shared' | 'all'; workspace?: string }
  /** Authorised shared resources, resolved on demand. `strict`: if nothing relevant is found, tell the model so instead of letting it improvise. */
  context?: { query?: string; resources?: string[]; limit?: number; strict?: boolean }
  /** The user's decision to let local context/memory go to a REMOTE (cloud) model. Never set this silently. */
  allow_remote_context?: boolean
  [k: string]: unknown
}

export interface ConnectOptions {
  appId: string
  name?: string
  /** Extra permissions to request; the user must grant them (`anvira app grant`). */
  permissions?: string[]
  requireApi?: number
  minVersion?: string
  autoStart?: boolean
  /**
   * Called when the runtime is missing. Ask the USER, then resolve: false (declined), true (install with defaults), or their choice:
   * `{ action: 'use-existing', path }` ("I already have it" - nothing is downloaded) or `{ action: 'download', dest, gpu }` (any folder).
   */
  install?: (info: RuntimeInfo) => Promise<InstallChoice> | InstallChoice
  /** Runtime bundle (.zip or source directory) used if the user agrees to install. */
  source?: string
  env?: Record<string, string | undefined>
  timeoutMs?: number
  onStatus?: (m: string) => void
  /** Hold a heartbeat lease while this handle is open (default true). The runtime is on-demand: it stops by itself after the last app closes. */
  keepAlive?: boolean
  /** Seconds the runtime waits after the last app closed before stopping (default 30). */
  idleGraceS?: number
}

export interface LifecycleStatus {
  mode: 'on-demand' | 'persistent'
  auto_stop: boolean
  idle_grace_s: number
  leases: { id: string; app: string; ttl_s: number; age_s: number; expires_in_s: number }[]
  busy: number
  idle_for_s: number | null
  shutdown_in_s: number | null
}

export class AnviraRuntime {
  /** Release this app's lease (call when the app closes). */
  close(): Promise<void>
  lifecycle(): Promise<LifecycleStatus>
  readonly info: RuntimeInfo
  readonly appId: string
  static detect(env?: Record<string, string | undefined>): Promise<RuntimeInfo>
  static connect(opts: ConnectOptions): Promise<AnviraRuntime>

  models: {
    list(o?: { installed?: boolean }): Promise<ModelRecord[]>
    installed(): Promise<ModelRecord[]>
    catalog(q?: string): Promise<ModelRecord[]>
    search(q: string, limit?: number): Promise<ModelRecord[]>
    recommended(limit?: number): Promise<ModelRecord[]>
    active(): Promise<Record<string, unknown>>
    get(id: string): Promise<ModelRecord>
    compatibility(id: string): Promise<Record<string, unknown>>
    use(id: string, o?: { waitS?: number }): Promise<Record<string, unknown>>
    /** Needs the `models.manage` permission (user-granted). */
    install(model?: string, o?: { url?: string; file?: string; dir?: string; wait?: boolean }): Promise<Job>
    remove(id: string, o?: { deleteFile?: boolean; confirm?: boolean }): Promise<Record<string, unknown>>
    /** Tell the runtime where a model file or models folder already lives (no copy, no download). */
    register(path: string): Promise<Record<string, unknown>>
    /** Model folders the runtime found from installed Anvira apps. */
    discovered(): Promise<{ enabled: boolean; locations: { app: string; dir: string; models: string[] }[] }>
    link(path: string, id?: string): Promise<ModelRecord>
    hardware(o?: { refresh?: boolean }): Promise<Record<string, unknown>>
    storage(): Promise<Record<string, unknown>>
    providers(): Promise<Record<string, unknown>[]>
    addProvider(o: { baseUrl: string; model: string; apiKey?: string; label?: string }): Promise<Record<string, unknown>>
  }
  orcha: {
    run(task: string, o?: { graph?: 'default' | 'research' | 'multi_agent'; wait?: boolean; timeoutMs?: number; [k: string]: unknown }): Promise<Job<OrchaResult>>
    status(): Promise<Record<string, unknown>>
    jobs(o?: { state?: string; limit?: number }): Promise<Record<string, unknown>[]>
    cancel(id: string): Promise<Job>
  }
  jobs: {
    get(id: string): Promise<Job>
    list(o?: { state?: string; kind?: string; limit?: number }): Promise<Record<string, unknown>[]>
    cancel(id: string): Promise<Job>
  }
  memory: {
    store(content: string, o?: { title?: string; type?: string; tags?: string[]; importance?: number; scope?: 'app' | 'shared'; workspace?: string; extra?: Record<string, unknown> }): Promise<MemoryItem>
    search(query: string, o?: { limit?: number; scope?: 'app' | 'shared' | 'all'; workspace?: string; tags?: string[]; type?: string }): Promise<MemoryItem[]>
    list(o?: { limit?: number; [k: string]: unknown }): Promise<MemoryItem[]>
    get(id: string): Promise<MemoryItem>
    delete(id: string): Promise<{ id: string; deleted: boolean }>
  }
  /** Per-app document index (chunk + BM25 retrieval) for grounding. */
  context: {
    put(collection: string, docId: string, text: string, o?: { title?: string; metadata?: Record<string, unknown> }): Promise<Record<string, unknown>>
    search(collection: string, query: string, o?: { limit?: number; docIds?: string[] }): Promise<{ id: string; doc_id: string; text: string; score: number; title?: string }[]>
    collections(): Promise<Record<string, unknown>[]>
    documents(collection: string): Promise<Record<string, unknown>[]>
    delete(collection: string, docId?: string): Promise<Record<string, unknown>>
  }
  /** Shared resources: private by default; shared with named apps by the owner; referenced, never copied. */
  resources: {
    create(title: string, o?: { type?: string; collection?: string; docIds?: string[]; text?: string; path?: string; workspace?: string; metadata?: Record<string, unknown> }): Promise<Resource>
    list(o?: { type?: string; workspace?: string; owned?: boolean }): Promise<Resource[]>
    get(id: string): Promise<Resource>
    update(id: string, o: { title?: string; metadata?: Record<string, unknown>; workspace?: string; text?: string }): Promise<Resource>
    /** Resolve a `runtime://res_...` reference (only if this app may see it). */
    resolve(ref: string): Promise<Resource & { document: string | null }>
    read(id: string, doc?: string): Promise<{ resource: Resource; text?: string; documents?: Record<string, unknown>[] }>
    write(id: string, docId: string, text: string, o?: { title?: string }): Promise<Record<string, unknown>>
    search(query: string, o?: { resources?: string[]; workspace?: string; types?: string[]; limit?: number }): Promise<{ searched: number; count: number; items: { resource: string; resource_title: string; owner: string; doc_id: string; text: string; score: number }[] }>
    share(id: string, apps: string[], o?: { access?: 'read' | 'write' }): Promise<Resource>
    revoke(id: string, apps?: string[]): Promise<Resource>
    permissions(id: string): Promise<Record<string, unknown>>
    requestAccess(id: string, o?: { access?: 'read' | 'write'; reason?: string }): Promise<{ id: string; state: string }>
    requests(state?: string): Promise<Record<string, unknown>[]>
    decide(requestId: string, approve: boolean): Promise<Record<string, unknown>>
    delete(id: string, o?: { purge?: boolean }): Promise<{ id: string; deleted: boolean }>
    audit(limit?: number, resource?: string): Promise<Record<string, unknown>[]>
    workspaces(): Promise<Record<string, unknown>[]>
    createWorkspace(name: string): Promise<{ id: string; name: string }>
    shareWorkspace(name: string, apps: string[], o?: { access?: 'read' | 'write' }): Promise<Record<string, unknown>>
  }

  health(): Promise<Record<string, unknown>>
  version(): Promise<Record<string, unknown>>
  status(): Promise<Record<string, unknown>>
  me(): Promise<{ kind: string; app_id: string | null; permissions: string[] }>
  hasCapability(name: string): boolean
  chat(messages: ChatMessage[], o: ChatOptions & { stream: true }): AsyncGenerator<string>
  chat(messages: ChatMessage[], o?: ChatOptions & { stream?: false }): Promise<Record<string, any>>
  chatText(messages: ChatMessage[], o?: ChatOptions): Promise<string>
  task(task: string, o?: { wait?: boolean; [k: string]: unknown }): Promise<Job<OrchaResult>>
  agentRun(task: string, o?: { workspaceRoots?: string[]; wait?: boolean; [k: string]: unknown }): Promise<Job<OrchaResult>>
}

export default AnviraRuntime

// ------------------------------------------------- locate / register / GitHub install
export const DEFAULT_REPO: string
export interface RuntimeLocation {
  found: boolean
  /** The folder the runtime lives in (may be on any drive). */
  home: string
  /** How it was found: ANVIRA_RUNTIME_HOME, a remembered custom location, or the default folder. */
  source: 'env' | 'pointer' | 'default'
  defaultHome: string
  running: boolean
  ready: boolean
  version: string | null
  portable: boolean
  gpuPack: boolean
  problem: string | null
}
export interface NvidiaGpu { name: string; driver: string; vramMib: number | null; cudaOk: boolean }
export interface GpuPackOffer { name: string; driver: string; size: number; cudaOk: boolean }
export type DownloadProgress = (asset: string, done: number, total: number) => void
export interface InstallFromGithubOptions {
  /** Any folder (any drive). Default: the standard per-user location. The choice is remembered for every app. */
  dest?: string
  repo?: string
  tag?: string
  /** true = fetch the NVIDIA GPU pack, false = never, undefined = ask confirmGpu on an NVIDIA machine. */
  gpu?: boolean
  confirmGpu?: (offer: GpuPackOffer) => boolean | Promise<boolean>
  env?: Record<string, string | undefined>
  onStatus?: (message: string) => void
  onProgress?: DownloadProgress
  force?: boolean
  stop?: () => unknown
}
export function defaultHome(env?: Record<string, string | undefined>): string
/** Where is the runtime, how do we know, and is it usable? Never throws and never installs. */
export function locate(env?: Record<string, string | undefined>): Promise<RuntimeLocation>
/** "I already have it": remember an existing install folder. Nothing is copied. Throws not_a_runtime_folder if it is not one. */
export function registerLocation(dir: string, env?: Record<string, string | undefined>): string
export function forgetLocation(env?: Record<string, string | undefined>): void
export function nvidiaGpu(): NvidiaGpu | null
export function updateInProgress(env?: Record<string, string | undefined>): boolean
export function waitForUpdate(env?: Record<string, string | undefined>, timeoutMs?: number, onStatus?: (m: string) => void): Promise<void>
export function downloadAsset(asset: { name: string; size?: number; browser_download_url: string }, dir: string, sha256?: string, onProgress?: DownloadProgress): Promise<string>
export function extractPackage(zip: string, dest: string): void
/** Install from a local AnviraRuntime zip (offline). */
export function installPackage(zip: string, o?: { dest?: string; env?: Record<string, string | undefined>; onStatus?: (m: string) => void; stop?: () => unknown }): Promise<Record<string, unknown>>
/** Download the latest release from GitHub (SHA-256 verified, resumable). Call only after the user chose to. */
export function installFromGithub(o?: InstallFromGithubOptions): Promise<Record<string, unknown> & { gpu_pack_available?: GpuPackOffer; unchanged?: boolean }>
/** What `connect({ install })` may return: false to decline, true for defaults, or the user's explicit choice. */
export type InstallChoice = boolean | { action: 'use-existing'; path: string } | { action: 'download'; dest?: string; gpu?: boolean; repo?: string; confirmGpu?: InstallFromGithubOptions['confirmGpu']; onProgress?: DownloadProgress }
