// @anvira/runtime-client — TypeScript/JavaScript SDK for Anvira Runtime (Node >= 18, ESM).
//
// Use it from an Electron *main* process or a Node backend. Do NOT call the runtime from a
// renderer/browser: the runtime rejects browser Origins by design and the app token must stay
// out of web content (expose your own IPC methods instead).
//
//   const runtime = await AnviraRuntime.connect({ appId: 'anvira-notes', install: askUser })
//   await runtime.models.use('qwen3-8b')
//   const reply = await runtime.chatText([{ role: 'user', content: 'hi' }])
//   const job = await runtime.orcha.run('Summarise my notes', { wait: true })
//   await runtime.memory.store('Prefers concise answers')

import { spawn, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

export const SDK_API_VERSION = 1

// ---------------------------------------------------------------- errors
export class AnviraError extends Error {
  constructor(code, message, { status, hint, details } = {}) {
    super(message)
    this.name = 'AnviraError'
    this.code = code
    this.status = status
    this.hint = hint
    this.details = details || {}
  }
  static fromResponse(status, body) {
    const err = body && body.error
    if (err && typeof err === 'object') {
      return new AnviraError(err.code || 'error', err.message || 'Request failed', { status, hint: err.hint, details: err.details })
    }
    return new AnviraError('http_error', `HTTP ${status}`, { status })
  }
}
export class RuntimeNotInstalled extends AnviraError {}
export class RuntimeNotRunning extends AnviraError {}
export class RuntimeStartFailed extends AnviraError {}
export class IncompatibleRuntime extends AnviraError {}
export class InstallDeclined extends AnviraError {}
export class PermissionDenied extends AnviraError {}

// ----------------------------------------------------------------- paths
/** Mirrors the runtime's own layout rules (parity-tested against the Python implementation). */
export function runtimeDirs(env = process.env, platform = process.platform, { ignorePointer = false } = {}) {
  const home = env.USERPROFILE || env.HOME || os.homedir()
  const j = path.join
  let d
  if (env.ANVIRA_RUNTIME_HOME) {
    const r = env.ANVIRA_RUNTIME_HOME
    d = { home: r, state: j(r, 'state'), config: j(r, 'config'), logs: j(r, 'logs'), models: j(r, 'models'), bin: j(r, 'bin') }
  } else if (platform === 'win32') {
    const r = j(env.LOCALAPPDATA || j(home, 'AppData', 'Local'), 'AnviraRuntime')
    d = { home: r, state: j(r, 'state'), config: j(r, 'config'), logs: j(r, 'logs'), models: j(r, 'models'), bin: j(r, 'bin') }
  } else if (platform === 'darwin') {
    const r = j(home, 'Library', 'Application Support', 'AnviraRuntime')
    d = { home: r, state: j(r, 'state'), config: j(r, 'config'), logs: j(home, 'Library', 'Logs', 'AnviraRuntime'), models: j(r, 'models'), bin: j(r, 'bin') }
  } else {
    const data = j(env.XDG_DATA_HOME || j(home, '.local', 'share'), 'anvira-runtime')
    const state = j(env.XDG_STATE_HOME || j(home, '.local', 'state'), 'anvira-runtime')
    d = { home: data, state, config: j(env.XDG_CONFIG_HOME || j(home, '.config'), 'anvira-runtime'), logs: j(state, 'logs'), models: j(data, 'models'), bin: j(data, 'bin') }
  }
  if (!env.ANVIRA_RUNTIME_HOME && !ignorePointer) {
    // A custom-location install: <default home>/location.json -> { home } (see anvira_runtime.config.paths)
    const custom = pointerTarget(d.home)
    if (custom) d = { home: custom, state: j(custom, 'state'), config: j(custom, 'config'), logs: j(custom, 'logs'), models: j(custom, 'models'), bin: j(custom, 'bin') }
  }
  if (env.ANVIRA_MODELS_DIR) d.models = env.ANVIRA_MODELS_DIR
  return d
}

function pointerTarget(defaultHome) {
  const target = readJson(path.join(defaultHome, 'location.json')).home
  return target && fs.existsSync(path.join(target, 'install.json')) ? target : null
}

function readJson(file) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')) } catch { return {} }
}

function pidAlive(pid) {
  if (!pid) return false
  try { process.kill(pid, 0); return true } catch (e) { return e.code === 'EPERM' }
}
// NB: process.kill(pid, 0) only *tests* on Node (unlike Python's os.kill on Windows).

async function getJson(url, timeoutMs = 2000) {
  try {
    const r = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) })
    return r.ok ? await r.json() : null
  } catch { return null }
}

// ------------------------------------------------------------- detection
/** Is the runtime installed / running? Never throws and never installs anything. */
export async function detect(env = process.env) {
  const dirs = runtimeDirs(env)
  const info = { installed: false, running: false, ready: false, home: dirs.home, install: readJson(path.join(dirs.home, 'install.json')),
    host: '127.0.0.1', port: null, pid: null, runtimeVersion: null, apiVersion: null, capabilities: [], staleDiscovery: false, error: null }
  const entry = info.install.entry && (path.isAbsolute(info.install.entry) ? info.install.entry : path.join(dirs.home, info.install.entry)) // portable installs record it relative
  info.installed = Object.keys(info.install).length > 0 && (!entry || fs.existsSync(entry))
  if (fs.existsSync(path.join(dirs.state, 'owner.token')) && !info.error) info.installed = true // a runtime has run here before
  const disc = readJson(path.join(dirs.state, 'runtime.json'))
  if (disc.pid) {
    info.host = disc.host || '127.0.0.1'; info.port = disc.port; info.pid = disc.pid; info.installed = true
    if (!pidAlive(info.pid)) { info.staleDiscovery = true; info.port = null }
  }
  if (info.port) {
    const base = `http://${info.host}:${info.port}`
    const [ver, health] = await Promise.all([getJson(`${base}/version`), getJson(`${base}/health`)])
    if (ver && ver.service === 'anvira-runtime' && health) {
      info.running = true
      info.ready = health.status === 'ok' || health.status === 'degraded'
      info.runtimeVersion = ver.runtime_version; info.apiVersion = ver.api_version; info.capabilities = ver.capabilities || []
    } else {
      info.staleDiscovery = true
      info.error = `Nothing answering as Anvira Runtime on ${info.host}:${info.port}`
    }
  }
  return info
}

function semver(v) {
  const p = String(v || '0').replace(/^v/i, '').split('-')[0].split('.').slice(0, 3).map((x) => parseInt(x, 10) || 0)
  while (p.length < 3) p.push(0)
  return p
}
function semverGte(a, b) {
  const x = semver(a), y = semver(b)
  for (let i = 0; i < 3; i++) { if (x[i] !== y[i]) return x[i] > y[i] }
  return true
}

function findPython(env) {
  const candidates = env.ANVIRA_PYTHON ? [env.ANVIRA_PYTHON] : process.platform === 'win32' ? ['py', 'python', 'python3'] : ['python3', 'python']
  for (const c of candidates) {
    const r = spawnSync(c, ['--version'], { stdio: 'ignore' })
    if (r.status === 0) return c
  }
  return null
}

function runtimePython(env) {
  const home = runtimeDirs(env).home
  const raw = readJson(path.join(home, 'install.json')).entry
  const entry = raw && (path.isAbsolute(raw) ? raw : path.join(home, raw))
  return entry && fs.existsSync(entry) ? entry : null
}

async function waitReady(env, timeoutMs, onStatus) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const info = await detect(env)
    if (info.running && info.ready) return info
    await new Promise((r) => setTimeout(r, 300))
  }
  throw new RuntimeStartFailed('runtime_start_failed', `Runtime did not become ready within ${Math.round(timeoutMs / 1000)}s.`, { hint: 'Run `anvira doctor` for diagnostics.' })
}

/** Start the runtime daemon (detached) and wait until it is healthy. */
export async function startRuntime({ env = process.env, timeoutMs = 90000, onStatus, autoStop = false, idleGraceS } = {}) {
  await waitForUpdate(env, 240000, onStatus) // an install/update is replacing files: never revive the runtime halfway through
  const info = await detect(env)
  if (info.running && info.ready) return info
  if (info.running) return waitReady(env, timeoutMs, onStatus)
  const py = runtimePython(env) || (env.ANVIRA_ALLOW_DEV_PYTHON === '1' ? findPython(env) : null)
  if (!py) throw new RuntimeNotInstalled('runtime_not_installed', 'Anvira Runtime is not installed.', { hint: 'Install it first (see INSTALLATION.md).' })
  const dirs = runtimeDirs(env)
  fs.mkdirSync(dirs.logs, { recursive: true })
  const log = fs.openSync(path.join(dirs.logs, 'daemon-boot.log'), 'a')
  // autoStop = on demand: the runtime stops by itself shortly after the last app closed
  const daemonArgs = ['-m', 'anvira_runtime', 'daemon', ...(autoStop ? ['--auto-stop'] : []), ...(idleGraceS ? ['--idle-grace', String(idleGraceS)] : [])]
  const child = spawn(py, daemonArgs, { env: { ...process.env, ...env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
    // Windows: NOT `detached` (that is DETACHED_PROCESS, which makes every child open a visible console window);
    // windowsHide gives the daemon a hidden console that its children inherit, and it still outlives this process.
    detached: process.platform !== 'win32', stdio: ['ignore', log, log], windowsHide: true })
  child.unref()
  let exited = null
  child.on('exit', (code) => { exited = code })
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (exited !== null) throw new RuntimeStartFailed('runtime_start_failed', 'Runtime failed to start.', { hint: 'Run `anvira doctor` for diagnostics.' })
    const cur = await detect(env)
    if (cur.running && cur.ready) { onStatus && onStatus('Runtime is up.'); return cur }
    await new Promise((r) => setTimeout(r, 300))
  }
  throw new RuntimeStartFailed('runtime_start_failed', `Runtime did not become ready within ${Math.round(timeoutMs / 1000)}s.`, { hint: 'Run `anvira doctor` for diagnostics.' })
}

// ------------------------------------------------- locate / register / GitHub install
const UPDATE_LOCK = '.updating'
/** Rewritten by scripts/publish.ps1 once the GitHub repository exists; ANVIRA_RUNTIME_REPO overrides it. */
export const DEFAULT_REPO = 'Vansh-synthetica/Anvira-Runtime'
const MIN_CUDA_DRIVER = 525

export function defaultHome(env = process.env) { return runtimeDirs(env, process.platform, { ignorePointer: true }).home }

/** Where is the runtime, how do we know, and is it usable? Never throws, never installs. */
export async function locate(env = process.env) {
  const dirs = runtimeDirs(env)
  const info = await detect(env)
  const source = env.ANVIRA_RUNTIME_HOME ? 'env' : path.resolve(dirs.home) !== path.resolve(defaultHome(env)) ? 'pointer' : 'default'
  return { found: info.installed, home: dirs.home, source, defaultHome: defaultHome(env), running: info.running, ready: info.ready,
    version: info.runtimeVersion || info.install.version || null, portable: !!info.install.portable, gpuPack: !!info.install.gpu_pack, problem: info.error }
}

/** "I already have it": remember an existing install folder so every app finds it there. Nothing is copied. */
export function registerLocation(dir, env = process.env) {
  const target = path.resolve(String(dir || ''))
  if (!fs.existsSync(path.join(target, 'install.json'))) {
    throw new AnviraError('not_a_runtime_folder', `'${target}' does not look like an Anvira Runtime folder.`,
      { hint: 'Pick the folder that contains install.json (the one with anvira.cmd and python/).' })
  }
  const def = defaultHome(env)
  if (path.resolve(def) === target) { forgetLocation(env); return target }
  fs.mkdirSync(def, { recursive: true })
  fs.writeFileSync(path.join(def, 'location.json'), JSON.stringify({ home: target, registered_at: Date.now() / 1000 }, null, 2))
  return target
}
export function forgetLocation(env = process.env) { try { fs.unlinkSync(path.join(defaultHome(env), 'location.json')) } catch { /* nothing to forget */ } }

/** The first NVIDIA GPU with its driver, or null (uses nvidia-smi, which ships with the driver). */
export function nvidiaGpu() {
  const r = spawnSync('nvidia-smi', ['--query-gpu=name,driver_version,memory.total', '--format=csv,noheader,nounits'], { encoding: 'utf8', windowsHide: true, timeout: 10000 })
  if (r.status !== 0 || !r.stdout || !r.stdout.trim()) return null
  const [name = '', driver = '', vram = ''] = r.stdout.trim().split('\n')[0].split(',').map((x) => x.trim())
  const major = parseInt(driver, 10) || 0
  return { name, driver, vramMib: /^\d+$/.test(vram) ? Number(vram) : null, cudaOk: major >= MIN_CUDA_DRIVER }
}

export function updateInProgress(env = process.env) {
  try { return Date.now() - fs.statSync(path.join(runtimeDirs(env).home, UPDATE_LOCK)).mtimeMs < 20 * 60 * 1000 } catch { return false }
}
export async function waitForUpdate(env = process.env, timeoutMs = 240000, onStatus) {
  const end = Date.now() + timeoutMs
  let told = false
  while (updateInProgress(env) && Date.now() < end) {
    if (!told && onStatus) { onStatus('Waiting for the Anvira Runtime update to finish...'); told = true }
    await new Promise((r) => setTimeout(r, 500))
  }
}
function lock(home, on) {
  try {
    if (on) { fs.mkdirSync(home, { recursive: true }); fs.writeFileSync(path.join(home, UPDATE_LOCK), String(process.pid)) } else fs.rmSync(path.join(home, UPDATE_LOCK), { force: true })
  } catch { /* best effort */ }
}

function repoName(repo, env) {
  const r = repo || env.ANVIRA_RUNTIME_REPO || DEFAULT_REPO
  if (r.startsWith('OWNER/')) throw new AnviraError('repo_not_configured', 'The Anvira Runtime GitHub repository is not configured in this build.', { hint: 'Set ANVIRA_RUNTIME_REPO=<owner>/Anvira-Runtime, or pass { repo }.' })
  return r
}

async function fetchRelease({ repo, tag, env }) {
  const r = repoName(repo, env)
  const api = String(env.ANVIRA_RELEASE_API || 'https://api.github.com').replace(/\/+$/, '')
  let res
  try { res = await fetch(`${api}/repos/${r}/releases/${tag ? `tags/${tag}` : 'latest'}`, { headers: { 'User-Agent': 'anvira-runtime-installer', Accept: 'application/vnd.github+json' } }) } catch (e) {
    throw new AnviraError('download_failed', `Could not reach GitHub: ${e.message}`, { hint: 'Check your internet connection.' })
  }
  if (res.status === 404) throw new AnviraError('release_not_found', `No Anvira Runtime release found in github.com/${r}${tag ? ` for tag ${tag}` : ''}.`, { hint: 'Check the repository name, or that a release is published.' })
  if (!res.ok) throw new AnviraError('download_failed', `GitHub answered HTTP ${res.status}.`, { hint: 'Try again later, or download the zip by hand and use installPackage().' })
  return res.json()
}

async function pickAssets(release) {
  const assets = new Map((release.assets || []).map((a) => [a.name, a]))
  const core = process.platform === 'win32' ? [...assets.values()].find((a) => /^AnviraRuntime-[\d.]+-win-x64\.zip$/.test(a.name)) : null
  if (!core) throw new AnviraError('no_release_asset', `This release has no Anvira Runtime package for ${process.platform}.`, { hint: 'Only Windows x64 packages are published so far.' })
  const cuda = [...assets.values()].find((a) => a.name.endsWith('-win-x64-cuda.zip')) || null
  const sums = {}
  if (assets.has('SHA256SUMS')) {
    const text = await (await fetch(assets.get('SHA256SUMS').browser_download_url, { headers: { 'User-Agent': 'anvira-runtime-installer' } })).text()
    for (const line of text.split(/\r?\n/)) { const m = line.trim().match(/^([0-9a-f]{64})\s+\*?(.+)$/i); if (m) sums[m[2]] = m[1].toLowerCase() }
  }
  return { core, cuda, sums, version: String(release.tag_name || '').replace(/^v/i, '') }
}

async function sha256File(file) {
  const hash = createHash('sha256')
  await new Promise((resolve, reject) => { fs.createReadStream(file).on('data', (c) => hash.update(c)).on('end', resolve).on('error', reject) })
  return hash.digest('hex')
}

/** Download one release asset (resumable), verifying size and SHA-256. Returns the file path. */
export async function downloadAsset(asset, dir, sha256, onProgress) {
  fs.mkdirSync(dir, { recursive: true })
  const final = path.join(dir, asset.name), part = final + '.part', total = Number(asset.size || 0)
  if (fs.existsSync(final) && (!sha256 || (await sha256File(final)) === sha256)) return final
  const have = fs.existsSync(part) ? fs.statSync(part).size : 0
  let res
  try { res = await fetch(asset.browser_download_url, { headers: { 'User-Agent': 'anvira-runtime-installer', ...(have ? { Range: `bytes=${have}-` } : {}) } }) } catch (e) {
    throw new AnviraError('download_failed', `Download of ${asset.name} failed: ${e.message}`, { hint: 'Run the install again to resume.' })
  }
  if (res.status !== 416) {
    if (!res.ok) throw new AnviraError('download_failed', `Download of ${asset.name} failed (HTTP ${res.status}).`)
    const resumed = res.status === 206
    const out = fs.createWriteStream(part, { flags: resumed ? 'a' : 'w' })
    let done = resumed ? have : 0
    try {
      for await (const chunk of res.body) {
        if (!out.write(chunk)) await new Promise((r) => out.once('drain', r))
        done += chunk.length
        onProgress && onProgress(asset.name, done, total)
      }
    } finally { await new Promise((r) => out.end(r)) }
  }
  if (total && fs.statSync(part).size !== total) throw new AnviraError('download_incomplete', `${asset.name} was cut short.`, { hint: 'Run the install again to resume.' })
  if (sha256 && (await sha256File(part)) !== sha256) {
    fs.rmSync(part, { force: true })
    throw new AnviraError('checksum_mismatch', `${asset.name} failed its SHA-256 check and was discarded.`, { hint: 'Run the install again; if it keeps failing the release may be corrupt.' })
  }
  fs.renameSync(part, final)
  return final
}

/** Unpack an AnviraRuntime zip into `dest` (its top folder stripped). Uses tar (bsdtar ships with Windows 10+ and macOS) or unzip. */
export function extractPackage(zip, dest) {
  fs.mkdirSync(dest, { recursive: true })
  const tmp = fs.mkdtempSync(path.join(dest, '.extract-'))
  try {
    let r = spawnSync('tar', ['-xf', zip, '-C', tmp], { encoding: 'utf8', windowsHide: true })
    if (r.status !== 0) r = spawnSync('unzip', ['-o', '-q', zip, '-d', tmp], { encoding: 'utf8', windowsHide: true })
    if (r.status !== 0) throw new AnviraError('bad_bundle', `Could not unpack ${zip}: ${(r.stderr || r.error?.message || '').slice(0, 300)}`)
    const top = path.join(tmp, 'AnviraRuntime')
    const from = fs.existsSync(top) ? top : tmp
    for (const name of fs.readdirSync(from)) {
      if (from === tmp && name === 'AnviraRuntime') continue
      fs.cpSync(path.join(from, name), path.join(dest, name), { recursive: true, force: true })
    }
  } finally { fs.rmSync(tmp, { recursive: true, force: true }) }
}

function finish(target, env, source, gpuPack) {
  const marker = path.join(target, 'install.json')
  const record = readJson(marker)
  if (!Object.keys(record).length) throw new AnviraError('bad_bundle', 'The package has no install.json.')
  Object.assign(record, { installed_at: Date.now() / 1000, source }, gpuPack === undefined ? {} : { gpu_pack: gpuPack })
  fs.writeFileSync(marker, JSON.stringify(record, null, 2))
  if (path.resolve(target) !== path.resolve(defaultHome(env))) registerLocation(target, env); else forgetLocation(env)
  return record
}

/** Install from a local AnviraRuntime zip (offline). `dest` may be any folder. */
export async function installPackage(zip, { dest, env = process.env, onStatus, stop } = {}) {
  const target = path.resolve(dest || defaultHome(env))
  lock(target, true)
  try {
    if (stop) await stop()
    onStatus && onStatus(`Unpacking Anvira Runtime into ${target} ...`)
    extractPackage(zip, target)
    return finish(target, env, zip)
  } finally { lock(target, false) }
}

/**
 * Download the latest release from GitHub into `dest` (any folder; default: the standard location). Call only after the user chose to.
 * `gpu`: true = fetch the CUDA pack, false = never, undefined = ask `confirmGpu(info)` on an NVIDIA machine (no callback -> not fetched;
 * the result's `gpu_pack_available` says it could be).
 */
export async function installFromGithub({ dest, repo, tag, gpu, confirmGpu, env = process.env, onStatus, onProgress, force = false, stop } = {}) {
  const say = onStatus || (() => {})
  const target = path.resolve(dest || defaultHome(env))
  say(`Looking up the latest Anvira Runtime release (${repoName(repo, env)}) ...`)
  const { core, cuda, sums, version } = await pickAssets(await fetchRelease({ repo, tag, env }))
  const existing = readJson(path.join(target, 'install.json'))
  if (Object.keys(existing).length && !force && existing.version === version && fs.existsSync(path.join(target, 'python'))) {
    say(`Anvira Runtime ${version} is already installed at ${target}.`)
    return { ...existing, unchanged: true }
  }
  fs.mkdirSync(target, { recursive: true })
  const need = Number(core.size || 0) * 3
  const st = fs.statfsSync ? fs.statfsSync(target) : null
  if (st && need && st.bavail * st.bsize < need) throw new AnviraError('insufficient_storage', `${target} has ${(st.bavail * st.bsize / 1e9).toFixed(1)} GB free; about ${(need / 1e9).toFixed(1)} GB is needed.`, { hint: 'Choose another folder or drive.' })
  lock(target, true)
  try {
    if (stop) await stop()
    const work = path.join(target, '.download')
    say(`Downloading ${core.name} (${Math.round(core.size / 1e6)} MB) ...`)
    extractPackage(await downloadAsset(core, work, sums[core.name], onProgress), target)
    let gpuDone = false
    const card = nvidiaGpu()
    const extra = {}
    if (cuda && card) {
      extra.gpu_pack_available = { name: card.name, driver: card.driver, size: Number(cuda.size || 0), cudaOk: card.cudaOk }
      let want = gpu === true || (gpu === undefined && !!confirmGpu && !!(await confirmGpu(extra.gpu_pack_available)))
      if (want && !card.cudaOk) { say(`Skipping the GPU pack: driver ${card.driver} is older than the required ${MIN_CUDA_DRIVER}.x. CPU mode will be used.`); want = false }
      if (want) {
        say(`Downloading the NVIDIA GPU pack (${Math.round(cuda.size / 1e6)} MB) ...`)
        extractPackage(await downloadAsset(cuda, work, sums[cuda.name], onProgress), target)
        gpuDone = true
      }
    }
    fs.rmSync(work, { recursive: true, force: true })
    const record = finish(target, env, `github:${repoName(repo, env)}@${tag || version}`, gpuDone)
    say(`Anvira Runtime ${record.version} installed at ${target}.`)
    return { ...record, ...extra }
  } finally { lock(target, false) }
}

/**
 * Install the shared runtime. Only call after the USER agreed. `source` must be a runtime bundle
 * (.zip) or a source checkout directory containing sdk/python. Requires Python 3.10+ on the machine.
 * Models are never downloaded here.
 */
export async function installRuntime({ source, env = process.env, force = false, onStatus } = {}) {
  const py = findPython(env)
  if (!py) throw new RuntimeNotInstalled('python_missing', 'Python 3.10+ is required to install Anvira Runtime.', { hint: 'Install Python from python.org or set ANVIRA_PYTHON.' })
  if (!source) throw new AnviraError('no_install_source', 'installRuntime needs a runtime bundle: { source: "<bundle.zip | directory>" }.')
  let dir = source
  let tmp = null
  if (source.toLowerCase().endsWith('.zip')) {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'anvira-bundle-'))
    const z = spawnSync(py, ['-m', 'zipfile', '-e', source, tmp], { encoding: 'utf8' })
    if (z.status !== 0) throw new AnviraError('bad_install_source', `Could not unpack ${source}: ${z.stderr}`)
    const entries = fs.readdirSync(tmp)
    dir = entries.length === 1 && !fs.existsSync(path.join(tmp, 'runtime')) ? path.join(tmp, entries[0]) : tmp
  }
  try {
    onStatus && onStatus('Installing Anvira Runtime...')
    const code = `import sys, json; sys.path.insert(0, ${JSON.stringify(path.join(dir, 'sdk', 'python'))}); ` +
      `from anvira_client import bootstrap; print(json.dumps(bootstrap.install_runtime(source=${JSON.stringify(dir)}, force=${force ? 'True' : 'False'})))`
    const r = spawnSync(py, ['-c', code], { encoding: 'utf8', env: { ...process.env, ...env } })
    if (r.status !== 0) throw new AnviraError('install_failed', 'Runtime installation failed.', { hint: (r.stderr || r.stdout || '').slice(-600) })
    return JSON.parse(r.stdout.trim().split('\n').pop())
  } finally {
    if (tmp) fs.rmSync(tmp, { recursive: true, force: true })
  }
}

// ------------------------------------------------------------------ http
class Http {
  constructor(baseUrl, token, timeoutMs = 30000, recover = null) { this.baseUrl = baseUrl; this.token = token; this.timeoutMs = timeoutMs; this.recover = recover }
  url(p, params) {
    const u = new URL(this.baseUrl + p)
    for (const [k, v] of Object.entries(params || {})) {
      if (v === undefined || v === null) continue
      for (const x of Array.isArray(v) ? v : [v]) u.searchParams.append(k, String(x))
    }
    return u
  }
  async raw(method, p, opts = {}) {
    try {
      return await this.rawOnce(method, p, opts)
    } catch (e) {
      // the runtime is gone (an on-demand runtime stopped while this app stayed open): start it again, retry once
      if (!(e instanceof RuntimeNotRunning) || !this.recover) throw e
      const nb = await this.recover()
      if (!nb) throw e
      this.baseUrl = nb
      return this.rawOnce(method, p, opts)
    }
  }
  async rawOnce(method, p, { body, params, headers, timeoutMs } = {}) {
    const h = { Accept: 'application/json', ...(headers || {}) }
    if (this.token) h.Authorization = `Bearer ${this.token}`
    if (body !== undefined) h['Content-Type'] = 'application/json'
    let res
    try {
      res = await fetch(this.url(p, params), { method, headers: h, body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(timeoutMs || this.timeoutMs) })
    } catch (e) {
      if (e && (e.name === 'TimeoutError' || e.name === 'AbortError')) throw new AnviraError('timeout', `The runtime did not answer within ${Math.round((timeoutMs || this.timeoutMs) / 1000)}s.`)
      throw new RuntimeNotRunning('runtime_unreachable', `Cannot reach Anvira Runtime at ${this.baseUrl}: ${e.cause?.code || e.message}`, { hint: 'Start it with `anvira runtime start`.' })
    }
    if (!res.ok) {
      let payload = null
      try { payload = await res.json() } catch { /* not json */ }
      const err = AnviraError.fromResponse(res.status, payload)
      throw err.code === 'permission_denied' || err.code === 'unauthorized' ? Object.assign(new PermissionDenied(err.code, err.message, { status: err.status, hint: err.hint }), {}) : err
    }
    return res
  }
  async json(method, p, opts) {
    const res = await this.raw(method, p, opts)
    const text = await res.text()
    return text ? JSON.parse(text) : {}
  }
  /** Async generator of { event, data } from an SSE response. */
  async *sse(method, p, opts = {}) {
    const res = await this.raw(method, p, { ...opts, headers: { Accept: 'text/event-stream', ...(opts.headers || {}) }, timeoutMs: opts.timeoutMs || 600000 })
    const decoder = new TextDecoder()
    let buf = ''
    let event = 'message'
    let data = []
    for await (const chunk of res.body) {
      buf += decoder.decode(chunk, { stream: true })
      let idx
      while ((idx = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, idx).replace(/\r$/, '')
        buf = buf.slice(idx + 1)
        if (line === '') { if (data.length) yield { event, data: data.join('\n') }; event = 'message'; data = [] }
        else if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data.push(line.slice(5).trimStart())
      }
    }
    if (data.length) yield { event, data: data.join('\n') }
  }
}

// ------------------------------------------------------------------- jobs
const TERMINAL = ['completed', 'failed', 'cancelled', 'interrupted']
export class Job {
  constructor(client, data) { this._c = client; this.data = data }
  get id() { return this.data.id }
  get kind() { return this.data.kind }
  get state() { return this.data.state }
  get result() { return this.data.result }
  get error() { return this.data.error }
  get progress() { return this.data.progress || {} }
  get done() { return TERMINAL.includes(this.data.state) }
  async refresh() { this.data = (await this._c._http.json('GET', `/v1/jobs/${this.id}`)).job; return this }
  async wait({ timeoutMs = 600000, pollMs = 500 } = {}) {
    const deadline = Date.now() + timeoutMs
    while (!this.done) {
      if (Date.now() > deadline) throw new AnviraError('timeout', `Job ${this.id} still ${this.state} after ${Math.round(timeoutMs / 1000)}s.`)
      await new Promise((r) => setTimeout(r, pollMs))
      await this.refresh()
    }
    return this
  }
  async cancel() { this.data = (await this._c._http.json('POST', `/v1/jobs/${this.id}/cancel`)).job; return this }
  /** Result of a completed job; throws AnviraError if it failed or was cancelled. */
  async unwrap() {
    if (!this.done) await this.wait()
    if (this.state !== 'completed') {
      const e = this.error || {}
      throw new AnviraError(e.code || this.state, e.message || `Job ${this.state}`, { hint: e.hint, details: e.details })
    }
    return this.result
  }
  /** Raw ORCHA event frames (advanced). */
  async *events() {
    for await (const { data } of this._c._http.sse('GET', `/v1/jobs/${this.id}/events`)) {
      try { yield JSON.parse(data) } catch { yield { raw: data } }
    }
  }
}

// ---------------------------------------------------------------- client
export class AnviraRuntime {
  constructor(http, info, appId) {
    this._http = http; this.info = info; this.appId = appId
    this._lease = null; this._timer = null; this._closed = false; this._opts = {}
    const c = this
    this.models = {
      list: async ({ installed = false } = {}) => (await http.json('GET', '/v1/models', { params: { installed: installed ? 'true' : undefined } })).models,
      installed: async () => (await http.json('GET', '/v1/models/installed')).models,
      catalog: async (q) => (await http.json('GET', '/v1/models/catalog', { params: { q } })).models,
      search: async (q, limit = 20) => (await http.json('GET', '/v1/models/search', { params: { q, limit } })).models,
      recommended: async (limit = 5) => (await http.json('GET', '/v1/models/recommended', { params: { limit } })).models,
      active: () => http.json('GET', '/v1/models/active'),
      get: (id) => http.json('GET', `/v1/models/${id}`),
      compatibility: (id) => http.json('GET', `/v1/models/${id}/compatibility`),
      use: (id, { waitS = 120 } = {}) => http.json('POST', '/v1/models/select', { body: { id, wait_s: waitS }, timeoutMs: (waitS + 40) * 1000 }),
      install: async (model, { url, file, dir, wait = false } = {}) => {
        const job = new Job(c, (await http.json('POST', '/v1/models/install', { body: { model, url, file, dir } })).job)
        return wait ? job.wait({ timeoutMs: 3600000 }) : job
      },
      remove: (id, { deleteFile, confirm = false } = {}) => http.json('POST', '/v1/models/remove', { body: { id, delete_file: deleteFile, confirm } }),
      register: (p) => http.json('POST', '/v1/models/register', { body: { path: p } }),
      discovered: () => http.json('GET', '/v1/models/discovered'),
      link: (p, id) => http.json('POST', '/v1/models/link', { body: { path: p, id } }),
      hardware: ({ refresh = false } = {}) => http.json('GET', '/v1/hardware', { params: { refresh: refresh ? 'true' : undefined } }),
      storage: () => http.json('GET', '/v1/models/storage'),
      providers: async () => (await http.json('GET', '/v1/providers')).providers,
      addProvider: ({ baseUrl, model, apiKey = '', label }) => http.json('POST', '/v1/providers', { body: { base_url: baseUrl, model, api_key: apiKey, label } }),
    }
    this.orcha = {
      run: async (task, { graph = 'default', wait = false, timeoutMs = 900000, ...options } = {}) => {
        const job = new Job(c, (await http.json('POST', '/v1/orcha/run', { body: { task, graph, ...options } })).job)
        return wait ? job.wait({ timeoutMs }) : job
      },
      status: () => http.json('GET', '/v1/orcha/status'),
      jobs: async ({ state, limit = 50 } = {}) => (await http.json('GET', '/v1/jobs', { params: { state, limit: 200 } })).jobs.filter((j) => ['orcha.run', 'task', 'agent.run'].includes(j.kind)).slice(0, limit),
      cancel: async (id) => new Job(c, (await http.json('POST', `/v1/jobs/${id}/cancel`)).job),
    }
    this.jobs = {
      get: async (id) => new Job(c, (await http.json('GET', `/v1/jobs/${id}`)).job),
      list: async ({ state, kind, limit = 50 } = {}) => (await http.json('GET', '/v1/jobs', { params: { state, kind, limit } })).jobs,
      cancel: async (id) => new Job(c, (await http.json('POST', `/v1/jobs/${id}/cancel`)).job),
    }
    this.memory = {
      store: (content, o = {}) => http.json('POST', '/v1/memory', { body: { content, title: o.title, type: o.type, tags: o.tags, importance: o.importance, scope: o.scope, workspace: o.workspace, extra: o.extra } }),
      search: async (q, { limit = 10, scope, workspace, tags, type } = {}) => (await http.json('GET', '/v1/memory/search', { params: { q, limit, scope, workspace, tag: tags, type } })).items,
      list: async ({ limit = 20, ...f } = {}) => (await http.json('GET', '/v1/memory', { params: { limit, ...f } })).items,
      get: (id) => http.json('GET', `/v1/memory/${id}`),
      delete: (id) => http.json('DELETE', `/v1/memory/${id}`),
    }
    this.context = {
      put: (collection, docId, text, { title, metadata } = {}) => http.json('PUT', `/v1/context/${collection}/documents/${docId}`, { body: { text, title, metadata } }),
      search: async (collection, query, { limit = 8, docIds } = {}) => (await http.json('POST', `/v1/context/${collection}/search`, { body: { query, limit, doc_ids: docIds } })).items,
      collections: async () => (await http.json('GET', '/v1/context')).collections,
      documents: async (collection) => (await http.json('GET', `/v1/context/${collection}/documents`)).documents,
      delete: (collection, docId) => http.json('DELETE', docId ? `/v1/context/${collection}/documents/${docId}` : `/v1/context/${collection}`),
    }
    // Shared resources: private by default, shared with named apps by the owner, referenced (`runtime://res_...`), never copied.
    const clean = (o) => Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined))
    this.resources = {
      create: (title, o = {}) => http.json('POST', '/v1/resources', { body: clean({
        type: o.type || 'document', title, kind: o.collection ? 'context' : o.text !== undefined ? 'text' : 'file',
        collection: o.collection, doc_ids: o.docIds, text: o.text, path: o.path, workspace: o.workspace, metadata: o.metadata }) }),
      list: async (o = {}) => (await http.json('GET', '/v1/resources', { params: { type: o.type, workspace: o.workspace, owned: o.owned ? 'true' : undefined } })).resources,
      get: (id) => http.json('GET', `/v1/resources/${id}`),
      update: (id, o = {}) => http.json('PATCH', `/v1/resources/${id}`, { body: clean({ title: o.title, metadata: o.metadata, workspace: o.workspace, text: o.text }) }),
      resolve: (ref) => http.json('GET', '/v1/resources/resolve', { params: { ref } }),
      read: (id, doc) => http.json('GET', `/v1/resources/${id}/read`, { params: { doc } }),
      write: (id, docId, text, { title } = {}) => http.json('PUT', `/v1/resources/${id}/documents/${docId}`, { body: clean({ text, title }) }),
      search: (query, o = {}) => http.json('POST', '/v1/resources/search', { body: clean({ query, resources: o.resources, workspace: o.workspace, types: o.types, limit: o.limit }) }),
      share: (id, apps, { access = 'read' } = {}) => http.json('POST', `/v1/resources/${id}/share`, { body: { with: apps, access } }),
      revoke: (id, apps) => http.json('POST', `/v1/resources/${id}/revoke`, { body: clean({ with: apps }) }),
      permissions: (id) => http.json('GET', `/v1/resources/${id}/permissions`),
      requestAccess: (id, { access = 'read', reason = '' } = {}) => http.json('POST', `/v1/resources/${id}/request`, { body: { access, reason } }),
      requests: async (state = 'pending') => (await http.json('GET', '/v1/resources/requests', { params: { state } })).requests,
      decide: (requestId, approve) => http.json('POST', `/v1/resources/requests/${requestId}/${approve ? 'approve' : 'deny'}`),
      delete: (id, { purge = false } = {}) => http.json('DELETE', `/v1/resources/${id}`, { params: { purge: purge ? 'true' : undefined } }),
      audit: async (limit = 50, resource) => (await http.json('GET', '/v1/resources/audit', { params: { limit, resource } })).events,
      workspaces: async () => (await http.json('GET', '/v1/workspaces')).workspaces,
      createWorkspace: (name) => http.json('POST', '/v1/workspaces', { body: { name } }),
      shareWorkspace: (name, apps, { access = 'read' } = {}) => http.json('POST', `/v1/workspaces/${name}/share`, { body: { with: apps, access } }),
    }
  }

  static detect(env = process.env) { return detect(env) }

  /**
   * Detect -> (install with consent) -> start -> authenticate -> verify compatibility.
   * `install(info)` is called when the runtime is missing; resolve `true` only after the USER agreed.
   * Without it a missing runtime throws RuntimeNotInstalled — nothing is ever installed silently.
   */
  static async connect({ appId, name, permissions = [], requireApi = SDK_API_VERSION, minVersion, autoStart = true, install, source, env = process.env, timeoutMs = 30000, onStatus, keepAlive = true, idleGraceS } = {}) {
    if (!appId) throw new AnviraError('invalid_request', 'connect({ appId }) is required.')
    const say = onStatus || (() => {})
    let info = await detect(env)
    if (!info.running) {
      if (!info.installed) {
        if (!install) throw new RuntimeNotInstalled('runtime_not_installed', 'Anvira Runtime is required but not installed.', { hint: 'Ask the user, then install it (see INSTALLATION.md) or pass install: async (info) => boolean to connect().' })
        // The callback returns: false (declined) | true (install with defaults) | { action: 'use-existing', path } | { action: 'download', dest?, gpu? }
        const choice = await install(info)
        if (!choice) throw new InstallDeclined('install_declined', 'Anvira Runtime is required. Installation was declined.')
        if (choice.action === 'use-existing') { registerLocation(choice.path, env); say(`Using the Anvira Runtime at ${choice.path}.`) }
        else if (choice.action === 'download' || (!source && !info.install.entry)) {
          await installFromGithub({ dest: choice.dest, gpu: choice.gpu, confirmGpu: choice.confirmGpu, repo: choice.repo, env, onStatus: say, onProgress: choice.onProgress })
        } else {
          say('Installing Anvira Runtime...')
          await installRuntime({ source, env, onStatus: say })
        }
        info = await detect(env)
      }
      if (!info.running) {
        if (!autoStart) throw new RuntimeNotRunning('runtime_not_running', 'Anvira Runtime is installed but not running.', { hint: 'Run `anvira runtime start`.' })
        say('Starting runtime...')
        info = await startRuntime({ env, onStatus: say, autoStop: true, idleGraceS })
      }
    }
    if (info.running && !info.ready) { say('Waiting for the runtime to finish starting...'); info = await waitReady(env, 120000, say) }
    if (info.apiVersion !== requireApi) {
      throw new IncompatibleRuntime('incompatible_runtime', `This app needs Runtime API v${requireApi} but the installed runtime speaks v${info.apiVersion} (runtime ${info.runtimeVersion}).`,
        { hint: (info.apiVersion || 0) < requireApi ? 'Update the runtime: `anvira runtime update`.' : 'Update this application to a version that supports the newer runtime API.' })
    }
    if (minVersion && !semverGte(info.runtimeVersion, minVersion)) {
      throw new IncompatibleRuntime('incompatible_runtime', `This app needs Anvira Runtime >= ${minVersion}; found ${info.runtimeVersion}.`, { hint: 'Update the runtime: `anvira runtime update`.' })
    }
    const baseUrl = `http://${info.host}:${info.port}`
    const dirs = runtimeDirs(env)
    const tokenFile = path.join(dirs.state, 'app-tokens', `${appId}.token`)
    let token
    if (fs.existsSync(tokenFile)) token = fs.readFileSync(tokenFile, 'utf8').trim()
    else {
      let reg
      try { reg = fs.readFileSync(path.join(dirs.state, 'register.token'), 'utf8').trim() } catch {
        throw new AnviraError('no_registration_token', 'Cannot register with the runtime (registration token unreadable).', { hint: 'Run the app as the same OS user that installed the runtime.' })
      }
      const out = await new Http(baseUrl, null, timeoutMs).json('POST', '/v1/apps/register', { body: { app_id: appId, name, permissions }, headers: { 'X-Anvira-Register-Token': reg } })
      fs.mkdirSync(path.dirname(tokenFile), { recursive: true })
      fs.writeFileSync(tokenFile, out.token, { mode: 0o600 })
      token = out.token
    }
    const rt = new AnviraRuntime(new Http(baseUrl, token, timeoutMs), info, appId)
    rt._opts = { env, onStatus: say, autoStart, idleGraceS }
    rt._http.recover = () => rt._recover()
    if (keepAlive) await rt._startLease()
    return rt
  }

  /** Tell the runtime this app is open and keep telling it until close(). The timer never keeps Node alive. */
  async _startLease() {
    await this._acquireLease()
    this._leasePid = this.info.pid
    this._timer = setInterval(async () => {
      const lid = this._lease.id
      try { await this._http.json('POST', `/v1/leases/${lid}/heartbeat`, { timeoutMs: 10000 }) }
      catch (e) { if (e && e.code === 'lease_not_found' && this._lease.id === lid) { try { await this._acquireLease() } catch { /* retried next beat */ } } }
    }, Math.max(1000, ((this._lease && this._lease.ttl_s) || 45) / 3 * 1000))
    if (this._timer.unref) this._timer.unref()
  }
  async _acquireLease() {
    const out = await this._http.json('POST', '/v1/leases', { body: {}, timeoutMs: 10000 })
    this._lease = { ...out.lease, mode: out.mode }
  }
  async _recover() {
    if (this._closed || !this._opts.autoStart) return null
    if (this._recovering) return this._recovering
    this._recovering = (async () => {
      try {
        let info = await detect(this._opts.env)
        if (!(info.running && info.ready)) {
          this._opts.onStatus && this._opts.onStatus('Starting runtime...')
          info = await startRuntime({ env: this._opts.env, onStatus: this._opts.onStatus, autoStop: true, idleGraceS: this._opts.idleGraceS })
        }
        this.info = info
        const base = `http://${info.host}:${info.port}`
        if (this._lease && info.pid !== this._leasePid) {
          this._leasePid = info.pid
          try { const o = await new Http(base, this._http.token, 10000).json('POST', '/v1/leases', { body: {} }); this._lease = { ...o.lease, mode: o.mode } } catch { /* next heartbeat re-acquires */ }
        }
        return base
      } finally { this._recovering = null }
    })()
    return this._recovering
  }
  /** This app is closing: release the lease so an on-demand runtime can stop when nobody else needs it. */
  async close() {
    if (this._closed) return
    this._closed = true
    if (this._timer) clearInterval(this._timer)
    if (this._lease) { try { await new Http(this._http.baseUrl, this._http.token, 5000).json('DELETE', `/v1/leases/${this._lease.id}`, { timeoutMs: 5000 }) } catch { /* already gone */ } }
  }
  /** Mode (on-demand / persistent), open leases and the idle countdown. */
  lifecycle() { return this._http.json('GET', '/v1/lifecycle') }

  health() { return this._http.json('GET', '/health') }
  version() { return this._http.json('GET', '/version') }
  status() { return this._http.json('GET', '/v1/status') }
  /** This app's identity and granted permissions. */
  me() { return this._http.json('GET', '/v1/apps/me') }
  hasCapability(name) { return this.info.capabilities.includes(name) }

  /** Non-streaming chat returns the OpenAI-style response; `stream: true` returns an async iterator of text deltas. */
  chat(messages, { stream = false, memory, ...options } = {}) {
    const body = { messages, stream, ...(memory ? { memory } : {}), ...options }
    if (!stream) return this._http.json('POST', '/v1/chat', { body, timeoutMs: 660000 })
    return this._streamChat(body)
  }
  async *_streamChat(body) {
    for await (const { event, data } of this._http.sse('POST', '/v1/chat', { body })) {
      if (event === 'error') throw AnviraError.fromResponse(502, JSON.parse(data))
      if (event !== 'message' || data.trim() === '[DONE]') continue
      let delta
      try { delta = JSON.parse(data).choices[0].delta.content } catch { continue }
      if (delta) yield delta
    }
  }
  async chatText(messages, options = {}) { return (await this.chat(messages, options)).choices[0].message.content }

  async task(task, { wait = true, ...options } = {}) {
    return new Job(this, (await this._http.json('POST', '/v1/task', { body: { task, wait, ...options }, timeoutMs: ((options.timeoutS || 900) + 30) * 1000 })).job)
  }
  async agentRun(task, { workspaceRoots, wait = false, ...options } = {}) {
    const job = new Job(this, (await this._http.json('POST', '/v1/agent/run', { body: { task, workspace_roots: workspaceRoots, ...options } })).job)
    return wait ? job.wait() : job
  }
}

export default AnviraRuntime
