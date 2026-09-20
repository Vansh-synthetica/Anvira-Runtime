// Standalone: custom install locations and GitHub-release installs against a local fake GitHub. No runtime needed.
import test from 'node:test'
import assert from 'node:assert/strict'
import crypto from 'node:crypto'
import fs from 'node:fs'
import http from 'node:http'
import os from 'node:os'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import {
  AnviraError, locate, registerLocation, runtimeDirs, defaultHome, installPackage, installFromGithub, downloadAsset,
  updateInProgress, waitForUpdate, detect,
} from '../index.js'

const tmp = (p) => fs.mkdtempSync(path.join(os.tmpdir(), `anvira-ts-${p}-`))
const py = process.env.ANVIRA_TEST_PYTHON || (process.platform === 'win32' ? 'python' : 'python3')

function makeZip(file, { version = '9.9.9', withPython = true, extra = {} } = {}) {
  const script = `
import json, sys, zipfile
out, version, with_python, extra = sys.argv[1], sys.argv[2], sys.argv[3] == '1', json.loads(sys.argv[4])
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    z.writestr('AnviraRuntime/install.json', json.dumps({'version': version, 'portable': True, 'entry': 'python/python.exe', 'services_dir': 'services'}))
    if with_python: z.writestr('AnviraRuntime/python/python.exe', b'MZ fake')
    z.writestr('AnviraRuntime/services/orcha/desktop_entry.py', 'print(1)')
    for k, v in extra.items(): z.writestr(k, v)
`
  const r = spawnSync(py, ['-c', script, file, version, withPython ? '1' : '0', JSON.stringify(extra)], { encoding: 'utf8' })
  assert.equal(r.status, 0, r.stderr)
  return file
}

function envFor(dir) {
  return { ...process.env, LOCALAPPDATA: path.join(dir, 'appdata'), APPDATA: path.join(dir, 'appdata'), USERPROFILE: dir, HOME: dir, XDG_DATA_HOME: path.join(dir, 'xdg'), ANVIRA_RUNTIME_HOME: '' }
}

async function fakeGithub(assets, { badSumFor, tag = 'v9.9.9' } = {}) {
  const files = { ...assets }
  const sums = Object.fromEntries(Object.entries(assets).map(([n, d]) => [n, crypto.createHash('sha256').update(d).digest('hex')]))
  if (badSumFor) sums[badSumFor] = '0'.repeat(64)
  files.SHA256SUMS = Buffer.from(Object.entries(sums).map(([n, h]) => `${h}  ${n}\n`).join(''))
  const hits = []
  let base = ''
  const server = http.createServer((req, res) => {
    hits.push(req.url)
    const rel = req.url.match(/^\/repos\/([^/]+\/[^/]+)\/releases\/(latest|tags\/.+)$/)
    if (rel) {
      if (rel[1] !== 'acme/Anvira-Runtime') { res.writeHead(404); return res.end() }
      const body = JSON.stringify({ tag_name: tag, assets: Object.entries(files).map(([name, d]) => ({ name, size: d.length, browser_download_url: `${base}/dl/${name}` })) })
      res.writeHead(200, { 'content-length': Buffer.byteLength(body) })
      return res.end(body)
    }
    const dl = req.url.match(/^\/dl\/(.+)$/)
    if (dl && files[dl[1]]) {
      const data = files[dl[1]]
      const range = /bytes=(\d+)-/.exec(req.headers.range || '')
      const start = range ? Number(range[1]) : 0
      res.writeHead(range ? 206 : 200, { 'content-length': data.length - start })
      return res.end(data.subarray(start))
    }
    res.writeHead(404); res.end()
  })
  await new Promise((r) => server.listen(0, '127.0.0.1', r))
  base = `http://127.0.0.1:${server.address().port}`
  return { base, hits, close: () => new Promise((r) => server.close(r)) }
}

test('a custom location is remembered and found by every client', async () => {
  const dir = tmp('loc'); const env = envFor(dir)
  const custom = path.join(dir, 'D-drive', 'My Anvira Runtime')
  const rec = await installPackage(makeZip(path.join(dir, 'pkg.zip')), { dest: custom, env })
  assert.equal(rec.version, '9.9.9')
  assert.ok(fs.existsSync(path.join(custom, 'python', 'python.exe')) && fs.existsSync(path.join(custom, 'services', 'orcha')))
  const loc = await locate(env)
  assert.deepEqual([loc.found, loc.source, path.resolve(loc.home), loc.version], [true, 'pointer', path.resolve(custom), '9.9.9'])
  assert.equal(path.resolve(runtimeDirs(env).state), path.resolve(custom, 'state'))
  assert.notEqual(path.resolve(defaultHome(env)), path.resolve(custom))
  assert.equal((await detect(env)).installed, true) // relative "entry" resolved against home
})

test('ANVIRA_RUNTIME_HOME beats the pointer, and a dead pointer falls back to the default', async () => {
  const dir = tmp('env'); const env = envFor(dir)
  const custom = path.join(dir, 'custom')
  await installPackage(makeZip(path.join(dir, 'pkg.zip')), { dest: custom, env })
  assert.equal(runtimeDirs({ ...env, ANVIRA_RUNTIME_HOME: path.join(dir, 'other') }).home, path.join(dir, 'other'))
  fs.rmSync(custom, { recursive: true, force: true })
  const loc = await locate(env)
  assert.equal(path.resolve(loc.home), path.resolve(defaultHome(env))); assert.equal(loc.found, false)
})

test('"I already have it" registers without copying and rejects non-runtime folders', async () => {
  const dir = tmp('have'); const env = envFor(dir)
  const existing = path.join(dir, 'already-here')
  await installPackage(makeZip(path.join(dir, 'pkg.zip')), { dest: existing, env: { ...env, LOCALAPPDATA: path.join(dir, 'other-appdata') } })
  assert.equal(registerLocation(existing, env), path.resolve(existing))
  assert.equal((await locate(env)).source, 'pointer')
  assert.equal(fs.existsSync(path.join(defaultHome(env), 'python')), false)
  fs.mkdirSync(path.join(dir, 'junk'))
  assert.throws(() => registerLocation(path.join(dir, 'junk'), env), (e) => e instanceof AnviraError && e.code === 'not_a_runtime_folder')
})

test('install from GitHub verifies checksums, reports progress, and never fetches the GPU pack without consent', async () => {
  const dir = tmp('gh'); const env = { ...envFor(dir), ANVIRA_RUNTIME_REPO: 'acme/Anvira-Runtime' }
  const core = fs.readFileSync(makeZip(path.join(dir, 'core.zip')))
  const cuda = fs.readFileSync(makeZip(path.join(dir, 'cuda.zip'), { withPython: false, extra: { 'AnviraRuntime/bin/llama-cpp/cuda/llama-server.exe': 'cuda-bin' } }))
  const gh = await fakeGithub({ 'AnviraRuntime-9.9.9-win-x64.zip': core, 'AnviraRuntime-9.9.9-win-x64-cuda.zip': cuda })
  env.ANVIRA_RELEASE_API = gh.base
  try {
    const dest = path.join(dir, 'E-drive', 'Anvira Runtime'); const said = []; let last = null
    const rec = await installFromGithub({ dest, env, onStatus: (m) => said.push(m), onProgress: (n, d, t) => { last = [n, d, t] }, confirmGpu: async () => false })
    assert.equal(rec.version, '9.9.9'); assert.ok(fs.existsSync(path.join(dest, 'python', 'python.exe')))
    assert.equal(last[1], last[2]); assert.equal(path.resolve((await locate(env)).home), path.resolve(dest))
    assert.ok(!fs.existsSync(path.join(dest, '.download')) && !fs.existsSync(path.join(dest, '.updating')))
    if (rec.gpu_pack_available) assert.equal(rec.gpu_pack, false) // declined: no GPU pack bytes
    assert.ok(!fs.existsSync(path.join(dest, 'bin', 'llama-cpp', 'cuda')))
    const n = gh.hits.length
    const again = await installFromGithub({ dest, env })
    assert.equal(again.unchanged, true); assert.ok(!gh.hits.slice(n).some((h) => h.startsWith('/dl/AnviraRuntime-9.9.9-win-x64.zip')))
    if (process.platform === 'win32' || true) {
      const gpu = await installFromGithub({ dest: path.join(dir, 'gpu'), env, gpu: true, force: true })
      if (gpu.gpu_pack_available && gpu.gpu_pack_available.cudaOk) assert.equal(fs.readFileSync(path.join(dir, 'gpu', 'bin', 'llama-cpp', 'cuda', 'llama-server.exe'), 'utf8'), 'cuda-bin')
    }
  } finally { await gh.close() }
})

test('a corrupt download is discarded and nothing is installed', async () => {
  const dir = tmp('bad'); const env = { ...envFor(dir), ANVIRA_RUNTIME_REPO: 'acme/Anvira-Runtime' }
  const core = fs.readFileSync(makeZip(path.join(dir, 'core.zip')))
  const gh = await fakeGithub({ 'AnviraRuntime-9.9.9-win-x64.zip': core }, { badSumFor: 'AnviraRuntime-9.9.9-win-x64.zip' })
  env.ANVIRA_RELEASE_API = gh.base
  try {
    await assert.rejects(installFromGithub({ dest: path.join(dir, 'dest'), env }), (e) => e.code === 'checksum_mismatch')
    assert.ok(!fs.existsSync(path.join(dir, 'dest', 'python')) && !fs.existsSync(path.join(dir, 'dest', '.updating')))
    await assert.rejects(installFromGithub({ dest: path.join(dir, 'dest'), env, repo: 'acme/nope' }), (e) => e.code === 'release_not_found')
    await assert.rejects(installFromGithub({ dest: path.join(dir, 'dest'), env: { ...env, ANVIRA_RUNTIME_REPO: 'OWNER/Anvira-Runtime' } }), (e) => e.code === 'repo_not_configured')
  } finally { await gh.close() }
})

test('an interrupted download resumes', async () => {
  const dir = tmp('resume'); const data = fs.readFileSync(makeZip(path.join(dir, 'core.zip')))
  const gh = await fakeGithub({ 'AnviraRuntime-9.9.9-win-x64.zip': data })
  try {
    const work = path.join(dir, 'w'); fs.mkdirSync(work)
    fs.writeFileSync(path.join(work, 'AnviraRuntime-9.9.9-win-x64.zip.part'), data.subarray(0, 40))
    const file = await downloadAsset({ name: 'AnviraRuntime-9.9.9-win-x64.zip', size: data.length, browser_download_url: `${gh.base}/dl/AnviraRuntime-9.9.9-win-x64.zip` }, work, crypto.createHash('sha256').update(data).digest('hex'))
    assert.deepEqual(fs.readFileSync(file), data)
  } finally { await gh.close() }
})

test('the update lock holds restarts back until it is released, and a stale lock is ignored', async () => {
  const dir = tmp('lock'); const env = { ...envFor(dir), ANVIRA_RUNTIME_HOME: dir }
  fs.writeFileSync(path.join(dir, '.updating'), '1')
  assert.equal(updateInProgress(env), true)
  setTimeout(() => fs.rmSync(path.join(dir, '.updating')), 800)
  const t0 = Date.now(); await waitForUpdate(env, 10000)
  assert.ok(Date.now() - t0 >= 500 && !updateInProgress(env))
  fs.writeFileSync(path.join(dir, '.updating'), 'x'); const old = new Date(Date.now() - 3600 * 1000); fs.utimesSync(path.join(dir, '.updating'), old, old)
  assert.equal(updateInProgress(env), false)
})
