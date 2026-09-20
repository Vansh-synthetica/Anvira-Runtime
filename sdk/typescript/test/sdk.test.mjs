// Runs against a REAL runtime started by tests/runtime/test_08_typescript_sdk.py (env: ANVIRA_RUNTIME_HOME, ANVIRA_TEST_*).
import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import {
  AnviraRuntime, AnviraError, RuntimeNotInstalled, InstallDeclined, IncompatibleRuntime, PermissionDenied,
  detect, runtimeDirs, Job,
} from '../index.js'

const providerId = process.env.ANVIRA_TEST_PROVIDER

test('runtimeDirs matches the Python implementation on every platform', () => {
  const expected = JSON.parse(process.env.ANVIRA_TEST_DIRS)
  for (const c of expected) {
    const got = runtimeDirs(c.env, c.platform)
    for (const k of ['home', 'state', 'config', 'logs', 'models', 'bin']) {
      assert.equal(path.normalize(got[k]), path.normalize(c.dirs[k]), `${c.platform}.${k}`)
    }
  }
})

test('detect reports an installed, running, ready runtime and never throws', async () => {
  const info = await AnviraRuntime.detect()
  assert.equal(info.installed, true)
  assert.equal(info.running, true)
  assert.equal(info.ready, true)
  assert.equal(info.apiVersion, 1)
  assert.ok(info.capabilities.includes('memory.search'))
  const none = await detect({ ANVIRA_RUNTIME_HOME: path.join(os.tmpdir(), 'anvira-nothing-' + Date.now()) })
  assert.equal(none.installed, false)
  assert.equal(none.running, false)
})

test('a missing runtime is never installed without consent', async () => {
  const home = path.join(os.tmpdir(), 'anvira-ts-missing-' + Date.now())
  const env = { ...process.env, ANVIRA_RUNTIME_HOME: home }
  await assert.rejects(AnviraRuntime.connect({ appId: 'anvira-notes', env }), (e) => e instanceof RuntimeNotInstalled && e.code === 'runtime_not_installed')
  let asked = 0
  await assert.rejects(AnviraRuntime.connect({ appId: 'anvira-notes', env, install: async () => { asked++; return false } }),
    (e) => e instanceof InstallDeclined && e.code === 'install_declined')
  assert.equal(asked, 1)
  assert.equal(fs.existsSync(home), false)
})

test('connect + models + chat + streaming', async () => {
  const rt = await AnviraRuntime.connect({ appId: 'ts-notes', name: 'TS Notes', install: async () => { throw new Error('must not be asked') } })
  assert.equal((await rt.health()).status, 'ok')
  assert.ok(rt.hasCapability('chat.stream'))
  const models = await rt.models.installed()
  assert.ok(models.some((m) => m.id === providerId))
  const active = await rt.models.use(providerId)
  assert.equal(active.active, providerId)
  assert.match(await rt.chatText([{ role: 'user', content: 'hello from node' }]), /^fake reply to: hello from node$/)
  let text = ''
  for await (const d of rt.chat([{ role: 'user', content: 'hi' }], { stream: true })) text += d
  assert.equal(text, 'fake stream reply')
  const me = await rt.me()
  assert.equal(me.app_id, 'ts-notes')
  assert.ok(me.permissions.includes('chat') && !me.permissions.includes('models.manage'))
})

test('errors are typed and structured', async () => {
  const rt = await AnviraRuntime.connect({ appId: 'ts-notes' })
  await assert.rejects(rt.models.install('acme/tiny-GGUF'), (e) => e instanceof PermissionDenied && e.code === 'permission_denied' && /anvira app grant/.test(e.message))
  await assert.rejects(rt.models.use('ghost'), (e) => e instanceof AnviraError && e.code === 'model_not_found' && e.status === 404)
  await assert.rejects(rt.chat([]), (e) => e.code === 'invalid_request')
  await assert.rejects(AnviraRuntime.connect({ appId: 'ts-notes', requireApi: 2 }), (e) => e instanceof IncompatibleRuntime)
  await assert.rejects(AnviraRuntime.connect({ appId: 'ts-notes', minVersion: '9.0.0' }), (e) => e instanceof IncompatibleRuntime)
})

test('ORCHA jobs', async () => {
  const rt = await AnviraRuntime.connect({ appId: 'ts-notes' })
  await rt.models.use(providerId)
  const job = await rt.orcha.run('What is a queue?', { wait: true })
  assert.ok(job instanceof Job)
  assert.equal(job.state, 'completed')
  const res = await job.unwrap()
  assert.match(res.answer, /fake reply/)
  assert.ok((await rt.orcha.jobs()).some((j) => j.id === job.id))
  const t = await rt.task('Define stack.')
  assert.equal(t.state, 'completed')
  await assert.rejects(rt.jobs.get('job_nope'), (e) => e.code === 'job_not_found')
})

test('memory and context are per-app', async () => {
  const rt = await AnviraRuntime.connect({ appId: 'ts-notes' })
  const other = await AnviraRuntime.connect({ appId: 'ts-study' })
  const m = await rt.memory.store('Node app remembers the moon landing', { title: 'Moon', tags: ['space'] })
  assert.equal(m.app, 'ts-notes')
  assert.equal((await rt.memory.search('moon landing'))[0].id, m.id)
  assert.deepEqual(await other.memory.search('moon landing'), [])
  await rt.context.put('c1', 'd1', 'Mitochondria produce ATP.', { title: 'Bio' })
  assert.equal((await rt.context.search('c1', 'ATP mitochondria'))[0].doc_id, 'd1')
  assert.deepEqual(await other.context.search('c1', 'ATP'), [])
})

test('an app can announce a model location', async () => {
  const rt = await AnviraRuntime.connect({ appId: 'ts-notes' })
  const f = path.join(os.tmpdir(), `ts-announced-${Date.now()}-Q4_K_M.gguf`)
  fs.writeFileSync(f, Buffer.concat([Buffer.from('GGUF'), Buffer.alloc(100)]))
  const res = await rt.models.register(f)
  assert.equal(res.kind, 'file')
  assert.ok((await rt.models.installed()).some((m) => m.id === res.model.id))
  fs.unlinkSync(f)
})

test('resources are private until shared, then referenced, never copied', async () => {
  const notes = await AnviraRuntime.connect({ appId: 'ts-notes' })
  const study = await AnviraRuntime.connect({ appId: 'ts-study' })
  await notes.context.put('res-bio', 'd1', 'The Krebs cycle oxidises acetyl-CoA in the mitochondrial matrix.', { title: 'Krebs' })
  const res = await notes.resources.create('TS notebook', { type: 'notebook', collection: 'res-bio' })
  assert.equal(res.visibility, 'private')
  assert.match(res.ref, /^runtime:\/\/res_/)
  assert.deepEqual(await study.resources.list(), [])
  await assert.rejects(study.resources.get(res.id), (e) => e instanceof AnviraError && e.code === 'resource_not_found')
  const shared = await notes.resources.share(res.id, ['ts-study'])
  assert.equal(shared.visibility, 'shared')
  const hit = (await study.resources.search('Krebs acetyl-CoA')).items[0]
  assert.equal(hit.resource, res.id)
  assert.equal((await study.resources.resolve(res.ref)).id, res.id)
  await assert.rejects(study.resources.write(res.id, 'x', 'nope'), (e) => e.code === 'access_denied')
  await assert.rejects(notes.resources.share(res.id, ['*']), (e) => e.code === 'user_approval_required')
  await notes.resources.revoke(res.id)
  await assert.rejects(study.resources.get(res.id), (e) => e.code === 'resource_not_found')
  await notes.resources.delete(res.id)
})
