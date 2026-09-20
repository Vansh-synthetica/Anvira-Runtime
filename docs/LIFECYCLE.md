# Runtime lifecycle: active only while an app is open

The runtime is a shared background service, but it should not sit in memory when nothing uses it. It runs **on demand**:

```
app opens ──► SDK finds the runtime (or starts it with auto_stop) ──► POST /v1/leases   (heartbeat every ~15 s)
app closes ──► DELETE /v1/leases/{id}
last lease gone + no work running + idle grace (30 s) ──► runtime stops ORCHA, Nomi, the model, then itself
```

## Rules

* A **lease** says "an app is open". It expires after `ttl_s` (default 45 s) without a heartbeat, so a **crashed app** cannot keep
  the runtime alive.
* Only runtimes started **on demand** (`auto_stop`) stop themselves. `anvira runtime start` is persistent (a service you chose to
  keep running) unless you pass `--auto-stop`. A runtime an app started for itself is always `auto_stop`.
* **Running work blocks shutdown**: ORCHA jobs, open event/chat streams, a model load and downloads all count as busy. The idle
  clock starts only when nothing is running and no lease is held.
* The CLI holds a lease while a command runs (and while `anvira ui` is open), and autostarts with a 120 s grace so a follow-up
  command finds it warm.
* If the runtime stops underneath an app that is still open, the SDK **recovers transparently**: it restarts the runtime,
  re-acquires its lease and retries the request.
* Nothing is loaded until asked: the context index and resource database open on first use; a model starts on `models.use`.

## API

| | |
|---|---|
| `POST /v1/leases` `{ttl_s?}` → `{lease, heartbeat_every_s, mode}` | an app is open |
| `POST /v1/leases/{id}/heartbeat` | keep it alive (`lease_not_found` 404 if it expired → re-acquire) |
| `DELETE /v1/leases/{id}` | app closed |
| `GET /v1/lifecycle` | `{mode: on-demand\|persistent, auto_stop, idle_grace_s, lease_ttl_s, leases[], busy, idle_for_s, shutdown_in_s}` |

Config: `lifecycle.auto_stop`, `lifecycle.idle_grace_s` (30), `lifecycle.lease_ttl_s` (45).
Daemon flags: `python -m anvira_runtime --auto-stop --idle-grace 30`.

## SDK

```python
app = AnviraRuntime.connect("anvira-notes", keep_alive=True)   # starts it if needed, holds a lease
...
app.close()                                                     # releases the lease; runtime may now stop
```
```ts
const rt = await AnviraRuntime.connect('anvira-notes', { keepAlive: true, idleGraceS: 30 })
await rt.close()
```
