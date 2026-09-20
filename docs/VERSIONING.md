# Versioning & compatibility

Applications depend on the **runtime API version**, never on ORCHA, Nomi or AICL versions.

| number | where | meaning |
|---|---|---|
| `runtime_version` (SemVer, `1.0.0`) | `GET /version`, `install.json`, `anvira --version` | the distribution: daemon + CLI + API |
| `api_version` (integer, `1`) | `GET /version`, every route is under `/v1` | **breaking** changes only. An app written for v1 keeps working on any runtime that still serves `/v1`. |
| `api_revision` (integer, `0`) | `GET /version` | additive changes within a major (new endpoints/fields) |
| `capabilities[]` | `GET /version` | feature flags (`chat.stream`, `memory.namespaces`, `models.register`…) — feature-detect instead of comparing versions |

Internal versions (ORCHA `0.4.0`, Nomi `0.2.0`, AICL `0.1.0`) are visible only in `anvira status`/`anvira aicl status` for diagnostics.

## How an app declares what it needs

```ts
AnviraRuntime.connect({ appId, requireApi: 1, minVersion: '1.0.0' })
```

* `requireApi` must equal the runtime's `api_version`. Older runtime → `IncompatibleRuntime` with *"Update the runtime: `anvira runtime update`"*;
  newer runtime the app does not know → *"Update this application"*.
* `minVersion` (optional) is a `>=` check on `runtime_version` for apps that need a specific fix.
* `hasCapability('memory.namespaces')` for optional features.

## Rules the runtime follows

* Adding endpoints, response fields or optional request fields is **not** breaking (`api_revision` bumps).
* Removing/renaming a field, changing a status/`code` meaning, or tightening validation of previously valid input **is** breaking → new
  `/v2` served alongside `/v1` for at least one runtime major; `/version` keeps `min_client_api`.
* Error `code`s are part of the contract; `message` and `hint` text are not.
* The CLI mirrors the API version; `anvira doctor` reports a CLI/runtime API mismatch.
* SDKs are versioned independently (`anvira-client 1.0.0`, `@anvira/runtime-client 1.0.0`) and tested against the runtime API in this repo.
* Config keys are additive; unknown keys are rejected on write so typos are caught, unknown persisted keys are ignored on read.
* Upgrades keep `state/` (tokens, apps, model registry), `config/`, `data/` (Nomi, context index) and the model library.

## Internal compatibility

ORCHA and Nomi are launched through their own `desktop_entry.py` and public HTTP APIs, so upgrading them is a runtime change invisible to
apps. AICL is loaded from `services/aicl` (or `ANVIRA_AICL_DIR`); if it cannot be imported the runtime reports `aicl_unavailable` and `anvira doctor` fails —
apps keep the same API.
