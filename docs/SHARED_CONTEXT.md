# Shared data, context and capabilities

The runtime owns the **mechanism** (records, grants, references, audit). **The user owns the permission decision.** Apps never see
each other's data unless it is explicitly shared, and the runtime never becomes a second app.

## Resources

A *resource* is a small record — owner app, type, title, workspace, metadata — that **points at** content instead of copying it:

| kind | points at | notes |
|---|---|---|
| `context` | a collection in the owner's context index | readable and (with `write`) writable through the same collection — one copy of the data |
| `file` | a text-like file (≤ 5 MB) | read on demand; follows edits; a deleted file is `resource_unavailable`, never a stale copy |
| `text` | a short inline text (≤ 200 000 chars) | for prompts, guides, small notes |

References look like `runtime://res_<12 hex>[/<document>]` and resolve only for apps allowed to see them.

## Visibility

| level | who | how |
|---|---|---|
| **private** (default) | the owner app (and the user via the CLI) | — |
| **shared** | named apps, `read` or `write` | the owner (`context.share`) or the user |
| **global** | every app, read-only | **the user only** (`anvira context share <id> --global`, asks first). An app gets `user_approval_required`. The owner may always narrow access again |

An app with no access gets `resource_not_found` — it cannot learn a private resource exists. Grantees see the resource but not its
content pointer or its grant list. Only the owner changes title/metadata, shares onward or deletes.

**Access requests:** an app calls `request_access(id, access, reason)`; the owner app or the user approves or denies
(`anvira context requests|approve|deny`). Requesting grants nothing.

**Workspaces** group resources by project. A workspace name alone reveals nothing; `share_workspace` shares every resource *you own* in it.

**Audit:** create / share / revoke / read / search / write / request / approve / deny are logged (bounded to 2 000 events). An
owner sees events on their resources and their own actions; the user sees all (`anvira context audit`).

## Using context in a request (on demand)

```python
app.chat(messages, context={"query": "zebrafish genome", "resources": ["res_..."], "limit": 5})
app.orcha.run("Summarise my notes", context={"query": "photosynthesis"})
```

Only resources the calling app is authorised for are searched, ranked once (BM25) across them, and only the top snippets are injected.
`runtime.context_used` in the response lists which resources contributed. Without a `context` option nothing is loaded.
The same context works with any model — it is injected as a system message, not tied to a model.

**Local data never goes to a cloud model silently.** If the active model is a remote provider, `context` and memory `recall` are
refused with `remote_context_not_allowed` (403) unless the request sets `allow_remote_context: true` — a decision for the user's UI to surface.

## API summary (`context.read` / `context.write` / `context.share`)

`POST/GET /v1/resources` · `GET|PATCH|DELETE /v1/resources/{id}` · `GET /v1/resources/{id}/read[?doc=]` ·
`PUT /v1/resources/{id}/documents/{doc}` · `POST /v1/resources/search` · `GET /v1/resources/resolve?ref=` ·
`POST /v1/resources/{id}/share|revoke|request` · `GET /v1/resources/{id}/permissions` ·
`GET /v1/resources/requests` · `POST /v1/resources/requests/{id}/approve|deny` · `GET /v1/resources/audit` ·
`GET|POST /v1/workspaces` · `POST /v1/workspaces/{name}/share` · `GET /v1/capabilities`.

## Capabilities

`GET /v1/capabilities` (CLI: `anvira capability list`) reports what the runtime offers and what is currently active or idle:
models, chat, orcha, agents, memory, context, resources, aicl. A capability that is idle costs nothing; "connected" does not
mean "loaded". Notes-, Study- and Code-specific experiences stay in those applications.

## Limits (honest)

* The runtime cannot let one app **invoke another app's features** (e.g. Study calling a Notes-only function). That needs the
  apps to expose them; the runtime only shares data and runtime capabilities.
* Same-OS-user processes can read the runtime's token files; grants protect against *apps using the API*, not against malware
  running as the user.
* `allow_remote_context` is a flag the *application* sets; it is the app's UI that must ask the user first.
