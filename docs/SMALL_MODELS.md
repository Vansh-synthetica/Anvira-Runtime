# Making small models useful for real work (ORCHA)

ORCHA is designed for edge models (1.5B–8B): a long-horizon task planner, tool-name aliases, argument aliases, tolerance for wrong tool calls.
Running it with real models (`examples/real_workflows.py`, real Snake build) exposed places where a small model still fell through. Every change
below is **additive, deterministic (no extra model calls to judge the model), and covered by tests** in `Orcha/tests`.

| what a real model did | what ORCHA now does | code |
|---|---|---|
| Wrote a fake transcript instead of calling tools; had no tools at all | The runtime's `agent.run` always gives the agent file/search tools scoped to its workspace roots. Running commands needs the user's `orcha.exec` grant | `runtime/core/runtime.py` |
| "Read X, find the bug and fix it" was routed to a **read-only** agent and could not edit | The intent router treats "look + change" (non-negated) as a file operation ("do not change…", "change nothing" stay read-only) | `nodes/intent.py` |
| Read a file, then said "done" | **Nothing-changed gate**: if the task asks for a change and no write happened, the next turn must be a tool call | `agent_runtime/completion_checks.py`, `nodes/agent.py` |
| Renamed a symbol in one file and stopped | **Rename gate**: scans the workspace; the old name still exists → concrete `file:line` list, forced tool call, max 3 nudges | same |
| Prose ("Done.") after a gate rejected it | After every gate nudge the next turn is **grammar-forced to a tool call** (ORCHA's own force-tool marker) | `nodes/agent.py` |
| `run_command` with `{}` arguments | Fills the obvious command (project tests, else the file just written). A call that never ran is not treated as a failing command (that used to trigger destructive "fix" rewrites) | same |
| `find_and_replace`, `modify_file`… | The alias table was bypassed on the native path; it is now applied before the "no such tool" answer | `nodes/agent.py` |
| `edit_file` did not count as writing code | Added to the write set, so verification/fix gates engage | `nodes/agent.py` |
| Fuzzy edit text, CRLF, line-number prefixes, multi-file edits | opencode-style edit ladder + atomic `apply_patch` | `capabilities/editing.py` (see ORCHA_EDITING.md) |
| Long build request → 17 overlapping planner steps, stub files, a missing file | **Project scaffold**: requests that name ≥2 files are built one file per step, as plain text, verified mechanically, written through the normal tools | `agent_runtime/scaffold.py` |

## The scaffold path in detail

Trigger: a request with a build verb (create/build/make/write/generate/implement/develop) naming two or more files (`app.py`, `templates/index.html`, …).
Not triggered by questions, single files, edits, or paths that would escape the workspace. `ORCHA_SCAFFOLD=0` turns it off.

1. **Plan**: one step per file, ordered backend → page → styles → script → tests.
2. **Generate**: one focused call per file ("reply with the complete file in a code block"), shown the project request, the file list and the files already written, plus per-kind guidance (real element ids, every button needs a handler, loop structure for canvas games). Responses cut by the token limit are continued.
3. **Verify** (no model): not a stub; no placeholders; syntax (Python `ast`, JS via `node --check`, JSON); every feature the request names is present (localStorage, touch, sound, WASD, pause, responsive CSS, hover, Flask route…); ids used by the script exist in the page; every button is wired; and, when Node is available, the script is **executed in a fake browser** (DOM, canvas, timers, localStorage, audio, touch/keys, button clicks) to catch crashes, undefined names, and "Start never draws".
4. **Repair**: an undefined name → ask for *only that function*, insert it in scope; later attempts use SEARCH/REPLACE patches (applied with the tolerant edit ladder); otherwise a full rewrite with the concrete problem list. The **best** attempt is kept (a rewrite can fix one bug and add another).
5. **Write** through the policy-checked executor; final **smoke test** (compile Python; if Flask is installed, GET `/`).
6. **Report honestly**: `Built N of N files` or `Partly built…` with the unresolved problems named.

## Limits

* A 3B model does not reliably finish a 6–7 KB `game.js` (see `REAL_WORLD_RESULTS.md`). The runtime makes the failure small, specific and visible instead of silent; a 7B finishes it.
* The fake browser is a stub, not Chrome: it catches crashes and dead loops, not visual bugs or subtle gameplay logic.
* Node is optional; without it the script check falls back to static checks.
* Nomi recall is word-based (BM25), not semantic.
