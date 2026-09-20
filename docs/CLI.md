# `anvira` CLI

```
anvira --help | --version          every command accepts --json (machine output) and -q
```

**Exit codes:** `0` ok · `1` failure or degraded status · `2` usage error · `3` runtime not installed · `4` runtime not running /
failed to start · `5` permission denied · `6` not found / no model · `8` `doctor` found failures · `130` interrupted.

Commands that *act* (`model`, `orcha`, `nomi`, `chat`, `ui`) start an installed-but-stopped runtime first (printing
`Starting runtime...`). Read-only ones (`status`, `doctor`, `config`, `version`, `runtime *`) never do.

## Running it

* **Windows, from a checkout:** use `anvira.cmd` in the repo root (`.\anvira.cmd status`) — it finds the project venv/Python and sets
  UTF-8. Add the repo folder to `PATH` to type plain `anvira`. Installed runtimes put an `anvira` launcher on the path instead.
* **A bare `anvira` in a terminal opens the live dashboard** (`anvira ui`); when piped it prints help.
* Action commands start the runtime **on demand** and hold a lease while they run; it stops itself ~2 minutes after the last
  command (`anvira runtime start` keeps it running persistently). See [LIFECYCLE.md](LIFECYCLE.md).

## Commands

| command | purpose |
|---|---|
| `anvira status` | runtime, ORCHA/Nomi/AICL, active model, jobs, apps (exit 0 healthy, 1 degraded, 4 stopped, 3 missing) |
| `anvira doctor` | diagnose: python, install, directories, config, storage, ORCHA/Nomi install+dependencies, AICL (+native core), port conflicts, version compatibility, hardware, llama-server, and (if running) services/model/model-vs-hardware. Prints a `fix:` per problem. Works with the runtime stopped. |
| `anvira version` | CLI, runtime and API versions |
| `anvira hardware` | CPU, RAM, GPU/VRAM/CUDA, accelerators, free storage |
| `anvira open [overview\|health\|models\|apps\|logs]` | the live dashboard (same as `ui`), starting on that page. **Tab / 1-5** switch pages, **r** refreshes the Health checkups. Bare `anvira` in a terminal does the same |
| `anvira runtime locate` · `register <folder>` · `install-gpu` | where the runtime is and how it was found · use a copy you already have (nothing copied) · add the NVIDIA GPU pack |
| `anvira runtime install [--github] [--dir D] [--gpu auto\|yes\|no] [--repo owner/name] [-y]` | asks: existing folder or download, where (any drive), GPU pack; downloads are SHA-256 verified |
| `anvira path add\|remove` | put `anvira` on your user PATH (asks first) |
| `anvira ui` | **live terminal dashboard** (see below) · `--once` prints one frame · `--exec "cmd;;cmd"` runs dashboard commands headlessly |
| `anvira runtime start [--foreground] [--port N]` · `stop` · `restart` · `status` | process control (`stop` is graceful, ~4 s) |
| `anvira runtime logs [-s runtime\|orcha\|nomi\|model:<id>] [-n N] [-f]` | logs (secrets redacted); works offline |
| `anvira runtime info` | install paths and config location |
| `anvira runtime install [--source zip\|URL\|dir] [-y] [--start]` · `update` | asks first; installs the runtime only, **never a model** |
| `anvira model list [--installed]` · `search <q>` · `info <id>` · `compat <id>` · `recommend` · `status` | discovery; `compat` says whether it fits this machine's VRAM/RAM/disk |
| `anvira model install <id\|owner/repo\|--url U> [--file F] [--dir D] [-y] [--no-wait]` | asks with the size and destination; live progress; Ctrl-C keeps the partial download |
| `anvira model use <id>` · `unuse` | select the active model (loads it) / stop it |
| `anvira model remove <id> [--delete-file] [-y]` | unlink; delete only when asked (and confirmed outside the download folder) |
| `anvira model dir [path] [--move]` | show / set where new models go — **any folder**; `--move` relocates existing ones |
| `anvira model dirs list\|add\|remove <path>` | extra folders scanned for existing models |
| `anvira model add <file-or-folder> [--id ID]` | register a `.gguf` (or a whole folder) in place — no copy |
| `anvira model discover` | model folders found from installed Anvira apps |
| `anvira model provider list\|add\|remove` | cloud/remote models: `add --base-url U --model M [--label L] (--api-key-env VAR \| --api-key-stdin \| --api-key K)` |
| `anvira orcha status` · `run "<task>" [--graph default\|research\|multi_agent] [--reasoning L] [--workspace DIR] [--no-wait]` · `jobs [--state]` · `cancel <id>` | orchestration |
| `anvira job get <id>` | any job (ORCHA run or model install) |
| `anvira nomi status` · `search <q> [--scope --app]` · `inspect <id>` · `store <text> [--title --tag --scope --app]` · `delete <id>` | memory (owner sees any app's namespace via `--app`) |
| `anvira context list\|add\|inspect\|read\|search\|share\|revoke\|delete\|requests\|approve\|deny\|audit` | shared resources: `add <title> (--file F \| --text T \| --collection C) [--owner APP]`; `share <id> <app…> [--write]`; `share <id> --global` asks first (user-only) — see [SHARED_CONTEXT.md](SHARED_CONTEXT.md) |
| `anvira workspace list\|create <name>\|share <name> <app…>` | group resources by project |
| `anvira capability list` | what the runtime offers and what is active vs idle |
| `anvira chat "<prompt>" [--system S] [--memory] [--no-stream]` | one prompt to the active model |
| `anvira config list\|get <key>\|set <key> <value>\|path` | validated; `api.host` can only be loopback |
| `anvira app list\|show <id>\|grant <id> <perm…>\|deny\|revoke <id>` | which apps are connected and what they may do |
| `anvira aicl status\|trace` | AICL bus: calls/errors/latency per module; the last packets |

## `anvira ui` — watch and drive the runtime

A full-screen dashboard (Windows Terminal, cmd, PowerShell, macOS/Linux terminals; no extra packages):

* **Services & model** — ORCHA / Nomi / AICL state, ports, restarts, active model with context size and GPU layers.
* **Jobs** — live list of ORCHA runs (state, age).
* **Activity** — ORCHA's own event frames for the run you started (`plan checkpoint -> select`, `execute node_end 90ms`), AICL
  packets (`orcha.sync_model [CALL] 189B/177B 27ms ok`) and the runtime log tail.
* **Output** and a prompt. Commands: `run <task> [--graph G] [--reasoning L]`, `chat <msg> [--memory]`, `models`, `use <id>`,
  `jobs`, `job <id>`, `cancel <id>`, `remember <text>`, `memory <query>`, `logs [service] [n]`, `status`, `doctor`, `clear`,
  `help`, `quit`. Keys: ↑/↓ history, PgUp/PgDn scroll, Ctrl+L redraw, Ctrl+C quit.

`anvira ui --once` prints a single frame (for logs, CI or screenshots); `anvira ui --exec "run hello;;jobs"` runs commands
without a TTY.

## Examples

```bash
anvira runtime start && anvira status
anvira model discover                       # models already downloaded by Anvira / Notes / Study / Dev
anvira model list --installed
anvira model use qwen2.5-coder-7b-instruct-q4_k_m
anvira model dir "E:\AI Models" --move      # put models wherever you want
anvira orcha run "Summarise the trade-offs of B-trees vs LSM-trees" --reasoning high
anvira doctor --json | jq '.checks[] | select(.status != "ok")'
```
