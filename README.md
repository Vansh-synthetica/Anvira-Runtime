# Anvira Runtime

One local, shared runtime for every Anvira app (Anvira, Anvira Notes, Anvira Study, Anvira Dev). It supervises **ORCHA** (agents and
orchestration), **Nomi** (memory) and **AICL** (the internal module bus), manages models (llama.cpp on CPU or NVIDIA GPU), and
exposes one token-authenticated loopback API with a CLI, a terminal dashboard and Python / TypeScript SDKs.

It is active **only while an app is open**, never downloads a model or installs anything without asking, and lets apps share data
only when the owner (or the user) says so.

## Install (Windows x64)

1. Download `AnviraRuntime-<version>-win-x64.zip` from **Releases** (about 45 MB: its own Python, all dependencies, ORCHA, Nomi,
   AICL and the CPU `llama-server`; no Python, CUDA toolkit or llama.cpp to install).
2. Unzip it anywhere (any folder or drive).
3. NVIDIA GPU? Also download `AnviraRuntime-<version>-win-x64-cuda.zip` (about 650 MB) and unzip it **into the same folder**, or let
   `anvira runtime install` fetch it for you. It carries the CUDA backend, so there is nothing else to set up.
4. Open a terminal in that folder and run:

```
anvira open
```

`anvira open` is a live dashboard (Overview, Health, Models, Apps & data, Logs). Useful commands: `anvira status`, `anvira doctor`,
`anvira model dirs add <folder with .gguf files>`, `anvira model use <id>`, `anvira orcha run "..."`, `anvira --help`.
Run `anvira path add` once if you want plain `anvira` to work from any terminal.

Apps find the runtime automatically, and if it is missing they ask the user whether to use an existing copy or download one
(see `docs/APP_BUILDER_PROMPT.md`).

## From source

```
python -m venv .venv && .venv\Scripts\pip install -e runtime[all] -e sdk/python
.\anvira.cmd doctor
```

## Layout

| folder | what |
|---|---|
| `runtime/` | the daemon, API, CLI, terminal dashboard, model layer, supervisor |
| `sdk/python`, `sdk/typescript` | client SDKs (detect / install / connect / models / chat / agents / memory / shared context) |
| `Orcha/`, `nomi/`, `AICL/` | the wrapped engines (unchanged in behaviour; small-model improvements are documented in `docs/ORCHA_EDITING.md`, `docs/SMALL_MODELS.md`) |
| `scripts/` | `build_portable.py` (the self-contained package), `finalize.py`, `publish.ps1` |
| `examples/real_workflows.py` | real end-to-end Notes / Study / Dev workflows against a real model |
| `docs/` | architecture, API, CLI, lifecycle, shared context, installation, integration, results |

## Tests

`python -m pytest tests/runtime` (starts real runtimes with fake model servers), `cd Orcha && python -m pytest tests`, `cd nomi && python -m pytest tests`,
`cd AICL && python -m pytest`, `cd sdk/typescript && node --test test/install.test.mjs`.

## Licence

Not chosen yet by the repository owner. `Orcha/LICENSE` and `nomi/LICENSE` apply to those folders. Bundled third-party components are listed in `NOTICE.txt` of the release package.
