# Installation

Runtime installation and **model** installation are separate. Installing the runtime downloads no model; models are installed
explicitly (`anvira model install …`) or discovered from folders you already have.

## The normal way: the self-contained package (Windows x64)

Two files on the GitHub **Releases** page:

| file | size | contains |
|---|---|---|
| `AnviraRuntime-<ver>-win-x64.zip` | ~45 MB | its own trimmed Python, every dependency (pinned to the tested versions), ORCHA, Nomi, AICL, the runtime, and the CPU `llama-server`. **No system Python, no pip, no internet after download.** |
| `AnviraRuntime-<ver>-win-x64-cuda.zip` | ~650 MB | the NVIDIA GPU pack: CUDA backend + NVIDIA runtime DLLs (cudart, cublas, cublasLt). Needs an NVIDIA driver 525 or newer. |

Unzip the first anywhere (any folder, any drive). NVIDIA card? Unzip the second **into the same folder**. Then, in a terminal in that folder:

```
anvira open          live dashboard
anvira doctor        checkups (finds the bundled llama-server, GPU, ports...)
anvira status
```

Run `anvira path add` once (it asks) if you want plain `anvira` to work from any terminal. The first time you run it, the folder registers
itself, so every Anvira app finds it wherever you put it.

**Why two files.** The CUDA libraries are about 1.1 GB unpacked and only help NVIDIA machines, so they are a separate, optional download
instead of taxing everyone. The installer (below) fetches the GPU pack only on a machine with an NVIDIA GPU and a new enough driver, and only if you say yes.

## How an app or the CLI gets it (GitHub, custom location)

When the runtime is missing, an app must ask the user (never install silently):

* **"I already have it"** → pick the folder (the one containing `install.json`). Nothing is downloaded; every app will use it.
* **"Download it"** → choose a folder (any drive). The package is downloaded from the GitHub release, **verified against `SHA256SUMS`**, unpacked, and the location is remembered. If an NVIDIA GPU is found, the app offers the GPU pack.

```ts
await AnviraRuntime.connect({ appId: 'anvira-notes', install: async (info) => {
  const c = await showDialog(info)                       // your UI
  return c.cancel ? false : c.haveIt ? { action: 'use-existing', path: c.folder } : { action: 'download', dest: c.folder, gpu: c.gpu }
}})
```

From a terminal: `anvira runtime install` (asks: existing folder or download, where, GPU pack), or non-interactively
`anvira runtime install --github --dir "E:\Anvira Runtime" --gpu yes -y`. `anvira runtime locate` shows where the runtime is and how it was found;
`anvira runtime register <folder>` points at an existing copy; `anvira runtime install-gpu` adds the GPU pack later.

The custom location is stored in a tiny pointer file, `%LOCALAPPDATA%\AnviraRuntime\location.json`. `ANVIRA_RUNTIME_HOME` overrides it.
If the target folder is deleted the pointer is ignored and the runtime is reported as not installed. Downloads resume if interrupted, a corrupt
download is discarded, and an **update lock** stops any open app from restarting the runtime while files are being replaced.

The GitHub repository is `ANVIRA_RUNTIME_REPO=<owner>/Anvira-Runtime` (or `DEFAULT_REPO` in the SDKs, set by `scripts/publish.ps1`).

## From a source checkout (developers)

Python 3.10+ on the machine.

```
python -m venv .venv && .venv\Scripts\pip install -e runtime[all] -e sdk/python
.\anvira.cmd doctor
python scripts/build_portable.py --llama-cpu <dir> --llama-cuda <dir>     # build the packages in dist/
anvira runtime install --source dist/AnviraRuntime-<ver>-win-x64.zip      # offline install of a package
```

`scripts/build_portable.py` builds a trimmed Python (stdlib as one compiled zip, no tests/Tk/IDLE), installs the **tested** dependency versions
(walked from the environment the tests ran in, not a fresh resolve), drops tests/type stubs/unused vendored trees, keeps only the llama.cpp
files `llama-server` loads, and writes `manifest.json` + `SHA256SUMS`. The pip-based installer (`anvira runtime install --source <repo>`)
still works for source checkouts.

## Inference backend (llama-server)

Bundled in the package under `bin/llama-cpp/cpu` and, with the GPU pack, `bin/llama-cpp/cuda`. The runtime picks CUDA on an NVIDIA machine and CPU
otherwise; lookup order is `models.llama_server_path`, `ANVIRA_LLAMA_SERVER`, `<home>/bin/llama-cpp/{cuda,cpu}/`, binaries an existing Anvira install
downloaded, then `PATH`. Cloud/remote providers work without it.

## Where things live

`<home>` is the folder you unzipped into (custom), else `%LOCALAPPDATA%\AnviraRuntime` (Windows), `~/Library/Application Support/AnviraRuntime` (macOS),
`~/.local/share/anvira-runtime` (Linux). Models can live anywhere (`anvira model dirs add <folder>`, `anvira model add <file>`); apps' model folders are discovered.

## Update

New package over the old one: `anvira runtime install --github` (or `--source`). The runtime is stopped for the swap; config, tokens, models, Nomi data
and registered apps are kept; apps reconnect on their own.

## Uninstall

`anvira runtime stop`, then delete the folder (this also deletes Nomi's data and tokens). Models stay wherever they are unless they were inside it.
`anvira path remove` undoes `path add`.

## Configuration

`anvira config list` shows every key; the important ones: `api.port` (default 47615, `0` = pick a free port), `models.models_dir`, `models.extra_dirs`,
`models.discover_apps`, `models.gpu` (`auto|off|on`), `models.context_size`, `models.llama_server_path`, `lifecycle.*`, `services.python`, `supervisor.*`.
Secrets are never in `config.json`.

## Troubleshooting

`anvira doctor` names the problem and the fix. `anvira runtime logs -s orcha|nomi|model:<id>`. `anvira open` for the live view (Health page runs the checkups).
Blank console windows popping up were a bug in older builds (the daemon was launched without a console); the daemon now runs hidden and shows nothing.
