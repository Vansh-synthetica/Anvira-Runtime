"""Assemble the clean, publishable Anvira Runtime tree.

    python scripts/finalize.py [--out Finalized] [--clean]

Copies exactly what belongs in the GitHub repository: the runtime, the SDKs, ORCHA / Nomi / AICL sources, docs, scripts, tests and
examples. It leaves out the application `src/` (Anvira, Notes, Study, Dev live elsewhere), virtual environments, caches, build
output, the vendored Qwen-Agent tree that nothing imports, and every binary (llama.cpp, models): those ship as GitHub Release
assets built by ``scripts/build_portable.py``, never as git history.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOP = ["runtime", "sdk", "Orcha", "nomi", "AICL", "docs", "scripts", "tests", "examples"]
FILES = ["anvira.cmd"]
SKIP_DIRS = {".git", ".venv", ".build-venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "build", "dist",
             "node_modules", "target", "_legacy_v0", "Qwen-Agent-0.0.26", ".idea", ".vscode", "benchmarks", "core-cpp", "native-cpp",
             "docker", "vendor"}
SKIP_SUFFIX = (".pyc", ".pyo", ".spec", ".log", ".gguf", ".dll", ".exe", ".zip", ".egg-info")
SKIP_NAMES = {"a.txt", "t.txt", "demo.py", ".DS_Store", "Thumbs.db"}
MAX_BYTES = 5_000_000


def wanted(rel: Path, src: Path) -> bool:
    if set(rel.parts) & SKIP_DIRS or any(p.endswith(".egg-info") for p in rel.parts):
        return False
    return not (src.name.endswith(SKIP_SUFFIX) or src.name in SKIP_NAMES or src.stat().st_size > MAX_BYTES)


README = """# Anvira Runtime

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
python -m venv .venv && .venv\\Scripts\\pip install -e runtime[all] -e sdk/python
.\\anvira.cmd doctor
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
"""

GITIGNORE = """__pycache__/
*.pyc
.venv/
.build-venv/
node_modules/
dist/
build/
*.egg-info/
*.gguf
*.log
.pytest_cache/
# binaries ship as release assets, never in git
bin/
*.dll
*.exe
*.zip
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "Finalized"))
    ap.add_argument("--clean", action="store_true", help="delete the output folder first (keeps .git)")
    a = ap.parse_args()
    out = Path(a.out)
    if a.clean and out.exists():
        for child in out.iterdir():
            if child.name == ".git":
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    out.mkdir(parents=True, exist_ok=True)
    n = size = 0
    for top in TOP:
        base = REPO / top
        if not base.is_dir():
            continue
        for f in base.rglob("*"):
            rel = f.relative_to(base)
            if f.is_file() and wanted(rel, f):
                dst = out / top / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                n += 1
                size += f.stat().st_size
    for name in FILES:
        if (REPO / name).is_file():
            shutil.copy2(REPO / name, out / name)
    (out / "README.md").write_text(README, encoding="utf-8")
    (out / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    print(f"{out}: {n} files, {size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
