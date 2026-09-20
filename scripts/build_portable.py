"""Build the self-contained Anvira Runtime release (Windows x64).

    python scripts/build_portable.py [--out dist] [--llama-cpu DIR] [--llama-cuda DIR] [--constraints FILE]

Produces, in ``--out``:
    AnviraRuntime-<ver>-win-x64.zip        core: its own trimmed Python + pinned dependencies + ORCHA/Nomi/AICL +
                                           the runtime + llama-server (CPU). No system Python, no pip, no internet.
    AnviraRuntime-<ver>-win-x64-cuda.zip   optional GPU pack: llama-server CUDA backend + NVIDIA runtime DLLs
                                           (extracted over the core; the installer fetches it only on NVIDIA machines).
    manifest.json + SHA256SUMS              what the installer reads to choose and verify assets.

Size work (lossless for what runs): stdlib as one compiled zip without the CPython test suite/Tk/IDLE (unittest, venv, ensurepip are kept); dependencies
without tests/type stubs/pythonwin/bytecode; the 17 MB vendored Qwen-Agent tree (never imported) left out; llama.cpp reduced
to what ``llama-server`` loads, with every CPU variant kept (the loader picks the best one for the machine) and the Microsoft VC++ runtime DLLs shipped beside it.
"""
from __future__ import annotations

import argparse
import compileall
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sdk" / "python"))
from anvira_client.bootstrap import _SKIP_DIRS, _SKIP_SUFFIXES  # noqa: E402

# Only things a user (or an ORCHA agent running "python -m unittest", "python -m venv", "pip") never needs are dropped: the CPython test suite,
# IDLE, Tk (needs the 10 MB Tcl/Tk DLLs) and turtle demos. unittest, venv and ensurepip (which bootstraps pip) are KEPT.
PY_SKIP_LIB = {"test", "tests", "idlelib", "tkinter", "turtledemo", "turtle.py", "lib2to3", "site-packages", "__pycache__",
               "__phello__", "xxlimited"}
PY_SKIP_DLL = ("tcl", "tk", "_tkinter", "_test", "_ctypes_test", "xxlimited", "_testbuffer", "_testimportmultiple",
               "_testmultiphase", "_testsinglephase", "_testinternalcapi", "_testclinic")
PKG_SKIP_DIRS = {"__pycache__", "tests", "test", "pythonwin", "win32comext", "bin", "testing_data", ".pytest_cache"}
PKG_SKIP_FILES = (".pyc", ".pyo", ".pyi")
SERVICE_PRUNE = {"Orcha": ["orcha/integrations/Qwen-Agent-0.0.26"], "nomi": [], "AICL": ["core-rust", "core-cpp"]}
# llama.cpp files that llama-server actually needs (tools like llama-cli/bench/quantize are dropped)
LLAMA_KEEP = re.compile(r"^(llama-server(\.exe|-impl\.dll)|llama-common\.dll|llama\.dll|mtmd\.dll|ggml(-base|-rpc)?\.dll|"
                        r"ggml-cuda\.dll|ggml-cpu-[\w]+\.dll|"
                        r"libomp[\w.]*\.dll|cudart64_\d+\.dll|cublas(Lt)?64_\d+\.dll|LICENSE[\w.-]*|.*\.txt)$", re.I)
NOTICE = """Anvira Runtime - third-party components bundled in this package

* Python (PSF License) - the trimmed interpreter in python/.
* Python packages in python/Lib/site-packages - each under its own license (see its *.dist-info/).
* llama.cpp llama-server (MIT License, https://github.com/ggml-org/llama.cpp) in bin/llama-cpp/.
* CUDA GPU pack only: NVIDIA CUDA runtime libraries cudart / cublas / cublasLt, redistributed under the NVIDIA CUDA Toolkit
  EULA "Attachment A" redistributable terms. Requires an NVIDIA driver that supports CUDA 12.
* LLVM OpenMP runtime (libomp) under the LLVM license.
* Microsoft Visual C++ runtime DLLs (msvcp140, vcruntime140, vcomp140 ...) redistributed as Microsoft "distributable code" so a clean PC needs no separate install.
No AI models are included; models are downloaded only when the user asks.
"""
LAUNCHER = r"""@echo off
rem Anvira Runtime launcher. Usage:  anvira open | status | doctor | model list | orcha run "..." | --help
setlocal
set "ROOT=%~dp0"
if not defined ANVIRA_RUNTIME_HOME set "ANVIRA_RUNTIME_HOME=%ROOT:~0,-1%"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
chcp 65001 >nul 2>&1
"%ROOT%python\python.exe" -m anvira_runtime %*
exit /b %ERRORLEVEL%
"""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def step(msg: str) -> None:
    print(f"==> {msg}", flush=True)


def build_python(dest: Path) -> None:
    base = Path(sys.base_prefix)
    dest.mkdir(parents=True)
    for name in ("python.exe", "pythonw.exe", "python3.dll", "python312.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        if (base / name).is_file():
            shutil.copy2(base / name, dest / name)
    for f in (base / "DLLs").iterdir():      # extension modules sit beside python.exe, as in the official embeddable build
        if f.is_file() and not f.name.lower().startswith(PY_SKIP_DLL) and f.suffix.lower() in (".pyd", ".dll"):
            shutil.copy2(f, dest / f.name)
    with tempfile.TemporaryDirectory() as tmp:
        lib = Path(tmp) / "Lib"
        shutil.copytree(base / "Lib", lib, ignore=lambda d, names: [n for n in names if n in PY_SKIP_LIB])
        compileall.compile_dir(str(lib), quiet=2, legacy=True, force=True)     # legacy => foo.pyc beside foo.py
        with zipfile.ZipFile(dest / "python312.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for f in sorted(lib.rglob("*.pyc")):
                z.write(f, f.relative_to(lib).as_posix())


ROOT_PACKAGES = ["fastapi", "uvicorn", "httpx", "pydantic", "langgraph", "PyYAML", "mcp", "sqlalchemy", "aiosqlite", "alembic",
                 "python-jose", "bcrypt", "pydantic-settings", "email-validator", "python-multipart", "requests", "beautifulsoup4"]


def constraints_from_env(path: Path) -> None:
    """Pin the *tested* dependency closure: walk the installed packages' requirements from the roots above.

    The tested environment's versions are the truth (a fresh `pip install` would resolve newer, untested ones).
    """
    from importlib import metadata

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    seen: dict[str, str] = {}
    todo = [canonicalize_name(p) for p in ROOT_PACKAGES]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        seen[name] = dist.version
        for line in dist.requires or []:
            req = Requirement(line)
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                todo.append(canonicalize_name(req.name))
    path.write_text("".join(f"{n}=={v}\n" for n, v in sorted(seen.items())), encoding="utf-8")


def build_site_packages(dest: Path, constraints: Path) -> None:
    dest.mkdir(parents=True)
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-warn-script-location", "--no-compile",
           "--no-deps", "--target", str(dest), "-r", str(constraints)]
    subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-warn-script-location", "--no-compile",
                    "--no-deps", "--target", str(dest), f"{REPO / 'runtime'}", str(REPO / "sdk" / "python")], check=True)
    for p in sorted(dest.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if p.is_dir() and p.name in PKG_SKIP_DIRS and p.parent != dest / "anvira_runtime":
            shutil.rmtree(p, ignore_errors=True)
        elif p.is_file() and (p.suffix in PKG_SKIP_FILES or p.name in ("RECORD", "INSTALLER", "direct_url.json")):
            p.unlink()
    shutil.rmtree(dest / "bin", ignore_errors=True)


def copy_service(name: str, dst: Path) -> None:
    src = REPO / name
    prune = [src / p for p in SERVICE_PRUNE.get(name, [])]
    for f in src.rglob("*"):
        rel = f.relative_to(src)
        if not f.is_file() or set(rel.parts) & _SKIP_DIRS or f.name.endswith(_SKIP_SUFFIXES) or any(p.endswith(".egg-info") for p in rel.parts):
            continue
        if any(pr in f.parents for pr in prune):
            continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst / rel)


def copy_llama(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src.iterdir():
        if f.is_file() and LLAMA_KEEP.match(f.name):
            shutil.copy2(f, dst / f.name)
            n += 1
    return n


VC_RUNTIME = ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "vcruntime140.dll", "vcruntime140_1.dll", "vcomp140.dll", "concrt140.dll")


def bundle_vc_runtime(*dirs: Path) -> int:
    """llama-server, ggml and the CUDA backend import the Microsoft VC++ runtime. A clean PC may not have it, so ship it beside them.

    These DLLs are Microsoft "distributable code" (Visual C++ Redistributable) and are copied from this machine's System32."""
    src = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    n = 0
    for d in dirs:
        if not d.is_dir():
            continue
        for name in VC_RUNTIME:
            if (src / name).is_file():
                shutil.copy2(src / name, d / name)
                n += 1
    return n


def zip_tree(root: Path, out: Path, only: Path | None = None, exclude: Path | None = None) -> None:
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        base = only or root
        for f in sorted(base.rglob("*")):
            if f.is_file() and not (exclude and exclude in f.parents):
                z.write(f, "AnviraRuntime/" + f.relative_to(root).as_posix())


def main() -> int:
    version = re.search(r'RUNTIME_VERSION\s*=\s*"([^"]+)"', (REPO / "runtime/anvira_runtime/version.py").read_text()).group(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "dist"))
    ap.add_argument("--llama-cpu", default=os.environ.get("ANVIRA_LLAMA_CPU_DIR"))
    ap.add_argument("--llama-cuda", default=os.environ.get("ANVIRA_LLAMA_CUDA_DIR"))
    ap.add_argument("--constraints")
    ap.add_argument("--keep-stage", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stage = out / "stage" / "AnviraRuntime"
    if stage.parent.exists():
        shutil.rmtree(stage.parent)
    stage.mkdir(parents=True)

    step("bundled Python (trimmed stdlib)")
    build_python(stage / "python")
    step("dependencies (pinned to the tested versions)")
    cons = Path(a.constraints) if a.constraints else out / "constraints.txt"
    if not a.constraints:
        constraints_from_env(cons)
    build_site_packages(stage / "python" / "Lib" / "site-packages", cons)
    step("ORCHA, Nomi, AICL")
    for name, dst in {"Orcha": "orcha", "nomi": "nomi", "AICL": "aicl"}.items():
        copy_service(name, stage / "services" / dst)
    (stage / "python" / "Lib" / "site-packages" / "anvira-services.pth").write_text(
        "../../../services/aicl\n", encoding="utf-8")            # AICL is imported by the runtime; ORCHA/Nomi get PYTHONPATH
    step("llama.cpp")
    counts = {}
    if a.llama_cpu:
        counts["cpu"] = copy_llama(Path(a.llama_cpu), stage / "bin" / "llama-cpp" / "cpu")
    cuda_dir = stage / "bin" / "llama-cpp" / "cuda"
    if a.llama_cuda:
        counts["cuda"] = copy_llama(Path(a.llama_cuda), cuda_dir)
    vc = bundle_vc_runtime(stage / "bin" / "llama-cpp" / "cpu", cuda_dir, stage / "python")
    print(f"    VC++ runtime DLLs bundled: {vc} copies")
    (stage / "anvira.cmd").write_text(LAUNCHER, encoding="ascii")
    (stage / "NOTICE.txt").write_text(NOTICE, encoding="utf-8")
    (stage / "README.txt").write_text(
        f"Anvira Runtime {version}\n\nOpen a terminal in this folder and run:\n    anvira open      live dashboard\n"
        "    anvira status    what is running\n    anvira doctor    checkups\n    anvira --help    everything else\n", encoding="utf-8")
    record = {"version": version, "portable": True, "entry": str(stage / "python" / "python.exe"), "os": "windows",
              "arch": "x86_64", "services_dir": str(stage / "services"), "source": "release"}
    (stage / "install.json").write_text(json.dumps({**record, "entry": "python/python.exe", "services_dir": "services"}, indent=2), encoding="utf-8")

    step("smoke test: the bundled interpreter imports the runtime")
    env = {**os.environ, "ANVIRA_RUNTIME_HOME": str(stage), "PYTHONPATH": ""}
    r = subprocess.run([str(stage / "python" / "python.exe"), "-c",
                        "import anvira_runtime, anvira_client, fastapi, uvicorn, httpx, langgraph, sqlalchemy, yaml, aicl; print('imports ok')"],
                       capture_output=True, text=True, env=env, cwd=str(stage))
    print(r.stdout.strip() or r.stderr[-800:])
    if r.returncode != 0:
        return 1

    step("zipping")
    core = out / f"AnviraRuntime-{version}-win-x64.zip"
    zip_tree(stage, core, exclude=cuda_dir)
    assets = [{"name": core.name, "kind": "core", "size": core.stat().st_size, "sha256": sha256(core)}]
    if a.llama_cuda:
        cuda = out / f"AnviraRuntime-{version}-win-x64-cuda.zip"
        zip_tree(stage, cuda, only=cuda_dir)
        assets.append({"name": cuda.name, "kind": "cuda", "size": cuda.stat().st_size, "sha256": sha256(cuda), "requires": "nvidia-gpu"})
    manifest = {"version": version, "os": "windows", "arch": "x86_64", "assets": assets, "llama_files": counts}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "SHA256SUMS").write_text("".join(f"{x['sha256']}  {x['name']}\n" for x in assets), encoding="utf-8")
    print(f"\nstaged: {size_mb(stage):.0f} MB unpacked")
    for x in assets:
        print(f"{x['name']}: {x['size'] / 1e6:.1f} MB")
    if not a.keep_stage:
        shutil.rmtree(out / "stage", ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
