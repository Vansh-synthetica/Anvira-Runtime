"""Find, download and install Anvira Runtime from GitHub Releases (stdlib only).

An app (or the CLI) asks the *user* one question when the runtime is missing:

    * "I already have it"  -> ``register_location(folder)``   (nothing is downloaded; every app will find it there)
    * "Download it"        -> ``install_from_github(dest)``   (dest may be ANY folder; the choice is remembered)

Downloads are verified against the release's ``SHA256SUMS``. The GPU (CUDA) pack is a separate, larger asset that is
fetched only on machines with an NVIDIA GPU, and only with the user's consent. No model is ever downloaded here.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable

from .discovery import probe, read_json
from .errors import AnviraError
from .paths import runtime_dirs

# Rewritten by scripts/publish.ps1 once the GitHub repository exists; ANVIRA_RUNTIME_REPO overrides it.
DEFAULT_REPO = "Vansh-synthetica/Anvira-Runtime"
MIN_CUDA_DRIVER = (525, 0)
UPDATE_LOCK = ".updating"
Status = Callable[[str], None]
Progress = Callable[[str, int, int], None]        # (asset name, bytes done, bytes total)


# ---------------------------------------------------------------------------------------------- locate / register
def default_home(env: dict[str, str] | None = None) -> Path:
    return runtime_dirs(env, ignore_pointer=True)["home"]


def locate(env: dict[str, str] | None = None) -> dict[str, Any]:
    """Where is the runtime, how do we know, and is it usable? Never raises, never installs."""
    e = dict(os.environ if env is None else env)
    dirs = runtime_dirs(env)
    info = probe(env)
    if e.get("ANVIRA_RUNTIME_HOME"):
        source = "env"
    elif dirs["home"] != default_home(env):
        source = "pointer"
    else:
        source = "default"
    return {"found": info.installed, "home": str(dirs["home"]), "source": source, "default_home": str(default_home(env)),
            "running": info.running, "ready": info.ready, "version": info.runtime_version or info.install.get("version"),
            "portable": bool(info.install.get("portable")), "gpu_pack": bool(info.install.get("gpu_pack")),
            "problem": info.error}


def looks_like_runtime(path: Path) -> bool:
    return (path / "install.json").is_file() or (path / "state" / "owner.token").is_file()


def register_location(path: str | os.PathLike[str], env: dict[str, str] | None = None) -> Path:
    """Remember an existing install ("I already have it"), so every app finds it there. Nothing is copied."""
    target = Path(path).expanduser().resolve()
    if not target.is_dir() or not looks_like_runtime(target):
        raise AnviraError("not_a_runtime_folder", f"'{target}' does not look like an Anvira Runtime folder.",
                          hint="Pick the folder that contains install.json (the one with anvira.cmd and python/).")
    if not (target / "install.json").is_file():
        raise AnviraError("not_a_runtime_folder", f"'{target}' has runtime data but no install.json.",
                          hint="Pick the folder that contains install.json (the one with anvira.cmd and python/).")
    default = default_home(env)
    if target == default.resolve():
        forget_location(env)
        return target
    default.mkdir(parents=True, exist_ok=True)
    (default / "location.json").write_text(json.dumps({"home": str(target), "registered_at": time.time()}, indent=2), encoding="utf-8")
    return target


def forget_location(env: dict[str, str] | None = None) -> None:
    try:
        (default_home(env) / "location.json").unlink()
    except OSError:
        pass


# ------------------------------------------------------------------------------------------------------- hardware
def nvidia_gpu() -> dict[str, Any] | None:
    """The first NVIDIA GPU with its driver version, or None. Uses nvidia-smi (ships with the driver)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    try:
        out = subprocess.run([exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, creationflags=flags).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    name, driver, vram = ([x.strip() for x in out[0].split(",")] + ["", "", ""])[:3]
    m = re.match(r"(\d+)\.(\d+)", driver)
    ver = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    return {"name": name, "driver": driver, "vram_mib": int(vram) if vram.isdigit() else None,
            "cuda_ok": ver >= MIN_CUDA_DRIVER}


# ------------------------------------------------------------------------------------------------- update locking
def update_lock_path(env: dict[str, str] | None = None) -> Path:
    return runtime_dirs(env)["home"] / UPDATE_LOCK


def update_in_progress(env: dict[str, str] | None = None) -> bool:
    p = update_lock_path(env)
    try:
        return time.time() - p.stat().st_mtime < 20 * 60          # a crashed installer must not block forever
    except OSError:
        return False


def wait_for_update(env: dict[str, str] | None, timeout: float = 240.0, say: Status | None = None) -> None:
    """Block while an install/update is replacing files (an open app must not revive the runtime mid-update)."""
    deadline = time.monotonic() + timeout
    told = False
    while update_in_progress(env) and time.monotonic() < deadline:
        if not told and say:
            say("Waiting for the Anvira Runtime update to finish...")
            told = True
        time.sleep(0.5)


# ----------------------------------------------------------------------------------------------- GitHub download
def _repo(repo: str | None, env: dict[str, str] | None) -> str:
    r = repo or (env or os.environ).get("ANVIRA_RUNTIME_REPO") or DEFAULT_REPO
    if r.startswith("OWNER/"):
        raise AnviraError("repo_not_configured", "The Anvira Runtime GitHub repository is not configured in this build.",
                          hint="Set ANVIRA_RUNTIME_REPO=<owner>/Anvira-Runtime, or pass repo=... / --repo.")
    return r


def _get(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "anvira-runtime-installer", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_release(repo: str | None = None, tag: str | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    r = _repo(repo, env)
    api = (env or os.environ).get("ANVIRA_RELEASE_API", "https://api.github.com").rstrip("/")   # overridable for tests / GitHub Enterprise
    url = f"{api}/repos/{r}/releases/" + (f"tags/{tag}" if tag else "latest")
    try:
        rel = json.loads(_get(url))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise AnviraError("release_not_found", f"No Anvira Runtime release found in github.com/{r}"
                              + (f" for tag {tag}" if tag else "") + ".", hint="Check the repository name, or that a release is published.") from None
        raise AnviraError("download_failed", f"GitHub answered HTTP {exc.code}.", hint="Try again later, or download the zip by hand and use --source.") from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise AnviraError("download_failed", f"Could not reach GitHub: {exc}", hint="Check your internet connection.") from None
    return rel


def pick_assets(release: dict[str, Any]) -> dict[str, Any]:
    """Choose the core and (optional) CUDA assets for this OS/arch, with their expected SHA-256."""
    assets = {a["name"]: a for a in release.get("assets", [])}
    core = next((a for n, a in assets.items() if re.fullmatch(r"AnviraRuntime-[\d.]+-win-x64\.zip", n)), None) \
        if sys.platform == "win32" else None
    if core is None:
        raise AnviraError("no_release_asset", f"This release has no Anvira Runtime package for {sys.platform}.",
                          hint="Only Windows x64 packages are published so far; use `--source <repo checkout>` elsewhere.")
    cuda = next((a for n, a in assets.items() if n.endswith("-win-x64-cuda.zip")), None)
    sums: dict[str, str] = {}
    if "SHA256SUMS" in assets:
        for line in _get(assets["SHA256SUMS"]["browser_download_url"]).decode("utf-8", "replace").splitlines():
            parts = line.split()
            if len(parts) == 2:
                sums[parts[1].lstrip("*")] = parts[0].lower()
    return {"core": core, "cuda": cuda, "sha256": sums, "version": (release.get("tag_name") or "").lstrip("vV")}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(asset: dict[str, Any], dest_dir: Path, sha256: str | None, on_progress: Progress | None = None) -> Path:
    """Download one release asset (resumable), verifying size and SHA-256. Returns the file path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    name, total = asset["name"], int(asset.get("size") or 0)
    final, part = dest_dir / name, dest_dir / (name + ".part")
    if final.is_file() and (not sha256 or _sha256(final) == sha256):
        return final
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "anvira-runtime-installer", **({"Range": f"bytes={have}-"} if have else {})}
    try:
        resp = urllib.request.urlopen(urllib.request.Request(asset["browser_download_url"], headers=headers), timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and have:                      # already complete
            resp = None
        else:
            raise AnviraError("download_failed", f"Download of {name} failed (HTTP {exc.code}).") from None
    except (urllib.error.URLError, OSError) as exc:
        raise AnviraError("download_failed", f"Download of {name} failed: {exc}", hint="Run the install again to resume.") from None
    if resp is not None:
        resumed = getattr(resp, "status", 200) == 206
        with resp, open(part, "ab" if resumed else "wb") as fh:
            done = have if resumed else 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if on_progress:
                    on_progress(name, done, total)
    if total and part.stat().st_size != total:
        raise AnviraError("download_incomplete", f"{name} was cut short ({part.stat().st_size} of {total} bytes).",
                          hint="Run the install again to resume.")
    if sha256 and _sha256(part) != sha256:
        part.unlink(missing_ok=True)
        raise AnviraError("checksum_mismatch", f"{name} failed its SHA-256 check and was discarded.",
                          hint="Run the install again; if it keeps failing the release may be corrupt.")
    part.replace(final)
    return final


# ------------------------------------------------------------------------------------------------------ extract
def is_portable_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return any(n.replace("\\", "/").endswith("python/python.exe") for n in z.namelist())
    except (OSError, zipfile.BadZipFile):
        return False


def extract_package(zip_path: Path, dest: Path, say: Status | None = None) -> int:
    """Unpack an AnviraRuntime zip into ``dest`` (top folder stripped), refusing paths that escape it."""
    dest = dest.resolve()
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = info.filename.replace("\\", "/")
            name = name.split("/", 1)[1] if name.startswith("AnviraRuntime/") else name
            if not name or name.endswith("/"):
                continue
            out = (dest / name).resolve()
            if dest != out and dest not in out.parents:
                raise AnviraError("bad_bundle", f"The package contains an unsafe path: {info.filename}")
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(out, "wb") as fh:
                shutil.copyfileobj(src, fh, 1 << 20)
            n += 1
    return n


# ---------------------------------------------------------------------------------------------------- install
def install_package(zip_path: str | os.PathLike[str], dest: str | os.PathLike[str] | None = None,
                    env: dict[str, str] | None = None, on_status: Status | None = None, stop: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Install from a local AnviraRuntime zip (offline). ``dest`` may be any folder."""
    say = on_status or (lambda _m: None)
    zp = Path(zip_path)
    target = Path(dest).expanduser().resolve() if dest else default_home(env)
    _lock(target, True)
    try:
        if stop:
            stop()
        say(f"Unpacking Anvira Runtime into {target} ...")
        target.mkdir(parents=True, exist_ok=True)
        extract_package(zp, target, say)
        return _finish(target, env, say, source=str(zp))
    finally:
        _lock(target, False)


def _lock(home: Path, on: bool) -> None:
    p = home / UPDATE_LOCK
    try:
        if on:
            home.mkdir(parents=True, exist_ok=True)
            p.write_text(str(os.getpid()), encoding="utf-8")
        else:
            p.unlink(missing_ok=True)
    except OSError:
        pass


def _finish(target: Path, env: dict[str, str] | None, say: Status, source: str, gpu_pack: bool | None = None) -> dict[str, Any]:
    marker = target / "install.json"
    record = read_json(marker)
    if not record:
        raise AnviraError("bad_bundle", "The package has no install.json.")
    record.update({"installed_at": time.time(), "source": source})
    if gpu_pack is not None:
        record["gpu_pack"] = gpu_pack
    marker.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if target.resolve() != default_home(env).resolve():
        register_location(target, env)
        say(f"Remembered {target} so every Anvira app finds it.")
    else:
        forget_location(env)
    return record


def install_from_github(dest: str | os.PathLike[str] | None = None, *, repo: str | None = None, tag: str | None = None,
                        gpu: bool | str | None = None, confirm_gpu: Callable[[dict[str, Any]], bool] | None = None,
                        env: dict[str, str] | None = None, on_status: Status | None = None, on_progress: Progress | None = None,
                        force: bool = False, stop: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Download the latest release from GitHub into ``dest`` (any folder; default: the standard location).

    Call this only after the user chose to download. ``gpu``: True = fetch the CUDA pack, False = never,
    None/"auto" = ask ``confirm_gpu(info)`` on an NVIDIA machine (no callback -> not fetched; see ``result["gpu_pack_available"]``).
    """
    say = on_status or (lambda _m: None)
    target = Path(dest).expanduser().resolve() if dest else default_home(env)
    say(f"Looking up the latest Anvira Runtime release ({_repo(repo, env)}) ...")
    rel = fetch_release(repo, tag, env)
    assets = pick_assets(rel)
    core, cuda, sums = assets["core"], assets["cuda"], assets["sha256"]
    existing = read_json(target / "install.json")
    if existing and not force and existing.get("version") == assets["version"] and (target / "python").is_dir():
        say(f"Anvira Runtime {assets['version']} is already installed at {target}.")
        return {**existing, "unchanged": True}
    need = int(core.get("size") or 0) * 3
    target.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target).free
    if need and free < need:
        raise AnviraError("insufficient_storage", f"{target} has {free / 1e9:.1f} GB free; about {need / 1e9:.1f} GB is needed.",
                          hint="Choose another folder or drive.")
    _lock(target, True)
    try:
        if stop:
            stop()
        work = target / ".download"
        say(f"Downloading {core['name']} ({int(core['size']) / 1e6:.0f} MB) ...")
        core_zip = download(core, work, sums.get(core["name"]), on_progress)
        say("Unpacking ...")
        extract_package(core_zip, target, say)
        gpu_done = False
        card = nvidia_gpu()
        result_extra: dict[str, Any] = {}
        if cuda and card:
            result_extra["gpu_pack_available"] = {"name": card["name"], "driver": card["driver"], "size": int(cuda.get("size") or 0),
                                                  "cuda_ok": card["cuda_ok"]}
            want = gpu is True or (gpu in (None, "auto") and bool(confirm_gpu and confirm_gpu(result_extra["gpu_pack_available"])))
            if want and not card["cuda_ok"]:
                say(f"Skipping the GPU pack: driver {card['driver']} is older than the required {MIN_CUDA_DRIVER[0]}.x. CPU mode will be used.")
                want = False
            if want:
                say(f"Downloading the NVIDIA GPU pack ({int(cuda['size']) / 1e6:.0f} MB) ...")
                cuda_zip = download(cuda, work, sums.get(cuda["name"]), on_progress)
                extract_package(cuda_zip, target, say)
                gpu_done = True
        shutil.rmtree(work, ignore_errors=True)
        record = _finish(target, env, say, source=f"github:{_repo(repo, env)}@{rel.get('tag_name')}", gpu_pack=gpu_done)
        say(f"Anvira Runtime {record.get('version')} installed at {target}.")
        return {**record, **result_extra}
    finally:
        _lock(target, False)


def install_gpu_pack(env: dict[str, str] | None = None, repo: str | None = None, on_status: Status | None = None,
                     on_progress: Progress | None = None) -> dict[str, Any]:
    """Add the CUDA pack to an existing install (after the user agreed)."""
    say = on_status or (lambda _m: None)
    home = runtime_dirs(env)["home"]
    card = nvidia_gpu()
    if not card:
        raise AnviraError("no_nvidia_gpu", "No NVIDIA GPU was found; the GPU pack would not be used.")
    if not card["cuda_ok"]:
        raise AnviraError("driver_too_old", f"NVIDIA driver {card['driver']} is older than {MIN_CUDA_DRIVER[0]}.x.",
                          hint="Update the NVIDIA driver, then run this again.")
    rel = fetch_release(repo, None, env)
    assets = pick_assets(rel)
    if not assets["cuda"]:
        raise AnviraError("no_release_asset", "This release has no GPU pack.")
    _lock(home, True)
    try:
        say(f"Downloading the NVIDIA GPU pack ({int(assets['cuda']['size']) / 1e6:.0f} MB) ...")
        z = download(assets["cuda"], home / ".download", assets["sha256"].get(assets["cuda"]["name"]), on_progress)
        extract_package(z, home, say)
        shutil.rmtree(home / ".download", ignore_errors=True)
        record = read_json(home / "install.json")
        record["gpu_pack"] = True
        (home / "install.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record
    finally:
        _lock(home, False)
