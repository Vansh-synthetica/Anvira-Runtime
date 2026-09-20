"""Installed-model registry, active-model state and cloud provider store.

The registry is the single source of truth that replaces Anvira's
renderer-side ``localStorage`` list. It is *self-healing*: model directories
are scanned on every read, so GGUF files dropped in by hand — or already
downloaded by an existing Anvira install — appear automatically, and files
that were deleted disappear. No duplicate download is ever required.

Where models live is the user's decision (see :class:`ModelStore`).
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..config.paths import RuntimeLayout
from ..security.secrets import read_json, write_private

SPLIT_RE = re.compile(r"^(?P<base>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$", re.I)
_SLUG_RE = re.compile(r"[^a-z0-9._]+")


def slugify(stem: str) -> str:
    stem = stem.split("__")[-1]  # Anvira prefixes "author__repo__" on some downloads
    return _SLUG_RE.sub("-", stem.lower()).strip("-.") or "model"


def parse_gguf_filename(filename: str) -> dict[str, str | None]:
    """Architecture + quantization from names like ``Qwen2.5-3B-Instruct-Q4_K_M.gguf``
    (port of Anvira's ``modelManager.parseGgufFilename``)."""
    name = re.sub(r"\.gguf$", "", filename, flags=re.I)
    q = re.search(r"[-.]((?:IQ\d[\w]*|Q\d[\w]*|F\d+|BF16))$", name, re.I)
    quant = q.group(1).upper() if q else None
    core = name[: q.start()] if q else name
    arch = re.sub(r"-(instruct|chat|dpo)$", "", core, flags=re.I)
    arch = re.sub(r"[-.]?\d+(?:\.\d+)?[Bb]$", "", arch).strip("-. ") or None
    return {"architecture": arch, "quantization": quant}


def validate_storage_dir(path: Path) -> tuple[bool, str | None]:
    """Create ``path`` if needed and prove it is readable and writable
    (port of Anvira's ``modelStorage.validateStorageDir``). Any local drive,
    external disk or folder with spaces is fine."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"Cannot create the directory: {exc}"
    probe = path / f".anvira-write-test-{os.getpid()}-{int(time.time() * 1000)}"
    try:
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        return False, f"The directory is not readable/writable: {exc}"
    return True, None


def _is_inside(directory: Path, path: Path) -> bool:
    try:
        Path(os.path.normcase(str(path.resolve()))).relative_to(Path(os.path.normcase(str(directory.resolve()))))
        return True
    except (ValueError, OSError):
        return False


class ModelStore:
    """Where models live is the user's decision.

    * ``primary_dir`` — where new downloads go (``models.models_dir``; any path).
    * ``extra_dirs`` — additional folders scanned for ``.gguf`` files (for
      example an existing Anvira or LM Studio models folder), so nothing is
      re-downloaded.
    * linked files — a ``.gguf`` at any path registered in place with
      :meth:`link` (never copied).
    """

    def __init__(self, layout: RuntimeLayout, primary_dir: Path, extra_dirs: list[Path] | None = None,
                 discover: Callable[[], dict[Path, str]] | None = None):
        self.layout = layout
        self.primary_dir = primary_dir
        self.extra_dirs = [d for d in (extra_dirs or []) if d != primary_dir]
        self._discover = discover
        self._lock = threading.RLock()

    def discovered(self) -> dict[Path, str]:
        """Model folders of installed Anvira apps (``{dir: app label}``), excluding folders already configured."""
        if self._discover is None:
            return {}
        try:
            found = self._discover()
        except Exception:  # noqa: BLE001 - discovery is best-effort
            return {}
        configured = {os.path.normcase(str(d)) for d in (self.primary_dir, *self.extra_dirs)}
        return {d: label for d, label in found.items() if os.path.normcase(str(d)) not in configured}

    @property
    def models_dir(self) -> Path:
        return self.primary_dir

    def scan_dirs(self) -> list[Path]:
        return [self.primary_dir, *self.extra_dirs, *self.discovered()]

    def _locate(self, p: Path) -> tuple[str, str | None]:
        if _is_inside(self.primary_dir, p):
            return "primary", None
        if any(_is_inside(d, p) for d in self.extra_dirs):
            return "extra", None
        for d, label in self.discovered().items():
            if _is_inside(d, p):
                return "app", label
        return "linked", None

    def set_dirs(self, primary: Path, extra: list[Path]) -> None:
        self.primary_dir = primary
        self.extra_dirs = [d for d in extra if d != primary]

    # -- registry (runtime state, independent of where model files are) -----
    @property
    def registry_file(self) -> Path:
        return self.layout.state_dir / "model-registry.json"

    def _registry(self) -> dict[str, dict[str, Any]]:
        return read_json(self.registry_file, {}).get("models", {})

    def _save_registry(self, models: dict[str, dict[str, Any]]) -> None:
        write_private(self.registry_file, json.dumps({"models": models}, indent=2, sort_keys=True))

    @staticmethod
    def check_filename(filename: str) -> str:
        """A bare ``*.gguf`` (or ``.gguf.part``) file name — no directories, no traversal."""
        name = Path(filename).name
        if not name or name != filename or "/" in filename or "\\" in filename or name.startswith("."):
            raise ValueError(f"Invalid model filename: {filename!r}")
        if not name.lower().endswith((".gguf", ".gguf.part")):
            raise ValueError("Only .gguf model files are supported.")
        return name

    def safe_path(self, filename: str, directory: Path | None = None) -> Path:
        return (directory or self.primary_dir) / self.check_filename(filename)

    def _is_managed(self, path: Path) -> bool:
        return any(_is_inside(d, path) for d in self.scan_dirs())

    # -- scanning -------------------------------------------------------------
    def scan(self) -> list[dict[str, Any]]:
        """Installed local models from every scanned dir plus linked files."""
        with self._lock:
            registry = self._registry()
            by_path = {os.path.normcase(str(Path(r["path"]))): mid
                       for mid, r in registry.items() if r.get("path")}
            entries: dict[str, dict[str, Any]] = {}

            def consider(files: list[Path]) -> None:
                groups: dict[str, list[Path]] = {}
                for p in sorted(files):
                    m = SPLIT_RE.match(p.name)
                    if m:
                        groups.setdefault(str(p.parent) + "|" + m["base"] + "|" + m["total"], []).append(p)
                        continue
                    entries.setdefault(os.path.normcase(str(p)), {
                        "path": p, "size_bytes": p.stat().st_size, "shards": 1, "complete": True})
                for key, shards in groups.items():
                    total = int(key.rsplit("|", 1)[1])
                    first = next((p for p in shards if SPLIT_RE.match(p.name)["idx"] == "00001"), None)  # type: ignore[index]
                    if first is None:
                        continue  # the entry shard is required to launch a split model
                    entries.setdefault(os.path.normcase(str(first)), {
                        "path": first, "size_bytes": sum(p.stat().st_size for p in shards),
                        "shards": total, "complete": len(shards) == total})

            for d in self.scan_dirs():
                if d.is_dir():
                    with contextlib.suppress(OSError):
                        consider(list(d.glob("*.gguf")))
            for rec in registry.values():  # linked files anywhere on disk
                if rec.get("path") and Path(rec["path"]).is_file():
                    with contextlib.suppress(OSError):
                        consider([Path(rec["path"])])

            out, used = [], set()
            for norm, meta in sorted(entries.items()):
                p: Path = meta["path"]
                base_id = by_path.get(norm) or slugify(re.sub(r"-00001-of-\d{5}", "", p.stem))
                mid, n = base_id, 2
                while mid in used:  # same file name in two folders
                    mid, n = f"{base_id}-{n}", n + 1
                used.add(mid)
                rec = registry.get(base_id) or {}
                parsed = parse_gguf_filename(p.name)
                out.append({
                    "id": mid, "kind": "local", "name": rec.get("name") or p.stem, "file": p.name,
                    "path": str(p), "directory": str(p.parent), "installed": True,
                    "location": self._locate(p)[0], "origin": self._locate(p)[1],
                    "size_bytes": meta["size_bytes"], "complete": meta["complete"], "shards": meta["shards"],
                    "quantization": rec.get("quantization") or parsed["quantization"],
                    "architecture": parsed["architecture"], "params_b": rec.get("params_b"),
                    "source": rec.get("source"), "installed_at": rec.get("installed_at"),
                })
            return sorted(out, key=lambda r: r["id"])

    def get(self, model_id: str) -> dict[str, Any] | None:
        return next((m for m in self.scan() if m["id"] == model_id), None)

    def register(self, model_id: str, path: Path, *, name: str | None = None, source: dict | None = None,
                 quantization: str | None = None, params_b: float | None = None) -> None:
        with self._lock:
            reg = self._registry()
            reg[model_id] = {"path": str(path), "name": name, "source": source, "quantization": quantization,
                             "params_b": params_b, "installed_at": time.time()}
            self._save_registry(reg)

    def link(self, file_path: str, *, model_id: str | None = None, name: str | None = None,
             source: dict | None = None) -> dict[str, Any]:
        """Register a .gguf that lives anywhere, in place (no copy)."""
        p = Path(file_path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"No such file: {p}")
        if not p.name.lower().endswith(".gguf"):
            raise ValueError("Only complete .gguf model files can be linked.")
        p = p.resolve()
        mid = model_id or slugify(re.sub(r"-00001-of-\d{5}", "", p.stem))
        self.register(mid, p, name=name or p.stem, source=source,
                      quantization=parse_gguf_filename(p.name)["quantization"])
        found = self.get(mid)
        if found is None:
            self.unregister(mid)
            raise ValueError("The file could not be registered (a split model must be linked by its 00001 shard).")
        return found

    def repoint(self, moved: dict[str, str]) -> None:
        """After files were moved, update registry paths (old path -> new path)."""
        if not moved:
            return
        norm = {os.path.normcase(k): v for k, v in moved.items()}
        with self._lock:
            reg = self._registry()
            changed = False
            for rec in reg.values():
                new = norm.get(os.path.normcase(rec.get("path") or ""))
                if new:
                    rec["path"], changed = new, True
            if changed:
                self._save_registry(reg)

    def unregister(self, model_id: str) -> None:
        with self._lock:
            reg = self._registry()
            if reg.pop(model_id, None) is not None:
                self._save_registry(reg)

    def remove(self, model_id: str, *, delete_file: bool = True) -> dict[str, Any]:
        """Unregister a model and (by default) delete its file(s).

        Callers (the API) additionally require explicit confirmation before
        deleting a file that is not in the primary download directory.
        """
        with self._lock:
            rec = self.get(model_id)
            if rec is None:
                raise KeyError(model_id)
            removed: list[str] = []
            path = Path(rec["path"])
            if delete_file:
                m = SPLIT_RE.match(path.name)
                targets = ([p for p in path.parent.glob("*.gguf")
                            if (sm := SPLIT_RE.match(p.name)) and sm["base"] == m["base"]  # type: ignore[index]
                            and sm["total"] == m["total"]] if m else [path])  # type: ignore[index]
                for t in targets:
                    t.unlink(missing_ok=True)
                    removed.append(str(t))
            self.unregister(model_id)
            return {"id": model_id, "deleted_files": removed, "location": rec["location"]}

    # -- active model ---------------------------------------------------------
    def active_id(self) -> str | None:
        return read_json(self.layout.model_state_file, {}).get("active")

    def set_active(self, model_id: str | None) -> None:
        with self._lock:
            state = read_json(self.layout.model_state_file, {})
            state["active"] = model_id
            state["updated_at"] = time.time()
            write_private(self.layout.model_state_file, json.dumps(state, indent=2))


def move_model_files(files: list[Path], target_dir: Path, *, is_running=lambda p: False) -> dict[str, list]:
    """Move model files into ``target_dir`` (rename, or copy+unlink across drives).

    Never overwrites: a same-size file already there counts as present, a
    different-size one is refused. A source is only removed after its copy is
    complete. Returns per-file ``moved`` / ``failed``.
    """
    moved, failed = [], []
    for src in files:
        dst = target_dir / src.name
        if is_running(src):
            failed.append({"source": str(src), "error": "The model is currently running; stop it first."})
            continue
        if not src.exists():
            failed.append({"source": str(src), "error": "File not found."})
            continue
        if dst.exists():
            if dst.stat().st_size == src.stat().st_size:
                moved.append({"source": str(src), "target": str(dst), "already_present": True})
            else:
                failed.append({"source": str(src),
                               "error": "A different file with the same name exists at the destination."})
            continue
        try:
            try:
                os.rename(src, dst)
            except OSError:
                shutil.copy2(src, dst)
                if dst.stat().st_size != src.stat().st_size:
                    dst.unlink(missing_ok=True)
                    raise OSError("copy size mismatch")
                src.unlink()
            moved.append({"source": str(src), "target": str(dst)})
        except OSError as exc:
            failed.append({"source": str(src), "error": str(exc)})
    return {"moved": moved, "failed": failed}


class ProviderStore:
    """Cloud / remote OpenAI-compatible connections. API keys never leave this file."""

    def __init__(self, layout: RuntimeLayout):
        self.layout = layout
        self._lock = threading.Lock()

    def _load(self) -> dict[str, dict[str, Any]]:
        return read_json(self.layout.providers_file, {})

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        write_private(self.layout.providers_file, json.dumps(data, indent=2))

    @staticmethod
    def model_id(pid: str) -> str:
        return f"cloud:{pid}"

    def add(self, *, label: str, base_url: str, model: str, api_key: str = "",
            provider: str = "openai-compatible") -> dict[str, Any]:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if not model.strip():
            raise ValueError("model is required")
        with self._lock:
            data = self._load()
            pid = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:32] or "provider"
            if pid in data:
                pid = f"{pid}-{secrets.token_hex(2)}"
            data[pid] = {"id": pid, "label": label, "provider": provider, "base_url": base_url.rstrip("/"),
                         "model": model.strip(), "api_key": api_key, "added_at": time.time()}
            self._save(data)
            return self.public(data[pid])

    def remove(self, pid: str) -> bool:
        with self._lock:
            data = self._load()
            if pid not in data:
                return False
            del data[pid]
            self._save(data)
            return True

    def get_private(self, pid: str) -> dict[str, Any] | None:
        return self._load().get(pid)

    @staticmethod
    def public(rec: dict[str, Any]) -> dict[str, Any]:
        key = rec.get("api_key") or ""
        return {"id": ProviderStore.model_id(rec["id"]), "provider_id": rec["id"], "kind": "provider",
                "name": rec["label"], "provider": rec["provider"], "base_url": rec["base_url"],
                "model": rec["model"], "has_api_key": bool(key),
                "key_hint": ("…" + key[-4:]) if len(key) >= 12 else None, "installed": True}

    def list(self) -> list[dict[str, Any]]:
        return [self.public(r) for r in self._load().values()]
