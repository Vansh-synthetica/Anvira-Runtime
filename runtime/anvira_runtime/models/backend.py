"""llama.cpp ``llama-server`` backend: discovery, launch flags, GPU offload.

Behaviour ported from Anvira's ``electron/llamaServer.cjs``: same flags
(``--jinja``, ``-np 1``, flash-attention + q8_0 KV cache on CUDA, ``-b 512``),
same empirically-calibrated GPU-layer fill (82% of total VRAM on cards up to
6 GiB, 90% above), same 4096-token context floor.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

from ..config.paths import RuntimeLayout
from ..config.settings import RuntimeConfig
from ..process.supervisor import ServiceSpec, free_port, port_in_use
from . import gguf

_MIB = 1024 * 1024
BASE_PORT = 18080  # avoids 8080 (commonly taken; Anvira's legacy default)


def _exe() -> str:
    return "llama-server.exe" if sys.platform == "win32" else "llama-server"


def candidate_binaries(layout: RuntimeLayout, config: RuntimeConfig, prefer_cuda: bool) -> list[tuple[Path, str]]:
    """Ordered (path, variant) candidates for llama-server."""
    exe = _exe()
    out: list[tuple[Path, str]] = []
    explicit = config.get("models.llama_server_path") or os.environ.get("ANVIRA_LLAMA_SERVER")
    if explicit:
        out.append((Path(explicit), "custom"))
    variants = ("cuda", "cpu") if prefer_cuda else ("cpu", "cuda")
    plat = {"win32": "win32", "darwin": "darwin"}.get(sys.platform, "linux")
    roots = [layout.bin_dir / "llama-cpp"]
    appdata = os.environ.get("APPDATA")
    localapp = os.environ.get("LOCALAPPDATA")
    if appdata:  # binaries an existing Anvira install already downloaded
        roots.append(Path(appdata) / "Anvira" / "bin" / "llama-cpp")
    if localapp:  # binaries bundled with a packaged Anvira
        roots.append(Path(localapp) / "Programs" / "Anvira" / "resources" / "llama-cpp" / plat)
    for root in roots:
        for v in variants:
            out.append((root / v / exe, v))
        out.append((root / exe, "unknown"))
    return out


def find_llama_server(layout: RuntimeLayout, config: RuntimeConfig, prefer_cuda: bool) -> tuple[Path | None, str, list[str]]:
    """Return (binary, variant, searched_paths)."""
    searched: list[str] = []
    for path, variant in candidate_binaries(layout, config, prefer_cuda):
        searched.append(str(path))
        if path.is_file():
            return path, variant, searched
    on_path = shutil.which("llama-server")
    searched.append("PATH:llama-server")
    if on_path:
        return Path(on_path), "unknown", searched
    return None, "none", searched


def compute_gpu_layers(model_path: Path, vram_total_mib: int, context: int, meta: dict | None) -> int:
    try:
        model_mib = model_path.stat().st_size / _MIB
    except OSError:
        return 0
    if model_mib <= 0 or not vram_total_mib:
        return 0
    n_layer = (meta or {}).get("n_layer") or 32
    per_layer = model_mib / n_layer
    fill = 0.82 if vram_total_mib <= 6144 else 0.90
    n_head_kv = (meta or {}).get("n_head_kv") or 8
    n_embd, n_head = (meta or {}).get("n_embd"), (meta or {}).get("n_head")
    head_dim = (n_embd // n_head) if n_embd and n_head else 128
    kv_mib = max(64.0, context * (2 * n_head_kv * head_dim) / _MIB)
    budget = vram_total_mib * fill - kv_mib
    if budget <= model_mib * 0.05:
        return 0
    return max(1, min(n_layer, int(budget // per_layer)))


def pick_context(meta: dict | None, free_vram_mib: int | None, requested: int | None) -> int:
    """Requested context, else the largest of 16384/8192/4096 whose KV fits 12% of free VRAM."""
    if requested:
        return requested
    if not free_vram_mib:
        return 4096
    n_head_kv = (meta or {}).get("n_head_kv") or 8
    n_embd, n_head = (meta or {}).get("n_embd"), (meta or {}).get("n_head")
    head_dim = (n_embd // n_head) if n_embd and n_head else 128
    per_tok = 2 * n_head_kv * head_dim
    for ctx in (16384, 8192, 4096):
        if ctx * per_tok / _MIB <= free_vram_mib * 0.12:
            return ctx
    return 4096


def build_launch(layout: RuntimeLayout, config: RuntimeConfig, model: dict[str, Any],
                 hardware: dict[str, Any], logs_dir: Path) -> tuple[ServiceSpec | None, dict[str, Any]]:
    """Build the llama-server ServiceSpec for ``model``. Returns (spec|None, info).

    ``spec`` is None when no binary can be found; ``info['error']`` says why.
    """
    gpu = hardware.get("gpu", {})
    gpu_mode = config.get("models.gpu")
    cuda_ok = gpu.get("backend") == "cuda" and bool(gpu.get("vram_total_mib")) and gpu_mode != "off"
    binary, variant, searched = find_llama_server(layout, config, prefer_cuda=cuda_ok)
    info: dict[str, Any] = {"binary": str(binary) if binary else None, "variant": variant, "searched": searched}
    if binary is None:
        info["error"] = "llama-server was not found"
        return None, info

    path = Path(model["path"])
    meta = gguf.read_metadata(path)
    use_cuda = cuda_ok and variant in ("cuda", "custom", "unknown")
    ctx = pick_context(meta, gpu.get("vram_free_mib") if use_cuda else None, config.get("models.context_size"))
    layers = compute_gpu_layers(path, gpu.get("vram_total_mib") or 0, ctx, meta) if use_cuda else 0
    use_cuda = use_cuda and layers > 0

    threads = hardware.get("cpu", {}).get("optimal_threads") or 4
    if use_cuda:
        threads = max(2, threads // 2)
    port = free_port()
    argv = [str(binary), "-m", str(path), "--host", "127.0.0.1", "--port", str(port), "--jinja", "-c", str(ctx)]
    if use_cuda:
        argv += ["-ngl", str(layers), "-fa", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    argv += ["-np", "1", "-t", str(threads), "-b", "512"]

    info.update({"port": port, "context": ctx, "gpu_layers": layers if use_cuda else 0,
                 "gpu": use_cuda, "threads": threads, "architecture": (meta or {}).get("architecture")})
    spec = ServiceSpec(
        name=f"model:{model['id']}", argv=argv, cwd=str(binary.parent), log_path=logs_dir / f"model-{model['id']}.log",
        port=port, health_url=f"http://127.0.0.1:{port}/health", ready_timeout_s=600.0, restart=True,
        kind="model", meta={"model_id": model["id"], "context": ctx, "gpu_layers": info["gpu_layers"],
                            "variant": variant})
    return spec, info


def base_port_taken() -> bool:  # used by doctor
    return port_in_use(BASE_PORT)
