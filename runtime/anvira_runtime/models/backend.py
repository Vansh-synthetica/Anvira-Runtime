"""llama.cpp ``llama-server`` backend: discovery, launch flags, GPU offload.

Behaviour ported from Anvira's ``electron/llamaServer.cjs``: same flags
(``--jinja``, flash-attention + q8_0 KV cache on CUDA, ``-b 512``), same
4096-token context floor.

A model must never take the computer down with it (1.0.4, after a tester's PC froze when the app opened with a
Discord call running). So the launch is sized from what is FREE right now, not from what the hardware has:
- GPU layers fit the VRAM that is free minus room for the display and other apps, with the KV cache (all layers,
  measured in bytes) and llama.cpp's working buffers counted. Overfilling the card makes Windows spill VRAM into
  system memory and the whole desktop stalls while only the mouse moves.
- System memory is checked before loading: if the model will not fit in what is free (keeping room for Windows and
  the apps in use), a smaller context is tried, then the load is refused with a clear message instead of thrashing.
- Two logical cores are left free and the process runs at below-normal priority (supervisor ``low_priority``).

Unlike the old launcher it serves several requests at once: ``-np N`` slots
over one unified KV pool (``--kv-unified``), so an app generating a quiz in
parallel chunks, or a chat running while autocomplete fires, is batched by
the server instead of queued behind a single slot. With a unified pool every
slot may use the whole context, so a long prompt never gets less room than
it had with one slot.
"""
from __future__ import annotations

import os
import shutil
import subprocess
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


# N-gram ("prompt lookup") speculative decoding: llama-server guesses the next tokens by finding the last 3 written
# tokens earlier in the prompt or answer and proposing the 24 that followed; the model checks them in one batch.
# Study items, notes chat and rewrites copy whole phrases from the notes they are given, so many guesses are right.
# Measured 2026-11-24 with Anvira Notes' real flashcard prompts, Llama 3.2 3B Q4_K_M, RTX 3050 4 GB, temperature 0:
# GPU 19.5 s -> 11.4 s (one request) and 11.7 s -> 7.3 s (three at once); quiz prompts even (19.4 vs 19.4 s).
# On the CPU the same flags made it ~2x SLOWER (31 s -> 57 s): checking a draft costs as much as writing it there.
SPECULATIVE_ARGS = ["--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", "3", "--spec-ngram-simple-size-m", "24"]
_flag_cache: dict[tuple[str, int, int], str] = {}


def _help_text(binary: Path) -> str:
    """``llama-server --help`` output (cached per binary file), '' when it cannot be run."""
    try:
        st = binary.stat()
        key = (str(binary), int(st.st_mtime), st.st_size)
    except OSError:
        return ""
    if key not in _flag_cache:
        try:
            r = subprocess.run([str(binary), "--help"], capture_output=True, text=True, timeout=15, cwd=str(binary.parent),
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            _flag_cache[key] = (r.stdout or "") + (r.stderr or "")
        except (OSError, subprocess.SubprocessError, ValueError):
            _flag_cache[key] = ""
    return _flag_cache[key]


def supports_speculative(binary: Path) -> bool:
    """Older or custom llama-server builds do not know these flags and would refuse to start with them."""
    help_text = _help_text(binary)
    return "--spec-type" in help_text and "ngram-simple" in help_text


KV_BYTES_GPU = 1.0625      # q8_0 K and V cache (--cache-type-k/v q8_0)
KV_BYTES_CPU = 2.0         # llama.cpp's f16 default on the CPU
GPU_WORK_MIB = 512         # CUDA context + llama.cpp compute buffers at -b 512
HOST_WORK_MIB = 400        # llama-server itself, tokenizer, compute buffers on the CPU side


def kv_cache_mib(meta: dict | None, context: int, bytes_per_value: float) -> float:
    """Size of the K+V cache for ``context`` tokens across ALL layers."""
    m = meta or {}
    n_layer = m.get("n_layer") or 32
    n_head_kv = m.get("n_head_kv") or 8
    n_embd, n_head = m.get("n_embd"), m.get("n_head")
    head_dim = (n_embd // n_head) if n_embd and n_head else 128
    return context * n_layer * 2 * n_head_kv * head_dim * bytes_per_value / _MIB


def usable_vram_mib(gpu: dict[str, Any]) -> int:
    """VRAM a model may take now: what is free, minus room for the display and the apps already using the card."""
    total = gpu.get("vram_total_mib") or 0
    if not total:
        return 0
    free = gpu.get("vram_free_mib")
    if free is None:                          # not reported (non-NVIDIA query): assume the desktop already uses some
        free = total * 0.7
    headroom = max(768.0, total * 0.10)       # on top of what is in use now: room for a call or a browser to grow
    return max(0, int(min(free, total) - headroom))


def ram_headroom_mib(total_mib: int) -> float:
    """System memory left free after loading, on top of what the apps in use already hold: Windows stays responsive
    with about 1 GB truly available; less and it starts paging the apps on screen."""
    return max(1024.0, total_mib * 0.06)


def compute_gpu_layers(model_path: Path, vram_mib: int, context: int, meta: dict | None) -> int:
    """Layers to offload into ``vram_mib`` (the USABLE VRAM, see usable_vram_mib), counting the KV cache and buffers."""
    try:
        model_mib = model_path.stat().st_size / _MIB
    except OSError:
        return 0
    if model_mib <= 0 or not vram_mib:
        return 0
    n_layer = (meta or {}).get("n_layer") or 32
    per_layer = model_mib / n_layer
    budget = vram_mib - kv_cache_mib(meta, context, KV_BYTES_GPU) - GPU_WORK_MIB
    if budget < per_layer * 2:                 # a layer or two on the GPU is not worth a CUDA context
        return 0
    return max(1, min(n_layer, int(budget // per_layer)))


def host_ram_needed_mib(model_mib: float, meta: dict | None, context: int, layers_on_gpu: int, on_gpu: bool) -> float:
    """System memory the model will occupy: the weights not on the GPU, their share of the KV cache, and buffers."""
    n_layer = (meta or {}).get("n_layer") or 32
    on_cpu = 1.0 - (min(layers_on_gpu, n_layer) / n_layer if on_gpu else 0.0)
    kv = kv_cache_mib(meta, context, KV_BYTES_GPU if on_gpu else KV_BYTES_CPU)
    return model_mib * on_cpu + kv * on_cpu + HOST_WORK_MIB


def pick_context(meta: dict | None, free_vram_mib: int | None, requested: int | None) -> int:
    """Requested context, else the largest of 16384/8192/4096 whose KV cache fits 12% of the usable VRAM."""
    if requested:
        return requested
    if not free_vram_mib:
        return 4096
    for ctx in (16384, 8192, 4096):
        if kv_cache_mib(meta, ctx, KV_BYTES_GPU) <= free_vram_mib * 0.12:
            return ctx
    return 4096


def pick_parallel(use_cuda: bool, ctx: int, requested: int | None) -> int:
    """Server slots. Requested wins; otherwise 4 on a GPU with a 16k+ pool, 2 with 8k+, else 1.

    Slots share one unified KV pool, so the pool must hold several typical prompts at
    once; below 8k tokens that is not guaranteed and a single slot is the safe choice.
    """
    if requested:
        return max(1, min(8, requested))
    if ctx >= 16384 and use_cuda:
        return 4
    return 2 if ctx >= 8192 else 1


def build_launch(layout: RuntimeLayout, config: RuntimeConfig, model: dict[str, Any],
                 hardware: dict[str, Any], logs_dir: Path, force_cpu: bool = False) -> tuple[ServiceSpec | None, dict[str, Any]]:
    """Build the llama-server ServiceSpec for ``model``. Returns (spec|None, info).

    ``spec`` is None when no binary can be found; ``info['error']`` says why.
    """
    gpu = hardware.get("gpu", {})
    gpu_mode = config.get("models.gpu")
    cuda_ok = gpu.get("backend") == "cuda" and bool(gpu.get("vram_total_mib")) and gpu_mode != "off" and not force_cpu
    binary, variant, searched = find_llama_server(layout, config, prefer_cuda=cuda_ok)
    info: dict[str, Any] = {"binary": str(binary) if binary else None, "variant": variant, "searched": searched}
    if binary is None:
        info["error"] = "llama-server was not found"
        return None, info

    path = Path(model["path"])
    meta = gguf.read_metadata(path)
    want_cuda = cuda_ok and variant in ("cuda", "custom", "unknown")
    vram = usable_vram_mib(gpu) if want_cuda else 0
    requested_ctx = config.get("models.context_size")

    def size(ctx_: int) -> tuple[int, bool]:
        layers_ = compute_gpu_layers(path, vram, ctx_, meta) if want_cuda else 0
        return layers_, want_cuda and layers_ > 0

    ctx = pick_context(meta, vram or None, requested_ctx)
    layers, use_cuda = size(ctx)
    if not use_cuda and not requested_ctx:
        # On the CPU the KV cache lives in system RAM, where 8k tokens of a small model is a few hundred MiB:
        # cheap room for two prompts at once instead of one (the memory check below takes it back if it won't fit).
        ctx = max(ctx, 8192)

    # Will it fit in the system memory that is free right now? If not, try the smallest context, then refuse:
    # loading anyway pushes Windows into the page file and the whole PC stops responding.
    ram = hardware.get("ram") or {}
    ram_free, ram_total = ram.get("free_mib") or 0, ram.get("total_mib") or 0
    try:
        model_mib = path.stat().st_size / _MIB
    except OSError:
        model_mib = 0.0
    if config.get("models.memory_guard") and ram_free and model_mib:
        room = ram_free - ram_headroom_mib(ram_total or ram_free)
        need = host_ram_needed_mib(model_mib, meta, ctx, layers, use_cuda)
        if need > room and not requested_ctx and ctx > 4096:
            ctx = 4096
            layers, use_cuda = size(ctx)
            need = host_ram_needed_mib(model_mib, meta, ctx, layers, use_cuda)
        info.update({"ram_needed_mib": round(need), "ram_free_mib": ram_free})
        if need > room:
            info["code"] = "not_enough_memory"
            info["error"] = (f"Not enough free memory to load this model: it needs about {need / 1024:.1f} GB "
                             f"and {ram_free / 1024:.1f} GB is free, with "
                             f"{ram_headroom_mib(ram_total or ram_free) / 1024:.1f} GB kept for Windows and your apps.")
            return None, info
    parallel = pick_parallel(use_cuda, ctx, config.get("models.parallel"))

    cpu = hardware.get("cpu", {})
    threads = cpu.get("optimal_threads") or 4
    if cpu.get("logical_cores"):
        threads = min(threads, cpu["logical_cores"] - 2)    # two logical cores stay free for the desktop and apps
    threads = max(1, threads)
    if use_cuda:
        threads = max(2, threads // 2)
    port = free_port()
    argv = [str(binary), "-m", str(path), "--host", "127.0.0.1", "--port", str(port), "--jinja", "-c", str(ctx)]
    if use_cuda:
        argv += ["-ngl", str(layers), "-fa", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    argv += ["-np", str(parallel)]
    if parallel > 1:
        argv += ["--kv-unified"]   # every slot may use the full context; without it each gets ctx / np
    argv += ["-t", str(threads), "-b", "512"]
    spec_mode = config.get("models.speculative") or "auto"
    n_layer = (meta or {}).get("n_layer") or 32
    want_spec = spec_mode == "on" or (spec_mode == "auto" and use_cuda and layers >= n_layer)
    speculative = want_spec and supports_speculative(binary)
    if speculative:
        argv += SPECULATIVE_ARGS
    info["speculative"] = "ngram-simple" if speculative else None

    info.update({"port": port, "context": ctx, "gpu_layers": layers if use_cuda else 0, "parallel": parallel,
                 "gpu": use_cuda, "threads": threads, "architecture": (meta or {}).get("architecture")})
    spec = ServiceSpec(
        name=f"model:{model['id']}", argv=argv, cwd=str(binary.parent), log_path=logs_dir / f"model-{model['id']}.log",
        port=port, health_url=f"http://127.0.0.1:{port}/health", ready_timeout_s=600.0, restart=True,
        # a busy model can miss health checks: never kill and reload it for that, and if it really dies restart it
        # once, not five times (each reload re-reads gigabytes); it runs below normal priority so the PC stays usable
        restart_limit=1, restart_when_unhealthy=False, low_priority=True,
        kind="model", meta={"model_id": model["id"], "context": ctx, "gpu_layers": info["gpu_layers"],
                            "parallel": parallel, "variant": variant})
    return spec, info


def base_port_taken() -> bool:  # used by doctor
    return port_in_use(BASE_PORT)
