"""llama.cpp ``llama-server`` backend: discovery, launch flags, GPU offload.

Behaviour ported from Anvira's ``electron/llamaServer.cjs``: same flags
(``--jinja``, flash-attention + q8_0 KV cache on CUDA, ``-b 512``), same
empirically-calibrated GPU-layer fill (82% of total VRAM on cards up to
6 GiB, 90% above), same 4096-token context floor.

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
    use_cuda = cuda_ok and variant in ("cuda", "custom", "unknown")
    requested_ctx = config.get("models.context_size")
    ctx = pick_context(meta, gpu.get("vram_free_mib") if use_cuda else None, requested_ctx)
    if not use_cuda and not requested_ctx:
        # On the CPU the KV cache lives in system RAM, where 8k tokens of a small model is a few hundred MiB:
        # cheap room for two prompts at once instead of one.
        ctx = max(ctx, 8192)
    layers = compute_gpu_layers(path, gpu.get("vram_total_mib") or 0, ctx, meta) if use_cuda else 0
    use_cuda = use_cuda and layers > 0
    parallel = pick_parallel(use_cuda, ctx, config.get("models.parallel"))

    threads = hardware.get("cpu", {}).get("optimal_threads") or 4
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
        kind="model", meta={"model_id": model["id"], "context": ctx, "gpu_layers": info["gpu_layers"],
                            "parallel": parallel, "variant": variant})
    return spec, info


def base_port_taken() -> bool:  # used by doctor
    return port_in_use(BASE_PORT)
