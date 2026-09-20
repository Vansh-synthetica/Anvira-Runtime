"""Model compatibility from hardware facts + model metadata.

No performance promises: this only answers "does the model's memory
footprint fit the detected VRAM / RAM / disk", and says which resource it
would run on. Overhead numbers are the same conservative ones Anvira's
launcher used (weights + KV cache + ~256 MiB).
"""
from __future__ import annotations

import re
from typing import Any

_MIB = 1024 * 1024
KV_AND_OVERHEAD_MIB = 768     # KV cache at default context + runtime overhead
GPU_FILL = 0.90               # fraction of total VRAM usable for full offload
RAM_FILL = 0.80               # fraction of total RAM usable for CPU inference
MIN_PARTIAL_VRAM_MIB = 1500

_PARAMS_RE = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*[bB](?![a-z0-9])")
_MOE_RE = re.compile(r"\d+x(\d+(?:\.\d+)?)[bB]")


def estimate_params_b(model_id: str, tags: list[str] | None = None) -> float | None:
    """Best-effort parameter count (billions) from a repo/model id.

    Port of Anvira's ``utils/modelSize.ts`` heuristic: GGUF repo names almost
    always spell the size out ("Qwen2.5-1.5B-Instruct-GGUF").
    """
    haystack = " ".join([model_id.replace("_", "-").replace("/", " ")] + list(tags or []))
    moe = _MOE_RE.search(haystack)
    if moe:
        return float(moe.group(1))
    m = _PARAMS_RE.search(haystack.replace("-", " "))
    return float(m.group(1)) if m else None


def assess(model: dict[str, Any], hardware: dict[str, Any]) -> dict[str, Any]:
    """Return ``{can_run, mode, needs_mib, reasons, ...}`` for ``model``."""
    if model.get("kind") == "provider":
        return {"can_run": True, "mode": "remote", "needs_mib": 0, "reasons": ["Runs on a remote provider."]}

    size = model.get("size_bytes") or 0
    weights_mib = size / _MIB
    needs = round(weights_mib + KV_AND_OVERHEAD_MIB) if size else None
    reasons: list[str] = []
    if needs is None:
        return {"can_run": None, "mode": "unknown", "needs_mib": None,
                "reasons": ["Model size unknown; cannot assess."]}

    ram = hardware.get("ram", {})
    gpu = hardware.get("gpu", {})
    storage = (hardware.get("storage") or {}).get("models") or {}
    ram_total = ram.get("total_mib") or 0
    ram_free = ram.get("free_mib") or 0
    backend = gpu.get("backend")
    vram = gpu.get("vram_total_mib") or 0

    disk_ok = True
    if not model.get("installed") and storage.get("free_bytes") is not None and size:
        disk_ok = storage["free_bytes"] > size * 1.05
        if not disk_ok:
            reasons.append(f"Not enough free disk space ({storage.get('free_gib')} GiB free, "
                           f"~{round(size / 1024**3, 1)} GiB needed).")

    mode = "insufficient"
    if backend == "metal":
        if needs <= ram_total * 0.70:
            mode = "gpu"
            reasons.append("Fits in Apple unified memory.")
    elif backend == "cuda" and vram:
        if needs <= vram * GPU_FILL:
            mode = "gpu"
            reasons.append(f"Fits fully in VRAM ({needs} MiB needed of {vram} MiB).")
        elif vram >= MIN_PARTIAL_VRAM_MIB and needs <= ram_total * RAM_FILL + vram * GPU_FILL:
            mode = "partial-gpu"
            reasons.append("Larger than VRAM; some layers would be offloaded to the GPU, the rest run on CPU/RAM.")
    if mode == "insufficient" and needs <= ram_total * RAM_FILL:
        mode = "cpu"
        reasons.append(f"Fits in system RAM ({needs} MiB needed of {ram_total} MiB total).")
    if mode == "insufficient":
        reasons.append(f"Needs about {needs} MiB; this machine reports {ram_total} MiB RAM"
                       + (f" and {vram} MiB VRAM." if vram else " and no usable GPU memory."))
    elif mode in ("cpu", "partial-gpu") and needs > ram_free > 0:
        reasons.append(f"Only {ram_free} MiB RAM is free right now; close other applications first.")

    return {
        "can_run": mode != "insufficient" and disk_ok,
        "mode": mode, "needs_mib": needs, "disk_ok": disk_ok, "reasons": reasons,
    }


def recommend(models: list[dict[str, Any]], hardware: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    """Rank runnable local models: full-GPU first, then by size (bigger = more capable)."""
    order = {"gpu": 0, "partial-gpu": 1, "cpu": 2}
    scored = []
    for m in models:
        if m.get("kind") == "provider":
            continue
        a = assess(m, hardware)
        if a["can_run"]:
            scored.append((order.get(a["mode"], 3), -(m.get("params_b") or 0), m, a))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [{**m, "compatibility": a} for _, _, m, a in scored[:limit]]
