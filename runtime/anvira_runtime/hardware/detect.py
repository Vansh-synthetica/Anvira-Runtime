"""Centralised hardware detection (CPU / RAM / GPU / VRAM / storage).

Ported from Anvira's ``electron/hardwareDetection.cjs`` (nvidia-smi first,
WMI fallback on Windows) and extended for macOS/Linux. Only *facts* are
reported; model compatibility is decided elsewhere from model metadata.
"""
from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

Runner = Callable[[list[str], float], str]
_CACHE: dict[str, tuple[float, dict]] = {}
CACHE_TTL_S = 30.0
_MIB = 1024 * 1024


def _run(cmd: list[str], timeout: float = 6.0) -> str:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                         creationflags=flags, check=False)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or f"exit {out.returncode}")
    return out.stdout


def find_nvidia_smi() -> str:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if sys.platform == "win32":
        for cand in (
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ):
            if os.path.exists(cand):
                return cand
    return "nvidia-smi"


def parse_nvidia_csv_line(line: str) -> list[str]:
    """nvidia-smi quotes fields containing commas; split quote-aware."""
    import csv
    return [p.strip() for p in next(csv.reader([line], skipinitialspace=True))]


def _cpu(runner: Runner) -> dict:
    logical = os.cpu_count() or 1
    model = platform.processor() or ""
    try:
        if sys.platform == "win32":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                model = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        elif sys.platform == "darwin":
            model = runner(["sysctl", "-n", "machdep.cpu.brand_string"], 3).strip() or model
        elif Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    physical = max(1, int(logical * 0.75))  # same heuristic as the Electron app
    return {
        "model": re.sub(r"\s+", " ", model).strip() or "Unknown CPU",
        "logical_cores": logical,
        "physical_cores_estimate": physical,
        "optimal_threads": physical,
    }


class _MemStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _ram(runner: Runner) -> dict:
    total = free = 0
    try:
        if sys.platform == "win32":
            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # type: ignore[attr-defined]
            total, free = st.ullTotalPhys, st.ullAvailPhys
        elif sys.platform == "darwin":
            total = int(runner(["sysctl", "-n", "hw.memsize"], 3).strip())
            vm = runner(["vm_stat"], 3)
            page = int(re.search(r"page size of (\d+)", vm).group(1))  # type: ignore[union-attr]
            pages = sum(int(m) for m in re.findall(r"Pages (?:free|inactive|speculative):\s+(\d+)", vm))
            free = pages * page
        else:
            info = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                k, _, v = line.partition(":")
                info[k] = int(v.strip().split()[0]) * 1024
            total, free = info["MemTotal"], info.get("MemAvailable", info.get("MemFree", 0))
    except Exception:
        pass
    return {
        "total_mib": round(total / _MIB), "free_mib": round(free / _MIB),
        "total_gib": round(total / 1024**3, 1), "free_gib": round(free / 1024**3, 1),
    }


def _gpu_nvidia(runner: Runner) -> dict | None:
    smi = find_nvidia_smi()
    try:
        out = runner([smi, "--query-gpu=name,memory.total,memory.free,driver_version",
                      "--format=csv,noheader,nounits"], 6)
    except Exception:
        return None
    line = out.strip().splitlines()[0] if out.strip() else ""
    parts = parse_nvidia_csv_line(line) if line else []
    if len(parts) < 4:
        return None
    try:
        name, total, free, driver = parts[0], int(parts[1]), int(parts[2]), parts[3]
    except ValueError:
        return None
    cuda = None
    try:
        m = re.search(r"CUDA Version:\s*([\d.]+)", runner([smi], 6), re.I)
        cuda = m.group(1) if m else None
    except Exception:
        pass
    return {
        "vendor": "nvidia", "name": name, "vram_total_mib": total, "vram_free_mib": free,
        "driver_version": driver, "cuda_version": cuda, "backend": "cuda",
    }


def _gpu_windows_wmi(runner: Runner) -> dict | None:
    if sys.platform != "win32":
        return None
    script = ("Get-CimInstance Win32_VideoController | ForEach-Object "
              "{ \"$($_.Name)|$($_.AdapterRAM)|$($_.DriverVersion)\" }")
    try:
        out = runner(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], 8)
    except Exception:
        return None
    best = None
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3 or not parts[0]:
            continue
        name = parts[0]
        vendor = ("nvidia" if "nvidia" in name.lower() else "amd" if ("amd" in name.lower() or "radeon" in name.lower())
                  else "intel" if "intel" in name.lower() else "unknown")
        if vendor == "unknown" or "basic" in name.lower():
            continue
        try:
            vram = round(int(parts[1]) / _MIB)
        except ValueError:
            vram = 0
        cand = {"vendor": vendor, "name": name, "vram_total_mib": vram, "vram_free_mib": None,
                "driver_version": parts[2] or None, "cuda_version": None,
                "backend": "cpu", "note": "adapter RAM from WMI is capped at 4 GiB and may be inaccurate"}
        if best is None or (vendor in ("nvidia", "amd") and best["vendor"] == "intel"):
            best = cand
    return best


def _gpu(runner: Runner) -> dict:
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return {"vendor": "apple", "name": "Apple Silicon (unified memory)", "vram_total_mib": None,
                "vram_free_mib": None, "driver_version": None, "cuda_version": None, "backend": "metal"}
    gpu = _gpu_nvidia(runner) or _gpu_windows_wmi(runner)
    return gpu or {"vendor": "none", "name": None, "vram_total_mib": 0, "vram_free_mib": 0,
                   "driver_version": None, "cuda_version": None, "backend": "cpu"}


def disk_info(path: Path) -> dict:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
        return {"path": str(path), "total_gib": round(usage.total / 1024**3, 1),
                "free_gib": round(usage.free / 1024**3, 1), "free_bytes": usage.free}
    except OSError as exc:
        return {"path": str(path), "error": str(exc), "free_bytes": 0, "free_gib": 0.0, "total_gib": 0.0}


def detect_hardware(models_dir: Path | None = None, *, runner: Runner = _run,
                    use_cache: bool = True) -> dict:
    """Return detected hardware facts. Cached for ``CACHE_TTL_S`` seconds."""
    key = str(models_dir)
    if use_cache and key in _CACHE and time.time() - _CACHE[key][0] < CACHE_TTL_S:
        return _CACHE[key][1]
    gpu = _gpu(runner)
    backends = ["cpu"]
    if gpu["backend"] not in backends:
        backends.insert(0, gpu["backend"])
    info = {
        "platform": {"os": {"win32": "windows", "darwin": "macos"}.get(sys.platform, sys.platform),
                     "arch": platform.machine().lower() or "unknown", "release": platform.release(),
                     "python": platform.python_version()},
        "cpu": _cpu(runner),
        "ram": _ram(runner),
        "gpu": gpu,
        "acceleration_backends": backends,
        "storage": {"models": disk_info(models_dir)} if models_dir else {},
        "detected_at": time.time(),
    }
    _CACHE[key] = (time.time(), info)
    return info
