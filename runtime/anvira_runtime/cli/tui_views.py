"""Extra dashboard pages for ``anvira open``: Health, Models, Apps & data, Logs.

Each function returns the *body* of a page (a list of boxed rows, exactly ``height`` rows tall) so the Overview page keeps
its layout. Data comes from ``Dashboard.extra`` (fetched lazily, only for the page being looked at), so an open dashboard
costs the runtime nothing on pages nobody is viewing.
"""
from __future__ import annotations

from typing import Any

MARK = {"ok": "ok  ", "info": "info", "warn": "warn", "fail": "FAIL"}


def _gib(n: float | None) -> str:
    return f"{n / 1024 ** 3:.1f} GiB" if n else "-"


def _halves(d: Any, left: tuple[str, list[str]], right: tuple[str, list[str]], width: int, height: int) -> list[str]:
    half = width // 2
    return d._side_by_side(d._box(left[0], left[1], half, height), d._box(right[0], right[1], width - half, height))


def _hardware_lines(d: Any, hw: dict[str, Any] | None, st: dict[str, Any] | None) -> list[str]:
    if not hw:
        return [d.dim(" reading hardware ...")]
    cpu, ram, gpu = hw.get("cpu", {}), hw.get("ram", {}), hw.get("gpu", {})
    lines = [f" CPU   {cpu.get('model', '?')}", f"       {cpu.get('physical_cores_estimate', '?')} cores / {cpu.get('logical_cores', '?')} threads",
             f" RAM   {ram.get('free_gib', '?')} GiB free of {ram.get('total_gib', '?')} GiB"]
    if gpu.get("vendor") in (None, "none"):
        lines.append(f" GPU   {d.yellow('none detected')} - models run on the CPU")
    else:
        lines.append(f" GPU   {gpu.get('name')}")
        if gpu.get("vram_total_mib"):
            lines.append(f"       {(gpu.get('vram_free_mib') or 0) / 1024:.1f} GiB free of {gpu['vram_total_mib'] / 1024:.1f} GiB VRAM  driver {gpu.get('driver_version') or '?'}")
    info = ((st or {}).get("model") or {}).get("backend_info") or {}
    if info:
        variant = info.get("variant") or "?"
        lines.append(f" using {d.cyan(str(variant).upper())} backend   ctx {info.get('context', '?')}   gpu layers {info.get('gpu_layers', '?')}")
    disk = (hw.get("storage") or {}).get("models") or {}
    if disk.get("free_bytes"):
        lines.append(f" disk  {_gib(disk['free_bytes'])} free where models are stored")
    return lines


def render_health(d: Any, width: int, height: int) -> list[str]:
    ex = d.extra.get("health") or {}
    st = d.status
    diag = ex.get("diag")
    lines: list[str] = []
    if diag is None:
        lines.append(d.dim(" running checkups ..." if not ex.get("error") else " " + ex["error"]))
    else:
        for c in diag["checks"]:
            col = {"ok": d.green, "info": d.dim, "warn": d.yellow, "fail": d.red}[c["status"]]
            lines.append(f" {col(MARK[c['status']])} {c['title']}: {c['message'].splitlines()[0][:60]}")
            if c.get("fix") and c["status"] in ("warn", "fail"):
                lines.append(d.dim(f"        fix: {c['fix'][:70]}"))
        s = diag["summary"]
        lines.insert(0, f" {s['ok'] + s['info']} ok   {s['warn']} warnings   {s['fail']} failures        {d.dim('r = re-run')}")
        lines.insert(1, "")
    right: list[str] = _hardware_lines(d, ex.get("hw"), st)
    if st:
        right += ["", " services"]
        for key in ("orcha", "nomi", "aicl"):
            sv = st["services"].get(key, {})
            right.append(f"  {d.dot(sv.get('state'))} {key:<6} {sv.get('state', '?'):<9} restarts {sv.get('restarts', 0)}")
        mo = st["model"]
        right += ["", f" model  {mo['active'] or 'none'}  {d.dot(mo['state'])} {mo['state']}"]
        if mo.get("backend") and mo["backend"].get("restarts"):
            right.append(d.yellow(f"  model server restarted {mo['backend']['restarts']}x"))
    return _halves(d, ("Checkups", lines), ("Hardware, backend & services", right), width, height)


def render_models(d: Any, width: int, height: int) -> list[str]:
    ex = d.extra.get("models") or {}
    st = d.status
    rows: list[str] = []
    models = ex.get("models")
    if models is None:
        rows.append(d.dim(" loading models ..."))
    elif not models:
        rows += [d.yellow(" No model is installed yet."), "",
                 "   put .gguf files anywhere, then:   anvira model dirs add <folder>",
                 "   or register one file:             anvira model add <file.gguf>",
                 "   or download one:                  anvira model install qwen3-4b"]
    else:
        rows.append(d.dim(f" {'':2}{'model':<38}{'size':>9}  {'runs on':<10} where"))
        for m in models:
            size = f"{m['size_bytes'] / 1024 ** 3:.1f} GiB" if m.get("size_bytes") else "cloud"
            mode = {"gpu": "GPU", "partial-gpu": "GPU+CPU", "cpu": "CPU", "remote": "remote", "insufficient": "too large"}.get(
                (m.get("compatibility") or {}).get("mode"), "?")
            col = d.green if m.get("active") else (d.red if mode == "too large" else (lambda x: x))
            rows.append(col(f" {'*' if m.get('active') else ' '} {m['id'][:37]:<38}{size:>9}  {mode:<10} {str(m.get('origin') or m.get('location') or 'cloud')[-30:]}"))
    right = _hardware_lines(d, ex.get("hw"), st)
    mo = (st or {}).get("model") or {}
    right += ["", f" folder  {str(mo.get('models_dir', '-'))[-(width // 2 - 12):]}"]
    for extra in mo.get("extra_dirs") or []:
        right.append(f"         {str(extra)[-(width // 2 - 12):]}")
    right += ["", d.dim(" use <id>             switch the active model"),
              d.dim(" anvira model install   download one (asks first)"),
              d.dim(" anvira model compat    will it fit this PC?")]
    return _halves(d, ("Installed models", rows), ("This machine", right), width, height)


def render_apps(d: Any, width: int, height: int) -> list[str]:
    ex = d.extra.get("apps") or {}
    apps = ex.get("apps")
    left: list[str] = []
    if apps is None:
        left.append(d.dim(" loading ..."))
    elif not apps:
        left += [d.dim(" No app has connected yet."), "",
                 d.dim(" Anvira, Anvira Notes, Anvira Study and Anvira Dev"), d.dim(" register themselves the first time they open.")]
    else:
        for a in apps:
            req = list(a.get("requested") or [])
            left.append(f" {d.dot('running')} {a['app_id']:<22} {len(a.get('permissions') or [])} permissions"
                        + (d.yellow(f"  {len(req)} waiting for you") if req else ""))
            if req:
                left.append(d.dim(f"     anvira app grant {a['app_id']} {' '.join(req)}"))
    lc = ex.get("lifecycle") or {}
    right: list[str] = []
    if lc:
        right.append(f" mode     {d.cyan(lc.get('mode', '?'))}" + (f"   stops in {lc['shutdown_in_s']}s" if lc.get("shutdown_in_s") is not None else ""))
        right.append(f" open     {len(lc.get('leases') or [])} app(s) holding the runtime,  busy: {lc.get('busy', 0)}")
        for ls in (lc.get("leases") or [])[:4]:
            right.append(d.dim(f"          {ls.get('app')}"))
    res = ex.get("resources") or []
    reqs = ex.get("requests") or []
    if ex:
        by_vis: dict[str, int] = {}
        for r in res:
            by_vis[r["visibility"]] = by_vis.get(r["visibility"], 0) + 1
        right += ["", f" shared data  {len(res)} resources   private {by_vis.get('private', 0)}  shared {by_vis.get('shared', 0)}  global {by_vis.get('global', 0)}"]
        if reqs:
            right.append(d.yellow(f" {len(reqs)} access request(s) waiting:  anvira context requests"))
        right.append("")
    for c in ex.get("capabilities") or []:
        right.append(f" {d.dot('running' if c['state'] in ('running', 'active') else 'stopped')} {c['name']:<10} {c['state']}")
    if not ex:
        right = [d.dim(" loading ...")]
    return _halves(d, ("Connected apps", left), ("Lifecycle, shared data & capabilities", right), width, height)


def render_logs(d: Any, width: int, height: int) -> list[str]:
    ex = d.extra.get("logs") or {}
    svc = ex.get("service", "runtime")
    lines = [d.dim(" " + ln[:width - 6]) for ln in (ex.get("lines") or [])] or [d.dim(" (no log yet)")]
    title = f"Log: {svc}    (logs runtime | orcha | nomi | model:<id> [lines])"
    return d._box(title, lines[-(height - 2):], width, height)


PAGES = {"health": render_health, "models": render_models, "apps": render_apps, "logs": render_logs}
