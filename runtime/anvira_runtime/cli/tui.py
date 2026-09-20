"""``anvira ui`` - live terminal dashboard with a command prompt.

Shows what the runtime is doing (services, active model, jobs, AICL bus traffic,
ORCHA events, logs) and lets you drive it: run ORCHA jobs, chat, switch models,
search memory. Pure stdlib (ANSI escapes + msvcrt/termios), so it runs in
Windows Terminal, cmd, PowerShell, macOS Terminal and Linux terminals.

    anvira ui                  interactive dashboard
    anvira ui --once           print one frame and exit (for logs / CI / screenshots)
    anvira ui --exec "run hi"  run one dashboard command headlessly and print its output
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sys
import threading
import time
from collections import deque
from typing import Any, Callable

from anvira_client.errors import AnviraError
from anvira_client.http import Http

from . import tui_views

_ANSI = re.compile(r"\033\[[0-9;?]*[A-Za-z]")
VIEWS = ("overview", "health", "models", "apps", "logs")
VIEW_TITLES = {"overview": "Overview", "health": "Health", "models": "Models", "apps": "Apps & data", "logs": "Logs"}
HELP = [
    "pages:  Tab / 1-5 switch page (when the prompt is empty)  ->  1 Overview  2 Health  3 Models  4 Apps & data  5 Logs",
    "        r re-runs the current page's checks.   view <name>  also works.",
    "commands:",
    "  run <task>            run an ORCHA job and follow its events   (--graph research|multi_agent, --reasoning high)",
    "  chat <message>        stream a reply from the active model     (--memory recalls memories)",
    "  models                list installed models        use <id>   switch the active model",
    "  jobs | job <id> | cancel <id>",
    "  remember <text>       store a memory               memory <query>   search memory",
    "  logs [runtime|orcha|nomi|model:<id>] [n]           status | doctor",
    "  clear | help | quit        keys: Up/Down history, PgUp/PgDn scroll output, Ctrl+L redraw, Ctrl+C quit",
]


def visible_len(s: str) -> int:
    return len(_ANSI.sub("", s))


def fit(s: str, width: int) -> str:
    """Truncate to ``width`` visible columns (ANSI-aware) and pad with spaces."""
    out, n, i = [], 0, 0
    while i < len(s) and n < width:
        m = _ANSI.match(s, i)
        if m:
            out.append(m.group())
            i = m.end()
            continue
        out.append(s[i])
        n += 1
        i += 1
    return "".join(out) + " " * (width - n) + ("\033[0m" if out and "\033[" in "".join(out) else "")


def _can_unicode() -> bool:
    try:
        "─│┌┐└┘●".encode(getattr(sys.stdout, "encoding", None) or "utf-8")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


class Dashboard:
    """State + rendering + commands. UI-agnostic so it can be tested headlessly."""

    def __init__(self, http_factory: Callable[[], Http], color: bool = True):
        self._http_factory = http_factory
        self._http: Http | None = None
        self.color = color
        self.uni = _can_unicode()
        self.lock = threading.RLock()
        self.status: dict[str, Any] | None = None
        self.jobs: list[dict[str, Any]] = []
        self.aicl: dict[str, Any] | None = None
        self.log_lines: list[str] = []
        self.error: str | None = None
        self.output: deque[str] = deque(maxlen=800)
        self.activity: deque[str] = deque(maxlen=200)
        self.scroll = 0
        self.history: list[str] = []
        self.busy: dict[str, str] = {}          # job id -> short label of things being followed
        self.quit = False
        self._stop = threading.Event()
        self.hold_lease = False
        self._lease_id: str | None = None
        self.view = "overview"
        self.extra: dict[str, dict[str, Any]] = {}     # per-page data, fetched only while that page is open
        self._extra_at: dict[str, float] = {}
        self._fetching: set[str] = set()
        self.log_service = "runtime"
        self.interactive = False               # only the live dashboard switches pages; --exec keeps printing

    # ------------------------------------------------------------------ styling
    def c(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def dim(self, t): return self.c(t, "2")
    def bold(self, t): return self.c(t, "1")
    def green(self, t): return self.c(t, "32")
    def red(self, t): return self.c(t, "31")
    def yellow(self, t): return self.c(t, "33")
    def cyan(self, t): return self.c(t, "36")

    def dot(self, state: str | None) -> str:
        d = "●" if self.uni else "*"
        if state in ("running", "ready", "ok", "remote", "completed"):
            return self.green(d)
        if state in ("starting", "loading", "degraded", "stopped", "queued", "cancelled", "none"):
            return self.yellow(d)
        return self.red(d)

    # --------------------------------------------------------------------- data
    def http(self) -> Http:
        if self._http is None:
            self._http = self._http_factory()
        return self._http

    def refresh(self) -> None:
        """Poll the runtime (called by the background thread and by --once)."""
        try:
            h = self.http()
            st = h.json("GET", "/v1/status", timeout=5)
            jobs = h.json("GET", "/v1/jobs", params={"limit": 12}, timeout=5)["jobs"]
            try:
                aicl = h.json("GET", "/v1/aicl/status", timeout=5)
            except AnviraError:
                aicl = None
            try:
                logs = h.json("GET", "/v1/runtime/logs", params={"service": "runtime", "lines": 12}, timeout=5)["lines"]
            except AnviraError:
                logs = []
            if self.hold_lease:                   # the open dashboard counts as an app: keep an on-demand runtime alive
                try:
                    if self._lease_id:
                        h.json("POST", f"/v1/leases/{self._lease_id}/heartbeat", timeout=5)
                    else:
                        self._lease_id = h.json("POST", "/v1/leases", {"ttl_s": 30}, timeout=5)["lease"]["id"]
                except AnviraError:
                    self._lease_id = None
            with self.lock:
                self.status, self.jobs, self.aicl, self.log_lines, self.error = st, jobs, aicl, logs, None
            self._refresh_view()
        except AnviraError as exc:
            with self.lock:
                self.error = exc.message
                self._http = None
        except Exception as exc:  # noqa: BLE001 - the UI must survive anything
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------- pages
    _PAGE_TTL = {"health": 25.0, "models": 4.0, "apps": 3.0, "logs": 2.0}

    def set_view(self, name: str) -> None:
        if name not in VIEWS:
            return self.say(self.red(f"no page '{name}'. Pages: {', '.join(VIEWS)}"))
        self.view = name
        self._extra_at.pop(name, None)             # fetch fresh data as soon as the page opens
        self._refresh_view()

    def next_view(self, step: int = 1) -> None:
        self.set_view(VIEWS[(VIEWS.index(self.view) + step) % len(VIEWS)])

    def _refresh_view(self, force: bool = False) -> None:
        name = self.view
        ttl = self._PAGE_TTL.get(name)
        if ttl is None or name in self._fetching or self._http_factory is None:
            return
        if not force and time.monotonic() - self._extra_at.get(name, 0.0) < ttl:
            return
        self._extra_at[name] = time.monotonic()
        self._fetching.add(name)

        def work() -> None:
            try:
                h = self.http()
                data: dict[str, Any] = {}
                if name == "health":
                    data["hw"] = h.json("GET", "/v1/hardware", timeout=10)
                    data["diag"] = h.json("GET", "/v1/diagnostics", timeout=90)
                elif name == "models":
                    data["hw"] = h.json("GET", "/v1/hardware", timeout=10)
                    data["models"] = h.json("GET", "/v1/models", params={"installed": "true"}, timeout=10)["models"]
                elif name == "apps":
                    data["apps"] = h.json("GET", "/v1/apps", timeout=10)["apps"]
                    data["lifecycle"] = h.json("GET", "/v1/lifecycle", timeout=10)
                    data["capabilities"] = h.json("GET", "/v1/capabilities", timeout=10)["capabilities"]
                    data["resources"] = h.json("GET", "/v1/resources", timeout=10)["resources"]
                    data["requests"] = h.json("GET", "/v1/resources/requests", timeout=10)["requests"]
                elif name == "logs":
                    svc = self.log_service
                    data = {"service": svc, "lines": h.json("GET", "/v1/runtime/logs", params={"service": svc, "lines": 60}, timeout=10)["lines"]}
                with self.lock:
                    self.extra[name] = data
            except AnviraError as exc:
                with self.lock:
                    self.extra[name] = {**self.extra.get(name, {}), "error": exc.message}
            except Exception as exc:  # noqa: BLE001 - a page must never take the dashboard down
                with self.lock:
                    self.extra[name] = {**self.extra.get(name, {}), "error": f"{type(exc).__name__}: {exc}"}
            finally:
                self._fetching.discard(name)
        threading.Thread(target=work, daemon=True, name=f"anvira-ui-{name}").start()

    def start_polling(self, interval: float = 1.0) -> None:
        self.error = "starting the runtime if it is not running (about 10 seconds)..."

        def loop() -> None:
            while not self._stop.is_set():
                self.refresh()
                self._stop.wait(interval)
        threading.Thread(target=loop, daemon=True, name="anvira-ui-poll").start()

    def stop(self) -> None:
        self._stop.set()
        if self._lease_id and self._http is not None:
            try:
                self._http.json("DELETE", f"/v1/leases/{self._lease_id}", timeout=5)
            except AnviraError:
                pass
            self._lease_id = None

    # ------------------------------------------------------------------- output
    def welcome(self) -> None:
        """First-screen guidance, adapted to what the runtime currently has (never an empty pane)."""
        st = self.status
        self.say(self.bold("Welcome to Anvira Runtime.") + "  This is a live view of ORCHA, Nomi, AICL and your models.")
        if st is None:
            self.say(self.yellow("Not connected yet: ") + (self.error or "starting") + "   (start manually: anvira runtime start)")
        elif not st["model"]["installed_count"]:
            self.say(self.yellow("No model yet.") + " Add a folder that already has .gguf files:  anvira model dirs add <folder>")
            self.say("   or download one:  anvira model install qwen3-4b     (then type  models  here, and  use <id>)")
        elif not st["model"]["active"]:
            self.say(self.yellow("No model selected.") + "  Type  models  to see what is installed, then  use <id>.")
        else:
            self.say(f"Active model: {self.cyan(st['model']['active'])}.  Try:")
            self.say("   run explain how a hash map works        - runs an ORCHA job; its steps appear in Activity")
            self.say("   chat say hello in three words           - streams a reply straight from the model")
        self.say("Pages: press Tab (or 1-5) for Health, Models, Apps & data, Logs.  Type  help  for every command,  quit  to leave.")
        self.say("The runtime stops by itself shortly after the last app closes.")
        self.say("")

    def say(self, text: str = "") -> None:
        with self.lock:
            for line in str(text).splitlines() or [""]:
                self.output.append(line)
            self.scroll = 0

    def note(self, text: str) -> None:
        with self.lock:
            self.activity.append(f"{time.strftime('%H:%M:%S')} {text}")

    # ------------------------------------------------------------------- render
    def _box(self, title: str, lines: list[str], width: int, height: int) -> list[str]:
        h, v, tl, tr, bl, br = ("─", "│", "┌", "┐", "└", "┘") if self.uni else ("-", "|", "+", "+", "+", "+")
        inner = max(1, width - 2)
        head = f"{h} {title[:max(0, inner - 3)]} "
        top = tl + self.bold(head) + h * max(0, inner - len(head)) + tr
        body = [v + fit(l, inner) + v for l in lines[: height - 2]]
        body += [v + " " * inner + v] * max(0, height - 2 - len(body))
        return [top] + body + [bl + h * inner + br]

    @staticmethod
    def _side_by_side(a: list[str], b: list[str]) -> list[str]:
        n = max(len(a), len(b))
        wa = visible_len(a[0]) if a else 0
        wb = visible_len(b[0]) if b else 0
        a = a + [" " * wa] * (n - len(a))
        b = b + [" " * wb] * (n - len(b))
        return [x + y for x, y in zip(a, b)]

    def render(self, width: int, height: int, prompt: str = "", cursor_hint: bool = True) -> list[str]:
        width, height = max(60, width), max(18, height)
        with self.lock:
            st, jobs, aicl, err = self.status, list(self.jobs), self.aicl, self.error
        # ---- title bar
        if st:
            m = st["model"]
            head = (f" {self.bold('Anvira Runtime')} {st['runtime_version']}  {self.dot(st['status'] if st['status'] != 'ok' else 'running')} "
                    f"{'running' if st['status'] == 'ok' else st['status']}   up {int(st['uptime_s'])}s   "
                    f"model: {self.cyan(m['active'] or 'none')} {self.dot(m['state']) if m['active'] else ''}   apps: {st['apps']}")
        else:
            head = f" {self.bold('Anvira Runtime')}  {self.red('not connected')}  {err or ''}"
        rows = [fit(head, width)]
        tabs = "  ".join((self.c(f" {i + 1} {VIEW_TITLES[v]} ", "7") if v == self.view else self.dim(f" {i + 1} {VIEW_TITLES[v]} "))
                         for i, v in enumerate(VIEWS))
        rows.append(fit(" " + tabs + self.dim("    Tab = next page"), width))
        if self.view != "overview" and self.view in tui_views.PAGES:
            out_h = 9
            body_h = max(8, height - len(rows) - out_h - 2)
            rows += tui_views.PAGES[self.view](self, width, body_h)
            with self.lock:
                out = list(self.output)
                end = len(out) - self.scroll
                view = out[max(0, end - (out_h - 2)):max(0, end)]
            rows += self._box("Output" + (f"  (scrolled {self.scroll})" if self.scroll else ""), view, width, out_h)
            rows.append(fit(f" {self.green('anvira>')} {prompt}{'_' if cursor_hint else ''}", width))
            rows.append(fit(self.dim(" Tab next page | r refresh | help | run <task> | chat <msg> | use <id> | logs <svc> | quit "), width))
            return rows[:height]
        # ---- top boxes
        half = width // 2
        svc_lines: list[str] = []
        if st:
            for key, label in (("orcha", "ORCHA"), ("nomi", "Nomi"), ("aicl", "AICL")):
                s = st["services"].get(key, {})
                extra = f"port {s['port']}" if s.get("port") else ("in-process bus" if key == "aicl" else "")
                if s.get("restarts"):
                    extra += f"  restarts {s['restarts']}"
                svc_lines.append(f" {self.dot(s.get('state'))} {label:<6} {s.get('state', '?'):<9} {extra}")
                if s.get("last_error") and s.get("state") != "running":
                    svc_lines.append(self.red("    " + str(s["last_error"]).splitlines()[0][:half - 8]))
            mo = st["model"]
            det = mo.get("backend_info") or {}
            svc_lines.append("")
            svc_lines.append(f" model   {mo['active'] or 'none'}  {mo['state']}")
            if det.get("context"):
                svc_lines.append(f"         ctx {det['context']}  gpu layers {det['gpu_layers']}  port {det.get('port')}")
            svc_lines.append(f" models  {mo['installed_count']} installed   dir {mo['models_dir'][-(half - 20):]}")
        else:
            svc_lines = [self.red(" cannot reach the runtime"), "", " start it:  anvira runtime start"]
        job_lines = []
        for j in jobs[:10]:
            age = int(time.time() - j["created_at"])
            job_lines.append(f" {self.dot(j['state'])} {j['id'][4:]:<12} {j['kind']:<10} {j['state']:<10} {age}s")
        if not job_lines:
            job_lines = [self.dim(" no jobs yet - try:  run explain what a mutex is")]
        top_h = 11
        rows += self._side_by_side(self._box("Services & model", svc_lines, half, top_h),
                                   self._box("Jobs", job_lines, width - half, top_h))
        # ---- activity (AICL trace + ORCHA events + log)
        act_lines: list[str] = []
        with self.lock:
            act_lines += list(self.activity)[-6:]
        if aicl and aicl.get("available"):
            for r in aicl["recent"][-4:]:
                act_lines.append(self.dim(f"{time.strftime('%H:%M:%S', time.localtime(r['ts']))} aicl {r['module']}.{r['action']} "
                                          f"[{r['op']}] {r['request_bytes']}B/{r['response_bytes']}B {r['ms']}ms {'ok' if r['ok'] else 'ERR'}"))
        with self.lock:
            for ln in self.log_lines[-3:]:
                act_lines.append(self.dim("log  " + ln[24:]))
        act_h = 9
        rows += self._box("Activity  (ORCHA events, AICL bus, runtime log)", act_lines[-(act_h - 2):], width, act_h)
        # ---- output pane
        out_h = height - len(rows) - 2
        with self.lock:
            out = list(self.output)
            end = len(out) - self.scroll
            view = out[max(0, end - (out_h - 2)):max(0, end)]
        rows += self._box("Output" + (f"  (scrolled {self.scroll})" if self.scroll else ""), view, width, max(4, out_h))
        # ---- prompt
        rows.append(fit(f" {self.green('anvira>')} {prompt}{'_' if cursor_hint else ''}", width))
        hint = " Tab pages | help | run <task> | chat <msg> | models | use <id> | jobs | cancel <id> | logs | quit "
        rows.append(fit(self.dim(hint), width))
        return rows[:height]

    # ----------------------------------------------------------------- commands
    def execute(self, line: str, wait: bool = False) -> None:
        """Run one dashboard command. Long commands run in a worker thread unless ``wait``."""
        line = line.strip()
        if not line:
            return
        self.say(f"{self.green('anvira>')} {line}")
        try:
            argv = shlex.split(line, posix=os.name != "nt")
        except ValueError as exc:
            self.say(self.red(f"parse error: {exc}"))
            return
        cmd, args = argv[0].lower(), argv[1:]
        handlers = {"help": self._help, "?": self._help, "clear": self._clear, "cls": self._clear,
                    "quit": self._quit, "exit": self._quit, "q": self._quit, "status": self._status, "models": self._models,
                    "use": self._use, "jobs": self._jobs, "job": self._job, "cancel": self._cancel, "logs": self._logs,
                    "remember": self._remember, "memory": self._memory, "doctor": self._doctor, "run": self._run,
                    "chat": self._chat, "view": self._view, "page": self._view, "health": self._page("health"),
                    "apps": self._page("apps"), "overview": self._page("overview")}
        fn = handlers.get(cmd)
        if fn is None:
            self.say(self.red(f"unknown command '{cmd}'. Type 'help'."))
            return
        long = cmd in ("run", "chat", "use", "doctor")
        def go() -> None:
            try:
                fn(args, line)
            except AnviraError as exc:
                self.say(self.red(f"error: {exc.message}") + (self.dim(f"   hint: {exc.hint}") if exc.hint else ""))
            except Exception as exc:  # noqa: BLE001
                self.say(self.red(f"error: {type(exc).__name__}: {exc}"))
        if long and not wait:
            threading.Thread(target=go, daemon=True).start()
        else:
            go()

    # individual commands ----------------------------------------------------------
    def _help(self, a, l): [self.say(x) for x in HELP]

    def _view(self, a, l):
        if not a:
            return self.say("pages: " + ", ".join(VIEWS) + "   (usage: view <name>)")
        self.set_view({"1": "overview", "2": "health", "3": "models", "4": "apps", "5": "logs"}.get(a[0], a[0].lower()))

    def _page(self, name: str):
        return lambda a, l: self.set_view(name)
    def _clear(self, a, l):
        with self.lock:
            self.output.clear()
    def _quit(self, a, l): self.quit = True

    def _status(self, a, l):
        self.refresh()
        st = self.status
        if not st:
            return self.say(self.red(self.error or "not connected"))
        self.say(f"{st['status']}  pid {st['pid']}  up {int(st['uptime_s'])}s  model {st['model']['active'] or 'none'} ({st['model']['state']})")
        for k, v in st["services"].items():
            self.say(f"  {k:<6} {v.get('state')}")

    def _models(self, a, l):
        d = self.http().json("GET", "/v1/models", params={"installed": "true"})
        if not d["models"]:
            return self.say(self.yellow("No model is installed.") + " Use `anvira model install <id>` in another terminal, or add a folder: anvira model dirs add <path>")
        for m in d["models"]:
            size = f"{m['size_bytes'] / 1024**3:.1f} GiB" if m.get("size_bytes") else "remote"
            where = m.get("origin") or m.get("location") or "cloud"
            self.say(f" {'*' if m.get('active') else ' '} {m['id']:<38} {size:>9}  {m['compatibility']['mode']:<11} {where}")

    def _use(self, a, l):
        if not a:
            return self.say("usage: use <model-id>")
        self.say(f"switching to {a[0]} ...")
        self.note(f"model select {a[0]}")
        st = self.http().json("POST", "/v1/models/select", {"id": a[0], "wait_s": 300}, timeout=340)
        self.say(f"active model: {st['active']}  ({st['state']})")
        self.refresh()

    def _jobs(self, a, l):
        self.refresh()
        for j in self.jobs:
            self.say(f" {j['id']}  {j['kind']:<10} {j['state']}")
        if not self.jobs:
            self.say("no jobs")

    def _job(self, a, l):
        if not a:
            return self.say("usage: job <id>")
        job = self.http().json("GET", f"/v1/jobs/{a[0]}")["job"]
        self._print_job(job)

    def _cancel(self, a, l):
        if not a:
            return self.say("usage: cancel <job-id>")
        job = self.http().json("POST", f"/v1/jobs/{a[0]}/cancel")["job"]
        self.say(f"{job['id']}: {job['state']}")

    def _logs(self, a, l):
        svc = a[0] if a else "runtime"
        if self.interactive and (not a or not a[-1].isdigit()):     # `logs orcha` opens the Logs page on that service
            self.log_service = svc
            self.view = "logs"
            self._extra_at.pop("logs", None)
            return self._refresh_view(force=True)
        n = int(a[1]) if len(a) > 1 and a[1].isdigit() else 25
        d = self.http().json("GET", "/v1/runtime/logs", params={"service": svc, "lines": n})
        for ln in d["lines"] or ["(no log yet)"]:
            self.say(self.dim(ln))

    def _remember(self, a, l):
        text = l.split(None, 1)[1] if len(l.split(None, 1)) > 1 else ""
        if not text:
            return self.say("usage: remember <text>")
        m = self.http().json("POST", "/v1/memory", {"content": text.strip("\"'"), "app": "owner"})
        self.say(f"stored {m['id']}")

    def _memory(self, a, l):
        q = l.split(None, 1)[1] if len(l.split(None, 1)) > 1 else ""
        d = self.http().json("GET", "/v1/memory/search", params={"q": q, "limit": 5})
        for m in d["items"] or []:
            self.say(f" {m['title']}  {self.dim('score ' + str(m.get('score')))}\n   {m['content'][:200]}")
        if not d["items"]:
            self.say("no matching memories")

    def _doctor(self, a, l):
        d = self.http().json("GET", "/v1/diagnostics", timeout=60)
        for c in d["checks"]:
            mark = {"ok": self.green("ok  "), "info": self.dim("info"), "warn": self.yellow("warn"), "fail": self.red("FAIL")}[c["status"]]
            self.say(f" {mark} {c['title']}: {c['message'].splitlines()[0][:110]}")
            if c.get("fix") and c["status"] in ("warn", "fail"):
                self.say(self.dim(f"        fix: {c['fix']}"))

    def _print_job(self, job: dict[str, Any]) -> None:
        self.say(f"{self.dot(job['state'])} {job['id']}  {job['kind']}  {job['state']}")
        res = job.get("result") or {}
        if job["state"] == "completed" and res.get("answer") is not None:
            self.say("")
            self.say(res["answer"])
            self.say(self.dim(f"confidence {res.get('confidence')} | iterations {res.get('iterations')} | "
                              f"contributors {', '.join(res.get('contributors') or []) or '-'} | {res.get('latency_s')}s"))
        if job.get("error"):
            e = job["error"]
            self.say(self.red(f"{e.get('code')}: {e.get('message')}"))
            if e.get("hint"):
                self.say(self.dim(f"hint: {e['hint']}"))

    def _run(self, a, l):
        graph, reasoning, words = "default", None, []
        it = iter(a)
        for w in it:
            if w == "--graph":
                graph = next(it, "default")
            elif w == "--reasoning":
                reasoning = next(it, None)
            else:
                words.append(w)
        task = " ".join(words).strip()
        if not task:
            return self.say("usage: run <task>")
        body: dict[str, Any] = {"task": task, "graph": graph}
        if reasoning:
            body["reasoning"] = reasoning
        job = self.http().json("POST", "/v1/orcha/run", body, timeout=60)["job"]
        self.note(f"ORCHA job {job['id']} started ({graph})")
        self.say(f"started {job['id']} ...")
        stop = threading.Event()
        threading.Thread(target=self._follow_events, args=(job["id"], stop), daemon=True).start()
        try:
            while job["state"] in ("queued", "running"):
                time.sleep(0.4)
                job = self.http().json("GET", f"/v1/jobs/{job['id']}")["job"]
        finally:
            stop.set()
        self.note(f"job {job['id']} {job['state']}")
        self._print_job(job)
        self.refresh()

    def _follow_events(self, job_id: str, stop: threading.Event) -> None:
        """Stream ORCHA's own event frames for a job into the Activity panel."""
        try:
            for _ev, data in self.http().sse("GET", f"/v1/jobs/{job_id}/events", timeout=120):
                if stop.is_set():
                    break
                try:
                    self.note("orcha " + summarize_event(json.loads(data)))
                except ValueError:
                    self.note("orcha " + data[:90])
        except Exception:  # noqa: BLE001 - events are best-effort decoration
            pass

    def _chat(self, a, l):
        memory = "--memory" in a
        text = " ".join(x for x in a if x != "--memory").strip()
        if not text:
            return self.say("usage: chat <message>")
        self.note("chat -> active model")
        buf = ""
        body: dict[str, Any] = {"messages": [{"role": "user", "content": text}], "stream": True}
        if memory:
            body["memory"] = {"recall": True}
        with self.lock:
            self.output.append("")
            idx = len(self.output) - 1
        for event, data in self.http().sse("POST", "/v1/chat", body):
            if event == "error":
                err = json.loads(data)["error"]
                return self.say(self.red(f"{err['code']}: {err['message']}") + (self.dim(f"  hint: {err['hint']}") if err.get("hint") else ""))
            if event != "message" or data.strip() == "[DONE]":
                continue
            try:
                delta = json.loads(data)["choices"][0]["delta"].get("content")
            except (ValueError, KeyError, IndexError):
                continue
            if delta:
                buf += delta
                with self.lock:
                    lines = buf.split("\n")
                    while len(self.output) - 1 < idx + len(lines) - 1:
                        self.output.append("")
                    for i, ln in enumerate(lines):
                        self.output[idx + i] = ln


def summarize_event(ev: dict[str, Any]) -> str:
    """One readable line from an ORCHA event frame: ``execute node_end 90ms`` / ``plan checkpoint -> select``."""
    data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
    node = ev.get("node") or ""
    kind = next((str(ev[k]) for k in ("kind", "type", "event", "stage") if ev.get(k)), "event")
    extra = ""
    if data.get("duration_ms") is not None:
        extra = f"{float(data['duration_ms']):.0f}ms"
    elif data.get("next_node"):
        extra = f"-> {data['next_node']}"
    elif ev.get("message") or data.get("message"):
        extra = str(ev.get("message") or data.get("message"))
    return f"{node} {kind} {extra}".strip()[:100]


# ------------------------------------------------------------------ terminal I/O
class Keys:
    """Cross-platform non-blocking key reader yielding tokens."""

    def __enter__(self):
        if os.name == "nt":
            return self
        import termios, tty
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *a):
        if os.name != "nt":
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def poll(self, timeout: float) -> str | None:
        if os.name == "nt":
            import msvcrt
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ("\x00", "\xe0"):
                        return {"H": "up", "P": "down", "K": "left", "M": "right", "I": "pgup", "Q": "pgdn",
                                "G": "home", "O": "end", "S": "delete"}.get(msvcrt.getwch(), "?")
                    return {"\r": "enter", "\x08": "backspace", "\x03": "quit", "\x0c": "redraw", "\x1b": "esc", "\t": "tab"}.get(ch, ch)
                time.sleep(0.02)
            return None
        import select
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if not r:
            return None
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            r, _, _ = select.select([sys.stdin], [], [], 0.02)
            if r:
                seq = sys.stdin.read(2)
                return {"[A": "up", "[B": "down", "[C": "right", "[D": "left", "[5": "pgup", "[6": "pgdn"}.get(seq, "?")
            return "esc"
        return {"\n": "enter", "\r": "enter", "\x7f": "backspace", "\x03": "quit", "\x0c": "redraw"}.get(ch, ch)


def run_ui(http_factory: Callable[[], Http], once: bool = False, exec_cmd: str | None = None, view: str | None = None) -> int:
    color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    dash = Dashboard(http_factory, color=color)
    if view:
        dash.view = view if view in VIEWS else "overview"
    size = shutil.get_terminal_size((110, 34))
    if exec_cmd is not None:                      # headless: run commands, print the output pane
        dash.refresh()
        for cmd in exec_cmd.split(";;"):
            dash.execute(cmd, wait=True)
        for ln in list(dash.output):
            print(_ANSI.sub("", ln))
        return 0
    if once:
        dash.refresh()
        if dash.view != "overview":
            for _ in range(200):                   # let the page's data arrive (the health checkups take a few seconds)
                if dash.extra.get(dash.view) is not None or not dash._fetching:
                    break
                time.sleep(0.2)
            time.sleep(0.3)
        dash.welcome()
        print("\n".join(_ANSI.sub("", r) if not color else r for r in dash.render(size.columns, min(size.lines, 34))))
        return 0 if dash.status else 4
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("anvira ui needs an interactive terminal (use --once for a snapshot).", file=sys.stderr)
        return 2
    if os.name == "nt":
        os.system("")                              # enable ANSI in the Windows console
    dash.hold_lease = True
    dash.interactive = True
    dash.start_polling()
    welcomed, t_start = False, time.monotonic()
    typed, hist_i = "", -1
    sys.stdout.write("\033[?1049h\033[?25l")     # alt screen, hide cursor
    try:
        with Keys() as keys:
            last = 0.0
            dirty = True
            while not dash.quit:
                if not welcomed and (dash.status is not None or time.monotonic() - t_start > 25):
                    dash.welcome()                # once the first status arrived (or the runtime could not be reached)
                    welcomed, dirty = True, True
                tok = keys.poll(0.1)
                if tok is not None:
                    dirty = True
                    if tok == "quit":
                        break
                    elif tok == "enter":
                        if typed.strip():
                            dash.history.append(typed)
                        dash.execute(typed)
                        typed, hist_i = "", -1
                    elif tok == "backspace":
                        typed = typed[:-1]
                    elif tok == "up" and dash.history:
                        hist_i = min(len(dash.history) - 1, hist_i + 1)
                        typed = dash.history[-1 - hist_i]
                    elif tok == "down":
                        hist_i = max(-1, hist_i - 1)
                        typed = dash.history[-1 - hist_i] if hist_i >= 0 else ""
                    elif tok == "pgup":
                        dash.scroll = min(len(dash.output), dash.scroll + 5)
                    elif tok == "pgdn":
                        dash.scroll = max(0, dash.scroll - 5)
                    elif tok == "tab" and not typed:
                        dash.next_view()
                    elif tok == "esc":
                        typed = ""
                    elif not typed and tok in ("1", "2", "3", "4", "5"):
                        dash.set_view(VIEWS[int(tok) - 1])
                    elif not typed and tok == "r":
                        dash._refresh_view(force=True)
                    elif tok == "redraw":
                        sys.stdout.write("\033[2J")
                    elif len(tok) == 1 and tok.isprintable():
                        typed += tok
                if dirty or time.monotonic() - last > 0.5:
                    size = shutil.get_terminal_size((110, 34))
                    frame = dash.render(size.columns, size.lines, typed)
                    sys.stdout.write("\033[H" + "\n".join(fit(r, size.columns) for r in frame) + "\033[J")
                    sys.stdout.flush()
                    last, dirty = time.monotonic(), False
    finally:
        dash.stop()
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()
    return 0
