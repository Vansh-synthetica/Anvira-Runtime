"""
orcha.capabilities.terminal
===========================
Secure terminal runtime: run commands, stream long-running processes,
stop them, and inspect process/history state.

Safety
------
- commands run with ``cwd`` resolved inside the workspace roots (never outside)
- a small deny-list blocks clearly destructive system commands
- every call has a timeout and returns structured results/errors
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from ..nodes.tool import ToolSpec
from .base import DANGEROUS_LEVEL, EXECUTE, READ, SAFE, CapabilityContext, ToolError, spec
from .pathing import ensure_dir, normalize_path, resolve_target

CAPABILITY_NAME = "terminal"
CAPABILITY_LABEL = "Terminal"
CAPABILITY_DESCRIPTION = (
    "Run shell commands, stream long-running processes, stop processes, and "
    "inspect process state — PowerShell, CMD and Bash."
)

# Commands that are never allowed — they mutate system state beyond any
# reasonable workspace scope.
_DENIED_COMMANDS = (
    "shutdown", "restart ", "reboot", "format ", "diskpart", "mkfs",
    "rm -rf /", "del /s /q c:\\", "rd /s /q c:\\", "dd if=",
    "> nul del", "powershell -enc", "powershell -encodedcommand",
)

_SHELLS = {
    "powershell": ["powershell", "-NoProfile", "-NonInteractive", "-Command"],
    "cmd": ["cmd", "/c"],
    "bash": ["bash", "-c"],
}


def _detect_shell() -> str:
    return "powershell" if os.name == "nt" else "bash"


# `a && b` / `a || b` are the universal way to chain shell commands, and it is
# what a model emits by default regardless of platform. Windows PowerShell 5.1
# — still the `powershell.exe` on a stock Windows box, and this module's
# auto-detected default — does not have those operators and fails the whole
# command with "The token '&&' is not a valid statement separator in this
# version". `cmd.exe` supports both with exactly the semantics the model
# intends, so a chained command auto-routes there.
#
# Translating `&&` to PowerShell's `;` would be wrong, not merely different:
# `;` runs the next command even when the previous one failed, which turns
# "build && test" into "test regardless of whether the build worked".
#
# Measured live: a 3B model wrote both source files correctly and then lost
# every one of its four attempts at `cd workspace && python -m unittest`, with
# the retry budget spent entirely on a shell incompatibility it could do
# nothing about.
_CHAIN_OPERATORS = ("&&", "||")


def _shell_for(command: str, shell: Optional[str]) -> str:
    """Resolve the shell, preferring cmd for chained commands on Windows.

    An explicitly requested shell is always honoured — the caller may have
    a reason, and silently overriding it would be worse than failing.
    """
    if shell:
        return shell
    detected = _detect_shell()
    if detected == "powershell" and any(op in command for op in _CHAIN_OPERATORS):
        return "cmd"
    return detected


def _validate_command(command: str, shell: str, cwd_abs: str) -> None:
    if not (command or "").strip():
        raise ToolError("invalid_command", "command must not be empty")
    if shell not in _SHELLS:
        raise ToolError("invalid_shell", f"Unsupported shell '{shell}'. Use one of: {', '.join(_SHELLS)}")
    low = command.lower()
    for denied in _DENIED_COMMANDS:
        if denied in low:
            raise ToolError("denied_command", f"Command blocked by the terminal safety policy: {denied}")


def _resolve_cwd(roots, cwd) -> str:
    if cwd:
        return ensure_dir(roots, cwd)
    if not roots:
        raise ToolError("no_workspace", "No workspace roots configured for the terminal.")
    return normalize_path(roots[0])


# ── Module-level process + history registry ───────────────────────────────────

_PROCESSES: Dict[str, Dict[str, Any]] = {}
_HISTORY: List[Dict[str, Any]] = []
_LOCK = threading.Lock()


def _record(entry: Dict[str, Any]) -> None:
    with _LOCK:
        _HISTORY.append(entry)
        if len(_HISTORY) > 100:
            _HISTORY[:] = _HISTORY[-100:]


def _run_command(roots):
    def tool(command: str, shell: Optional[str] = None, cwd: Optional[str] = None, timeout_s: int = 120) -> Dict[str, Any]:
        shell = _shell_for(command, shell)
        cwd_abs = _resolve_cwd(roots, cwd)
        _validate_command(command, shell, cwd_abs)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [*_SHELLS[shell], command],
                cwd=cwd_abs, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=int(timeout_s or 120),
            )
            result = {
                "command": command, "shell": shell, "cwd": cwd_abs,
                "exit_code": proc.returncode, "ok": proc.returncode == 0,
                "stdout": proc.stdout[-16000:], "stderr": proc.stderr.strip()[-8000:],
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
        except subprocess.TimeoutExpired as exc:
            raise ToolError("command_timeout", f"Command timed out after {timeout_s}s",
                            detail={"command": command, "partial_stdout": (exc.stdout or "")[-2000:]})
        except FileNotFoundError:
            raise ToolError("shell_missing", f"Shell '{shell}' is not available on this system.")
        _record({**result, "kind": "run", "ts": time.time()})
        return result
    return tool


def _stream_output(roots):
    def tool(command: str, shell: Optional[str] = None, cwd: Optional[str] = None) -> Dict[str, Any]:
        shell = _shell_for(command, shell)
        cwd_abs = _resolve_cwd(roots, cwd)
        _validate_command(command, shell, cwd_abs)
        try:
            proc = subprocess.Popen(
                [*_SHELLS[shell], command], cwd=cwd_abs,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
        except (OSError, FileNotFoundError):
            raise ToolError("shell_missing", f"Could not start shell '{shell}' on this system.")

        sid = str(uuid.uuid4())
        buffer: List[str] = []
        done = threading.Event()

        def _reader():
            try:
                for chunk in iter(proc.stdout.readline, ""):
                    buffer.append(chunk)
            finally:
                done.set()

        threading.Thread(target=_reader, daemon=True).start()
        with _LOCK:
            _PROCESSES[sid] = {
                "stream_id": sid, "proc": proc, "buffer": buffer,
                "command": command, "shell": shell, "cwd": cwd_abs,
                "started": time.time(), "done": done,
            }
        _record({"kind": "stream", "command": command, "shell": shell, "cwd": cwd_abs,
                 "stream_id": sid, "ts": time.time()})
        return {"stream_id": sid, "status": "running", "command": command,
                "shell": shell, "cwd": cwd_abs, "output": "".join(buffer)[-4000:]}
    return tool


def _stop_process(_roots):
    def tool(stream_id: str) -> Dict[str, Any]:
        entry = _PROCESSES.get(stream_id)
        if entry is None:
            raise ToolError("process_not_found", f"No process with stream_id {stream_id}.")
        proc = entry["proc"]
        killed = False
        if proc.poll() is None:
            try:
                proc.kill()
                killed = True
            except OSError:
                pass
        return {"stream_id": stream_id, "killed": killed,
                "exit_code": proc.poll(), "output": "".join(entry["buffer"])[-4000:]}
    return tool


def _running_processes(_roots):
    def tool() -> Dict[str, Any]:
        now = time.time()
        procs = []
        for sid, entry in list(_PROCESSES.items()):
            proc = entry["proc"]
            running = proc.poll() is None
            procs.append({
                "stream_id": sid, "running": running,
                "command": entry["command"], "shell": entry["shell"],
                "elapsed_s": round(now - entry["started"], 1),
                "exit_code": None if running else proc.returncode,
                "output_tail": "".join(entry["buffer"])[-3000:],
            })
        return {"processes": procs, "count": len(procs)}
    return tool


def _command_history(_roots):
    def tool(limit: int = 25) -> Dict[str, Any]:
        n = max(1, min(int(limit or 25), 100))
        return {"history": _HISTORY[-n:], "count": min(n, len(_HISTORY))}
    return tool


def build_tools(ctx: CapabilityContext) -> List[ToolSpec]:
    roots = ctx.roots
    return [
        spec("run_command",
             "Execute a shell command (PowerShell, CMD, or Bash) and return stdout, stderr, and exit code. USE THIS WHEN: you need to run build tools (npm, cargo, python), install packages, run tests, check versions, execute scripts, or perform any system operation. Returns {command, exit_code, stdout, stderr, ok}. If exit_code is 0 the command succeeded. The command runs in the project root unless you specify cwd.",
             {"type": "object", "properties": {
                 "command": {"type": "string", "description": "The shell command to execute. Examples: 'npm install', 'python app.py --port 3000', 'cargo build --release', 'dir /b src' (list files on Windows), 'Get-ChildItem -Recurse *.ts'."},
                 "shell": {"type": "string", "description": "Shell to use: 'powershell' (Windows default), 'cmd', or 'bash' (Linux/Mac default). Default: auto-detect based on OS."},
                 "cwd": {"type": "string", "description": "Working directory for the command. Default: project root. Example: 'src/tests' to run tests from a subdirectory."},
                 "timeout_s": {"type": "integer", "description": "Max seconds before the command is killed. Default: 120. Increase for long builds (e.g. 300 for 'npm run build')."}},
              "required": ["command"]},
             _run_command(roots), permissions=[EXECUTE, "dangerous"], safety_level=DANGEROUS_LEVEL,
             capability=CAPABILITY_NAME, result_format={"type": "object",
                 "properties": {"exit_code": {"type": "integer"}, "stdout": {"type": "string"}, "stderr": {"type": "string"}}}),
        spec("stream_output",
             "Start a long-running command in the background (e.g. a dev server, file watcher, or build). Returns a stream_id you can use to check output or stop the process later. USE THIS WHEN: the command takes too long for run_command, or you need to start a server and let it run.",
             {"type": "object", "properties": {
                 "command": {"type": "string", "description": "The command to run in the background. Example: 'npm run dev', 'python -m http.server 8000'."},
                 "shell": {"type": "string", "description": "Shell: 'powershell', 'cmd', or 'bash'. Default: auto-detect."},
                 "cwd": {"type": "string", "description": "Working directory. Default: project root."}},
              "required": ["command"]},
             _stream_output(roots), permissions=[EXECUTE, "dangerous"], safety_level=DANGEROUS_LEVEL,
             capability=CAPABILITY_NAME, result_format={"type": "object",
                 "properties": {"stream_id": {"type": "string"}, "status": {"type": "string"}, "output": {"type": "string"}}}),
        spec("stop_process",
             "Kill a background process that was started with stream_output. USE THIS WHEN: you need to stop a dev server, a running build, or any long-running background command.",
             {"type": "object", "properties": {"stream_id": {"type": "string", "description": "The stream_id returned by stream_output."}}, "required": ["stream_id"]},
             _stop_process(roots), permissions=[EXECUTE, "dangerous"], safety_level=DANGEROUS_LEVEL, capability=CAPABILITY_NAME),
        spec("running_processes",
             "List all background processes started with stream_output, showing their status, command, elapsed time, and recent output. USE THIS WHEN: checking if a background process is still running or viewing its output.",
             {"type": "object"}, _running_processes(roots), permissions=[READ], safety_level=SAFE,
             capability=CAPABILITY_NAME, result_format={"type": "object", "properties": {"processes": {"type": "array"}, "count": {"type": "integer"}}}),
        spec("command_history",
             "Show recent terminal commands that were executed in this session. USE THIS WHEN: you want to see what commands were already run, or find a command you ran earlier to re-run it.",
             {"type": "object", "properties": {"limit": {"type": "integer", "description": "Max entries to return (1-100). Default: 25."}}},
             _command_history(roots), permissions=[READ], safety_level=SAFE, capability=CAPABILITY_NAME,
             result_format={"type": "object", "properties": {"history": {"type": "array"}, "count": {"type": "integer"}}}),
    ]
