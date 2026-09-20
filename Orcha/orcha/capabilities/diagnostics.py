"""
orcha.capabilities.diagnostics
==============================
Run build/test/lint/type-check/dependency checks inside the workspace and read
project log files. Uses the project toolchain detected from its manifest files.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..nodes.tool import ToolSpec
from .base import CAUTIOUS, DANGEROUS_LEVEL, EXECUTE, READ, SAFE, WRITE, CapabilityContext, ToolError, spec
from .pathing import ensure_dir, normalize_path

CAPABILITY_NAME = "diagnostics"
CAPABILITY_LABEL = "Diagnostics"
CAPABILITY_DESCRIPTION = "Run builds, tests, linters, type checks, dependency checks and read log files."

_TIMEOUT_S = 300


def _manifest_data(root: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {"root": root}
    pkg = Path(root) / "package.json"
    py = Path(root) / "pyproject.toml"
    if pkg.exists():
        try:
            data["package"] = json.loads(pkg.read_text(encoding="utf-8"))
        except Exception:
            data["package"] = {}
    if py.exists():
        data["pyproject"] = True
    if (Path(root) / "requirements.txt").exists():
        data["requirements"] = True
    if (Path(root) / "Cargo.toml").exists():
        data["cargo"] = True
    return data


def _run(roots, argv: List[str], cwd: str) -> Dict[str, Any]:
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=_TIMEOUT_S)
    except FileNotFoundError:
        raise ToolError("tool_missing", f"Executable not found: {argv[0]}")
    except subprocess.TimeoutExpired:
        raise ToolError("tool_timeout", f"Command timed out after {_TIMEOUT_S}s: {' '.join(argv)}")
    return {
        "command": " ".join(argv), "cwd": cwd,
        "exit_code": proc.returncode, "ok": proc.returncode == 0,
        "stdout": proc.stdout[-16000:], "stderr": proc.stderr.strip()[-8000:],
    }


def _scripts(m: Dict[str, Any]) -> Dict[str, str]:
    pkg = m.get("package") or {}
    return pkg.get("scripts") or {}


def _pick_command(m: Dict[str, Any], names: List[str]) -> List[str]:
    scripts = _scripts(m)
    for name in names:
        if name in scripts:
            return ["npm", "run", name]
    return []


def _find_logs(root: str, pattern: str, limit: int) -> List[Dict[str, Any]]:
    base = Path(root)
    patterns = pattern.split(",") if pattern else ["*.log", "logs/**/*.log", "**/*.log"]
    seen = set()
    out: List[Dict[str, Any]] = []
    for pat in patterns[:8]:
        for path in glob.glob(str(base / pat), recursive=True):
            p = Path(path)
            if p.is_file() and p not in seen and ".git" not in p.parts:
                seen.add(p)
                try:
                    stat = p.stat()
                except OSError:
                    continue
                out.append({"path": str(p), "size": stat.st_size, "mtime": stat.st_mtime})
                if len(out) >= limit:
                    return out
    return out


def build_tools(ctx: CapabilityContext) -> List[ToolSpec]:
    roots = ctx.roots

    def _target_dir(path: str) -> str:
        return ensure_dir(roots, path)

    def run_build(path: str = ".") -> Dict[str, Any]:
        cwd = _target_dir(path)
        m = _manifest_data(cwd)
        if m.get("package"):
            cmd = _pick_command(m, ["build"]) or ["npm", "run", "build"]
        elif m.get("pyproject") or m.get("requirements"):
            cmd = [os.environ.get("PYTHON", "python"), "-m", "build"]
        elif m.get("cargo"):
            cmd = ["cargo", "build"]
        else:
            raise ToolError("no_build_system", "No supported build system detected (package.json, pyproject.toml, requirements.txt, Cargo.toml).")
        return _run(roots, cmd, cwd)

    def run_tests(path: str = ".", pattern: str = "") -> Dict[str, Any]:
        cwd = _target_dir(path)
        m = _manifest_data(cwd)
        if m.get("package"):
            cmd = _pick_command(m, ["test"]) or ["npm", "test"]
        elif m.get("pyproject"):
            cmd = [os.environ.get("PYTHON", "python"), "-m", "pytest", pattern] if pattern else [os.environ.get("PYTHON", "python"), "-m", "pytest"]
        elif m.get("cargo"):
            cmd = ["cargo", "test"]
        else:
            raise ToolError("no_test_runner", "No supported test runner detected.")
        return _run(roots, cmd, cwd)

    def run_linter(path: str = ".") -> Dict[str, Any]:
        cwd = _target_dir(path)
        m = _manifest_data(cwd)
        if m.get("package"):
            cmd = _pick_command(m, ["lint"]) or ["npx", "--no-install", "eslint", "."]
        elif m.get("pyproject"):
            cmd = [os.environ.get("PYTHON", "python"), "-m", "ruff", "check", "."]
        elif m.get("cargo"):
            cmd = ["cargo", "clippy"]
        else:
            raise ToolError("no_linter", "No supported linter detected.")
        return _run(roots, cmd, cwd)

    def type_check(path: str = ".") -> Dict[str, Any]:
        cwd = _target_dir(path)
        m = _manifest_data(cwd)
        if m.get("package"):
            cmd = _pick_command(m, ["type-check", "typecheck"]) or ["npx", "--no-install", "tsc", "--noEmit"]
        elif m.get("pyproject"):
            cmd = [os.environ.get("PYTHON", "python"), "-m", "mypy", "."]
        else:
            raise ToolError("no_type_checker", "No supported type checker detected.")
        return _run(roots, cmd, cwd)

    def dependency_check(path: str = ".", severity: str = "moderate") -> Dict[str, Any]:
        cwd = _target_dir(path)
        m = _manifest_data(cwd)
        if m.get("package"):
            cmd = ["npm", "audit", "--audit-level", severity or "moderate"]
        elif m.get("pyproject") or m.get("requirements"):
            cmd = [os.environ.get("PYTHON", "python"), "-m", "pip", "check"]
        else:
            raise ToolError("no_dependency_manager", "No supported dependency manager detected.")
        return _run(roots, cmd, cwd)

    def read_logs(path: str = ".", pattern: str = "", limit: int = 5) -> Dict[str, Any]:
        cwd = _target_dir(path)
        n = max(1, min(int(limit or 5), 20))
        logs = _find_logs(cwd, pattern, n)
        entries = []
        for log in logs:
            try:
                with open(log["path"], "r", encoding="utf-8", errors="replace") as fh:
                    tail = fh.read()[-4000:]
            except OSError as exc:
                tail = f"<unreadable: {exc}>"
            entries.append({"path": log["path"], "size": log["size"], "tail": tail})
        return {"logs": entries, "count": len(entries)}

    return [
        spec("run_build",
             "Run the project's build command (auto-detected from package.json/pyproject.toml/Cargo.toml). USE THIS WHEN: compiling the project, generating output bundles, or checking for build errors. Returns stdout/stderr and exit code. Exit code 0 means success.",
             {"type": "object", "properties": {"path": {"type": "string", "description": "Project directory inside the workspace. Default: '.' (workspace root)."}}},
             run_build, permissions=[EXECUTE, WRITE], safety_level=CAUTIOUS, capability=CAPABILITY_NAME),
        spec("run_tests",
             "Run the project's test suite (auto-detected: pytest/npm test/cargo test). USE THIS WHEN: verifying your changes don't break anything, checking test coverage, or running specific test files. Returns test output with pass/fail counts.",
             {"type": "object", "properties": {
                 "path": {"type": "string", "description": "Project directory. Default: '.'."},
                 "pattern": {"type": "string", "description": "Optional filter. For pytest: keyword expression like 'test_login' or 'not slow'. For npm: test file pattern."}}},
             run_tests, permissions=[EXECUTE], safety_level=CAUTIOUS, capability=CAPABILITY_NAME),
        spec("run_linter",
             "Run the project's linter (auto-detected: eslint/ruff/clippy). USE THIS WHEN: checking code style, finding potential bugs, or verifying code follows project conventions. Returns linting issues found.",
             {"type": "object", "properties": {"path": {"type": "string", "description": "Project directory. Default: '.'."}}},
             run_linter, permissions=[EXECUTE], safety_level=CAUTIOUS, capability=CAPABILITY_NAME),
        spec("type_check",
             "Run the project's type checker (auto-detected: tsc --noEmit/mypy). USE THIS WHEN: verifying TypeScript types are correct, or checking Python type annotations. Returns type errors found.",
             {"type": "object", "properties": {"path": {"type": "string", "description": "Project directory. Default: '.'."}}},
             type_check, permissions=[EXECUTE], safety_level=CAUTIOUS, capability=CAPABILITY_NAME),
        spec("dependency_check",
             "Check project dependencies for known vulnerabilities (npm audit / pip check). USE THIS WHEN: auditing security of installed packages, or diagnosing dependency conflicts.",
             {"type": "object", "properties": {
                 "path": {"type": "string", "description": "Project directory. Default: '.'."},
                 "severity": {"type": "string", "description": "Minimum severity to report. Default: 'moderate'. Options: 'low', 'moderate', 'high', 'critical'."}}},
             dependency_check, permissions=[EXECUTE, "dangerous"], safety_level=DANGEROUS_LEVEL, capability=CAPABILITY_NAME),
        spec("read_logs",
             "Find and read the tail of log files in the project. USE THIS WHEN: debugging errors, checking application output, or investigating failures. Returns the last 4KB of each log file found.",
             {"type": "object", "properties": {
                 "path": {"type": "string", "description": "Project directory. Default: '.'."},
                 "pattern": {"type": "string", "description": "Comma-separated glob patterns. Example: '*.log' or 'logs/**/*.log'. Default: auto-detect common log patterns."},
                 "limit": {"type": "integer", "description": "Max log files to read (1-20). Default: 5."}}},
             read_logs, permissions=[READ], safety_level=SAFE, capability=CAPABILITY_NAME),
    ]
