"""
orcha.capabilities.git
======================
Git integration via the system ``git`` CLI (no heavyweight Python deps).
All commands run with ``cwd`` inside the workspace roots.
"""
from __future__ import annotations

import subprocess
from typing import Any, Dict, List, Optional

from ..nodes.tool import ToolSpec
from .base import CAUTIOUS, DANGEROUS_LEVEL, EXECUTE, NETWORK, READ, SAFE, WRITE, CapabilityContext, ToolError, spec
from .pathing import ensure_dir, normalize_path

CAPABILITY_NAME = "git"
CAPABILITY_LABEL = "Git"
CAPABILITY_DESCRIPTION = "Status, diff, add, commit, branch, checkout, log, show, restore, pull and push."

_GIT_TIMEOUT_S = 60


def _git(roots, args: List[str], path: str = ".", timeout_s: int = _GIT_TIMEOUT_S) -> Dict[str, Any]:
    cwd = ensure_dir(roots, path)
    try:
        proc = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=int(timeout_s),
        )
    except FileNotFoundError:
        raise ToolError("git_missing", "git is not installed or not on PATH.")
    except subprocess.TimeoutExpired:
        raise ToolError("git_timeout", f"git {args[0]} timed out after {timeout_s}s")
    result = {
        "command": "git " + " ".join(args), "cwd": cwd,
        "exit_code": proc.returncode, "ok": proc.returncode == 0,
        "stdout": proc.stdout[-12000:], "stderr": proc.stderr.strip()[-4000:],
    }
    if proc.returncode != 0:
        raise ToolError("git_error", f"git {args[0]} failed", detail=result)
    return result


def _list_args(paths: Optional[List[str]]) -> List[str]:
    return list(paths or [])


def build_tools(ctx: CapabilityContext) -> List[ToolSpec]:
    roots = ctx.roots

    def git_status(path: str = ".") -> Dict[str, Any]:
        return _git(roots, ["status", "--short", "--branch"], path)

    def git_diff(path: str = ".", staged: bool = False) -> Dict[str, Any]:
        args = ["diff", "--stat", "--patch"] if not staged else ["diff", "--cached", "--stat", "--patch"]
        return _git(roots, args, path)

    def git_add(path: str = ".", paths: Optional[List[str]] = None) -> Dict[str, Any]:
        files = _list_args(paths)
        return _git(roots, ["add", "--", *files] if files else ["add", "-A"], path)

    def git_commit(path: str = ".", message: str = "") -> Dict[str, Any]:
        if not (message or "").strip():
            raise ToolError("invalid_message", "A commit message is required.")
        return _git(roots, ["commit", "-m", message], path)

    def git_branch(path: str = ".") -> Dict[str, Any]:
        return _git(roots, ["branch", "-a"], path)

    def git_checkout(path: str = ".", branch: str = "") -> Dict[str, Any]:
        if not (branch or "").strip():
            raise ToolError("invalid_branch", "branch must not be empty")
        return _git(roots, ["checkout", branch], path)

    def git_log(path: str = ".", limit: int = 20) -> Dict[str, Any]:
        n = max(1, min(int(limit or 20), 100))
        return _git(roots, ["log", "--oneline", "--decorate", "-n", str(n)], path)

    def git_show(path: str = ".", ref: str = "HEAD") -> Dict[str, Any]:
        return _git(roots, ["show", "--stat", "--oneline", ref or "HEAD"], path)

    def git_restore(path: str = ".", paths: Optional[List[str]] = None) -> Dict[str, Any]:
        files = _list_args(paths)
        return _git(roots, ["restore", "--", *files] if files else ["restore", "."], path)

    def git_pull(path: str = ".") -> Dict[str, Any]:
        return _git(roots, ["pull", "--ff-only"], path)

    def git_push(path: str = ".", remote: str = "", branch: str = "") -> Dict[str, Any]:
        args = ["push"]
        if remote:
            args.append(remote)
            if branch:
                args.append(branch)
        elif branch:
            args.extend(["origin", branch])
        return _git(roots, args, path)

    return [
        spec("git_status",
             "Show the working tree status: which files are modified, staged, untracked, or conflicted. USE THIS WHEN: you want to see what has changed, what needs to be staged, or what the current state of the repo is. Shows branch name too.",
             {"type": "object", "properties": {"path": {"type": "string", "description": "Repo directory. Default: '.' (workspace root)."}}},
             git_status, permissions=[READ], safety_level=SAFE, capability=CAPABILITY_NAME),
        spec("git_diff",
             "Show uncommitted changes (or staged changes with staged=true) as a diff patch. USE THIS WHEN: reviewing what exactly changed in files before committing, or checking if your edits look correct.",
             {"type": "object", "properties": {
                 "path": {"type": "string", "description": "Repo directory. Default: '.'."},
                 "staged": {"type": "boolean", "description": "Show staged changes (git diff --cached). Default: false (shows unstaged changes)."}}},
             git_diff, permissions=[READ], safety_level=SAFE, capability=CAPABILITY_NAME),
        spec("git_add",
             "Stage files for the next commit. Pass specific files or leave paths empty to stage ALL changes. USE THIS WHEN: preparing to commit — you must add files before committing them.",
             {"type": "object", "properties": {
                 "path": {"type": "string", "description": "Repo directory. Default: '.'."},
                 "paths": {"type": "array", "items": {"type": "string"}, "description": "Specific files to stage. Examples: ['src/app.ts', 'package.json']. Leave empty to stage all changes (git add -A)."}}},
             git_add, permissions=[WRITE], safety_level=CAUTIOUS, capability=CAPABILITY_NAME),
        spec("git_commit",
             "Create a git commit with the given message. You must git_add files first. USE THIS WHEN: saving a logical snapshot of changes. Write clear, concise commit messages.",
             {"type": "object", "properties": {
                 "path": {"type": "string", "description": "Repo directory. Default: '.'."},
                 "message": {"type": "string", "description": "Commit message. Example: 'fix: resolve auth token expiry bug'. Keep under 72 chars."}},
              "required": ["message"]},
             git_commit, permissions=[WRITE, EXECUTE], safety_level=CAUTIOUS, capability=CAPABILITY_NAME),
        spec("git_branch",
             "List all local and remote branches. USE THIS WHEN: seeing what branches exist, checking the current branch, or finding a branch name to switch to.",
             {"type": "object", "properties": {"path": {"type": "string", "description": "Repo directory. Default: '.'."}}},
             git_branch, permissions=[READ], safety_level=SAFE, capability=CAPABILITY_NAME),
        spec("git_checkout",
             "Switch to a different branch. USE THIS WHEN: changing your working branch. WARNING: this may discard uncommitted changes — commit or stash first if needed.",
             {"type": "object", "properties": {
                 "path": {"type": "string", "description": "Repo directory. Default: '.'."},
                 "branch": {"type": "string", "description": "Branch name to switch to. Example: 'main', 'feature/auth'. Use git_branch to see available branches."}},
              "required": ["branch"]},
             git_checkout, permissions=[WRITE, EXECUTE, "dangerous"], safety_level=DANGEROUS_LEVEL, capability=CAPABILITY_NAME),
        spec("git_log",
             "Show recent commit history. USE THIS WHEN: understanding what changed recently, finding a commit to revert, or checking if your changes were committed.",
             {"type": "object", "properties": {
                 "path": {"type": "string", "description": "Repo directory. Default: '.'."},
                 "limit": {"type": "integer", "description": "Number of commits to show (1-100). Default: 20."}}},
             git_log, permissions=[READ], safety_level=SAFE, capability=CAPABILITY_NAME),
        spec("git_show",
             "Show details of a specific commit including its diff. USE THIS WHEN: examining what a particular commit changed, or viewing the HEAD commit.",
             {"type": "object", "properties": {
                 "path": {"type": "string", "description": "Repo directory. Default: '.'."},
                 "ref": {"type": "string", "description": "Commit hash, branch, or tag. Default: 'HEAD' (latest commit). Example: 'abc1234' or 'HEAD~1' (one commit back)."}}},
             git_show, permissions=[READ], safety_level=SAFE, capability=CAPABILITY_NAME),
        spec("git_restore",
             "Discard uncommitted changes in files (restore to last commit). USE THIS WHEN: undoing changes you haven't committed yet. WARNING: This is permanent — staged and unstaged changes are lost.",
             {"type": "object", "properties": {
                 "path": {"type": "string", "description": "Repo directory. Default: '.'."},
                 "paths": {"type": "array", "items": {"type": "string"}, "description": "Files to restore. Leave empty to restore all files."}}},
             git_restore, permissions=[WRITE, "dangerous"], safety_level=DANGEROUS_LEVEL, capability=CAPABILITY_NAME),
        spec("git_pull",
             "Pull latest changes from remote (fast-forward only). USE THIS WHEN: syncing your local branch with the remote before pushing or starting new work.",
             {"type": "object", "properties": {"path": {"type": "string", "description": "Repo directory. Default: '.'."}}},
             git_pull, permissions=[NETWORK, EXECUTE, "dangerous"], safety_level=DANGEROUS_LEVEL, capability=CAPABILITY_NAME),
        spec("git_push",
             "Push committed changes to the remote repository. USE THIS WHEN: sharing your committed changes with the remote. Make sure you've committed and pulled first.",
             {"type": "object", "properties": {
                 "path": {"type": "string", "description": "Repo directory. Default: '.'."},
                 "remote": {"type": "string", "description": "Remote name. Default: 'origin'."},
                 "branch": {"type": "string", "description": "Branch to push. Default: current branch."}}},
             git_push, permissions=[NETWORK, EXECUTE, "dangerous"], safety_level=DANGEROUS_LEVEL, capability=CAPABILITY_NAME),
    ]
