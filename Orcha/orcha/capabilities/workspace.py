"""
orcha.capabilities.workspace
============================
Workspace awareness tools — project context, attachment lists, tree and
summary. Gives agents the same high-level view the app has (projects,
attachments, active/recent files) so they can orient before touching files.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..nodes.tool import ToolSpec
from .base import READ, SAFE, CapabilityContext, ToolError, spec
from .pathing import match_ignored, normalize_path, resolve_target

CAPABILITY_NAME = "workspace"
CAPABILITY_LABEL = "Workspace"
CAPABILITY_DESCRIPTION = (
    "Project awareness: current workspace, attached files/folders, project "
    "tree and summary, active and recent files."
)

_READ_PERMS = [READ]


def _current_workspace(ctx):
    def tool() -> Dict[str, Any]:
        return {
            "roots": list(ctx.roots),
            "project_path": ctx.project_path,
            "attachments": list(ctx.attachments),
            "active_file": ctx.active_file,
            "recent_files": list(ctx.recent_files),
            "attached": bool(ctx.roots),
        }
    return tool


def _root_or_project(ctx) -> str:
    if ctx.project_path and os.path.isdir(ctx.project_path):
        return ctx.project_path
    if ctx.roots:
        return normalize_path(ctx.roots[0])
    raise ToolError("no_workspace", "No workspace is attached.")


def _workspace_tree(ctx):
    def _render(base: str, depth: int, cur: int) -> List[str]:
        if cur > depth:
            return []
        try:
            names = sorted(os.listdir(base))
        except OSError:
            return []
        out: List[str] = []
        for name in names:
            full = os.path.join(base, name)
            is_dir = os.path.isdir(full)
            if match_ignored(name, is_dir):
                continue
            out.append("  " * cur + name + ("/" if is_dir else ""))
            if is_dir and cur < depth:
                out.extend(_render(full, depth, cur + 1))
        return out

    def tool(path: str = ".", depth: int = 2) -> str:
        fp = resolve_target(ctx.roots, path) if ctx.roots else normalize_path(os.path.join(_root_or_project(ctx), path))
        if not os.path.isdir(fp):
            raise ToolError("directory_not_found", f"Directory not found: {path}")
        lines = [os.path.basename(fp) or fp, *["  " + l for l in _render(fp, int(depth), 0)]]
        return "\n".join(lines)

    return tool


def _list_projects(ctx):
    def tool() -> Dict[str, Any]:
        projects = []
        for root in ctx.roots:
            projects.append({
                "path": root,
                "name": os.path.basename(root.rstrip("\\/")) or root,
                "has_vcs": os.path.isdir(os.path.join(root, ".git")),
                "has_package_json": os.path.isfile(os.path.join(root, "package.json")),
                "has_pyproject": os.path.isfile(os.path.join(root, "pyproject.toml")),
            })
        return {"projects": projects, "count": len(projects)}
    return tool


def _attached_files(ctx):
    def tool() -> Dict[str, Any]:
        files = []
        for a in ctx.attachments:
            kind = (a.get("kind") or "").lower()
            if kind in ("file", "attachment") or (kind not in ("folder", "directory") and not os.path.isdir(a.get("path", ""))):
                files.append({"name": a.get("name") or os.path.basename(a.get("path", "") or ""), "path": a.get("path")})
        # Fall back to top-level files in the first root when no explicit attachments.
        if not files and ctx.roots:
            root = normalize_path(ctx.roots[0])
            for name in sorted(os.listdir(root)):
                full = os.path.join(root, name)
                if os.path.isfile(full) and not match_ignored(name, False) and not name.startswith("."):
                    files.append({"name": name, "path": full})
        return {"files": files, "count": len(files)}
    return tool


def _attached_folders(ctx):
    def tool() -> Dict[str, Any]:
        folders = []
        for a in ctx.attachments:
            kind = (a.get("kind") or "").lower()
            p = a.get("path") or ""
            if kind in ("folder", "directory") or os.path.isdir(p):
                folders.append({"name": a.get("name") or os.path.basename(p.rstrip("\\/")) or p, "path": p})
        if not folders and ctx.roots:
            folders.append({"name": os.path.basename(normalize_path(ctx.roots[0]).rstrip("\\/")), "path": normalize_path(ctx.roots[0])})
        return {"folders": folders, "count": len(folders)}
    return tool


def _active_file(ctx):
    def tool() -> Dict[str, Any]:
        if ctx.active_file:
            return {"active_file": ctx.active_file, "present": os.path.isfile(ctx.active_file)}
        return {"active_file": None, "present": False}
    return tool


def _recent_files(ctx):
    def tool(limit: int = 10) -> Dict[str, Any]:
        n = max(1, min(int(limit or 10), 50))
        if ctx.recent_files:
            return {"recent_files": ctx.recent_files[:n], "count": len(ctx.recent_files[:n])}
        # Fall back to most recently modified text files in the first root.
        root = _root_or_project(ctx)
        candidates = []
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if not match_ignored(d, True)]
            for name in files:
                if match_ignored(name, False) or name.startswith("."):
                    continue
                full = os.path.join(dirpath, name)
                try:
                    candidates.append((os.path.getmtime(full), full))
                except OSError:
                    continue
        candidates.sort(reverse=True)
        return {"recent_files": [p for _, p in candidates[:n]], "count": min(n, len(candidates))}
    return tool


def _project_summary(ctx):
    def tool() -> Dict[str, Any]:
        root = _root_or_project(ctx)
        counts: Dict[str, int] = {}
        total = 0
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if not match_ignored(d, True)]
            for name in files:
                if match_ignored(name, False) or name.startswith("."):
                    continue
                total += 1
                ext = os.path.splitext(name)[1].lstrip(".").lower() or "(no extension)"
                counts[ext] = counts.get(ext, 0) + 1
        top = sorted(os.listdir(root))
        top_level = []
        for name in top:
            if name.startswith(".") or match_ignored(name, os.path.isdir(os.path.join(root, name))):
                continue
            top_level.append({"name": name, "type": "dir" if os.path.isdir(os.path.join(root, name)) else "file"})
        return {
            "root": root,
            "file_count": total,
            "by_extension": dict(sorted(counts.items(), key=lambda kv: -kv[1])[:15]),
            "top_level_entries": top_level[:50],
        }
    return tool


def build_tools(ctx: CapabilityContext) -> List[ToolSpec]:
    return [
        spec("current_workspace",
             "Show the current project context: workspace root paths, attached files/folders, and active file. USE THIS FIRST when starting a task to understand what project you're working with.",
             {"type": "object"}, _current_workspace(ctx), permissions=_READ_PERMS, safety_level=SAFE,
             capability=CAPABILITY_NAME, result_format={"type": "object"}),
        spec("workspace_tree",
             "Show a nested directory tree of the project. USE THIS WHEN: you need to understand the project structure at a glance — see all folders and key files. Good first step before diving into specific files.",
             {"type": "object", "properties": {
                 "path": {"type": "string", "description": "Directory to show. Default: '.' (project root). Example: 'src' to see just the src tree."},
                 "depth": {"type": "integer", "description": "How deep to go (1-20). Default: 2. Use 3-4 for more detail."}},
              }, _workspace_tree(ctx), permissions=_READ_PERMS, safety_level=SAFE, capability=CAPABILITY_NAME),
        spec("list_projects",
             "List all attached projects with their type (has git? has package.json? has pyproject.toml?). USE THIS WHEN: understanding what projects are open and what tech stack each uses.",
             {"type": "object"}, _list_projects(ctx), permissions=_READ_PERMS, safety_level=SAFE,
             capability=CAPABILITY_NAME, result_format={"type": "object"}),
        spec("attached_files",
             "List files that are attached to the current conversation. USE THIS WHEN: seeing which specific files the user has provided or is focused on.",
             {"type": "object"}, _attached_files(ctx), permissions=_READ_PERMS, safety_level=SAFE,
             capability=CAPABILITY_NAME, result_format={"type": "object"}),
        spec("attached_folders",
             "List folders that are attached to the current conversation. USE THIS WHEN: seeing which directories the user has provided or is focused on.",
             {"type": "object"}, _attached_folders(ctx), permissions=_READ_PERMS, safety_level=SAFE,
             capability=CAPABILITY_NAME, result_format={"type": "object"}),
        spec("active_file",
             "Return the currently active/selected file in the editor (if any). USE THIS WHEN: the user says 'this file' or 'the current file' and you need to know which file they mean.",
             {"type": "object"}, _active_file(ctx), permissions=_READ_PERMS, safety_level=SAFE,
             capability=CAPABILITY_NAME, result_format={"type": "object"}),
        spec("recent_files",
             "Return the most recently used or modified files. USE THIS WHEN: finding files the user was recently working on, or getting a sense of what's been active.",
             {"type": "object", "properties": {"limit": {"type": "integer", "description": "Max files to return (1-50). Default: 10."}}},
             _recent_files(ctx), permissions=_READ_PERMS, safety_level=SAFE,
             capability=CAPABILITY_NAME, result_format={"type": "object"}),
        spec("project_summary",
             "Show a project overview: file count by extension, top-level entries. USE THIS WHEN: understanding what kind of project this is (how many .ts files, .py files, etc.) and its overall structure.",
             {"type": "object"}, _project_summary(ctx), permissions=_READ_PERMS, safety_level=SAFE,
             capability=CAPABILITY_NAME, result_format={"type": "object"}),
    ]
