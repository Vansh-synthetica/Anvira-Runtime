"""
orcha.api.skills_manager
========================
Desktop-side registry of local SKILL.md packages: scan configured
directories, list them for the UI, and hand every discovered skill's tool
to agent runs (prompt skills expand their markdown body; script skills run
their entrypoint lazily).

Directories come from ``ORCHA_SKILLS_DIR`` (path-separator-separated, like
PATH). Missing directories are skipped with a warning — never an error.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from ..agent_runtime.skills import SkillsDirectory
from ..agent_runtime.tools import Tool, ToolRegistry
from ..nodes.tool import ToolSpec

logger = logging.getLogger("orcha.api.skills_manager")


def _skills_dirs() -> List[Path]:
    raw = os.environ.get("ORCHA_SKILLS_DIR", "")
    out: List[Path] = []
    for chunk in raw.split(os.pathsep):
        chunk = chunk.strip()
        if chunk:
            out.append(Path(chunk))
    return out


class SkillsManager:
    """Scans every configured skills directory and caches the result until
    :meth:`reload`. Resilient by contract: one bad folder never breaks the
    rest."""

    def __init__(self) -> None:
        self._loaders: List[SkillsDirectory] = []
        self._tools: Dict[str, Tool] = {}

    def reload(self) -> Dict[str, Any]:
        """Rescan all directories; returns counts for the API response."""
        self._loaders = []
        tools: Dict[str, Tool] = {}
        dirs = _skills_dirs()
        scanned = 0
        for directory in dirs:
            loader = SkillsDirectory(directory)
            found = loader.list_skills()
            scanned += len(found)
            self._loaders.append(loader)
            for tool in loader.build_tools():
                # First directory wins on name conflicts (deterministic).
                tools.setdefault(tool.name, tool)
        self._tools = tools
        logger.info(
            "skills: %d skill(s) from %d director(y/ies)", len(tools), len(dirs),
        )
        return {"directories": [str(d) for d in dirs], "scanned": scanned, "registered": len(tools)}

    def ensure_loaded(self) -> None:
        if not self._loaders and _skills_dirs():
            self.reload()

    def list_skills(self) -> List[Dict[str, Any]]:
        self.ensure_loaded()
        out: List[Dict[str, Any]] = []
        for loader in self._loaders:
            for spec_obj in loader.list_skills():
                registered = self._tools.get(spec_obj.name) is not None
                out.append(
                    {
                        "name": spec_obj.name,
                        "description": spec_obj.description,
                        "mode": spec_obj.mode,
                        "when_to_use": spec_obj.when_to_use,
                        "argument_hint": spec_obj.argument_hint,
                        "execution_context": spec_obj.execution_context,
                        "allowed_tools": list(spec_obj.allowed_tools),
                        "triggers": list(spec_obj.triggers),
                        "source_dir": str(spec_obj.source_dir),
                        "available": registered,
                    }
                )
        return out

    def active_tool_specs(self) -> List[ToolSpec]:
        """ToolSpecs of all registered skills, for injection into runs."""
        self.ensure_loaded()
        specs: List[ToolSpec] = []
        for tool in self._tools.values():
            spec_obj = getattr(tool, "spec", None)
            if spec_obj is not None:
                specs.append(spec_obj)
        return specs

    def register_into(self, registry: ToolRegistry) -> int:
        """Register every skill tool into a ToolRegistry (dev surfaces)."""
        count = 0
        for name, tool in self._tools.items():
            if not registry.has(name):
                registry.register(tool)
                count += 1
        return count


_manager: SkillsManager | None = None


def skills_manager() -> SkillsManager:
    global _manager
    if _manager is None:
        _manager = SkillsManager()
    return _manager


__all__ = ["SkillsManager", "skills_manager"]
