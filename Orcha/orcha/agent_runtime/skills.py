"""
orcha.agent_runtime.skills
==========================
Local skills loader — Orcha's folder-per-skill convention, upgraded to the
interoperable SKILL.md package format:

    skills_dir/
      <skill-name>/
        SKILL.md        # required. Metadata as frontmatter (--- … ---):
                        #   name, description, triggers (hint words)
                        #   when_to_use, argument_hint            (optional)
                        #   parameters (JSON schema)             (optional)
                        #   allowed_tools: [rule strings]        (optional)
                        #   context: inline | fork               (optional)
                        #   entrypoint (default "entrypoint.py") (optional*)
                        # The body below the frontmatter is the skill's
                        # prompt/instructions for the model.
        entrypoint.py   # optional* when mode is prompt. Exposes
                        # ``def run(**kwargs)``.

Two execution flavors
---------------------
**script** (classic) — ``entrypoint.py`` present: the tool call imports and
runs it lazily; the return value is the tool result.

**prompt** (new, no code) — a SKILL.md whose body carries instructions and
no usable entrypoint: calling the tool expands the body into a directive
returned to the model — ``$ARGUMENTS`` becomes the validated JSON of the
call arguments and ``${SKILL_DIR}`` the skill folder path (so auxiliary
files can be referenced and read). This makes skills shareable as pure
markdown packages with zero executable code.

``allowed_tools`` frontmatter lists permission rule strings (see
``orcha.capabilities.rules``) that should apply while the skill executes;
they are surfaced on the ToolSpec's ``meta`` so hosts can scope an
executor (fork-style runs) without re-parsing SKILL.md.

Behavior
--------
- Discovery reads ONLY metadata (cheap, no code execution).
- Loading is lazy: entrypoints import on first call, never at scan time.
- Resilience by contract: malformed folders are warned and skipped —
  discovery never crashes AgentRuntime boot.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..capabilities.base import CAUTIOUS, EXECUTE, READ, SAFE, ToolError, spec
from ..nodes.tool import ToolSchema
from .tools import Tool, ToolRegistry

logger = logging.getLogger("orcha.agent_runtime.skills")

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]{0,63}$")
_DEFAULT_PARAMETERS: ToolSchema = {
    "type": "object", "properties": {}, "required": [],
}


# ── Metadata: SKILL.md frontmatter (YAML-ish subset, dependency-free) ───────

def parse_skill_metadata(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse the frontmatter block of a SKILL.md. Returns a metadata dict, or
    None when the block is missing or malformed (the folder is then
    skipped).

    Supported subset: ``key: value`` scalars, quoted strings, inline JSON
    (``{...}`` / ``[...]`` — used for ``parameters`` and inline trigger
    lists), and block lists (``- item`` lines under an empty ``key:``).
    """
    match = re.match(r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)", text, re.DOTALL)
    if not match:
        return None
    raw = match.group(1)
    data: Dict[str, Any] = {}
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        if ":" not in line or line.startswith(" ") or stripped.startswith("-"):
            return None  # not a key: value line — malformed metadata
        key, _, rest = line.partition(":")
        key = key.strip()
        value = rest.strip()
        if not value:
            # block list (triggers etc.)
            items: List[str] = []
            j = i + 1
            while j < len(lines):
                item = lines[j].strip()
                if not item.startswith("- "):
                    break
                items.append(_unquote(item[2:].strip()))
                j += 1
            if items:
                data[key] = items
                i = j
                continue
            data[key] = None
        else:
            data[key] = _parse_value(value)
            if data[key] is None and value not in ("null", "~"):
                return None  # unparseable value — malformed metadata
        i += 1
    return data


def _parse_value(value: str) -> Any:
    if value.startswith("{") or value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return _unquote(value)
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    if value in ("null", "~"):
        return None
    if value in ("", "[]"):
        return []
    return value


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


# ── The skill descriptor ─────────────────────────────────────────────────────

_MAX_PROMPT_RESULT_CHARS = 24_000


@dataclass(frozen=True)
class SkillSpec:
    """One well-formed skill folder, ready to be surfaced as a Tool."""
    name: str
    description: str
    triggers: Tuple[str, ...] = ()
    parameters: ToolSchema = field(default_factory=lambda: dict(_DEFAULT_PARAMETERS))
    entrypoint: Optional[Path] = None
    source_dir: Path = None  # type: ignore[assignment]
    instructions: str = ""
    # "script" (entrypoint.py) or "prompt" (markdown body expansion).
    mode: str = "script"
    # Permission rule strings that should scope execution, e.g.
    # ("run_command(git *)", "edit_file(src/**)"). Empty = no scoping asked.
    allowed_tools: Tuple[str, ...] = ()
    # inline = expand in this conversation; fork = host should prefer a
    # sub-agent run. Metadata only — enforcement lives with the caller.
    execution_context: str = "inline"
    argument_hint: str = ""
    when_to_use: str = ""

    @property
    def is_prompt_skill(self) -> bool:
        return self.mode == "prompt"

    def to_prompt_line(self) -> str:
        hints = f" (triggers: {', '.join(self.triggers)})" if self.triggers else ""
        use = f" — {self.when_to_use}" if self.when_to_use else ""
        return f"- {self.name}: {self.description}{use}{hints}"

    def render_prompt(self, arguments: Dict[str, Any]) -> str:
        """
        Expand the SKILL.md body for one invocation: ``$ARGUMENTS`` becomes
        the validated call arguments (the single string value when there is
        exactly one, JSON otherwise), ``${SKILL_DIR}`` the skill folder.
        Deterministic and side-effect free.
        """
        body = self.instructions
        if "$ARGUMENTS" in body or "${ARGUMENTS}" in body:
            if len(arguments) == 1 and isinstance(next(iter(arguments.values())), str):
                rendered = next(iter(arguments.values()))
            elif arguments:
                rendered = json.dumps(arguments, ensure_ascii=False, default=str)
            else:
                rendered = "(no arguments provided)"
            body = body.replace("${ARGUMENTS}", rendered).replace("$ARGUMENTS", rendered)
        if "${SKILL_DIR}" in body:
            body = body.replace("${SKILL_DIR}", str(self.source_dir))
        if len(body) > _MAX_PROMPT_RESULT_CHARS:
            body = (
                body[:_MAX_PROMPT_RESULT_CHARS]
                + f"\n… (skill instructions truncated at {_MAX_PROMPT_RESULT_CHARS} chars)"
            )
        return body


# ── The loader ───────────────────────────────────────────────────────────────

class SkillsDirectory:
    """
    Scans one skills directory. Metadata is read eagerly (cheap); skill
    entrypoints are imported lazily on first call and cached.

    Resilient by contract: malformed folders are warned and skipped.
    """

    def __init__(
        self, directory: str | Path, *, log: Optional[logging.Logger] = None,
    ) -> None:
        self._dir = Path(directory)
        self._log = log or logger
        self._loaded: Dict[str, Any] = {}

    @property
    def directory(self) -> Path:
        return self._dir

    # ── Discovery (metadata only) ─────────────────────────────────────

    def list_skills(self) -> List[SkillSpec]:
        if not self._dir.is_dir():
            self._log.warning(
                "skills directory %r does not exist — skipping", str(self._dir),
            )
            return []
        skills: List[SkillSpec] = []
        for folder in sorted(self._dir.iterdir()):
            if not folder.is_dir():
                continue  # stray files are not skill folders
            spec_obj = self._scan_folder(folder)
            if spec_obj is not None:
                skills.append(spec_obj)
        return skills

    def _scan_folder(self, folder: Path) -> Optional[SkillSpec]:
        meta_path = folder / "SKILL.md"
        if not meta_path.is_file():
            self._log.warning(
                "skipping skill folder %r: missing SKILL.md", folder.name,
            )
            return None
        try:
            text = meta_path.read_text(encoding="utf-8")
        except OSError as exc:
            self._log.warning(
                "skipping skill folder %r: cannot read SKILL.md: %s", folder.name, exc,
            )
            return None

        metadata = parse_skill_metadata(text)
        if metadata is None:
            self._log.warning(
                "skipping skill folder %r: missing or malformed metadata "
                "frontmatter in SKILL.md", folder.name,
            )
            return None

        name = metadata.get("name")
        if not isinstance(name, str) or not _NAME_RE.match(name):
            self._log.warning(
                "skipping skill folder %r: invalid or missing skill name %r",
                folder.name, name,
            )
            return None

        instructions = _body_after_frontmatter(text)

        # Execution flavor: explicit mode wins; otherwise a usable
        # entrypoint means script, and a non-empty body alone means prompt.
        entrypoint_name = metadata.get("entrypoint") or "entrypoint.py"
        entrypoint = folder / str(entrypoint_name)
        declared_mode = metadata.get("mode")
        if declared_mode not in (None, "script", "prompt"):
            self._log.warning(
                "skipping skill %r: unknown mode %r", name, declared_mode,
            )
            return None
        if declared_mode == "script" or (declared_mode is None and entrypoint.is_file()):
            if not entrypoint.is_file():
                self._log.warning(
                    "skipping skill %r: script mode but entrypoint %r not found",
                    name, str(entrypoint_name),
                )
                return None
            mode = "script"
        else:
            if not instructions:
                self._log.warning(
                    "skipping skill %r: no executable entrypoint and an empty "
                    "SKILL.md body — nothing to run or to expand", name,
                )
                return None
            mode = "prompt"

        description = metadata.get("description") or name
        if not isinstance(description, str):
            description = name
        triggers = metadata.get("triggers") or ()
        if not isinstance(triggers, (list, tuple)):
            triggers = ()
        triggers = tuple(str(t) for t in triggers if isinstance(t, str))
        parameters = metadata.get("parameters")
        if not isinstance(parameters, dict):
            parameters = dict(_DEFAULT_PARAMETERS)

        allowed_raw = metadata.get("allowed_tools") or ()
        if isinstance(allowed_raw, str):
            allowed_raw = [allowed_raw]
        if not isinstance(allowed_raw, (list, tuple)):
            allowed_raw = ()
        allowed_tools = tuple(
            str(r) for r in allowed_raw if isinstance(r, str) and r.strip()
        )

        context = metadata.get("context")
        execution_context = context if context in ("inline", "fork") else "inline"
        argument_hint = metadata.get("argument_hint") or ""
        when_to_use = metadata.get("when_to_use") or ""

        return SkillSpec(
            name=name,
            description=description,
            triggers=triggers,
            parameters=parameters,
            entrypoint=entrypoint if mode == "script" else None,
            source_dir=folder,
            instructions=instructions,
            mode=mode,
            allowed_tools=allowed_tools,
            execution_context=execution_context,
            argument_hint=str(argument_hint) if argument_hint else "",
            when_to_use=str(when_to_use) if when_to_use else "",
        )

    # ── Lazy loading ──────────────────────────────────────────────────

    def load_entrypoint(self, skill: SkillSpec) -> Any:
        """Import (once) the skill's entrypoint module. The module must
        expose a callable ``run``. Raises ToolError on any load failure —
        the Tool executor turns it into a clean ErrorObservation."""
        cached = self._loaded.get(skill.name)
        if cached is not None:
            return cached
        module_name = f"orcha_skill_{re.sub(r'[^A-Za-z0-9_]', '_', skill.name)}"
        try:
            module_spec = importlib.util.spec_from_file_location(
                module_name, skill.entrypoint,
            )
            if module_spec is None or module_spec.loader is None:
                raise ToolError(
                    "skill_load_failed", f"cannot build module spec for {skill.entrypoint}",
                )
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            module_spec.loader.exec_module(module)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(
                "skill_load_failed", f"skill {skill.name!r} failed to load: {exc}",
            ) from exc
        if not callable(getattr(module, "run", None)):
            raise ToolError(
                "skill_bad_entrypoint",
                f"skill {skill.name!r} entrypoint does not define run(**kwargs)",
            )
        self._loaded[skill.name] = module
        return module

    # ── Building Orcha Tools ──────────────────────────────────────────

    def build_tools(self) -> List[Tool]:
        return [skill_to_tool(spec_obj, self) for spec_obj in self.list_skills()]

    def prompt_listing(self, max_chars: int = 8_000) -> str:
        """
        Budget-capped catalog of the directory's skills for injection into a
        system prompt or tool description (the "1% of context" convention).
        Newest-discovered last; entries beyond the cap are summarized in a
        trailing count line.
        """
        lines: List[str] = []
        used = 0
        skipped = 0
        for skill_obj in self.list_skills():
            line = skill_obj.to_prompt_line()
            if used + len(line) > max_chars and lines:
                skipped += 1
                continue
            lines.append(line)
            used += len(line) + 1
        if not lines:
            return ""
        if skipped:
            lines.append(f"… and {skipped} more skill(s) omitted (listing budget)")
        return "\n".join(lines)


def _body_after_frontmatter(text: str) -> str:
    match = re.match(r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)", text, re.DOTALL)
    if not match:
        return ""
    return text[match.end():].strip()


def skill_to_tool(skill: SkillSpec, loader: SkillsDirectory) -> Tool:
    """
    A skill is just a Tool.

    script mode — the entrypoint runs lazily behind the Tool contract.
    prompt mode — the tool result is the SKILL.md body expanded for this
    call ($ARGUMENTS / ${SKILL_DIR} substituted): the model receives the
    skill's directive as its working instructions. No executable code
    involved, so prompt skills are read-only and concurrency-safe.

    ``allowed_tools`` rule strings ride on the spec's ``meta`` so a host can
    scope permissions for (forked) execution without re-reading files.
    """

    if skill.is_prompt_skill:
        def invoke_prompt(**kwargs: Any) -> Any:
            return skill.render_prompt(dict(kwargs or {}))

        spec_obj = spec(
            skill.name,
            skill.description,
            dict(skill.parameters),
            invoke_prompt,
            permissions=[READ],
            safety_level=SAFE,
            capability="skill",
            read_only=True,
            concurrency_safe=True,
            result_format={"type": "string"},
            meta={
                "kind": "skill",
                "mode": "prompt",
                "execution_context": skill.execution_context,
                "argument_hint": skill.argument_hint,
                "allowed_tools": list(skill.allowed_tools),
                "source_dir": str(skill.source_dir),
            },
        )
        return Tool(spec_obj)

    def invoke(**kwargs: Any) -> Any:
        module = loader.load_entrypoint(skill)
        runner = getattr(module, "run")
        return runner(**kwargs)

    spec_obj = spec(
        skill.name,
        skill.description,
        dict(skill.parameters),
        invoke,
        permissions=[EXECUTE],
        safety_level=CAUTIOUS,
        capability="skill",
        meta={
            "kind": "skill",
            "mode": "script",
            "execution_context": skill.execution_context,
            "argument_hint": skill.argument_hint,
            "allowed_tools": list(skill.allowed_tools),
            "source_dir": str(skill.source_dir),
        },
    )
    return Tool(spec_obj)


def register_skills(
    registry: ToolRegistry,
    directory: str | Path,
    *,
    log: Optional[logging.Logger] = None,
) -> List[Tool]:
    """
    Scan a skills directory and register every well-formed skill into
    ``registry`` as an ordinary Tool. Resilient by contract: malformed
    folders and missing directories are warned and skipped — never raises.
    """
    log = log or logger
    loader = SkillsDirectory(directory, log=log)
    tools = loader.build_tools()
    for tool in tools:
        if registry.has(tool.name):
            log.warning("skill tool %r shadows an existing tool", tool.name)
        registry.register(tool)
    log.info("registered %d skill tool(s) from %r", len(tools), str(directory))
    return tools


__all__ = [
    "SkillSpec", "SkillsDirectory", "skill_to_tool", "register_skills",
    "parse_skill_metadata",
]
