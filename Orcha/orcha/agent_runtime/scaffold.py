"""Project scaffolding for small models: one focused step per named file, verified mechanically.

Why this exists. Asked for "a Snake game: app.py, templates/index.html, static/style.css and static/game.js", a 3B model driven
through the general planner produced 17 overlapping steps ("Create Flask app" / "Setup Flask app" / "Add styling" ...), finished
four, and wrote stubs (game.js: 249 bytes; index.html never created). Small models are weak at long plans and at emitting
kilobytes of escaped JSON inside a tool call; they are much better at "write THIS one file, as plain text".

So when a request explicitly names two or more files to create, ORCHA (a) plans exactly one step per file, in dependency order
(backend, page, styles, script, tests); (b) asks the model for that file's content only, as a fenced code block, showing it the
files already written so names, element ids and URLs stay consistent; (c) writes it through the normal, policy-checked tool
executor; (d) verifies it with plain code - not a stub, no placeholders, syntax OK, the features the request names are present,
ids used by the script exist in the page - and feeds concrete problems back for up to two rewrites; (e) finally smoke-tests what
it can. No extra "are you sure?" model calls: every check is mechanical, which suits edge hardware.
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

_EXT = r"py|js|mjs|ts|tsx|jsx|html?|css|json|md|txt|ya?ml|toml|sh|bat|ps1|sql|java|c|cpp|h|cs|go|rs|rb|php"
_PATH_RE = re.compile(rf"(?<![\w/.\\-])((?:[\w.-]+/)*[\w-][\w.-]*\.(?:{_EXT}))(?![\w/])", re.I)
_BUILD_VERB_RE = re.compile(r"\b(create|build|make|write|generate|implement|develop|scaffold|set up)\b", re.I)
_QUESTION_LEAD_RE = re.compile(r"^\W*(explain|describe|what|how|why|who|where|when|which|is|are|does|do|can|could|should|would)\b", re.I)
_MAX_RETRIES = 2
_MAX_RETRIES_BY_KIND = {"js": 3}      # the script is the hardest file for a small model: one more rewrite
_MAX_CONTINUATIONS = 3
_MIN_CHARS = {"py": 80, "test": 150, "html": 250, "css": 250, "js": 600, "json": 2, "md": 20, "other": 10}
_PLACEHOLDER_RE = re.compile(r"rest of (the )?(code|file|implementation)|your (code|logic) here|implement (this|here|later)|"
                             r"\bTODO\b|//\s*\.\.\.|#\s*\.\.\.|/\*\s*\.\.\.\s*\*/|^\s*\.\.\.\s*$|code (goes|omitted)|same as before", re.I | re.M)
_KIND_ORDER = {"manifest": 0, "py": 1, "html": 2, "css": 3, "js": 4, "other": 4, "test": 5, "md": 6}

# spec phrase -> (code regex that must appear, human description, file kinds it applies to)
_FEATURES: List[Tuple[str, str, str, Tuple[str, ...]]] = [
    (r"localstorage", r"localStorage", "saves/reads with localStorage", ("js",)),
    (r"touch|swipe", r"touchstart|touchmove|touchend|pointerdown", "touch/swipe handlers (touchstart/touchend)", ("js",)),
    (r"sound|audio", r"AudioContext|new Audio|\.play\(|oscillator", "sound effects (AudioContext or Audio)", ("js",)),
    (r"\bcanvas\b", r"getContext\(", "canvas drawing (getContext)", ("js",)),
    (r"\bcanvas\b", r"<canvas", "a <canvas> element", ("html",)),
    (r"arrow keys|wasd|keyboard", r"keydown|keyup|addEventListener\(\s*['\"]key", "keyboard handling (keydown)", ("js",)),
    (r"wasd", r"['\"]w['\"]|['\"]a['\"]|KeyW|KeyA", "WASD keys (case 'w', 'a', 's', 'd')", ("js",)),
    (r"\bpause\b", r"pause", "a pause function/button", ("js", "html")),
    (r"\brestart\b", r"restart|reset", "restart/reset logic", ("js", "html")),
    (r"high[- ]?score", r"high", "a high score", ("js", "html")),
    (r"game[- ]?over", r"game[- ]?over|gameOver", "a game-over state/message", ("js", "html")),
    (r"responsive|mobile", r"@media|clamp\(|\bvw\b|\bvh\b|max-width|min\(", "responsive CSS (@media / relative units)", ("css",)),
    (r"animation|transition|smooth", r"@keyframes|transition|animation", "CSS transitions/animations", ("css",)),
    (r"\bhover\b", r":hover", "a :hover style", ("css",)),
    (r"\bflask\b", r"from flask import|import flask", "a Flask import", ("py",)),
    (r"\bflask\b", r"@\w+\.route\(|add_url_rule", "a Flask route", ("py",)),
    (r"\bflask\b.*templates|templates.*\bflask\b", r"render_template", "render_template for the page", ("py",)),
]


@dataclass
class PlannedFile:
    path: str
    kind: str


def file_kind(path: str) -> str:
    base = os.path.basename(path).lower()
    ext = base.rsplit(".", 1)[-1] if "." in base else ""
    if base in ("requirements.txt", "package.json", "pyproject.toml"):
        return "manifest"
    if ext == "py":
        return "test" if base.startswith("test_") or base.endswith("_test.py") else "py"
    if ext in ("html", "htm"):
        return "html"
    if ext == "css":
        return "css"
    if ext in ("js", "mjs", "ts", "tsx", "jsx"):
        return "js"
    if ext == "json":
        return "json"
    if ext == "md":
        return "md"
    return "other"


def _safe(path: str) -> bool:
    p = path.replace("\\", "/")
    return not (p.startswith("/") or ".." in p.split("/") or re.match(r"^[A-Za-z]:", p))


def plan_files(task: str) -> Optional[List[PlannedFile]]:
    """The files a build request names, in build order - or None when this is not an explicit multi-file build."""
    text = task or ""
    if os.environ.get("ORCHA_SCAFFOLD", "1") == "0" or not _BUILD_VERB_RE.search(text) or _QUESTION_LEAD_RE.match(text):
        return None
    found: Dict[str, str] = {}
    for m in _PATH_RE.finditer(text):
        path = m.group(1).replace("\\", "/")
        if not _safe(path) or re.search(r"\d+\.\d+\.\d+", path):
            continue
        base = os.path.basename(path).lower()
        if base not in found or path.count("/") > found[base].count("/"):     # prefer "templates/index.html" over "index.html"
            found[base] = path
    paths = list(found.values())
    if len(paths) < 2:
        return None
    files = [PlannedFile(p, file_kind(p)) for p in paths]
    files.sort(key=lambda f: (_KIND_ORDER.get(f.kind, 4), paths.index(f.path)))
    return files


# ------------------------------------------------------------------------------------------------- generation
_SYSTEM = ("You write ONE file of a small software project. Reply with ONLY the complete contents of that file inside a single "
           "fenced code block. The file must be complete and working: no placeholders, no '...', no 'rest of code', no TODO, "
           "no explanations before or after the block.")


def extract_code(text: str, tidy: bool = True) -> str:
    """Contents of the first fenced block (tolerating a missing closing fence); else the text itself.

    ``tidy=False`` keeps the exact text (needed to splice a continuation onto a response that was cut mid-line)."""
    m = re.search(r"```[\w+-]*[ \t]*\r?\n(.*?)(?:```|\Z)", text or "", re.S)
    body = m.group(1) if m else (text or "")
    return body if not tidy else body.strip("\n").rstrip() + "\n"


def _ids_in_html(html: str) -> List[str]:
    return sorted(set(re.findall(r"""\bid\s*=\s*["']([\w-]+)["']""", html)))


def _guidance(f: PlannedFile, task: str, written: Dict[str, str]) -> str:
    low = task.lower()
    if f.kind == "py" and "flask" in low:
        tpl = [p for p in written if p.endswith(".html")]
        return ("Use Flask. Serve the page at '/' with render_template('index.html') and start it with app.run(debug=True) under "
                "`if __name__ == '__main__'`. Add any small JSON API routes only if the request asks for them.")
    if f.kind == "html":
        flask = "flask" in low and f.path.replace("\\", "/").startswith("templates/")
        return ("Give EVERY element that a script will need a unique id (score, high score, buttons, canvas, overlay, message). "
                + ("Link the stylesheet and script with {{ url_for('static', filename='...') }}. " if flask else
                   "Link the stylesheet and script with relative paths. ")
                + "Include all the visible sections the request lists (title, scores, canvas, buttons, overlay, instructions).")
    if f.kind == "js":
        html = next((c for p, c in written.items() if p.endswith(".html")), "")
        ids = _ids_in_html(html)
        buttons = re.findall(r"""<button[^>]*\bid\s*=\s*["']([\w-]+)["']""", html)
        text = ("Use EXACTLY these element ids from the page (do not invent others): " + (", ".join(ids) or "(none)") + ". "
                "Implement EVERY behaviour the request lists for the browser side, completely, in this one file. "
                "Declare every variable, constant and function BEFORE it is used (const startBtn = document.getElementById(...)).")
        if buttons:
            text += " Give EVERY one of these buttons a click handler: " + ", ".join(buttons) + "."
        if re.search(r"\bcanvas\b", low):
            text += (" Structure it like this: (1) constants and state; (2) small functions draw(), update(), and helpers; (3) ONE loop function "
                     "that calls update() then draw(), started with setInterval when Start is pressed (and cleared on pause/game over); "
                     "(4) event listeners at the bottom. draw() must paint the whole scene every tick (clearRect, then the shapes). "
                     "Keep positions on a grid of equal-sized cells so collisions with food compare equal numbers.")
        return text
    if f.kind == "css":
        return "Style every id and class used by the page. Cover every visual requirement of the request."
    if f.kind == "test":
        return "Use unittest. Import from the module under test by its file name. Include the specific checks the request names."
    return ""


def _context(written: Dict[str, str]) -> str:
    if not written:
        return "No files have been written yet."
    parts = []
    for path, code in written.items():
        cap = 4500 if path.endswith((".html", ".py", ".css")) else 1500
        parts.append(f"--- {path} ---\n{code[:cap]}")
    return "Files ALREADY written (keep names, ids, routes and URLs consistent with them):\n" + "\n".join(parts)


def _prompt(f: PlannedFile, task: str, plan: List[PlannedFile], written: Dict[str, str], feedback: str = "", previous: str = "") -> str:
    lines = [f"PROJECT REQUEST:\n{task.strip()[:6500]}", "", "ALL PROJECT FILES (written in this order): " + ", ".join(p.path for p in plan), "",
             _context(written), "", f"NOW WRITE THE COMPLETE FILE: `{f.path}`", _guidance(f, task, written)]
    if feedback:
        lines += ["", "YOUR PREVIOUS VERSION HAD THESE PROBLEMS - fix ALL of them:", feedback,
                  "Previous version:\n```\n" + previous[:6000] + "\n```", "Rewrite the COMPLETE file."]
    lines.append("Reply with one fenced code block containing the entire file.")
    return "\n".join(x for x in lines if x is not None)


# ------------------------------------------------------------------------------------------------ verification
def _node_check(code: str, suffix: str) -> Optional[str]:
    node = shutil.which("node")
    if not node or suffix not in (".js", ".mjs"):
        return None
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as fh:
        fh.write(code)
        name = fh.name
    try:
        r = subprocess.run([node, "--check", name], capture_output=True, text=True, timeout=30)
        return None if r.returncode == 0 else (r.stderr.strip().splitlines() or ["syntax error"])[-1][:200] if r.stderr else "syntax error"
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass


def _balanced(code: str, pairs: str = "(){}[]") -> bool:
    """Bracket balance ignoring strings/comments roughly - a cheap syntax proxy when no parser is available."""
    stripped = re.sub(r"//[^\n]*|/\*.*?\*/|'(?:\\.|[^'\\\n])*'|\"(?:\\.|[^\"\\\n])*\"|`(?:\\.|[^`\\])*`", "", code, flags=re.S)
    for a, b in (("(", ")"), ("{", "}"), ("[", "]")):
        if stripped.count(a) != stripped.count(b):
            return False
    return True


def verify_file(f: PlannedFile, code: str, task: str, written: Dict[str, str], plan: List[PlannedFile]) -> List[str]:
    """Concrete, mechanical problems with a generated file (empty list = it passes)."""
    problems: List[str] = []
    low = task.lower()
    if len(code.strip()) < _MIN_CHARS.get(f.kind, 10):
        problems.append(f"It is only {len(code.strip())} characters - a stub. Write the full, working file.")
    ph = _PLACEHOLDER_RE.search(code)
    if ph:
        problems.append(f"It contains a placeholder ('{ph.group(0).strip()[:30]}'). Write every part out completely.")
    ext = os.path.splitext(f.path)[1].lower()
    if f.kind in ("py", "test"):
        try:
            ast.parse(code)
        except SyntaxError as exc:
            problems.append(f"Python syntax error on line {exc.lineno}: {exc.msg}.")
    elif f.kind == "js":
        err = _node_check(code, ext)
        if err:
            problems.append(f"JavaScript syntax error: {err}")
        elif not _balanced(code):
            problems.append("Unbalanced brackets or braces - the file is probably cut off or malformed.")
    elif f.kind == "json":
        try:
            json.loads(code)
        except ValueError as exc:
            problems.append(f"Invalid JSON: {exc}")
    elif f.kind == "css" and not _balanced(code, "{}"):
        problems.append("Unbalanced braces in the CSS.")
    elif f.kind == "html":
        if code.count("<") != code.count(">") and abs(code.count("<") - code.count(">")) > 2:
            problems.append("The HTML has unbalanced angle brackets.")
        for other in plan:
            if other.kind in ("css", "js") and os.path.basename(other.path) not in code:
                problems.append(f"The page never references {os.path.basename(other.path)}.")
    missing = []
    for spec_re, code_re, desc, kinds in _FEATURES:
        if f.kind in kinds and re.search(spec_re, low) and not re.search(code_re, code, re.I):
            missing.append(desc)
    if missing:
        problems.append("It is missing: " + "; ".join(dict.fromkeys(missing)) + ".")
    if f.kind == "js":
        html = next((c for p, c in written.items() if p.endswith(".html")), "")
        if html:
            # every button the page shows must be wired up by the script
            buttons = re.findall(r"""<button[^>]*\bid\s*=\s*["']([\w-]+)["']""", html)
            unwired = [b for b in buttons if b not in code]
            if unwired:
                problems.append("These buttons exist on the page but the script never uses them: " + ", ".join(unwired) + ".")
            if True:              # always run the fake-browser pass: problem COUNTS must be comparable between attempts
                from . import scaffold_js
                problems += scaffold_js.run(code, html, needs_canvas=bool(re.search(r"\bcanvas\b", low)))
            have = set(_ids_in_html(html))
            used = set(re.findall(r"""getElementById\(\s*['"]([\w-]+)['"]""", code)) | set(re.findall(r"""querySelector\(\s*['"]#([\w-]+)""", code))
            absent = sorted(used - have)
            if absent:
                problems.append("It uses element ids that the page does not have: " + ", ".join(absent) +
                                ". Page ids: " + ", ".join(sorted(have)) + ".")
    return problems


# ---------------------------------------------------------------------------------------------------- runner
@dataclass
class ScaffoldResult:
    steps: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    completed: bool = False
    written: Dict[str, str] = field(default_factory=dict)
    problems: Dict[str, List[str]] = field(default_factory=dict)


async def _generate(completion_fn: Callable, prompt: str) -> str:
    """One file's text, continuing if the model hit its token ceiling."""
    messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
    parts: List[str] = []
    for _ in range(_MAX_CONTINUATIONS + 1):
        msg = await completion_fn(messages, _SYSTEM, None)
        piece = str(msg.get("content") or "")
        parts.append(extract_code(piece, tidy=False))
        if msg.get("finish_reason") != "length":
            break
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": piece},
                    {"role": "user", "content": "You were cut off. Continue EXACTLY where you stopped and output only the rest of "
                                                "the file in a fenced code block."}]
    return "".join(parts).strip("\n").rstrip() + "\n"


_UNDEF_RE = re.compile(r"\b([A-Za-z_$][\w$]{2,}) is not defined")
_DEF_SYSTEM = ("You write ONE missing definition for a JavaScript file. Reply with ONLY a fenced code block containing that single "
               "function or const/let declaration - not the rest of the file.")


def undefined_names(problems: List[str]) -> List[str]:
    """Identifiers the fake browser reported as 'X is not defined' (3+ characters: shorter ones are usually a missing parameter)."""
    seen: List[str] = []
    for p in problems:
        for n in _UNDEF_RE.findall(p):
            if n not in seen and n not in ("window", "document"):
                seen.append(n)
    return seen[:3]


def insert_definition(code: str, snippet: str) -> str:
    """Insert ``snippet`` at the outermost scope of the file, before its first top-level declaration."""
    lines = code.split("\n")
    decl = re.compile(r"^(\s*)(function\b|async function\b|const\b|let\b|var\b|class\b)")
    idx = [(len(m.group(1).expandtabs(4)), i) for i, ln in enumerate(lines) if (m := decl.match(ln))]
    if not idx:
        return snippet.rstrip() + "\n\n" + code
    indent, at = min(idx, key=lambda t: (t[0], t[1]))
    pad = " " * indent
    block = "\n".join((pad + ln if ln.strip() else ln) for ln in snippet.strip("\n").split("\n"))
    return "\n".join(lines[:at] + [block, ""] + lines[at:])


async def repair_undefined(completion_fn: Callable, f: PlannedFile, code: str, problems: List[str]) -> str:
    """Ask for ONLY each missing definition (a tiny, well-scoped task for a small model) and insert it. Returns the new code."""
    out = code
    for name in undefined_names(problems):
        uses = [ln.strip() for ln in out.split("\n") if re.search(r"(?<![\w$.])" + re.escape(name) + r"(?![\w$])", ln)][:6]
        prompt = (f"In `{f.path}` the name `{name}` is used but never defined, so the script crashes. These lines use it:\n"
                  + "\n".join(f"  {u}" for u in uses) + f"\n\nThe rest of the file, for context:\n```js\n{out[:9000]}\n```\n\n"
                  f"Write the definition of `{name}` (a function, or a const/let) so those lines work with the existing code. "
                  "Use only variables that already exist in the file.")
        msg = await completion_fn([{"role": "user", "content": prompt}], _DEF_SYSTEM, None)
        snippet = extract_code(str(msg.get("content") or ""))
        if len(snippet.strip()) < 8 or not re.search(rf"\b{re.escape(name)}\b", snippet.split("\n", 3)[0] + snippet[:200]):
            continue
        out = insert_definition(out, snippet)
    return out


_SR_RE = re.compile(r"<<<<<<<[ \t]*SEARCH[ \t]*\r?\n(.*?)\r?\n=======[ \t]*\r?\n(.*?)\r?\n?>>>>>>>[ \t]*REPLACE", re.S)


async def _generate_raw(completion_fn: Callable, prompt: str) -> str:
    msg = await completion_fn([{"role": "user", "content": prompt}], _PATCH_SYSTEM, None)
    return str(msg.get("content") or "")


_PATCH_SYSTEM = ("You fix a file with small patches. Reply with ONLY patch blocks, nothing else. Each block is exactly:\n"
                 "<<<<<<< SEARCH\n(lines copied EXACTLY from the file)\n=======\n(the replacement lines)\n>>>>>>> REPLACE")


def _patch_prompt(f: PlannedFile, code: str, problems: List[str]) -> str:
    return (f"This is the current `{f.path}`:\n```\n{code[:12000]}\n```\n\nIt has these problems:\n" + "\n".join(f"- {p}" for p in problems) +
            "\n\nFix ALL of them with as few SEARCH/REPLACE blocks as possible. Copy each SEARCH text exactly from the file. "
            "To ADD code (a missing function, variable or handler), SEARCH one nearby existing line and put that line plus the new code in REPLACE. "
            "Every variable and function you use must be declared.")


def apply_search_replace(code: str, reply: str) -> Tuple[str, int]:
    """Apply SEARCH/REPLACE blocks with ORCHA's tolerant edit ladder; returns (new code, blocks applied)."""
    from ..capabilities.editing import EditError, replace
    text, applied = code, 0
    for m in _SR_RE.finditer(reply or ""):
        try:
            text = replace(text, m.group(1), m.group(2)).text
            applied += 1
        except EditError:
            continue
    return text, applied


def smoke_test(root: str, written: Dict[str, str], task: str) -> List[str]:
    """Best-effort checks that need a runtime: compile Python, and hit a Flask '/' route when Flask is installed."""
    notes: List[str] = []
    py = "python"
    for path in written:
        if path.endswith(".py"):
            r = subprocess.run([py, "-m", "py_compile", path], cwd=root, capture_output=True, text=True, timeout=60)
            notes.append(f"{path}: compiles" if r.returncode == 0 else f"{path}: does NOT compile ({r.stderr.strip()[-120:]})")
    entry = next((p for p, c in written.items() if p.endswith(".py") and re.search(r"Flask\(", c)), None)
    if entry:
        mod = os.path.splitext(os.path.basename(entry))[0]
        code = ("import importlib, sys; sys.path.insert(0, '.'); m = importlib.import_module(%r); a = getattr(m, 'app'); "
                "r = a.test_client().get('/'); print(r.status_code, b'<canvas' in r.data or b'<html' in r.data.lower())" % mod)
        try:
            r = subprocess.run([py, "-c", code], cwd=os.path.join(root, os.path.dirname(entry)) if os.path.dirname(entry) else root,
                               capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and r.stdout.strip().startswith("200"):
                notes.append("Flask: GET / returned 200 with the page")
            elif "No module named 'flask'" in r.stderr:
                notes.append("Flask is not installed here, so the server was not started (pip install flask)")
            else:
                notes.append("Flask: GET / failed: " + (r.stderr.strip().splitlines() or [r.stdout.strip()])[-1][:140])
        except (OSError, subprocess.SubprocessError):
            notes.append("Flask smoke test could not run")
    return notes


async def run_scaffold(*, task: str, plan: List[PlannedFile], completion_fn: Callable, write: Callable[[str, str], Any],
                       root: Optional[str], cancelled: Callable[[], bool] = lambda: False) -> ScaffoldResult:
    """Build every planned file. ``write(path, content)`` performs the (policy-checked) write and returns a tool result."""
    res = ScaffoldResult()
    for i, f in enumerate(plan, 1):
        if cancelled():
            break
        t0 = time.perf_counter()
        feedback, previous, attempts, code, problems = "", "", 0, "", []
        best: Optional[Tuple[int, str, List[str]]] = None       # a rewrite can fix one bug and add another: keep the best version
        patched = False
        while True:
            attempts += 1
            new_code = ""
            if previous and f.kind == "js" and undefined_names(problems):
                fixed = await repair_undefined(completion_fn, f, previous, problems)
                if fixed != previous:
                    new_code = fixed
            if not new_code and previous and problems and len(previous) > 1200 and attempts >= 3 and not patched:
                # Later attempts: ask for small SEARCH/REPLACE patches instead of rewriting kilobytes again (whack-a-mole otherwise).
                reply = await _generate_raw(completion_fn, _patch_prompt(f, previous, problems))
                patched_code, n = apply_search_replace(previous, reply)
                if n:
                    new_code = patched_code
                else:
                    patched = True                         # patches did not apply; fall back to full rewrites from here on
            if not new_code:
                new_code = await _generate(completion_fn, _prompt(f, task, plan, res.written, feedback, previous))
            code = new_code
            problems = verify_file(f, code, task, res.written, plan)
            if best is None or len(problems) <= best[0]:
                best = (len(problems), code, problems)
            if not problems or attempts > _MAX_RETRIES_BY_KIND.get(f.kind, _MAX_RETRIES):
                break
            feedback, previous = "\n".join(f"- {p}" for p in problems), code
        if best is not None:
            _, code, problems = best
        if root:
            os.makedirs(os.path.dirname(os.path.join(root, f.path)) or root, exist_ok=True)
        result = await write(f.path, code)
        ok = bool(getattr(result, "ok", True))
        res.tool_calls.append({"name": "write_file", "arguments": {"path": f.path, "content": code[:200] + ("..." if len(code) > 200 else "")},
                               "iteration": i, "result_type": "ok" if ok else "error",
                               "result": (result.to_message(300) if hasattr(result, "to_message") else str(result))[:300],
                               "note": f"scaffold step {i}/{len(plan)}, attempt {attempts}" + ("" if not problems else " (still had problems)")})
        if ok:
            res.written[f.path] = code
        if problems or not ok:
            res.problems[f.path] = problems or ["the file could not be written"]
        res.steps.append({"iteration": i, "response_preview": f"{f.path}: {len(code)} chars", "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
                          "status": "completed" if ok and not problems else "continued",
                          "note": f"scaffold: {f.path} written after {attempts} attempt(s)" + (f"; unresolved: {'; '.join(problems)[:200]}" if problems else "")})
    smoke: List[str] = []
    if root and res.written:
        try:
            smoke = await asyncio.to_thread(smoke_test, root, res.written, task)
        except Exception as exc:  # noqa: BLE001 - a smoke test must never fail the build
            smoke = [f"smoke test skipped: {exc}"]
    done = len(res.written) == len(plan) and not res.problems
    lines = [("Built " if done else "Partly built ") + f"{len(res.written)} of {len(plan)} files:"]
    lines += [f"  - {p}" + (f"  (PROBLEMS: {'; '.join(res.problems[p])[:160]})" if p in res.problems else "") for p in [f.path for f in plan] if p in res.written or p in res.problems]
    if smoke:
        lines += ["Checks run:"] + [f"  - {s}" for s in smoke]
    res.answer, res.completed = "\n".join(lines), done
    return res
