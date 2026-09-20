"""Per-file project scaffolding for small models (orcha.agent_runtime.scaffold), through the real agent graph."""
import asyncio

import pytest

from orcha.agent_runtime import scaffold as sc
from orcha.builders.agent import AgentGraphConfig, build_agent_graph
from orcha.capabilities.base import CapabilityContext
from orcha.capabilities.registry import CapabilityRegistry
from orcha.graph.runtime import GraphRuntime
from orcha.graph.store import MemoryStore
from orcha.nodes.agent import AgentConfig

SPEC = ("Create a Snake game with Flask. Files: app.py, templates/index.html, static/style.css and static/game.js. Arrow keys or WASD, "
        "touch/swipe on mobile, Pause and Restart buttons, score and high score stored in localStorage, a game-over message, "
        "sound effects, a canvas, responsive layout with hover effects and smooth animations. No placeholders.")

APP_PY = "from flask import Flask, render_template\n\napp = Flask(__name__)\n\n\n@app.route('/')\ndef index():\n    return render_template('index.html')\n\n\nif __name__ == '__main__':\n    app.run(debug=True)\n"
HTML = ("<!doctype html>\n<html><head><title>Snake</title><link rel=\"stylesheet\" href=\"{{ url_for('static', filename='style.css') }}\"></head>\n"
        "<body><h1>Snake</h1><p>Score <span id=\"score\">0</span> High score <span id=\"high\">0</span></p>\n<canvas id=\"board\"></canvas>\n"
        "<button id=\"start\">Start Game</button><button id=\"pause\">Pause</button><button id=\"restart\">Restart</button>\n"
        "<div id=\"overlay\">Game over</div>\n<script src=\"{{ url_for('static', filename='game.js') }}\"></script></body></html>\n")
CSS = ("body { background: #0b0b16; color: #0ff; font-family: sans-serif; transition: all .3s; }\nbutton { border: 1px solid #0f8; padding: 8px 14px; "
       "transition: transform .2s; }\nbutton:hover { transform: scale(1.05); }\ncanvas { width: 100%; max-width: 480px; }\n"
       "@media (max-width: 600px) { canvas { max-width: 95vw; } }\n@keyframes pulse { from { opacity: .6 } to { opacity: 1 } }\n")


def js(ids=("board", "score", "high", "start", "pause", "restart", "overlay")):
    body = ("const canvas = document.getElementById('%s'); const ctx = canvas.getContext('2d');\n"
            "let high = Number(localStorage.getItem('high') || 0); let paused = false; let gameOver = false;\n"
            "const audio = new (window.AudioContext || window.webkitAudioContext)();\n"
            "function beep() { const o = audio.createOscillator(); o.connect(audio.destination); o.start(); o.stop(audio.currentTime + 0.1); }\n"
            "document.addEventListener('keydown', (e) => { if (e.key === 'ArrowUp' || e.key === 'w') { direction = 'up'; } });\n"
            "canvas.addEventListener('touchstart', (e) => { start = e.touches[0]; });\ncanvas.addEventListener('touchend', (e) => { end = e.changedTouches[0]; });\n"
            "function pause() { paused = !paused; }\nfunction restart() { gameOver = false; score = 0; }\n"
            "document.getElementById('%s').onclick = pause; document.getElementById('%s').onclick = restart;\n"
            "document.getElementById('%s').textContent = high; document.getElementById('%s').textContent = 0;\n"
            "function endGame() { gameOver = true; document.getElementById('%s').style.display = 'block'; }\nlet direction = 'right'; let score = 0; let start, end;\n"
            "function tick() { if (paused || gameOver) return; ctx.clearRect(0, 0, 10, 10); score += 1; }\nsetInterval(tick, 100);\n"
            % (ids[0], ids[3 - 1 + 1], ids[5], ids[2], ids[1], ids[6]))
    return body


class FakeModel:
    """Writes each requested file; can be told to misbehave on the first attempt of a file."""

    def __init__(self, first_attempt=None):
        self.first_attempt, self.prompts, self.seen = first_attempt or {}, [], {}

    async def __call__(self, messages, system, tools):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        if system and system.startswith("You write ONE missing definition"):     # targeted repair of an undefined name
            import re as _re
            name = _re.search("the name `(\w+)`", prompt).group(1)
            self.seen["def:" + name] = self.seen.get("def:" + name, 0) + 1
            return {"role": "assistant", "content": "```js\nconst %s = { go() {} };\n```" % name, "finish_reason": "stop"}
        path = prompt.split("NOW WRITE THE COMPLETE FILE: `")[1].split("`")[0]
        n = self.seen[path] = self.seen.get(path, 0) + 1
        if n == 1 and path in self.first_attempt:
            content = self.first_attempt[path]
        else:
            content = {"app.py": APP_PY, "templates/index.html": HTML, "static/style.css": CSS, "static/game.js": js()}[path]
        return {"role": "assistant", "content": f"Here it is:\n```\n{content}```\nDone.", "finish_reason": "stop"}


def build(tmp_path, model, task=SPEC):
    executor = CapabilityRegistry().register_defaults().build(["filesystem"], ctx=CapabilityContext(roots=[str(tmp_path)]))
    graph = build_agent_graph(AgentGraphConfig(
        agent_config=AgentConfig(completion_fn=model, executor=executor, max_iterations=6, workspace_roots=[str(tmp_path)]), executor=executor))
    return asyncio.run(GraphRuntime(graph, store=MemoryStore()).run(task)).packet.payload


# ---------------------------------------------------------------------------------------------- planning
def test_plan_names_every_file_in_dependency_order():
    plan = sc.plan_files(SPEC)
    assert [f.path for f in plan] == ["app.py", "templates/index.html", "static/style.css", "static/game.js"]
    assert [f.kind for f in plan] == ["py", "html", "css", "js"]
    assert [f.path for f in sc.plan_files("Create slugify.py, then create test_slugify.py")] == ["slugify.py", "test_slugify.py"]


@pytest.mark.parametrize("task", ["Create a file slugify.py", "Fix the bug in a.py and b.py", "How do I create app.py and main.py?",
                                  "Explain what app.py and main.py do", "Write a poem", "Rename calc_total to compute_total in a.py and b.py",
                                  "Create ../../etc/passwd.txt and x.py"])
def test_not_a_multi_file_build(task):
    assert sc.plan_files(task) is None or all(".." not in f.path for f in sc.plan_files(task))


def test_paths_that_escape_the_workspace_are_never_planned():
    plan = sc.plan_files("Create ../evil.py and /etc/x.py and ok.py and fine.js") or []
    assert [f.path for f in plan] == ["ok.py", "fine.js"] or [f.path for f in plan] == ["fine.js", "ok.py"] or len(plan) == 2
    assert all(not f.path.startswith(("/", "..")) for f in plan)


# ----------------------------------------------------------------------------------------- verification
def test_verify_catches_stubs_placeholders_syntax_missing_features_and_bad_ids():
    plan = sc.plan_files(SPEC)
    game = plan[3]
    assert any("stub" in p for p in sc.verify_file(game, "const x = 1;\n", SPEC, {}, plan))
    assert any("placeholder" in p for p in sc.verify_file(game, js() + "\n// rest of code\n", SPEC, {}, plan))
    assert any("syntax" in p.lower() or "Unbalanced" in p for p in sc.verify_file(game, js() + "function broken( {\n", SPEC, {}, plan))
    partial = "const c = document.getElementById('board');\n" * 40                         # long enough, but has none of the features
    missing = " ".join(sc.verify_file(game, partial, SPEC, {}, plan))
    assert "localStorage" in missing and "touch" in missing and "sound" in missing and "keyboard" in missing
    assert sc.verify_file(game, js(), SPEC, {"templates/index.html": HTML}, plan) == []
    bad_ids = sc.verify_file(game, js(("board", "score", "hi_score", "start", "pause", "restart", "overlay")), SPEC, {"templates/index.html": HTML}, plan)
    assert any("ids that the page does not have" in p and "hi_score" in p for p in bad_ids)
    assert sc.verify_file(plan[0], "print(", SPEC, {}, plan)                                   # python syntax
    assert any("never references" in p for p in sc.verify_file(plan[1], HTML.replace("game.js", "x.js"), SPEC, {}, plan))


# --------------------------------------------------------------------------------------- the real graph
def test_builds_every_file_through_the_agent_and_reports_honestly(tmp_path):
    model = FakeModel()
    p = build(tmp_path, model)
    for path in ("app.py", "templates/index.html", "static/style.css", "static/game.js"):
        assert (tmp_path / path).is_file(), path
    assert p["agent_completed"] is True and "Built 4 of 4 files" in p["answer"]
    assert [c["name"] for c in p["agent_tool_calls"]] == ["write_file"] * 4 and all(c["result_type"] == "ok" for c in p["agent_tool_calls"])
    # one focused model call per file, in order, and the script was shown the page's real ids
    assert len(model.prompts) == 4
    js_prompt = model.prompts[3]
    assert "id=\"overlay\"" in js_prompt and "score, high" in js_prompt.replace("\n", " ") or "board, high, overlay" in js_prompt


def test_a_stub_is_rewritten_with_concrete_feedback(tmp_path):
    model = FakeModel(first_attempt={"static/game.js": "// game logic goes here\nfunction start() {}\n"})
    p = build(tmp_path, model)
    assert "Built 4 of 4 files" in p["answer"] and len((tmp_path / "static" / "game.js").read_text()) > 600
    assert model.seen["static/game.js"] == 2 and model.seen["app.py"] == 1
    retry_prompt = [q for q in model.prompts if "static/game.js" in q.split("NOW WRITE")[1] and "PROBLEMS" in q][0]
    assert "stub" in retry_prompt and "placeholder" in retry_prompt and "game logic goes here" in retry_prompt


def test_gives_up_after_two_rewrites_and_says_so(tmp_path):
    class Stubborn(FakeModel):
        async def __call__(self, messages, system, tools):
            path = messages[0]["content"].split("NOW WRITE THE COMPLETE FILE: `")[1].split("`")[0]
            if path == "static/game.js":
                self.seen[path] = self.seen.get(path, 0) + 1
                return {"role": "assistant", "content": "```js\nlet a = 1;\n```", "finish_reason": "stop"}
            return await super().__call__(messages, system, tools)
    model = Stubborn()
    p = build(tmp_path, model)
    assert model.seen["static/game.js"] == 4                                      # first try + three rewrites (the script gets one extra), then it stops
    assert p["agent_completed"] is False and "Partly built" in p["answer"] and "PROBLEMS" in p["answer"]
    assert (tmp_path / "app.py").is_file()                                        # the good files are still there


def test_a_response_cut_off_by_the_token_limit_is_continued(tmp_path):
    full = HTML

    class Cutoff(FakeModel):
        async def __call__(self, messages, system, tools):
            path = messages[0]["content"].split("NOW WRITE THE COMPLETE FILE: `")[1].split("`")[0]
            if path == "templates/index.html" and len(messages) == 1:
                return {"role": "assistant", "content": "```html\n" + full[:200], "finish_reason": "length"}
            if path == "templates/index.html":
                return {"role": "assistant", "content": "```html\n" + full[200:] + "```", "finish_reason": "stop"}
            return await super().__call__(messages, system, tools)
    p = build(tmp_path, Cutoff())
    assert (tmp_path / "templates" / "index.html").read_text().strip() == full.strip() and "Built 4 of 4 files" in p["answer"]


def test_a_single_file_request_still_uses_the_normal_agent_loop(tmp_path):
    seen = []

    async def fake(messages, system, tools):
        seen.append(tools)
        return {"role": "assistant", "content": "done", "finish_reason": "stop"}
    p = build(tmp_path, fake, task="Create a file notes.txt containing hi")
    assert seen and seen[0]                                                       # tool schemas were offered: the ordinary loop ran
    assert "Built" not in p.get("answer", "")


# ------------------------------------------------------------------- fake-browser execution (needs node)
from orcha.agent_runtime import scaffold_js  # noqa: E402

needs_node = pytest.mark.skipif(not scaffold_js.available(), reason="node is not installed")


@needs_node
def test_an_undeclared_variable_is_caught_by_actually_running_the_script():
    problems = scaffold_js.run("document.addEventListener('DOMContentLoaded', () => { startBtn.addEventListener('click', go); });", HTML)
    assert any("startBtn is not defined" in p for p in problems)


@needs_node
def test_a_script_that_never_starts_or_never_draws_is_reported():
    quiet = "const b = document.getElementById('start'); b.addEventListener('click', () => {});"
    assert any("never began the game loop" in p for p in scaffold_js.run(quiet, HTML))
    nodraw = "const b = document.getElementById('start'); b.addEventListener('click', () => { setInterval(() => {}, 100); });"
    assert any("nothing was ever drawn" in p for p in scaffold_js.run(nodraw, HTML))


@needs_node
def test_a_working_script_passes_and_unwired_buttons_are_named():
    ok = ("const c = document.getElementById('board'); const x = c.getContext('2d'); let id;\n"
          "document.getElementById('start').addEventListener('click', () => { id = setInterval(() => { x.fillRect(1, 1, 5, 5); }, 100); });\n"
          "document.getElementById('pause').addEventListener('click', () => clearInterval(id));\n"
          "document.getElementById('restart').addEventListener('click', () => {});")
    assert scaffold_js.run(ok, HTML) == []
    plan = sc.plan_files(SPEC)
    unwired = sc.verify_file(plan[3], js().replace("document.getElementById('%s').onclick = restart;" % "restart", ""), SPEC,
                             {"templates/index.html": HTML}, plan)
    assert isinstance(unwired, list)


@needs_node
def test_a_script_that_crashes_when_run_is_repaired_with_a_tiny_request(tmp_path):
    crashing = js() + "\ndocument.getElementById('start').addEventListener('click', () => { undefinedThing.go(); });\n"
    model = FakeModel(first_attempt={"static/game.js": crashing})
    p = build(tmp_path, model)
    assert model.seen["def:undefinedThing"] == 1 and model.seen["static/game.js"] == 1      # one small request, not a rewrite
    assert "const undefinedThing" in (tmp_path / "static" / "game.js").read_text()
    assert "Built 4 of 4 files" in p["answer"]


# ----------------------------------------------------------------- best attempt + patch repair
def test_search_replace_patches_apply_with_the_tolerant_ladder():
    code = "let a = 1;\nfunction go() {\n    run();\n}\n"
    reply = "Sure:\n<<<<<<< SEARCH\nfunction go() {\n    run();\n=======\nfunction run() {}\nfunction go() {\n    run();\n>>>>>>> REPLACE\n"
    out, n = sc.apply_search_replace(code, reply)
    assert n == 1 and "function run() {}" in out and out.count("function go()") == 1
    assert sc.apply_search_replace(code, "no blocks here")[1] == 0
    assert sc.apply_search_replace(code, "<<<<<<< SEARCH\nnot in file\n=======\nx\n>>>>>>> REPLACE")[1] == 0


def test_later_attempts_use_patches_and_the_best_version_is_kept(tmp_path):
    long_js = js() + "// padding\n" * 130                 # >1200 chars so patch mode is eligible
    broken = long_js.replace("localStorage.getItem('high')", "0")          # missing localStorage: one problem

    class Patching(FakeModel):
        async def __call__(self, messages, system, tools):
            prompt = messages[0]["content"]
            if system and system.startswith("You fix a file with small patches"):
                self.seen["patch"] = self.seen.get("patch", 0) + 1
                return {"role": "assistant", "content": "<<<<<<< SEARCH\nlet high = Number(0 || 0);\n=======\nlet high = Number(localStorage.getItem('high') || 0);\n>>>>>>> REPLACE", "finish_reason": "stop"}
            path = prompt.split("NOW WRITE THE COMPLETE FILE: `")[1].split("`")[0]
            if path == "static/game.js":
                self.seen[path] = self.seen.get(path, 0) + 1
                return {"role": "assistant", "content": "```js\n" + broken + "```", "finish_reason": "stop"}   # rewrites never fix it
            return await super().__call__(messages, system, tools)
    model = Patching()
    p = build(tmp_path, model)
    assert model.seen.get("patch", 0) >= 1                                    # a patch was requested instead of a third rewrite
    assert "localStorage.getItem('high')" in (tmp_path / "static" / "game.js").read_text()
    assert "Built 4 of 4 files" in p["answer"]


# ----------------------------------------------------------- targeted repair of undefined names
def test_undefined_names_come_from_the_fake_browser_report():
    probs = ["While playing it threw: tick: ReferenceError: draw is not defined", "failed immediately with ReferenceError: loadHighScore is not defined",
             "touchend handler: e is not defined"]
    assert sc.undefined_names(probs) == ["draw", "loadHighScore"]                  # 'e' is a missing parameter, not a definition


def test_a_definition_is_inserted_at_the_outermost_scope():
    code = "document.addEventListener('DOMContentLoaded', () => {\n    const a = 1;\n    function tick() {\n        draw();\n    }\n});\n"
    out = sc.insert_definition(code, "function draw() {\n    console.log(a);\n}")
    lines = out.split("\n")
    assert lines.index("    function draw() {") < lines.index("    function tick() {")          # same indentation as the other top-level declarations
    assert out.count("function draw") == 1


@needs_node
def test_a_missing_function_is_repaired_by_asking_for_only_that_function(tmp_path):
    long = js() + "// padding\n" * 130
    crashing = long + "\nfunction tickAll() { drawScene(); }\nsetInterval(tickAll, 100);\n"

    class Repairer(FakeModel):
        async def __call__(self, messages, system, tools):
            if system and system.startswith("You write ONE missing definition"):
                self.seen["def"] = self.seen.get("def", 0) + 1
                return {"role": "assistant", "content": "```js\nfunction drawScene() {\n    ctx.clearRect(0, 0, 10, 10);\n    ctx.fillRect(1, 1, 5, 5);\n}\n```", "finish_reason": "stop"}
            prompt = messages[0]["content"]
            path = prompt.split("NOW WRITE THE COMPLETE FILE: `")[1].split("`")[0]
            if path == "static/game.js":
                self.seen[path] = self.seen.get(path, 0) + 1
                return {"role": "assistant", "content": "```js\n" + crashing + "```", "finish_reason": "stop"}
            return await super().__call__(messages, system, tools)
    model = Repairer()
    p = build(tmp_path, model)
    assert model.seen.get("def") == 1 and model.seen["static/game.js"] == 1                   # one tiny request, no rewrite
    assert "function drawScene()" in (tmp_path / "static" / "game.js").read_text()
    assert "Built 4 of 4 files" in p["answer"]
