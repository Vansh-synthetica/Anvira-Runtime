"""Real end-to-end workflows against a REAL model - not a test double.

    python examples/real_workflows.py --models "D:\\AI\\models" --model qwen2.5-coder-3b [--only notes,study,dev] [--home <runtime folder>]

What it does (each step is what an Anvira app would do through the SDK):

  Notes  - index a notebook, register it as a resource, ask grounded questions (answers must come from the notes and
           cite them; a question the notes cannot answer must NOT be answered from thin air), keep memory across sessions.
  Study  - the user shares the notebook with Study; Study builds flashcards from it. A different app that was not
           granted access must see nothing.
  Dev    - agentic work with a small model in a real workspace (create code + tests, fix a bug, rename across files);
           every result is verified by RUNNING the code, not by trusting the model.

It prints what happened and exits non-zero if a check fails. Nothing is downloaded; the model must already exist.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "sdk" / "python", REPO / "runtime"):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
if (REPO / "Orcha").is_dir():
    sys.path.insert(0, str(REPO / "Orcha"))                       # lets the Snake check reuse ORCHA's fake-browser harness

from anvira_client import AnviraError, AnviraRuntime  # noqa: E402

NOTEBOOK = {
    "photosynthesis": """# Photosynthesis
Photosynthesis converts light energy into chemical energy. It happens in chloroplasts, organelles found in plant cells.

The light-dependent reactions take place in the thylakoid membranes. They split water, release oxygen, and make ATP and NADPH.

The Calvin cycle takes place in the stroma. It uses ATP and NADPH to fix carbon dioxide into sugars. The key enzyme that fixes carbon dioxide is RuBisCO.""",
    "respiration": """# Cellular respiration
Cellular respiration releases energy from glucose. It happens mostly in the mitochondria.

Glycolysis happens in the cytoplasm and splits one glucose into two pyruvate, making a net gain of 2 ATP.

The Krebs cycle happens in the mitochondrial matrix. The electron transport chain on the inner membrane makes most of the ATP - about 30 to 32 ATP per glucose overall.""",
    "genetics": """# Genetics
DNA is a double helix made of nucleotides. Adenine pairs with thymine and guanine pairs with cytosine.

Mendel studied pea plants. A dominant allele shows in the phenotype even when only one copy is present. A recessive allele shows only when two copies are present.""",
}

results: list[tuple[str, bool, str]] = []
RUN_COLLECTION = "biology-101-" + time.strftime("%H%M%S")      # unique per run: a persistent runtime keeps earlier runs' data


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""), flush=True)
    return bool(ok)


def head(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(3, 70 - len(title)), flush=True)


def ask(app: AnviraRuntime, question: str, **opts) -> tuple[str, dict]:
    t0 = time.time()
    out = app.chat([{"role": "system", "content": "Answer ONLY from the provided context. If the context does not contain the answer, "
                                                    "say exactly: I could not find that in your notes. Cite the source title in brackets."},
                    {"role": "user", "content": question}], temperature=0.1, max_tokens=300, **opts)
    text = out["choices"][0]["message"]["content"].strip()
    print(f"    Q: {question}\n    A: {text[:400]}   ({time.time() - t0:.1f}s)", flush=True)
    return text, out.get("runtime", {})


def connect(app_id: str, name: str, env=None, permissions=None) -> AnviraRuntime:
    return AnviraRuntime.connect(app_id, name=name, env=env, permissions=permissions, on_status=lambda m: print(f"    [{app_id}] {m}"),
                                 install=lambda info: (_ for _ in ()).throw(SystemExit("Anvira Runtime is not installed: run `anvira runtime install`.")))


# ------------------------------------------------------------------------------------------------- Notes
def notes_workflow(notes: AnviraRuntime) -> str:
    head("Anvira Notes: grounded answers from a notebook")
    for doc_id, text in NOTEBOOK.items():
        notes.context.put(RUN_COLLECTION, doc_id, text, title=doc_id.capitalize())
    res = notes.resources.create("Biology 101 notebook", type="notebook", collection=RUN_COLLECTION, workspace="school")
    check("notebook indexed and registered as a private resource", res["visibility"] == "private" and res["ref"].startswith("runtime://"), res["ref"])

    # the notes app grounds a question itself (retrieval by the runtime's BM25 index), then asks the model
    hits = notes.context.search(RUN_COLLECTION, "Which enzyme fixes carbon dioxide in the Calvin cycle?", limit=2)
    check("retrieval finds the right page", hits and hits[0]["doc_id"] == "photosynthesis", hits[0]["doc_id"] if hits else "none")

    q1 = "Which enzyme fixes carbon dioxide in the Calvin cycle, and where does that cycle happen?"
    a1, meta = ask(notes, q1, context={"resources": [res["id"]]})
    check("answer uses the notebook (RuBisCO, stroma)", "rubisco" in a1.lower() and "stroma" in a1.lower(), f"context_used={meta.get('context_used')}")
    check("only the notebook resource was sent as context", meta.get("context_used") == [res["id"]])

    a2, _ = ask(notes, "How much ATP does one glucose yield overall in cellular respiration?", context={"resources": [res["id"]]})
    check("second question answered from the notebook (30-32 ATP)", "30" in a2 or "32" in a2, a2[:60])

    a3, _ = ask(notes, "Who won the 1998 football World Cup?", context={"resources": [res["id"]], "strict": True})
    refused = "could not find" in a3.lower() or "not find" in a3.lower() or "not in" in a3.lower() or "no information" in a3.lower()
    check("a question the notes cannot answer is NOT invented", refused, a3[:80])

    m = notes.memory.store("The student is preparing for a biology midterm on Friday and struggles with the Krebs cycle.",
                           title="Study goal", tags=["profile"])
    time.sleep(0.5)
    found = notes.memory.search("what is the student preparing for")
    check("memory stored and recalled by the runtime (Nomi)", any(x["id"] == m["id"] for x in found))
    # Nomi recall is word-based (BM25): the question shares words with the memory ("biology", "midterm"). It does not do semantic matching.
    a4, meta = ask(notes, "What should I revise for my biology midterm?", memory={"recall": True}, context={"resources": [res["id"]]})
    check("memory recall reaches the model", bool(meta.get("memories_used")), f"memories_used={len(meta.get('memories_used') or [])}")
    return res["id"]


# ------------------------------------------------------------------------------------------------- Study
def study_workflow(study: AnviraRuntime, other: AnviraRuntime, notes: AnviraRuntime, rid: str) -> None:
    head("Anvira Study: shared notebook, flashcards, and a stranger who sees nothing")
    check("Study cannot see the notebook before it is shared", all(r["id"] != rid for r in study.resources.list()))
    check("an unrelated app cannot find it by search", other.resources.search("Calvin cycle RuBisCO")["count"] == 0)
    req = study.resources.request_access(rid, reason="build flashcards for the biology unit")
    print(f"    Study asked for access: {req['id']} ({req['state']})  <- the user/Notes decides")
    notes.resources.decide(req["id"], True)                     # what `anvira context approve` does
    check("after approval Study can read it", "RuBisCO" in study.resources.read(rid, "photosynthesis")["text"])
    check("the unrelated app still sees nothing", other.resources.list() == [])
    notes.resources.revoke(rid)                                  # leave the runtime as we found it

    prompt = ("Create 3 flashcards from the context as a JSON array of objects with keys front and back. "
              "Output only the JSON array, no other text.")
    out = study.chat([{"role": "user", "content": prompt}], context={"query": "photosynthesis respiration flashcards", "resources": [rid]},
                     temperature=0.1, max_tokens=500)
    text = out["choices"][0]["message"]["content"]
    print("    " + text.replace("\n", "\n    ")[:500])
    cards = None
    try:
        start, end = text.index("["), text.rindex("]") + 1
        cards = json.loads(text[start:end])
    except ValueError:
        pass
    check("flashcards came back as valid JSON", isinstance(cards, list) and len(cards) >= 2 and all("front" in c and "back" in c for c in cards),
          f"{len(cards) if isinstance(cards, list) else 0} cards")
    audit = [e["action"] for e in notes.resources.audit(50, rid)]
    check("the owner's audit log shows the request, approval and Study's read", {"request", "approve", "read"} <= set(audit), ",".join(sorted(set(audit))))


# --------------------------------------------------------------------------------------------------- Dev
def run_tests(cwd: Path, module: str = "") -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-m", "unittest", *([module] if module else ["discover", "-v"])], cwd=str(cwd), capture_output=True,
                       text=True, timeout=120)
    return r.returncode == 0, (r.stdout + r.stderr)[-600:]


def agent(dev: AnviraRuntime, task: str, root: Path, timeout: float = 900):
    t0 = time.time()
    job = dev.agent_run(task, workspace_roots=[str(root)], wait=False)
    job.wait(timeout)
    res = job.data.get("result") or {}
    calls = res.get("agent_tool_calls") or []
    names = [c.get("tool") or c.get("name") or "?" for c in calls if isinstance(c, dict)]
    print(f"    job {job.id}: {job.state} in {time.time() - t0:.0f}s, {len(calls)} tool calls: {', '.join(names[:14])}", flush=True)
    if job.state != "completed":
        print(f"    error: {job.data.get('error')}")
    return job, names


def user_grants(env: dict, app_id: str, permission: str, revoke: bool = False) -> None:
    """What `anvira app grant <app> <permission>` does - the USER's decision, made with the owner token."""
    from anvira_client.discovery import probe, read_token
    from anvira_client.http import Http
    info = probe(env)
    Http(info.base_url, read_token("owner", env), 30).json("POST", f"/v1/apps/{app_id}/{'deny' if revoke else 'grant'}", {"permissions": [permission]})
    print(f"    (the user ran: anvira app {'deny' if revoke else 'grant'} {app_id} {permission})")


def dev_workflow(dev: AnviraRuntime, base: Path, env: dict) -> None:
    head("Anvira Dev: agentic work in a real workspace")
    user_grants(env, "anvira-dev", "orcha.exec", revoke=True)      # start from the default: apps may not run commands
    try:
        dev.agent_run("Say hi", workspace_roots=[str(base)], capabilities=["terminal"], wait=False)
        check("(security) running commands is refused until the user grants orcha.exec", False)
    except AnviraError as exc:
        check("(security) running commands is refused until the user grants orcha.exec", exc.code == "permission_denied", exc.code)
    user_grants(env, "anvira-dev", "orcha.exec")
    # 1) create code + tests from a spec
    w1 = base / "create"
    w1.mkdir(parents=True)
    print("  Task 1: create a module and its tests from a spec")
    agent(dev, "Create a file slugify.py with a function slugify(text) that lowercases the text, replaces every run of characters that are "
               "not letters or digits with a single hyphen, and strips hyphens from both ends. Then create test_slugify.py using unittest "
               "with three tests, including that slugify('Hello, World!') == 'hello-world'.", w1)
    ok = (w1 / "slugify.py").is_file()
    passed = False
    if ok:
        sys.path.insert(0, str(w1))
        try:
            ns: dict = {}
            exec(compile((w1 / "slugify.py").read_text(encoding="utf-8"), "slugify.py", "exec"), ns)
            passed = ns["slugify"]("Hello, World!") == "hello-world" and ns["slugify"]("  --A  b--  ") == "a-b"
        except Exception as exc:  # noqa: BLE001
            print(f"    slugify.py failed: {exc}")
        finally:
            sys.path.remove(str(w1))
    check("slugify.py exists and behaves correctly", ok and passed)
    if (w1 / "test_slugify.py").is_file():
        good, tail = run_tests(w1, "test_slugify")
        check("its own unit tests pass when run", good, "" if good else tail[-200:])
    else:
        check("test_slugify.py was created", False)

    # 2) fix a bug found by a failing test
    w2 = base / "bugfix"
    w2.mkdir()
    (w2 / "stats.py").write_text(textwrap.dedent('''\
        def mean(values):
            """Arithmetic mean of a non-empty list."""
            return sum(values) / (len(values) + 1)


        def median(values):
            ordered = sorted(values)
            mid = len(ordered) // 2
            if len(ordered) % 2:
                return ordered[mid]
            return (ordered[mid - 1] + ordered[mid]) / 2
        '''), encoding="utf-8")
    (w2 / "test_stats.py").write_text(textwrap.dedent('''\
        import unittest
        from stats import mean, median


        class T(unittest.TestCase):
            def test_mean(self):
                self.assertEqual(mean([2, 4, 6]), 4)

            def test_median(self):
                self.assertEqual(median([3, 1, 2]), 2)
        '''), encoding="utf-8")
    before, _ = run_tests(w2)
    check("(setup) the test fails before the agent runs", not before)
    print("  Task 2: find and fix a bug")
    agent(dev, "The unit tests in test_stats.py are failing. Read stats.py, find the bug, and fix it in stats.py. Do not change the tests.", w2)
    after, tail = run_tests(w2)
    check("the agent fixed the bug (tests pass)", after, "" if after else tail[-200:])
    check("the agent did not tamper with the tests", "assertEqual(mean([2, 4, 6]), 4)" in (w2 / "test_stats.py").read_text(encoding="utf-8"))

    # 3) multi-file rename
    w3 = base / "rename"
    w3.mkdir()
    (w3 / "billing.py").write_text("def calc_total(items):\n    return sum(i['price'] * i['qty'] for i in items)\n", encoding="utf-8")
    (w3 / "report.py").write_text("from billing import calc_total\n\n\ndef summary(items):\n    return f'Total: {calc_total(items)}'\n", encoding="utf-8")
    (w3 / "test_billing.py").write_text("import unittest\nfrom billing import calc_total\nfrom report import summary\n\n\nclass T(unittest.TestCase):\n"
                                        "    def test_it(self):\n        items = [{'price': 2, 'qty': 3}]\n"
                                        "        self.assertEqual(calc_total(items), 6)\n        self.assertEqual(summary(items), 'Total: 6')\n", encoding="utf-8")
    print("  Task 3: rename a function across three files")
    agent(dev, "Rename the function calc_total to compute_total everywhere in this project (billing.py, report.py and test_billing.py), "
               "so the tests still pass.", w3)
    text = "".join((w3 / f).read_text(encoding="utf-8") for f in ("billing.py", "report.py", "test_billing.py"))
    good, tail = run_tests(w3)
    check("no calc_total left and the tests still pass", "calc_total" not in text and "compute_total" in text and good, "" if good else tail[-200:])


SNAKE_TASK = """Create a complete, polished Snake Game using HTML, CSS, JavaScript, and Python.
Requirements: arrow keys or WASD; the snake moves continuously; food appears at random positions; eating food grows the snake, raises the score and spawns new food;
the game ends when the snake hits a wall or itself; Start Game, Pause and Restart buttons; show the score, the high score and a game-over message; the game speeds up as the score rises;
responsive on desktop and mobile with touch/swipe controls; a modern dark UI with neon snake and food, hover effects and animations; sound effects for eating, game over and starting;
the high score is stored in localStorage.
Use Flask. Create app.py, templates/index.html, static/style.css and static/game.js. The Flask server serves the game from the / route.
All gameplay logic goes in game.js. Use an HTML canvas. Add a title, score and high-score display, the canvas, control buttons, a game-over overlay and instructions.
Do not omit any code or use placeholders. The final code must work together."""


def snake_workflow(dev: AnviraRuntime, base: Path) -> None:
    head("Anvira Dev: a multi-file project from one prompt (Snake: Flask + HTML + CSS + JS)")
    w = base / "snake"
    w.mkdir(parents=True)
    t0 = time.time()
    job = dev.agent_run(SNAKE_TASK, workspace_roots=[str(w)])
    job.wait(1500)
    answer = (job.data.get("result") or {}).get("answer", "")
    print(f"    job {job.id}: {job.state} in {time.time() - t0:.0f}s")
    for line in answer.splitlines():
        print("    " + line)
    files = {"app.py": w / "app.py", "index.html": w / "templates" / "index.html", "style.css": w / "static" / "style.css", "game.js": w / "static" / "game.js"}
    for name, path in files.items():
        check(f"{name} was created and is not a stub", path.is_file() and path.stat().st_size > {"app.py": 80, "index.html": 250, "style.css": 250, "game.js": 1500}[name], f"{path.stat().st_size if path.is_file() else 0} bytes")
    if files["app.py"].is_file():
        r = subprocess.run([sys.executable, "-m", "py_compile", str(files["app.py"])], capture_output=True, text=True)
        check("app.py compiles and defines the / route", r.returncode == 0 and "@app.route('/')" in files["app.py"].read_text(encoding="utf-8").replace('"', "'"))
    if files["game.js"].is_file() and files["index.html"].is_file():
        try:
            from orcha.agent_runtime import scaffold_js
        except ImportError:
            scaffold_js = None
        if scaffold_js and scaffold_js.available():
            problems = scaffold_js.run(files["game.js"].read_text(encoding="utf-8"), files["index.html"].read_text(encoding="utf-8"))
            check("game.js runs in a fake browser: Start begins a loop, it draws, keys/touch/buttons do not crash", not problems, "; ".join(problems)[:160])
        else:
            print("    (node or ORCHA sources not available: skipped the fake-browser check)")
    js = files["game.js"].read_text(encoding="utf-8") if files["game.js"].is_file() else ""
    for label, pat in (("localStorage high score", "localStorage"), ("touch/swipe", "touchstart|touchend"), ("sound", "AudioContext|new Audio"), ("WASD", "'w'|\"w\"|KeyW")):
        import re as _re
        check(f"game.js has {label}", bool(_re.search(pat, js)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="folder with .gguf files to register (nothing is copied or downloaded)")
    ap.add_argument("--model", help="model id, or part of it (default: the active model)")
    ap.add_argument("--only", default="notes,study,dev")
    ap.add_argument("--home", help="use the runtime in this folder (sets ANVIRA_RUNTIME_HOME)")
    ap.add_argument("--keep", action="store_true", help="keep the temporary workspaces")
    a = ap.parse_args()
    env = {**os.environ, **({"ANVIRA_RUNTIME_HOME": a.home} if a.home else {})}
    only = set(a.only.split(","))

    head("Connecting like an app")
    notes = connect("anvira-notes", "Anvira Notes", env)
    print(f"    runtime {notes.info.runtime_version} at {notes.info.home}")
    if a.models:
        reg = notes.models.register(a.models)
        print(f"    registered {a.models}: {len(reg.get('models') or [reg.get('model')])} model file(s) (none copied)")
    installed = notes.models.installed()
    if a.model:
        pick = next((m for m in installed if a.model.lower() in m["id"].lower()), None)
        if not pick:
            print(f"    no installed model matches '{a.model}'. Installed: {[m['id'] for m in installed]}")
            return 2
        t0 = time.time()
        st = notes.models.use(pick["id"])
        print(f"    model {pick['id']} -> {st['state']} in {time.time() - t0:.0f}s   {st.get('backend_info') or ''}")
    elif not notes.models.active().get("active"):
        print("    no active model; pass --model")
        return 2
    print(f"    active model: {notes.models.active()['active']}")

    study = connect("anvira-study", "Anvira Study", env)
    stranger = connect("anvira-dev", "Anvira Dev", env, permissions=["orcha.exec"])
    rid = None
    try:
        if "notes" in only or "study" in only:
            rid = notes_workflow(notes)
        if "study" in only and rid:
            study_workflow(study, stranger, notes, rid)
        if "dev" in only:
            base = Path(tempfile.mkdtemp(prefix="anvira-dev-"))
            dev_workflow(stranger, base, env)
            if "snake" in only or "dev" in only:
                snake_workflow(stranger, base)
            if not a.keep:
                import shutil
                shutil.rmtree(base, ignore_errors=True)
    except AnviraError as exc:
        print(f"\n!! runtime error: {exc}")
        results.append(("workflow aborted", False, str(exc)))
    finally:
        for app in (notes, study, stranger):
            app.close()

    head("Result")
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
