"""Application integration: how Anvira Notes, Anvira Study and Anvira Dev use one shared runtime.

Each scenario goes: detect -> connect -> model -> request -> ORCHA / Nomi / context -> result, using only the
public SDK (no ORCHA, Nomi or AICL knowledge).
"""
import json

import pytest

from anvira_client import AnviraError, AnviraRuntime, RuntimeNotInstalled
from anvira_client.discovery import probe
from conftest import wait_for

pytestmark = pytest.mark.slow

NOTEBOOK = ("Photosynthesis converts light energy into chemical energy inside chloroplasts.\n\n"
            "The Calvin cycle fixes carbon dioxide into sugars using ATP and NADPH.\n\n"
            "Cellular respiration in mitochondria releases energy as ATP.")


@pytest.fixture(scope="module", autouse=True)
def _model(runtime, provider_model):
    runtime.owner.json("POST", "/v1/models/select", {"id": provider_model})


def connect_like_an_app(runtime, app_id, **kw):
    """What an app does at launch. The install callback must never fire for an existing shared runtime."""
    asked = []
    app = AnviraRuntime.connect(app_id, env=runtime.env, install=lambda info: asked.append(info) or False, **kw)
    assert asked == [], "an installed, running runtime must not trigger an install prompt"
    return app


def test_anvira_notes_flow(runtime, fake, provider_model):
    info = AnviraRuntime.detect(runtime.env)
    assert info.installed and info.running and info.api_version == 1                      # 1. runtime detection
    notes = connect_like_an_app(runtime, "anvira-notes", name="Anvira Notes", min_version="1.0.0")   # 2. connect (already running)
    assert notes.has_capability("memory.search") and notes.health()["status"] == "ok"
    models = notes.models.installed()                                                      # 3. model discovery + selection
    assert any(m["id"] == provider_model for m in models)                                   # (other tests add local models too)
    assert notes.models.use(provider_model)["state"] in ("remote", "running")

    notes.context.put("biology", "src1", NOTEBOOK, title="Biology notes")                  # 4. index the notebook sources
    question = "How is carbon dioxide fixed?"
    hits = notes.context.search("biology", question, limit=2)                               #    retrieve grounding chunks
    assert hits and "Calvin" in hits[0]["text"]
    grounded = [{"role": "system", "content": "Answer only from these notes:\n" + "\n---\n".join(h["text"] for h in hits)},
                {"role": "user", "content": question}]
    answer = notes.chat_text(grounded)                                                     # 5. request
    assert answer.startswith("fake reply") and "Calvin cycle" in json.dumps(fake.requests[-1]["messages"])

    memo = notes.memory.store("User keeps a biology notebook", title="Interest", tags=["profile"])   # 6. Nomi through the runtime
    assert notes.memory.search("biology notebook")[0]["id"] == memo["id"]
    job = notes.orcha.run("Summarise my biology notes in one paragraph.", wait=True)        # 7. ORCHA
    assert job.state == "completed" and job.unwrap()["answer"]
    assert notes.jobs.get(job.id).unwrap()["run_id"] == job.result["run_id"]               # 8. result


def test_anvira_study_flow(runtime, fake):
    study = connect_like_an_app(runtime, "anvira-study", name="Anvira Study")
    study.context.put("course-1", "syllabus", "Grading: homework 20%, midterm 30%, final 50%.", title="Syllabus")
    study.context.put("course-1", "lecture-3", NOTEBOOK, title="Lecture 3")
    grounded = study.context.search("course-1", "how is the grade calculated final exam weight", limit=1)
    assert grounded[0]["doc_id"] == "syllabus"                                             # grounded material retrieval
    cards = study.chat_text([{"role": "system", "content": "Generate flashcards as JSON."},
                             {"role": "user", "content": grounded[0]["text"]}])
    assert isinstance(cards, str) and cards
    study.memory.store("Exam on Friday; weak on the Calvin cycle", title="Study state", workspace="course-1")
    plan_job = study.task("Make a 3-day study plan for the Calvin cycle.")
    assert plan_job.state == "completed"
    # Study's data stays separate from Notes': same collection name, different app namespace
    notes = connect_like_an_app(runtime, "anvira-notes")
    assert notes.context.search("course-1", "grading") == []
    assert notes.memory.search("Calvin cycle", workspace="course-1") == []
    assert [m["title"] for m in study.memory.search("Calvin", workspace="course-1")] == ["Study state"]


def test_anvira_dev_flow_uses_orcha_heavily(runtime, fake, tmp_path):
    dev = connect_like_an_app(runtime, "anvira-dev", name="Anvira Dev")
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("print('hello')\n")
    job = dev.agent_run("Explain what main.py does.", workspace_roots=[str(project)])       # workspace-scoped agent job
    assert job.kind == "agent.run" and job.id.startswith("job_")
    job.wait(120)
    assert job.state in ("completed", "failed") and job.data["meta"].get("orcha_run_id")   # it reached ORCHA's agent path
    assert any(j["id"] == job.id for j in dev.orcha.jobs())
    frames = []
    try:
        for ev in job.events():                                                            # ORCHA event frames pass straight through
            frames.append(ev)
            if len(frames) > 500:
                break
    except AnviraError:
        pass                                                                               # stream may already be closed for finished runs
    graph_job = dev.orcha.run("Research: what is a B-tree?", graph="research", wait=True)
    assert graph_job.state in ("completed", "failed") and graph_job.data["meta"]["graph"] == "research"
    fake.mode = "slow"
    fake.delay = 20
    slow = dev.orcha.run("Cancel me")
    wait_for(lambda: dev.jobs.get(slow.id).data["meta"].get("orcha_run_id"), 20, message="run started")
    assert dev.jobs.get(slow.id).cancel().state == "cancelled"


def test_all_three_apps_share_one_runtime_and_one_model(runtime):
    apps = [connect_like_an_app(runtime, a) for a in ("anvira-notes", "anvira-study", "anvira-dev")]
    pids = {probe(runtime.env).pid} | {a.info.pid for a in apps}
    assert len(pids) == 1                                                                  # a single runtime process
    assert len({a.models.active()["active"] for a in apps}) == 1                           # a single active model
    assert len({tuple(m["id"] for m in a.models.installed()) for a in apps}) == 1          # a single model library
    listed = runtime.owner.json("GET", "/v1/apps")["apps"]
    assert {"anvira-notes", "anvira-study", "anvira-dev"} <= {x["app_id"] for x in listed}
    home_files = {p.name for p in (runtime.home).iterdir()}
    assert not home_files & {"notes", "study", "workspace"}                                # runtime state != application data


def test_second_registration_reuses_the_token_not_a_new_install(runtime):
    a1 = runtime.connect("reuse-app")
    a2 = runtime.connect("reuse-app")
    assert a1._http.token == a2._http.token                                                # idempotent connect
    assert len([x for x in runtime.owner.json("GET", "/v1/apps")["apps"] if x["app_id"] == "reuse-app"]) == 1


def test_missing_runtime_never_installs_silently(tmp_path):
    env = {"ANVIRA_RUNTIME_HOME": str(tmp_path / "empty")}
    with pytest.raises(RuntimeNotInstalled) as e:
        AnviraRuntime.connect("anvira-notes", env=env)                                     # no install callback -> refuse
    assert "required" in str(e.value) and e.value.code == "runtime_not_installed"
    assert not (tmp_path / "empty" / "install.json").exists()
    declined = []
    with pytest.raises(AnviraError) as e:
        AnviraRuntime.connect("anvira-notes", env=env, install=lambda info: declined.append(1) or False)   # user says Cancel
    assert e.value.code == "install_declined" and declined == [1]
    assert not (tmp_path / "empty").exists() or not any((tmp_path / "empty").iterdir())    # nothing was created
