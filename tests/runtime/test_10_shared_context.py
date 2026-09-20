"""Shared data layer: resources are private by default, shared explicitly, referenced (never copied), audited.

Covers the rules apps must never bend: no silent cross-app access, global is the user's decision only, and local
data never goes to a remote model unless the user opts in.
"""
import json

import pytest

from anvira_client import AnviraError
from conftest import wait_for  # noqa: F401

pytestmark = pytest.mark.slow

NOTES = ("The Calvin cycle fixes carbon dioxide into sugars using ATP and NADPH.\n\n"
         "Mitochondria release energy as ATP during cellular respiration.")


@pytest.fixture(scope="module", autouse=True)
def _model(runtime, provider_model):
    runtime.owner.json("POST", "/v1/models/select", {"id": provider_model})


def raises(code, fn, *a, **kw):
    with pytest.raises(AnviraError) as ei:
        fn(*a, **kw)
    assert ei.value.code == code, f"expected {code}, got {ei.value.code}: {ei.value.message}"
    return ei.value


@pytest.fixture()
def apps(runtime):
    return runtime.connect("anvira-notes", keep_alive=False), runtime.connect("anvira-study", keep_alive=False), \
        runtime.connect("anvira-dev", keep_alive=False)


def test_private_by_default_then_shared_read_then_revoked(runtime, apps):
    notes, study, dev = apps
    notes.context.put("bio", "s1", NOTES, title="Biology")
    res = notes.resources.create("Biology notebook", type="notebook", collection="bio")
    assert res["visibility"] == "private" and res["owner"] == "anvira-notes" and res["ref"].startswith("runtime://res_")
    assert res["access"] == "admin"

    # another app cannot see it, resolve it, read it, or find it by search: not even its existence
    assert study.resources.list() == []
    raises("resource_not_found", study.resources.get, res["id"])
    raises("resource_not_found", study.resources.resolve, res["ref"])
    raises("resource_not_found", study.resources.read, res["id"])
    assert study.resources.search("Calvin cycle carbon")["count"] == 0

    shared = notes.resources.share(res["id"], ["anvira-study"])
    assert shared["visibility"] == "shared" and shared["grants"] == [{"app": "anvira-study", "access": "read"}]
    seen = study.resources.get(res["id"])
    assert seen["access"] == "read" and "grants" not in seen and "content" not in seen   # grantees do not see internals
    assert study.resources.resolve(res["ref"])["id"] == res["id"]
    hit = study.resources.search("How is carbon dioxide fixed Calvin")["items"][0]
    assert hit["resource"] == res["id"] and "Calvin" in hit["text"] and hit["owner"] == "anvira-notes"
    assert "Calvin" in study.resources.read(res["id"], "s1")["text"]
    assert dev.resources.list() == []                                                    # a third app still sees nothing

    raises("access_denied", study.resources.write, res["id"], "s2", "study was here")   # read is not write
    raises("access_denied", study.resources.share, res["id"], ["anvira-dev"])         # only the owner shares onward

    notes.resources.revoke(res["id"])
    raises("resource_not_found", study.resources.get, res["id"])
    assert study.resources.search("Calvin")["count"] == 0


def test_write_access_updates_the_owners_collection_not_a_copy(runtime, apps):
    notes, study, _ = apps
    notes.context.put("shared-notes", "a", "Original page about enzymes.", title="A")
    res = notes.resources.create("Team notes", type="notebook", collection="shared-notes")
    notes.resources.share(res["id"], ["anvira-study"], access="write")
    out = study.resources.write(res["id"], "b", "Study added a page on ribosomes and translation.", title="B")
    assert out["by"] == "anvira-study"
    # the owner sees the new document in ITS OWN collection: one copy of the data, referenced
    assert {d["doc_id"] for d in notes.context.documents("shared-notes")} == {"a", "b"}
    assert "ribosomes" in notes.context.search("shared-notes", "ribosomes translation")[0]["text"]
    # the writer's own index has nothing: nothing was duplicated into Study's namespace
    assert "shared-notes" not in {c["collection"] for c in study.context.collections()}
    raises("access_denied", study.resources.update, res["id"], title="renamed by a grantee")   # owner-only metadata
    raises("access_denied", study.resources.delete, res["id"])


def test_global_is_only_the_users_decision(runtime, apps):
    notes, study, dev = apps
    res = notes.resources.create("Style guide", type="guide", text="Always answer in British English, briefly.")
    e = raises("user_approval_required", notes.resources.share, res["id"], ["*"])
    assert "anvira context share" in (e.hint or "")
    assert study.resources.list(type="guide") == []
    # the user (owner token) decides
    r = runtime.owner.json("POST", f"/v1/resources/{res['id']}/share", {"global": True})
    assert r["visibility"] == "global"
    assert study.resources.get(res["id"])["access"] == "read" and dev.resources.get(res["id"])["visibility"] == "global"
    raises("access_denied", dev.resources.write, res["id"], "x", "nope")                # global is always read-only
    assert "British English" in dev.resources.read(res["id"])["text"]
    notes.resources.revoke(res["id"], ["*"])                                          # the owner may always narrow access
    raises("resource_not_found", study.resources.get, res["id"])


def test_access_requests_are_decided_by_the_owner_or_user(runtime, apps):
    notes, study, dev = apps
    res = notes.resources.create("Lecture recap", type="recap", text="Week 4: Krebs cycle and the electron transport chain.")
    raises("resource_not_found", study.resources.read, res["id"])
    req = study.resources.request_access(res["id"], access="read", reason="Building flashcards")
    assert req["state"] == "pending" and req["requester"] == "anvira-study"
    raises("request_pending", study.resources.request_access, res["id"])
    assert [q["id"] for q in notes.resources.requests()] == [req["id"]]               # the owner sees it
    assert dev.resources.requests() == []                                               # unrelated apps do not
    raises("resource_not_found", study.resources.read, res["id"])                       # requesting grants nothing
    raises("request_not_found", dev.resources.decide, req["id"], True)                  # a third app cannot approve
    raises("request_not_found", study.resources.decide, req["id"], True)                # nor can the requester
    assert notes.resources.decide(req["id"], True)["state"] == "approved"
    assert "Krebs" in study.resources.read(res["id"])["text"]
    raises("request_decided", notes.resources.decide, req["id"], False)

    other = notes.resources.create("Private draft", type="recap", text="Not for others.")
    q2 = dev.resources.request_access(other["id"])
    r = runtime.owner.json("POST", f"/v1/resources/requests/{q2['id']}/deny")           # the user can decide too
    assert r["state"] == "denied"
    raises("resource_not_found", dev.resources.get, other["id"])
    assert any(q["id"] == q2["id"] for q in runtime.owner.json("GET", "/v1/resources/requests", params={"state": "all"})["requests"])


def test_workspaces_group_and_share_in_bulk(runtime, apps):
    notes, study, dev = apps
    notes.resources.create_workspace("thesis")
    a = notes.resources.create("Chapter 1", type="draft", text="Introduction to enzyme kinetics.", workspace="thesis")
    b = notes.resources.create("Chapter 2", type="draft", text="Michaelis-Menten model derivation.", workspace="thesis")
    notes.resources.create("Diary", type="note", text="Personal.", workspace="elsewhere")
    assert {w["name"] for w in notes.resources.workspaces()} >= {"thesis", "elsewhere"}
    assert study.resources.workspaces() == []                                           # workspaces reveal nothing by themselves
    out = notes.resources.share_workspace("thesis", ["anvira-study"])
    assert set(out["shared"]) == {a["id"], b["id"]}
    assert {r["title"] for r in study.resources.list(workspace="thesis")} == {"Chapter 1", "Chapter 2"}
    assert study.resources.list(workspace="elsewhere") == []
    assert study.resources.search("Michaelis-Menten", workspace="thesis")["items"][0]["resource"] == b["id"]
    assert dev.resources.list(workspace="thesis") == []


def test_files_are_referenced_not_copied_and_read_on_demand(runtime, apps, tmp_path):
    notes, study, _ = apps
    f = tmp_path / "paper.md"
    f.write_text("Original abstract about chloroplast thylakoid membranes.", encoding="utf-8")
    res = notes.resources.create("Paper", type="paper", path=str(f))
    assert res["kind"] == "file" and res["content"]["path"] == str(f.resolve())
    notes.resources.share(res["id"], ["anvira-study"])
    assert "thylakoid" in study.resources.read(res["id"])["text"]
    f.write_text("Revised abstract about mitochondrial cristae.", encoding="utf-8")      # a reference follows the file
    assert "cristae" in study.resources.read(res["id"])["text"]
    f.unlink()
    raises("resource_unavailable", study.resources.read, res["id"])                     # a clear error, not a stale copy
    binary = tmp_path / "blob.exe"
    binary.write_bytes(b"MZ")
    raises("invalid_request", notes.resources.create, "Bad", type="x", path=str(binary))


def test_audit_records_who_did_what_and_hides_others(runtime, apps):
    notes, study, dev = apps
    res = notes.resources.create("Audited", type="doc", text="Secret plan about osmosis.")
    notes.resources.share(res["id"], ["anvira-study"])
    study.resources.read(res["id"])
    study.resources.search("osmosis")
    notes.resources.revoke(res["id"], ["anvira-study"])
    actions = [(e["actor"], e["action"]) for e in notes.resources.audit(50, res["id"])]
    for expected in [("anvira-notes", "create"), ("anvira-notes", "share"), ("anvira-study", "read"), ("anvira-notes", "revoke")]:
        assert expected in actions
    assert all(e["resource"] == res["id"] for e in notes.resources.audit(50, res["id"]))
    assert dev.resources.audit(50, res["id"]) == []                                     # outsiders learn nothing
    user_view = runtime.owner.json("GET", "/v1/resources/audit", params={"limit": 200})["events"]
    assert any(e["resource"] == res["id"] and e["action"] == "share" for e in user_view)


def test_context_injection_into_chat_is_authorised_and_on_demand(runtime, apps, fake):
    notes, study, dev = apps
    res = notes.resources.create("Biology facts", type="notebook", text="The zebrafish genome has twenty-five chromosome pairs.")
    notes.resources.share(res["id"], ["anvira-study"])
    ask = [{"role": "user", "content": "How many chromosome pairs does the zebrafish genome have?"}]
    out = study.chat(ask, context={"query": "zebrafish chromosome pairs"})
    assert out["choices"][0]["message"]["content"].startswith("fake reply")
    sent = json.dumps(fake.requests[-1]["messages"])
    assert "twenty-five chromosome pairs" in sent                                       # resolved on demand, at request time
    assert out["runtime"]["context_used"] == [res["id"]]                                # and reported back
    fake.requests.clear()
    dev.chat(ask, context={"query": "zebrafish chromosome pairs"})                      # not authorised: nothing is injected
    assert "twenty-five" not in json.dumps(fake.requests[-1]["messages"])
    fake.requests.clear()
    study.chat(ask)                                                                     # no `context` option: nothing is loaded
    assert "twenty-five" not in json.dumps(fake.requests[-1]["messages"])


def test_local_data_is_never_sent_to_a_remote_model_without_opt_in(runtime, apps, fake, provider_model):
    notes, study, _ = apps
    res = notes.resources.create("Sensitive", type="doc", text="Patient zero has an unusual quokka allergy.")
    notes.resources.share(res["id"], ["anvira-study"])
    remote = runtime.owner.json("POST", "/v1/providers", {"label": "Cloud", "base_url": "https://api.remote-llm.example/v1",
                                                          "model": "cloud-1", "api_key": "sk-cloud-secret-000000000000"})
    try:
        runtime.owner.json("POST", "/v1/models/select", {"id": remote["id"]})
        ask = [{"role": "user", "content": "quokka allergy?"}]
        raises("remote_context_not_allowed", study.chat, ask, context={"query": "quokka"})
        raises("remote_context_not_allowed", study.chat, ask, memory={"recall": True})
        e = raises("remote_context_not_allowed", study.orcha.run, "Summarise the quokka allergy", context={"query": "quokka"})
        assert e.status == 403 and "allow_remote_context" in (e.hint or e.message)
        assert not any("quokka" in json.dumps(r) for r in fake.requests)                # nothing reached any model
    finally:
        runtime.owner.json("POST", "/v1/models/select", {"id": provider_model})


def test_capabilities_report_state_and_do_not_load_everything(runtime, apps):
    caps = {c["name"]: c for c in runtime.owner.json("GET", "/v1/capabilities")["capabilities"]}
    assert {"models", "chat", "orcha", "agents", "memory", "context", "resources", "aicl"} <= set(caps)
    assert caps["orcha"]["state"] == "running" and caps["aicl"]["state"] == "running"
    assert "Notes, Study and Code" in runtime.owner.json("GET", "/v1/capabilities")["note"]
    notes = apps[0]
    assert notes.me()["permissions"]                                                    # apps are told what they may do


def test_invalid_input_is_a_clean_4xx(runtime, apps):
    notes, *_ = apps
    raises("invalid_request", notes.resources.create, "", type="x", text="t")
    raises("invalid_request", notes.resources.create, "T", type="x", collection=None, text="")
    raises("invalid_reference", notes.resources.resolve, "runtime://nonsense")
    raises("resource_not_found", notes.resources.get, "res_000000000000")
    raises("resource_not_found", notes.resources.get, "res_ZZZZ%20bad")
    res = notes.resources.create("Ok", type="x", text="content here")
    raises("invalid_request", notes.resources.share, res["id"], ["Not A Valid App!"])
    raises("invalid_request", notes.resources.share, res["id"], ["anvira-notes"])       # the owner already has access
    raises("invalid_request", notes.resources.share, res["id"], ["anvira-study"], access="admin")
    assert notes.resources.delete(res["id"])["deleted"] is True
    raises("resource_not_found", notes.resources.get, res["id"])


def test_cli_context_commands(runtime, tmp_path, capsys):
    from test_04_cli_tui_discovery import run_cli  # reuse the in-process CLI runner: (exit code, stdout, stderr)

    def vis(rid):
        return json.loads(run_cli(capsys, "context", "inspect", rid, "--json")[1])["visibility"]

    f = tmp_path / "plan.md"
    f.write_text("Migration plan: move the ledger to the new database on Friday.", encoding="utf-8")
    code, out, _ = run_cli(capsys, "context", "add", "Plan", "--type", "plan", "--file", str(f), "--owner", "anvira-dev", "--json")
    assert code == 0
    rid = json.loads(out)["id"]
    assert vis(rid) == "private"
    assert run_cli(capsys, "context", "share", rid, "anvira-study", "--json")[0] == 0
    code, out, _ = run_cli(capsys, "context", "list")
    assert code == 0 and "Plan" in out and "shared" in out and "anvira-study" in out
    assert "Plan" in run_cli(capsys, "context", "search", "ledger database Friday")[1]
    code, out, err = run_cli(capsys, "context", "share", rid, "--global")           # non-interactive: must not silently go global
    assert code != 0 and "confirm" in (out + err).lower()
    assert vis(rid) == "shared"
    assert run_cli(capsys, "context", "share", rid, "--global", "--yes")[0] == 0
    assert vis(rid) == "global"
    assert run_cli(capsys, "context", "revoke", rid)[0] == 0
    assert vis(rid) == "private"
    assert "share" in run_cli(capsys, "context", "audit")[1]
    assert "resources" in run_cli(capsys, "capability", "list")[1]
    assert run_cli(capsys, "workspace", "create", "demo")[0] == 0
    assert "demo" in run_cli(capsys, "workspace", "list")[1]
    assert run_cli(capsys, "context", "delete", rid, "--yes")[0] == 0
