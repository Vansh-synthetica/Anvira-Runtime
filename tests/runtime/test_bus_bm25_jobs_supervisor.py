"""Runtime: AICL bus, BM25 (parity with Anvira's TypeScript), jobs, supervisor."""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from anvira_runtime.core.bm25 import chunk_text, rank_chunks, tokenize
from anvira_runtime.core.bus import AICLBus, BusUnavailable, load_aicl, native_available
from anvira_runtime.core.errors import RuntimeApiError
from anvira_runtime.core.jobs import JobRegistry
from anvira_runtime.process.infra import locate_aicl
from anvira_runtime.process.supervisor import ServiceSpec, Supervisor, free_port, pid_alive, port_in_use

REPO = Path(__file__).resolve().parents[2]
AICL_ROOT = REPO / "AICL"


def run(coro):
    return asyncio.run(coro)


# -------------------------------------------------------------------- AICL bus
@pytest.fixture()
def bus():
    b = AICLBus(AICL_ROOT)

    async def echo(action, args, ctx):
        if action == "boom":
            raise ValueError("kaput")
        if action == "api_error":
            raise RuntimeApiError("nope", "denied by module", 403)
        if action == "slow":
            await asyncio.sleep(5)
        if action == "big":
            return {"blob": "x" * 200_000}
        return {"action": action, "args": args, "origin": ctx.origin, "app": ctx.app_id, "deadline_ms": ctx.deadline_ms}
    b.register("echo", echo, ["echo.test"])
    return b


def test_bus_uses_the_new_aicl_package():
    import aicl
    assert "AICL" in str(Path(aicl.__file__)) and (AICL_ROOT / "aicl").exists()
    assert AICLBus(AICL_ROOT).version


def test_bus_locates_aicl_from_layout():
    from anvira_runtime.config.paths import resolve_layout
    assert locate_aicl(resolve_layout({"ANVIRA_RUNTIME_HOME": "/tmp/x"}, "linux")) == AICL_ROOT


def test_bus_call_roundtrips_through_real_aicl_packets(bus):
    res = run(bus.call("echo", "hello", {"a": [1, 2], "ü": "ünï"}, app_id="notes", timeout=5))
    assert res["args"] == {"a": [1, 2], "ü": "ünï"} and res["app"] == "notes" and res["deadline_ms"] == 5000
    st = bus.stats()
    assert st["modules"]["echo"]["calls"] == 1 and st["modules"]["echo"]["bytes_in"] > 0 and st["codec"].startswith("aicl-bin")
    rec = bus.recent()[-1]
    assert rec["module"] == "echo" and rec["ok"] and rec["op"] == "CALL" and len(rec["message_id"]) == 32


def test_bus_uses_semantic_opcodes(bus):
    run(bus.call("echo", "store", {}, op=bus.ops.OP_MEMORY_WRITE))
    run(bus.call("echo", "run", {}, op=bus.ops.OP_EXECUTE))
    assert [r["op"] for r in bus.recent()] == ["MEMORY_WRITE", "EXECUTE"]


def test_bus_large_payload(bus):
    assert len(run(bus.call("echo", "big", {}))["blob"]) == 200_000


def test_bus_error_propagation(bus):
    with pytest.raises(RuntimeApiError) as e:
        run(bus.call("echo", "api_error", {}))
    assert e.value.code == "nope" and e.value.status == 403                    # structured error preserved
    with pytest.raises(RuntimeApiError) as e:
        run(bus.call("echo", "boom", {}))
    assert e.value.code == "module_error" and "kaput" in e.value.message
    with pytest.raises(RuntimeApiError) as e:
        run(bus.call("ghost", "x", {}))
    assert e.value.code == "module_unavailable" and e.value.status == 503
    st = bus.stats()["modules"]["echo"]
    assert st["errors"] == 2 and not bus.recent()[-1]["ok"]


def test_bus_timeout_does_not_hang(bus):
    t0 = time.monotonic()
    with pytest.raises(RuntimeApiError) as e:
        run(bus.call("echo", "slow", {}, timeout=0.3))
    assert e.value.code == "timeout" and e.value.status == 504 and time.monotonic() - t0 < 2


def test_bus_unavailable_when_aicl_missing(tmp_path, monkeypatch):
    for name in [n for n in sys.modules if n == "aicl" or n.startswith("aicl.")]:
        monkeypatch.delitem(sys.modules, name)
    saved = list(sys.path)
    monkeypatch.setattr(sys, "path", [p for p in saved if "AICL" not in p])
    with pytest.raises(BusUnavailable):
        load_aicl(tmp_path)          # a directory without an `aicl` package


def test_native_core_reported_honestly():
    assert native_available() in (True, False)


# ------------------------------------------------------------------------ BM25
def test_bm25_tokenizer_and_chunking():
    assert tokenize("The Quick brown FOX, and a 42!") == ["quick", "brown", "fox", "42"]
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(10))
    chunks = chunk_text("Source: a.pdf", text)
    assert len(chunks) > 1 and chunks[0]["id"] == "Source: a.pdf#0" and all(len(c["text"]) <= 1400 for c in chunks)
    assert chunk_text("t", "   ") == []
    assert len(chunk_text("t", "x" * 3000)) == 4                                # hard-split oversized paragraph


def test_bm25_ranks_relevant_first():
    docs = [{"id": "a", "text": "Cats purr and sleep all day"}, {"id": "b", "text": "The Calvin cycle fixes carbon dioxide in plants"},
            {"id": "c", "text": "Stock markets rose on Tuesday"}]
    ranked = rank_chunks(docs, "calvin cycle carbon")
    assert ranked[0]["id"] == "b" and ranked[0]["score"] > 0 and ranked[-1]["score"] == 0
    assert all(r["score"] == 0 for r in rank_chunks(docs, "the a of"))         # stopwords only


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_bm25_parity_with_anvira_typescript(tmp_path):
    """Same chunks + same ranking as the original ``src/utils/bm25.ts`` (run by Node)."""
    ts = REPO / "src" / "utils" / "bm25.ts"
    if not ts.exists():
        pytest.skip("src/utils/bm25.ts not present")
    corpus = ("Photosynthesis converts light energy into chemical energy. " * 12 + "\n\n"
              + "The Calvin cycle fixes carbon dioxide into sugars.\n\n" + "Mitochondria produce ATP through respiration. " * 15
              + "\n\n" + "Stock markets and interest rates affect the economy.\n\n" + "x" * 2200)
    queries = ["calvin cycle carbon dioxide", "ATP mitochondria respiration", "interest rates", "photosynthesis energy"]
    script = tmp_path / "run.mjs"
    script.write_text(textwrap.dedent(f"""
        import {{ chunkText, rankChunks }} from {json.dumps(ts.as_uri())}
        const chunks = chunkText('S', {json.dumps(corpus)})
        const out = {{ chunks: chunks.map(c => c.id + ':' + c.text.length), ranks: {{}} }}
        for (const q of {json.dumps(queries)}) out.ranks[q] = rankChunks(chunks, q).map(c => [c.id, Number(c.score.toFixed(6))])
        console.log(JSON.stringify(out))
    """), encoding="utf-8")
    proc = subprocess.run(["node", "--experimental-strip-types", "--no-warnings", str(script)], capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"node could not load the .ts module: {proc.stderr[-200:]}")
    expected = json.loads(proc.stdout)
    chunks = chunk_text("S", corpus)
    assert [f"{c['id']}:{len(c['text'])}" for c in chunks] == expected["chunks"]
    for q in queries:
        got = [[c["id"], round(c["score"], 6)] for c in rank_chunks(chunks, q)]
        assert got == expected["ranks"][q], q


# ------------------------------------------------------------------------ jobs
def test_job_lifecycle_and_ownership():
    async def scenario():
        jobs = JobRegistry()
        ok = jobs.create("x", "notes")
        jobs.submit(ok, lambda j: asyncio.sleep(0, result={"v": 1}))
        await asyncio.sleep(0.05)
        assert ok.state == "completed" and ok.result == {"v": 1} and ok.finished_at

        async def boom(j): raise RuntimeApiError("bad", "nope", 400, hint="fix it")
        bad = jobs.submit(jobs.create("x", "notes"), boom)
        await asyncio.sleep(0.05)
        assert bad.state == "failed" and bad.error["code"] == "bad" and bad.error["hint"] == "fix it"

        async def crash(j): raise KeyError("k")
        crashed = jobs.submit(jobs.create("x", None), crash)
        await asyncio.sleep(0.05)
        assert crashed.state == "failed" and crashed.error["code"] == "job_failed"      # never propagates

        hooked = []
        async def hook(): hooked.append(1)
        slow = jobs.create("x", "study")
        jobs.set_cancel_hook(slow.id, hook)
        jobs.submit(slow, lambda j: asyncio.sleep(30))
        await asyncio.sleep(0.05)
        assert slow.state == "running"
        await jobs.cancel(slow.id)
        assert slow.state == "cancelled" and hooked == [1]
        with pytest.raises(RuntimeApiError) as e:
            await jobs.cancel(slow.id)
        assert e.value.code == "job_not_cancellable"

        # ownership: an app only sees its own jobs; the owner sees everything
        assert {j.id for j in jobs.list("notes")} == {ok.id, bad.id}
        with pytest.raises(RuntimeApiError) as e:
            jobs.get(slow.id, "notes")
        assert e.value.status == 404
        assert jobs.get(slow.id, None, owner=True) is slow
        assert jobs.counts()["completed"] == 1
    run(scenario())


def test_jobs_persist_and_running_become_interrupted(tmp_path):
    f = tmp_path / "jobs.json"

    async def first():
        j = JobRegistry(f)
        done = j.submit(j.create("a", "x"), lambda job: asyncio.sleep(0, result=1))
        running = j.submit(j.create("b", "x"), lambda job: asyncio.sleep(30))
        await asyncio.sleep(0.05)
        j._persist()
        return done.id, running.id, f.read_text()      # snapshot while "running", like a crash
    done_id, running_id, snapshot = run(first())
    f.write_text(snapshot)
    again = JobRegistry(f)
    assert again.get(done_id, None, True).state == "completed"
    r = again.get(running_id, None, True)
    assert r.state == "interrupted" and r.error["code"] == "runtime_restarted"


# ------------------------------------------------------------------ supervisor
SERVICE = textwrap.dedent('''
    import sys, os, json
    from http.server import BaseHTTPRequestHandler, HTTPServer
    port = int(sys.argv[1]); mode = sys.argv[2] if len(sys.argv) > 2 else "ok"
    if mode == "die_at_start":
        print("boom at startup", flush=True); sys.exit(3)
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    print("listening", port, flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()
''')


@pytest.fixture()
def svc(tmp_path):
    script = tmp_path / "svc.py"
    script.write_text(SERVICE, encoding="utf-8")

    def spec(name="svc", port=None, mode="ok", restart=True, timeout=20.0):
        port = port or free_port()
        return ServiceSpec(name=name, argv=[sys.executable, str(script), str(port), mode], log_path=tmp_path / f"{name}.log",
                           port=port, health_url=f"http://127.0.0.1:{port}/", ready_timeout_s=timeout, restart=restart)
    return spec


def sup(tmp_path, **kw):
    return Supervisor(tmp_path / "children.json", health_interval_s=0.2, **kw)


def test_supervisor_start_health_stop(tmp_path, svc):
    async def scenario():
        s = sup(tmp_path)
        spec = svc()
        s.add(spec)
        st = await s.start("svc")
        assert st.state == "running" and pid_alive(st.pid) and port_in_use(spec.port)
        assert json.loads((tmp_path / "children.json").read_text())["svc"]["pid"] == st.pid
        pid = st.pid
        await s.stop_all()
        assert not pid_alive(pid) and not port_in_use(spec.port) and json.loads((tmp_path / "children.json").read_text()) == {}
    run(scenario())


def test_supervisor_reports_startup_crash_with_log_tail(tmp_path, svc):
    async def scenario():
        s = sup(tmp_path)
        s.add(svc(mode="die_at_start"))
        with pytest.raises(RuntimeError) as e:
            await s.start("svc")
        assert "exited during startup" in str(e.value) and "boom at startup" in str(e.value)
        assert s.services["svc"].state == "failed"
        await s.stop_all()
    run(scenario())


def test_supervisor_restarts_crashed_service_then_gives_up(tmp_path, svc):
    async def scenario():
        s = sup(tmp_path, restart_limit=2, restart_window_s=60)
        s.add(svc())
        st = await s.start("svc")
        s.start_monitor()
        first = st.pid
        subprocess_kill(first)
        for _ in range(80):
            await asyncio.sleep(0.25)
            if st.total_restarts >= 1 and st.state == "running" and st.pid != first:
                break
        assert st.total_restarts >= 1 and st.state == "running" and st.pid != first      # recovered
        for _ in range(4):                                                                # crash loop -> failed
            subprocess_kill(st.pid)
            for _ in range(60):
                await asyncio.sleep(0.25)
                if st.state == "failed" or (st.state == "running" and pid_alive(st.pid)):
                    if st.state == "failed":
                        break
            if st.state == "failed":
                break
        assert st.state == "failed" and "gave up" in (st.last_error or "")
        await s.stop_all()
    run(scenario())


def subprocess_kill(pid):
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        os.kill(pid, 9)


def test_supervisor_no_restart_when_disabled(tmp_path, svc):
    async def scenario():
        s = sup(tmp_path)
        s.add(svc(restart=False))
        st = await s.start("svc")
        s.start_monitor()
        subprocess_kill(st.pid)
        for _ in range(40):
            await asyncio.sleep(0.25)
            if st.state == "failed":
                break
        assert st.state == "failed" and st.total_restarts == 0
        await s.stop_all()
    run(scenario())


def test_supervisor_adopts_healthy_existing_service_and_never_kills_it(tmp_path, svc):
    async def scenario():
        spec = svc()
        s1 = sup(tmp_path)
        s1.add(spec)
        st1 = await s1.start("svc")
        s2 = Supervisor(tmp_path / "c2.json", health_interval_s=0.2)
        s2.add(ServiceSpec(**{**spec.__dict__, "name": "svc"}))
        st2 = await s2.start("svc")
        assert st2.adopted and st2.state == "running" and st2.proc is None
        await s2.stop("svc")                       # stopping an adopted service must NOT kill the owner's process
        assert pid_alive(st1.pid)
        await s1.stop_all()
        await s2._client.aclose()
    run(scenario())


def test_supervisor_port_conflict_with_foreign_program(tmp_path, svc):
    import socket
    async def scenario():
        blocker = socket.socket()
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        port = blocker.getsockname()[1]
        s = sup(tmp_path)
        s.add(svc(port=port))
        with pytest.raises(RuntimeError) as e:
            await s.start("svc")
        assert "in use by another program" in str(e.value)
        blocker.close()
        await s.stop_all()
    run(scenario())


def test_supervisor_reaps_stale_children_only_if_health_answers(tmp_path, svc):
    async def scenario():
        s1 = sup(tmp_path)
        spec = svc()
        s1.add(spec)
        st = await s1.start("svc")
        pid = st.pid
        # simulate a runtime crash: forget the process but keep children.json
        s1.services.clear()
        s2 = Supervisor(tmp_path / "children.json")
        assert await s2.reap_stale() == ["svc"] and not pid_alive(pid)
        await s2._client.aclose()
        await s1._client.aclose()
        # recorded pid whose health URL no longer answers (recycled pid) is left alone
        (tmp_path / "children.json").write_text(json.dumps({"x": {"pid": os.getpid(), "health_url": "http://127.0.0.1:9/"}}))
        s3 = Supervisor(tmp_path / "children.json")
        assert await s3.reap_stale() == [] and pid_alive(os.getpid())
        await s3._client.aclose()
    run(scenario())


def test_pid_alive_is_side_effect_free():
    assert pid_alive(os.getpid()) and not pid_alive(0) and not pid_alive(None) and not pid_alive(4_000_000_000 % 999_983 + 3_000_000)
