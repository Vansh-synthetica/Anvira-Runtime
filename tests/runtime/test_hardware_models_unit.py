"""Runtime: hardware detection, model compatibility, store, downloader, catalog, backend launch."""
import asyncio
import json
import struct
from pathlib import Path

import httpx
import pytest

from anvira_runtime.config.paths import resolve_layout
from anvira_runtime.config.settings import RuntimeConfig
from anvira_runtime.hardware.detect import detect_hardware, parse_nvidia_csv_line
from anvira_runtime.models import backend, gguf
from anvira_runtime.models.catalog import HuggingFaceClient, load_catalog, pick_preferred_file
from anvira_runtime.models.compat import assess, estimate_params_b, recommend
from anvira_runtime.models.download import DownloadCancelled, DownloadError, download_file
from anvira_runtime.models.store import ModelStore, ProviderStore, move_model_files, parse_gguf_filename, slugify, validate_storage_dir

GIB = 1024**3


# -------------------------------------------------------------------- hardware
def fake_runner(nvidia=True):
    def run(cmd, timeout=6):
        if "nvidia-smi" in cmd[0] or cmd[0].endswith("nvidia-smi.exe"):
            if not nvidia:
                raise RuntimeError("not found")
            if len(cmd) == 1:
                return "NVIDIA-SMI 555.1   Driver Version: 555.1   CUDA Version: 12.5"
            return '"NVIDIA GeForce RTX 3050 Laptop GPU, Ti", 4096, 3500, 555.99\n'
        raise RuntimeError("unsupported")
    return run


def test_nvidia_csv_with_commas_in_name():
    assert parse_nvidia_csv_line('"RTX 3050, Laptop", 4096, 3500, 555.99') == ["RTX 3050, Laptop", "4096", "3500", "555.99"]


def test_detect_nvidia(tmp_path):
    hw = detect_hardware(tmp_path, runner=fake_runner(True), use_cache=False)
    g = hw["gpu"]
    assert g["vendor"] == "nvidia" and g["vram_total_mib"] == 4096 and g["cuda_version"] == "12.5" and g["backend"] == "cuda"
    assert "cuda" in hw["acceleration_backends"] and "cpu" in hw["acceleration_backends"]
    assert hw["ram"]["total_mib"] > 0 and hw["cpu"]["logical_cores"] >= 1 and hw["storage"]["models"]["free_bytes"] > 0
    assert hw["platform"]["os"] and hw["platform"]["arch"]


def test_detect_without_gpu_falls_back_to_cpu(tmp_path, monkeypatch):
    import anvira_runtime.hardware.detect as d
    monkeypatch.setattr(d, "_gpu_windows_wmi", lambda r: None)
    hw = detect_hardware(tmp_path, runner=fake_runner(False), use_cache=False)
    assert hw["gpu"]["vendor"] == "none" and hw["gpu"]["backend"] == "cpu" and hw["acceleration_backends"] == ["cpu"]


# ---------------------------------------------------------------- compatibility
def hw(ram_gib=16, vram_mib=4096, backend_="cuda", free_gib=100):
    return {"ram": {"total_mib": ram_gib * 1024, "free_mib": ram_gib * 1024 - 2048},
            "gpu": {"vram_total_mib": vram_mib, "backend": backend_},
            "storage": {"models": {"free_bytes": free_gib * GIB, "free_gib": free_gib}}}


def model(gib, installed=False):
    return {"id": "m", "kind": "local", "size_bytes": int(gib * GIB), "installed": installed}


def test_compat_modes():
    assert assess(model(1.0), hw())["mode"] == "gpu"
    assert assess(model(4.4), hw())["mode"] == "partial-gpu"       # the RTX 3050 4 GiB case
    assert assess(model(4.4), hw(vram_mib=0, backend_="cpu"))["mode"] == "cpu"
    big = assess(model(40), hw())
    assert big["mode"] == "insufficient" and big["can_run"] is False and any("Needs about" in r for r in big["reasons"])
    assert assess({"kind": "provider"}, hw())["mode"] == "remote"
    assert assess({"kind": "local"}, hw())["can_run"] is None        # unknown size: no fake promises
    assert assess(model(1.0, installed=False), hw(free_gib=0))["can_run"] is False   # disk
    assert assess(model(1.0, installed=True), hw(free_gib=0))["can_run"] is True     # already downloaded
    assert assess(model(6), hw(vram_mib=0, backend_="metal", ram_gib=16))["mode"] == "gpu"


def test_recommend_orders_by_fit():
    models = [{**model(1), "id": "small", "params_b": 1}, {**model(4.4), "id": "mid", "params_b": 7},
              {**model(40), "id": "huge", "params_b": 70}, {"id": "cloud", "kind": "provider"}]
    ids = [m["id"] for m in recommend(models, hw())]
    assert ids == ["small", "mid"]


@pytest.mark.parametrize("name,expected", [("Qwen2.5-1.5B-Instruct-GGUF", 1.5), ("Llama-3.1-70B", 70.0), ("Mixtral-8x7B", 7.0),
                                           ("phi-3-mini", None), ("qwen__Qwen3-8B-GGUF", 8.0)])
def test_param_estimate(name, expected):
    assert estimate_params_b(name) == expected


# ------------------------------------------------------------------------ store
@pytest.fixture()
def store(tmp_path):
    lay = resolve_layout({"ANVIRA_RUNTIME_HOME": str(tmp_path / "home")}, "linux").ensure()
    return ModelStore(lay, tmp_path / "models")


def put(dirpath: Path, name: str, size=100):
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / name
    p.write_bytes(b"GGUF" + b"0" * size)
    return p


def test_filename_helpers():
    assert slugify("MaziyarPanahi__Qwen2.5-7B-Instruct-GGUF__Qwen2.5-7B-Instruct.IQ1_M") == "qwen2.5-7b-instruct.iq1_m"
    assert parse_gguf_filename("Qwen2.5-3B-Instruct-Q4_K_M.gguf") == {"architecture": "Qwen2.5", "quantization": "Q4_K_M"}


def test_scan_finds_models_in_any_folder(store, tmp_path):
    put(store.primary_dir, "a-Q4_K_M.gguf")
    other = tmp_path / "Other Drive" / "my models"
    put(other, "b-Q8_0.gguf")
    store.set_dirs(store.primary_dir, [other])
    ids = {m["id"]: m for m in store.scan()}
    assert set(ids) == {"a-q4_k_m", "b-q8_0"}
    assert ids["a-q4_k_m"]["location"] == "primary" and ids["b-q8_0"]["location"] == "extra"


def test_link_in_place_and_no_copy(store, tmp_path):
    f = put(tmp_path / "elsewhere", "linked-Q4_K_M.gguf")
    rec = store.link(str(f))
    assert rec["location"] == "linked" and Path(rec["path"]) == f.resolve()
    assert not (store.primary_dir / f.name).exists()
    with pytest.raises(FileNotFoundError):
        store.link(str(tmp_path / "nope.gguf"))
    with pytest.raises(ValueError):
        (tmp_path / "x.txt").write_text("x")
        store.link(str(tmp_path / "x.txt"))


def test_split_gguf_grouping_and_incomplete(store):
    for i in (1, 2, 3):
        put(store.primary_dir, f"big-0000{i}-of-00003.gguf", 50)
    m = store.scan()
    assert len(m) == 1 and m[0]["shards"] == 3 and m[0]["complete"] and m[0]["size_bytes"] == 3 * 54
    (store.primary_dir / "big-00002-of-00003.gguf").unlink()
    assert store.scan()[0]["complete"] is False
    (store.primary_dir / "big-00001-of-00003.gguf").unlink()
    assert store.scan() == []                       # no entry shard, nothing launchable


def test_same_filename_in_two_folders_gets_distinct_ids(store, tmp_path):
    put(store.primary_dir, "same.gguf")
    other = tmp_path / "o"
    put(other, "same.gguf")
    store.set_dirs(store.primary_dir, [other])
    assert len({m["id"] for m in store.scan()}) == 2


def test_remove_only_deletes_when_asked(store, tmp_path):
    a = put(store.primary_dir, "a.gguf")
    linked = put(tmp_path / "far", "b.gguf")
    store.link(str(linked))
    assert store.remove("b", delete_file=False)["deleted_files"] == [] and linked.exists()
    assert store.remove("a")["deleted_files"] == [str(a)] and not a.exists()
    with pytest.raises(KeyError):
        store.remove("zzz")


def test_filename_traversal_is_rejected(store):
    for bad in ("../evil.gguf", "..\\evil.gguf", "/abs.gguf", "sub/dir.gguf", ".hidden.gguf", "notes.txt", ""):
        with pytest.raises(ValueError):
            store.safe_path(bad)
    assert store.safe_path("ok.gguf").parent == store.primary_dir


def test_active_state_persists(store):
    assert store.active_id() is None
    store.set_active("x")
    assert store.active_id() == "x"
    store.set_active(None)
    assert store.active_id() is None


def test_validate_storage_dir(tmp_path):
    assert validate_storage_dir(tmp_path / "Deep Path" / "new") == (True, None)
    f = tmp_path / "file"
    f.write_text("x")
    ok, err = validate_storage_dir(f / "sub")
    assert not ok and err


def test_move_model_files_semantics(tmp_path):
    src, dst = tmp_path / "s", tmp_path / "d"
    dst.mkdir()
    a, b, c = put(src, "a.gguf", 10), put(src, "b.gguf", 20), put(src, "c.gguf", 30)
    (dst / "b.gguf").write_bytes(b"GGUF" + b"0" * 20)          # identical size: already present
    (dst / "c.gguf").write_bytes(b"different size")            # collision: refuse
    res = move_model_files([a, b, c, src / "gone.gguf"], dst, is_running=lambda p: False)
    moved = {Path(m["target"]).name for m in res["moved"]}
    assert moved == {"a.gguf", "b.gguf"} and not a.exists() and (dst / "a.gguf").exists()
    assert c.exists()                                             # never lost
    assert len(res["failed"]) == 2
    running = move_model_files([put(src, "r.gguf")], dst, is_running=lambda p: True)
    assert running["failed"][0]["error"].startswith("The model is currently running")


def test_provider_store_never_leaks_key(tmp_path):
    lay = resolve_layout({"ANVIRA_RUNTIME_HOME": str(tmp_path)}, "linux").ensure()
    ps = ProviderStore(lay)
    pub = ps.add(label="OpenRouter", base_url="https://openrouter.ai/api/v1/", model="llama", api_key="sk-or-1234567890abcdef")
    assert "api_key" not in pub and pub["has_api_key"] and pub["key_hint"] == "…cdef" and pub["id"] == "cloud:openrouter"
    assert "sk-or-1234567890abcdef" not in json.dumps(ps.list())
    assert ps.get_private("openrouter")["api_key"] == "sk-or-1234567890abcdef"
    with pytest.raises(ValueError):
        ps.add(label="x", base_url="ftp://bad", model="m")
    assert ps.remove("openrouter") and not ps.remove("openrouter")


# ----------------------------------------------------------------------- GGUF
def make_gguf(path: Path, n_layer=32):
    def s(x: str) -> bytes:
        return struct.pack("<Q", len(x)) + x.encode()

    kv = [("general.architecture", 8, s("llama")), ("llama.block_count", 4, struct.pack("<I", n_layer)),
          ("llama.attention.head_count", 4, struct.pack("<I", 32)), ("llama.attention.head_count_kv", 4, struct.pack("<I", 8)),
          ("llama.embedding_length", 4, struct.pack("<I", 4096)),
          ("tokenizer.tokens", 9, struct.pack("<I", 8) + struct.pack("<Q", 2) + s("a") + s("b"))]
    body = b"".join(s(k) + struct.pack("<I", t) + v for k, t, v in kv)
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv)) + body + b"\0" * 1000)


def test_gguf_metadata_reader(tmp_path):
    f = tmp_path / "m.gguf"
    make_gguf(f, 28)
    meta = gguf.read_metadata(f)
    assert meta == {"architecture": "llama", "n_layer": 28, "n_head": 32, "n_head_kv": 8, "n_embd": 4096, "context_length": None}
    (tmp_path / "bad.gguf").write_bytes(b"NOPE" + b"\0" * 100)
    assert gguf.read_metadata(tmp_path / "bad.gguf") is None
    assert gguf.read_metadata(tmp_path / "missing.gguf") is None


def test_gpu_layer_heuristic_matches_anvira_launcher(tmp_path):
    f = tmp_path / "m.gguf"
    make_gguf(f)
    f.write_bytes(f.read_bytes() + b"\0" * (4400 * 1024 * 1024 - f.stat().st_size))   # ~4.4 GiB model
    meta = gguf.read_metadata(f)
    layers = backend.compute_gpu_layers(f, 4096, 4096, meta)
    assert 0 < layers < 32                       # partial offload on a 4 GiB card, like the Electron app
    assert backend.compute_gpu_layers(f, 0, 4096, meta) == 0
    assert backend.pick_context(meta, None, None) == 4096 and backend.pick_context(meta, 100000, None) == 16384
    assert backend.pick_context(meta, 100000, 2048) == 2048


def test_build_launch_flags_and_missing_binary(tmp_path, fake_llama):
    lay = resolve_layout({"ANVIRA_RUNTIME_HOME": str(tmp_path / "h")}, "linux").ensure()
    cfg = RuntimeConfig.load(lay)
    m = tmp_path / "m.gguf"
    make_gguf(m)
    rec = {"id": "m", "path": str(m)}
    hw_gpu = {"gpu": {"backend": "cuda", "vram_total_mib": 24000, "vram_free_mib": 20000}, "cpu": {"optimal_threads": 8}}
    cfg.set("models.llama_server_path", "/definitely/not/here")
    spec, info = backend.build_launch(lay, cfg, rec, hw_gpu, lay.logs_dir)
    if spec is None:   # no other llama-server on this machine: structured "missing" path
        assert info["error"] and info["searched"]
    cfg.set("models.llama_server_path", str(fake_llama))
    spec, info = backend.build_launch(lay, cfg, rec, hw_gpu, lay.logs_dir)
    assert spec and spec.argv[0] == str(fake_llama)
    a = spec.argv
    assert "--host" in a and a[a.index("--host") + 1] == "127.0.0.1" and "--jinja" in a and "-np" in a
    assert spec.health_url.endswith("/health") and spec.kind == "model" and info["gpu"] is True
    cfg.set("models.gpu", "off")
    spec, info = backend.build_launch(lay, cfg, rec, hw_gpu, lay.logs_dir)
    assert "-ngl" not in spec.argv and info["gpu"] is False


# -------------------------------------------------------------------- catalog
def test_catalog_is_well_formed():
    cat = load_catalog()
    assert len(cat) >= 5 and len({m["id"] for m in cat}) == len(cat)
    for m in cat:
        assert m["source"]["repo"] and m["source"]["file"].endswith(".gguf") and m["size_bytes"] > 0 and m["kind"] == "local"
    assert "qwen3-8b" in {m["id"] for m in cat}


def test_pick_preferred_file():
    files = [{"name": "m-Q8_0.gguf"}, {"name": "m-Q4_K_M.gguf"}, {"name": "README.md"}, {"name": "m-Q4_K_M-00002-of-00002.gguf"}]
    assert pick_preferred_file(files)["name"] == "m-Q4_K_M.gguf"
    assert pick_preferred_file([{"name": "x.txt"}]) is None


def test_hf_client_over_mock_transport():
    def handler(req: httpx.Request):
        if req.url.path == "/api/models":
            assert req.url.params["filter"] == "gguf" and req.url.params["search"] == "qwen"
            return httpx.Response(200, json=[{"id": "Qwen/Qwen3-4B-GGUF", "downloads": 5, "tags": ["gguf"]}])
        return httpx.Response(200, json={"siblings": [{"rfilename": "a.gguf", "size": 7}]})
    hf = HuggingFaceClient("https://hf.test", transport=httpx.MockTransport(handler))
    res = asyncio.run(hf.search("qwen"))
    assert res[0]["id"] == "Qwen/Qwen3-4B-GGUF" and res[0]["params_b"] == 4.0
    assert asyncio.run(hf.files("o/r")) == [{"name": "a.gguf", "size_bytes": 7}]
    assert hf.download_url("o/r", "a b.gguf") == "https://hf.test/o/r/resolve/main/a%20b.gguf"


# ------------------------------------------------------------------ downloader
DATA = bytes(range(256)) * 1000


def range_transport(data=DATA, honor_range=True, truncate_to=None):
    def handler(req: httpx.Request):
        rng = req.headers.get("range")
        body = data
        if rng and honor_range:
            start = int(rng.split("=")[1].rstrip("-"))
            if start >= len(data):
                return httpx.Response(416)
            return httpx.Response(206, content=data[start:], headers={"content-length": str(len(data) - start)})
        if truncate_to is not None:
            return httpx.Response(200, content=body[:truncate_to], headers={"content-length": str(len(body))})
        return httpx.Response(200, content=body, headers={"content-length": str(len(body))})
    return httpx.MockTransport(handler)


def test_download_writes_atomically(tmp_path):
    dest = tmp_path / "m.gguf"
    seen = []
    asyncio.run(download_file("http://x/m.gguf", dest, transport=range_transport(), on_progress=lambda d, t, s: seen.append((d, t))))
    assert dest.read_bytes() == DATA and not (tmp_path / "m.gguf.part").exists() and seen[-1] == (len(DATA), len(DATA))


def test_download_resumes_from_partial(tmp_path):
    dest = tmp_path / "m.gguf"
    (tmp_path / "m.gguf.part").write_bytes(DATA[:1000])
    asyncio.run(download_file("http://x/m.gguf", dest, transport=range_transport()))
    assert dest.read_bytes() == DATA


def test_download_restarts_if_server_ignores_range(tmp_path):
    dest = tmp_path / "m.gguf"
    (tmp_path / "m.gguf.part").write_bytes(b"garbage")
    asyncio.run(download_file("http://x/m.gguf", dest, transport=range_transport(honor_range=False)))
    assert dest.read_bytes() == DATA


def test_download_detects_truncation_and_keeps_partial(tmp_path):
    dest = tmp_path / "m.gguf"
    with pytest.raises(DownloadError):
        asyncio.run(download_file("http://x/m.gguf", dest, transport=range_transport(honor_range=False, truncate_to=500)))
    assert not dest.exists() and (tmp_path / "m.gguf.part").exists()      # resumable, never a truncated "model"


def test_download_http_error_and_cancel(tmp_path):
    with pytest.raises(DownloadError):
        asyncio.run(download_file("http://x/m", tmp_path / "m.gguf", transport=httpx.MockTransport(lambda r: httpx.Response(404))))
    ev = asyncio.Event()
    ev.set()
    with pytest.raises(DownloadCancelled):
        asyncio.run(download_file("http://x/m", tmp_path / "n.gguf", transport=range_transport(), cancel=ev, chunk_size=10))
    assert not (tmp_path / "n.gguf").exists()
