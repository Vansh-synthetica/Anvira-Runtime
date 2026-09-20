"""Shared fixtures. ``runtime`` starts a REAL daemon (real ORCHA + Nomi processes) in an isolated home."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "runtime", REPO / "sdk" / "python", REPO / "AICL"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.path.insert(0, str(Path(__file__).parent))

from anvira_client import AnviraRuntime, bootstrap  # noqa: E402
from anvira_client.discovery import probe, read_token  # noqa: E402
from anvira_client.http import Http  # noqa: E402
from fakes import FakeLLM, make_fake_llama_server  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: starts real processes or installs (deselect with -m 'not slow')")


class RuntimeHandle:
    """A running isolated runtime plus helpers."""

    def __init__(self, home: Path, env: dict[str, str], fake: FakeLLM, llama: Path):
        self.home, self.env, self.fake, self.llama = home, env, fake, llama
        self.info = probe(env)

    @property
    def owner(self) -> Http:
        return Http(self.info.base_url, read_token("owner", self.env), timeout=60)

    def connect(self, app_id: str, **kw) -> AnviraRuntime:
        return AnviraRuntime.connect(app_id, env=self.env, **kw)

    def refresh(self) -> None:
        self.info = probe(self.env)


def start_isolated(tmp: Path, fake: FakeLLM, llama: Path, extra_config: dict | None = None,
                   extra_env: dict[str, str] | None = None, auto_stop: bool = False,
                   idle_grace_s: float | None = None, start: bool = True) -> RuntimeHandle:
    home = tmp / "home"
    (home / "config").mkdir(parents=True, exist_ok=True)
    cfg = {"api": {"port": 0}, "models": {"llama_server_path": str(llama), "huggingface_endpoint": fake.base,
                                          "gpu": "off", "autostart_active": False},
           "supervisor": {"health_interval_s": 0.5, "restart_window_s": 300, "restart_limit": 3, "start_timeout_s": 90},
           "orcha": {"allow_mock": False, "job_timeout_s": 60}}
    for k, v in (extra_config or {}).items():
        cfg.setdefault(k, {}).update(v)
    (home / "config" / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    appdata = tmp / "appdata"                       # hermetic: never discover the developer's real Anvira folders
    appdata.mkdir(exist_ok=True)
    env = {"ANVIRA_RUNTIME_HOME": str(home), "APPDATA": str(appdata), "LOCALAPPDATA": str(appdata),
           "XDG_CONFIG_HOME": str(appdata), "HOME": str(tmp), **(extra_env or {})}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        if start:
            bootstrap.start_runtime(env=env, timeout=120, auto_stop=auto_stop, idle_grace_s=idle_grace_s)
    except Exception:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        raise
    handle = RuntimeHandle(home, env, fake, llama)
    handle._old_env = old  # type: ignore[attr-defined]
    return handle


def stop_isolated(h: RuntimeHandle) -> None:
    bootstrap.stop_runtime(h.env, timeout=30)
    for k, v in getattr(h, "_old_env", {}).items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


@pytest.fixture(scope="session")
def fake_llm():
    f = FakeLLM()
    yield f
    f.close()


@pytest.fixture(scope="session")
def fake_llama(tmp_path_factory):
    return make_fake_llama_server(tmp_path_factory.mktemp("fakebin"))


@pytest.fixture(scope="session")
def runtime(tmp_path_factory, fake_llm, fake_llama):
    """Real runtime (ORCHA + Nomi + AICL bus), isolated home, fake inference backends."""
    h = start_isolated(tmp_path_factory.mktemp("rt"), fake_llm, fake_llama)
    yield h
    stop_isolated(h)


@pytest.fixture(scope="session")
def provider_model(runtime):
    """Register the fake OpenAI-compatible provider (as owner) and return its model id."""
    rec = runtime.owner.json("POST", "/v1/providers", {"label": "Fake", "base_url": runtime.fake.base + "/v1",
                                                       "model": "fake-model", "api_key": "sk-test-secret-1234567890"})
    return rec["id"]


@pytest.fixture()
def fake(fake_llm):
    fake_llm.mode, fake_llm.delay = "ok", 0.0
    fake_llm.requests.clear()
    return fake_llm


def wait_for(fn, timeout=30.0, interval=0.3, message="condition"):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = fn()
            if last:
                return last
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {message} (last={last!r})")
