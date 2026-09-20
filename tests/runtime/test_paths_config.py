"""Runtime: configuration and platform paths."""
import json
from pathlib import Path

import pytest

from anvira_client.paths import runtime_dirs
from anvira_runtime.config.paths import resolve_layout
from anvira_runtime.config.settings import ConfigError, RuntimeConfig, flat_keys
from anvira_runtime.version import parse_version, satisfies


ENVS = {
    "win32": {"LOCALAPPDATA": r"C:\Users\u\AppData\Local", "USERPROFILE": r"C:\Users\u"},
    "darwin": {"HOME": "/Users/u"},
    "linux": {"HOME": "/home/u"},
}


@pytest.mark.parametrize("plat", ["win32", "darwin", "linux"])
def test_layout_per_platform_and_client_parity(plat):
    """The runtime and the (duplicated) client path logic must agree on every OS."""
    lay = resolve_layout(ENVS[plat], plat)
    cl = runtime_dirs(ENVS[plat], plat)
    assert lay.home == cl["home"] and lay.state_dir == cl["state"] and lay.config_dir == cl["config"]
    assert lay.logs_dir == cl["logs"] and lay.models_dir == cl["models"] and lay.bin_dir == cl["bin"]


def test_platform_conventions():
    assert "AnviraRuntime" in str(resolve_layout(ENVS["win32"], "win32").home)
    assert "Application Support" in str(resolve_layout(ENVS["darwin"], "darwin").home)
    assert ".local" in str(resolve_layout(ENVS["linux"], "linux").home)


def test_xdg_and_overrides(tmp_path):
    env = {"HOME": "/h", "XDG_DATA_HOME": str(tmp_path / "d"), "XDG_CONFIG_HOME": str(tmp_path / "c")}
    lay = resolve_layout(env, "linux")
    assert lay.home == tmp_path / "d" / "anvira-runtime" and lay.config_dir == tmp_path / "c" / "anvira-runtime"
    env = {"ANVIRA_RUNTIME_HOME": str(tmp_path / "x"), "ANVIRA_MODELS_DIR": str(tmp_path / "big")}
    lay = resolve_layout(env, "win32")
    assert lay.home == tmp_path / "x" and lay.models_dir == tmp_path / "big" and lay.state_dir == tmp_path / "x" / "state"


def test_runtime_state_is_separate_from_app_data(tmp_path):
    lay = resolve_layout({"ANVIRA_RUNTIME_HOME": str(tmp_path)}, "linux").ensure()
    # runtime-owned dirs only; there is no notes/study/workspace directory in the runtime home
    names = {p.name for p in tmp_path.iterdir()}
    assert not names & {"notes", "study", "workspace", "notebooks"}
    assert lay.state_dir.is_dir() and lay.models_dir.is_dir()


def test_config_defaults_roundtrip_and_validation(tmp_path):
    lay = resolve_layout({"ANVIRA_RUNTIME_HOME": str(tmp_path)}, "linux").ensure()
    cfg = RuntimeConfig.load(lay)
    assert cfg.get("api.host") == "127.0.0.1" and cfg.get("models.gpu") == "auto"
    assert cfg.set("api.port", "5000") == 5000
    assert cfg.set("models.autostart_active", "false") is False
    assert cfg.set("models.extra_dirs", "a,b") == ["a", "b"]
    assert cfg.set("models.context_size", "auto") is None
    cfg.save()
    again = RuntimeConfig.load(lay)
    assert again.get("api.port") == 5000 and again.get("models.extra_dirs") == ["a", "b"] and again.load_error is None


@pytest.mark.parametrize("key,val", [("api.host", "0.0.0.0"), ("api.host", "192.168.1.5"), ("models.gpu", "maybe"),
                                     ("api.port", "70000"), ("api.port", "abc"), ("nope.key", "1"),
                                     ("logging.level", "LOUD"), ("models.autostart_active", "perhaps")])
def test_config_rejects_bad_values(tmp_path, key, val):
    cfg = RuntimeConfig.load(resolve_layout({"ANVIRA_RUNTIME_HOME": str(tmp_path)}, "linux").ensure())
    with pytest.raises(ConfigError):
        cfg.set(key, val)


def test_config_never_binds_externally(tmp_path):
    """Security invariant: the API host can only ever be loopback."""
    cfg = RuntimeConfig.load(resolve_layout({"ANVIRA_RUNTIME_HOME": str(tmp_path)}, "linux").ensure())
    for host in ("0.0.0.0", "::", "example.com"):
        with pytest.raises(ConfigError):
            cfg.set("api.host", host)
    assert cfg.set("api.host", "localhost") == "localhost"


def test_broken_config_is_reported_not_fatal(tmp_path):
    lay = resolve_layout({"ANVIRA_RUNTIME_HOME": str(tmp_path)}, "linux").ensure()
    lay.config_file.write_text("{ not json", encoding="utf-8")
    cfg = RuntimeConfig.load(lay)
    assert cfg.load_error and "config.json" in cfg.load_error
    assert cfg.get("api.port") == 47615  # defaults keep the runtime usable
    lay.config_file.write_text(json.dumps({"api": {"host": "0.0.0.0"}}), encoding="utf-8")
    assert RuntimeConfig.load(lay).load_error  # hand-edited unsafe value is caught too


def test_config_lists_every_key(tmp_path):
    keys = flat_keys()
    assert "models.models_dir" in keys and "models.extra_dirs" in keys and "api.allowed_origins" in keys


def test_models_dir_anywhere(tmp_path):
    lay = resolve_layout({"ANVIRA_RUNTIME_HOME": str(tmp_path / "home")}, "linux").ensure()
    cfg = RuntimeConfig.load(lay)
    assert cfg.models_dir() == lay.models_dir
    weird = tmp_path / "Drive D" / "my models (big)"
    cfg.set("models.models_dir", str(weird))
    assert cfg.models_dir() == weird


def test_versions():
    assert parse_version("1.2.3") == (1, 2, 3) and parse_version("v2.0") == (2, 0, 0) and parse_version("1.0.0-rc1") == (1, 0, 0)
    assert satisfies("1.10.0", "1.9.9") and not satisfies("1.0.0", "1.0.1") and satisfies("1.0.0", "1.0.0")
