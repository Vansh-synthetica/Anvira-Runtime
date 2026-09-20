"""The TypeScript SDK (Node) against a real runtime, plus path parity with the Python layout code."""
import json
import os
import shutil
import subprocess

import pytest

from anvira_runtime.config.paths import resolve_layout
from conftest import REPO

pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("node") is None, reason="node not available")]


def test_typescript_sdk_end_to_end(runtime, provider_model):
    cases = []
    for plat, env in (("win32", {"LOCALAPPDATA": "C:\\Users\\u\\AppData\\Local", "USERPROFILE": "C:\\Users\\u"}),
                      ("darwin", {"HOME": "/Users/u"}), ("linux", {"HOME": "/home/u", "XDG_DATA_HOME": "/x/data"}),
                      ("linux", {"ANVIRA_RUNTIME_HOME": "/custom/home", "HOME": "/home/u"})):
        lay = resolve_layout(env, plat)
        cases.append({"platform": plat, "env": env, "dirs": {
            "home": str(lay.home), "state": str(lay.state_dir), "config": str(lay.config_dir),
            "logs": str(lay.logs_dir), "models": str(lay.models_dir), "bin": str(lay.bin_dir)}})
    env = {**os.environ, **runtime.env, "ANVIRA_TEST_PROVIDER": provider_model, "ANVIRA_TEST_DIRS": json.dumps(cases)}
    proc = subprocess.run(["node", "--test", str(REPO / "sdk" / "typescript" / "test" / "sdk.test.mjs")], capture_output=True,
                          text=True, env=env, timeout=240)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-3000:]
    assert "# fail 0" in out and "# pass 9" in out, out[-1500:]
