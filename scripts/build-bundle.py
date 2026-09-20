"""Build the Anvira Runtime release bundle (a .zip the installer can consume).

    python scripts/build-bundle.py [--out dist/anvira-runtime-1.0.0.zip]

The bundle holds exactly what the installer copies or pip-installs: ``runtime/``, ``sdk/python/``, and the
infrastructure source trees ``Orcha/``, ``nomi/``, ``AICL/`` (without virtualenvs, build output, vendored
research trees, tests, docs or caches). It contains no models.
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sdk" / "python"))
from anvira_client.bootstrap import SERVICE_DIRS, _SKIP_DIRS, _SKIP_SUFFIXES  # noqa: E402

TREES = {"runtime": "runtime", "sdk/python": "sdk/python", **{k: k for k in SERVICE_DIRS}}


def wanted(path: Path) -> bool:
    parts = set(path.parts)
    return not (parts & _SKIP_DIRS or path.name.endswith(_SKIP_SUFFIXES) or any(p.endswith(".egg-info") for p in path.parts))


def main() -> int:
    version = re.search(r'RUNTIME_VERSION\s*=\s*"([^"]+)"', (REPO / "runtime/anvira_runtime/version.py").read_text()).group(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "dist" / f"anvira-runtime-{version}.zip"))
    out = Path(ap.parse_args().out)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for tree in TREES:
            root = REPO / tree
            for f in sorted(root.rglob("*")):
                rel = f.relative_to(root)
                if f.is_file() and wanted(rel):
                    z.write(f, f"{tree}/{rel.as_posix()}")
                    count += 1
        z.writestr("README.txt", f"Anvira Runtime {version} bundle. Install: anvira runtime install --source <this zip>\n")
    print(f"{out}  ({count} files, {out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
