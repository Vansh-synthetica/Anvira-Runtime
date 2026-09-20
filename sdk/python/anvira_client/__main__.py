"""``python -m anvira_client <detect|install|start|stop>`` — JSON-speaking bootstrap helper.

Used by the TypeScript SDK (and any non-Python host) to detect, install (after the
user agreed), start and stop the shared runtime. Always prints one JSON object.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import bootstrap
from .discovery import probe
from .errors import AnviraError


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m anvira_client")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("detect")
    i = sub.add_parser("install")
    i.add_argument("--source")
    i.add_argument("--force", action="store_true")
    sub.add_parser("start")
    sub.add_parser("stop")
    args = p.parse_args(argv)
    try:
        if args.cmd == "detect":
            out = probe().as_dict()
        elif args.cmd == "install":
            out = bootstrap.install_runtime(source=args.source, force=args.force,
                                            on_status=lambda m: print(m, file=sys.stderr))
        elif args.cmd == "start":
            out = bootstrap.start_runtime(on_status=lambda m: print(m, file=sys.stderr)).as_dict()
        else:
            out = {"stopped": bootstrap.stop_runtime()}
        print(json.dumps(out, default=str))
        return 0
    except AnviraError as exc:
        print(json.dumps({"error": {"code": exc.code, "message": exc.message, "hint": exc.hint}}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
