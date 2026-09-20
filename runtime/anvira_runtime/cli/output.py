"""CLI output: human tables/key-values, ``--json`` mode, colour only on a TTY."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Iterable


def _can_encode(text: str) -> bool:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


class Printer:
    def __init__(self, json_mode: bool = False, quiet: bool = False):
        self.json_mode, self.quiet = json_mode, quiet
        self.color = (sys.stdout.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb")
        if sys.platform == "win32" and self.color:
            os.system("")  # enable ANSI escapes in the Windows console
        uni = _can_encode("●✓✗▲")
        self.sym = {"ok": "✓" if uni else "OK", "fail": "✗" if uni else "X", "warn": "▲" if uni else "!",
                    "info": "•" if uni else "-", "up": "●" if uni else "*"}

    # -- colour -----------------------------------------------------------------
    def c(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    green = lambda self, t: self.c(t, "32")   # noqa: E731
    red = lambda self, t: self.c(t, "31")     # noqa: E731
    yellow = lambda self, t: self.c(t, "33")  # noqa: E731
    dim = lambda self, t: self.c(t, "2")      # noqa: E731
    bold = lambda self, t: self.c(t, "1")     # noqa: E731

    def status_mark(self, status: str) -> str:
        return {"ok": self.green(self.sym["ok"]), "fail": self.red(self.sym["fail"]),
                "warn": self.yellow(self.sym["warn"]), "info": self.dim(self.sym["info"])}.get(status, status)

    # -- emit ---------------------------------------------------------------------
    def emit(self, data: Any, render: Callable[[Any], None] | None = None) -> None:
        """JSON mode prints ``data``; otherwise call ``render(data)``."""
        if self.json_mode:
            print(json.dumps(data, indent=2, default=str))
        elif render:
            render(data)

    def line(self, text: str = "") -> None:
        if not self.json_mode and not self.quiet:
            print(text)

    def err(self, text: str) -> None:
        print(text, file=sys.stderr)

    def progress(self, text: str) -> None:
        """Ephemeral status to stderr (never pollutes JSON on stdout)."""
        if not self.quiet:
            print(text, file=sys.stderr)

    def kv(self, pairs: Iterable[tuple[str, Any]], indent: int = 0) -> None:
        pairs = [(k, v) for k, v in pairs]
        width = max((len(k) for k, _ in pairs), default=0)
        for k, v in pairs:
            print(f"{' ' * indent}{self.dim(k.ljust(width))}  {v}")

    def table(self, headers: list[str], rows: list[list[Any]]) -> None:
        cells = [[str(x) if x is not None else "-" for x in r] for r in rows]
        widths = [max(len(h), *(len(_strip(r[i])) for r in cells)) if cells else len(h) for i, h in enumerate(headers)]
        print("  ".join(self.bold(h.ljust(widths[i])) for i, h in enumerate(headers)))
        for r in cells:
            print("  ".join(r[i] + " " * (widths[i] - len(_strip(r[i]))) for i in range(len(headers))))


def _strip(s: str) -> str:
    import re
    return re.sub(r"\033\[[0-9;]*m", "", s)


def human_size(n: int | float | None) -> str:
    if not n:
        return "-"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return "-"


def human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    return f"{s // 3600}h {(s % 3600) // 60}m"
