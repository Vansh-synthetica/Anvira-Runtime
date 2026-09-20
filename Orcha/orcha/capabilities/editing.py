"""Robust file editing, adapted from opencode's ``edit`` and ``patch`` tools.

Local models are unreliable at reproducing text character-for-character: they emit curly quotes, normalise
dashes, drop trailing spaces, copy the ``12: `` line-number prefix from ``read_file`` output, or flatten CRLF.
An exact-match ``edit_file`` then fails, the model retries blindly, and the run burns iterations. This module
gives ``edit_file`` a strategy ladder (exact -> Unicode-normalised -> trailing-whitespace-tolerant lines ->
line-number-prefix-stripped), preserves the file's line endings, and explains failures with the closest line
so the model can self-correct. It also implements opencode's multi-file ``apply_patch`` envelope
(``*** Begin Patch`` / ``Add`` / ``Update`` / ``Delete`` / ``Move to``), applied atomically with rollback.

Behavioural compatibility: for an EXACT match ORCHA's documented semantics are unchanged (first occurrence unless
``replace_all``); the result now says how many other occurrences remain. Fuzzy strategies are guesses, so they
must match exactly one place or the edit is refused.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

_UNICODE_MAP = {
    **{ord(c): "'" for c in "‘’‚‛"},
    **{ord(c): '"' for c in "“”„‟"},
    **{ord(c): "-" for c in "‐‑‒–—―−"},
    **{ord(c): " " for c in "            　"},
}
_PREFIX_RE = re.compile(r"^[ \t]*\d{1,7}[:|→\t] ?")


class EditError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def normalize_for_match(s: str) -> str:
    """One-to-one character mapping, so offsets in the normalised text are offsets in the original."""
    return s.translate(_UNICODE_MAP)


def _find_all(hay: str, needle: str) -> List[Tuple[int, int]]:
    out, i = [], 0
    while needle and (i := hay.find(needle, i)) != -1:
        out.append((i, i + len(needle)))
        i += len(needle)
    return out


def _find_line_matches(content: str, search: str) -> List[Tuple[int, int]]:
    """Whole-line matches ignoring trailing whitespace and Unicode punctuation variants."""
    trailing_nl = search.endswith("\n")
    expected = search.split("\n")
    if trailing_nl:
        expected.pop()
    lines, pos = [], 0
    for raw in content.splitlines(keepends=True):
        text = raw[:-1] if raw.endswith("\n") else raw
        content_end = pos + len(text) - (1 if text.endswith("\r") else 0)
        lines.append((pos, pos + len(raw), text, content_end, raw.endswith("\n")))
        pos += len(raw)
    want = [normalize_for_match(e.rstrip()) for e in expected]
    found: List[Tuple[int, int]] = []
    for i in range(len(lines) - len(want) + 1):
        window = lines[i:i + len(want)]
        if all(normalize_for_match(w[2].rstrip()) == want[k] for k, w in enumerate(window)):
            last = window[-1]
            if trailing_nl and not last[4]:
                continue
            span = (window[0][0], last[1] if trailing_nl else last[3])
            if not found or span[0] >= found[-1][1]:
                found.append(span)
    return found


def strip_line_numbers(s: str) -> Optional[str]:
    """If EVERY non-empty line starts with a ``12: `` style prefix (copied from read output), remove it."""
    lines = s.split("\n")
    body = [l for l in lines if l.strip()]
    if not body or not all(_PREFIX_RE.match(l) for l in body):
        return None
    return "\n".join(_PREFIX_RE.sub("", l, count=1) for l in lines)


def closest_line_hint(text: str, old: str) -> str:
    """Point at the most similar line in the file so the model can fix its old_string."""
    first = next((l.strip() for l in old.split("\n") if l.strip()), "")
    if not first:
        return ""
    candidates = {l.strip(): n for n, l in enumerate(text.split("\n"), 1) if l.strip()}
    best = difflib.get_close_matches(first, list(candidates), n=1, cutoff=0.6)
    return f" Closest line in the file is {candidates[best[0]]}: {best[0][:100]!r}." if best else ""


@dataclass
class Replaced:
    text: str
    count: int                 # replacements performed
    strategy: str              # exact | unicode | trailing-whitespace | line-numbers
    remaining: int = 0         # other exact occurrences left untouched


def replace(text: str, old: str, new: str, replace_all: bool = False, *, label: str = "the file") -> Replaced:
    """Apply one edit with the strategy ladder. Raises :class:`EditError` with a model-actionable message."""
    if old == new:
        raise EditError("no_change", "No changes to apply: old_string and new_string are identical.")
    if old == "":
        raise EditError("empty_old_string", "old_string must not be empty. Use write_file to create or overwrite a file, "
                                            "or append_file to add text.")
    ending = "\r\n" if "\r\n" in text else "\n"
    old_n = old.replace("\r\n", "\n").replace("\n", ending)
    new_n = new.replace("\r\n", "\n").replace("\n", ending)

    def apply(spans: Sequence[Tuple[int, int]], replacement: str) -> str:
        out = text
        for s, e in reversed(list(spans)):
            out = out[:s] + replacement + out[e:]
        return out

    exact = _find_all(text, old_n)
    if exact:
        chosen = exact if replace_all else exact[:1]
        return Replaced(apply(chosen, new_n), len(chosen), "exact", 0 if replace_all else len(exact) - 1)

    ladder: List[Tuple[str, Callable[[], Tuple[List[Tuple[int, int]], str]]]] = [
        ("unicode", lambda: (_find_all(normalize_for_match(text), normalize_for_match(old_n)), new_n)),
        ("trailing-whitespace", lambda: (_find_line_matches(text, old_n), new_n)),
    ]
    stripped = strip_line_numbers(old)
    if stripped is not None:
        s_old = stripped.replace("\r\n", "\n").replace("\n", ending)
        s_new = (strip_line_numbers(new) or new).replace("\r\n", "\n").replace("\n", ending)
        ladder.append(("line-numbers", lambda: (_find_all(text, s_old) or _find_line_matches(text, s_old), s_new)))
    for name, finder in ladder:
        spans, replacement = finder()
        if not spans:
            continue
        if len(spans) > 1 and not replace_all:
            raise EditError("ambiguous_match",
                            f"old_string only matches {label} after normalising whitespace/punctuation, and then it matches "
                            f"{len(spans)} places. Add more surrounding lines to make it unique, or set replace_all=true.")
        chosen = spans if replace_all else spans[:1]
        return Replaced(apply(chosen, replacement), len(chosen), name)
    raise EditError("old_string_not_found",
                    f"old_string not found in {label}. It must match the file including whitespace and indentation; "
                    f"do not include line-number prefixes.{closest_line_hint(text, old)} Use read_file to copy the exact text.")


# ─────────────────────────────── apply_patch (opencode's multi-file envelope) ───────────────────────────────
@dataclass
class Chunk:
    old: List[str] = field(default_factory=list)
    new: List[str] = field(default_factory=list)
    context: Optional[str] = None
    end_of_file: bool = False


@dataclass
class Hunk:
    kind: str                       # add | delete | update
    path: str
    contents: str = ""
    move_to: Optional[str] = None
    chunks: List[Chunk] = field(default_factory=list)


class PatchError(EditError):
    pass


_BOUNDARY = ("*** End Patch", "*** Add File: ", "*** Delete File: ", "*** Update File: ")


def _is_boundary(line: str) -> bool:
    return line == "*** End Patch" or line.startswith(_BOUNDARY[1:])


def parse_patch(patch_text: str) -> List[Hunk]:
    text = patch_text.strip()
    m = re.match(r"^(?:cat\s*<<\s*['\"]?(\w+)['\"]?\s*\n)", text)      # tolerate a shell heredoc wrapper
    if m:
        text = text[m.end():]
        text = re.sub(r"\n" + re.escape(m.group(1)) + r"\s*$", "", text)
    lines = [l[:-1] if l.endswith("\r") else l for l in text.split("\n")]
    if not lines or lines[0].strip() != "*** Begin Patch":
        raise PatchError("invalid_patch", "The first line of the patch must be '*** Begin Patch'.")
    if lines[-1].strip() != "*** End Patch":
        raise PatchError("invalid_patch", "The last line of the patch must be '*** End Patch'.")
    end = len(lines) - 1
    hunks: List[Hunk] = []
    i = 1
    while i < end:
        header = lines[i].strip()
        if header.startswith("*** Add File: "):
            path, body, i = header[len("*** Add File: "):].strip(), [], i + 1
            while i < end and not _is_boundary(lines[i].strip()):
                if not lines[i].startswith("+"):
                    raise PatchError("invalid_patch", f"Invalid Add File line for '{path}' (line {i + 1}): expected '+', got '{lines[i][:60]}'.")
                body.append(lines[i][1:])
                i += 1
            hunks.append(Hunk("add", path, contents="\n".join(body) + ("\n" if body else "")))
        elif header.startswith("*** Delete File: "):
            path = header[len("*** Delete File: "):].strip()
            i += 1
            if i < end and not _is_boundary(lines[i].strip()):
                raise PatchError("invalid_patch", f"Delete hunks have no body (line {i + 1}: '{lines[i][:60]}').")
            hunks.append(Hunk("delete", path))
        elif header.startswith("*** Update File: "):
            path, i = header[len("*** Update File: "):].strip(), i + 1
            move = None
            if i < end and lines[i].strip().startswith("*** Move to:"):
                move = lines[i].strip()[len("*** Move to:"):].strip()
                if not move:
                    raise PatchError("invalid_patch", f"Move destination for '{path}' must not be empty.")
                i += 1
            chunks: List[Chunk] = []
            while i < end and not _is_boundary(lines[i].strip()):
                raw = lines[i]
                s = raw.rstrip()
                if s == "*** End of File":
                    if chunks:
                        chunks[-1].end_of_file = True
                    i += 1
                    continue
                if s == "@@" or s.startswith("@@ "):
                    chunks.append(Chunk(context=None if s == "@@" else s[3:]))
                    i += 1
                    continue
                if not chunks:
                    chunks.append(Chunk())
                c = chunks[-1]
                if raw == "":
                    c.old.append(""); c.new.append("")
                elif raw[0] == " ":
                    c.old.append(raw[1:]); c.new.append(raw[1:])
                elif raw[0] == "-":
                    c.old.append(raw[1:])
                elif raw[0] == "+":
                    c.new.append(raw[1:])
                else:
                    raise PatchError("invalid_patch", f"Update hunk line {i + 1} must start with ' ', '+' or '-': '{raw[:60]}'.")
                i += 1
            if not chunks or all(not c.old and not c.new for c in chunks):
                raise PatchError("invalid_patch", f"Update hunk for '{path}' contains no changes.")
            hunks.append(Hunk("update", path, move_to=move, chunks=chunks))
        else:
            raise PatchError("invalid_patch", f"'{header[:60]}' (line {i + 1}) is not a valid hunk header. Use "
                                              "'*** Add File: <path>', '*** Delete File: <path>' or '*** Update File: <path>'.")
    if not hunks:
        raise PatchError("invalid_patch", "Empty patch.")
    return hunks


def _seek(lines: Sequence[str], pattern: Sequence[str], start: int, eof: bool = False) -> int:
    if not pattern:
        return start
    n = len(pattern)
    key_fns = (lambda s: s, lambda s: s.rstrip(), lambda s: s.strip(), lambda s: normalize_for_match(s.strip()))
    for key in key_fns:
        want = [key(p) for p in pattern]
        if eof:
            at = len(lines) - n
            if at >= start and [key(x) for x in lines[at:at + n]] == want:
                return at
        for i in range(start, len(lines) - n + 1):
            if [key(x) for x in lines[i:i + n]] == want:
                return i
    return -1


def derive_update(path: str, chunks: Sequence[Chunk], original: str) -> str:
    """New file text after applying an Update hunk (context-anchored, whitespace-tolerant)."""
    ending = "\r\n" if "\r\n" in original else "\n"
    lines = original.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    edits: List[Tuple[int, int, List[str]]] = []
    cursor = 0
    for ch in chunks:
        if ch.context:
            at = _seek(lines, [ch.context], cursor)
            if at == -1:
                raise PatchError("patch_context_not_found", f"Failed to find context '{ch.context}' in {path}.")
            cursor = at + 1
        if not ch.old:
            edits.append((len(lines), 0, ch.new))
            continue
        old, new = list(ch.old), list(ch.new)
        at = _seek(lines, old, cursor, ch.end_of_file)
        if at == -1 and old and old[-1] == "":
            old = old[:-1]
            new = new[:-1] if new and new[-1] == "" else new
            at = _seek(lines, old, cursor, ch.end_of_file)
        if at == -1:
            raise PatchError("patch_hunk_not_found", f"Failed to find the expected lines in {path}:\n" + "\n".join(old[:6]))
        edits.append((at, len(old), new))
        cursor = at + len(old)
    for start, remove, insert in sorted(edits, key=lambda e: e[0], reverse=True):
        lines[start:start + remove] = insert
    return ending.join(lines) + ending
