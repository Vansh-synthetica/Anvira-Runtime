"""
opencode-derived editing for ORCHA: a strategy ladder for edit_file, the multi-file apply_patch tool, and the
edge-model tolerance around them (tool-name aliases, argument aliases, edit<->patch cross-routing, file tracking).
"""
import json
import os

import pytest

from orcha.capabilities import editing
from orcha.capabilities.base import CapabilityContext, ToolExecutor
from orcha.capabilities.filesystem import build_tools as build_fs_tools
from orcha.capabilities.tool_aliases import resolve_tool_name
from orcha.core.packets import _paths_in_tool_args


@pytest.fixture
def ex(tmp_path):
    return ToolExecutor(build_fs_tools(CapabilityContext(roots=[str(tmp_path)])))


def write(tmp_path, name, text):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text.encode("utf-8"))
    return p


def read(p):
    return p.read_bytes().decode("utf-8")


# ---------------------------------------------------------------- strategy ladder
def test_exact_match_keeps_first_occurrence_semantics_and_reports_the_rest():
    r = editing.replace("foo bar foo", "foo", "baz")
    assert r.text == "baz bar foo" and r.count == 1 and r.strategy == "exact" and r.remaining == 1
    assert editing.replace("foo foo", "foo", "x", replace_all=True).text == "x x"


def test_smart_quotes_and_dashes_from_a_small_model_still_match():
    text = 'print("it\'s a test - ok")\n'
    r = editing.replace(text, "print(“it’s a test – ok”)", 'print("done")')
    assert r.strategy == "unicode" and r.text == 'print("done")\n'


def test_trailing_whitespace_differences_are_tolerated_per_line():
    text = "def f():   \n    return 1  \n\nx = 2\n"
    r = editing.replace(text, "def f():\n    return 1\n", "def f():\n    return 2\n")
    assert r.strategy == "trailing-whitespace" and r.text.startswith("def f():\n    return 2\n") and "x = 2" in r.text


def test_line_number_prefixes_copied_from_read_output_are_stripped():
    text = "alpha\nbeta\ngamma\n"
    r = editing.replace(text, "2: beta\n3: gamma", "2: BETA\n3: GAMMA")
    assert r.strategy == "line-numbers" and r.text == "alpha\nBETA\nGAMMA\n"


def test_crlf_files_keep_their_line_endings():
    text = "one\r\ntwo\r\nthree\r\n"
    r = editing.replace(text, "one\ntwo", "ONE\nTWO")            # the model sends LF, the file is CRLF
    assert r.text == "ONE\r\nTWO\r\nthree\r\n"


def test_fuzzy_matches_must_be_unique_but_exact_ones_are_not_second_guessed():
    exact = editing.replace("a = 1\nb = 2\na = 1\n", "a = 1", "a = 9")
    assert exact.count == 1 and exact.remaining == 1                    # documented ORCHA behaviour: first occurrence
    text = "x = “q”\ny = 2\nx = “q”\n"
    with pytest.raises(editing.EditError) as e:
        editing.replace(text, 'x = "q"', 'x = "z"')                       # a guess that hits twice is refused
    assert e.value.code == "ambiguous_match" and "replace_all" in e.value.message
    assert editing.replace(text, 'x = "q"', 'x = "z"', replace_all=True).count == 2


def test_errors_are_actionable_for_a_weak_model():
    with pytest.raises(editing.EditError) as e:
        editing.replace("def compute_total(items):\n    pass\n", "def compute_totl(items):", "x")
    assert e.value.code == "old_string_not_found" and "Closest line in the file is 1" in e.value.message
    for old, new, code in (("a", "a", "no_change"), ("", "x", "empty_old_string")):
        with pytest.raises(editing.EditError) as e:
            editing.replace("a", old, new)
        assert e.value.code == code


# ---------------------------------------------------------------- edit_file tool
def test_edit_file_tool_uses_the_ladder_and_says_so(ex, tmp_path):
    p = write(tmp_path, "a.py", "name = “bob”   \nage = 3\n")
    r = ex.invoke("edit_file", path="a.py", old_string='name = "bob"', new_string='name = "amy"')
    assert r.ok and "unicode normalisation" in str(r.value)
    assert read(p).startswith('name = "amy"')


def test_edit_file_reports_remaining_occurrences(ex, tmp_path):
    write(tmp_path, "b.txt", "x x x")
    r = ex.invoke("edit_file", path="b.txt", old_string="x", new_string="y")
    assert r.ok and "2 other occurrence" in str(r.value)


def test_edit_file_not_found_error_carries_the_hint(ex, tmp_path):
    write(tmp_path, "c.py", "def hello_world():\n    pass\n")
    r = ex.invoke("edit_file", path="c.py", old_string="def hello_wrld():", new_string="x")
    assert not r.ok and r.error["code"] == "old_string_not_found" and "Closest line" in r.error["message"]


# ---------------------------------------------------------------- apply_patch
PATCH = """*** Begin Patch
*** Add File: docs/new.md
+# Title
+hello
*** Update File: src/app.py
@@ def greet():
-    print("Hi")
+    print("Hello, world!")
*** Delete File: old.txt
*** End Patch"""


def test_apply_patch_add_update_delete_in_one_step(ex, tmp_path):
    app = write(tmp_path, "src/app.py", 'def greet():\n    print("Hi")\n\nprint(greet())\n')
    old = write(tmp_path, "old.txt", "bye\n")
    r = ex.invoke("apply_patch", patch_text=PATCH)
    assert r.ok, r
    assert read(tmp_path / "docs" / "new.md") == "# Title\nhello\n"
    assert 'print("Hello, world!")' in read(app) and "print(greet())" in read(app)
    assert not old.exists()


def test_apply_patch_move_and_context_tolerance(ex, tmp_path):
    src = write(tmp_path, "a.py", "import os  \nx = “q”\n")
    r = ex.invoke("apply_patch", patch_text="*** Begin Patch\n*** Update File: a.py\n*** Move to: pkg/b.py\n@@\n import os\n-x = \"q\"\n+x = \"z\"\n*** End Patch")
    assert r.ok, r
    assert not src.exists() and read(tmp_path / "pkg" / "b.py") == 'import os  \nx = "z"\n'.replace("import os  ", "import os")


def test_apply_patch_is_atomic_validation_failure_writes_nothing(ex, tmp_path):
    a = write(tmp_path, "a.txt", "one\n")
    bad = "*** Begin Patch\n*** Update File: a.txt\n@@\n-one\n+two\n*** Update File: missing.txt\n@@\n-x\n+y\n*** End Patch"
    r = ex.invoke("apply_patch", patch_text=bad)
    assert not r.ok and read(a) == "one\n"
    bad2 = "*** Begin Patch\n*** Update File: a.txt\n@@\n-nothere\n+two\n*** End Patch"
    r = ex.invoke("apply_patch", patch_text=bad2)
    assert not r.ok and read(a) == "one\n" and "not applied" in json.dumps(r.__dict__, default=str)


def test_apply_patch_rolls_back_when_a_write_fails_midway(ex, tmp_path, monkeypatch):
    from orcha.capabilities import filesystem as fs
    a = write(tmp_path, "a.txt", "one\n")
    real = fs._atomic_write
    calls = {"n": 0}

    def flaky(fp, content):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real(fp, content)
    monkeypatch.setattr(fs, "_atomic_write", flaky)
    patch = "*** Begin Patch\n*** Update File: a.txt\n@@\n-one\n+two\n*** Add File: b.txt\n+new\n*** End Patch"
    r = ex.invoke("apply_patch", patch_text=patch)
    assert not r.ok and "rolled back" in json.dumps(r.__dict__, default=str)
    assert read(a) == "one\n" and not (tmp_path / "b.txt").exists()


@pytest.mark.parametrize("bad", ["", "no envelope", "*** Begin Patch\n*** End Patch",
                                 "*** Begin Patch\n*** Add File: x\nnot plus\n*** End Patch",
                                 "*** Begin Patch\n*** Frobnicate: x\n*** End Patch"])
def test_apply_patch_rejects_malformed_patches_with_guidance(ex, bad):
    r = ex.invoke("apply_patch", patch_text=bad)
    assert not r.ok


def test_apply_patch_refuses_to_add_over_an_existing_file_and_stays_in_root(ex, tmp_path):
    write(tmp_path, "e.txt", "x\n")
    assert not ex.invoke("apply_patch", patch_text="*** Begin Patch\n*** Add File: e.txt\n+y\n*** End Patch").ok
    r = ex.invoke("apply_patch", patch_text="*** Begin Patch\n*** Add File: ../escape.txt\n+y\n*** End Patch")
    assert not r.ok and not (tmp_path.parent / "escape.txt").exists()


def test_heredoc_wrapped_patch_is_accepted():
    hunks = editing.parse_patch("cat <<'EOF'\n*** Begin Patch\n*** Add File: a.txt\n+hi\n*** End Patch\nEOF")
    assert hunks[0].kind == "add" and hunks[0].contents == "hi\n"


# ------------------------------------------- edge-model tolerance (the ORCHA way)
def test_wrong_tool_names_resolve_to_the_right_tool():
    real = {"apply_patch", "edit_file", "write_file"}
    for wrong, expected in (("patch", "apply_patch"), ("apply_diff", "apply_patch"), ("multi_edit", "apply_patch"),
                            ("patch_file", "edit_file"), ("str_replace", "edit_file"), ("apply_patchs", "apply_patch")):
        assert resolve_tool_name(wrong, real) == expected, wrong


def test_wrong_argument_names_are_mapped(ex, tmp_path):
    p = write(tmp_path, "m.txt", "hello world\n")
    assert ex.invoke("edit_file", file_path="m.txt", oldString="hello", newString="bye").ok            # opencode-style camelCase
    assert ex.invoke("edit_file", filename="m.txt", old_str="bye", new_str="hi").ok
    assert read(p) == "hi world\n"
    q = ex.invoke("apply_patch", patchText="*** Begin Patch\n*** Add File: z.txt\n+z\n*** End Patch")
    assert q.ok and read(tmp_path / "z.txt") == "z\n"
    assert ex.invoke("patch", diff="*** Begin Patch\n*** Add File: y.txt\n+y\n*** End Patch").ok      # wrong name AND wrong arg name


def test_edit_style_call_sent_to_apply_patch_runs_as_edit_file(ex, tmp_path):
    p = write(tmp_path, "r.txt", "abc\n")
    r = ex.invoke("apply_patch", path="r.txt", old_string="abc", new_string="xyz")
    assert r.ok and read(p) == "xyz\n"


def test_patch_envelope_sent_to_edit_file_runs_as_apply_patch(ex, tmp_path):
    r = ex.invoke("edit_file", path="q.txt", old_string="*** Begin Patch\n*** Add File: q.txt\n+q\n*** End Patch", new_string="")
    r2 = ex.invoke("edit_file", patch="*** Begin Patch\n*** Add File: q2.txt\n+q\n*** End Patch")
    assert r2.ok and read(tmp_path / "q2.txt") == "q\n"


def test_apply_patch_is_a_write_tool_so_read_only_policies_still_block_it(tmp_path):
    specs = {s.name: s for s in build_fs_tools(CapabilityContext(roots=[str(tmp_path)]))}
    assert specs["apply_patch"].permissions == specs["edit_file"].permissions
    assert not specs["apply_patch"].is_read_only()


def test_files_touched_by_apply_patch_are_tracked_for_the_progress_report():
    args = json.dumps({"patch_text": PATCH})
    assert _paths_in_tool_args(args) == ["docs/new.md", "src/app.py", "old.txt"]
    assert _paths_in_tool_args('{"path": "x.py"}') == ["x.py"]


def test_compact_small_model_tool_listing_does_not_advertise_apply_patch():
    from orcha.agent_runtime import tools as t
    import inspect
    src = inspect.getsource(t)
    assert '"edit_file", "append_file"' in src and "apply_patch" not in src
