# ORCHA editing improvements (from opencode)

Ideas taken from the opencode repo, adapted to ORCHA's **edge / small-model design**: long-horizon task planner, tool-name
aliases, argument aliases, and tolerance of wrong tool calls. Nothing was replaced; everything is additive.

## `edit_file` — a matching ladder (`orcha/capabilities/editing.py`)

Small models rarely reproduce a snippet byte-for-byte. `old_string` is tried against progressively looser strategies:

1. exact match — keeps ORCHA's documented **first-occurrence** semantics and now reports how many further occurrences remain;
2. Unicode-normalised (smart quotes/dashes, NBSP);
3. trailing-whitespace-tolerant, line by line;
4. line-number-prefix stripped (`  12→` / `12: ` copied from a file view).

Fuzzy matches must be **unique**, otherwise the edit fails with the candidate count. CRLF files stay CRLF. A failed edit returns the
closest line ("did you mean line 41?") so the model can self-correct on the next step instead of looping.

## `apply_patch` — multi-file edits in one call

`*** Begin Patch` / `*** Add File` / `*** Update File` (+ `*** Move to`) / `*** Delete File` / `*** End Patch`. Two-phase: every
hunk is validated first, then written atomically with **rollback** if any write fails. Heredoc wrappers (`<<EOF`) are tolerated.

## Fit with the edge-model machinery

* Tool aliases: `patch`, `apply_diff`, `multi_edit`, `multiedit`, `patch_files` → `apply_patch`.
* Argument aliases: `old_str/oldString/…` → `old_string`, `replaceAll/all/global` → `replace_all`, `patch/patchText/diff/input/…` → `patch_text`.
* Cross-routing in `ToolExecutor`: an edit-style call sent to `apply_patch` is rerouted to `edit_file`, and a patch envelope sent to
  `edit_file` is rerouted to `apply_patch`, so a confused small model still succeeds.
* `apply_patch` is **not** advertised in the compact small-model tool listing (keeps prompts short); it is available to capable
  models and reachable through aliases.
* File tracking, write-tool detection (task executor, agent, memory extraction) and workspace-path checks all recognise `apply_patch`.

Tests: `Orcha/tests/test_editing_opencode.py` (28 tests).
