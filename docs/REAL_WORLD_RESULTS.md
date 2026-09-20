# Real-world results (real models, packaged runtime)

Not unit tests with fakes: the **packaged** runtime (`AnviraRuntime-1.0.0-win-x64` + GPU pack, own Python, bundled CUDA `llama-server`) started on demand by the SDK exactly as an
app would, driving real GGUF models on an RTX 3050 Laptop (4 GB VRAM), i5-12500H, 16 GB RAM. Script: `examples/real_workflows.py`.

## Workflow matrix (21 checks, identical tasks and prompts)

| model | Notes (8) | Study (6) | security (1) | Dev: create module + tests | Dev: fix a bug | Dev: rename across 3 files | total |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder **7B** q4_k_m (GPU+CPU, 20 layers) | pass | pass | pass | pass (138 s) | pass (26 s: read, edit, run tests) | pass (26 s) | **21/21** |
| Qwen2.5-Coder **3B** q4_k_m (GPU, 36 layers) | pass | pass | pass | fail | fail | pass (13 s) | **18/21** |
| Qwen2.5-Coder **1.5B** q4_k_m (GPU) | pass | pass | pass | fail | fail | fail | **17/21** |

(The 3B/1.5B rows were measured before the project-scaffold path existed; "create module + tests" is now handled by it.)

* **Notes**: index a notebook, register it as a private resource, grounded answers that cite the notes, an unanswerable question is refused ("I could not find that in your notes") thanks to `context.strict`, memory recall reaches the model. Answers took 0.3-0.6 s on the 3B.
* **Study**: cannot see the notebook until it asks and the owner approves; an unrelated app sees nothing; flashcards come back as valid JSON; the owner's audit log shows request, approval and read.
* **Security**: an app cannot give an agent `terminal` until the user grants `orcha.exec`.

## The Snake project (the request in the master prompt: Flask + HTML + CSS + JS, four files)

Same prompt, real 3B, fresh workspace each time; the result was checked by running `game.js` against a fake browser (Start begins a loop, it draws, keys/touch/buttons do not crash).

| stage | result |
|---|---|
| Before the scaffold path (general planner) | 17 overlapping steps, 4 finished, stub files (`game.js` 249 bytes), `index.html` missing |
| Scaffold, static checks only | 4 of 4 files every time, but `game.js` had a crash in 3 of 3 runs (e.g. an undeclared variable) |
| + fake-browser execution, targeted repair, patches, best-attempt | **3B: clean in 2 of 3 runs** (73 s and 42 s), the third named its remaining defect honestly; **7B: clean** (231 s) |

## What broke on the way (and is fixed)

`DETACHED_PROCESS` made every child open a blank console window; `nvidia_gpu()` crashed on any NVIDIA machine; an update could be revived by an open app mid-swap (update lock); the agent had no tools (default capabilities);
the intent router made "read, find the bug, fix it" read-only; ORCHA's alias table was bypassed on the native path; `edit_file` did not count as writing code; empty `run_command` calls sent the model rewriting working files;
my own test harness once flagged a correct 7B game (a "Start" regex matched "restart") - the harness is now tested against that game.

## Honest limits

* A 3B model is **not** reliable for agent work: it fails bug-fix and some create-from-spec tasks and the Snake build about a third of the time. Use 7B+ for agent work; 1.5B-3B are good for grounded Notes/Study.
* Passing the fake browser means "runs, draws, does not crash" - not "plays well" or "looks good". A person still has to play the game.
* Nomi recall is lexical, not semantic.
* Only Windows x64 is packaged; macOS/Linux run from source. The native Rust AICL core is not built (the in-process Python codec is used). The frontend `src/` cutover is not done.
* Numbers come from single machines and 1-3 runs per cell; small models are non-deterministic, so treat them as indicative.
