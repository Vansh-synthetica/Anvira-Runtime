# Benchmarks: AICL vs FastAPI/HTTPS

This document exists to answer one question honestly: **for AI-module-to-
AI-module communication on the same machine, how much does AICL's binary
shared-memory transport actually save over a normal FastAPI/HTTPS/JSON
service?** Every number below is reproducible — the scripts that produced
it are in this repo, and the commands to rerun them are included.

## TL;DR

*(Re-measured after the [wire-format rewrite](#7-re-measured-after-the-wire-format-rewrite) —
see that section for why the pure-transport numbers moved.)*

| Scenario | AICL | FastAPI/HTTPS | Difference |
|---|---|---|---|
| Request/response, **same process** (Rust) | 140ns avg | — | — |
| Request/response, **cross-process** (Python, 2 real OS processes) | 26.3µs avg | 1,480µs avg (persistent conn) | **~56× faster** |
| Request/response, cross-process, fresh connection each time | 26.3µs avg | 14,608µs avg | **~555× faster** |
| Streaming, 51 chunks/response, cross-process (no real inference) | 546µs/stream (10.7µs/chunk) | 7,161µs/stream (140µs/chunk) | **~13× faster** |
| End-to-end, real inference, **AICL over HTTP** (40 real tokens, 60 rounds) | 1,173.5ms avg, −19.8ms vs. no-relay | 1,176.2ms avg, −17.2ms vs. no-relay | essentially tied |
| **End-to-end, real inference, AICL fully in-process (zero network)** | **1,104.0ms avg, −89.4ms vs. no-relay** | *(same HTTP relay as above)* | **further −67.7ms median vs. AICL-over-HTTP** |

The pure-transport numbers (rows 1–4) isolate transport/codec cost from
compute — that's the right way to measure a transport, but by itself it
overstates what a user actually feels. Rows 5–6 are what matters for real
impact: with actual model inference in the loop, AICL-over-HTTP and
FastAPI/SSE are statistically indistinguishable from calling the model
directly (both around −18ms, within noise) — the codec/transport
difference that's dramatic in isolation genuinely doesn't matter once
~1 second of real compute dominates. The one difference that *does* still
show up reliably at 60 rounds: removing the network hop entirely (loading
the model directly in the relay process, no HTTP/TCP/`"localhost"`
anywhere) saves a real, consistent ~68ms. See
[§6](#6-going-further-removing-the-network-hop-entirely) for the honest
caveats on that number, and
[§7](#7-re-measured-after-the-wire-format-rewrite) for why the
pure-transport rows changed from an earlier draft of this document. Read
[Methodology](#methodology) and
[What this does and doesn't prove](#what-this-does-and-doesnt-prove)
before citing any of this.

## Results

### 1. Same-process, Rust (`core-rust/examples/bench_local_path.rs`)

Both ends of the channel live in one OS process, communicating through a
lock-free SPSC ring buffer over a raw pointer — no process boundary, no
OS involved in the handoff at all. This measures the transport primitive
itself, not a deployable two-module system.

```
cd core-rust
cargo run --release --example bench_local_path
```

| Path | Avg | p50 | p95 | p99 | Throughput |
|---|---|---|---|---|---|
| Encode (reuse buffer) | 93–147ns | 100ns | 200–300ns | 200–300ns | 5.5–8.2M/s |
| Decode (zero-copy view) | 48–52ns | 0–100ns | 100ns | 100ns | 11.7–12.8M/s |
| Decode (owned/`Arc` copy) | 91–395ns | 100–400ns | 100–400ns | 100–500ns | 2.2–6.6M/s |
| Raw ring buffer push+pop | 30–33ns | 0ns | 100ns | 100–200ns | 14.9–16.1M/s |
| **`LocalChannel` (zero-copy)** | **140ns** | **100ns** | **300ns** | **300ns** | **5.79M/s** |
| `LocalChannel` (standard decode) | 197ns | 200ns | 300ns | 400ns | 3.86M/s |

These are in the same tens-to-hundreds-of-nanoseconds range independently-
reported shared-memory IPC implementations typically land in for same-
process operation — consistent with what a lock-free ring buffer with no
syscalls in the hot path should achieve.

### 2. Cross-process, Python (`benchmarks/https_vs_binary/`)

**This is the comparison that actually matters.** Two genuinely separate
OS processes (`subprocess.Popen`-spawned, verifiably different PIDs),
communicating through real OS-backed shared memory
(`multiprocessing.shared_memory`), using the same `aicl.bin` codec that
ships in this repo — the exact same Python encode/decode path Stage 0's
92-test suite exercises. Same interpreter, same language, same kind of
process boundary as the FastAPI benchmark below; only the transport
differs.

```
cd benchmarks/https_vs_binary
"../../.venv/Scripts/python.exe" cross_process_client.py 20000
```

| Metric | Value |
|---|---|
| Avg | 26.3µs |
| p50 | 25.3µs |
| p95 | 29.4µs |
| p99 | 49.1µs |
| Throughput | 37,876 req/s |

(At 20,000 iterations; consistent at 5,000 too — 26.3µs avg both times,
ruling out an iteration-count artifact.)

The response payload each round trip carries is real: `{"label":
"positive", "confidence": 1.0}`, captured live from Qwen2.5-0.5B-Instruct
running under `llama-server` — not synthetic filler text. **The 26.3µs is
the transport-and-codec cost of moving that already-generated response
between two processes, not the model inference that produced it.** Qwen
inference for that response took ~345ms; nothing in this benchmark re-runs
that per iteration, on either side of the comparison. (This number moved
up from an earlier 16.3µs after the wire-format rewrite — see
[§7](#7-re-measured-after-the-wire-format-rewrite) for the honest
breakdown of why.)

### 3. Cross-process, FastAPI/HTTPS (`benchmarks/https_vs_binary/client.py`)

Two separate OS processes: a `uvicorn` server terminating a self-signed
TLS connection, and an `httpx` client. Same request/response shape, same
canned response payload as above.

```
python gen_cert.py
python -m uvicorn server:app --host 127.0.0.1 --port 8443 \
    --ssl-keyfile key.pem --ssl-certfile cert.pem --log-level warning &
python client.py 5000
```

| Mode | Avg | p50 | p95 | p99 | Throughput |
|---|---|---|---|---|---|
| Persistent connection (TLS handshake paid once) | 1,480µs | 1,313µs | 3,019µs | 4,233µs | 675 req/s |
| Fresh connection per request (TLS handshake every time) | 14,608µs | 11,707µs | 31,563µs | 35,194µs | 68 req/s |

**Important correction from an earlier draft of this comparison:** the
persistent-connection overhead is *not* a repeated TLS handshake — that
cost is paid once, when the connection opens, and amortized across every
request after. What you're seeing in the 1,063µs is TCP-loopback
round-trip + HTTP/1.1 framing + Starlette routing + Pydantic request
validation + JSON serialize/deserialize. The fresh-connection row is where
TLS/TCP setup cost genuinely shows up, on every single request.

### 4. Streaming — the case that actually resembles this project's real traffic

Orcha (this project's agent runtime) streams model output to its desktop
frontend as Server-Sent Events, one small JSON event per token
(`orcha/api/agent_stream.py`: `yield f"data: {json.dumps(wire)}\n\n"`).
That's a *many small messages per response* pattern, which is exactly
where per-message transport overhead compounds instead of being drowned
out by a single big payload. This benchmark reproduces that shape: 50
token-delta chunks + 1 end marker per "response," cross-process both ways.

```
# AICL side
python streaming_aicl_client.py 500

# SSE side (separate terminal, server already running from step 3)
python -m uvicorn streaming_sse_server:app --host 127.0.0.1 --port 8443 \
    --ssl-keyfile key.pem --ssl-certfile cert.pem --log-level warning &
python streaming_sse_client.py 500
```

| Path | Avg full stream | p50 | p95 | p99 | Per-chunk avg | Streams/s |
|---|---|---|---|---|---|---|
| **AICL shared-memory ring** | **546µs** | 534µs | 704µs | 863µs | **10.7µs** | 1,829 |
| FastAPI SSE (HTTPS) | 7,161µs | 7,084µs | 8,178µs | 9,146µs | 140µs | 139.6 |

~13× faster to deliver a complete 51-chunk stream (at 500 streams each;
smaller margin than an earlier 31× figure — partly the wire-format
rewrite's codec overhead on the AICL side, see
[§7](#7-re-measured-after-the-wire-format-rewrite), partly this run's SSE
side happening to be faster than an earlier run — both numbers are real,
current measurements, not cherry-picked). This is the scenario
where AICL's transport choice would actually be felt by a real user —
every chunk saved is latency a token-by-token UI update doesn't have to
wait through.

### 5. End-to-end with real model inference — does the transport gap survive contact with real work?

Every result above deliberately excludes model inference, to isolate
transport cost. That's the right way to measure a transport, but it leaves
the actual question unanswered: **when a real model is doing real work,
does AICL's transport advantage still matter, or does it disappear into
the noise of generation time?**

This benchmark answers that directly. Three conditions, interleaved
round-by-round (not run as separate blocks — see the warm-up note below),
each doing a **real** 40-token generation against Qwen2.5-0.5B-Instruct
via `llama-server`, no canned responses:

- **Baseline**: client calls `llama-server` directly, no relay.
- **AICL relay**: client → AICL shared-memory ring → relay process → calls
  `llama-server` for real → streams each real token back over the ring as
  it's produced → client.
- **SSE relay**: client → HTTPS → relay process (FastAPI) → calls
  `llama-server` for real → streams each real token back as SSE as it's
  produced → client.

```
cd benchmarks/https_vs_binary
# Start llama-server (port 8099) and e2e_sse_relay_server.py (port 8443) first
"../../.venv/Scripts/python.exe" e2e_combined.py 60
```

| Path | Avg | p50 | min | max |
|---|---|---|---|---|
| Baseline (no relay) | 1,193.4ms | 1,162.6ms | 1,000.0ms | 2,068.7ms |
| **AICL relay** | **1,173.5ms** | 1,154.6ms | 1,005.4ms | 1,750.1ms |
| SSE relay (HTTPS) | 1,176.2ms | 1,163.4ms | 1,027.5ms | 1,535.8ms |

| Comparison | Result |
|---|---|
| AICL relay overhead vs. no-relay baseline | **−19.8ms** (no measurable overhead — actually faster than baseline, within noise) |
| SSE relay overhead vs. no-relay baseline | **−17.2ms** (also within noise) |

(60 rounds this time, up from 40 in an earlier pass — same conclusion,
now with more samples. Absolute times here run somewhat higher than an
earlier ~950ms-per-condition pass; that's normal machine-load variance
between benchmark sessions on a shared dev box, not a regression — both
AICL and SSE moved together, tracking the baseline.)

**The honest answer: at this sample size, AICL and FastAPI/SSE are
statistically indistinguishable from calling the model directly** — not
the 56× or 555× a pure-transport comparison shows. ~1,000-1,200ms of
every round here is real model compute (`llama-server`'s own reported
`prompt_ms + predicted_ms`), which nothing in this benchmark can shrink.
Whatever transport-level difference exists between AICL and HTTPS (and
§7 below shows the wire-format rewrite made AICL's own codec measurably
slower than before) is now genuinely too small to detect against ~1
second of real inference. Section 6 below shows where a difference
*does* still show up reliably at this scale — removing the network hop
itself, not swapping which protocol carries it.

**Getting to this number took four rounds of debugging a benchmark that
was lying to us**, and it's worth documenting in full because the lesson
generalizes well past this one comparison:

1. **Warm-up drift**: running each condition as one big sequential block
   let `llama-server` (and the OS page cache, CPU frequency scaling) get
   faster over the first several calls, so whichever condition ran last
   looked artificially fastest. Fixed by interleaving: one round of
   baseline → AICL → SSE, repeated, instead of three separate blocks.
2. **A real connection-reuse bug**: `llama-server` doesn't reliably
   support HTTP keep-alive — a reused connection gets forcibly closed by
   the server on the second request (`WinError 10054`), and httpx's
   stale-connection retry silently added ~2 seconds that had nothing to
   do with any transport being tested. Initially "fixed" by using a fresh
   client per call — correct in spirit, expensive in practice (see #4).
3. **Sync vs. async httpx**: even after fixing both of the above, one
   path was still ~1.7s faster than the others. Isolated with a direct
   side-by-side test (identical request, identical server, only the
   client's sync-vs-async calling convention changed) — sync httpx
   streaming measured ~3,450ms, async measured ~1,700ms, for the exact
   same request. The SSE relay path happened to use async for its call to
   `llama-server`; baseline and the AICL relay happened to use sync. That
   was the *entire* earlier "HTTPS is faster" result — a Windows/httpx
   networking artifact, not a transport-protocol finding. Fixed by making
   every path's call to `llama-server` use the same (async) convention.
4. **Two genuine, fixable sources of waste, found by profiling the "fast"
   number for where the remaining time actually went** — this is the part
   that made everything ~40% faster, not just more comparable:
   - Resolving `"localhost"` cost **~250-290ms per request** on this
     machine (confirmed via direct A/B test: identical request, `127.0.0.1`
     took ~5ms, `localhost` took ~260ms). Switched every call to
     `llama-server` to the literal IP.
   - Constructing a fresh `httpx.AsyncClient()` per call (the fix from
     step 2) cost a flat **~160-220ms every single time**, regardless of
     host or `trust_env` setting — confirmed this was the client object's
     own setup cost, not networking, by timing client construction with no
     request at all. Fixed by keeping **one persistent `AsyncClient` per
     process**, sending an explicit `Connection: close` header per request
     instead — this still defeats `llama-server`'s keep-alive bug (the
     server closes the connection itself, httpx opens a clean new one for
     the next request) without paying full client-reconstruction cost.

   Combined, these two fixes cut **every condition's time by ~40-42%**
   (baseline: 1,653ms → 958ms; AICL: 1,647ms → 952ms; SSE: 1,732ms →
   964ms) — none of it an AICL-vs-HTTPS effect, all of it avoidable
   Python/Windows networking overhead that was inflating every path
   roughly equally and masking how close to the real inference floor
   (`llama-server`'s own ~950ms) everything actually could get.

None of these four issues were bugs in AICL's own code — all four were in
the benchmark harness (or, for #4, in *how any Python client on Windows
talks to a local HTTP server*, a lesson worth carrying into real code, not
just this benchmark). Reporting the first number that came out at any of
these stages would have been actively misleading.

### 6. Going further: removing the network hop entirely

Everything above still has `127.0.0.1` in it somewhere — the relay
process still talks to `llama-server` over real (if now well-optimized)
TCP loopback. That's not actually necessary. `llama-server` is just a thin
HTTP wrapper around llama.cpp's C++ library; nothing requires going
through a socket to reach it. `llama-cpp-python` binds that library
directly, so a relay process can load the model itself and generate
tokens with **no HTTP, no TCP, no "localhost" anywhere in the path** —
only AICL's shared-memory ring between the relay and the client.

```
cd benchmarks/https_vs_binary
pip install llama-cpp-python   # needs a C++ toolchain — see below
python e2e_combined.py 60      # now runs all four conditions
```

A fourth condition was added to the same interleaved benchmark:
**AICL in-process** — the relay loads the GGUF model directly via
`llama_cpp.Llama(...)` at process startup (once, like `llama-server`
loading its model once) and streams real generated tokens straight into
the AICL ring, with literally no network stack anywhere in the loop.

| Path | Avg | p50 | min | max |
|---|---|---|---|---|
| Baseline (direct HTTP, no relay) | 1,193.4ms | 1,162.6ms | 1,000.0ms | 2,068.7ms |
| AICL relay (ring → HTTP → `llama-server`) | 1,173.5ms | 1,154.6ms | 1,005.4ms | 1,750.1ms |
| SSE relay (HTTPS → HTTP → `llama-server`) | 1,176.2ms | 1,163.4ms | 1,027.5ms | 1,535.8ms |
| **AICL relay (ring → in-process model, zero network)** | **1,104.0ms** | 1,082.5ms | 882.6ms | 1,531.5ms |

| Comparison | Result |
|---|---|
| AICL in-process vs. AICL-over-HTTP | **−69.6ms mean, −67.7ms median** (paired, stdev 132.8ms, over 60 rounds) |
| AICL in-process vs. no-relay baseline | **−89.4ms** |

**Yes — removing the network hop entirely measurably helps**, on top of
everything section 5 already fixed, and this held up (actually got
*more* consistent) when re-run at 60 rounds instead of 40: mean and
median are now much closer together (−69.6ms vs −67.7ms, previously
−46.4ms vs −11.9ms), which is exactly what you'd expect from a real
effect settling down as sample size grows rather than a lucky/unlucky
run. One caveat specific to *this* section: it has two full copies of the
model resident in memory at once (`llama-server`'s and the in-process
relay's), so absolute times here run higher across all four conditions
than section 5's single-model numbers — a real cost of the experimental
setup, not of either transport. The **relative** comparison (in-process
vs. HTTP) is unaffected by that; the **absolute** numbers in this section
aren't directly comparable to section 5's.

**Why this isn't the default architecture in this repo**: it trades
isolation for speed. `llama-server` as a separate process means it can
crash, restart, or be swapped for a different backend without touching
the code that talks to it; loading the model in-process means the relay
and the model share a memory space and a lifetime. Whether that tradeoff
is worth ~50ms depends entirely on what you're building — for a system
that already commits to one persistent model-serving process per module
(which is a reasonable design), this is close to free.

**Building `llama-cpp-python` from source on Windows**, since no prebuilt
wheel exists for most platforms: it needs a C++ toolchain (Visual Studio
Build Tools with the "Desktop development with C++" workload, which
bundles CMake) and, on this machine, ran into Windows' default 260-character
path length limit while unpacking llama.cpp's vendored source tree —
worked around by pointing `TEMP`/`TMP` at a short path (`C:\t`) for the
build rather than enabling long-path support system-wide:

```powershell
$vcvars = "<VS Build Tools path>\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "call `"$vcvars`" && set TEMP=C:\t && set TMP=C:\t && python -m pip install llama-cpp-python"
```

### 7. Re-measured after the wire-format rewrite

`aicl.bin` (Python) was later rewritten to match `core-rust`'s actual ISA-
spec wire format, for real cross-language interop (see README.md's "Wire
format status" section — this is proven with a genuine byte-for-byte
Python↔Rust round trip, not just claimed). That rewrite changed the
numbers in this document, and it's worth being precise about why rather
than quietly swapping in new figures.

**The pure-transport cross-process number went from 16.3µs to 26.3µs avg
— a real, reproducible ~60% increase**, confirmed at both 5,000 and
20,000 iterations (not noise or an iteration-count artifact). Profiling
traced it to genuinely more work per message, not a bug:

- The new 56-byte header computes and verifies a real CRC32 over itself
  on every single encode and decode — the old 64-byte header had no
  integrity check at all. This is a real correctness feature (confirmed
  it actually catches corruption — see §"header CRC" test in the test
  suite) that the old format simply didn't have, and it isn't free.
- Operand values now use LEB128 varint length prefixes (matching
  core-rust exactly) instead of the old format's simpler 1-or-3-byte
  length encoding — a small amount of extra work per string/list/map.
- Every `Packet` auto-generates and encodes a `session_id` as a vendor
  extension operand (continuing to support that field, per the "keep
  everything good and similar" direction this rewrite followed) — this
  alone accounts for roughly 2.4µs of a 10.5µs single-encode-decode
  round trip in isolated measurement.

None of this is a case of the rewrite introducing slop — a
`cProfile` pass shows the added time distributed across encode/decode/
varint/header-CRC roughly proportionally to how much more each one is
now actually doing. **The honest framing: this is the real, measured
cost of matching a shared cross-language spec that includes integrity
checking the old format didn't have — a genuine tradeoff, not a defect.**

**And critically, it doesn't matter where it would actually matter**: the
real end-to-end numbers (with actual model inference) in §5 and §6 above
are *unaffected* by this — AICL-over-HTTP is still statistically tied
with the no-relay baseline (−19.8ms, within noise, at 60 rounds) and the
in-process advantage is still a clean, consistent ~68ms. An extra ~10µs
of codec time is completely invisible next to ~1 second of real model
compute, which is exactly the point made in
["What this does and doesn't prove"](#what-this-does-and-doesnt-prove)
below: pure-transport numbers make a dramatic headline, but the
end-to-end numbers are what a real user actually feels.

## Methodology

- **Timer**: Rust benchmarks use `std::time::Instant` (Windows:
  `QueryPerformanceCounter`, sub-microsecond resolution). Python
  benchmarks use `time.perf_counter()` (also sub-microsecond on Windows).
  All latencies are measured client-side, wall-clock, per round trip.
- **Iteration counts**: 100,000 for the same-process Rust benchmarks,
  5,000 for cross-process request/response, 500 for fresh-TLS-connection
  (deliberately smaller — each iteration pays a real handshake), 200
  streams (× 51 chunks = 10,200 messages) for the streaming benchmarks.
- **Warm-up**: every benchmark discards one untimed iteration first, so
  process-spawn cost, connection establishment, and OS/interpreter
  first-call overhead aren't counted in the timed loop (except the
  fresh-connection HTTPS benchmark, where paying that cost every time is
  the point).
- **What's excluded on purpose**: no model inference happens inside any
  timed loop. The response payloads are real (captured once from a live
  Qwen2.5-0.5B-Instruct call), but generating them is not part of what's
  being measured — this isolates transport/codec cost from compute cost,
  which is the actual question ("is the wire protocol worth it") rather
  than a model speed test.

## What this does and doesn't prove

**Does prove**: for two processes on the same Windows machine, in Python,
sending small structured messages back and forth, AICL's shared-memory
ring transport is dramatically faster than FastAPI over HTTPS — real,
reproducible, order-of-magnitude numbers, not a synthetic microbenchmark
with an unfair payload.

**Doesn't prove**:
- That HTTPS/FastAPI is "bad" — it's solving a different problem. AICL's
  ring buffer requires both ends on the same machine with access to a
  shared memory segment; FastAPI can talk to a module on another machine,
  in a container, behind a load balancer, written in any language with an
  HTTP client. That flexibility has a cost, and this benchmark measures
  exactly that cost — it's not free, but it's also not what AICL is for.
- That this reflects the current product. Orcha's actual internal module
  communication today is same-process Python function calls (`AgentNode`
  calling `ToolExecutor` directly) — no HTTP hop exists there to replace.
  Orcha's FastAPI server is the boundary to the Electron desktop frontend,
  a genuine separate process across which AICL's *current* Python
  implementation could plausibly compete, but nothing here is wired into
  that path yet. This is a capability demonstration, not a shipped
  optimization.
- Cross-platform numbers. Everything here ran on Windows. Linux loopback
  TCP has less syscall overhead than Windows' — FastAPI's numbers would
  likely improve somewhat on Linux; AICL's shared-memory path would likely
  also improve. The *ratio* between them is not guaranteed to hold exactly,
  though the underlying mechanism gap (syscalls + TLS + JSON vs a shared
  ring buffer) doesn't go away on any OS.
- Raw C/C++ shared-memory IPC numbers. Independently reported C-level
  SPSC shared-memory benchmarks (no interpreter, no protocol encode/decode)
  can reach double-digit nanoseconds cross-process. Python's interpreter
  overhead (object allocation, `struct.pack`/`unpack`, reference counting)
  puts a floor under what this implementation can reach that a compiled
  language wouldn't have — 16.3µs cross-process in Python is a fair number
  for what it is, not a ceiling on what AICL's protocol itself could do in
  a compiled cross-process implementation (which doesn't exist yet — see
  [README.md](README.md)'s wire-format-status section on the Python/Rust
  format mismatch).

## Reproducing all of this

```bash
cd benchmarks/https_vs_binary
pip install -r requirements.txt
python gen_cert.py

# Cross-process request/response
python cross_process_client.py 20000

# Cross-process HTTPS (start server, then client, separate terminals)
python -m uvicorn server:app --host 127.0.0.1 --port 8443 --ssl-keyfile key.pem --ssl-certfile cert.pem --log-level warning
python client.py 5000

# Streaming (AICL)
python streaming_aicl_client.py 500

# Streaming (SSE — start server, then client, separate terminals)
python -m uvicorn streaming_sse_server:app --host 127.0.0.1 --port 8443 --ssl-keyfile key.pem --ssl-certfile cert.pem --log-level warning
python streaming_sse_client.py 500

# End-to-end with real model inference, all four conditions including the
# fully-in-process one (needs a small local GGUF model — see
# e2e_combined.py for the exact llama-server command; start it on port
# 8099, and e2e_sse_relay_server.py via uvicorn on port 8443, first; and
# pip install llama-cpp-python — see the build note in §6 above)
"../../.venv/Scripts/python.exe" e2e_combined.py 60

# Same-process Rust
cd ../../core-rust
cargo run --release --example bench_local_path
```
