# Orcha

**Local-first multi-model orchestration engine.**

Orcha runs as many local LLMs as you want **in parallel**, then uses the
largest one to **synthesize** every answer into a single, refined response —
no cloud API needed, no vendor lock-in.

```
your query
    │
    ▼
Decompose ──► detect domain (finance, code, reasoning, science…)
    │
    ▼
Select ──────► score every local model against the domain
    │
    ▼
Execute ────► llama3:8b ──┐
             mistral:7b ──┤  all run concurrently (asyncio)
             codellama ───┤
             qwen2.5:32b ─┘
    │
    ▼
Synthesize ──► qwen2.5:70b reads every candidate and writes ONE refined answer
    │
    ▼
Evaluate ────► confidence check → retry if below threshold
    │
    ▼
  answer
```

Everything is wired through a single typed `OrchaPacket` message that
every stage reads, writes back to, and stamps with a trace entry — so you
always know exactly what happened and why.

---

## Quickstart

```bash
git clone https://github.com/LocalHouseLLM/orcha01
cd orcha01
pip install -e ".[all]"          # install everything
```

**With mock experts (zero setup, works immediately):**

```python
from orcha import Orchestrator
from orcha.experts.mock import load_mock_experts

orc = Orchestrator(
    experts=load_mock_experts(),
    synthesizer_expert="mock_synthesizer",
    run_all_experts=True,
)
result = orc.run("How should I diversify a small portfolio?")
print(result.answer)
print(result.contributors)   # which models answered
print(result.synthesized)    # True = answer was synthesized by a local model
```

**With real Ollama models:**

```bash
ollama pull llama3:8b
ollama pull mistral:7b
ollama pull qwen2.5:32b     # will become the synthesizer
```

```python
import asyncio
from orcha import Orchestrator, LocalModelRegistry

async def main():
    registry = LocalModelRegistry()
    await registry.discover_ollama()          # finds every pulled model automatically

    orc = Orchestrator(
        experts=registry.build(),
        synthesizer_expert=registry.pick_synthesizer(),   # largest model wins
        run_all_experts=True,
    )
    result = await orc.run_async("Explain quantum entanglement simply.")
    print(result.answer)

asyncio.run(main())
```

**Web UI:**

```bash
uvicorn orcha.api.server:app --reload --port 8420
# open http://localhost:8420
```

> **API versioning:** The stable HTTP API lives under `/v1` (e.g. `POST /v1/query`,
> `GET /v1/health`). Un-versioned paths (`/query`, `/health`) redirect to `/v1` for
> backwards compatibility and are deprecated.

---

## Installation

```bash
pip install -e ".[all]"      # everything (recommended)
pip install -e "."           # core only (no HTTP, no YAML)
pip install -e ".[local]"    # + httpx for Ollama / local servers
pip install -e ".[api]"      # + FastAPI web UI
pip install -e ".[config]"   # + PyYAML for config files
pip install -e ".[dev]"      # + pytest for development
```

Requires Python 3.10+. No cloud API key is ever required.

---

## Adding models

### Ollama (recommended)

```python
from orcha.experts import LocalModelRegistry

registry = LocalModelRegistry()

# Auto-discover everything in `ollama list`
import asyncio
asyncio.run(registry.discover_ollama())

# Or add specific models with metadata
registry.add_ollama("llama3.1:8b",  domain="general")
registry.add_ollama("codellama:13b", domain="code")
registry.add_ollama("llama3.1:70b", synthesizer=True)   # explicit synthesizer

pool  = registry.build()
synth = registry.pick_synthesizer()

orc = Orchestrator(experts=pool, synthesizer_expert=synth, run_all_experts=True)
```

### Any OpenAI-compatible local server

Works with **llama.cpp**, **LM Studio**, **vLLM**, **text-generation-webui**,
**LocalAI**, **Jan**, and anything else that serves
`POST /v1/chat/completions`.

```python
from orcha.experts import LocalChatExpert

expert = LocalChatExpert(
    model="qwen2.5-32b-instruct",
    base_url="http://localhost:8080/v1",   # llama.cpp --port 8080
    domain="reasoning",
    description="32B reasoning model via llama.cpp",
)
orc.register_expert("qwen32b", expert)
```

### Config file (JSON or YAML)

```json
{
  "models": [
    {"backend": "ollama", "model": "llama3:8b",   "domain": "general"},
    {"backend": "ollama", "model": "codellama:13b", "domain": "code"},
    {"backend": "ollama", "model": "llama3:70b",   "domain": "reasoning", "synthesizer": true},
    {
      "backend": "openai_compatible",
      "model": "qwen2.5-32b-instruct",
      "base_url": "http://localhost:8080/v1",
      "domain": "reasoning"
    }
  ]
}
```

```python
registry = LocalModelRegistry.from_config("models.json")
orc = Orchestrator(
    experts=registry.build(),
    synthesizer_expert=registry.pick_synthesizer(),
    run_all_experts=True,
)
```

### Write your own expert

```python
from orcha.experts.base import BaseExpert, ExpertOutput

class MyExpert(BaseExpert):
    name        = "my_model"
    domain      = "finance"           # used for routing
    description = "Custom fine-tuned finance model"

    async def execute(self, query: str) -> ExpertOutput:
        answer = await my_inference_call(query)
        return ExpertOutput(answer=answer, confidence=0.85, tokens_used=200)

orc.register_expert("my_model", MyExpert())
```

---

## How the synthesizer works

When `synthesizer_expert` is configured and at least two models produced
usable answers, the aggregator builds this prompt and hands it to the
synthesizer:

```
You are a synthesis expert. Several specialist models independently answered
the question below. Combine their insights into ONE refined, accurate,
well-organized final answer. Resolve disagreements by favoring the
better-reasoned position. Do not simply concatenate the answers.

Question:
{query}

Candidate answers:
--- Candidate 1 (llama3:8b) ---
{answer from llama3}

--- Candidate 2 (mistral:7b) ---
{answer from mistral}

Final answer:
```

The synthesizer (typically the largest model in your pool) then writes one
clean, reconciled response. If synthesis fails for any reason, the system
falls back to confidence-weighted aggregation automatically.

---

## Architecture

### ORCHA2 — Classic pipeline

| File | Purpose |
|---|---|
| `orcha/core/packets.py` | `OrchaPacket` — the typed message flowing through every stage |
| `orcha/orchestrator.py` | Main loop: decompose → select → execute → aggregate → evaluate → retry |
| `orcha/orchestration/decomposer.py` | Domain detection, subtask decomposition |
| `orcha/orchestration/planner.py` | Budget-aware effort scaling per iteration |
| `orcha/orchestration/selector.py` | Domain-match scoring, `force_all_experts` mode |
| `orcha/orchestration/executor.py` | `asyncio.gather` parallel execution, fault isolation |
| `orcha/orchestration/aggregator.py` | Synthesis prompt + confidence-weighted fallback |
| `orcha/orchestration/evaluator.py` | Quality gate; triggers retry |
| `orcha/orchestration/retry.py` | Retry decision vs budget |
| `orcha/experts/ollama.py` | Ollama backend |
| `orcha/experts/local_chat.py` | OpenAI-compatible local server backend |
| `orcha/experts/registry.py` | `LocalModelRegistry` — discover, configure, pick synthesizer |
| `orcha/experts/mock.py` | Zero-dependency mock experts + `MockSynthesizer` |
| `orcha/api/server.py` | FastAPI server, Ollama auto-discovery on startup (lifespan), versioned `/v1` routes |
| `orcha/observability.py` | Structured logging with trace IDs |
| `orcha/ui/index.html` | Web dashboard |

### ORCHA3 — Graph engine + durability + agents

ORCHA3 adds a full graph execution engine on top of the existing ORCHA2
infrastructure. The `OrchaPacket` is preserved as the single state object
on every edge — no rewrite, pure evolution.

**What's new in v0.4.0:**

```
                    ┌──────────────────────────────────────┐
                    │  orcha.graph (graph execution engine)  │
                    │                                      │
                    │  Graph         ─ topology builder     │
                    │  GraphRuntime  ─ executor             │
                    │  Node / to_node() ─ compute units    │
                    │  Store         ─ checkpoint durability │
                    │  RunContext    ─ per-run ambient       │
                    │  CancelToken   ─ cooperative cancel    │
                    │  EventEmitter ─ typed event stream     │
                    └──────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              orcha.nodes    orcha.builders   orcha.result
              Stage wrappers  Pre-built graphs  RunResult
              VerifyNode      default_graph
              AgentNode       research_graph
              ToolNode        multi_agent_graph
              RetrievalNode
```

| Module | Purpose |
|---|---|
| `orcha/graph/graph.py` | `Graph` — declarative topology (nodes, edges, conditional, scatter/gather) |
| `orcha/graph/runtime.py` | `GraphRuntime` — executes a graph with timeout, retry, checkpointing, events |
| `orcha/graph/node.py` | `Node` ABC, `GatherNode`, `to_node()` adapter for legacy stages |
| `orcha/graph/edge.py` | `Edge`, `ConditionalEdge`, `FanOutEdge`, `FanInEdge`, `END` sentinel |
| `orcha/graph/store.py` | `MemoryStore`, `FileStore`, `SqliteStore` — checkpoint durability backends |
| `orcha/graph/context.py` | `RunContext`, `CancelToken`, `EventEmitter`, `ScatterResult`, trace propagation |
| `orcha/graph/errors.py` | `GraphError` taxonomy — `NodeTimeout`, `NodeFailed`, `BudgetExceeded`, etc. |
| `orcha/nodes/stages.py` | Wraps all 7 ORCHA2 stages as graph nodes (zero porting) |
| `orcha/nodes/verify.py` | `VerifyNode`, `CriticNode`, `FactCheckNode` — quality gates |
| `orcha/nodes/agent.py` | `AgentNode` — autonomous multi-step reasoning with tool use |
| `orcha/nodes/tool.py` | `ToolNode`, `ToolSpec` — invoke external tools as graph nodes |
| `orcha/nodes/retrieval.py` | `RetrievalNode`, `keyword_retriever`, `embedding_retriever` — RAG plugin |
| `orcha/builders/default.py` | `build_default_graph()` — ORCHA2 regression bridge (identical behavior) |
| `orcha/builders/research.py` | `build_research_graph()` — RAG + fact-check pipeline |
| `orcha/builders/multi_agent.py` | `build_multi_agent_graph()` — scatter/gather agent collaboration |
| `orcha/result.py` | `RunResult` — terminal projection of a completed graph run |
| `orcha/api/server.py` | Extended: `/v1/run`, `/v1/run/{id}`, SSE events, resume, replay |

#### Graph API quickstart

```python
import asyncio
from orcha.graph import Graph, GraphRuntime, MemoryStore, END
from orcha.graph.node import to_node

async def my_node(pkt, ctx):
    return pkt.fork(pkt.kind, result="done")

g = Graph(name="demo")
g.add_node(to_node(my_node, name="worker"), entry=True)
g.add_edge("worker", END)
g.validate()

rt = GraphRuntime(g, store=MemoryStore())
result = await rt.run("hello world")
print(result.answer)
```

#### Pre-built graphs (zero wiring)

```python
from orcha.builders import build_default_graph, build_research_graph
from orcha.graph import GraphRuntime

# Same behavior as ORCHA2 Orchestrator, but on the graph engine.
graph = build_default_graph(experts=my_experts, synthesizer="llama3:70b")
rt = GraphRuntime(graph)
result = await rt.run("What causes climate change?")
```

#### Fan-out / fan-in (parallel branches)

```python
from orcha.graph import Graph, GraphRuntime, END
from orcha.graph.node import Node, GatherNode
from orcha.graph.context import ScatterResult

class ScatterNode(Node):
    name = "scatter"; is_scatter = True
    async def run(self, pkt, ctx):
        branches = [(f"b{i}", pkt.fork(pkt.kind, i=i), "worker") for i in range(3)]
        return ScatterResult(branches=branches, gather_to="gather")

g = Graph("fan")
g.add_node(ScatterNode(), entry=True)
g.add_node(to_node(worker_fn, name="worker"))
g.add_node(MyGatherNode())
g.fan_out("scatter", "gather")
g.add_edge("gather", END)
```

#### Durability (crash-resume)

```python
from orcha.graph.store import FileStore

store = FileStore()  # checkpoints at ~/.orcha/runs/
rt = GraphRuntime(graph, store=store)

# Run, crash, resume — picks up from last checkpoint.
result = await rt.run("query", run_id="my-run")
# ... server crashes ...
result = await rt.run("query", resume_from=store, run_id="my-run")
```

#### HTTP API (new endpoints)

| Endpoint | Description |
|---|---|
| `POST /v1/run` | Execute a graph (`graph=default|research|multi_agent`, `stream=true/false`) |
| `GET /v1/run/{id}` | Get run status / latest checkpoint |
| `GET /v1/run/{id}/events` | SSE event stream |
| `POST /v1/run/{id}/resume` | Resume a crashed run |
| `POST /v1/run/{id}/replay` | Deterministically replay from history |
| `GET /v1/runs` | List all known runs |

#### Migration from ORCHA2 to ORCHA3

ORCHA2 is **fully preserved** — `Orchestrator.run()` still works unchanged.
ORCHA3 is additive:

| ORCHA2 | ORCHA3 equivalent |
|---|---|
| `Orchestrator.run(query)` | `build_default_graph(experts=...)` + `GraphRuntime(graph).run(query)` |
| `Orchestrator.run_async(query)` | Same as above (graph runtime is always async) |
| `OrchaResult` | `RunResult` (same `.answer/.confidence/.explain()` surface) |
| Implicit loop | Explicit graph topology (conditional edges for retry/stop) |
| No checkpointing | `FileStore` / `SqliteStore` — automatic resume after crash |
| No streaming | `EventEmitter` + SSE on `/v1/run/{id}/events` |

---

## Tests

```bash
pip install -e ".[dev]"
pytest -v                  # 151 tests
```

---

## License

MIT
