# Quickstart

## 1. Install

```bash
git clone https://github.com/LocalHouseLLM/orcha01
cd orcha01
pip install -e ".[all]"
```

## 2. Zero-setup demo (no local models needed)

```bash
python demo.py
```

This runs 3 example queries through the built-in mock expert pool
(5 specialist models + a mock synthesizer). No Ollama, no API keys, nothing
to configure. You'll see the synthesis pipeline in action immediately.

## 3. Use your real local models (Ollama)

Install Ollama from https://ollama.ai, then pull as many models as you want:

```bash
ollama pull llama3.2:3b       # fast, small
ollama pull llama3.1:8b       # good all-rounder
ollama pull mistral:7b        # great for reasoning
ollama pull codellama:13b     # code expert
ollama pull llama3.1:70b      # largest → becomes the synthesizer
```

Then run:

```bash
python demo.py --ollama "How should I structure a microservices architecture?"
```

Orcha auto-discovers every pulled model, runs them all in parallel, and has
the largest one synthesize their answers into one refined response.

## 4. Python API

```python
import asyncio
from orcha import Orchestrator, LocalModelRegistry

async def main():
    # Auto-discover all local Ollama models
    registry = LocalModelRegistry()
    await registry.discover_ollama()

    print(f"Found {len(registry)} models: {registry.names()}")
    print(f"Synthesizer will be: {registry.pick_synthesizer()}")

    orc = Orchestrator(
        experts=registry.build(),
        synthesizer_expert=registry.pick_synthesizer(),
        run_all_experts=True,    # send query to EVERY model
        max_iterations=2,        # retry once if confidence is low
    )

    result = await orc.run_async("Explain the risks of intermittent fasting")

    print(result.answer)
    print(f"Confidence:    {result.confidence:.0%}")
    print(f"Synthesized:   {result.synthesized}")
    print(f"Contributors:  {result.contributors}")
    print(f"Primary:       {result.primary}")
    print()
    print(result.explain())    # full stage-by-stage trace

asyncio.run(main())
```

## 5. Config file (manage many models cleanly)

Create `models.json`:

```json
{
  "models": [
    {"backend": "ollama", "model": "llama3.2:3b",  "domain": "general",   "description": "Fast generalist"},
    {"backend": "ollama", "model": "codellama:13b", "domain": "code",      "description": "Code specialist"},
    {"backend": "ollama", "model": "mistral:7b",    "domain": "reasoning", "description": "Strong reasoner"},
    {"backend": "ollama", "model": "llama3.1:70b",  "domain": "reasoning", "synthesizer": true, "description": "Synthesis model"}
  ]
}
```

```python
from orcha import Orchestrator
from orcha.experts import LocalModelRegistry

registry = LocalModelRegistry.from_config("models.json")
orc = Orchestrator(
    experts=registry.build(),
    synthesizer_expert=registry.pick_synthesizer(),
    run_all_experts=True,
)
result = orc.run("Your question here")
print(result.answer)
```

## 6. llama.cpp / LM Studio / vLLM / any local server

Orcha speaks OpenAI-compatible `/chat/completions` — any local inference
server works:

```python
from orcha import Orchestrator
from orcha.experts import LocalChatExpert, LocalModelRegistry

registry = LocalModelRegistry()

# llama.cpp server: ./llama-server -m model.gguf --port 8080
registry.add_local_server(
    model="qwen2.5-32b-instruct",
    base_url="http://localhost:8080/v1",
    domain="reasoning",
    synthesizer=True,
)

# LM Studio: starts at localhost:1234 by default
registry.add_local_server(
    model="mistral-7b-instruct",
    base_url="http://localhost:1234/v1",
    domain="general",
)

orc = Orchestrator(
    experts=registry.build(),
    synthesizer_expert=registry.pick_synthesizer(),
    run_all_experts=True,
)
result = orc.run("Analyse the tradeoffs of REST vs GraphQL")
print(result.answer)
```

## 7. Web UI

```bash
uvicorn orcha.api.server:app --reload --port 8420
```

Open http://localhost:8420. The UI:
- Shows your full local model roster on the left
- Lets you type a query and hit **Run all models**
- Renders a live **pipeline trace** (every stage, every ms, every model score)
- Shows the final answer with a **◈ synthesized** badge when synthesis ran
- Lists which models contributed and which one wrote the final answer

After pulling a new Ollama model, click the **reload** endpoint
(`POST /reload`) or restart the server to pick it up.

## 8. Write your own expert

Any Python class implementing `async execute() -> ExpertOutput` works:

```python
from orcha.experts.base import BaseExpert, ExpertOutput

class MyFineTunedModel(BaseExpert):
    name        = "my_finance_model"
    domain      = "finance"
    description = "Fine-tuned finance analyst"

    async def execute(self, query: str) -> ExpertOutput:
        # call your model however you like
        answer = await my_custom_inference(query)
        return ExpertOutput(
            answer=answer,
            confidence=0.88,
            tokens_used=300,
        )

orc.register_expert("my_finance_model", MyFineTunedModel())
```

## 9. Run the tests

```bash
pip install -e ".[dev]"
pytest -v          # 55 tests, all green
```
