"""Example: Grammar-constrained AICL output with llama.cpp.

Shows how to use AICLGrammar to force a model to output valid AICL
semantic actions instead of free-form text.

Without grammar constraints:
    Model might output: "I think the sentiment is positive because..."
    -> Requires parsing, often fails, wastes tokens

With grammar constraints:
    Model outputs: {"intent":"CLASSIFY","target":"orch","inputs":{"text":"..."},"result":{"label":"positive","confidence":0.95}}
    -> Always valid, parse guaranteed, fewer tokens

Usage with llama-cpp-python:
    pip install llama-cpp-python
    python examples/llama_grammar_example.py

Usage without llama.cpp (演示 only):
    python examples/llama_grammar_example.py --demo
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, ".")

from aicl.semantic.grammar import AICLGrammar
from aicl.semantic.types import ModelIntent, ModelAction
from aicl.semantic.adapter import SemanticAdapter


def classify_with_grammar(
    text: str,
    labels: list[str],
    model_path: str = "",
) -> dict:
    """Classify text using grammar-constrained generation.

    Args:
        text: Input text to classify.
        labels: Possible labels (e.g., ["positive", "negative"]).
        model_path: Path to GGUF model file.

    Returns:
        Dict with label and confidence.
    """
    grammar = AICLGrammar.for_classify(labels=labels)

    # Build the prompt
    prompt = f"Classify the following text into one of {labels}.\nText: \"{text}\"\nOutput a JSON action."

    if not model_path:
        # Demo mode: show what would happen
        print(f"\n--- Grammar-constrained classification ---")
        print(f"Prompt: {prompt}")
        print(f"Grammar BNF (first 500 chars):")
        print(grammar.bnf[:500])
        print(f"\nJSON Schema:")
        print(json.dumps(grammar.json_schema, indent=2))
        print(f"\nExpected model output (constrained):")
        expected = {
            "intent": "CLASSIFY",
            "target": "orchestrator",
            "inputs": {"text": text},
            "result": {"label": labels[0], "confidence": 0.95},
            "params": {"labels": labels},
        }
        print(json.dumps(expected, indent=2))
        return expected["result"]

    # Real inference with llama-cpp-python
    try:
        from llama_cpp import Llama

        llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)

        # Grammar-constrained generation
        t0 = time.perf_counter()
        output = llm.create_completion(
            prompt=prompt,
            max_tokens=128,
            temperature=0.1,
            grammar=grammar.bnf,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        raw_output = output["choices"][0]["text"]
        print(f"Raw output ({latency_ms:.1f}ms): {raw_output}")

        # Parse (guaranteed valid due to grammar)
        action = json.loads(raw_output)
        result = action.get("result", {})
        print(f"Classification: {result.get('label')} (confidence: {result.get('confidence')})")
        return result

    except ImportError:
        print("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        return {}


def generate_with_grammar(
    prompt: str,
    model_path: str = "",
) -> str:
    """Generate text using grammar-constrained AICL output.

    The model must output a valid AICL GENERATE action with the result.
    """
    grammar = AICLGrammar.for_generate()

    if not model_path:
        print(f"\n--- Grammar-constrained generation ---")
        print(f"Prompt: {prompt}")
        print(f"Grammar accepts intents: GENERATE")
        expected = {
            "intent": "GENERATE",
            "target": "orchestrator",
            "inputs": {"prompt": prompt},
            "result": {"text": "Generated response here..."},
        }
        print(f"Expected output: {json.dumps(expected, indent=2)}")
        return expected["result"]["text"]

    try:
        from llama_cpp import Llama

        llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)

        t0 = time.perf_counter()
        output = llm.create_completion(
            prompt=f"Respond to the following prompt with an AICL GENERATE action.\nPrompt: {prompt}\n",
            max_tokens=256,
            temperature=0.7,
            grammar=grammar.bnf,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        raw_output = output["choices"][0]["text"]
        print(f"Raw output ({latency_ms:.1f}ms): {raw_output}")

        action = json.loads(raw_output)
        return action.get("result", {}).get("text", "")

    except ImportError:
        print("llama-cpp-python not installed.")
        return ""


def tool_call_with_grammar(
    query: str,
    available_tools: list[str],
    model_path: str = "",
) -> dict:
    """Route a query to the appropriate tool using grammar-constrained output."""
    grammar = AICLGrammar.for_tool()

    if not model_path:
        print(f"\n--- Grammar-constrained tool routing ---")
        print(f"Query: {query}")
        print(f"Available tools: {available_tools}")
        expected = {
            "intent": "CALL_TOOL",
            "target": "tool_router",
            "params": {"tool_name": available_tools[0], "parameters": {"query": query}},
        }
        print(f"Expected output: {json.dumps(expected, indent=2)}")
        return expected["params"]

    try:
        from llama_cpp import Llama

        llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)

        t0 = time.perf_counter()
        output = llm.create_completion(
            prompt=f"Route this query to the appropriate tool.\nQuery: {query}\nTools: {available_tools}\n",
            max_tokens=128,
            temperature=0.1,
            grammar=grammar.bnf,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        raw_output = output["choices"][0]["text"]
        print(f"Raw output ({latency_ms:.1f}ms): {raw_output}")

        action = json.loads(raw_output)
        return action.get("params", {})

    except ImportError:
        print("llama-cpp-python not installed.")
        return {}


def main():
    demo_mode = "--demo" in sys.argv or not any(
        arg.endswith(".gguf") for arg in sys.argv
    )

    if demo_mode:
        print("=== AICL Grammar-Constrained Generation Demo ===")
        print("(Use --model path/to/model.gguf for real inference)\n")

    # 1. Classification
    classify_with_grammar(
        text="I absolutely love this product! Best purchase ever.",
        labels=["positive", "negative", "neutral"],
    )

    # 2. Generation
    generate_with_grammar(
        prompt="Explain quantum entanglement in one sentence.",
    )

    # 3. Tool routing
    tool_call_with_grammar(
        query="What's the weather in London?",
        available_tools=["weather_api", "search", "calculator"],
    )

    # Show the token savings
    print("\n=== Token Comparison ===")
    adapter = SemanticAdapter(origin="demo")

    # Without grammar: model outputs free text, needs parsing
    free_text = "I think the sentiment is positive because the user expresses enthusiasm. The confidence is about 0.95."
    free_tokens = len(free_text.split())  # Rough approximation

    # With grammar: model outputs structured JSON directly
    grammar_output = json.dumps({
        "intent": "CLASSIFY",
        "target": "orchestrator",
        "inputs": {"text": "I love this product"},
        "result": {"label": "positive", "confidence": 0.95},
        "params": {"labels": ["positive", "negative"]},
    })
    grammar_tokens = len(grammar_output.split())

    print(f"  Free text output: ~{free_tokens} tokens (needs parsing)")
    print(f"  Grammar output:   ~{grammar_tokens} tokens (parse guaranteed)")
    print(f"  Savings:          ~{free_tokens - grammar_tokens} tokens + no parse failures")


if __name__ == "__main__":
    main()
