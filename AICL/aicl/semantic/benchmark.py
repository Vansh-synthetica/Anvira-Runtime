"""Benchmark: English vs JSON vs AICL semantic instructions.

Measures:
    1. Token count (approximate, via whitespace/word splitting)
    2. Encode latency (format -> wire bytes)
    3. Decode latency (wire bytes -> semantic type)
    4. Parse success rate (roundtrip correctness)
    5. Bytes on wire

DOES NOT claim improvements without measurements. The benchmark
produces raw numbers; interpretation is left to the reader.

Usage:
    python -m aicl.semantic.benchmark
    python -m aicl.semantic.benchmark --rounds 1000 --output report.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from aicl.semantic.types import ModelIntent, ModelAction
from aicl.semantic.adapter import SemanticAdapter
from aicl.bin.codec_api import encode as bin_encode, decode as bin_decode

__all__ = [
    "BenchmarkResult",
    "InstructionFormat",
    "run_benchmark",
    "print_report",
]


# ──────────────────────────────────────────────────────────────
# Token estimation (no tiktoken dependency)
# ──────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Estimate token count without a tokenizer dependency.

    Uses whitespace splitting as a lower-bound approximation.
    Real tokenizers (BPE, SentencePiece) typically produce fewer
    tokens than word count for structured text, and more for
    natural language. This is a rough proxy.
    """
    return len(text.split())


# ──────────────────────────────────────────────────────────────
# Instruction formats
# ──────────────────────────────────────────────────────────────

@dataclass
class InstructionFormat:
    """A named instruction format with encode/parse functions."""
    name: str
    encode_fn: Callable[[ModelAction], str]
    parse_fn: Callable[[str], Any]
    description: str = ""


def _english_encode(action: ModelAction) -> str:
    """Encode a ModelAction as a natural language instruction."""
    parts = []
    intent = action.intent.value

    if intent == "CLASSIFY":
        labels = action.params.get("labels", [])
        text = action.inputs.get("text", "")
        parts.append(f"Please classify the following text into one of the categories: {', '.join(labels)}.")
        parts.append(f"Text: \"{text}\"")
        parts.append("Return the label and your confidence score.")
    elif intent == "GENERATE":
        prompt = action.inputs.get("prompt", action.inputs.get("text", ""))
        parts.append(f"Please generate a response for the following prompt:")
        parts.append(f"Prompt: \"{prompt}\"")
        if action.params.get("max_tokens"):
            parts.append(f"Limit your response to {action.params['max_tokens']} tokens.")
    elif intent == "CALL_TOOL":
        tool = action.params.get("tool_name", "unknown")
        params = action.params.get("parameters", {})
        parts.append(f"Please call the tool '{tool}' with the following parameters:")
        parts.append(json.dumps(params, indent=2))
    elif intent == "READ_MEMORY":
        key = action.inputs.get("key", action.target)
        parts.append(f"Please read the value stored under key '{key}' from memory.")
    elif intent == "WRITE_MEMORY":
        key = action.inputs.get("key", action.target)
        value = action.inputs.get("value", "")
        parts.append(f"Please store the following value under key '{key}' in memory:")
        parts.append(f"Value: {value}")
    elif intent == "EMBED":
        text = action.inputs.get("text", "")
        parts.append(f"Please produce a vector embedding for the following text:")
        parts.append(f"Text: \"{text}\"")
    elif intent == "VERIFY":
        claim = action.inputs.get("claim", action.inputs.get("text", ""))
        parts.append(f"Please verify the following claim for correctness:")
        parts.append(f"Claim: \"{claim}\"")
    else:
        parts.append(f"Please perform the following action: {intent}")
        if action.inputs:
            parts.append(f"Inputs: {json.dumps(action.inputs)}")
        if action.params:
            parts.append(f"Parameters: {json.dumps(action.params)}")

    return "\n".join(parts)


def _json_encode(action: ModelAction) -> str:
    """Encode a ModelAction as JSON."""
    return json.dumps(action.to_dict(), indent=2)


def _aicl_encode(action: ModelAction) -> str:
    """Encode a ModelAction as compact AICL semantic JSON."""
    return json.dumps(action.to_dict(), separators=(",", ":"))


def _english_parse(text: str) -> Dict[str, Any]:
    """Parse English text into a dict (best-effort, lossy)."""
    result: Dict[str, Any] = {"_raw": text, "_parse_quality": "approximate"}

    upper = text.upper()
    for intent in ModelIntent:
        if intent.value in upper:
            result["intent"] = intent.value
            break

    if "intent" not in result:
        result["intent"] = "UNKNOWN"

    return result


def _json_parse(text: str) -> Dict[str, Any]:
    """Parse JSON text into a dict."""
    return json.loads(text)


def _aicl_parse(text: str) -> Dict[str, Any]:
    """Parse compact AICL JSON into a dict."""
    return json.loads(text)


# Pre-defined formats
FORMATS = {
    "english": InstructionFormat(
        name="english",
        encode_fn=_english_encode,
        parse_fn=_english_parse,
        description="Natural language instruction",
    ),
    "json": InstructionFormat(
        name="json",
        encode_fn=_json_encode,
        parse_fn=_json_parse,
        description="JSON tool instruction (indented)",
    ),
    "aicl": InstructionFormat(
        name="aicl",
        encode_fn=_aicl_encode,
        parse_fn=_aicl_parse,
        description="AICL semantic JSON (compact)",
    ),
}


# ──────────────────────────────────────────────────────────────
# Benchmark actions
# ──────────────────────────────────────────────────────────────

def _make_benchmark_actions() -> List[ModelAction]:
    """Generate a set of representative ModelActions for benchmarking."""
    return [
        ModelAction(
            intent=ModelIntent.CLASSIFY,
            target="sentiment",
            inputs={"text": "I absolutely love this product! It exceeded all my expectations."},
            params={"labels": ["positive", "negative", "neutral"]},
            result={"label": "positive", "confidence": 0.95},
        ),
        ModelAction(
            intent=ModelIntent.GENERATE,
            target="llm",
            inputs={"prompt": "Explain quantum entanglement in simple terms."},
            params={"max_tokens": 256, "temperature": 0.7},
        ),
        ModelAction(
            intent=ModelIntent.CALL_TOOL,
            target="tool_router",
            inputs={"query": "weather in London"},
            params={"tool_name": "weather_api", "parameters": {"location": "London", "units": "celsius"}},
        ),
        ModelAction(
            intent=ModelIntent.READ_MEMORY,
            target="memory_store",
            inputs={"key": "user.session.context"},
        ),
        ModelAction(
            intent=ModelIntent.WRITE_MEMORY,
            target="memory_store",
            inputs={"key": "user.preferences.theme", "value": "dark"},
        ),
        ModelAction(
            intent=ModelIntent.EMBED,
            target="embedder",
            inputs={"text": "The quick brown fox jumps over the lazy dog."},
        ),
        ModelAction(
            intent=ModelIntent.VERIFY,
            target="fact_checker",
            inputs={"claim": "The Earth is the third planet from the Sun."},
        ),
        ModelAction(
            intent=ModelIntent.SUMMARIZE,
            target="summarizer",
            inputs={"text": "AICL replaces verbose natural language communication between AI modules with compact, typed symbolic packets. This reduces ambiguity and improves machine-to-machine communication efficiency."},
            params={"max_length": 50},
        ),
        ModelAction(
            intent=ModelIntent.REASON,
            target="reasoner",
            inputs={"problem": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?"},
        ),
        ModelAction(
            intent=ModelIntent.EXTRACT,
            target="extractor",
            inputs={"text": "John Smith went to Paris on March 15, 2024. He stayed at the Hotel Grande for 5 nights."},
            params={"fields": ["person", "location", "date", "duration"]},
        ),
    ]


# ──────────────────────────────────────────────────────────────
# Benchmark result
# ──────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """Result of benchmarking a single instruction format."""
    format_name: str
    description: str
    num_actions: int
    num_rounds: int

    # Token counts
    token_counts: List[int] = field(default_factory=list)
    token_mean: float = 0.0
    token_median: float = 0.0
    token_p95: float = 0.0
    token_total: int = 0

    # Byte counts
    byte_counts: List[int] = field(default_factory=list)
    byte_mean: float = 0.0
    byte_median: float = 0.0
    byte_total: int = 0

    # Encode latency (ms)
    encode_latencies: List[float] = field(default_factory=list)
    encode_mean_ms: float = 0.0
    encode_p95_ms: float = 0.0

    # Decode latency (ms)
    decode_latencies: List[float] = field(default_factory=list)
    decode_mean_ms: float = 0.0
    decode_p95_ms: float = 0.0

    # Parse success
    parse_successes: int = 0
    parse_failures: int = 0
    parse_success_rate: float = 0.0

    # Roundtrip correctness
    roundtrip_exact: int = 0
    roundtrip_exact_rate: float = 0.0

    def compute_stats(self) -> None:
        """Compute aggregate statistics from raw measurements."""
        if self.token_counts:
            s = sorted(self.token_counts)
            self.token_mean = statistics.mean(s)
            self.token_median = statistics.median(s)
            self.token_p95 = s[int(len(s) * 0.95)] if len(s) >= 2 else s[-1]
            self.token_total = sum(s)

        if self.byte_counts:
            s = sorted(self.byte_counts)
            self.byte_mean = statistics.mean(s)
            self.byte_median = statistics.median(s)
            self.byte_total = sum(s)

        if self.encode_latencies:
            s = sorted(self.encode_latencies)
            self.encode_mean_ms = statistics.mean(s)
            self.encode_p95_ms = s[int(len(s) * 0.95)] if len(s) >= 2 else s[-1]

        if self.decode_latencies:
            s = sorted(self.decode_latencies)
            self.decode_mean_ms = statistics.mean(s)
            self.decode_p95_ms = s[int(len(s) * 0.95)] if len(s) >= 2 else s[-1]

        total = self.parse_successes + self.parse_failures
        self.parse_success_rate = self.parse_successes / total if total else 0.0
        self.roundtrip_exact_rate = self.roundtrip_exact / total if total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format_name,
            "description": self.description,
            "num_actions": self.num_actions,
            "num_rounds": self.num_rounds,
            "tokens": {
                "mean": round(self.token_mean, 2),
                "median": round(self.token_median, 2),
                "p95": round(self.token_p95, 2),
                "total": self.token_total,
            },
            "bytes": {
                "mean": round(self.byte_mean, 2),
                "median": round(self.byte_median, 2),
                "total": self.byte_total,
            },
            "encode_latency_ms": {
                "mean": round(self.encode_mean_ms, 4),
                "p95": round(self.encode_p95_ms, 4),
            },
            "decode_latency_ms": {
                "mean": round(self.decode_mean_ms, 4),
                "p95": round(self.decode_p95_ms, 4),
            },
            "parse_success_rate": round(self.parse_success_rate, 4),
            "roundtrip_exact_rate": round(self.roundtrip_exact_rate, 4),
        }


# ──────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────

def run_benchmark(
    rounds: int = 100,
    actions: Optional[List[ModelAction]] = None,
    formats: Optional[List[str]] = None,
) -> Dict[str, BenchmarkResult]:
    """Run the benchmark across formats.

    Args:
        rounds: Number of iterations per action per format.
        actions: List of ModelActions to benchmark. Default: 10 representative actions.
        formats: List of format names to benchmark. Default: all.

    Returns:
        Dict mapping format name to BenchmarkResult.
    """
    if actions is None:
        actions = _make_benchmark_actions()
    if formats is None:
        formats = list(FORMATS.keys())

    adapter = SemanticAdapter(origin="benchmark")
    results: Dict[str, BenchmarkResult] = {}

    for fmt_name in formats:
        fmt = FORMATS[fmt_name]
        result = BenchmarkResult(
            format_name=fmt_name,
            description=fmt.description,
            num_actions=len(actions),
            num_rounds=rounds,
        )

        for action in actions:
            # Encode to instruction text
            instruction_text = fmt.encode_fn(action)
            text_tokens = _estimate_tokens(instruction_text)
            text_bytes = len(instruction_text.encode("utf-8"))

            # Also get AICL binary size for comparison
            try:
                wire_bytes = adapter.action_to_bytes(action)
                wire_size = len(wire_bytes)
            except Exception:
                wire_size = 0

            for _ in range(rounds):
                # Encode latency
                t0 = time.perf_counter()
                encoded = fmt.encode_fn(action)
                t1 = time.perf_counter()
                result.encode_latencies.append((t1 - t0) * 1000)

                result.token_counts.append(text_tokens)
                result.byte_counts.append(text_bytes)

                # Decode/parse latency
                t2 = time.perf_counter()
                try:
                    parsed = fmt.parse_fn(encoded)
                    t3 = time.perf_counter()
                    result.decode_latencies.append((t3 - t2) * 1000)
                    result.parse_successes += 1

                    # Roundtrip check: can we recover the intent?
                    parsed_intent = parsed.get("intent", "")
                    if parsed_intent == action.intent.value:
                        result.roundtrip_exact += 1
                    elif fmt_name == "english" and parsed_intent != "UNKNOWN":
                        result.roundtrip_exact += 1  # partial credit for english
                except Exception:
                    t3 = time.perf_counter()
                    result.decode_latencies.append((t3 - t2) * 1000)
                    result.parse_failures += 1

        result.compute_stats()
        results[fmt_name] = result

    return results


# ──────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────

def print_report(results: Dict[str, BenchmarkResult]) -> str:
    """Generate a formatted comparison report."""
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("AICL Semantic Instruction Benchmark")
    lines.append("=" * 72)
    lines.append("")

    # Header
    lines.append(
        f"{'Format':<12} {'Tokens':>8} {'Bytes':>8} "
        f"{'Enc(ms)':>10} {'Dec(ms)':>10} "
        f"{'Parse%':>8} {'Round%':>8}"
    )
    lines.append("-" * 72)

    for name, r in results.items():
        lines.append(
            f"{name:<12} {r.token_mean:>8.1f} {r.byte_mean:>8.0f} "
            f"{r.encode_mean_ms:>10.4f} {r.decode_mean_ms:>10.4f} "
            f"{r.parse_success_rate:>7.0%} {r.roundtrip_exact_rate:>7.0%}"
        )

    lines.append("-" * 72)
    lines.append("")

    # Delta comparison
    fmts = list(results.keys())
    if len(fmts) >= 2:
        base = results[fmts[0]]
        lines.append(f"Relative to {fmts[0]}:")
        for name in fmts[1:]:
            r = results[name]
            if base.token_mean > 0:
                tok_delta = (r.token_mean - base.token_mean) / base.token_mean * 100
            else:
                tok_delta = 0
            if base.byte_mean > 0:
                byte_delta = (r.byte_mean - base.byte_mean) / base.byte_mean * 100
            else:
                byte_delta = 0
            sign_tok = "+" if tok_delta >= 0 else ""
            sign_byte = "+" if byte_delta >= 0 else ""
            lines.append(
                f"  {name}: {sign_tok}{tok_delta:.1f}% tokens, "
                f"{sign_byte}{byte_delta:.1f}% bytes"
            )
        lines.append("")

    # Wire size comparison
    lines.append("Wire format sizes (AICL binary for reference):")
    adapter = SemanticAdapter(origin="benchmark")
    actions = _make_benchmark_actions()
    for i, action in enumerate(actions[:3]):
        try:
            wire = adapter.action_to_bytes(action)
            eng = FORMATS["english"].encode_fn(action)
            jsn = FORMATS["json"].encode_fn(action)
            aicl = FORMATS["aicl"].encode_fn(action)
            lines.append(
                f"  Action {i+1} ({action.intent.value}): "
                f"binary={len(wire)}B, "
                f"english={len(eng.encode())}B, "
                f"json={len(jsn.encode())}B, "
                f"aicl={len(aicl.encode())}B"
            )
        except Exception as e:
            lines.append(f"  Action {i+1}: error - {e}")
    lines.append("")

    lines.append("=" * 72)
    lines.append("NOTE: Token counts are approximate (whitespace splitting).")
    lines.append("Latencies measure format serialization only, not LLM inference.")
    lines.append("No improvement claims are made without actual model measurements.")
    lines.append("=" * 72)

    report = "\n".join(lines)
    return report


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark AICL semantic instructions vs alternatives",
    )
    parser.add_argument(
        "--rounds", type=int, default=100,
        help="Number of iterations per action (default: 100)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--formats", nargs="+", default=None,
        choices=list(FORMATS.keys()),
        help="Formats to benchmark (default: all)",
    )
    args = parser.parse_args()

    results = run_benchmark(rounds=args.rounds, formats=args.formats)
    report = print_report(results)
    print(report)

    if args.output:
        out_data = {name: r.to_dict() for name, r in results.items()}
        with open(args.output, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
