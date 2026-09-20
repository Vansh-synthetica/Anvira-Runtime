"""
orcha.experts.registry
========================
LocalModelRegistry: the single source of truth for all local-model experts.

Responsibilities
----------------
1. Manual registration  — add_ollama(), add_local_server(), add()
2. Auto-discovery       — discover_ollama() finds every pulled model
3. Config-file loading  — from_config() reads JSON or YAML
4. Synthesizer selection — pick_synthesizer() chooses the largest/best model
5. Health monitoring    — check_health() probes every registered expert
6. Performance stats    — integrated with the selector's history

Synthesizer selection heuristic
---------------------------------
Priority order:
  1. An expert marked synthesizer=True explicitly.
  2. The expert with the largest inferred parameter count from its model name
     (e.g. "llama3.1:70b" → 70.0B > "mistral:7b" → 7.0B).
  3. The first registered expert if no size info is available.

Domain inference from model name
---------------------------------
Uses a keyword table to assign a sensible routing domain automatically
when the user doesn't specify one. Always overridable per-model.

Example
-------
    import asyncio
    from orcha.experts import LocalModelRegistry

    registry = LocalModelRegistry()
    asyncio.run(registry.discover_ollama())
    print(registry.names())
    # ['llama3.2:3b', 'qwen2.5:32b', 'codellama:13b']

    orc = Orchestrator(
        experts=registry.build(),
        synthesizer_expert=registry.pick_synthesizer(),
        run_all_experts=True,
    )
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from .base import BaseExpert
from .local_chat import LocalChatExpert
from .ollama import OllamaExpert, list_ollama_models


# ── Domain inference ──────────────────────────────────────────────────────────

_DOMAIN_HINTS: Dict[str, List[str]] = {
    "code":      ["code", "coder", "codellama", "starcoder", "deepseek-coder",
                  "wizardcoder", "qwen-coder", "devstral", "granite-code"],
    "science":   ["math", "deepseek-math", "wizardmath", "mathstral", "numina"],
    "reasoning": ["reasoning", "r1", "qwq", "o1", "think", "deepthink",
                  "deepseek-r1", "sky-t1"],
    "finance":   ["finance", "fin", "bloombergGPT"],
    "creative":  ["story", "creative", "writer", "novel", "mistral-nemo"],
    "medical":   ["medllama", "clinical", "med", "biomed", "meditron"],
}


def infer_domain(model_name: str) -> str:
    name = model_name.lower()
    for domain, hints in _DOMAIN_HINTS.items():
        if any(h in name for h in hints):
            return domain
    return "general"


# ── Parameter size inference ──────────────────────────────────────────────────

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[xX]?\s*b\b", re.IGNORECASE)


def infer_param_size(model_name: str) -> float:
    """
    Extract parameter count in billions from a model name.
    Returns 0.0 if no size pattern found.
    Examples:
        'llama3.1:70b'           → 70.0
        'qwen2.5-32b-instruct'   → 32.0
        'phi3:3.8b'              → 3.8
        'mixtral-8x7b'           → 7.0  (takes the first match)
    """
    match = _SIZE_RE.search(model_name)
    return float(match.group(1)) if match else 0.0


# ── Registry ──────────────────────────────────────────────────────────────────

class LocalModelRegistry:
    """
    Manages a pool of local-model experts and selects the best synthesizer.
    """

    def __init__(self) -> None:
        self._experts:              Dict[str, BaseExpert] = {}
        self._explicit_synthesizer: Optional[str]        = None

    # ── Registration ──────────────────────────────────────────────────

    def add(self, expert: BaseExpert, synthesizer: bool = False) -> "LocalModelRegistry":
        """Register a pre-constructed BaseExpert instance."""
        self._experts[expert.name] = expert
        if synthesizer:
            self._explicit_synthesizer = expert.name
        return self

    def add_ollama(
        self,
        model:         str,
        base_url:      str           = "http://localhost:11434",
        domain:        Optional[str] = None,
        description:   Optional[str] = None,
        synthesizer:   bool          = False,
        **kwargs,
    ) -> "LocalModelRegistry":
        """
        Register a single Ollama model (must already be `ollama pull`ed).

        Extra kwargs are forwarded to OllamaExpert.__init__() — use them
        to set temperature, num_predict, system_prompt, etc.
        """
        expert = OllamaExpert(
            model=model,
            base_url=base_url,
            domain=domain or infer_domain(model),
            description=description,
            **kwargs,
        )
        return self.add(expert, synthesizer=synthesizer)

    def add_local_server(
        self,
        model:         str,
        base_url:      str           = "http://localhost:8080/v1",
        domain:        Optional[str] = None,
        description:   Optional[str] = None,
        synthesizer:   bool          = False,
        **kwargs,
    ) -> "LocalModelRegistry":
        """
        Register a model served by any OpenAI-compatible local server.
        (llama.cpp, LM Studio, vLLM, LocalAI, Jan, Kobold.cpp, …)
        """
        expert = LocalChatExpert(
            model=model,
            base_url=base_url,
            domain=domain or infer_domain(model),
            description=description,
            **kwargs,
        )
        return self.add(expert, synthesizer=synthesizer)

    # ── Auto-discovery ────────────────────────────────────────────────

    async def discover_ollama(
        self,
        base_url:         str                   = "http://localhost:11434",
        domain_overrides: Optional[Dict[str, str]] = None,
        skip_patterns:    Optional[List[str]]   = None,
    ) -> "LocalModelRegistry":
        """
        Register every model currently pulled into a local Ollama instance.
        Silently registers nothing if Ollama isn't running.

        Parameters
        ----------
        domain_overrides  {model_name: domain} to override auto-inference.
        skip_patterns     List of substrings — models whose names contain any
                          of these will be skipped (e.g. ["embed", "vision"]).
        """
        domain_overrides = domain_overrides or {}
        skip_patterns    = skip_patterns    or []

        models = await list_ollama_models(base_url)
        for model in models:
            if any(p in model.lower() for p in skip_patterns):
                continue
            domain = domain_overrides.get(model, infer_domain(model))
            self.add_ollama(model, base_url=base_url, domain=domain)

        return self

    # ── Config-file loading ───────────────────────────────────────────

    @classmethod
    def from_config(cls, path: Union[str, Path]) -> "LocalModelRegistry":
        """
        Load a registry from a JSON or YAML config file.

        JSON format::

            {
              "models": [
                {
                  "backend": "ollama",
                  "model": "llama3.2:3b",
                  "domain": "general",
                  "description": "Fast generalist"
                },
                {
                  "backend": "ollama",
                  "model": "qwen2.5:32b",
                  "domain": "reasoning",
                  "synthesizer": true
                },
                {
                  "backend": "openai_compatible",
                  "model": "mistral-7b-instruct",
                  "base_url": "http://localhost:8080/v1",
                  "domain": "general",
                  "temperature": 0.5
                }
              ]
            }

        backend aliases: "ollama", "openai_compatible", "local_server",
                         "llama_cpp", "lmstudio", "vllm", "localai"
        """
        path = Path(path)
        raw  = path.read_text()

        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
                data = yaml.safe_load(raw)
            except ImportError:
                raise ImportError(
                    "PyYAML is needed for .yaml config files.\n"
                    "Install: pip install -e \".[config]\"  or  pip install PyYAML\n"
                    "Alternatively, use a .json config file (no extra dependency)."
                )
        else:
            data = json.loads(raw)

        registry = cls()
        for entry in data.get("models", []):
            entry       = dict(entry)
            backend     = entry.pop("backend", "ollama").lower()
            synthesizer = bool(entry.pop("synthesizer", False))

            _OPENAI_COMPAT = {
                "openai_compatible", "local_server",
                "llama_cpp", "lmstudio", "vllm", "localai",
            }

            if backend == "ollama":
                model = entry.pop("model")
                registry.add_ollama(model, synthesizer=synthesizer, **entry)
            elif backend in _OPENAI_COMPAT:
                model = entry.pop("model")
                registry.add_local_server(model, synthesizer=synthesizer, **entry)
            else:
                raise ValueError(
                    f"Unknown backend '{backend}'. Expected 'ollama' or "
                    f"one of: {sorted(_OPENAI_COMPAT)}"
                )

        return registry

    # ── Health checks ─────────────────────────────────────────────────

    async def check_health(self) -> Dict[str, bool]:
        """
        Probe every registered expert concurrently and return
        {name: is_healthy}.
        """
        async def probe(name: str, expert: BaseExpert):
            try:
                return name, await asyncio.wait_for(expert.healthcheck(), timeout=8.0)
            except Exception:
                return name, False

        tasks   = [probe(n, e) for n, e in self._experts.items()]
        results = await asyncio.gather(*tasks)
        return dict(results)

    async def remove_unhealthy(self, base_url: str = "http://localhost:11434") -> List[str]:
        """
        Run healthchecks and remove any experts that fail.
        Returns the list of removed expert names.
        """
        health  = await self.check_health()
        removed = [name for name, ok in health.items() if not ok]
        for name in removed:
            self._experts.pop(name, None)
        if self._explicit_synthesizer in removed:
            self._explicit_synthesizer = None
        return removed

    # ── Synthesizer selection ─────────────────────────────────────────

    def pick_synthesizer(self) -> Optional[str]:
        """
        Return the name of the expert best suited to synthesize all answers.

        Priority:
          1. Explicit synthesizer=True (from config or add() call).
          2. Largest model by inferred parameter count (70b > 32b > 13b …).
          3. First registered expert as a last resort.
          4. None if the registry is empty.
        """
        if not self._experts:
            return None

        if (
            self._explicit_synthesizer
            and self._explicit_synthesizer in self._experts
        ):
            return self._explicit_synthesizer

        def _size(expert: BaseExpert) -> float:
            model_attr = getattr(expert, "model", expert.name)
            return infer_param_size(model_attr)

        best = max(self._experts.values(), key=_size)
        if _size(best) > 0:
            return best.name

        return next(iter(self._experts.values())).name

    # ── Inspection ────────────────────────────────────────────────────

    def build(self) -> Dict[str, BaseExpert]:
        """Return the assembled {name: expert} mapping for Orchestrator."""
        return dict(self._experts)

    def names(self) -> List[str]:
        return list(self._experts.keys())

    def domains(self) -> Dict[str, List[str]]:
        """Return {domain: [expert_names]} grouping."""
        out: Dict[str, List[str]] = {}
        for name, expert in self._experts.items():
            out.setdefault(expert.domain, []).append(name)
        return out

    def summary(self) -> str:
        lines = [f"LocalModelRegistry ({len(self._experts)} experts)"]
        synth = self.pick_synthesizer()
        for name, expert in self._experts.items():
            marker = " ◈ [synthesizer]" if name == synth else ""
            lines.append(f"  {name:<40} {expert.domain:<12}{marker}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._experts)

    def __repr__(self) -> str:
        return f"LocalModelRegistry({self.names()})"
