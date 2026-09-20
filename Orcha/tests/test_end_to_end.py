"""End-to-end tests for the Orchestrator — sync, async, synthesis, registry."""
import json
import pytest

from orcha import Orchestrator, OrchaResult
from orcha.experts.base import BaseExpert, ExpertOutput
from orcha.experts.mock import load_mock_experts, MockSynthesizer
from orcha.experts.registry import LocalModelRegistry, infer_domain, infer_param_size


# ── Deterministic test experts ──────────────────────────────────────────────

class EchoExpert(BaseExpert):
    name = "echo"; domain = "general"
    description = "echoes the query with high confidence"
    async def execute(self, query):
        return ExpertOutput(answer=f"echo: {query}", confidence=0.92, tokens_used=8)

class LowConfExpert(BaseExpert):
    name = "low_conf"; domain = "general"
    description = "always low confidence"
    async def execute(self, query):
        return ExpertOutput(answer="uncertain", confidence=0.20, tokens_used=5)

class SpecificExpert(BaseExpert):
    """Returns a realistic answer with numbers and proper nouns for evaluation."""
    name = "specific"; domain = "general"
    description = "gives detailed specific answers"
    async def execute(self, query):
        return ExpertOutput(
            answer=(
                "Research across 47 studies shows that regular aerobic exercise "
                "reduces cardiovascular risk by 30-40%. The American Heart Association "
                "recommends 150 minutes of moderate activity weekly. Key benefits "
                "include reduced LDL cholesterol (by ~10mg/dL) and lower blood pressure."
            ),
            confidence=0.88, tokens_used=60,
        )


# ── Basic orchestration ─────────────────────────────────────────────────────

def test_sync_run_basic():
    orc = Orchestrator(experts={"echo": EchoExpert()}, max_iterations=1)
    result = orc.run("hello world")
    assert isinstance(result, OrchaResult)
    assert "echo: hello world" in result.answer

def test_sync_run_with_mock_experts():
    orc = Orchestrator(experts=load_mock_experts(), max_iterations=1)
    result = orc.run("What is compound interest?")
    assert result.answer
    assert 0.0 <= result.confidence <= 1.0

@pytest.mark.asyncio
async def test_async_run():
    orc = Orchestrator(experts={"echo": EchoExpert()}, max_iterations=1)
    result = await orc.run_async("async test")
    assert "echo: async test" in result.answer

@pytest.mark.asyncio
async def test_sync_raises_inside_event_loop():
    orc = Orchestrator(experts={"echo": EchoExpert()})
    with pytest.raises(RuntimeError, match="running event loop"):
        orc.run("should fail")

@pytest.mark.asyncio
async def test_high_confidence_stops_after_one_iteration():
    orc = Orchestrator(experts={"echo": EchoExpert()}, max_iterations=5)
    result = await orc.run_async("hello")
    assert result.iterations == 1

@pytest.mark.asyncio
async def test_low_confidence_exhausts_iterations():
    orc = Orchestrator(experts={"low": LowConfExpert()}, max_iterations=2)
    result = await orc.run_async("hard question")
    assert result.iterations == 2

@pytest.mark.asyncio
async def test_result_fields_all_present():
    orc = Orchestrator(experts={"echo": EchoExpert()}, max_iterations=1)
    result = await orc.run_async("test")
    assert result.answer
    assert isinstance(result.confidence, float)
    assert isinstance(result.synthesized, bool)
    assert isinstance(result.contributors, list)
    assert isinstance(result.domains, list)
    assert isinstance(result.iterations, int)
    assert isinstance(result.cost, float)
    assert isinstance(result.latency_s, float)
    assert isinstance(result.trace, list)
    assert result.trace[0]["stage"] == "input"
    assert result.trace[-1]["stage"] == "response"

@pytest.mark.asyncio
async def test_result_explain_and_to_dict():
    orc = Orchestrator(experts={"echo": EchoExpert()}, max_iterations=1)
    result = await orc.run_async("explain test")
    d = result.to_dict()
    assert d["answer"] == result.answer
    assert "trace" in d
    explain = result.explain()
    assert "explain test" in explain
    assert "confidence=" in explain
    assert result.final_answer == result.answer  # alias


# ── Synthesis ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_synthesis_with_mock_synthesizer():
    experts = load_mock_experts()
    orc = Orchestrator(
        experts=experts,
        synthesizer_expert="mock_synthesizer",
        run_all_experts=True,
        max_iterations=1,
    )
    result = await orc.run_async("What is the best approach to managing technical debt?")
    assert result.synthesized is True
    assert result.primary == "mock_synthesizer"
    assert len(result.answer) > 30
    assert len(result.contributors) > 1

@pytest.mark.asyncio
async def test_run_all_experts_uses_full_pool():
    experts = load_mock_experts()
    orc = Orchestrator(experts=experts, run_all_experts=True, max_iterations=1)
    result = await orc.run_async("broad general question")
    assert len(result.contributors) >= 2


# ── Expert management ────────────────────────────────────────────────────────

def test_register_and_unregister():
    orc = Orchestrator(experts={}, max_iterations=1)
    assert orc.list_experts() == []
    orc.register_expert("echo", EchoExpert())
    assert any(e["name"] == "echo" for e in orc.list_experts())
    orc.unregister_expert("echo")
    assert orc.list_experts() == []

def test_register_returns_self_for_chaining():
    orc = Orchestrator(experts={})
    result = orc.register_expert("echo", EchoExpert())
    assert result is orc

def test_set_synthesizer():
    orc = Orchestrator(experts={"echo": EchoExpert(), "synth": MockSynthesizer()})
    assert orc.synthesizer_expert is None
    orc.set_synthesizer("synth")
    assert orc.synthesizer_expert == "synth"
    orc.set_synthesizer(None)
    assert orc.synthesizer_expert is None

def test_list_experts_includes_synthesizer_flag():
    orc = Orchestrator(
        experts={"echo": EchoExpert(), "synth": MockSynthesizer()},
        synthesizer_expert="synth",
    )
    listing = orc.list_experts()
    synth_entry = next(e for e in listing if e["name"] == "synth")
    echo_entry  = next(e for e in listing if e["name"] == "echo")
    assert synth_entry["is_synthesizer"] is True
    assert echo_entry["is_synthesizer"] is False


# ── LocalModelRegistry ───────────────────────────────────────────────────────

def test_registry_add_ollama_and_build():
    r = LocalModelRegistry()
    r.add_ollama("llama3:8b",  domain="general")
    r.add_ollama("llama3:70b", domain="reasoning", synthesizer=True)
    pool = r.build()
    assert len(pool) == 2
    assert r.pick_synthesizer() == "ollama_llama3_70b"

def test_registry_explicit_synthesizer_wins_over_size():
    r = LocalModelRegistry()
    r.add_ollama("llama3.1:70b")
    r.add_ollama("phi3:3b", synthesizer=True)   # small but explicitly marked
    assert r.pick_synthesizer() == "ollama_phi3_3b"

def test_registry_picks_largest_by_size():
    r = LocalModelRegistry()
    r.add_ollama("mistral:7b")
    r.add_ollama("qwen2.5:32b")
    r.add_ollama("llama3.2:3b")
    assert r.pick_synthesizer() == "ollama_qwen2_5_32b"

def test_registry_add_local_server():
    from orcha.experts.local_chat import LocalChatExpert
    r = LocalModelRegistry()
    r.add_local_server("qwen2.5-32b", base_url="http://localhost:8080/v1", domain="reasoning")
    pool = r.build()
    assert len(pool) == 1
    assert isinstance(list(pool.values())[0], LocalChatExpert)

def test_registry_from_json_config(tmp_path):
    cfg = {"models": [
        {"backend": "ollama", "model": "llama3:8b",  "domain": "general"},
        {"backend": "ollama", "model": "llama3:70b", "synthesizer": True},
    ]}
    p = tmp_path / "models.json"
    p.write_text(json.dumps(cfg))
    r = LocalModelRegistry.from_config(p)
    assert len(r.build()) == 2
    assert r.pick_synthesizer() == "ollama_llama3_70b"

def test_registry_domains_grouping():
    r = LocalModelRegistry()
    r.add_ollama("llama3:8b",     domain="general")
    r.add_ollama("codellama:13b", domain="code")
    r.add_ollama("mistral:7b",    domain="general")
    domains = r.domains()
    assert set(domains["general"]) == {"ollama_llama3_8b", "ollama_mistral_7b"}
    assert domains["code"] == ["ollama_codellama_13b"]

def test_registry_summary_string():
    r = LocalModelRegistry()
    r.add_ollama("llama3:8b")
    summary = r.summary()
    assert "LocalModelRegistry" in summary
    assert "llama3" in summary.lower() or "ollama_llama3_8b" in summary

def test_infer_domain_known_models():
    assert infer_domain("codellama:13b")      == "code"
    assert infer_domain("deepseek-math:7b")   == "science"
    assert infer_domain("qwq-32b-preview")    == "reasoning"
    assert infer_domain("mistral:7b")         == "general"
    assert infer_domain("meditron-70b")       == "medical"

def test_infer_param_size():
    assert infer_param_size("llama3.1:70b")           == 70.0
    assert infer_param_size("qwen2.5-32b-instruct")    == 32.0
    assert infer_param_size("phi3:3.8b")               == 3.8
    assert infer_param_size("unknown-model")           == 0.0
