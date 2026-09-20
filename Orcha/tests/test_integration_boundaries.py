"""
Tests for orcha.integrations — the external-framework integration
boundaries. The key safety property: Orcha core imports and runs with
NO Lang ecosystem packages installed; every external framework is
lazily loaded behind its boundary and reports availability.
"""
import pytest

from orcha import availability_report, IntegrationUnavailable, IntegrationStatus
from orcha.integrations import (
    DeepAgentsBoundary, LangChainBoundary, LangGraphBoundary,
    McpBoundary, PhoenixBoundary, RagasBoundary,
)
from orcha.integrations.base import ExecutionEngine


# ── Availability report ──────────────────────────────────────────────────────

def test_availability_report_lists_all_boundaries():
    report = availability_report()
    assert set(report) == {
        "langchain", "langgraph", "deepagents", "mcp", "ragas", "phoenix",
    }

def test_report_entries_are_status_objects():
    for status in availability_report().values():
        assert isinstance(status, IntegrationStatus)
        assert status.to_dict()["name"] == status.name

def test_lang_boundaries_unavailable_without_extra():
    report = availability_report()
    for name in ("langchain", "langgraph", "deepagents", "ragas", "phoenix"):
        if report[name].available:
            pytest.skip(f"{name} installed on this machine")
        assert report[name].extra is not None


# ── Lazy loading ─────────────────────────────────────────────────────────────

def test_boundaries_importable_without_lang_installed():
    # Importing the boundary modules themselves must never import the
    # external frameworks. If any of these failed, the whole import of
    # `orcha` above would have errored — assert explicitly anyway.
    for boundary in (LangChainBoundary(), LangGraphBoundary(), DeepAgentsBoundary(),
                     RagasBoundary(), PhoenixBoundary()):
        assert boundary.name


def test_missing_boundary_load_raises_with_hint():
    for boundary in (LangChainBoundary(), LangGraphBoundary(), DeepAgentsBoundary(),
                     RagasBoundary()):
        if boundary.status().available:
            continue
        with pytest.raises(IntegrationUnavailable) as exc:
            boundary.load()
        assert "orcha[" in str(exc.value)


def test_phoenix_otel_error_is_local_first():
    boundary = PhoenixBoundary()
    if boundary.status().available:
        pytest.skip("phoenix otel installed on this machine")
    with pytest.raises(IntegrationUnavailable, match="localhost:6006"):
        boundary.load_otel()


# ── MCP boundary (native, always available) ──────────────────────────────────

def test_mcp_boundary_always_available():
    status = McpBoundary().status()
    assert status.available is True
    assert status.name == "mcp"

def test_mcp_boundary_re_exports_native_client():
    from orcha.integrations.mcp import McpServer, build_mcp_tool
    assert callable(build_mcp_tool)
    assert McpServer is not None


# ── Execution contract seam ──────────────────────────────────────────────────

def test_execution_engine_contract_is_defined():
    import inspect
    assert issubclass(type(ExecutionEngine()), ExecutionEngine)
    params = inspect.signature(ExecutionEngine.execute).parameters
    assert "packet" in params and "ctx" in params


# ── RagasJudge still importable from the public evaluation API ───────────────

def test_ragas_judge_re_exported_from_evaluation():
    from orcha.evaluation import RagasJudge
    from orcha.integrations.ragas import RagasJudge as BoundaryRagasJudge
    assert RagasJudge is BoundaryRagasJudge
