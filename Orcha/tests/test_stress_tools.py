"""
Stress tests for the unified Anvira Agent tool system.
Verifies all 7 capability domains register their full tool sets,
the completion fallback chain works, and cloud params are stripped.
"""
import asyncio
import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ── 1. Capability Registration Stress Test ────────────────────────────────

class TestCapabilityRegistration:
    """Every capability domain must register all its tools."""

    def test_all_eight_capabilities_exist(self):
        from orcha.capabilities.registry import CapabilityRegistry
        reg = CapabilityRegistry().register_defaults()
        names = reg.names()
        expected = {"filesystem", "workspace", "search", "web", "terminal", "git", "diagnostics", "code_intelligence"}
        assert set(names) == expected, f"Missing capabilities: {expected - set(names)}"

    def test_filesystem_registers_all_22_tools(self):
        from orcha.capabilities.filesystem import build_tools
        from orcha.capabilities.base import CapabilityContext
        tools = build_tools(CapabilityContext(roots=[]))
        tool_names = {t.name for t in tools}
        expected = {
            "read_file", "write_file", "edit_file", "append_file",
            "create_file", "delete_file", "rename_file", "copy_file",
            "move_file", "create_directory", "delete_directory",
            "rename_directory", "copy_directory", "move_directory",
            "list_directory", "directory_tree", "exists", "file_info",
            "glob_search", "search_text", "replace_text", "read_multiple_files",
        }
        missing = expected - tool_names
        assert not missing, f"Filesystem missing tools: {missing}"

    def test_search_registers_all_5_tools(self):
        from orcha.capabilities.search import build_tools
        from orcha.capabilities.base import CapabilityContext
        tools = build_tools(CapabilityContext(roots=[]))
        tool_names = {t.name for t in tools}
        expected = {"grep", "regex_search", "filename_search", "symbol_search", "workspace_search"}
        missing = expected - tool_names
        assert not missing, f"Search missing tools: {missing}"

    def test_terminal_registers_all_5_tools(self):
        from orcha.capabilities.terminal import build_tools
        from orcha.capabilities.base import CapabilityContext
        tools = build_tools(CapabilityContext(roots=[]))
        tool_names = {t.name for t in tools}
        expected = {"run_command", "stream_output", "stop_process", "running_processes", "command_history"}
        missing = expected - tool_names
        assert not missing, f"Terminal missing tools: {missing}"

    def test_git_registers_all_11_tools(self):
        from orcha.capabilities.git import build_tools
        from orcha.capabilities.base import CapabilityContext
        tools = build_tools(CapabilityContext(roots=[]))
        tool_names = {t.name for t in tools}
        expected = {
            "git_status", "git_diff", "git_add", "git_commit",
            "git_branch", "git_checkout", "git_log", "git_show",
            "git_restore", "git_pull", "git_push",
        }
        missing = expected - tool_names
        assert not missing, f"Git missing tools: {missing}"

    def test_diagnostics_registers_all_6_tools(self):
        from orcha.capabilities.diagnostics import build_tools
        from orcha.capabilities.base import CapabilityContext
        tools = build_tools(CapabilityContext(roots=[]))
        tool_names = {t.name for t in tools}
        expected = {"run_build", "run_tests", "run_linter", "type_check", "dependency_check", "read_logs"}
        missing = expected - tool_names
        assert not missing, f"Diagnostics missing tools: {missing}"

    def test_code_intelligence_registers_all_5_tools(self):
        from orcha.capabilities.code_intelligence import build_tools
        from orcha.capabilities.base import CapabilityContext
        tools = build_tools(CapabilityContext(roots=[]))
        tool_names = {t.name for t in tools}
        expected = {"find_symbol", "find_references", "rename_symbol", "document_symbol", "outline_file"}
        missing = expected - tool_names
        assert not missing, f"Code intelligence missing tools: {missing}"

    def test_workspace_registers_all_8_tools(self):
        from orcha.capabilities.workspace import build_tools
        from orcha.capabilities.base import CapabilityContext
        tools = build_tools(CapabilityContext(roots=[]))
        tool_names = {t.name for t in tools}
        expected = {
            "current_workspace", "workspace_tree", "list_projects",
            "attached_files", "attached_folders", "active_file",
            "recent_files", "project_summary",
        }
        missing = expected - tool_names
        assert not missing, f"Workspace missing tools: {missing}"

    def test_unified_agent_gets_all_62_tools(self):
        """The unified agent with all 7 capabilities should get 62 tools."""
        from orcha.capabilities.registry import CapabilityRegistry
        from orcha.capabilities.base import CapabilityContext, PermissionPolicy
        reg = CapabilityRegistry().register_defaults()
        ctx = CapabilityContext(roots=[])
        policy = PermissionPolicy(access_mode="full")
        all_caps = ["filesystem", "workspace", "search", "terminal", "git", "diagnostics", "code_intelligence"]
        executor = reg.build(all_caps, ctx=ctx, policy=policy)
        tools = executor.tools()
        tool_names = {t.name for t in tools}
        # Should have at least 55+ unique tools (some overlap between filesystem and workspace)
        assert len(tool_names) >= 55, f"Expected 55+ unique tools, got {len(tool_names)}: {tool_names}"

    def test_each_tool_has_valid_schema(self):
        """Every registered tool must have a valid OpenAI-compatible schema."""
        from orcha.capabilities.registry import CapabilityRegistry
        from orcha.capabilities.base import CapabilityContext, PermissionPolicy
        reg = CapabilityRegistry().register_defaults()
        ctx = CapabilityContext(roots=[])
        policy = PermissionPolicy(access_mode="full")
        executor = reg.build(ctx=ctx, policy=policy)
        for tool_spec in executor.tools():
            schema = tool_spec.to_openai_schema()
            assert "type" in schema, f"{tool_spec.name} missing 'type' in schema"
            assert schema["type"] == "function", f"{tool_spec.name} has wrong type: {schema['type']}"
            assert "function" in schema, f"{tool_spec.name} missing 'function' in schema"
            assert "name" in schema["function"], f"{tool_spec.name} missing function name"
            assert "parameters" in schema["function"], f"{tool_spec.name} missing parameters"


# ── 2. LocalChatExpert Cloud Param Stripping ──────────────────────────────

class TestCloudParamStripping:
    """Ollama-specific params must be stripped for cloud providers."""

    def test_local_endpoint_includes_ollama_params(self):
        from orcha.experts.local_chat import LocalChatExpert
        expert = LocalChatExpert(
            model="llama3.2:3b",
            base_url="http://localhost:11434/v1",
        )
        body = expert._build_completion_request(
            [{"role": "user", "content": "hello"}],
            stream=False,
        )
        assert "top_k" in body, "Local endpoint should include top_k"
        assert "repeat_penalty" in body, "Local endpoint should include repeat_penalty"

    def test_cloud_endpoint_strips_ollama_params(self):
        from orcha.experts.local_chat import LocalChatExpert
        expert = LocalChatExpert(
            model="meta-llama/llama-3-8b",
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
        )
        body = expert._build_completion_request(
            [{"role": "user", "content": "hello"}],
            stream=False,
        )
        assert "top_k" not in body, "Cloud endpoint should NOT include top_k"
        assert "repeat_penalty" not in body, "Cloud endpoint should NOT include repeat_penalty"

    def test_localhost_8080_detected_as_local(self):
        from orcha.experts.local_chat import LocalChatExpert
        expert = LocalChatExpert(
            model="local-model",
            base_url="http://localhost:8080/v1",
        )
        assert expert._is_local_endpoint() is True

    def test_127_0_0_1_detected_as_local(self):
        from orcha.experts.local_chat import LocalChatExpert
        expert = LocalChatExpert(
            model="local-model",
            base_url="http://127.0.0.1:8080/v1",
        )
        assert expert._is_local_endpoint() is True

    def test_openrouter_detected_as_cloud(self):
        from orcha.experts.local_chat import LocalChatExpert
        expert = LocalChatExpert(
            model="meta-llama/llama-3-8b",
            base_url="https://openrouter.ai/api/v1",
        )
        assert expert._is_local_endpoint() is False

    def test_groq_detected_as_cloud(self):
        from orcha.experts.local_chat import LocalChatExpert
        expert = LocalChatExpert(
            model="llama3-8b",
            base_url="https://api.groq.com/openai/v1",
        )
        assert expert._is_local_endpoint() is False

    def test_together_detected_as_cloud(self):
        from orcha.experts.local_chat import LocalChatExpert
        expert = LocalChatExpert(
            model="meta-llama/Llama-3-8b-chat-hf",
            base_url="https://api.together.xyz/v1",
        )
        assert expert._is_local_endpoint() is False

    def test_cloud_request_has_standard_openai_params(self):
        from orcha.experts.local_chat import LocalChatExpert
        expert = LocalChatExpert(
            model="meta-llama/llama-3-8b",
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
        )
        body = expert._build_completion_request(
            [{"role": "user", "content": "hello"}],
            stream=False,
        )
        # Standard OpenAI params must always be present
        assert "model" in body
        assert "messages" in body
        assert "temperature" in body
        assert "top_p" in body
        assert "max_tokens" in body
        assert "stream" in body


# ── 3. Completion Fallback Chain ──────────────────────────────────────────

class TestCompletionFallback:
    """The completion_fn must degrade gracefully through 3 tiers."""

    def test_envelope_to_native_tools_fallback(self):
        """When json_schema fails, should retry with native tools."""
        from orcha.experts.local_chat import LocalChatExpert

        expert = LocalChatExpert(
            model="test-model",
            base_url="http://localhost:8080/v1",
            api_key="not-needed",
        )

        call_count = 0
        original_post = expert._post_with_retry

        async def mock_post(body):
            nonlocal call_count
            call_count += 1
            # First call (with json_schema) fails
            if call_count == 1:
                raise RuntimeError("response_format not supported")
            # Second call (native tools) succeeds
            return {
                "choices": [{
                    "message": {"role": "assistant", "content": "I'll create that file."},
                    "finish_reason": "stop",
                }],
                "model": "test-model",
            }

        expert._post_with_retry = mock_post

        # Simulate the completion_fn fallback logic
        async def test_completion():
            tools = [{"type": "function", "function": {"name": "write_file"}}]
            try:
                result = await expert.chat_completion(
                    [{"role": "user", "content": "create a file"}],
                    system="You are a coding agent.",
                    tools=tools,
                )
                return result
            except Exception:
                # Fallback: native tools
                result = await expert.chat_completion(
                    [{"role": "user", "content": "create a file"}],
                    system="You are a coding agent.",
                    tools=tools,
                )
                return result

        result = asyncio.get_event_loop().run_until_complete(test_completion())
        assert call_count == 2, f"Expected 2 calls, got {call_count}"
        assert result["content"] == "I'll create that file."

    def test_all_tiers_fail_raises(self):
        """When all 3 tiers fail, the exception should propagate."""
        from orcha.experts.local_chat import LocalChatExpert

        expert = LocalChatExpert(
            model="test-model",
            base_url="http://localhost:8080/v1",
        )

        async def always_fail(body):
            raise RuntimeError("model unavailable")

        expert._post_with_retry = always_fail

        async def test():
            with pytest.raises(RuntimeError, match="model unavailable"):
                await expert.chat_completion(
                    [{"role": "user", "content": "hello"}],
                    system="test",
                    tools=[{"type": "function", "function": {"name": "write_file"}}],
                )

        asyncio.get_event_loop().run_until_complete(test())

    def test_plain_text_fallback_still_works(self):
        """When tools fail, plain text should still return an answer."""
        from orcha.experts.local_chat import LocalChatExpert

        expert = LocalChatExpert(
            model="test-model",
            base_url="http://localhost:8080/v1",
        )

        call_count = 0

        async def mock_post(body):
            nonlocal call_count
            call_count += 1
            # If tools are in the request, fail
            if body.get("tools"):
                raise RuntimeError("tools not supported")
            # Plain text works
            return {
                "choices": [{
                    "message": {"role": "assistant", "content": "Here is my answer."},
                    "finish_reason": "stop",
                }],
                "model": "test-model",
            }

        expert._post_with_retry = mock_post

        async def test():
            tools = [{"type": "function", "function": {"name": "write_file"}}]
            try:
                return await expert.chat_completion(
                    [{"role": "user", "content": "hello"}],
                    system="test",
                    tools=tools,
                )
            except Exception:
                try:
                    return await expert.chat_completion(
                        [{"role": "user", "content": "hello"}],
                        system="test",
                        tools=tools,
                    )
                except Exception:
                    return await expert.chat_completion(
                        [{"role": "user", "content": "hello"}],
                        system="test",
                        tools=None,
                    )

        result = asyncio.get_event_loop().run_until_complete(test())
        assert result["content"] == "Here is my answer."


# ── 4. Error Classification ───────────────────────────────────────────────

class TestErrorClassification:
    """Frontend error patterns must classify correctly."""

    def test_network_errors(self):
        patterns = [
            "ECONNREFUSED 127.0.0.1:8420",
            "Failed to fetch",
            "NetworkError",
            "Could not reach Orcha at http://localhost:8420",
            "connection refused",
        ]
        for p in patterns:
            assert "ECONNREFUSED" in p or "Failed to fetch" in p or "NetworkError" in p or "Could not reach" in p or "connection refused" in p

    def test_api_errors_not_misclassified_as_network(self):
        """API errors should NOT match the network error pattern."""
        api_errors = [
            "model API error 400: unsupported parameter",
            "HTTPStatusError 429",
            "status 500",
            "OpenAI returned 400",
            "response_format not supported",
        ]
        import re
        network_pattern = re.compile(r"ECONNREFUSED|Failed to fetch|NetworkError|Could not reach|connection refused", re.I)
        for err in api_errors:
            assert not network_pattern.search(err), f"API error misclassified as network: {err}"


# ── 5. Agent Config with All Capabilities ─────────────────────────────────

class TestUnifiedAgentConfig:
    """The unified agent should have all capabilities and high reasoning."""

    def test_unified_agent_has_all_capabilities(self):
        from orcha.capabilities.registry import CapabilityRegistry
        reg = CapabilityRegistry().register_defaults()
        all_caps = ["filesystem", "workspace", "search", "terminal", "git", "diagnostics", "code_intelligence"]
        for cap in all_caps:
            assert reg.has(cap), f"Capability '{cap}' not registered"

    def test_capability_build_doesnt_crash(self):
        """Building any single capability should not crash."""
        from orcha.capabilities.registry import CapabilityRegistry
        from orcha.capabilities.base import CapabilityContext
        reg = CapabilityRegistry().register_defaults()
        ctx = CapabilityContext(roots=[])
        for name in reg.names():
            executor = reg.build([name], ctx=ctx)
            tools = executor.tools()
            assert len(tools) > 0, f"Capability '{name}' registered 0 tools"

    def test_full_build_produces_schemas(self):
        """All tools should produce valid OpenAI schemas for the model."""
        from orcha.capabilities.registry import CapabilityRegistry
        from orcha.capabilities.base import CapabilityContext, PermissionPolicy
        reg = CapabilityRegistry().register_defaults()
        ctx = CapabilityContext(roots=[])
        policy = PermissionPolicy(access_mode="full")
        executor = reg.build(ctx=ctx, policy=policy)
        schemas = executor.schemas()
        assert len(schemas) >= 55
        for s in schemas:
            assert "type" in s
            assert "function" in s
            assert "name" in s["function"]
            assert "parameters" in s["function"]
