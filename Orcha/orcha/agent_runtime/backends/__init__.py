"""
orcha.agent_runtime.backends
============================
Concrete ModelBackend implementations. Only OpenAI-compatible is shipped;
custom backends implement ``ModelBackend`` and register themselves with the
AgentRuntime via its constructor — nothing else needs to change.
"""
from .openai_compat import (
    DEFAULT_LLAMACPP_V1, DEFAULT_LMSTUDIO_V1, DEFAULT_OLLAMA_V1,
    OpenAICompatBackend, OpenAICompatBackendConfig,
)

__all__ = [
    "OpenAICompatBackend", "OpenAICompatBackendConfig",
    "DEFAULT_OLLAMA_V1", "DEFAULT_LLAMACPP_V1", "DEFAULT_LMSTUDIO_V1",
]
