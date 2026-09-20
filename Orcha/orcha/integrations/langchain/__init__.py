"""
orcha.integrations.langchain
============================
Boundary around LangChain — the primary component / model / tool
abstraction layer.

Orcha core never imports langchain directly; it goes through this
boundary. The boundary is lazy: ``LangChainBoundary()`` is cheap, and
``load()`` / ``load_core()`` import langchain / langchain_core on
demand, raising ``IntegrationUnavailable`` with an install hint when
the optional ``orcha[lang]`` extra is missing.

Adapters in this package:
- ``messages``   Orcha message dicts ↔ LangChain BaseMessage.
- ``tools``      Orcha Tool ↔ LangChain StructuredTool (execution stays
                 in Orcha's permission boundary in both directions).
- ``backend``    LangChainModelBackend — any LangChain chat model behind
                 Orcha's ModelBackend contract (local models first-class:
                 Ollama / llama.cpp / OpenAI-compatible localhost).

LangChain is the provider-interoperability surface, while Orcha keeps
ownership of policy, permissions, workspace context, and the public
API contract.
"""
from __future__ import annotations

from ..base import Boundary
from .backend import LangChainModelBackend
from .messages import (
    langchain_message_to_orcha, orcha_message_to_langchain,
    orcha_messages_to_langchain,
)
from .tools import langchain_tool_to_orcha, orcha_tool_to_langchain

__all__ = [
    "LangChainBoundary",
    "LangChainModelBackend",
    "orcha_message_to_langchain", "orcha_messages_to_langchain",
    "langchain_message_to_orcha",
    "orcha_tool_to_langchain", "langchain_tool_to_orcha",
]


class LangChainBoundary(Boundary):
    name = "langchain"
    package = "langchain"
    extra = "lang"

    def load(self) -> object:
        """Return the ``langchain`` module (raises if not installed)."""
        return self._import("langchain")

    def load_core(self) -> object:
        """Return the ``langchain_core`` module (raises if not installed)."""
        return self._import("langchain_core")