"""
orcha.agent_runtime.errors
==========================
Error surface for the event-sourced AgentRuntime.
"""
from __future__ import annotations


class AgentRuntimeError(Exception):
    """Base class for all AgentRuntime failures."""


class EmptyLogError(AgentRuntimeError):
    """The Conversation loop was started before any user message was logged."""
