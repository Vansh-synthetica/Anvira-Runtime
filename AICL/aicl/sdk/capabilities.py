"""
Capability discovery for the AICL SDK.

Capabilities describe what operations a connected model or module supports.
This module provides structured discovery and filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Capability:
    """A single capability offered by a model or module."""

    name: str
    """Capability name (e.g., 'classify', 'generate', 'reasoning')."""

    version: str = "1.0"
    """Capability version."""

    description: str = ""
    """Human-readable description."""

    max_payload: int = 0
    """Maximum payload size in bytes (0 = unlimited)."""

    tags: tuple[str, ...] = ()
    """Searchable tags (e.g., ('text', 'multilingual'))."""

    def matches(self, query: str) -> bool:
        """Check if this capability matches a query string."""
        q = query.lower()
        return (
            q in self.name.lower()
            or q in self.description.lower()
            or any(q in t.lower() for t in self.tags)
        )


@dataclass
class CapabilitySet:
    """A collection of capabilities offered by a runtime or model."""

    capabilities: list[Capability] = field(default_factory=list)

    def add(self, cap: Capability) -> None:
        self.capabilities.append(cap)

    def find(self, query: str) -> list[Capability]:
        """Find capabilities matching a query string."""
        return [c for c in self.capabilities if c.matches(query)]

    def has(self, name: str) -> bool:
        """Check if a capability with the given name exists."""
        return any(c.name == name for c in self.capabilities)

    def get(self, name: str) -> Optional[Capability]:
        """Get a capability by exact name."""
        for c in self.capabilities:
            if c.name == name:
                return c
        return None

    def names(self) -> list[str]:
        return [c.name for c in self.capabilities]

    def __len__(self) -> int:
        return len(self.capabilities)

    def __iter__(self):
        return iter(self.capabilities)

    def __repr__(self) -> str:
        return f"<CapabilitySet {self.names()}>"


# ── Built-in AICL ISA capabilities ──────────────────────────────────────────

BUILTIN_CAPABILITIES = CapabilitySet([
    Capability("classify", description="Classify text into categories"),
    Capability("generate", description="Generate text from a prompt"),
    Capability("embed", description="Compute text embeddings"),
    Capability("reason", description="Chain-of-thought reasoning"),
    Capability("rank", description="Rank documents by relevance"),
    Capability("summarize", description="Summarize text"),
    Capability("verify", description="Verify claims against evidence"),
    Capability("extract", description="Extract structured data"),
    Capability("transform", description="Transform text"),
    Capability("plan", description="Generate action plans"),
    Capability("evaluate", description="Evaluate text quality"),
    Capability("model_call", description="Direct model invocation"),
    Capability("tool_call", description="Invoke external tools"),
    Capability("memory_read", description="Read from memory store"),
    Capability("memory_write", description="Write to memory store"),
    Capability("memory_delete", description="Delete from memory store"),
    Capability("index_query", description="Query the index"),
    Capability("index_upsert", description="Update the index"),
])
