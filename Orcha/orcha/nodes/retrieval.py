"""
orcha.nodes.retrieval
======================
Retrieval-as-a-plugin nodes for RAG, web search, and document lookup.

A RetrievalNode enriches the packet with relevant context from an external
source (vector store, web index, document corpus, etc.) before downstream
nodes (experts, agents, fact-checkers) consume it.

The retrieval mechanism is pluggable via a ``retriever`` function:
``async (query: str, top_k: int) -> list[str]``. This means any retrieval
backend — ChromaDB, Qdrant, Pinecone, simple keyword search, or even a
mock — can be dropped in without changing the node or the graph.

Pre-built retrievers are provided for common cases:
  - ``keyword_retriever``: simple keyword-overlap ranking over a document list.
  - ``embedding_retriever``: cosine-similarity ranking using pre-computed
    embeddings (requires numpy or a vector store).

Contract
--------
Input packet payload:
  - ``retrieval_query`` (str, optional): override the query used for retrieval.
    Defaults to ``packet.query``.

Output packet payload:
  - ``retrieval_results`` (list[str]): the top-K retrieved passages/snippets.
  - ``retrieval_count`` (int): number of results returned.
  - ``retrieval_source`` (str): identifier for the retrieval backend used.
  - ``retrieval_duration_ms`` (float): time spent retrieving.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..core.packets import OrchaPacket, PacketKind
from ..graph.context import RunContext
from ..graph.node import Node


# ── Types ────────────────────────────────────────────────────────────────────

RetrieverFn = Callable[[str, int], Any]
AsyncRetrieverFn = Callable[[str, int], Any]  # async version


@dataclass
class RetrievalConfig:
    """
    Configuration for a RetrievalNode.

    Attributes
    ----------
    retriever      The retrieval function: (query, top_k) -> list[str] or
                    list[dict]. May be sync or async.
    top_k           Number of results to retrieve (default 5).
    source_name     Identifier for the retrieval backend (for logging/tracing).
    query_key      Packet payload key to read the query from (default
                    "retrieval_query"; falls back to packet.query).
    result_key     Packet payload key to write results to (default
                    "retrieval_results").
    metadata       Additional metadata attached to results.
    """
    retriever: Optional[object] = None
    top_k: int = 5
    source_name: str = "unknown"
    query_key: str = "retrieval_query"
    result_key: str = "retrieval_results"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── RetrievalNode ─────────────────────────────────────────────────────────────

class RetrievalNode(Node):
    """
    A graph node that retrieves relevant context from an external source.

    The retriever function is called with the query and top_k; results are
    written into the packet payload for downstream consumption.

    If no retriever is configured, the node passes through unchanged (with
    empty results). This makes it safe to include in a graph even when the
    retrieval backend is not yet available.

    Parameters
    ----------
    name       Node name (default "retrieval").
    config     Retrieval configuration.
    timeout_s  Per-node timeout.
    retries    Retry budget.
    """

    def __init__(
        self,
        name: str = "retrieval",
        config: Optional[RetrievalConfig] = None,
        timeout_s: Optional[float] = 30.0,
        retries: int = 0,
    ) -> None:
        self.name = name
        self.timeout_s = timeout_s
        self.retries = retries
        self.config = config or RetrievalConfig()

    async def run(self, packet: OrchaPacket, ctx: RunContext) -> OrchaPacket:
        t0 = time.perf_counter()

        # Extract query.
        query = packet.payload.get(self.config.query_key) or packet.query

        # Call retriever.
        results: List[str] = []
        if self.config.retriever is not None:
            try:
                import asyncio
                import inspect
                raw = self.config.retriever(query, self.config.top_k)
                if inspect.isawaitable(raw):
                    raw = await raw
                # Normalize results to list of strings.
                if isinstance(raw, list):
                    for item in raw:
                        if isinstance(item, dict):
                            # Extract text from dict (supports {"text": ...} or {"content": ...}).
                            results.append(str(item.get("text") or item.get("content") or item.get("snippet") or item))
                        elif isinstance(item, str):
                            results.append(item)
                        else:
                            results.append(str(item))
                elif isinstance(raw, str):
                    results.append(raw)
            except Exception as exc:
                ctx.logger.warning(
                    "retrieval_error source=%s error=%s", self.config.source_name, exc,
                )
                results.append(f"[Retrieval error: {exc}]")

        duration_ms = (time.perf_counter() - t0) * 1000

        return packet.fork(
            packet.kind,
            **{self.config.result_key: results},
            retrieval_count=len(results),
            retrieval_source=self.config.source_name,
            retrieval_duration_ms=round(duration_ms, 2),
        )


# ── Pre-built retrievers ─────────────────────────────────────────────────────

def keyword_retriever(
    documents: List[str],
    case_sensitive: bool = False,
) -> RetrieverFn:
    """
    Build a simple keyword-overlap retriever over a static document list.

    Scores each document by the fraction of query tokens it contains.
    Returns the top-k documents as strings.

    This is a zero-dependency retriever for prototyping and testing. For
    production use, plug in a vector store retriever instead.

    Parameters
    ----------
    documents         List of document texts.
    case_sensitive     Whether matching should be case-sensitive.
    """
    def _retrieve(query: str, top_k: int) -> List[str]:
        if not documents:
            return []
        q_tokens = set(query.split())
        if not case_sensitive:
            q_tokens = {t.lower() for t in q_tokens}

        scored = []
        for i, doc in enumerate(documents):
            d_tokens = set(doc.split())
            if not case_sensitive:
                d_tokens = {t.lower() for t in d_tokens}
            if q_tokens:
                overlap = len(q_tokens & d_tokens) / len(q_tokens)
            else:
                overlap = 0.0
            scored.append((i, overlap, doc))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for _, _, doc in scored[:top_k]]

    return _retrieve


def embedding_retriever(
    documents: List[str],
    embeddings: Optional[List[List[float]]] = None,
) -> RetrieverFn:
    """
    Build a cosine-similarity retriever over pre-computed embeddings.

    If embeddings are not provided, falls back to keyword overlap
    (same as ``keyword_retriever``) with a warning.

    Parameters
    ----------
    documents    List of document texts.
    embeddings   Pre-computed embedding vectors (one per document).
    """
    import math

    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # If no embeddings provided, fall back to keyword retrieval.
    if embeddings is None or len(embeddings) != len(documents):
        return keyword_retriever(documents)

    doc_embeddings = list(embeddings)

    def _retrieve(query: str, top_k: int) -> List[str]:
        # Use a simple bag-of-words embedding for the query as a proxy.
        # In production, you'd call an embedding model here.
        query_tokens = set(query.lower().split())
        all_tokens = set()
        for doc in documents:
            all_tokens.update(doc.lower().split())

        token_index = {t: i for i, t in enumerate(sorted(all_tokens))}
        dim = len(token_index)

        # Build query embedding.
        q_emb = [0.0] * dim
        for t in query_tokens:
            if t in token_index:
                q_emb[token_index[t]] = 1.0

        # Score each document.
        scored = []
        for i, doc_emb in enumerate(doc_embeddings):
            sim = _cosine(q_emb, doc_emb)
            scored.append((i, sim, documents[i]))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for _, _, doc in scored[:top_k]]

    return _retrieve


__all__ = [
    "RetrievalNode", "RetrievalConfig",
    "RetrieverFn", "AsyncRetrieverFn",
    "keyword_retriever", "embedding_retriever",
]
