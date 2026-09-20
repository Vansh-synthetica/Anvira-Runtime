"""Dependency-free BM25 + paragraph chunking.

Faithful port of Anvira's ``src/utils/bm25.ts`` (same stopword list, same
tokenizer, same 900/200-char chunking, Okapi BM25 with k1=1.5, b=0.75) so a
document ranks identically whether it is scored in the renderer or in the
runtime. Product-neutral: Notes, Study, Dev and chat all need "which pieces of
this text answer that question", and this is the one implementation.
"""
from __future__ import annotations

import math
import re
from typing import Any

STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the their then there "
    "these they this to was will with what when where who how why do does i you we my your our can could "
    "would should".split())

CHUNK_TARGET_CHARS = 900
CHUNK_MIN_CHARS = 200
K1, B = 1.5, 0.75

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]


def chunk_text(tag: str, text: str) -> list[dict[str, str]]:
    """Paragraph-aligned chunks near ``CHUNK_TARGET_CHARS`` (small paragraphs merge, huge ones hard-split)."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[dict[str, str]] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.append({"id": f"{tag}#{len(chunks)}", "tag": tag, "text": buffer.strip()})
        buffer = ""

    for para in paragraphs:
        if len(para) > CHUNK_TARGET_CHARS * 1.5:
            flush()
            for i in range(0, len(para), CHUNK_TARGET_CHARS):
                chunks.append({"id": f"{tag}#{len(chunks)}", "tag": tag, "text": para[i:i + CHUNK_TARGET_CHARS]})
            continue
        if buffer and len(buffer) + len(para) > CHUNK_TARGET_CHARS:
            flush()
        buffer = f"{buffer}\n\n{para}" if buffer else para
        if len(buffer) >= CHUNK_MIN_CHARS and len(buffer) >= CHUNK_TARGET_CHARS:
            flush()
    flush()
    return chunks


def rank_chunks(chunks: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Rank ``chunks`` (each with a ``text``) against ``query``; zero-overlap chunks score 0 and sort last."""
    docs = [tokenize(c["text"]) for c in chunks]
    lengths = [len(d) for d in docs]
    avg_len = sum(lengths) / (len(lengths) or 1)
    df: dict[str, int] = {}
    for tokens in docs:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    n = len(docs)
    q_terms = tokenize(query)
    scored = []
    for chunk, tokens, length in zip(chunks, docs, lengths):
        counts: dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        score = 0.0
        for term in q_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            nt = df.get(term, 0)
            idf = math.log(1 + (n - nt + 0.5) / (nt + 0.5))
            score += idf * (tf * (K1 + 1)) / (tf + K1 * (1 - B + (B * length) / (avg_len or 1)))
        scored.append({**chunk, "score": score})
    return sorted(scored, key=lambda c: c["score"], reverse=True)
