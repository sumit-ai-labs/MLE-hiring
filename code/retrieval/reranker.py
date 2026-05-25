"""Deterministic reranking for retrieved chunks."""

from __future__ import annotations

import re
from dataclasses import replace

from agent.safety import analyze_doc_poisoning
from retrieval.markdown_chunker import DocumentChunk


_CROSS_ENCODER = None
_CROSS_ENCODER_AVAILABLE: bool | None = None


def rerank(
    query: str,
    chunks: list[DocumentChunk],
    company: str,
    product_area: str,
    top_k: int,
) -> list[DocumentChunk]:
    cross_ranked = _cross_encoder_rerank(query, chunks, company, product_area)
    if cross_ranked is not None:
        return cross_ranked[:top_k]
    return heuristic_rerank(query, chunks, company, product_area, top_k)


def heuristic_rerank(
    query: str,
    chunks: list[DocumentChunk],
    company: str,
    product_area: str,
    top_k: int,
) -> list[DocumentChunk]:
    query_tokens = set(_tokens(query))
    for chunk in chunks:
        heading_tokens = set(_tokens(chunk.heading))
        path_tokens = set(_tokens(chunk.path))
        specificity = 0.0
        if company != "none" and chunk.company == company:
            specificity += 0.08
        if product_area and product_area in {chunk.product, chunk.category, chunk.heading.lower().replace(" ", "_")}:
            specificity += 0.06
        if heading_tokens & query_tokens:
            specificity += min(0.08, 0.015 * len(heading_tokens & query_tokens))
        if path_tokens & query_tokens:
            specificity += min(0.05, 0.01 * len(path_tokens & query_tokens))
        poison_penalty = 0.18 if analyze_doc_poisoning(chunk.content) else 0.0
        chunk.rerank_score = round(chunk.score + specificity - poison_penalty, 6)
    return sorted(chunks, key=lambda c: (c.rerank_score, c.score, c.path, c.heading), reverse=True)[:top_k]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]{3,}", (text or "").lower())


def _cross_encoder_rerank(
    query: str,
    chunks: list[DocumentChunk],
    company: str,
    product_area: str,
) -> list[DocumentChunk] | None:
    model = _load_cross_encoder()
    if model is None or not chunks:
        return None
    try:
        pairs = [(query, f"{chunk.heading}\n{chunk.content[:1200]}") for chunk in chunks]
        scores = model.predict(pairs, show_progress_bar=False)
        ranked: list[DocumentChunk] = []
        min_score = min(float(score) for score in scores)
        max_score = max(float(score) for score in scores)
        spread = max(max_score - min_score, 1e-9)
        for chunk, raw_score in zip(chunks, scores):
            semantic_score = (float(raw_score) - min_score) / spread
            specificity = 0.0
            if company != "none" and chunk.company == company:
                specificity += 0.03
            if product_area and product_area in {chunk.product, chunk.category, chunk.heading.lower().replace(" ", "_")}:
                specificity += 0.03
            poison_penalty = 0.2 if analyze_doc_poisoning(chunk.content) else 0.0
            ranked.append(replace(chunk, rerank_score=round(0.85 * semantic_score + 0.15 * chunk.score + specificity - poison_penalty, 6)))
        return sorted(ranked, key=lambda c: (c.rerank_score, c.score, c.path, c.heading), reverse=True)
    except Exception:
        return None


def _load_cross_encoder():
    global _CROSS_ENCODER, _CROSS_ENCODER_AVAILABLE
    if _CROSS_ENCODER_AVAILABLE is False:
        return None
    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER
    try:
        from sentence_transformers import CrossEncoder

        try:
            _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", local_files_only=True)
        except TypeError:
            _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        _CROSS_ENCODER_AVAILABLE = True
        return _CROSS_ENCODER
    except Exception:
        _CROSS_ENCODER_AVAILABLE = False
        return None
