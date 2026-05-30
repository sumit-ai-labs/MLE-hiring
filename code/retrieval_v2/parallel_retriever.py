"""Deterministic multi-query retrieval fanout and merge."""

from __future__ import annotations

from dataclasses import replace

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.markdown_chunker import DocumentChunk
from retrieval_v2.query_expander import RetrievalQueryPlan
from retrieval_v2.retrieval_trace import RetrievedDocTrace, RetrievalTrace


def retrieve_many(
    retriever: HybridRetriever,
    plan: RetrievalQueryPlan,
    top_k_per_query: int = 5,
    final_top_k: int = 5,
    seed_results: list[DocumentChunk] | None = None,
) -> tuple[list[DocumentChunk], RetrievalTrace]:
    trace = RetrievalTrace(
        company_route=plan.route.company,
        product_route=plan.route.product_area,
        issue_route=plan.route.issue_type,
        expanded_queries=list(plan.expanded_queries),
    )
    merged: dict[str, DocumentChunk] = {}
    first_seen: dict[str, int] = {}
    for query_index, query in enumerate(plan.expanded_queries):
        chunks = list(seed_results or []) if query_index == 0 and seed_results is not None else _cached_retrieve(
            retriever,
            query,
            plan.route.company,
            plan.route.product_area,
            plan.route.confidence,
            top_k_per_query,
        )
        for chunk in chunks:
            reason = _selection_reason(query, chunk, query_index)
            trace.retrieved_docs.append(
                RetrievedDocTrace(
                    query=query,
                    path=chunk.path,
                    bm25_score=chunk.bm25_score,
                    embedding_score=chunk.embedding_score,
                    rerank_score=chunk.rerank_score,
                    final_selection_reason=reason,
                )
            )
            if chunk.path not in merged or _rank_key(chunk, query_index) > _rank_key(merged[chunk.path], first_seen[chunk.path]):
                merged[chunk.path] = replace(chunk)
                first_seen[chunk.path] = query_index
    ranked = sorted(
        merged.values(),
        key=lambda chunk: (chunk.rerank_score, chunk.score, -first_seen.get(chunk.path, 999), chunk.path, chunk.heading),
        reverse=True,
    )
    return ranked[:final_top_k], trace


def _cached_retrieve(
    retriever: HybridRetriever,
    query: str,
    company: str,
    product_area: str,
    confidence: float,
    top_k: int,
) -> list[DocumentChunk]:
    cache = getattr(retriever, "_v2_retrieval_cache", None)
    if cache is None:
        cache = {}
        setattr(retriever, "_v2_retrieval_cache", cache)
    key = (query, company, product_area, round(float(confidence), 3), int(top_k))
    if key not in cache:
        cache[key] = tuple(retriever.retrieve(query, company, product_area, confidence, top_k=top_k))
        if len(cache) > 512:
            oldest = next(iter(cache))
            cache.pop(oldest, None)
    return [replace(chunk) for chunk in cache[key]]


def _rank_key(chunk: DocumentChunk, query_index: int) -> tuple[float, float, int, str]:
    return (chunk.rerank_score, chunk.score, -query_index, chunk.path)


def _selection_reason(query: str, chunk: DocumentChunk, query_index: int) -> str:
    if chunk.heading and chunk.heading.lower() in query.lower():
        return f"heading_match_query_{query_index}"
    if chunk.rerank_score >= chunk.score:
        return f"rerank_specificity_query_{query_index}"
    return f"hybrid_score_query_{query_index}"
