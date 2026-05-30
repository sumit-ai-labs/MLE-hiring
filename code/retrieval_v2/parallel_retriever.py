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
        chunks = retriever.retrieve(
            query,
            plan.route.company,
            plan.route.product_area,
            plan.route.confidence,
            top_k=top_k_per_query,
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


def _rank_key(chunk: DocumentChunk, query_index: int) -> tuple[float, float, int, str]:
    return (chunk.rerank_score, chunk.score, -query_index, chunk.path)


def _selection_reason(query: str, chunk: DocumentChunk, query_index: int) -> str:
    if chunk.heading and chunk.heading.lower() in query.lower():
        return f"heading_match_query_{query_index}"
    if chunk.rerank_score >= chunk.score:
        return f"rerank_specificity_query_{query_index}"
    return f"hybrid_score_query_{query_index}"

