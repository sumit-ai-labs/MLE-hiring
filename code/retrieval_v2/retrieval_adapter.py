"""Shadow adapter for V2 advanced retrieval routing."""

from __future__ import annotations

from agent.classifier import ClassificationResult
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.markdown_chunker import DocumentChunk
from retrieval_v2.parallel_retriever import retrieve_many
from retrieval_v2.query_expander import expand_query
from retrieval_v2.router import route_retrieval


def run_shadow_retrieval(
    retriever: HybridRetriever,
    v1_chunks: list[DocumentChunk],
    query: str,
    classification: ClassificationResult,
    support_text: str,
) -> tuple[list[DocumentChunk], str]:
    route = route_retrieval(classification, " ".join([query, support_text]))
    plan = expand_query(query, route)
    v2_chunks, trace = retrieve_many(retriever, plan)
    trace.mismatch_with_v1 = _paths(v1_chunks) != _paths(v2_chunks)
    return v1_chunks, trace.to_json()


def _paths(chunks: list[DocumentChunk]) -> tuple[str, ...]:
    return tuple(chunk.path for chunk in chunks[:5])
