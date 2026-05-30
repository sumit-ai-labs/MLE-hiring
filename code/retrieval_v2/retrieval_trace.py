"""Explainable V2 retrieval traces."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievedDocTrace:
    query: str
    path: str
    bm25_score: float
    embedding_score: float
    rerank_score: float
    final_selection_reason: str


@dataclass
class RetrievalTrace:
    company_route: str
    product_route: str
    issue_route: str
    expanded_queries: list[str]
    retrieved_docs: list[RetrievedDocTrace] = field(default_factory=list)
    mismatch_with_v1: bool = False

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "company_route": self.company_route,
            "product_route": self.product_route,
            "issue_route": self.issue_route,
            "expanded_queries": self.expanded_queries,
            "retrieved_docs": [asdict(doc) for doc in self.retrieved_docs],
            "mismatch_with_v1": self.mismatch_with_v1,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=True)

    def human_readable(self) -> str:
        top = self.retrieved_docs[:5]
        docs = "; ".join(f"{doc.path} rerank={doc.rerank_score:.3f}" for doc in top)
        return (
            f"company={self.company_route}; product={self.product_route}; "
            f"issue={self.issue_route}; queries={self.expanded_queries}; docs=[{docs}]"
        )

