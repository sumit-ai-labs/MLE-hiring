"""Deterministic query expansion for V2 retrieval shadow mode."""

from __future__ import annotations

from dataclasses import dataclass

from retrieval_v2.router import RetrievalRoute


@dataclass(frozen=True)
class RetrievalQueryPlan:
    base_query: str
    expanded_queries: tuple[str, ...]
    route: RetrievalRoute


SYNONYMS: dict[str, tuple[str, ...]] = {
    "refund": ("duplicate charge", "billing issue", "wrong charge", "payment reversal", "chargeback"),
    "security": ("account takeover", "compromised account", "unauthorized login", "fraud", "password reset"),
    "subscription": ("plan downgrade", "plan change", "cancel plan", "billing tier"),
    "technical_issue": ("error message", "endpoint failure", "deployment issue", "sandbox access"),
    "bug": ("error message", "unexpected behavior", "broken workflow"),
    "travel_claim": ("travel reimbursement", "trip benefits", "emergency assistance"),
    "feature_request": ("product enhancement", "feature request", "add support"),
}


def expand_query(base_query: str, route: RetrievalRoute, max_queries: int = 5) -> RetrievalQueryPlan:
    clean_base = " ".join((base_query or "").split())
    queries: list[str] = [clean_base]
    for synonym in SYNONYMS.get(route.issue_type, ()):
        expanded = " ".join(part for part in [clean_base, synonym, route.product_area] if part).strip()
        if expanded and expanded not in queries:
            queries.append(expanded)
        if len(queries) >= max_queries:
            break
    return RetrievalQueryPlan(base_query=clean_base, expanded_queries=tuple(queries), route=route)

