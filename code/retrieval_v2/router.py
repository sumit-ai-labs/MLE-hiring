"""Hierarchical deterministic retrieval routing."""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.classifier import ClassificationResult


@dataclass(frozen=True)
class RetrievalRoute:
    company: str
    product_area: str
    issue_type: str
    strategy: str
    confidence: float


ISSUE_HINTS: list[tuple[str, re.Pattern[str]]] = [
    ("refund", re.compile(r"\b(refund|reimburse|money back|chargeback|duplicate charge|wrong charge|reembolso|remboursement|erstattung)\b", re.I)),
    ("security", re.compile(r"\b(hacked|compromised|account takeover|unauthorized login|fraud|stolen|pirate|comprometida)\b", re.I)),
    ("subscription", re.compile(r"\b(subscription|plan|downgrade|upgrade|cancel|billing tier|suscripcion|abonnement)\b", re.I)),
    ("technical_support", re.compile(r"\b(error|bug|endpoint|deployment|sandbox|api|timeout|failure)\b", re.I)),
    ("travel_claim", re.compile(r"\b(travel|trip|reimbursement|benefit|emergency assistance|viaje)\b", re.I)),
    ("feature_request", re.compile(r"\b(feature|add support|request|enhancement)\b", re.I)),
]


def route_retrieval(classification: ClassificationResult, text: str) -> RetrievalRoute:
    issue_type = _issue_type(classification.internal_request_type, text)
    product_area = classification.product_area or _product_for_issue(issue_type)
    strategy = _strategy(issue_type, classification.company, product_area)
    confidence = _route_confidence(classification.confidence, issue_type, text)
    return RetrievalRoute(
        company=classification.company,
        product_area=product_area,
        issue_type=issue_type,
        strategy=strategy,
        confidence=confidence,
    )


def _issue_type(internal_request_type: str, text: str) -> str:
    if internal_request_type in {"refund", "billing"}:
        return "refund"
    if internal_request_type in {"account_compromise", "fraud"}:
        return "security"
    if internal_request_type in {"subscription", "technical_issue", "feature_request", "bug", "legal"}:
        return internal_request_type
    for label, pattern in ISSUE_HINTS:
        if pattern.search(text or ""):
            return label
    return internal_request_type or "general"


def _product_for_issue(issue_type: str) -> str:
    return {
        "refund": "billing",
        "security": "security",
        "subscription": "subscription",
        "technical_issue": "technical_support",
        "bug": "technical_support",
        "travel_claim": "travel_support",
    }.get(issue_type, "general_support")


def _strategy(issue_type: str, company: str, product_area: str) -> str:
    if issue_type in {"refund", "security", "subscription"}:
        return f"{company}:{product_area}:{issue_type}:expanded"
    return f"{company}:{product_area}:{issue_type}:focused"


def _route_confidence(classifier_confidence: float, issue_type: str, text: str) -> float:
    bonus = 0.08 if issue_type != "general" else 0.0
    if re.search(r"\b(txn_|refund|hacked|fraud|subscription|endpoint|travel)\b", text or "", re.I):
        bonus += 0.04
    return round(min(1.0, max(0.0, classifier_confidence + bonus)), 3)

