"""Resolve contradictory multi-turn intents before rule decisions."""

from __future__ import annotations

import re
from dataclasses import replace

from agent.classifier import ClassificationResult, detect_product_area, map_request_type
from utils.parser import ConversationState


PRIORITY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("account_compromise", re.compile(r"\b(hacked|compromised|account takeover|unauthorized access|someone logged in|stolen account)\b", re.I)),
    ("legal", re.compile(r"\b(sue|lawsuit|lawyer|attorney|legal|court|regulator|gdpr complaint|demand\w*|abogado)\b", re.I)),
    ("fraud", re.compile(r"\b(fraud|fraudulent|identity theft|stolen card|unauthorized charge|unauthorized transaction)\b", re.I)),
    ("refund", re.compile(r"\b(refund|money back|give me my money|reimburse)\b", re.I)),
    ("subscription", re.compile(r"\b(cancel|upgrade|downgrade|pause|subscription|plan)\b", re.I)),
    ("technical_issue", re.compile(r"\b(api|endpoint|deployment|sandbox|429|rate limit|login|password|access|error|bug|broken)\b", re.I)),
]

PRIORITY_ORDER = {
    "account_compromise": 0,
    "legal": 1,
    "fraud": 2,
    "refund": 3,
    "billing": 3,
    "subscription": 4,
    "technical_issue": 5,
    "bug": 5,
    "faq": 6,
    "invalid": 7,
}


def resolve_conflicts(state: ConversationState, classification: ClassificationResult) -> ClassificationResult:
    text = " ".join([state.subject, state.full_text])
    detected = [intent for intent, pattern in PRIORITY_PATTERNS if pattern.search(text)]
    if not detected:
        return classification
    best = sorted(detected + [classification.internal_request_type], key=lambda item: PRIORITY_ORDER.get(item, 99))[0]
    if best == classification.internal_request_type:
        return classification
    product_area = detect_product_area(text.lower(), classification.company, best)
    return replace(
        classification,
        internal_request_type=best,
        final_request_type=map_request_type(best, classification.company),
        product_area=product_area or classification.product_area,
    )
