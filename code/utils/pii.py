"""Deterministic PII detection and masking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class PIIResult:
    detected: bool
    categories: list[str] = field(default_factory=list)
    masked_text: str = ""


PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    ("api_key", re.compile(r"\b(?:sk-ant|sk|api[_-]?key|token)[-_]?[A-Za-z0-9]{12,}\b", re.I)),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("dob", re.compile(r"\b(?:dob|date of birth|born)\s*[:\-]?\s*(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b", re.I)),
    ("phone", re.compile(r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3,5}\)?[-.\s]?)\d{3,4}[-.\s]?\d{4}(?!\d)")),
    ("passport", re.compile(r"\b(?:passport|ppn)\s*(?:number|no\.?|#)?\s*[:#]?\s*[A-Z][A-Z0-9]{6,9}\b", re.I)),
    ("account_id", re.compile(r"\b(?:account|customer|user)\s*(?:id|number|#)\s*[:#]?\s*[A-Za-z0-9_-]{5,}\b", re.I)),
    ("address", re.compile(r"\b\d{1,6}\s+[A-Za-z0-9.'-]+(?:\s+[A-Za-z0-9.'-]+){0,5}\s+(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|boulevard|blvd|way|apt|suite)\b", re.I)),
]

CARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def detect_pii(text: str) -> PIIResult:
    masked = text or ""
    categories: list[str] = []
    for match in list(CARD_PATTERN.finditer(masked)):
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_like(digits):
            categories.append("payment_card")
            masked = masked.replace(match.group(0), "[REDACTED_PAYMENT_CARD]")
    for name, pattern in PATTERNS:
        if pattern.search(masked):
            categories.append(name)
            masked = pattern.sub(f"[REDACTED_{name.upper()}]", masked)
    categories = sorted(set(categories))
    return PIIResult(detected=bool(categories), categories=categories, masked_text=masked)


def mask_text(text: str) -> str:
    return detect_pii(text).masked_text


def _luhn_like(digits: str) -> bool:
    total = 0
    reverse = digits[::-1]
    for idx, char in enumerate(reverse):
        value = int(char)
        if idx % 2:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0
