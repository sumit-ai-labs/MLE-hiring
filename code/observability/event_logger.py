"""Safe trace event helpers."""

from __future__ import annotations

import re
from typing import Any


SECRET_PATTERN = re.compile(r"\b(?:AIza[0-9A-Za-z_-]+|sk-[A-Za-z0-9_-]{10,}|sk-ant-[A-Za-z0-9_-]+)\b")
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
CARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list):
        return [safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [safe_value(item) for item in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return sanitize_text(str(value))


def sanitize_text(text: str) -> str:
    clean = text or ""
    clean = SECRET_PATTERN.sub("[REDACTED_SECRET]", clean)
    clean = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", clean)
    clean = CARD_PATTERN.sub("[REDACTED_PAYMENT_CARD]", clean)
    clean = SSN_PATTERN.sub("[REDACTED_SSN]", clean)
    return clean[:2000]

