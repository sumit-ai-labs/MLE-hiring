"""Conversation parsing and deterministic multi-turn state extraction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Turn:
    role: str
    content: str


@dataclass
class ConversationState:
    turns: list[Turn]
    subject: str
    input_company: str
    full_text: str
    latest_user_text: str
    identity_verified: bool = False
    prior_failed_attempt: bool = False
    prior_escalation: bool = False
    prior_promise: bool = False
    follow_up: bool = False
    repeated_failure: bool = False
    extracted_amount: float | None = None
    extracted_transaction_id: str | None = None
    extracted_days_old: int | None = None
    extracted_user_identifier: str | None = None
    notes: list[str] = field(default_factory=list)


VERIFY_PATTERNS = [
    r"\b(identity|account|email|phone)\s+(has\s+been\s+)?(verified|confirmed)\b",
    r"\bverification\s+(complete|completed|passed|successful)\b",
    r"\bverified\s+(my|the)\s+(identity|account|email|phone)\b",
]

FAILED_PATTERNS = [
    r"\bstill\s+(not|isn't|cannot|can't|unable)\b",
    r"\bfailed\b",
    r"\bdid\s+not\s+work\b",
    r"\bnot\s+resolved\b",
]

PROMISE_PATTERNS = [
    r"\bwe\s+(will|would|can)\s+(refund|escalate|follow up|restore|fix)\b",
    r"\bpromised\b",
]


def parse_issue(issue_raw: str, subject: str = "", company: str = "") -> ConversationState:
    turns = _parse_turns(issue_raw)
    user_turns = [turn.content for turn in turns if turn.role.lower() == "user"]
    full_text = "\n".join(f"{turn.role}: {turn.content}" for turn in turns)
    latest_user = user_turns[-1] if user_turns else (turns[-1].content if turns else str(issue_raw))
    combined = " ".join([subject or "", full_text])
    lower = combined.lower()

    state = ConversationState(
        turns=turns,
        subject=subject or "",
        input_company=(company or "").strip(),
        full_text=full_text,
        latest_user_text=latest_user,
    )
    state.identity_verified = any(re.search(p, lower, re.I) for p in VERIFY_PATTERNS)
    state.prior_failed_attempt = any(re.search(p, lower, re.I) for p in FAILED_PATTERNS)
    state.prior_escalation = bool(re.search(r"\b(escalated|case\s+number|ticket\s+#?|human\s+agent)\b", lower, re.I))
    state.prior_promise = any(re.search(p, lower, re.I) for p in PROMISE_PATTERNS)
    state.follow_up = len(user_turns) > 1 or bool(re.search(r"\b(following up|follow up|again|still)\b", lower, re.I))
    state.repeated_failure = state.follow_up and state.prior_failed_attempt
    state.extracted_amount = extract_amount(combined)
    state.extracted_transaction_id = extract_transaction_id(combined)
    state.extracted_days_old = extract_days_old(combined)
    state.extracted_user_identifier = extract_user_identifier(combined)
    return state


def _parse_turns(issue_raw: str) -> list[Turn]:
    if not issue_raw:
        return []
    text = str(issue_raw)
    try:
        parsed: Any = json.loads(text)
        if isinstance(parsed, list):
            turns = []
            for item in parsed:
                if isinstance(item, dict):
                    role = str(item.get("role", "user")).strip() or "user"
                    content = str(item.get("content", "")).strip()
                    if content:
                        turns.append(Turn(role=role, content=content))
            if turns:
                return turns
    except Exception:
        pass

    parts = re.split(r"(?im)^\s*(user|assistant|agent|support)\s*:\s*", text)
    if len(parts) > 2:
        turns = []
        for i in range(1, len(parts), 2):
            role = parts[i].strip().lower()
            content = parts[i + 1].strip()
            if content:
                turns.append(Turn(role=role, content=content))
        if turns:
            return turns
    return [Turn(role="user", content=text.strip())]


def extract_amount(text: str) -> float | None:
    matches = re.findall(r"(?:\$|usd\s*)\s*([0-9]+(?:\.[0-9]{1,2})?)|([0-9]+(?:\.[0-9]{1,2})?)\s*(?:usd|dollars?)", text, re.I)
    values = [a or b for a, b in matches if (a or b)]
    if not values:
        return None
    try:
        return float(values[0])
    except ValueError:
        return None


def extract_transaction_id(text: str) -> str | None:
    match = re.search(r"\b(txn_[A-Za-z0-9_-]+|transaction\s*(?:id|#)\s*[:#]?\s*([A-Za-z0-9_-]+))\b", text, re.I)
    if not match:
        return None
    return match.group(1) if match.group(1).lower().startswith("txn_") else match.group(2)


def extract_days_old(text: str) -> int | None:
    match = re.search(r"\b([0-9]{1,4})\s+(days?|months?|years?)\s+(old|ago)\b", text, re.I)
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("month"):
        return value * 30
    if unit.startswith("year"):
        return value * 365
    return value


def extract_user_identifier(text: str) -> str | None:
    email = re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.I)
    if email:
        return email.group(0)
    account = re.search(r"\b(?:account|user|customer)\s*(?:id|#)\s*[:#]?\s*([A-Za-z0-9_-]{5,})\b", text, re.I)
    if account:
        return account.group(1)
    return None
