"""Deterministic structured memory extraction."""

from __future__ import annotations

import re

from agent.classifier import ClassificationResult
from agent.decision_engine import DecisionResult
from agent.safety import SafetyResult
from memory.ticket_memory import TicketMemory
from utils.parser import ConversationState
from utils.pii import PIIResult


def extract_memory(
    state: ConversationState,
    classification: ClassificationResult,
    safety: SafetyResult,
    pii: PIIResult,
    language: str,
    decision: DecisionResult | None = None,
    actions: list[dict] | None = None,
) -> TicketMemory:
    request_type = decision.internal_request_type if decision else classification.internal_request_type
    risk_level = decision.risk_level if decision else safety.risk_level
    prior_actions = [str(action.get("action", "")) for action in actions or [] if action.get("action")]
    security_signal = _security_signal(state.full_text, classification.internal_request_type, safety)
    refund_signal = _refund_signal(state.full_text, state.extracted_amount, state.extracted_days_old)
    verification_status = "verified" if state.identity_verified else "unverified"
    if "verify_identity" in prior_actions and not state.identity_verified:
        verification_status = "pending"
    return TicketMemory(
        company=_none_if_empty(classification.company),
        product=_none_if_empty(classification.display_company),
        product_area=_none_if_empty(classification.product_area),
        transaction_id=_none_if_empty(state.extracted_transaction_id),
        user_id=_none_if_empty(state.extracted_user_identifier),
        verification_status=verification_status,
        security_signal=security_signal,
        refund_signal=refund_signal,
        risk_level=_none_if_empty(risk_level),
        language=_none_if_empty(language),
        pii_detected=bool(pii.detected),
        prior_actions=prior_actions,
        request_type=_none_if_empty(request_type),
        current_issue=_current_issue(state),
        escalation_needed=bool(decision and decision.status == "escalated"),
    )


def _security_signal(text: str, request_type: str, safety: SafetyResult) -> str | None:
    lower = (text or "").lower()
    if request_type in {"account_compromise", "fraud"}:
        return request_type
    if "legal" == request_type:
        return "legal"
    if safety.attack_detected:
        return "prompt_injection"
    if re.search(r"\b(hacked|compromised|unauthorized|fraud|stolen)\b", lower):
        return "security_keyword"
    return None


def _refund_signal(text: str, amount: float | None, days_old: int | None) -> str | None:
    lower = (text or "").lower()
    if not re.search(r"\b(refund|reimburse|money back|charged back)\b", lower):
        return None
    if amount is not None and amount > 500:
        return "over_limit"
    if days_old is not None and days_old > 90:
        return "too_old"
    return "requested"


def _current_issue(state: ConversationState) -> str | None:
    text = (state.latest_user_text or state.full_text or "").strip()
    if not text:
        return None
    sanitized = re.sub(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[EMAIL]", text, flags=re.I)
    sanitized = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", sanitized)
    sanitized = re.sub(r"\b(?:\d[ -]*?){13,16}\b", "[CARD]", sanitized)
    return sanitized[:180]


def _none_if_empty(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
