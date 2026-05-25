"""Rule-first decision engine."""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.classifier import ClassificationResult
from agent.safety import SafetyResult
from utils.parser import ConversationState


@dataclass
class DecisionResult:
    status: str
    internal_status: str
    risk_level: str
    internal_request_type: str
    final_request_type: str
    product_area: str
    decision: str
    department: str
    certainty: float
    needs_tool: bool
    response_mode: str


def decide(
    state: ConversationState,
    safety: SafetyResult,
    classification: ClassificationResult,
    retrieval_count: int,
) -> DecisionResult:
    text = " ".join([state.subject, state.full_text]).lower()
    req = classification.internal_request_type
    risk = "high" if safety.attack_detected else "low"
    status = "replied"
    department = "general"
    decision = "reply_from_corpus"
    needs_tool = False
    response_mode = "answer"
    certainty = 0.78

    if safety.attack_detected and _adversarial_probe_without_support(text):
        return DecisionResult(
            status="replied",
            internal_status="replied",
            risk_level="high",
            internal_request_type="invalid",
            final_request_type="invalid",
            product_area="",
            decision="prompt_injection_probe",
            department="security",
            certainty=0.72,
            needs_tool=False,
            response_mode="scope",
        )
    if req == "legal":
        return _result("escalated", "critical", req, classification, "legal_escalation", "legal", 0.96, True, "escalate")
    if req in {"account_compromise", "fraud"}:
        return _result("escalated", "critical", req, classification, "lock_account", "security", 0.93, True, "escalate")
    if classification.company == "claude" and re.search(r"\b(restore|add|grant)\b.{0,80}\b(access|seat|admin|owner)\b", text) and re.search(r"\b(not|n't|removed|without|even though)\b.{0,80}\b(admin|owner|workspace owner)\b", text):
        return _result("replied", "medium", req, classification, "admin_permission_required", "general", 0.86, False, "unsupported_action")
    if classification.company == "devplatform" and re.search(r"\b(increase|change|adjust|override)\b.{0,60}\b(score|grade|result)\b|\bmove me\b.{0,50}\b(next round|forward)\b", text):
        return _result("replied", "medium", req, classification, "unsupported_score_change", "general", 0.88, False, "unsupported_action")
    if classification.company == "visa" and req in {"refund", "billing"} and re.search(r"\b(merchant|seller|wrong product|refund|chargeback|dispute)\b", text):
        return _result("replied", "medium", req, classification, "visa_dispute_guidance", "billing", 0.84, False, "visa_dispute")
    if re.search(r"\b(delete all data|erase account|remove account|close account)\b", text):
        return _result("escalated", "high", req, classification, "destructive_account_request", "security", 0.86, True, "escalate")

    if req == "billing":
        risk = "medium"
        department = "billing"
        decision = "billing_guidance"
        certainty = 0.72
    elif req == "refund":
        risk = "medium"
        department = "billing"
        needs_tool = True
        decision = "refund_flow"
        certainty = 0.84
        if state.extracted_amount is not None and state.extracted_amount > 500:
            return _result("escalated", "high", req, classification, "refund_over_limit", "billing", 0.92, True, "escalate")
        if state.extracted_days_old is not None and state.extracted_days_old > 90:
            return _result("escalated", "high", req, classification, "refund_too_old", "billing", 0.92, True, "escalate")
        if not state.identity_verified or not state.extracted_transaction_id or state.extracted_amount is None:
            status = "replied"
            response_mode = "clarify_or_verify"
            certainty = 0.72
        else:
            response_mode = "tool_ack"

    elif req == "subscription":
        risk = "medium"
        department = "billing"
        needs_tool = True
        decision = "subscription_flow"
        response_mode = "clarify_or_verify" if not state.identity_verified else "tool_ack"
        certainty = 0.74
    elif req in {"bug", "technical_issue"}:
        risk = "medium" if state.repeated_failure else risk
        department = "technical"
        decision = "technical_support"
        certainty = 0.75
    elif req == "feature_request":
        decision = "feature_request"
        certainty = 0.76
    elif req == "invalid" or classification.company == "none":
        status = "replied"
        risk = "low" if not safety.attack_detected else "high"
        decision = "out_of_scope"
        response_mode = "scope"
        certainty = 0.9

    if safety.attack_detected and risk not in {"critical"}:
        risk = "high"
    if retrieval_count == 0 and classification.company != "none" and response_mode == "answer":
        status = "escalated"
        decision = "missing_grounding"
        response_mode = "escalate"
        certainty = min(certainty, 0.55)

    return DecisionResult(
        status=status,
        internal_status=status,
        risk_level=risk,
        internal_request_type=req,
        final_request_type=classification.final_request_type,
        product_area=classification.product_area,
        decision=decision,
        department=department,
        certainty=certainty,
        needs_tool=needs_tool,
        response_mode=response_mode,
    )


def _adversarial_probe_without_support(text: str) -> bool:
    support_terms = re.compile(
        r"\b(refund|billing|invoice|payment|charge|subscription|cancel|upgrade|login|password|hacked|fraud|card|travel|candidate|assessment|test|interview|api|endpoint|deployment|workspace|access|error|bug)\b",
        re.I,
    )
    return not support_terms.search(text or "")


def _result(
    status: str,
    risk: str,
    req: str,
    classification: ClassificationResult,
    decision: str,
    department: str,
    certainty: float,
    needs_tool: bool,
    response_mode: str,
) -> DecisionResult:
    return DecisionResult(
        status=status,
        internal_status=status,
        risk_level=risk,
        internal_request_type=req,
        final_request_type=classification.final_request_type,
        product_area=classification.product_area,
        decision=decision,
        department=department,
        certainty=certainty,
        needs_tool=needs_tool,
        response_mode=response_mode,
    )
