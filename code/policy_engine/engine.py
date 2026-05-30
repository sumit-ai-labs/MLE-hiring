"""Declarative V2 policy evaluator.

The engine intentionally supports a small condition vocabulary first. This is
enough to mirror the highest-risk V1 decision rules while keeping the adapter
auditable and easy to roll back.
"""

from __future__ import annotations

from typing import Any

from agent.classifier import ClassificationResult
from agent.decision_engine import DecisionResult
from agent.safety import SafetyResult
from policy_engine.loader import load_policies
from utils.parser import ConversationState


class PolicyEngine:
    def __init__(self, policies: list[dict[str, Any]] | None = None):
        self.policies = policies if policies is not None else load_policies()

    def evaluate(
        self,
        state: ConversationState,
        safety: SafetyResult,
        classification: ClassificationResult,
        retrieval_count: int,
    ) -> DecisionResult | None:
        for policy in self.policies:
            if _matches(policy.get("when", {}), state, safety, classification, retrieval_count):
                return _to_decision(policy.get("result", {}), classification)
        return None


def _matches(
    condition: dict[str, Any],
    state: ConversationState,
    safety: SafetyResult,
    classification: ClassificationResult,
    retrieval_count: int,
) -> bool:
    request_type = classification.internal_request_type
    if "request_type" in condition and request_type != condition["request_type"]:
        return False
    if "request_type_one_of" in condition and request_type not in set(condition["request_type_one_of"]):
        return False
    if "company" in condition and classification.company != condition["company"]:
        return False
    if "attack_detected" in condition and bool(safety.attack_detected) != bool(condition["attack_detected"]):
        return False
    if "retrieval_count_eq" in condition and retrieval_count != int(condition["retrieval_count_eq"]):
        return False
    if "amount_gt" in condition and not (state.extracted_amount is not None and state.extracted_amount > float(condition["amount_gt"])):
        return False
    if "age_days_gt" in condition and not (state.extracted_days_old is not None and state.extracted_days_old > int(condition["age_days_gt"])):
        return False
    for field in condition.get("missing_any", []):
        if not _present(state, field):
            return True
    if condition.get("missing_any"):
        return False
    for field in condition.get("present_all", []):
        if not _present(state, field):
            return False
    return True


def _present(state: ConversationState, field: str) -> bool:
    value = getattr(state, field, None)
    if isinstance(value, bool):
        return value
    return value is not None and value != ""


def _to_decision(result: dict[str, Any], classification: ClassificationResult) -> DecisionResult:
    status = str(result["status"])
    return DecisionResult(
        status=status,
        internal_status=status,
        risk_level=str(result["risk_level"]),
        internal_request_type=str(result.get("internal_request_type", classification.internal_request_type)),
        final_request_type=classification.final_request_type,
        product_area=classification.product_area,
        decision=str(result["decision"]),
        department=str(result["department"]),
        certainty=float(result["certainty"]),
        needs_tool=bool(result["needs_tool"]),
        response_mode=str(result["response_mode"]),
    )

