"""Feature-gated V2 policy adapter around the frozen V1 decision result."""

from __future__ import annotations

from agent.classifier import ClassificationResult
from agent.decision_engine import DecisionResult
from agent.safety import SafetyResult
from policy_engine.engine import PolicyEngine
from utils.parser import ConversationState


COMPARE_FIELDS = (
    "status",
    "internal_status",
    "risk_level",
    "internal_request_type",
    "final_request_type",
    "product_area",
    "decision",
    "department",
    "certainty",
    "needs_tool",
    "response_mode",
)


def decide_with_policy_engine(
    v1_result: DecisionResult,
    state: ConversationState,
    safety: SafetyResult,
    classification: ClassificationResult,
    retrieval_count: int,
    engine: PolicyEngine | None = None,
) -> DecisionResult:
    """Return a V2 policy result only when it is byte-for-byte compatible.

    This keeps V1 behavior safe while allowing V2 policy coverage to grow under
    differential tests. Unsupported or mismatched cases fall back to V1.
    """

    policy_result = (engine or PolicyEngine()).evaluate(state, safety, classification, retrieval_count)
    if policy_result is None:
        return v1_result
    return policy_result if decisions_match(v1_result, policy_result) else v1_result


def decisions_match(left: DecisionResult, right: DecisionResult) -> bool:
    return all(getattr(left, field) == getattr(right, field) for field in COMPARE_FIELDS)

