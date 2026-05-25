"""Deterministic tool proposal planner."""

from __future__ import annotations

from agent.decision_engine import DecisionResult
from utils.parser import ConversationState


def plan_tools(decision: DecisionResult, state: ConversationState) -> list[dict]:
    req = decision.internal_request_type
    actions: list[dict] = []
    identifier = state.extracted_user_identifier or "account_contact_on_file"

    if not decision.needs_tool and decision.status != "escalated":
        return actions

    if decision.decision == "legal_escalation":
        actions.append(_escalate("urgent", "legal", "Legal threat or regulatory escalation requires human review."))
    elif decision.decision in {"lock_account"} or req in {"account_compromise", "fraud"}:
        actions.append(
            {
                "action": "lock_account",
                "parameters": {
                    "user_identifier": identifier,
                    "lock_reason": "suspected_fraud" if req == "fraud" else "user_requested",
                },
            }
        )
        actions.append(_escalate("urgent", "security", "Potential account compromise or fraud requires security review."))
    elif decision.decision in {"refund_over_limit", "refund_too_old"}:
        actions.append(_escalate("high", "billing", "Refund request exceeds automated authorization limits."))
    elif req == "refund":
        if not state.identity_verified:
            actions.append({"action": "verify_identity", "parameters": {"method": "email_otp", "target": identifier}})
        elif state.extracted_transaction_id and state.extracted_amount is not None:
            actions.append(
                {
                    "action": "issue_refund",
                    "parameters": {
                        "transaction_id": state.extracted_transaction_id,
                        "amount": state.extracted_amount,
                        "reason": "customer_request",
                    },
                }
            )
    elif req == "subscription":
        if not state.identity_verified:
            actions.append({"action": "verify_identity", "parameters": {"method": "email_otp", "target": identifier}})
    elif decision.status == "escalated":
        actions.append(_escalate("normal", decision.department, "Ticket requires human review."))
    return actions


def _escalate(priority: str, department: str, summary: str) -> dict:
    return {
        "action": "escalate_to_human",
        "parameters": {"priority": priority, "department": department, "summary": summary},
    }
