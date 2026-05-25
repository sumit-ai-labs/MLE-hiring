"""Validate planned tool calls against schemas and preconditions."""

from __future__ import annotations

from dataclasses import dataclass, field

from tools.registry import ToolRegistry
from utils.parser import ConversationState


@dataclass
class ValidationResult:
    actions: list[dict]
    valid: bool
    errors: list[str] = field(default_factory=list)


def validate_actions(actions: list[dict], registry: ToolRegistry, state: ConversationState, internal_request_type: str) -> ValidationResult:
    valid_actions: list[dict] = []
    errors: list[str] = []
    verified_in_plan = state.identity_verified

    for action in actions:
        name = action.get("action")
        params = action.get("parameters", {})
        spec = registry.get(name)
        if not spec:
            errors.append(f"unknown_tool:{name}")
            continue
        required = spec.get("parameters", {}).get("required", [])
        missing = [field for field in required if field not in params or params[field] in {None, ""}]
        if missing:
            errors.append(f"missing_required:{name}:{','.join(missing)}")
            continue
        if name == "verify_identity":
            verified_in_plan = True
            valid_actions.append(action)
            continue
        if name == "issue_refund":
            if not verified_in_plan:
                errors.append("refund_without_identity_verification")
                continue
            if float(params.get("amount", 0)) > 500:
                errors.append("refund_amount_over_limit")
                continue
            if state.extracted_days_old is not None and state.extracted_days_old > 90:
                errors.append("refund_age_over_limit")
                continue
        if name == "reset_password" and internal_request_type in {"account_compromise", "fraud"}:
            errors.append("password_reset_blocked_for_compromise")
            continue
        if name == "modify_subscription" and not verified_in_plan:
            errors.append("subscription_change_without_identity_verification")
            continue
        valid_actions.append(action)
    return ValidationResult(actions=valid_actions, valid=not errors, errors=errors)
