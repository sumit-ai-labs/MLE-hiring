"""Schema-driven ticket memory for V2 shadow mode."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


MEMORY_FIELDS = (
    "company",
    "product",
    "product_area",
    "transaction_id",
    "user_id",
    "verification_status",
    "security_signal",
    "refund_signal",
    "risk_level",
    "language",
    "pii_detected",
    "prior_actions",
    "request_type",
    "current_issue",
    "escalation_needed",
)


@dataclass(frozen=True)
class TicketMemory:
    company: str | None = None
    product: str | None = None
    product_area: str | None = None
    transaction_id: str | None = None
    user_id: str | None = None
    verification_status: str | None = None
    security_signal: str | None = None
    refund_signal: str | None = None
    risk_level: str | None = None
    language: str | None = None
    pii_detected: bool = False
    prior_actions: list[str] = field(default_factory=list)
    request_type: str | None = None
    current_issue: str | None = None
    escalation_needed: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {field_name: data.get(field_name) for field_name in MEMORY_FIELDS}

    def validate(self) -> bool:
        data = self.to_dict()
        if tuple(data.keys()) != MEMORY_FIELDS:
            return False
        if not isinstance(data["pii_detected"], bool):
            return False
        if not isinstance(data["prior_actions"], list):
            return False
        if not isinstance(data["escalation_needed"], bool):
            return False
        allowed_verification = {None, "unverified", "pending", "verified"}
        return data["verification_status"] in allowed_verification

