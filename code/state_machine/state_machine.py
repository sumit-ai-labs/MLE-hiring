"""Explicit V2 ticket lifecycle state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class State(str, Enum):
    NEW = "NEW"
    SAFETY_CHECKED = "SAFETY_CHECKED"
    PII_CHECKED = "PII_CHECKED"
    CLASSIFIED = "CLASSIFIED"
    RETRIEVED = "RETRIEVED"
    RERANKED = "RERANKED"
    TOOL_PLANNED = "TOOL_PLANNED"
    VALIDATED = "VALIDATED"
    RESPONDED = "RESPONDED"
    COMPLETED = "COMPLETED"
    VERIFICATION_PENDING = "VERIFICATION_PENDING"
    VERIFIED = "VERIFIED"
    ESCALATED = "ESCALATED"
    LOCKED = "LOCKED"
    REFUND_ELIGIBLE = "REFUND_ELIGIBLE"
    REFUND_DENIED = "REFUND_DENIED"


LINEAR_TRANSITIONS: dict[State, set[State]] = {
    State.NEW: {State.SAFETY_CHECKED},
    State.SAFETY_CHECKED: {State.PII_CHECKED, State.ESCALATED, State.LOCKED},
    State.PII_CHECKED: {State.CLASSIFIED, State.ESCALATED, State.LOCKED},
    State.CLASSIFIED: {State.RETRIEVED, State.ESCALATED, State.LOCKED},
    State.RETRIEVED: {State.RERANKED, State.ESCALATED, State.LOCKED},
    State.RERANKED: {State.TOOL_PLANNED, State.ESCALATED, State.LOCKED},
    State.TOOL_PLANNED: {State.VALIDATED, State.VERIFICATION_PENDING, State.VERIFIED, State.ESCALATED, State.LOCKED, State.REFUND_DENIED},
    State.VERIFICATION_PENDING: {State.VALIDATED, State.ESCALATED},
    State.VERIFIED: {State.VALIDATED, State.REFUND_ELIGIBLE, State.ESCALATED},
    State.REFUND_ELIGIBLE: {State.VALIDATED, State.RESPONDED},
    State.REFUND_DENIED: {State.VALIDATED, State.RESPONDED, State.ESCALATED},
    State.VALIDATED: {State.RESPONDED, State.ESCALATED, State.LOCKED},
    State.RESPONDED: {State.COMPLETED},
    State.ESCALATED: {State.RESPONDED, State.COMPLETED},
    State.LOCKED: {State.ESCALATED, State.RESPONDED, State.COMPLETED},
    State.COMPLETED: set(),
}


@dataclass(frozen=True)
class TransitionRecord:
    state_before: str
    trigger: str
    state_after: str
    reason: str
    allowed: bool
    timestamp: str


@dataclass
class TicketState:
    ticket_id: int
    current_state: State = State.NEW
    history: list[TransitionRecord] = field(default_factory=list)
    verification_status: str = "unverified"
    security_signals: list[str] = field(default_factory=list)
    refund_eligibility: str | None = None
    prior_actions: list[str] = field(default_factory=list)
    escalation_status: str = "not_escalated"
    trace_enabled: bool = False
    _logical_clock: int = 0

    def transition(self, target: State | str, trigger: str, reason: str = "") -> bool:
        target_state = State(target)
        allowed = self._can_transition(target_state, trigger)
        before = self.current_state
        if allowed:
            self.current_state = target_state
            self._apply_side_effects(target_state, trigger)
        self._record(before, trigger, target_state, reason, allowed)
        return allowed

    def apply_action(self, action_name: str) -> bool:
        self.prior_actions.append(action_name)
        if action_name == "verify_identity":
            return self.transition(State.VERIFICATION_PENDING, "tool:verify_identity", "identity verification requested")
        if action_name == "issue_refund":
            if self.verification_status != "verified":
                return self.transition(State.REFUND_DENIED, "tool:issue_refund", "refund attempted before verification")
            return self.transition(State.REFUND_ELIGIBLE, "tool:issue_refund", "verified refund action")
        if action_name == "lock_account":
            return self.transition(State.LOCKED, "tool:lock_account", "security lock action")
        if action_name == "escalate_to_human":
            return self.transition(State.ESCALATED, "tool:escalate_to_human", "human escalation action")
        return True

    def mark_verified(self, reason: str = "conversation indicates verification") -> bool:
        if self.current_state == State.LOCKED:
            return self.transition(State.VERIFIED, "memory:verified", "cannot verify after account lock")
        return self.transition(State.VERIFIED, "memory:verified", reason)

    def mark_security_signal(self, signal: str) -> bool:
        if signal not in self.security_signals:
            self.security_signals.append(signal)
        return self.transition(State.LOCKED, "memory:security_signal", signal)

    def _can_transition(self, target: State, trigger: str) -> bool:
        if self.current_state == State.COMPLETED:
            return False
        if self.current_state == State.LOCKED and target in {State.VERIFIED, State.REFUND_ELIGIBLE}:
            return False
        if target == State.REFUND_ELIGIBLE and self.verification_status != "verified":
            return False
        if target == State.VERIFIED and self.current_state == State.LOCKED:
            return False
        return target in LINEAR_TRANSITIONS.get(self.current_state, set())

    def _apply_side_effects(self, target: State, trigger: str) -> None:
        if target == State.VERIFICATION_PENDING:
            self.verification_status = "pending"
        elif target == State.VERIFIED:
            self.verification_status = "verified"
        elif target == State.ESCALATED:
            self.escalation_status = "escalated"
        elif target == State.LOCKED:
            self.escalation_status = "security_review"
        elif target == State.REFUND_ELIGIBLE:
            self.refund_eligibility = "eligible"
        elif target == State.REFUND_DENIED:
            self.refund_eligibility = "denied"

    def _record(self, before: State, trigger: str, after: State, reason: str, allowed: bool) -> None:
        self._logical_clock += 1
        record = TransitionRecord(
            state_before=before.value,
            trigger=trigger,
            state_after=after.value if allowed else before.value,
            reason=_sanitize_reason(reason),
            allowed=allowed,
            timestamp=f"t{self._logical_clock:04d}",
        )
        self.history.append(record)

    def trace(self) -> list[dict[str, Any]]:
        return [record.__dict__.copy() for record in self.history]


def _sanitize_reason(reason: str) -> str:
    text = str(reason or "")[:160]
    for marker in ("@", "sk-", "4242", "123-45"):
        if marker in text:
            return "redacted"
    return text


def build_shadow_lifecycle(
    ticket_id: int,
    *,
    identity_verified: bool,
    security_signal: str | None,
    risk_level: str,
    actions: list[dict],
    trace_enabled: bool = False,
) -> TicketState:
    ticket = TicketState(ticket_id=ticket_id, trace_enabled=trace_enabled)
    for state in (
        State.SAFETY_CHECKED,
        State.PII_CHECKED,
        State.CLASSIFIED,
        State.RETRIEVED,
        State.RERANKED,
        State.TOOL_PLANNED,
    ):
        ticket.transition(state, f"stage:{state.value.lower()}", "pipeline stage completed")
    if security_signal:
        ticket.mark_security_signal(security_signal)
    elif identity_verified:
        ticket.mark_verified()
    if risk_level in {"high", "critical"} and ticket.current_state not in {State.LOCKED, State.ESCALATED}:
        ticket.transition(State.ESCALATED, "risk:escalation", "high-risk ticket")
    for action in actions:
        name = str(action.get("action", ""))
        if name:
            ticket.apply_action(name)
    if ticket.current_state not in {State.VALIDATED, State.RESPONDED, State.COMPLETED}:
        ticket.transition(State.VALIDATED, "stage:validated", "tool validation completed")
    ticket.transition(State.RESPONDED, "stage:responded", "response generated")
    ticket.transition(State.COMPLETED, "stage:completed", "csv row completed")
    return ticket
