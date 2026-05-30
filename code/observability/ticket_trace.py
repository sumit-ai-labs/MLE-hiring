"""Trace model for passive V2 observability."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from observability.event_logger import safe_value


@dataclass
class TicketTrace:
    ticket_id: int
    classification: dict[str, Any] = field(default_factory=dict)
    memory_extraction: dict[str, Any] = field(default_factory=dict)
    state_machine: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    confidence: dict[str, Any] = field(default_factory=dict)
    tool_planning: list[dict[str, Any]] = field(default_factory=list)
    tool_validation: dict[str, Any] = field(default_factory=dict)
    fallbacks_triggered: list[str] = field(default_factory=list)
    gemini_usage: dict[str, Any] = field(default_factory=dict)
    risk_classification: dict[str, Any] = field(default_factory=dict)
    response_generation: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, int] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return safe_value(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True)

