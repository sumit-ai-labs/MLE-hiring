"""Passive trace manager for V2 observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from observability.event_logger import safe_value
from observability.performance_metrics import PerformanceMetrics
from observability.ticket_trace import TicketTrace


@dataclass
class TraceManager:
    traces: dict[int, TicketTrace] = field(default_factory=dict)
    metrics: dict[int, PerformanceMetrics] = field(default_factory=dict)

    def start_ticket(self, ticket_id: int) -> TicketTrace:
        trace = TicketTrace(ticket_id=int(ticket_id))
        self.traces[int(ticket_id)] = trace
        self.metrics[int(ticket_id)] = PerformanceMetrics()
        self.record_event(ticket_id, "ticket_started", {})
        return trace

    def get(self, ticket_id: int) -> TicketTrace | None:
        return self.traces.get(int(ticket_id))

    def record_event(self, ticket_id: int, name: str, payload: dict[str, Any] | None = None) -> None:
        trace = self.traces.setdefault(int(ticket_id), TicketTrace(ticket_id=int(ticket_id)))
        trace.events.append({"event": name, "payload": safe_value(payload or {})})

    def record_stage(self, ticket_id: int, stage: str, units: int = 1) -> None:
        metrics = self.metrics.setdefault(int(ticket_id), PerformanceMetrics())
        metrics.record(stage, units)
        trace = self.traces.setdefault(int(ticket_id), TicketTrace(ticket_id=int(ticket_id)))
        trace.latency = metrics.as_dict()

    def finalize_ticket(self, ticket_id: int) -> TicketTrace:
        self.record_event(ticket_id, "ticket_completed", {})
        trace = self.traces[int(ticket_id)]
        trace.latency = self.metrics.get(int(ticket_id), PerformanceMetrics()).as_dict()
        return trace

