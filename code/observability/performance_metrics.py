"""Deterministic performance metrics for trace comparison."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PerformanceMetrics:
    """Logical latency counters.

    Wall-clock timings are useful in production, but deterministic logical
    units are safer for this evaluation branch because traces stay reproducible.
    """

    stage_units: dict[str, int] = field(default_factory=dict)

    def record(self, stage: str, units: int = 1) -> None:
        self.stage_units[stage] = self.stage_units.get(stage, 0) + max(0, int(units))

    def total(self) -> int:
        return sum(self.stage_units.values())

    def as_dict(self) -> dict[str, int]:
        data = {stage: self.stage_units[stage] for stage in sorted(self.stage_units)}
        data["overall_ticket_runtime"] = self.total()
        return data

    def compare(self, other: "PerformanceMetrics") -> dict[str, int]:
        keys = sorted(set(self.stage_units) | set(other.stage_units))
        return {key: self.stage_units.get(key, 0) - other.stage_units.get(key, 0) for key in keys}

