"""Optional wall-clock profiler for V2 performance work.

Profiler output is never written to CSV and is disabled by default.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageProfiler:
    timings: dict[str, float] = field(default_factory=dict)
    _starts: dict[str, float] = field(default_factory=dict)

    def start(self, stage: str) -> None:
        self._starts[stage] = time.perf_counter()

    def stop(self, stage: str) -> None:
        started = self._starts.pop(stage, None)
        if started is None:
            return
        self.timings[stage] = self.timings.get(stage, 0.0) + max(0.0, time.perf_counter() - started)

    def record(self, stage: str, seconds: float) -> None:
        self.timings[stage] = self.timings.get(stage, 0.0) + max(0.0, float(seconds))

    def report(self) -> dict[str, Any]:
        total = sum(self.timings.values())
        return {
            "total_seconds": round(total, 6),
            "stages": {stage: round(self.timings[stage], 6) for stage in sorted(self.timings)},
        }
