"""V1/V2 comparison report utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ComparisonReport:
    v1: dict[str, Any] = field(default_factory=dict)
    v2: dict[str, Any] = field(default_factory=dict)

    def diff(self) -> dict[str, Any]:
        keys = sorted(set(self.v1) | set(self.v2))
        return {key: {"v1": self.v1.get(key), "v2": self.v2.get(key)} for key in keys if self.v1.get(key) != self.v2.get(key)}

    def determinism_preserved(self) -> bool:
        return self.v1.get("output_hash") == self.v2.get("output_hash")

    def to_json(self) -> str:
        return json.dumps(
            {"v1": self.v1, "v2": self.v2, "diff": self.diff(), "determinism_preserved": self.determinism_preserved()},
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
        )

