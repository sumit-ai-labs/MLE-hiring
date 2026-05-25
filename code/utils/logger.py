"""Deterministic readable per-ticket logging."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class TriageLogger:
    def __init__(self, path: Path | None):
        self.path = path
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_ticket(self, ticket_id: int, sections: dict[str, Any]) -> None:
        if not self.path:
            return
        lines = [f"=== Ticket {ticket_id} ==="]
        for key in [
            "INPUT",
            "SAFETY",
            "PII",
            "LANGUAGE",
            "CLASSIFICATION",
            "RETRIEVAL",
            "DECISION",
            "TOOLS",
            "RESPONSE",
        ]:
            lines.append(f"[{key}]")
            value = sections.get(key, "")
            lines.append(str(value))
        lines.append("")
        self.path.open("a", encoding="utf-8", newline="\n").write("\n".join(lines))
