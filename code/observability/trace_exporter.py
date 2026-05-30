"""Export trace data as JSON or human-readable reports."""

from __future__ import annotations

import json
from pathlib import Path

from observability.ticket_trace import TicketTrace


def traces_to_json(traces: list[TicketTrace]) -> str:
    return json.dumps([trace.to_dict() for trace in traces], sort_keys=True, indent=2, ensure_ascii=True)


def write_json(path: Path, traces: list[TicketTrace]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(traces_to_json(traces), encoding="utf-8")


def human_report(traces: list[TicketTrace]) -> str:
    lines: list[str] = []
    for trace in sorted(traces, key=lambda item: item.ticket_id):
        data = trace.to_dict()
        classification = data.get("classification", {})
        retrieval = data.get("retrieval", {})
        confidence = data.get("confidence", {})
        lines.append(
            "ticket={ticket_id} company={company} issue={issue} confidence={confidence_score} docs={docs}".format(
                ticket_id=data["ticket_id"],
                company=classification.get("company"),
                issue=classification.get("issue"),
                confidence_score=confidence.get("score"),
                docs=",".join(retrieval.get("selected_docs", [])[:3]),
            )
        )
    return "\n".join(lines)
