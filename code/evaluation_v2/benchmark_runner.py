"""Standalone V2 benchmark runner."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluation_v2.comparison_report import ComparisonReport
from evaluation_v2.confidence_metrics import confidence_distribution
from evaluation_v2.retrieval_metrics import cross_company_confusion_rate, retrieval_coverage
from evaluation_v2.safety_metrics import security_incident_handling
from evaluation_v2.tool_metrics import tool_correctness_rate


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    actions = [_actions(row.get("actions_taken", "[]")) for row in rows]
    scores = [_float(row.get("confidence_score", "0")) for row in rows]
    return {
        "row_count": len(rows),
        "output_hash": hash_rows(rows),
        "retrieval_coverage": retrieval_coverage(rows),
        "cross_company_confusion_rate": cross_company_confusion_rate(rows),
        "tool_correctness": tool_correctness_rate(list(zip(actions, [row.get("request_type", "") for row in rows]))),
        "confidence_distribution": confidence_distribution(scores),
        "security_incident_handling": security_incident_handling([row.get("risk_level", "") for row in rows], actions),
    }


def compare_rows(v1_rows: list[dict[str, str]], v2_rows: list[dict[str, str]]) -> ComparisonReport:
    return ComparisonReport(v1=summarize_rows(v1_rows), v2=summarize_rows(v2_rows))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def hash_rows(rows: list[dict[str, str]]) -> str:
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _actions(value: str) -> list[dict]:
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0
