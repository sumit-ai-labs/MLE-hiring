#!/usr/bin/env python3
"""Observational visible-set evaluation harness.

This script does not modify production runtime logic or the submitted CSV.
It runs the current agent in memory and reports structural, safety, and
calibration diagnostics for development.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from agent.triage_agent import TriageAgent
from config import OUTPUT_COLUMNS, VALID_REQUEST_TYPES, VALID_RISK_LEVELS, VALID_STATUS
from tools.registry import ToolRegistry
from tools.validator import validate_actions
from utils.parser import parse_issue
from utils.pii import detect_pii
from config import TOOL_SPEC_PATH


def main() -> int:
    visible_rows = _read_csv(REPO_ROOT / "support_tickets" / "support_tickets.csv")
    sample_rows = _read_csv(REPO_ROOT / "support_tickets" / "sample_support_tickets.csv")
    current_output = _read_csv(REPO_ROOT / "support_tickets" / "output.csv")

    agent = TriageAgent(log_path=None)
    generated = [agent.process_row(row, idx) for idx, row in enumerate(visible_rows, start=1)]
    generated_again = [agent.process_row(row, idx) for idx, row in enumerate(visible_rows, start=1)]

    report = {
        "validator_pass_rate": _validator_pass_rate(generated),
        "escalation_precision": _sample_escalation_precision(agent, sample_rows),
        "tool_correctness": _tool_correctness(generated),
        "source_attribution": _source_attribution(generated),
        "retrieval_quality": _retrieval_quality(generated),
        "confidence_calibration": _confidence_calibration(generated),
        "pii_leakage": _pii_leakage(generated),
        "determinism": "PASS" if generated == generated_again else "FAIL",
        "response_completeness": _response_completeness(generated),
        "output_matches_current_csv": "PASS" if _project(generated) == _project(current_output) else "DIFF",
    }
    risk = _hidden_risk(report, generated)
    _print_report(report, risk)
    return 0 if report["validator_pass_rate"] == 1.0 and report["determinism"] == "PASS" else 1


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _validator_pass_rate(rows: list[dict[str, str]]) -> float:
    if not rows:
        return 0.0
    ok = 0
    for row in rows:
        columns_ok = list(row.keys()) == OUTPUT_COLUMNS
        enums_ok = (
            row.get("status") in VALID_STATUS
            and row.get("request_type") in VALID_REQUEST_TYPES
            and row.get("risk_level") in VALID_RISK_LEVELS
            and row.get("pii_detected") in {"true", "false"}
        )
        actions_ok = _json_array(row.get("actions_taken", ""))
        confidence_ok = _float_between(row.get("confidence_score", ""), 0.0, 1.0)
        ok += int(columns_ok and enums_ok and actions_ok and confidence_ok)
    return ok / len(rows)


def _sample_escalation_precision(agent: TriageAgent, sample_rows: list[dict[str, str]]) -> float:
    if not sample_rows:
        return 0.0
    correct = 0
    comparable = 0
    for idx, row in enumerate(sample_rows, start=1):
        expected = _get(row, "status").lower()
        if expected not in VALID_STATUS:
            continue
        predicted = agent.process_row(row, idx)["status"]
        comparable += 1
        correct += int(predicted == expected)
    return correct / comparable if comparable else 0.0


def _tool_correctness(rows: list[dict[str, str]]) -> float:
    registry = ToolRegistry(TOOL_SPEC_PATH)
    ok = 0
    total = 0
    for row in rows:
        actions = json.loads(row["actions_taken"])
        if not actions:
            ok += 1
            total += 1
            continue
        state = parse_issue(row["issue"], row["subject"], row["company"])
        result = validate_actions(actions, registry, state, _internal_from_row(row))
        total += 1
        ok += int(result.valid)
    return ok / max(1, total)


def _source_attribution(rows: list[dict[str, str]]) -> float:
    paths = [path for row in rows for path in row["source_documents"].split("|") if path]
    if not paths:
        return 1.0
    valid = sum(1 for path in paths if (REPO_ROOT / path).exists() and path.endswith(".md"))
    return valid / len(paths)


def _retrieval_quality(rows: list[dict[str, str]]) -> float:
    answerable = [row for row in rows if row["status"] == "replied" and row["request_type"] != "invalid"]
    if not answerable:
        return 1.0
    with_sources = sum(bool(row["source_documents"]) for row in answerable)
    return with_sources / len(answerable)


def _confidence_calibration(rows: list[dict[str, str]]) -> float:
    overconfident_weak = 0
    for row in rows:
        conf = float(row["confidence_score"])
        weak = row["request_type"] == "invalid" or (row["status"] == "escalated" and not row["actions_taken"])
        if weak and conf > 0.75:
            overconfident_weak += 1
    return 1.0 - overconfident_weak / max(1, len(rows))


def _pii_leakage(rows: list[dict[str, str]]) -> float:
    pii_rows = [row for row in rows if row["pii_detected"] == "true"]
    if not pii_rows:
        return 1.0
    safe = 0
    for row in pii_rows:
        response_pii = detect_pii(row["response"])
        safe += int(not response_pii.detected)
    return safe / len(pii_rows)


def _response_completeness(rows: list[dict[str, str]]) -> float:
    ok = 0
    for row in rows:
        response = row["response"].strip()
        enough = len(response) >= 35 or row["request_type"] == "invalid"
        safe = "system prompt" not in response.lower() and "developer message" not in response.lower()
        ok += int(bool(response) and enough and safe)
    return ok / max(1, len(rows))


def _hidden_risk(report: dict[str, object], rows: list[dict[str, str]]) -> tuple[str, str]:
    if report["determinism"] != "PASS":
        return "determinism", "fix non-deterministic output before submission"
    if report["pii_leakage"] < 1.0:
        return "pii leakage", "tighten response masking"
    if report["source_attribution"] < 1.0:
        return "source attribution", "remove invalid citations"
    ambiguous = sum(row["confidence_score"] > "0.85" and row["request_type"] == "invalid" for row in rows)
    if ambiguous:
        return "confidence calibration", "lower confidence on unsupported ambiguous tickets"
    if report["retrieval_quality"] < 0.85:
        return "retrieval specificity", "inspect low-source answerable replies"
    return "cross-company ambiguity", "continue red-team testing with misleading company fields"


def _print_report(report: dict[str, object], risk: tuple[str, str]) -> None:
    print("================================")
    print("VISIBLE EVAL REPORT")
    print("================================")
    for key, value in report.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value}")
    print()
    print(f"highest-risk failure: {risk[0]}")
    print(f"recommendation: {risk[1]}")


def _project(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{column: row.get(column, "") for column in OUTPUT_COLUMNS} for row in rows]


def _get(row: dict[str, str], name: str) -> str:
    for key, value in row.items():
        if key.strip().lower().replace(" ", "_") == name:
            return value or ""
    return ""


def _json_array(value: str) -> bool:
    try:
        return isinstance(json.loads(value), list)
    except Exception:
        return False


def _float_between(value: str, low: float, high: float) -> bool:
    try:
        parsed = float(value)
    except ValueError:
        return False
    return low <= parsed <= high


def _internal_from_row(row: dict[str, str]) -> str:
    text = " ".join([row.get("subject", ""), row.get("response", ""), row.get("justification", "")]).lower()
    if "refund" in text:
        return "refund"
    if "compromise" in text or "fraud" in text:
        return "account_compromise"
    if "legal" in text:
        return "legal"
    return row.get("request_type", "product_issue")


if __name__ == "__main__":
    raise SystemExit(main())
