"""Retrieval metrics for V2 evaluation."""

from __future__ import annotations


def precision_at_k(predicted_paths: list[str], expected_paths: set[str], k: int = 5) -> float:
    if k <= 0:
        return 0.0
    predicted = predicted_paths[:k]
    if not predicted:
        return 0.0
    return sum(1 for path in predicted if path in expected_paths) / len(predicted)


def retrieval_coverage(rows: list[dict[str, str]]) -> float:
    answerable = [row for row in rows if row.get("status") == "replied" and row.get("request_type") != "invalid"]
    if not answerable:
        return 1.0
    return sum(bool(row.get("source_documents")) for row in answerable) / len(answerable)


def cross_company_confusion_rate(rows: list[dict[str, str]]) -> float:
    total = 0
    confused = 0
    for row in rows:
        company = (row.get("company") or "").lower()
        paths = [path for path in (row.get("source_documents") or "").split("|") if path]
        for path in paths:
            total += 1
            confused += int(company and company not in {"none", "unknown"} and company not in path.lower())
    return confused / total if total else 0.0


def rerank_gain(v1_scores: list[float], v2_scores: list[float]) -> float:
    if not v1_scores or not v2_scores:
        return 0.0
    return round((sum(v2_scores) / len(v2_scores)) - (sum(v1_scores) / len(v1_scores)), 4)


def poisoning_resistance(selected_paths: list[str]) -> float:
    if not selected_paths:
        return 1.0
    poisoned = sum(1 for path in selected_paths if "poison" in path.lower())
    return 1.0 - poisoned / len(selected_paths)

