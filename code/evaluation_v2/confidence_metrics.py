"""Confidence metrics for V2 evaluation."""

from __future__ import annotations


def confidence_distribution(scores: list[float]) -> dict[str, int]:
    buckets = {"low": 0, "medium": 0, "high": 0}
    for score in scores:
        if score < 0.5:
            buckets["low"] += 1
        elif score < 0.8:
            buckets["medium"] += 1
        else:
            buckets["high"] += 1
    return buckets


def calibration_error(predicted: list[float], observed: list[float]) -> float:
    if not predicted or len(predicted) != len(observed):
        return 0.0
    return round(sum(abs(p - o) for p, o in zip(predicted, observed)) / len(predicted), 4)


def uncertainty_handling(scores: list[float], ambiguous_flags: list[bool]) -> float:
    if not scores or len(scores) != len(ambiguous_flags):
        return 0.0
    ambiguous = [(score, flag) for score, flag in zip(scores, ambiguous_flags) if flag]
    if not ambiguous:
        return 1.0
    return sum(1 for score, _ in ambiguous if score <= 0.65) / len(ambiguous)

