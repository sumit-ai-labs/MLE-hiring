"""Deterministic no-side-effect executor."""

from __future__ import annotations


def execute_actions(actions: list[dict]) -> list[dict]:
    executed = []
    for action in actions:
        executed.append({**action, "status": "planned"})
    return executed
