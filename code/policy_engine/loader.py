"""Load declarative policy files without adding runtime dependencies."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


POLICY_DIR = Path(__file__).resolve().parent / "policies"


@lru_cache(maxsize=4)
def _load_policies_cached(policy_dir: str) -> tuple[dict[str, Any], ...]:
    policies: list[dict[str, Any]] = []
    for path in sorted(Path(policy_dir).glob("*.yaml")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for policy in data.get("policies", []):
            item = dict(policy)
            item["source"] = path.name
            policies.append(item)
    return tuple(sorted(policies, key=lambda item: (int(item.get("priority", 1000)), str(item.get("name", "")))))


def load_policies(policy_dir: Path = POLICY_DIR) -> list[dict[str, Any]]:
    """Load JSON-compatible YAML policy files.

    JSON is valid YAML, so these `.yaml` files remain declarative while avoiding
    a PyYAML dependency that could disturb V1 reproducibility.
    """

    return [dict(policy) for policy in _load_policies_cached(str(policy_dir))]
