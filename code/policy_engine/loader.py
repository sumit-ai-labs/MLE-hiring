"""Load declarative policy files without adding runtime dependencies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


POLICY_DIR = Path(__file__).resolve().parent / "policies"


def load_policies(policy_dir: Path = POLICY_DIR) -> list[dict[str, Any]]:
    """Load JSON-compatible YAML policy files.

    JSON is valid YAML, so these `.yaml` files remain declarative while avoiding
    a PyYAML dependency that could disturb V1 reproducibility.
    """

    policies: list[dict[str, Any]] = []
    for path in sorted(policy_dir.glob("*.yaml")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for policy in data.get("policies", []):
            policy = dict(policy)
            policy["source"] = path.name
            policies.append(policy)
    return sorted(policies, key=lambda item: (int(item.get("priority", 1000)), str(item.get("name", ""))))

