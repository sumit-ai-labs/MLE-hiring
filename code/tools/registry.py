"""Internal tool registry loaded from repository schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ToolRegistry:
    def __init__(self, spec_path: Path):
        self.spec_path = spec_path
        self.tools = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        raw = json.loads(self.spec_path.read_text(encoding="utf-8"))
        return {tool["name"]: tool for tool in raw}

    def get(self, name: str) -> dict[str, Any] | None:
        return self.tools.get(name)
