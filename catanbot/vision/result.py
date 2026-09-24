"""Common result type for the screenshot parsers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..state import GameState


@dataclass
class ParseResult:
    parsed: dict                      # parsed screenshot in the schema of catanbot.vision.schema.PARSE_SCHEMA
    state: GameState                  # GameState built from ``parsed`` (parsed_to_state)
    confidence: Dict[str, float] = field(default_factory=dict)   # per field, 0..1
    warnings: List[str] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)          # geometry, intermediate masks, ...

    def to_dict(self) -> dict:
        return {"parsed": self.parsed, "state": self.state.to_dict(), "confidence": self.confidence,
                "warnings": list(self.warnings)}
