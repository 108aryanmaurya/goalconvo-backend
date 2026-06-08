"""JSON extraction for small research-side LLM calls (avoid import cycles with planner_agent)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


def extract_json_dict(text: str) -> Optional[Dict[str, Any]]:
    if not text or not str(text).strip():
        return None
    raw = str(text).strip()
    if "```" in raw:
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I | re.M)
        raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        val = json.loads(raw[start : end + 1])
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        return None
