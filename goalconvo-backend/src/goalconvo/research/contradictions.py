"""LLM-assisted contradiction / consistency checks (prompt-only)."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .json_util import extract_json_dict

logger = logging.getLogger(__name__)

_CONTRADICTION_PROMPT = """You check goal-oriented dialogue for factual consistency with tracked memory.

Output ONLY valid JSON:
{{
  "contradictions": [{{"description": "string", "severity": "low|medium|high"}}],
  "repair_hint": "one imperative sentence for the next assistant reply (empty if none)"
}}

Rules:
- Compare RECENT_TRANSCRIPT with MEMORY_JSON and the USER_GOAL.
- A contradiction is a direct clash (e.g. different party size, date, venue, price) not mere underspecification.
- If none, return "contradictions": [] and repair_hint "".

USER_GOAL:
{goal}

MEMORY_JSON:
{memory_json}

RECENT_TRANSCRIPT:
{transcript}
"""


def analyze_contradictions(
    llm_client: Any,
    config: Any,
    *,
    goal: str,
    transcript: str,
    memory_blob: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Returns dict with ``contradictions`` (list), ``repair_hint`` (str), and optional ``severity_max``.
    """
    try:
        memory_json = json.dumps(memory_blob or {}, ensure_ascii=False)[:6000]
    except (TypeError, ValueError):
        memory_json = "{}"
    prompt = _CONTRADICTION_PROMPT.format(
        goal=(goal or "").strip() or "(unspecified)",
        memory_json=memory_json,
        transcript=(transcript or "").strip() or "(empty)",
    )
    max_tok = int(getattr(config, "max_tokens_research_aux", 256))
    try:
        raw = llm_client.generate_completion(
            prompt,
            temperature=0.05,
            max_tokens=max_tok,
        )
    except Exception as e:
        logger.warning("analyze_contradictions: LLM failed: %s", e)
        return {"contradictions": [], "repair_hint": "", "error": str(e)}
    data = extract_json_dict(raw or "")
    if not data:
        return {"contradictions": [], "repair_hint": "", "parse_error": True}
    raw_list = data.get("contradictions") or []
    contradictions: List[Dict[str, str]] = []
    if isinstance(raw_list, list):
        for item in raw_list[:8]:
            if not isinstance(item, dict):
                continue
            desc = str(item.get("description") or "").strip()
            if not desc:
                continue
            sev = str(item.get("severity") or "low").strip().lower()
            if sev not in ("low", "medium", "high"):
                sev = "low"
            contradictions.append({"description": desc, "severity": sev})
    repair = str(data.get("repair_hint") or "").strip()
    sev_rank = {"low": 1, "medium": 2, "high": 3}
    max_sev = 0
    for c in contradictions:
        max_sev = max(max_sev, sev_rank.get(c.get("severity", "low"), 1))
    inv_rank = {1: "low", 2: "medium", 3: "high"}
    return {
        "contradictions": contradictions,
        "repair_hint": repair,
        "severity_max": inv_rank.get(max_sev, "none") if contradictions else "none",
    }
