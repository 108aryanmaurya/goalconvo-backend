"""One-shot LLM goal decomposition into ordered milestones (prompt-only)."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .json_util import extract_json_dict

logger = logging.getLogger(__name__)

_DECOMPOSE_PROMPT = """You decompose a user goal for a multi-turn task-oriented dialogue simulator.
Output ONLY valid JSON (no markdown fences) with exactly these keys:
{{
  "milestones": ["string", ...],
  "notes": "one short paragraph"
}}

Rules:
- 3–8 milestones, ordered from first conversational move to successful completion.
- Each milestone is testable in dialogue (what should be true before moving on).
- "notes" explains dependencies or domain assumptions (max ~80 words).

Domain: {domain}
Context:
{context}

User goal:
{goal}
"""


def run_dynamic_goal_decomposition(
    llm_client: Any,
    config: Any,
    *,
    goal: str,
    context: str,
    domain: str,
) -> Dict[str, Any]:
    """
    Returns ``{"milestones": [...], "notes": str}``; safe on LLM/parse failure (empty milestones).
    """
    prompt = _DECOMPOSE_PROMPT.format(
        domain=domain or "general",
        context=(context or "(none)").strip() or "(none)",
        goal=(goal or "").strip() or "(unspecified)",
    )
    max_tok = int(getattr(config, "max_tokens_research_aux", 256))
    try:
        raw = llm_client.generate_completion(
            prompt,
            temperature=0.12,
            max_tokens=max_tok,
        )
    except Exception as e:
        logger.warning("run_dynamic_goal_decomposition: LLM failed: %s", e)
        return {"milestones": [], "notes": "", "error": str(e)}
    data = extract_json_dict(raw or "")
    if not data:
        logger.warning("run_dynamic_goal_decomposition: JSON parse failed")
        return {"milestones": [], "notes": "", "parse_error": True}
    milestones_raw = data.get("milestones") or data.get("subgoals") or []
    milestones: List[str] = []
    if isinstance(milestones_raw, list):
        milestones = [str(x).strip() for x in milestones_raw if str(x).strip()][:12]
    notes = str(data.get("notes") or "").strip()
    return {"milestones": milestones, "notes": notes}
