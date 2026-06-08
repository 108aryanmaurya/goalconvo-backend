"""Rolling transcript summarization for long dialogues (prompt-only)."""

from __future__ import annotations

import logging
from typing import Any

from .json_util import extract_json_dict

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """You maintain a compact rolling summary for a goal-oriented dialogue simulator.
Merge PRIOR_SUMMARY with NEW_LINES into an updated summary for the planner and agents.

Output ONLY valid JSON: {{"summary": "<= ~12 short bullet sentences separated by semicolons; no markdown>"}}

Rules:
- Preserve grounded facts (names, times, counts, references) when stated in NEW_LINES.
- Drop pleasantries; flag unresolved conflicts if both sides disagree on a fact.
- If PRIOR_SUMMARY is empty, summarize NEW_LINES only.

User goal: {goal}

PRIOR_SUMMARY:
{prior}

NEW_LINES:
{chunk}
"""


def update_rolling_summary(
    llm_client: Any,
    config: Any,
    *,
    goal: str,
    prior_summary: str,
    new_transcript_chunk: str,
) -> str:
    if not (new_transcript_chunk or "").strip():
        return (prior_summary or "").strip()
    prompt = _SUMMARY_PROMPT.format(
        goal=(goal or "").strip() or "(unspecified)",
        prior=(prior_summary or "").strip() or "(empty)",
        chunk=new_transcript_chunk.strip(),
    )
    max_tok = int(getattr(config, "max_tokens_research_aux", 256))
    try:
        raw = llm_client.generate_completion(
            prompt,
            temperature=0.1,
            max_tokens=max_tok,
        )
    except Exception as e:
        logger.warning("update_rolling_summary: LLM failed: %s", e)
        return (prior_summary or "").strip()
    data = extract_json_dict(raw or "")
    if not data:
        return (prior_summary or "").strip()
    summary = str(data.get("summary") or "").strip()
    if not summary:
        return (prior_summary or "").strip()
    return summary[:4000]
