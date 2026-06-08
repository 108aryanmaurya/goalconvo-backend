"""Heuristics for adaptive dialogue length using planner metadata (no extra LLM)."""

from __future__ import annotations

from typing import Any, Dict, Optional


def _goal_progress_from_meta(meta: Optional[Dict[str, Any]]) -> str:
    if not meta or not isinstance(meta, dict):
        return ""
    sp = meta.get("structured_planner")
    if not isinstance(sp, dict):
        return ""
    return str(sp.get("goal_progress") or "").lower()


def both_planners_near_complete(
    support_meta: Optional[Dict[str, Any]],
    user_meta: Optional[Dict[str, Any]],
) -> bool:
    """True when both agents' last structured plans report ``near_complete``-style progress."""
    s = _goal_progress_from_meta(support_meta)
    u = _goal_progress_from_meta(user_meta)
    if not s or not u:
        return False

    def _is_near_complete(progress: str) -> bool:
        p = (progress or "").lower()
        return "near" in p and "complete" in p

    return _is_near_complete(s) and _is_near_complete(u)
