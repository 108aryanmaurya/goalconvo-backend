"""Extra planner instructions when research flags are enabled (prompt-only)."""

from __future__ import annotations

from typing import Any


def structured_planner_research_suffix(config: Any) -> str:
    """Appended to the structured planner user message (after history block)."""
    parts: list[str] = []
    if getattr(config, "research_hierarchical_subgoals", False):
        parts.append(
            "Research extension — hierarchical subgoals:\n"
            'Include optional JSON keys alongside the required keys: '
            '"subgoal_hierarchy" (array of strings, coarse parent → fine leaf; the last item should align with "subgoal") '
            'and "decomposition_notes" (short string explaining how this hierarchy fits the transcript). '
            "Use [] and \"\" when not applicable."
        )
    if getattr(config, "research_dynamic_goal_decomposition", False):
        parts.append(
            "Research extension — dynamic decomposition:\n"
            "If the user goal block listed simulator-provided milestones (dynamic subgoals), keep your "
            '"subgoal_hierarchy" and active "subgoal" consistent with the next unfinished milestone unless the transcript clearly completed it.'
        )
    if not parts:
        return ""
    return "\n\n" + "\n\n".join(parts)
