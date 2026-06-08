"""Lightweight, prompt-based research extensions (no fine-tuning)."""

from .adaptive import both_planners_near_complete
from .contradictions import analyze_contradictions
from .decomposition import run_dynamic_goal_decomposition
from .summarization import update_rolling_summary

__all__ = [
    "both_planners_near_complete",
    "analyze_contradictions",
    "run_dynamic_goal_decomposition",
    "update_rolling_summary",
]
