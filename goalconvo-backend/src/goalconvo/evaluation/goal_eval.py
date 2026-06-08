"""Goal relevance / satisfaction heuristics for synthetic dialogues."""

from __future__ import annotations

from typing import Any, Dict, List


class GoalEvaluator:
    """Keyword-style goal satisfaction; swap for LLM-as-judge in experiments."""

    @staticmethod
    def goal_satisfied(goal: str, turns: List[Dict[str, str]]) -> bool:
        if not turns:
            return False
        recent_turns = turns[-3:] if len(turns) >= 3 else turns
        completion_keywords = [
            "thank you",
            "thanks",
            "perfect",
            "great",
            "excellent",
            "that's great",
            "that works",
            "sounds good",
            "all set",
            "i'm all set",
            "that's exactly what I needed",
            "that'll work",
            "booked",
            "confirmed",
            "reserved",
            "done",
            "completed",
            "appreciate it",
            "good, thank",
        ]
        for turn in recent_turns:
            if turn.get("role") == "User":
                text = turn.get("text", "").lower()
                if any(keyword in text for keyword in completion_keywords):
                    return True
        return False

    def corpus_goal_relevance(
        self, synthetic_dialogues: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        goal_satisfied_count = 0
        domain_goal_satisfaction: Dict[str, Dict[str, int]] = {}
        for dialogue in synthetic_dialogues:
            domain = dialogue.get("domain", "unknown")
            goal = dialogue.get("goal", "")
            turns = dialogue.get("turns", [])
            is_satisfied = self.goal_satisfied(goal, turns)
            if is_satisfied:
                goal_satisfied_count += 1
            if domain not in domain_goal_satisfaction:
                domain_goal_satisfaction[domain] = {"satisfied": 0, "total": 0}
            domain_goal_satisfaction[domain]["total"] += 1
            if is_satisfied:
                domain_goal_satisfaction[domain]["satisfied"] += 1
        total_dialogues = len(synthetic_dialogues)
        overall = goal_satisfied_count / total_dialogues if total_dialogues > 0 else 0.0
        domain_goal_relevance = {}
        for domain, stats in domain_goal_satisfaction.items():
            domain_goal_relevance[domain] = {
                "satisfied": stats["satisfied"],
                "total": stats["total"],
                "percentage": stats["satisfied"] / stats["total"] if stats["total"] > 0 else 0.0,
            }
        return {
            "overall_goal_relevance": overall,
            "domain_goal_relevance": domain_goal_relevance,
            "target_goal_relevance": 0.85,
        }

    def goal_completed(self, dialogue: Dict[str, Any]) -> bool:
        """Binary goal completion (same heuristic as corpus aggregation)."""
        goal = dialogue.get("goal", "") or ""
        turns = dialogue.get("turns", []) or []
        return bool(self.goal_satisfied(goal, turns))
