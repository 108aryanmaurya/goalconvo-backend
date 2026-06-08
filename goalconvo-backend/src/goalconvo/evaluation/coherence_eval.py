"""Turn-level coherence proxies (interpretability / lightweight metrics)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..utils import calculate_similarity


class CoherenceEvaluator:
    """Mean adjacent-turn similarity over User/SupportBot text (research baseline)."""

    @staticmethod
    def adjacent_similarity_mean(dialogue: Dict[str, Any]) -> float:
        turns = dialogue.get("turns", [])
        if len(turns) < 2:
            return 0.0
        vals = []
        for i in range(1, len(turns)):
            a = (turns[i - 1].get("text") or "").strip()
            b = (turns[i].get("text") or "").strip()
            if a and b:
                vals.append(calculate_similarity(a, b))
        return sum(vals) / len(vals) if vals else 0.0

    def corpus_mean(self, dialogues: List[Dict[str, Any]]) -> Dict[str, float]:
        scores = [self.adjacent_similarity_mean(d) for d in dialogues]
        if not scores:
            return {"mean_adjacent_similarity": 0.0, "std_adjacent_similarity": 0.0}
        arr = np.asarray(scores, dtype=float)
        std = float(np.std(arr, ddof=1)) if len(scores) > 1 else 0.0
        return {
            "mean_adjacent_similarity": float(np.mean(arr)),
            "std_adjacent_similarity": std,
        }

    def per_dialogue_scores(self, dialogues: List[Dict[str, Any]]) -> List[float]:
        return [float(self.adjacent_similarity_mean(d)) for d in dialogues]
