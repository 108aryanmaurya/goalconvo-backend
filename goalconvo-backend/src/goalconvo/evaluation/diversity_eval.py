"""Lexical diversity metrics (Distinct-n style)."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np


class DiversityEvaluator:
    """Per-dialogue Distinct-1 / Distinct-2 averages over tokenized text."""

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    @staticmethod
    def distinct_for_tokens(tokens: List[str]) -> Tuple[float, float]:
        if not tokens:
            return 0.0, 0.0
        uniq_1 = len(set(tokens)) / len(tokens)
        bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
        uniq_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
        return uniq_1, uniq_2

    def dialogue_diversity(self, texts: List[str]) -> Dict[str, float]:
        if not texts:
            return {"distinct_1": 0.0, "distinct_2": 0.0, "combined": 0.0}
        per_d1, per_d2 = [], []
        for text in texts:
            tokens = self.tokenize(text)
            d1, d2 = self.distinct_for_tokens(tokens)
            if tokens:
                per_d1.append(d1)
                per_d2.append(d2)
        if not per_d1:
            return {"distinct_1": 0.0, "distinct_2": 0.0, "combined": 0.0}
        distinct_1 = float(np.mean(per_d1))
        distinct_2 = float(np.mean(per_d2))
        combined = (distinct_1 + distinct_2) / 2
        return {"distinct_1": distinct_1, "distinct_2": distinct_2, "combined": combined}

    def distinct_for_dialogue_text(self, text: str) -> Dict[str, float]:
        """Distinct-1 / Distinct-2 for a single concatenated dialogue string."""
        tokens = self.tokenize(text or "")
        d1, d2 = self.distinct_for_tokens(tokens)
        combined = (d1 + d2) / 2 if tokens else 0.0
        return {"distinct_1": float(d1), "distinct_2": float(d2), "combined": float(combined)}
