"""Lexical semantic similarity vs a reference corpus (e.g. MultiWOZ) via TF–IDF cosine."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SemanticSimilarityEvaluator:
    """
    Max TF–IDF cosine similarity between a candidate dialogue and reference texts
    in the same domain (complements neural BERTScore).
    """

    def __init__(self, max_features: int = 8000, min_df: int = 1):
        self.max_features = max_features
        self.min_df = min_df
        # domain -> (vectorizer, ref_matrix, global_ref_indices aligned with matrix rows)
        self._by_domain: Dict[str, Tuple[TfidfVectorizer, Any, List[int]]] = {}

    def fit(
        self,
        reference_dialogues: List[Dict[str, Any]],
        extract_text: Callable[[Dict[str, Any]], str],
    ) -> None:
        self._by_domain.clear()
        by_dom: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for i, d in enumerate(reference_dialogues):
            dom = (d.get("domain") or "unknown").lower()
            text = (extract_text(d) or "").strip()
            if text:
                by_dom[dom].append((i, text))
        for dom, pairs in by_dom.items():
            indices = [p[0] for p in pairs]
            texts = [p[1] for p in pairs]
            if not texts:
                continue
            md = min(self.min_df, len(texts))
            vec = TfidfVectorizer(
                max_features=self.max_features,
                min_df=md,
                max_df=0.95,
                stop_words="english",
            )
            try:
                mat = vec.fit_transform(texts)
            except ValueError as e:
                logger.warning("SemanticSimilarity fit skipped for domain=%s: %s", dom, e)
                continue
            self._by_domain[dom] = (vec, mat, indices)

    def _pick_domain_key(self, dialogue: Dict[str, Any]) -> str:
        dom = (dialogue.get("domain") or "unknown").lower()
        if dom in self._by_domain:
            return dom
        if self._by_domain:
            return next(iter(self._by_domain.keys()))
        return dom

    def max_cosine(
        self,
        dialogue: Dict[str, Any],
        extract_text: Callable[[Dict[str, Any]], str],
    ) -> float:
        if not self._by_domain:
            return 0.0
        dom = self._pick_domain_key(dialogue)
        vec, mat, _ = self._by_domain[dom]
        qtext = (extract_text(dialogue) or "").strip()
        if not qtext:
            return 0.0
        try:
            q = vec.transform([qtext])
        except ValueError:
            return 0.0
        sims = cosine_similarity(q, mat).flatten()
        return float(np.max(sims)) if sims.size else 0.0

    def pick_best_reference_index(
        self,
        dialogue: Dict[str, Any],
        extract_text: Callable[[Dict[str, Any]], str],
    ) -> int:
        """Global index in the original reference list used at ``fit`` time."""
        if not self._by_domain:
            return 0
        dom = self._pick_domain_key(dialogue)
        vec, mat, indices = self._by_domain[dom]
        qtext = (extract_text(dialogue) or "").strip()
        if not qtext or not indices:
            return int(indices[0]) if indices else 0
        q = vec.transform([qtext])
        sims = cosine_similarity(q, mat).flatten()
        if not sims.size:
            return int(indices[0])
        j = int(np.argmax(sims))
        return int(indices[j]) if j < len(indices) else int(indices[0])
