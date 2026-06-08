"""BERTScore-based semantic similarity (synthetic vs reference dialogues)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from bert_score import score as bert_score

logger = logging.getLogger(__name__)


class BertScoreEvaluator:
    """Isolated BERTScore computation for ablations and alternative encoders."""

    BERTSCORE_FALLBACK_MODEL = "bert-base-uncased"

    def __init__(self, model_type: str):
        self.model_type = model_type

    def score_one_pair(self, cand_text: str, ref_text: str, max_chars: int = 1000) -> Optional[float]:
        def run_bertscore(cand: str, ref: str, model: str):
            _p, _r, f1 = bert_score([cand], [ref], model_type=model, verbose=False)
            return float(f1.item())

        try:
            return run_bertscore(cand_text, ref_text, self.model_type)
        except (OverflowError, ValueError, Exception) as e:
            err_str = str(e).lower()
            if "int too big to convert" not in err_str and "overflow" not in err_str:
                logger.warning("Error computing BERTScore: %s", e)
                return None
        for cap in [400, 200]:
            c = cand_text[:cap] if len(cand_text) > cap else cand_text
            r = ref_text[:cap] if len(ref_text) > cap else ref_text
            try:
                return run_bertscore(c, r, self.model_type)
            except (OverflowError, ValueError, Exception):
                continue
        try:
            c = cand_text[:512] if len(cand_text) > 512 else cand_text
            r = ref_text[:512] if len(ref_text) > 512 else ref_text
            return run_bertscore(c, r, self.BERTSCORE_FALLBACK_MODEL)
        except Exception as e:
            logger.warning("Error computing BERTScore (fallback model): %s", e)
            return None

    def score_pairs_batch(
        self,
        cand_texts: List[str],
        ref_texts: List[str],
        max_chars: int = 1000,
    ) -> List[Optional[float]]:
        """Batch BERTScore F1 for parallel candidate/reference strings (same length)."""
        if len(cand_texts) != len(ref_texts):
            raise ValueError("cand_texts and ref_texts must have the same length")

        def trunc(xs: List[str]) -> List[str]:
            return [t[:max_chars] if len(t) > max_chars else t for t in xs]

        cands = trunc(cand_texts)
        refs = trunc(ref_texts)

        def run_batch(c: List[str], r: List[str], model: str) -> List[float]:
            _p, _r, f1 = bert_score(c, r, model_type=model, verbose=False)
            arr = np.asarray(f1.detach().cpu().numpy() if hasattr(f1, "detach") else f1).ravel()
            return [float(arr[i]) for i in range(len(c))]

        try:
            return run_batch(cands, refs, self.model_type)
        except (OverflowError, ValueError, Exception) as e:
            err_str = str(e).lower()
            if "int too big" not in err_str and "overflow" not in err_str:
                logger.warning("BERTScore batch failed: %s; falling back per-pair", e)
        out: List[Optional[float]] = []
        for c, r in zip(cands, refs):
            out.append(self.score_one_pair(c, r, max_chars))
        return out

    def best_match_score(
        self,
        synthetic_text: str,
        real_dialogues_same_domain: List[Dict[str, Any]],
        extract_text_fn,
        max_chars: int = 1000,
    ) -> float:
        syn_trunc = synthetic_text[:max_chars] if len(synthetic_text) > max_chars else synthetic_text
        best = 0.0
        for real_dialogue in real_dialogues_same_domain:
            real_text = extract_text_fn(real_dialogue)
            ref_trunc = real_text[:max_chars] if len(real_text) > max_chars else real_text
            score = self.score_one_pair(syn_trunc, ref_trunc, max_chars)
            if score is not None:
                best = max(best, score)
        return best
