"""Research-grade evaluation vs MultiWOZ: batch metrics, statistics, CSV, and viz exports."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ..config import Config
from .bertscore_eval import BertScoreEvaluator
from .coherence_eval import CoherenceEvaluator
from .diversity_eval import DiversityEvaluator
from .goal_eval import GoalEvaluator
from .semantic_eval import SemanticSimilarityEvaluator
from .stats_utils import mean_std_ci, proportion_summary, stack_metric_rows

logger = logging.getLogger(__name__)


def extract_dialogue_text(dialogue: Dict[str, Any]) -> str:
    return " ".join((t.get("text") or "").strip() for t in dialogue.get("turns", []) or [])


@dataclass
class ResearchEvaluationReport:
    """Per-dialogue metrics, corpus aggregates with CIs, and export helpers."""

    per_dialogue: List[Dict[str, Any]]
    aggregate: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.per_dialogue)

    def to_viz_dict(self) -> Dict[str, Any]:
        """Payload suitable for error bars / faceted plots (Plotly, Seaborn, Vega-Lite)."""
        series: List[Dict[str, Any]] = []
        for name, stats in (self.aggregate.get("metrics") or {}).items():
            if not isinstance(stats, dict):
                continue
            series.append(
                {
                    "metric": name,
                    "mean": stats.get("mean"),
                    "std": stats.get("std"),
                    "ci_low": stats.get("ci_low"),
                    "ci_high": stats.get("ci_high"),
                    "n": stats.get("n"),
                }
            )
        gc = self.aggregate.get("goal_completion") or {}
        return {
            "meta": self.meta,
            "metric_series": series,
            "goal_completion": gc,
            "diversity_corpus": self.aggregate.get("diversity_corpus_comparison"),
        }

    def save_csv(self, output_prefix: str | Path) -> Dict[str, str]:
        """
        Write:
        - ``*_per_dialogue.csv`` — one row per candidate dialogue (viz-friendly long table).
        - ``*_summary_long.csv`` — one row per metric with mean/std or Wilson bounds.
        - ``*_viz.json`` — compact series for dashboards.
        """
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        per_path = prefix.parent / f"{prefix.name}_per_dialogue.csv"
        sum_path = prefix.parent / f"{prefix.name}_summary_long.csv"
        viz_path = prefix.parent / f"{prefix.name}_viz.json"
        self.to_dataframe().to_csv(per_path, index=False)
        rows: List[Dict[str, Any]] = []
        for name, stats in (self.aggregate.get("metrics") or {}).items():
            if isinstance(stats, dict):
                rows.append({"metric": name, **stats})
        gc = self.aggregate.get("goal_completion") or {}
        if gc.get("n"):
            rows.append(
                {
                    "metric": "goal_completion_rate",
                    "mean": gc.get("rate"),
                    "std": None,
                    "ci_low": gc.get("wilson_ci_low"),
                    "ci_high": gc.get("wilson_ci_high"),
                    "n": gc.get("n"),
                    "successes": gc.get("successes"),
                }
            )
        pd.DataFrame(rows).to_csv(sum_path, index=False)
        with open(viz_path, "w", encoding="utf-8") as fp:
            json.dump(self.to_viz_dict(), fp, indent=2, default=str)
        return {
            "per_dialogue_csv": str(per_path),
            "summary_csv": str(sum_path),
            "viz_json": str(viz_path),
        }


class ResearchEvaluationSuite:
    """
    Compare candidate dialogues against a MultiWOZ reference corpus.

    Metrics:
    - **BERTScore** F1 vs TF–IDF–picked reference in the same domain (batched).
    - **Distinct-1 / Distinct-2** per dialogue + corpus-level comparison.
    - **Goal completion** (binary) with Wilson CI on the rate.
    - **Coherence** — mean adjacent-turn lexical similarity per dialogue.
    - **Semantic similarity** — max TF–IDF cosine to in-domain MultiWOZ references.
    """

    def __init__(
        self,
        config: Config,
        *,
        confidence: float = 0.95,
        bert_evaluator: Optional[BertScoreEvaluator] = None,
    ):
        self.config = config
        self.confidence = confidence
        self._bert = bert_evaluator or BertScoreEvaluator(config.bertscore_model)
        self._div = DiversityEvaluator()
        self._goal = GoalEvaluator()
        self._coh = CoherenceEvaluator()
        self._sem = SemanticSimilarityEvaluator()

    @staticmethod
    def default_multiwoz_path(config: Config) -> Path:
        return Path(config.multiwoz_dir) / "processed_dialogues.json"

    @staticmethod
    def load_multiwoz_processed(
        config: Config,
        *,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        from ..utils import load_json

        path = ResearchEvaluationSuite.default_multiwoz_path(config)
        if not path.exists():
            raise FileNotFoundError(f"MultiWOZ processed file not found: {path}")
        data = load_json(str(path))
        if not isinstance(data, list):
            raise TypeError("processed_dialogues.json must contain a JSON list")
        return data[:limit] if limit else data

    def evaluate(
        self,
        candidate_dialogues: List[Dict[str, Any]],
        multiwoz_references: List[Dict[str, Any]],
        extract_text: Callable[[Dict[str, Any]], str] = extract_dialogue_text,
        *,
        max_chars_bert: int = 1000,
    ) -> ResearchEvaluationReport:
        if not candidate_dialogues:
            raise ValueError("candidate_dialogues must be non-empty")
        if not multiwoz_references:
            raise ValueError("multiwoz_references must be non-empty")

        self._sem.fit(multiwoz_references, extract_text)

        syn_texts = [extract_text(d) for d in candidate_dialogues]
        ref_texts_all = [extract_text(d) for d in multiwoz_references]

        ref_idx_for_bert = [
            self._sem.pick_best_reference_index(d, extract_text) for d in candidate_dialogues
        ]
        paired_refs = [
            ref_texts_all[i] if 0 <= i < len(ref_texts_all) else ref_texts_all[0]
            for i in ref_idx_for_bert
        ]

        try:
            bert_scores = self._bert.score_pairs_batch(
                syn_texts, paired_refs, max_chars=max_chars_bert
            )
        except Exception as e:
            logger.exception("BERTScore batch failed: %s", e)
            bert_scores = [None] * len(candidate_dialogues)

        per_rows: List[Dict[str, Any]] = []
        goals_ok = 0
        for i, d in enumerate(candidate_dialogues):
            txt = syn_texts[i]
            dist = self._div.distinct_for_dialogue_text(txt)
            coh = float(self._coh.adjacent_similarity_mean(d))
            sem = float(self._sem.max_cosine(d, extract_text))
            gok = self._goal.goal_completed(d)
            if gok:
                goals_ok += 1
            bf = bert_scores[i] if i < len(bert_scores) else None
            per_rows.append(
                {
                    "dialogue_id": d.get("dialogue_id", f"candidate_{i}"),
                    "domain": d.get("domain", "unknown"),
                    "bertscore_f1": bf,
                    "semantic_cosine": sem,
                    "distinct_1": dist["distinct_1"],
                    "distinct_2": dist["distinct_2"],
                    "distinct_combined": dist["combined"],
                    "coherence_adjacent": coh,
                    "goal_completed": int(gok),
                    "matched_multiwoz_index": ref_idx_for_bert[i],
                }
            )

        metric_keys = [
            "bertscore_f1",
            "semantic_cosine",
            "distinct_1",
            "distinct_2",
            "distinct_combined",
            "coherence_adjacent",
        ]
        metrics_summary = stack_metric_rows(per_rows, metric_keys, self.confidence)
        goal_summary = proportion_summary(goals_ok, len(candidate_dialogues), self.confidence)

        div_syn = self._div.dialogue_diversity(syn_texts)
        div_ref = self._div.dialogue_diversity(ref_texts_all)

        aggregate: Dict[str, Any] = {
            "metrics": metrics_summary,
            "goal_completion": goal_summary,
            "diversity_corpus_comparison": {
                "candidate": div_syn,
                "multiwoz_reference_sample": div_ref,
            },
            "coherence_corpus_mean": mean_std_ci(
                [r["coherence_adjacent"] for r in per_rows],
                self.confidence,
            ),
        }

        meta = {
            "confidence": self.confidence,
            "n_candidates": len(candidate_dialogues),
            "n_multiwoz_refs": len(multiwoz_references),
            "bertscore_model": self.config.bertscore_model,
            "reference_source": "MultiWOZ",
        }

        return ResearchEvaluationReport(
            per_dialogue=per_rows, aggregate=aggregate, meta=meta
        )


def evaluate_batch_chunks(
    suite: ResearchEvaluationSuite,
    candidate_dialogues: List[Dict[str, Any]],
    multiwoz_references: List[Dict[str, Any]],
    *,
    chunk_size: int = 256,
    extract_text: Callable[[Dict[str, Any]], str] = extract_dialogue_text,
) -> ResearchEvaluationReport:
    """Evaluate large candidate sets in chunks; recomputes global summaries on merged rows."""
    if chunk_size <= 0 or len(candidate_dialogues) <= chunk_size:
        return suite.evaluate(candidate_dialogues, multiwoz_references, extract_text)

    all_rows: List[Dict[str, Any]] = []
    for start in range(0, len(candidate_dialogues), chunk_size):
        chunk = candidate_dialogues[start : start + chunk_size]
        rep = suite.evaluate(chunk, multiwoz_references, extract_text)
        all_rows.extend(rep.per_dialogue)

    syn_texts_all = [extract_text(d) for d in candidate_dialogues]
    ref_texts_all = [extract_text(d) for d in multiwoz_references]
    metric_keys = [
        "bertscore_f1",
        "semantic_cosine",
        "distinct_1",
        "distinct_2",
        "distinct_combined",
        "coherence_adjacent",
    ]
    goals_ok = sum(int(r["goal_completed"]) for r in all_rows)
    metrics_summary = stack_metric_rows(all_rows, metric_keys, suite.confidence)
    aggregate_merged: Dict[str, Any] = {
        "metrics": metrics_summary,
        "goal_completion": proportion_summary(goals_ok, len(all_rows), suite.confidence),
        "diversity_corpus_comparison": {
            "candidate": suite._div.dialogue_diversity(syn_texts_all),
            "multiwoz_reference_sample": suite._div.dialogue_diversity(ref_texts_all),
        },
        "coherence_corpus_mean": mean_std_ci(
            [r["coherence_adjacent"] for r in all_rows],
            suite.confidence,
        ),
    }
    meta_info = {
        "confidence": suite.confidence,
        "n_candidates": len(all_rows),
        "n_multiwoz_refs": len(multiwoz_references),
        "bertscore_model": suite.config.bertscore_model,
        "reference_source": "MultiWOZ",
        "batched": True,
        "chunk_size": chunk_size,
    }
    return ResearchEvaluationReport(
        per_dialogue=all_rows, aggregate=aggregate_merged, meta=meta_info
    )
