"""Tests for research-grade evaluation (suite, stats, CSV export)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.append(str(Path(__file__).parent.parent / "src"))

from goalconvo.config import Config
from goalconvo.evaluation.stats_utils import mean_std_ci, proportion_summary, wilson_interval
from goalconvo.evaluation.research_evaluator import (
    ResearchEvaluationSuite,
    evaluate_batch_chunks,
)


def test_mean_std_ci():
    s = mean_std_ci([1.0, 2.0, 3.0, 4.0], confidence=0.95)
    assert s["n"] == 4
    assert s["mean"] == 2.5
    assert s["ci_low"] < s["mean"] < s["ci_high"]


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(8, 10, confidence=0.95)
    assert 0.0 <= lo <= hi <= 1.0


def test_proportion_summary():
    p = proportion_summary(2, 10, confidence=0.95)
    assert p["rate"] == 0.2
    assert p["wilson_ci_low"] <= 0.2 <= p["wilson_ci_high"]


def test_research_suite_with_mock_bert(tmp_path):
    cfg = Config()
    mock_bert = MagicMock()
    mock_bert.score_pairs_batch.return_value = [0.72, 0.71]
    suite = ResearchEvaluationSuite(cfg, bert_evaluator=mock_bert)
    cands = [
        {
            "dialogue_id": "a",
            "domain": "hotel",
            "goal": "book",
            "turns": [
                {"role": "User", "text": "I need a hotel"},
                {"role": "SupportBot", "text": "Confirmed booking ref HOT-1"},
                {"role": "User", "text": "Thank you, perfect!"},
            ],
        },
        {
            "dialogue_id": "b",
            "domain": "hotel",
            "goal": "book",
            "turns": [{"role": "User", "text": "Hi"}],
        },
    ]
    refs = [
        {"domain": "hotel", "turns": [{"text": "multiwoz hotel reference one two three"}]},
        {"domain": "hotel", "turns": [{"text": "another hotel dialogue reference text"}]},
    ]
    rep = suite.evaluate(cands, refs)
    assert len(rep.per_dialogue) == 2
    assert rep.aggregate["goal_completion"]["n"] == 2
    assert "bertscore_f1" in rep.aggregate["metrics"]
    assert "semantic_cosine" in rep.aggregate["metrics"]
    paths = rep.save_csv(tmp_path / "run_eval")
    assert Path(paths["per_dialogue_csv"]).exists()
    assert Path(paths["summary_csv"]).exists()
    assert Path(paths["viz_json"]).exists()


def test_evaluate_batch_chunks_merges():
    cfg = Config()
    mock_bert = MagicMock()
    mock_bert.score_pairs_batch.side_effect = [[0.5], [0.6]]
    suite = ResearchEvaluationSuite(cfg, bert_evaluator=mock_bert)
    cands = [
        {"dialogue_id": "1", "domain": "hotel", "goal": "g", "turns": [{"role": "User", "text": "thanks"}]},
        {"dialogue_id": "2", "domain": "hotel", "goal": "g", "turns": [{"role": "User", "text": "thanks"}]},
    ]
    refs = [
        {"domain": "hotel", "turns": [{"text": "ref " * 5}]},
    ]
    rep = evaluate_batch_chunks(suite, cands, refs, chunk_size=1)
    assert len(rep.per_dialogue) == 2
    assert rep.meta.get("batched") is True
