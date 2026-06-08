"""Smoke tests for ablation study runner (no live LLM in dialogue loop)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.append(str(Path(__file__).parent.parent / "src"))

from goalconvo.config import Config
from goalconvo.evaluation.bertscore_eval import BertScoreEvaluator
from goalconvo.experiments.ablation import (
    ABLATION_ARM_SPECS,
    AblationStudyRunner,
    apply_ablation_config,
)


def test_apply_ablation_config():
    c = Config()
    spec = next(s for s in ABLATION_ARM_SPECS if s.name == "no_planner")
    c2 = apply_ablation_config(c, spec)
    assert c2.structured_planner_enabled is False
    assert c.structured_planner_enabled == c.structured_planner_enabled


def test_ablation_runner_smoke(tmp_path):
    cfg = Config()
    llm = MagicMock()

    def fake_runner(_cfg: Config, exp: dict) -> dict:
        return {
            "dialogue_id": exp.get("dialogue_id", "fake-1"),
            "goal": exp["goal"],
            "domain": exp.get("domain", "hotel"),
            "turns": [
                {"role": "User", "text": "I need a hotel tonight centre"},
                {"role": "SupportBot", "text": "Confirmed booking reference HOT-9 for 2 nights."},
                {"role": "User", "text": "Thank you, perfect!"},
            ],
        }

    refs = [
        {
            "domain": "hotel",
            "turns": [
                {"text": "multiwoz style user hotel"},
                {"text": "confirmed reservation ref ABC"},
            ],
        }
    ]

    mock_bert = MagicMock()
    mock_bert.score_pairs_batch.return_value = [0.55, 0.56]

    arms = tuple(s for s in ABLATION_ARM_SPECS if s.name in ("full", "no_reflection"))
    runner = AblationStudyRunner(cfg, llm, max_turns=2, multiwoz_reference_limit=5, arm_specs=arms)
    seeds = [
        {
            "goal": "Book hotel",
            "context": "c",
            "domain": "hotel",
            "user_persona": "u",
            "first_utterance": "Hi",
        }
    ]
    summary = runner.run(
        seeds,
        tmp_path,
        multiwoz_references=refs,
        bert_evaluator=mock_bert,
        dialogue_runner=fake_runner,
    )
    assert len(summary) == 2
    assert (tmp_path / "ablation_summary.csv").exists()
    assert (tmp_path / "ablation_comparison.md").exists()
