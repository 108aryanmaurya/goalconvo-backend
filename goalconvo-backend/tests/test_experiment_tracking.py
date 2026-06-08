"""Tests for experiment tracking and reproducibility metadata."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from goalconvo.experiments.tracking import (
    ExperimentRun,
    merge_repro_metadata,
    prompt_fingerprints,
)


@pytest.fixture
def tmp_cfg_llm(tmp_path):
    cfg = SimpleNamespace(
        data_dir=str(tmp_path),
        experiments_dir=str(tmp_path / "experiments_root"),
        experiment_seed=99,
        experiment_name="pytest",
        temperature=0.55,
        top_p=0.91,
        max_tokens=100,
        max_tokens_user_turn=40,
        max_tokens_supportbot_turn=80,
        max_tokens_planning=90,
        planner_temperature=0.11,
        planner_top_p=0.86,
        max_tokens_structured_planner=300,
        reflection_temperature=0.12,
        max_tokens_reflection=310,
        max_turns=12,
        min_turns=4,
        structured_planner_enabled=True,
        reflection_on_utterances_enabled=True,
    )
    llm = SimpleNamespace(
        api_config={"provider": "testprov", "model": "test-model-x"},
        config=cfg,
    )
    return cfg, llm


def test_prompt_fingerprints_non_empty():
    fp = prompt_fingerprints()
    assert isinstance(fp, dict)
    assert len(fp) >= 4


def test_experiment_run_creates_layout(tmp_cfg_llm):
    cfg, llm = tmp_cfg_llm
    run = ExperimentRun.start(cfg, llm, experiment_name="layout_test")
    assert run.run_root.is_dir()
    assert (run.run_root / "metadata.json").is_file()
    assert (run.run_root / "reproducibility.log").is_file()
    assert run.dialogues_dir.is_dir()
    assert run.planner_dir.is_dir()
    assert run.reflection_dir.is_dir()
    meta = json.loads((run.run_root / "metadata.json").read_text(encoding="utf-8"))
    assert meta["dialogue_count"] == 0
    assert meta["generation_hyperparameters"]["llm_model"] == "test-model-x"
    assert meta["generation_hyperparameters"]["experiment_seed"] == 99

    d = {
        "dialogue_id": "did-1",
        "goal": "g",
        "domain": "hotel",
        "turns": [
            {
                "role": "User",
                "text": "hi",
                "metadata": {"utterance_reflection": {"attempts": [{"verdict": {"score": 5}}]}},
            }
        ],
        "pipeline_turns": [{"turn": 0, "planner_output": {"user_phase": {"subgoal": "x"}}}],
        "metadata": {},
    }
    merge_repro_metadata(d, cfg, llm, experiment_run_id=run.run_id, prompt_versions=run._prompt_versions)
    assert "reproducibility" in d["metadata"]
    assert d["metadata"]["reproducibility"]["experiment_run_id"] == run.run_id

    run.record_dialogue(d)
    assert (run.dialogues_dir / "did-1.json").exists()
    assert (run.planner_dir / "did-1.json").exists()
    assert (run.reflection_dir / "did-1.json").exists()

    run.finalize({"ok": True})
    meta2 = json.loads((run.run_root / "metadata.json").read_text(encoding="utf-8"))
    assert meta2["dialogue_count"] == 1
    assert meta2["completed_at"] is not None
