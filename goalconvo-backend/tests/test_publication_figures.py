"""Tests for publication matplotlib figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from goalconvo.visualization.publication_figures import (
    export_ablation_figure_bundle,
    export_evaluation_figure_bundle,
    plot_ablation_study_figure,
)


def test_ablation_figure_png_pdf(tmp_path: Path):
    summary = pd.DataFrame(
        [
            {
                "arm": "full",
                "n_dialogues": 3,
                "bertscore_mean": 0.4,
                "distinct1_mean": 0.5,
                "coherence_mean": 0.35,
                "goal_completion_score_mean": 0.7,
                "goal_completed_rate": 0.33,
            },
            {
                "arm": "no_planner",
                "n_dialogues": 3,
                "bertscore_mean": 0.35,
                "distinct1_mean": 0.48,
                "coherence_mean": 0.3,
                "goal_completion_score_mean": 0.55,
                "goal_completed_rate": 0.0,
            },
        ]
    )
    stem = tmp_path / "fig_ab"
    paths = plot_ablation_study_figure(summary, stem)
    assert Path(paths["png"]).is_file()
    assert Path(paths["pdf"]).is_file()


def test_export_ablation_bundle(tmp_path: Path):
    summary = pd.DataFrame(
        [
            {
                "arm": "a",
                "n_dialogues": 2,
                "bertscore_mean": 0.2,
                "distinct1_mean": 0.3,
                "coherence_mean": 0.25,
                "goal_completion_score_mean": 0.5,
                "goal_completed_rate": 0.5,
            }
        ]
    )
    summary.to_csv(tmp_path / "ablation_summary.csv", index=False)
    out = export_ablation_figure_bundle(tmp_path)
    assert "ablation_study" in out
    assert Path(out["ablation_study"][0]).exists()


def test_export_evaluation_bundle(tmp_path: Path):
    per = pd.DataFrame(
        [
            {
                "dialogue_id": "1",
                "domain": "hotel",
                "bertscore_f1": 0.5,
                "coherence_adjacent": 0.4,
                "goal_completed": 1,
            },
            {
                "dialogue_id": "2",
                "domain": "hotel",
                "bertscore_f1": 0.45,
                "coherence_adjacent": 0.35,
                "goal_completed": 0,
            },
        ]
    )
    prefix = tmp_path / "ev"
    per.to_csv(prefix.parent / f"{prefix.name}_per_dialogue.csv", index=False)
    outd = tmp_path / "figout"
    paths = export_evaluation_figure_bundle(prefix, outd)
    assert "domain_wise" in paths
    assert (outd / "publication" / "figure_manifest.json").is_file()
