"""
Ablation study framework: toggle planner, memory, reflection, and single-agent (support-only LLM).

Produces CSV summaries, Markdown comparison tables, and matplotlib bar plots.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..evaluation.bertscore_eval import BertScoreEvaluator
from ..evaluation.goal_completion_eval import GoalCompletionEvaluator
from ..evaluation.research_evaluator import ResearchEvaluationSuite
from ..llm_client import LLMClient
from ..pipeline.dialogue_pipeline import DialoguePipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AblationArmSpec:
    """Named ablation arm: human label + Config field overrides."""

    name: str
    description: str
    overrides: Dict[str, Any]


# Presets aligned with common GoalConvo ablations
ABLATION_ARM_SPECS: Tuple[AblationArmSpec, ...] = (
    AblationArmSpec("full", "Full multi-agent system", {}),
    AblationArmSpec(
        "no_planner",
        "Structured planner + planning instructions off",
        {"structured_planner_enabled": False, "agent_planning_enabled": False},
    ),
    AblationArmSpec(
        "no_memory",
        "DialogueMemory refresh / memory sections off",
        {"agent_memory_enabled": False},
    ),
    AblationArmSpec(
        "no_reflection",
        "Utterance-level reflection / regen loop off",
        {"reflection_on_utterances_enabled": False},
    ),
    AblationArmSpec(
        "single_agent",
        "Support-only LLM; user lines are templated (no User-agent generation)",
        {
            "single_agent_support_only": True,
            "reflection_on_utterances_enabled": False,
        },
    ),
)


def apply_ablation_config(base: Config, spec: AblationArmSpec) -> Config:
    """Return a shallow copy of ``Config`` with arm-specific overrides."""
    if not spec.overrides:
        return base
    return replace(base, **spec.overrides)


def compute_arm_metrics(
    dialogues: List[Dict[str, Any]],
    multiwoz_references: List[Dict[str, Any]],
    config: Config,
    *,
    bert_evaluator: Optional[BertScoreEvaluator] = None,
) -> Dict[str, Any]:
    """
    Aggregate BERTScore (mean F1 vs matched ref), Distinct-1, coherence, goal completion
    for one arm's dialogue list.
    """
    if not dialogues:
        return {
            "n_dialogues": 0,
            "bertscore_mean": float("nan"),
            "distinct1_mean": float("nan"),
            "coherence_mean": float("nan"),
            "goal_completion_score_mean": float("nan"),
            "goal_completed_rate": float("nan"),
        }

    if not multiwoz_references:
        dom = (dialogues[0].get("domain") or "hotel").lower()
        multiwoz_references = [
            {
                "domain": dom,
                "turns": [
                    {"role": "User", "text": "I need a booking in the centre tonight."},
                    {"role": "SupportBot", "text": "Confirmed reservation reference REF-001."},
                ],
            }
        ]

    suite = ResearchEvaluationSuite(config, bert_evaluator=bert_evaluator)
    rep = suite.evaluate(dialogues, multiwoz_references)
    df = rep.to_dataframe()

    bert_col = df["bertscore_f1"].dropna()
    bert_mean = float(bert_col.mean()) if len(bert_col) else float("nan")

    gc_eval = GoalCompletionEvaluator(config)
    gc_scores: List[float] = []
    gc_hits = 0
    for d in dialogues:
        r = gc_eval.evaluate(d, use_llm_judge=False)
        gc_scores.append(float(r["completion_score"]))
        if r.get("goal_completed"):
            gc_hits += 1

    return {
        "n_dialogues": len(dialogues),
        "bertscore_mean": bert_mean,
        "distinct1_mean": float(df["distinct_1"].mean()),
        "coherence_mean": float(df["coherence_adjacent"].mean()),
        "goal_completion_score_mean": float(np.mean(gc_scores)) if gc_scores else float("nan"),
        "goal_completed_rate": gc_hits / len(dialogues),
    }


def _summary_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    sep = "|" + "|".join("---" for _ in headers) + "|"
    head = "|" + "|".join(headers) + "|"
    lines = [head, sep]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, bool):
                cells.append(str(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("|" + "|".join(cells) + "|")
    return "\n".join(lines) + "\n"


def _plot_ablation_bars(summary: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        "bertscore_mean",
        "distinct1_mean",
        "coherence_mean",
        "goal_completion_score_mean",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes.ravel()
    arms = summary["arm"].tolist()
    x = np.arange(len(arms))
    for ax, m in zip(axes_flat, metrics):
        vals = summary[m].astype(float).values
        ax.bar(x, vals, color="steelblue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(arms, rotation=25, ha="right")
        ax.set_ylabel(m.replace("_", " "))
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


class AblationStudyRunner:
    """
    Automatic runner: for each ablation arm, simulate dialogues and compute metrics vs MultiWOZ.
    """

    def __init__(
        self,
        base_config: Config,
        llm_client: LLMClient,
        *,
        max_turns: int = 4,
        multiwoz_reference_limit: int = 40,
        arm_specs: Sequence[AblationArmSpec] = ABLATION_ARM_SPECS,
    ):
        self.base_config = base_config
        self.llm_client = llm_client
        self.max_turns = max_turns
        self.multiwoz_reference_limit = multiwoz_reference_limit
        self.arm_specs = tuple(arm_specs)

    def load_multiwoz_references(self) -> List[Dict[str, Any]]:
        try:
            return ResearchEvaluationSuite.load_multiwoz_processed(
                self.base_config,
                limit=self.multiwoz_reference_limit,
            )
        except (FileNotFoundError, OSError, TypeError, ValueError) as e:
            logger.warning("Could not load MultiWOZ references: %s", e)
            return []

    def run(
        self,
        experience_seeds: List[Dict[str, Any]],
        output_dir: str | Path,
        *,
        multiwoz_references: Optional[List[Dict[str, Any]]] = None,
        bert_evaluator: Optional[BertScoreEvaluator] = None,
        dialogue_runner: Optional[Callable[[Config, Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """
        Run all arms on the same ``experience_seeds``, write CSV/MD/plots under ``output_dir``.

        ``dialogue_runner(cfg, exp)`` defaults to ``DialoguePipeline(cfg, llm).simulate_dialogue(exp, max_turns=...)``.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        refs = multiwoz_references if multiwoz_references is not None else self.load_multiwoz_references()
        if not refs:
            logger.warning("No MultiWOZ references; using placeholder reference for metrics.")
            refs = [
                {
                    "domain": "hotel",
                    "turns": [
                        {"text": "reference user"},
                        {"text": "reference bot confirmation REF-ZZ"},
                    ],
                }
            ]

        per_rows: List[Dict[str, Any]] = []
        all_dialogues_by_arm: Dict[str, List[Dict[str, Any]]] = {s.name: [] for s in self.arm_specs}

        def default_run(cfg: Config, exp: Dict[str, Any]) -> Dict[str, Any]:
            pipe = DialoguePipeline(cfg, self.llm_client)
            return pipe.simulate_dialogue(exp, max_turns=self.max_turns)

        runner = dialogue_runner or default_run

        for spec in self.arm_specs:
            cfg = apply_ablation_config(self.base_config, spec)
            for i, exp in enumerate(experience_seeds):
                logger.info("Ablation arm=%s experience=%s", spec.name, i)
                dial = runner(cfg, exp)
                all_dialogues_by_arm[spec.name].append(dial)
                per_rows.append(
                    {
                        "arm": spec.name,
                        "arm_description": spec.description,
                        "seed_index": i,
                        "dialogue_id": dial.get("dialogue_id"),
                        "domain": dial.get("domain"),
                        "n_turns": len(dial.get("turns", [])),
                    }
                )

        pd.DataFrame(per_rows).to_csv(out / "ablation_per_run.csv", index=False)

        summary_rows: List[Dict[str, Any]] = []
        for spec in self.arm_specs:
            dials = all_dialogues_by_arm[spec.name]
            cfg = apply_ablation_config(self.base_config, spec)
            m = compute_arm_metrics(dials, refs, cfg, bert_evaluator=bert_evaluator)
            summary_rows.append({"arm": spec.name, "description": spec.description, **m})

        summary = pd.DataFrame(summary_rows)
        summary.to_csv(out / "ablation_summary.csv", index=False)

        with open(out / "ablation_comparison.md", "w", encoding="utf-8") as fp:
            fp.write("# Ablation comparison\n\n")
            fp.write(_summary_to_markdown(summary))
            fp.write("\n")

        with open(out / "ablation_summary.json", "w", encoding="utf-8") as fp:
            json.dump(summary_rows, fp, indent=2, default=str)

        try:
            _plot_ablation_bars(summary, out / "ablation_metrics.png")
        except ImportError:
            logger.warning("matplotlib not installed; skipping plots.")
        except Exception as e:
            logger.warning("Plot generation failed: %s", e)

        try:
            from ..visualization.publication_figures import export_ablation_figure_bundle

            pub_paths = export_ablation_figure_bundle(out, summary)
            logger.info("Publication figures (PNG/PDF): %s", pub_paths)
        except Exception as e:
            logger.warning("Publication figure export failed: %s", e)

        return summary
