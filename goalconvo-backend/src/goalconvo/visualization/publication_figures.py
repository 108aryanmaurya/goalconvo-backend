"""
Publication-quality GoalConvo experiment figures (matplotlib only).

IEEE-inspired defaults: serif fonts, restrained grid, no top/right spines, vector PDF.
Each figure is saved as ``<stem>.png`` (raster) and ``<stem>.pdf`` (vector) with an
automatic caption paragraph under the axes.
"""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- Optional heavy imports (lazy in functions too for CLI safety) ---
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, plt


IEEE_RC: Dict[str, Any] = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman", "serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.85,
    "axes.edgecolor": "0.15",
    "axes.labelcolor": "0.05",
    "xtick.color": "0.05",
    "ytick.color": "0.05",
    "grid.linewidth": 0.45,
    "grid.linestyle": ":",
    "lines.linewidth": 1.35,
    "lines.markersize": 4.5,
    "axes.grid": True,
    "grid.alpha": 0.45,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.75",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_ieee_matplotlib_style() -> None:
    """Apply matplotlib rcParams for IEEE-style figures (call once per process)."""
    _, plt = _plt()
    plt.rcParams.update(IEEE_RC)


def save_figure_png_pdf(fig: Any, output_stem: Path, caption: str) -> Dict[str, str]:
    """
    Save ``fig`` to ``output_stem.png`` and ``output_stem.pdf``.

    ``caption`` is rendered below the subplot area (figure-level).
    """
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    wrapped = textwrap.fill(caption.strip(), width=108) if caption else ""

    fig.subplots_adjust(bottom=0.26 if wrapped else 0.12)
    if wrapped:
        fig.text(
            0.5,
            0.02,
            wrapped,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="0.15",
            linespacing=1.25,
        )

    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    paths = {"png": str(png), "pdf": str(pdf)}
    return paths


def _close(fig: Any) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_ablation_study_figure(
    summary: pd.DataFrame,
    output_stem: Path,
    *,
    title: str = "Ablation study: corpus metrics by arm",
) -> Dict[str, str]:
    """
    Grouped bar chart of ablation_summary columns (bertscore, distinct-1, coherence, goal scores).
    """
    apply_ieee_matplotlib_style()
    _, plt = _plt()

    metrics = [
        ("bertscore_mean", "BERTScore F1 (mean)"),
        ("distinct1_mean", "Distinct-1"),
        ("coherence_mean", "Coherence (adj.)"),
        ("goal_completion_score_mean", "Goal completion score"),
        ("goal_completed_rate", "Goal completed rate"),
    ]
    present = [(c, lab) for c, lab in metrics if c in summary.columns]
    if not present:
        raise ValueError("summary DataFrame missing expected metric columns")

    nmet = len(present)
    ncols = min(3, nmet)
    nrows = int(np.ceil(nmet / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols + 0.5, 2.6 * nrows + 0.9), squeeze=False)
    axes_flat = axes.ravel()
    arms = summary["arm"].astype(str).tolist()
    x = np.arange(len(arms))
    colors = plt.cm.Greys(np.linspace(0.35, 0.75, len(arms))) if len(arms) > 1 else ["0.45"]

    for ax, (col, ylab) in zip(axes_flat, present):
        vals = summary[col].astype(float).values
        ax.bar(x, vals, color=colors[: len(arms)], edgecolor="0.2", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(arms, rotation=22, ha="right")
        ax.set_ylabel(ylab)
        ax.set_xlim(-0.55, len(arms) - 0.45)

    for j in range(len(present), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, y=1.02, fontsize=11)
    n = int(summary["n_dialogues"].max()) if "n_dialogues" in summary.columns else 0
    caption = (
        f"Ablation arms compared on the same experience seeds against MultiWOZ references. "
        f"Bars show corpus means (N dialogues per arm = {n} when uniform). "
        f"Higher BERTScore and goal metrics indicate stronger reference alignment and completion heuristics."
    )
    paths = save_figure_png_pdf(fig, output_stem, caption)
    _close(fig)
    return paths


def plot_domain_wise_performance(
    per_dialogue: pd.DataFrame,
    output_stem: Path,
    *,
    title: str = "Domain-wise performance",
) -> Dict[str, str]:
    """Mean ± SEM bars per domain for BERTScore, coherence, and goal-completion rate."""
    apply_ieee_matplotlib_style()
    _, plt = _plt()

    if "domain" not in per_dialogue.columns:
        raise ValueError("per_dialogue must include a 'domain' column")

    df = per_dialogue.copy()
    df["domain"] = df["domain"].fillna("unknown").astype(str)
    domains = sorted(df["domain"].unique())
    metrics = [
        ("bertscore_f1", "BERTScore F1"),
        ("coherence_adjacent", "Coherence"),
        ("goal_completed", "Goal completed (rate)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9), sharex=False)
    x = np.arange(len(domains))
    for ax, (col, ylab) in zip(axes, metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        means: List[float] = []
        sems: List[float] = []
        for d in domains:
            sub = pd.to_numeric(df.loc[df["domain"] == d, col], errors="coerce").dropna()
            if len(sub) == 0:
                means.append(float("nan"))
                sems.append(0.0)
            else:
                means.append(float(sub.mean()))
                sems.append(float(sub.std(ddof=1) / np.sqrt(len(sub))) if len(sub) > 1 else 0.0)
        ax.bar(x, means, yerr=sems, color="0.55", edgecolor="0.2", linewidth=0.55, capsize=3, error_kw={"linewidth": 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels(domains, rotation=28, ha="right")
        ax.set_ylabel(ylab)
        ax.set_xlabel("Domain")

    fig.suptitle(title, y=1.05, fontsize=11)
    n = len(df)
    caption = (
        f"Per-domain means with standard error of the mean (SEM) over n={n} dialogues. "
        f"BERTScore and coherence are dialogue-level; goal completed is the fraction passing the rule-based goal check."
    )
    paths = save_figure_png_pdf(fig, output_stem, caption)
    _close(fig)
    return paths


def plot_coherence_distribution(
    per_dialogue: pd.DataFrame,
    output_stem: Path,
    *,
    title: str = "Adjacent-turn coherence distribution",
) -> Dict[str, str]:
    """Histogram of coherence_adjacent with corpus mean vertical line."""
    apply_ieee_matplotlib_style()
    _, plt = _plt()

    col = "coherence_adjacent"
    if col not in per_dialogue.columns:
        raise ValueError(f"per_dialogue must include '{col}'")

    s = pd.to_numeric(per_dialogue[col], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.hist(s, bins=min(24, max(8, len(s) // 3)), color="0.72", edgecolor="0.25", linewidth=0.6, density=True)
    mu = float(s.mean())
    ax.axvline(mu, color="0.1", linestyle="--", linewidth=1.2, label=f"Mean = {mu:.3f}")
    ax.set_xlabel("Coherence (mean adjacent-turn similarity)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", framealpha=0.95)
    fig.suptitle(title, y=1.02, fontsize=11)
    caption = (
        f"Distribution of lexical coherence (n={len(s)} dialogues). "
        f"Values near 1 indicate high word overlap between consecutive turns; near 0 suggests topic or style shifts."
    )
    paths = save_figure_png_pdf(fig, output_stem, caption)
    _close(fig)
    return paths


def plot_coherence_comparison_overlay(
    series_a: np.ndarray,
    series_b: np.ndarray,
    label_a: str,
    label_b: str,
    output_stem: Path,
    *,
    title: str = "Coherence comparison (overlaid densities)",
) -> Dict[str, str]:
    """Overlaid normalized histograms for two corpora (e.g. two experimental conditions)."""
    apply_ieee_matplotlib_style()
    _, plt = _plt()

    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    bins = min(28, max(10, int(min(len(a), len(b)) // 2) + 6))
    ax.hist(
        a,
        bins=bins,
        density=True,
        alpha=0.55,
        color="0.25",
        edgecolor="0.15",
        linewidth=0.45,
        label=f"{label_a} (n={len(a)})",
    )
    ax.hist(
        b,
        bins=bins,
        density=True,
        alpha=0.45,
        color="0.65",
        edgecolor="0.2",
        linewidth=0.45,
        label=f"{label_b} (n={len(b)})",
    )
    ax.set_xlabel("Coherence (adjacent-turn similarity)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", framealpha=0.95)
    fig.suptitle(title, y=1.02, fontsize=11)
    caption = (
        "Overlaid density estimates for adjacent-turn lexical coherence under two conditions. "
        "Greater mass at higher coherence often indicates more stable topical continuity turn-to-turn."
    )
    paths = save_figure_png_pdf(fig, output_stem, caption)
    _close(fig)
    return paths


def plot_goal_completion_figure(
    per_dialogue: pd.DataFrame,
    aggregate: Optional[Dict[str, Any]],
    output_stem: Path,
    *,
    title: str = "Goal completion",
) -> Dict[str, str]:
    """Bar for completion rate with Wilson interval when aggregate provides ci bounds; else binomial SE."""
    apply_ieee_matplotlib_style()
    _, plt = _plt()

    if "goal_completed" not in per_dialogue.columns:
        raise ValueError("per_dialogue must include 'goal_completed'")

    g = pd.to_numeric(per_dialogue["goal_completed"], errors="coerce").fillna(0)
    n = len(g)
    p = float(g.mean()) if n else 0.0
    se = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
    ci_lo, ci_hi = p - 1.96 * se, p + 1.96 * se
    if aggregate and isinstance(aggregate.get("goal_completion"), dict):
        gc = aggregate["goal_completion"]
        wlo, whi = gc.get("wilson_ci_low"), gc.get("wilson_ci_high")
        if wlo is not None and whi is not None and not (isinstance(wlo, float) and np.isnan(wlo)):
            ci_lo, ci_hi = float(wlo), float(whi)

    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.bar([0], [p], width=0.55, color="0.5", edgecolor="0.15", linewidth=0.7, yerr=[[p - ci_lo], [ci_hi - p]], capsize=4)
    ax.set_xticks([0])
    ax.set_xticklabels(["Candidate corpus"])
    ax.set_ylabel("Goal completion rate")
    ax.set_ylim(0, min(1.05, max(0.15, ci_hi + 0.12)))
    fig.suptitle(title, y=1.02, fontsize=11)
    caption = (
        f"Fraction of dialogues marked goal-completed (n={n}). "
        f"Error bars show approximate 95% interval (Wilson if available from aggregate, else normal approximation)."
    )
    paths = save_figure_png_pdf(fig, output_stem, caption)
    _close(fig)
    return paths


def plot_metric_comparison_from_aggregates(
    series: Sequence[Tuple[str, Dict[str, Any]]],
    output_stem: Path,
    *,
    title: str = "Metric comparison (corpus means with 95% CI)",
) -> Dict[str, str]:
    """
    ``series`` is a list of (label, aggregate_dict) where aggregate_dict matches
    :class:`ResearchEvaluationReport` ``aggregate`` field (``metrics`` + optional goal_completion).
    """
    apply_ieee_matplotlib_style()
    _, plt = _plt()

    metric_keys = ["bertscore_f1", "distinct_1", "coherence_adjacent", "semantic_cosine"]
    labels = [s[0] for s in series]
    ncond = len(labels)
    nmet = len(metric_keys)
    x = np.arange(nmet)
    width = min(0.22, 0.8 / max(1, ncond))

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for i, (lab, agg) in enumerate(series):
        metrics = (agg or {}).get("metrics") or {}
        means: List[float] = []
        lo: List[float] = []
        hi: List[float] = []
        for m in metric_keys:
            st = metrics.get(m) or {}
            mu = st.get("mean")
            if mu is None:
                means.append(float("nan"))
                lo.append(float("nan"))
                hi.append(float("nan"))
            else:
                means.append(float(mu))
                lo.append(float(st.get("ci_low", mu)))
                hi.append(float(st.get("ci_high", mu)))
        offset = (i - (ncond - 1) / 2) * width
        pos = x + offset
        lo_a = np.array(lo, dtype=float)
        hi_a = np.array(hi, dtype=float)
        mu_a = np.array(means, dtype=float)
        yerr = np.vstack([np.nan_to_num(mu_a - lo_a, nan=0.0), np.nan_to_num(hi_a - mu_a, nan=0.0)])
        ax.bar(pos, means, width=width * 0.92, label=lab, yerr=yerr, capsize=2.2, edgecolor="0.2", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metric_keys], fontsize=7.5)
    ax.set_ylabel("Score")
    ax.legend(loc="upper right", ncol=min(2, ncond))
    fig.suptitle(title, y=1.02, fontsize=11)
    caption = (
        "Corpus-level means with 95% confidence intervals from bootstrap or analytic summaries in "
        "``ResearchEvaluationReport.aggregate['metrics']``. Compare runs or candidate pools side-by-side."
    )
    paths = save_figure_png_pdf(fig, output_stem, caption)
    _close(fig)
    return paths


def export_ablation_figure_bundle(ablation_output_dir: Union[str, Path], summary: Optional[pd.DataFrame] = None) -> Dict[str, List[str]]:
    """
    Write publication figures under ``<ablation_output_dir>/publication/``.

    If ``summary`` is None, loads ``ablation_summary.csv``.
    Returns mapping figure key -> [png path, pdf path].
    """
    root = Path(ablation_output_dir)
    pub = root / "publication"
    pub.mkdir(parents=True, exist_ok=True)
    if summary is None:
        csv_path = root / "ablation_summary.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {csv_path}; pass summary DataFrame explicitly.")
        summary = pd.read_csv(csv_path)

    out: Dict[str, List[str]] = {}
    p = plot_ablation_study_figure(summary, pub / "fig_ablation_study_metrics")
    out["ablation_study"] = [p["png"], p["pdf"]]
    return out


def export_evaluation_figure_bundle(
    report_or_prefix: Union[Any, str, Path],
    output_dir: Union[str, Path],
    *,
    second_report: Optional[Any] = None,
    second_label: str = "Condition B",
    first_label: str = "Condition A",
) -> Dict[str, List[str]]:
    """
    Build figures from a :class:`goalconvo.evaluation.research_evaluator.ResearchEvaluationReport`
    or from CSV prefix ``path`` such that ``path_per_dialogue.csv`` and ``path_summary_long.csv`` exist.

    Writes to ``output_dir/publication/``.
    """
    from ..evaluation.research_evaluator import ResearchEvaluationReport

    out_dir = Path(output_dir) / "publication"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, List[str]] = {}

    if isinstance(report_or_prefix, (str, Path)):
        prefix = Path(report_or_prefix)
        per_path = prefix.parent / f"{prefix.name}_per_dialogue.csv"
        if not per_path.exists():
            per_path = Path(str(report_or_prefix) + "_per_dialogue.csv")
        df = pd.read_csv(per_path)
        agg_path = prefix.parent / f"{prefix.name}_summary_long.csv"
        aggregate = None
        if agg_path.exists():
            sl = pd.read_csv(agg_path)
            aggregate = _aggregate_from_summary_long(sl)
    else:
        rep = report_or_prefix
        if not isinstance(rep, ResearchEvaluationReport):
            raise TypeError("report_or_prefix must be ResearchEvaluationReport or CSV prefix path")
        df = rep.to_dataframe()
        aggregate = rep.aggregate

    p1 = plot_domain_wise_performance(df, out_dir / "fig_domain_wise_performance")
    results["domain_wise"] = [p1["png"], p1["pdf"]]
    p2 = plot_coherence_distribution(df, out_dir / "fig_coherence_distribution")
    results["coherence"] = [p2["png"], p2["pdf"]]
    p3 = plot_goal_completion_figure(df, aggregate, out_dir / "fig_goal_completion")
    results["goal_completion"] = [p3["png"], p3["pdf"]]

    if second_report is not None and isinstance(second_report, ResearchEvaluationReport):
        df2 = second_report.to_dataframe()
        if "coherence_adjacent" in df.columns and "coherence_adjacent" in df2.columns:
            p5 = plot_coherence_comparison_overlay(
                pd.to_numeric(df["coherence_adjacent"], errors="coerce").values,
                pd.to_numeric(df2["coherence_adjacent"], errors="coerce").values,
                first_label,
                second_label,
                out_dir / "fig_coherence_comparison",
            )
            results["coherence_comparison"] = [p5["png"], p5["pdf"]]
        if aggregate is not None:
            p4 = plot_metric_comparison_from_aggregates(
                [(first_label, aggregate), (second_label, second_report.aggregate)],
                out_dir / "fig_metric_comparison",
            )
            results["metric_comparison"] = [p4["png"], p4["pdf"]]
    elif aggregate is not None:
        p4 = plot_metric_comparison_from_aggregates([(first_label, aggregate)], out_dir / "fig_metric_comparison_single")
        results["metric_comparison"] = [p4["png"], p4["pdf"]]

    # captions index
    with open(out_dir / "figure_manifest.json", "w", encoding="utf-8") as fp:
        json.dump({k: {"png": v[0], "pdf": v[1]} for k, v in results.items()}, fp, indent=2)
    return results


def _aggregate_from_summary_long(summary_long: pd.DataFrame) -> Dict[str, Any]:
    """Reconstruct a minimal aggregate dict for goal bar / metric comparison from summary CSV."""
    metrics: Dict[str, Any] = {}
    goal_completion: Optional[Dict[str, Any]] = None
    for _, row in summary_long.iterrows():
        m = row.get("metric")
        if pd.isna(m):
            continue
        if str(m) == "goal_completion_rate":
            goal_completion = {
                "rate": float(row.get("mean", 0) or 0),
                "wilson_ci_low": float(row["ci_low"]) if pd.notna(row.get("ci_low")) else None,
                "wilson_ci_high": float(row["ci_high"]) if pd.notna(row.get("ci_high")) else None,
                "n": int(row.get("n") or 0),
            }
        else:
            metrics[str(m)] = {
                "mean": row.get("mean"),
                "ci_low": row.get("ci_low"),
                "ci_high": row.get("ci_high"),
                "std": row.get("std"),
                "n": row.get("n"),
            }
    out: Dict[str, Any] = {"metrics": metrics}
    if goal_completion:
        out["goal_completion"] = goal_completion
    return out
