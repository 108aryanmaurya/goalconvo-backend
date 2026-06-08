#!/usr/bin/env python3
"""
Export publication figures (PNG + PDF, IEEE-style) from ablation outputs or evaluation CSVs.

Examples::

  PYTHONPATH=src python scripts/export_publication_figures.py ablation --dir ./runs/ablation_001
  PYTHONPATH=src python scripts/export_publication_figures.py eval --prefix ./data/results/my_eval --out ./data/results/my_eval_figures
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from goalconvo.visualization.publication_figures import (
    export_ablation_figure_bundle,
    export_evaluation_figure_bundle,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description="GoalConvo publication figures (matplotlib, PNG+PDF)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("ablation", help="From ablation_summary.csv directory")
    pa.add_argument("--dir", type=Path, required=True, help="Ablation output directory (contains ablation_summary.csv)")

    pe = sub.add_parser("eval", help="From research evaluation CSV prefix")
    pe.add_argument(
        "--prefix",
        type=Path,
        required=True,
        help="Path prefix as used by ResearchEvaluationReport.save_csv (e.g. dir/run yields dir/run_per_dialogue.csv)",
    )
    pe.add_argument("--out", type=Path, required=True, help="Directory to write publication/ subfolder into")

    args = p.parse_args()
    if args.cmd == "ablation":
        paths = export_ablation_figure_bundle(args.dir)
        logger.info("Wrote: %s", paths)
        return 0
    if args.cmd == "eval":
        paths = export_evaluation_figure_bundle(args.prefix, args.out)
        logger.info("Wrote %s figure groups", len(paths))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
