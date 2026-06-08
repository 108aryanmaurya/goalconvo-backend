#!/usr/bin/env python3
"""
Automatic GoalConvo ablation study runner.

Runs preset arms (full, no planner, no memory, no reflection, single-agent),
writes CSV summaries, Markdown comparison table, and matplotlib plots.

Example:
  PYTHONPATH=src python scripts/run_ablation_study.py \\
    --output-dir ./runs/ablation_001 \\
    --dialogues-per-arm 2 \\
    --max-turns 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from goalconvo.config import Config
from goalconvo.experiments.ablation import AblationStudyRunner
from goalconvo.llm_client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _default_experience_seeds(n: int) -> list:
    seeds = []
    for i in range(n):
        seeds.append(
            {
                "goal": f"Book a hotel room for tonight in the city centre ({i})",
                "context": "Business trip, prefer central location.",
                "domain": "hotel",
                "user_persona": "Business traveler",
                "first_utterance": f"Hi, I need a hotel room for tonight, run {i}.",
            }
        )
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="GoalConvo ablation study runner")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for CSV/MD/PNG outputs")
    parser.add_argument("--dialogues-per-arm", type=int, default=2, help="Number of experience seeds per arm")
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--multiwoz-limit", type=int, default=40)
    parser.add_argument("--seeds-json", type=str, default="", help="Optional JSON file: list of experience dicts")
    args = parser.parse_args()

    config = Config()
    llm = LLMClient(config)

    if args.seeds_json:
        seeds = json.loads(Path(args.seeds_json).read_text(encoding="utf-8"))
        if not isinstance(seeds, list):
            raise SystemExit("--seeds-json must contain a JSON list")
    else:
        seeds = _default_experience_seeds(args.dialogues_per_arm)

    runner = AblationStudyRunner(
        config,
        llm,
        max_turns=args.max_turns,
        multiwoz_reference_limit=args.multiwoz_limit,
    )
    summary = runner.run(seeds, args.output_dir)
    logger.info("Wrote results under %s", args.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
