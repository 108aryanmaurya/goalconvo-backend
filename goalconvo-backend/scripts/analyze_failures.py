#!/usr/bin/env python3
"""
Run automated failure analysis on a JSON list or JSONL of dialogues.

Writes failure_report.json, failure_summary.csv, failure_report.md, and
per-dialogue JSON under failed_dialogues/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).parent.parent / "src"))

from goalconvo.analysis import FailureAnalyzer, FailureCorpusAnalyzer, export_failure_bundle
from goalconvo.config import Config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_dialogues(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        out: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
        return out
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "dialogues" in data:
        return list(data["dialogues"])
    raise ValueError("JSON must be a list of dialogues or an object with key 'dialogues'")


def main() -> None:
    p = argparse.ArgumentParser(description="GoalConvo failure analysis export")
    p.add_argument("input_path", type=Path, help="Path to .json or .jsonl dialogues")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory for exports")
    p.add_argument("--limit", type=int, default=None, help="Max dialogues to analyze")
    p.add_argument("--examples", type=int, default=5, help="Max examples per category in aggregate")
    args = p.parse_args()

    dialogues = _load_dialogues(args.input_path)
    if args.limit is not None:
        dialogues = dialogues[: args.limit]

    cfg = Config()
    analyzer = FailureAnalyzer(cfg)
    corpus = FailureCorpusAnalyzer(cfg)
    aggregate = corpus.analyze(dialogues, max_examples_per_category=args.examples)
    per = [analyzer.analyze(d) for d in dialogues]

    paths = export_failure_bundle(
        args.output_dir,
        dialogues,
        aggregate,
        per_dialogue_results=per,
        analyzer=analyzer,
    )
    logger.info("Wrote %s", paths)


if __name__ == "__main__":
    main()
