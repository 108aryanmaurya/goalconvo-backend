"""Experiment drivers (ablation studies, reproducibility tracking).

Ablation symbols are lazy-exported to avoid an import cycle:
``DialoguePipeline`` imports ``merge_repro_metadata`` from ``.tracking``, while
``ablation`` imports ``DialoguePipeline``.
"""

from __future__ import annotations

from typing import Any

from .tracking import (
    ExperimentRun,
    create_experiment_run,
    generation_hyperparameter_snapshot,
    merge_repro_metadata,
    prompt_fingerprints,
)

__all__ = [
    "ABLATION_ARM_SPECS",
    "AblationStudyRunner",
    "apply_ablation_config",
    "compute_arm_metrics",
    "ExperimentRun",
    "create_experiment_run",
    "generation_hyperparameter_snapshot",
    "merge_repro_metadata",
    "prompt_fingerprints",
]


def __getattr__(name: str) -> Any:
    if name in (
        "ABLATION_ARM_SPECS",
        "AblationStudyRunner",
        "apply_ablation_config",
        "compute_arm_metrics",
    ):
        from . import ablation as _ab

        return getattr(_ab, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
