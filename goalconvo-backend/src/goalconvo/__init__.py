"""
GoalConvo: A framework for generating goal-oriented dialogue data via multi-agent simulation.

This package provides tools for:
- Multi-agent dialogue simulation using Mistral-7B
- Few-shot experience generation with dynamic hub
- Quality filtering and evaluation
- Comprehensive metrics computation
"""

__version__ = "0.1.0"
__author__ = "GoalConvo Team"

from .llm_client import LLMClient
from .config import Config
from .experience_generator import ExperienceGenerator
from .pipeline import DialoguePipeline, export_dialogue_json
from .evaluation import ResearchEvaluationReport, ResearchEvaluationSuite
from .multi_agent_simulator import DialogueSimulator
from .quality_judge import QualityJudge
from .dataset_store import DatasetStore
from .evaluator import Evaluator
from .experiments import AblationStudyRunner, ABLATION_ARM_SPECS
from .experiments.tracking import ExperimentRun, create_experiment_run
from .analysis import FailureAnalyzer, FailureCorpusAnalyzer, export_failure_bundle

__all__ = [
    "LLMClient",
    "Config", 
    "ExperienceGenerator",
    "DialoguePipeline",
    "export_dialogue_json",
    "ResearchEvaluationSuite",
    "ResearchEvaluationReport",
    "DialogueSimulator",
    "QualityJudge",
    "DatasetStore",
    "Evaluator",
    "AblationStudyRunner",
    "ABLATION_ARM_SPECS",
    "ExperimentRun",
    "create_experiment_run",
    "FailureAnalyzer",
    "FailureCorpusAnalyzer",
    "export_failure_bundle",
]
