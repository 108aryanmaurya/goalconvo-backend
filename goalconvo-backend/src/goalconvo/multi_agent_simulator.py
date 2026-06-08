"""
Backward-compatible entry point for the dialogue simulator.

Implementation lives in ``goalconvo.pipeline.dialogue_pipeline``.
"""

from .pipeline.dialogue_pipeline import DialoguePipeline as DialogueSimulator

__all__ = ["DialogueSimulator"]
