"""Modular dialogue generation pipeline."""

from .dialogue_pipeline import DialoguePipeline

export_dialogue_json = DialoguePipeline.export_dialogue_json

__all__ = ["DialoguePipeline", "export_dialogue_json"]
