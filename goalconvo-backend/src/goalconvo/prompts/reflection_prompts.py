"""Shim: reflection prompts from ``prompts.reflection.templates``."""

from .reflection.templates import build_reflection_prompts

REFLECTION_PROMPTS = build_reflection_prompts()
