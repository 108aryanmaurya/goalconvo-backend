"""Shim: support-bot prompts from ``prompts.support.templates``."""

from .support.templates import build_support_prompts

SUPPORT_PROMPTS = build_support_prompts()
