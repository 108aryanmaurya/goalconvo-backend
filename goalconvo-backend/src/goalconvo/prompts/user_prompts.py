"""Shim: user-agent prompts from ``prompts.user.templates``."""

from .user.templates import build_user_prompts

USER_PROMPTS = build_user_prompts()
