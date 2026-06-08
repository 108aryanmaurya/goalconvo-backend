"""Prompt templates: modular builders + backward-compatible dict exports."""

from .domains import map_domain_to_vertical, vertical_guidance_for_domain
from .planner_prompts import PLANNER_PROMPTS, STRUCTURED_DIALOGUE_PLANNER_PROMPTS
from .reflection_prompts import REFLECTION_PROMPTS
from .support_prompts import SUPPORT_PROMPTS
from .user_prompts import USER_PROMPTS

__all__ = [
    "USER_PROMPTS",
    "SUPPORT_PROMPTS",
    "REFLECTION_PROMPTS",
    "PLANNER_PROMPTS",
    "STRUCTURED_DIALOGUE_PLANNER_PROMPTS",
    "map_domain_to_vertical",
    "vertical_guidance_for_domain",
]
