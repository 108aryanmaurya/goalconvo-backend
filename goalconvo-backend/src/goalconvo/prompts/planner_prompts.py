"""Shim: experience + structured planner prompts composed from modular ``prompts.planner``."""

from .planner.experience import build_experience_planner_prompts
from .planner.structured import build_structured_planner_prompts

PLANNER_PROMPTS = build_experience_planner_prompts()
STRUCTURED_DIALOGUE_PLANNER_PROMPTS = build_structured_planner_prompts()
