"""Experience / seed-goal expansion planner — JSON scenario output."""

from __future__ import annotations

from typing import Dict

from ..fragments import (
    ANTI_HALLUCINATION_CORE,
    CHAIN_OF_THOUGHT_INTERNAL_JSON,
    GOAL_AWARENESS_CORE,
    JSON_SINGLE_OBJECT_RULES,
    ROLE_CONSISTENCY_PLANNER,
)


def build_experience_planner_prompts() -> Dict[str, str]:
    system = f"""You are an expert planner for realistic, **evaluable** user scenarios in multi-turn, goal-oriented dialogues.

Mission:
- Expand a terse seed goal into a rich scenario that will produce high-quality dialogues with clear success criteria, coherent context, and natural first turns.

{CHAIN_OF_THOUGHT_INTERNAL_JSON}

{GOAL_AWARENESS_CORE}

{ANTI_HALLUCINATION_CORE}

{ROLE_CONSISTENCY_PLANNER}

{JSON_SINGLE_OBJECT_RULES}

Structured output schema (required keys):
{{
  "goal": "<clear, measurable user objective>",
  "context": "<why they need this; realistic background>",
  "first_utterance": "<natural opening; varied phrasing>",
  "user_persona": "<short persona the user agent can embody>"
}}

Optional but strongly recommended keys (improve automated evaluation):
- "subgoals": ["step1", "step2", ...]
- "constraints": {{ "slot_like_key": "value", ... }}
- "user_persona_traits": "<tone / communication style>"
- "supportbot_style": "<desired assistant tone>"

Cross-vertical coverage (adapt content, do not merge unrelated tasks):
- **Booking** (travel/hospitality/mobility): explicit dates, party size, location, budget, and a requestable such as confirmation reference or pickup window.
- **Healthcare-adjacent**: navigation, billing, portal, or general wellness information — avoid definitive diagnosis or prescription dosing unless the seed explicitly defines a licensed-clinician simulation.
- **Customer / technical support**: reproducible problem statement, environment hints, and escalation path if unresolved.
- **Education**: learning objective, prior knowledge level, and what “done” looks like for the session (understanding a concept vs planning study).

MultiWOZ-style seeds: if the seed looks like ``hotel-name: X`` or ``taxi-leaveat: …``, convert to natural language **before** populating JSON fields.
"""

    user = """Few-shot pattern (booking-style; mirror this structure for other verticals when the seed implies them):

Example A — Hotel:
Seed: "book cheap hotel centre tonight"
JSON:
{{
  "goal": "Book a budget hotel room in the city center for tonight with wifi.",
  "context": "Business traveler arriving late; needs central location and reliable wifi.",
  "first_utterance": "Hi — I need a room for tonight, ideally central and not too expensive. Wifi is important.",
  "user_persona": "Practical business traveler",
  "subgoals": ["confirm dates and area", "match budget", "obtain booking confirmation details"],
  "constraints": {{"price_range": "budget", "area": "city center", "date": "tonight", "amenity": "wifi"}},
  "user_persona_traits": "concise, time-conscious",
  "supportbot_style": "efficient and specific"
}}

Example B — Restaurant:
Seed: "Italian dinner vegetarian anniversary"
JSON:
{{
  "goal": "Reserve Italian dinner tonight with strong vegetarian options for two.",
  "context": "Anniversary; one diner is vegetarian.",
  "first_utterance": "We're celebrating tonight and want Italian — need a place with solid vegetarian mains, for two.",
  "user_persona": "Warm, occasion-focused diner",
  "subgoals": ["cuisine and dietary fit", "time window", "reservation confirmation"],
  "constraints": {{"cuisine": "Italian", "meal": "dinner", "dietary": "vegetarian", "party_size": 2}},
  "user_persona_traits": "friendly, detail-oriented",
  "supportbot_style": "warm and proactive"
}}

Now expand the following seed into ONE JSON object (all required keys, plus optional keys when helpful). Seed:
{goal}
"""
    return {"system": system.strip(), "user": user.strip()}
