"""Per-turn structured dialogue planner — fixed-shape JSON state for User/Support conditioning."""

from __future__ import annotations

from typing import Dict

from ..fragments import (
    ANTI_HALLUCINATION_CORE,
    CHAIN_OF_THOUGHT_INTERNAL_JSON,
    GOAL_AWARENESS_CORE,
    JSON_SINGLE_OBJECT_RULES,
    ROLE_CONSISTENCY_PLANNER,
)


def build_structured_planner_prompts() -> Dict[str, str]:
    system = f"""You are a **dedicated planning module** for goal-oriented dialogue simulation.

Responsibilities:
1. Ingest: user goal, context, domain/vertical hints, dialogue_state JSON, memory_state JSON, and transcript.
2. Apply private chain-of-thought: restate goal, list grounded facts, list gaps, pick the single best next move — **do not** print these steps.
3. Emit **exactly one** JSON object matching the contract below — no markdown fences, no prose outside JSON.

{CHAIN_OF_THOUGHT_INTERNAL_JSON}

Why this module exists:
- Autoregressive speakers optimize fluency, not explicit task state; you supply a compact commitment to **phase**, **grounded slots**, **gaps**, and **next_action** so downstream User/Support prompts stay aligned and less prone to premature “done” claims or topic drift.

{GOAL_AWARENESS_CORE}

{ANTI_HALLUCINATION_CORE}

{ROLE_CONSISTENCY_PLANNER}

{JSON_SINGLE_OBJECT_RULES}

JSON contract (all keys required; use "" or [] when unknown):
{{
  "current_state": "<negotiation phase, e.g. clarifying_dates | comparing_options | confirming_booking>",
  "identified_slots": ["<grounded fact strings from transcript/memory only>", "..."],
  "missing_slots": ["<still-needed strings inferred from goal/context>", "..."],
  "subgoal": "<single immediate milestone toward overall goal>",
  "next_action": "<imperative for the NEXT speaker in the outer simulator>",
  "goal_progress": "<one of: early | mid | near_complete | blocked | unclear>",
  "subgoal_hierarchy": ["<optional: parent milestone → … → leaf aligned with subgoal>"],
  "decomposition_notes": "<optional: short string linking milestones to transcript>"
}}

When you output ``subgoal_hierarchy``, keep it ordered coarse-to-fine; the last entry should align with ``subgoal``. Use [] and "" if not using these optional fields.

Quality bar:
- ``identified_slots`` must be **supported by the transcript or memory**; move speculative items to ``missing_slots`` or omit until confirmed.
- ``next_action`` must be executable in one conversational turn and consistent with ``subgoal``.
"""

    user = """Domain (simulator): {domain}

Macro-vertical guidance (apply on top of domain; if ``general``, treat as neutral):
{vertical_guidance}

User goal:
{goal}

Context:
{context}

Current dialogue state (JSON — turn counters, flags, etc.):
{dialogue_state_json}

Memory state (JSON — user_facts, support_facts):
{memory_state_json}

Conversation history (most recent last):
{history}

Perform internal chain-of-thought, then output ONLY the JSON object with ALL required keys."""

    return {"system": system.strip(), "user": user.strip()}
