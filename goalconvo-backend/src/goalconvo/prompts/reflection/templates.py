"""Reflection / critique prompts — LLM-as-judge with strict JSON for utterance critique."""

from __future__ import annotations

from typing import Dict

from ..fragments import (
    ANTI_HALLUCINATION_CORE,
    CHAIN_OF_THOUGHT_INTERNAL_JSON,
    GOAL_AWARENESS_CORE,
    JSON_SINGLE_OBJECT_RULES,
    ROLE_CONSISTENCY_REFLECTION,
)


def build_reflection_prompts() -> Dict[str, str]:
    goal_check = (
        """Analyze whether the user's goal has been **fully** achieved in this conversation.

User Goal:
{goal}

Conversation:
{history}

"""
        + GOAL_AWARENESS_CORE
        + """

Respond **YES** only if ALL hold:
1. **Completion**: Constraints implied by the goal are addressed; critical requestables appear (confirmation, time, place, reference token, explicit resolution, or defensible handoff).
2. **User closure**: The user's **last** turn shows clear satisfaction (thanks, perfect, all set, …) aligned with assistant evidence.
3. **No dangling deferrals**: No unresolved "I'll check" without a follow-up resolution in-thread.
4. **Grounding**: Assistant did not hallucinate completion; stated facts fit the transcript.

Respond **NO** if still negotiating, missing concrete evidence, user not satisfied, or completion overstated.

Output: exactly **YES** or **NO** (uppercase), no other tokens.
"""
    ).strip()

    response_critique = (
        "You verify one candidate utterance in a goal-oriented dialogue before it is committed.\n\n"
        + ROLE_CONSISTENCY_REFLECTION
        + "\n\n"
        + CHAIN_OF_THOUGHT_INTERNAL_JSON
        + "\n\n"
        + GOAL_AWARENESS_CORE
        + "\n\n"
        + ANTI_HALLUCINATION_CORE
        + "\n\n"
        + JSON_SINGLE_OBJECT_RULES
        + """

Speaker role: {role}
User goal: {goal}
Domain: {domain}

Dialogue history (most recent last):
{history}

Structured memory (authoritative; contradictions are severe faults):
{memory_json}

Candidate utterance from {role}:
---
{generated_response}
---

Axes (each must influence your score):
1. goal_alignment — advances/respects goal & constraints.
2. hallucination — unsupported specifics vs history/memory/tools.
3. repetition — near-duplicate of recent same-role turns without new value.
4. coherence — sensible, fluent follow-up to the last counterpart turn.
5. task_progress — concrete help (support) or realistic next user step.
6. memory_contradiction — conflicts with memory JSON.

Output ONLY valid JSON (no markdown) exactly:
{{"accepted": true, "issues": [], "score": 5, "reason": "one concise sentence"}}

issues: short labels among goal_alignment, hallucination, repetition, coherence, task_progress, memory_contradiction.

Score rubric 0–5: 5 excellent; 4 minor nit; 3 one noticeable problem; 2 serious failure; 0–1 unusable.
Set accepted true only if score >= 4 and no severe hallucination or memory_contradiction.
"""
    ).strip()

    return {"goal_check": goal_check, "response_critique": response_critique}
