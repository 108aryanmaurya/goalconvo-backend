"""User-role dialogue agent."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..agent_capabilities import build_planning_instruction, parse_plan_and_reply
from ..prompts.domains import vertical_guidance_for_domain
from ..prompts.user_prompts import USER_PROMPTS
from ..pipeline.dialogue_helpers import (
    clean_response,
    format_structured_goal,
    last_k_turns,
    progress_hint_for_user,
    truncate_prompt,
    vary_user_response,
)
from ..utils import format_conversation_history
from .base import AgentGenerationResult, BaseDialogueAgent

logger = logging.getLogger(__name__)


class UserAgent(BaseDialogueAgent):
    def generate(
        self,
        goal: str,
        context: str,
        user_persona: str,
        history: List[Dict[str, str]],
        domain: str = "general",
        experience_data: Optional[Dict[str, Any]] = None,
        user_memory: Optional[List[str]] = None,
        support_memory: Optional[List[str]] = None,
        memory_section: str = "",
        structured_planning_block: Optional[str] = None,
        reflection_repair_hint: Optional[str] = None,
        **_: Any,
    ) -> AgentGenerationResult:
        k = getattr(self.config, "prompt_last_k_turns", 6)
        recent = last_k_turns(history, k)
        history_text = format_conversation_history(recent)
        progress_hint = progress_hint_for_user(goal)
        structured_goal = format_structured_goal(experience_data)
        persona_traits = (experience_data or {}).get("user_persona_traits", "") or ""
        if persona_traits:
            persona_traits = f"Communication style: {persona_traits}"
        if structured_planning_block:
            planning_instruction = structured_planning_block
        elif getattr(self.config, "agent_planning_enabled", True):
            planning_instruction = build_planning_instruction()
        else:
            planning_instruction = ""
        vg = (vertical_guidance_for_domain(domain) or "").strip()
        if not vg:
            vg = "General — follow domain and stated goal only."

        prompt = USER_PROMPTS["user"].format(
            domain=domain,
            vertical_guidance=vg,
            goal=goal,
            context=context,
            user_persona=user_persona,
            structured_goal=structured_goal,
            persona_traits=persona_traits,
            memory_section=memory_section,
            history=history_text,
            progress_hint=progress_hint,
            planning_instruction=planning_instruction,
        )
        full_prompt = f"{USER_PROMPTS['system']}\n\n{prompt}"
        repair = (reflection_repair_hint or "").strip()
        if repair:
            full_prompt += (
                "\n\n## Reflection repair (required)\n"
                "Your previous reply failed automated verification. Revise your next reply to fix this "
                "(stay in character as the user; one short message):\n"
                f"{repair}\n"
            )
        max_words = getattr(self.config, "prompt_max_words", 1000)
        instr_words = getattr(self.config, "prompt_instruction_words", 250)
        truncated = truncate_prompt(full_prompt, max_words=max_words, instruction_words=instr_words)
        max_tokens_user = getattr(self.config, "max_tokens_user_turn", 60)
        if getattr(self.config, "agent_planning_enabled", True):
            max_tokens_user = max(max_tokens_user, getattr(self.config, "max_tokens_planning", 180))
        response = self.llm_client.generate_completion(
            truncated,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=max_tokens_user,
        )
        plan_note = None
        if getattr(self.config, "agent_planning_enabled", True):
            if structured_planning_block:
                pass
            else:
                plan_note, response = parse_plan_and_reply(response)
        cleaned = clean_response(response, role="User").strip()
        if history:
            for prev_turn in history:
                if prev_turn.get("role") != "User":
                    continue
                prev_text = (prev_turn.get("text") or "").strip()
                if prev_text and cleaned.lower() == prev_text.lower():
                    logger.warning("Prevented exact repetition for User: '%s' matches previous turn", cleaned)
                    cleaned = vary_user_response(cleaned, goal, domain)
                    break
        if not cleaned:
            cleaned = "I need help with this."
        meta: Dict[str, Any] = {}
        if plan_note:
            meta["plan"] = plan_note
        return AgentGenerationResult(text=cleaned, metadata=meta)

    def update_memory(self, memory, recent_turns: List[Dict[str, Any]], **ctx: Any) -> None:
        return None
