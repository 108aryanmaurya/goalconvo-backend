"""Support-role dialogue agent."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..agent_capabilities import build_planning_instruction, parse_plan_and_reply
from ..prompts.domains import vertical_guidance_for_domain
from ..prompts.support_prompts import SUPPORT_PROMPTS
from ..pipeline.dialogue_helpers import (
    clean_response,
    format_structured_goal,
    get_domain_grounding,
    last_k_turns,
    supportbot_tool_outputs,
    truncate_prompt,
    vary_supportbot_response,
)
from ..utils import format_conversation_history
from .base import AgentGenerationResult, BaseDialogueAgent

logger = logging.getLogger(__name__)


class SupportAgent(BaseDialogueAgent):
    def generate(
        self,
        goal: str,
        context: str,
        history: List[Dict[str, str]],
        domain: str = "general",
        experience_data: Optional[Dict[str, Any]] = None,
        memory_section: str = "",
        structured_planning_block: Optional[str] = None,
        reflection_repair_hint: Optional[str] = None,
        **_: Any,
    ) -> AgentGenerationResult:
        k = getattr(self.config, "prompt_last_k_turns", 6)
        recent = last_k_turns(history, k)
        history_text = format_conversation_history(recent)
        structured_goal = format_structured_goal(experience_data)
        supportbot_style = (experience_data or {}).get("supportbot_style", "") or ""
        if supportbot_style:
            supportbot_style = f"Style: {supportbot_style}"
        domain_grounding = get_domain_grounding(domain)
        tool_outputs, tools_used = supportbot_tool_outputs(
            history, domain, goal, context, getattr(self.config, "agent_tools_enabled", True)
        )
        if structured_planning_block:
            planning_instruction = structured_planning_block
        elif getattr(self.config, "agent_planning_enabled", True):
            planning_instruction = build_planning_instruction()
        else:
            planning_instruction = ""
        vg = (vertical_guidance_for_domain(domain) or "").strip()
        if not vg:
            vg = "General — follow domain and stated goal only."

        prompt = SUPPORT_PROMPTS["supportbot"].format(
            domain=domain,
            vertical_guidance=vg,
            goal=goal,
            context=context,
            structured_goal=structured_goal,
            domain_grounding=domain_grounding,
            supportbot_style=supportbot_style,
            tool_outputs=tool_outputs,
            memory_section=memory_section,
            history=history_text,
            planning_instruction=planning_instruction,
        )
        full_prompt = f"{SUPPORT_PROMPTS['system']}\n\n{prompt}"
        repair = (reflection_repair_hint or "").strip()
        if repair:
            full_prompt += (
                "\n\n## Reflection repair (required)\n"
                "Your previous reply failed automated verification. Revise your next reply to fix this "
                "(stay in character as support; be concrete):\n"
                f"{repair}\n"
            )
        max_words = getattr(self.config, "prompt_max_words", 1000)
        instr_words = getattr(self.config, "prompt_instruction_words", 250)
        truncated = truncate_prompt(full_prompt, max_words=max_words, instruction_words=instr_words)
        max_tokens_supportbot = getattr(self.config, "max_tokens_supportbot_turn", 120)
        if getattr(self.config, "agent_planning_enabled", True):
            max_tokens_supportbot = max(max_tokens_supportbot, getattr(self.config, "max_tokens_planning", 180))
        response = self.llm_client.generate_completion(
            truncated,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=max_tokens_supportbot,
        )
        plan_note = None
        if getattr(self.config, "agent_planning_enabled", True):
            if structured_planning_block:
                pass
            else:
                plan_note, response = parse_plan_and_reply(response)
        cleaned = clean_response(response, role="SupportBot").strip()
        word_count = len(cleaned.split()) if cleaned else 0
        generic_phrases = ("i'm sorry", "i cannot", "i can't help", "let me check", "i'll get back", "i don't have")
        is_generic = cleaned and any(cleaned.lower().strip().startswith(p) for p in generic_phrases)
        if (word_count < 5 or is_generic) and word_count > 0:
            retry_prompt = truncated + "\n\nProvide a concrete, helpful response with at least one specific detail (e.g. option, time, reference). Avoid generic apologies or deferrals."
            try:
                retry_response = self.llm_client.generate_completion(
                    retry_prompt,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=max_tokens_supportbot,
                )
                if getattr(self.config, "agent_planning_enabled", True):
                    _, retry_response = parse_plan_and_reply(retry_response)
                retry_cleaned = clean_response(retry_response, role="SupportBot").strip()
                if len(retry_cleaned.split()) >= 5 and retry_cleaned:
                    cleaned = retry_cleaned
                    logger.info("SupportBot turn improved after retry (was too short or generic)")
            except Exception as e:
                logger.warning("SupportBot retry failed, using first response: %s", e)
        if history:
            for prev_turn in history:
                if prev_turn.get("role") == "SupportBot":
                    prev_text = (prev_turn.get("text") or "").strip()
                    if prev_text and cleaned.lower() == prev_text.lower():
                        logger.warning("SupportBot response matched previous turn exactly; varying phrasing.")
                        last_user_msg = ""
                        for turn in reversed(recent):
                            if turn.get("role") == "User":
                                last_user_msg = turn.get("text", "")
                                break
                        cleaned = vary_supportbot_response(cleaned, goal, domain, last_user_msg)
                        break
        if not cleaned:
            cleaned = "I can help you with that."
        meta: Dict[str, Any] = {}
        if plan_note:
            meta["plan"] = plan_note
        if tools_used:
            meta["tools_used"] = tools_used
        return AgentGenerationResult(text=cleaned, metadata=meta)

    def update_memory(self, memory, recent_turns: List[Dict[str, Any]], **ctx: Any) -> None:
        return None
