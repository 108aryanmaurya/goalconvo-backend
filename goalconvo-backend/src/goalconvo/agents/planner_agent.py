"""
Structured dialogue planner for GoalConvo.

This module implements a PlannerAgent that runs *before* user/support utterance
generation. It asks the LLM (Mistral-style instruction models supported via
``LLMClient``) to reason about the dialogue, then emits a fixed-shape JSON plan.

Why planning improves goal-oriented dialogue:
    Autoregressive models optimize next-token likelihood, not explicit task graphs.
    In multi-turn goal settings that mismatch shows up as: forgotten constraints,
    premature “done” claims, and topic drift. A planner pass gives the model a
    compact *commitment* to state and missing information so the downstream
    speaker prompt can stay on-mission—similar in spirit to policy + state
    tracking in classical task-oriented systems, but kept JSON-native for research.

How state tracking works here:
    We do not enforce a hotel/restaurant ontology in code. Instead we ask the
    planner to maintain ``identified_slots`` and ``missing_slots`` as string
    lists summarizing grounded facts vs gaps. The simulator also passes a small
    ``dialogue_state`` dict (turn index, last speaker, domain, etc.) so the
    planner can align with the outer loop. Memory buffers (user_facts,
    support_facts) are passed as JSON for cross-turn consistency checks.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..agent_capabilities import build_planning_instruction
from ..prompts.domains import vertical_guidance_for_domain
from ..prompts.planner_prompts import STRUCTURED_DIALOGUE_PLANNER_PROMPTS
from ..research.planner_augment import structured_planner_research_suffix
from ..utils import format_conversation_history
from .base import AgentGenerationResult, BaseDialogueAgent, ValidationResult

logger = logging.getLogger(__name__)

# Required keys for the structured planner contract (research / eval hooks).
_PLAN_REQUIRED_KEYS: Tuple[str, ...] = (
    "current_state",
    "identified_slots",
    "missing_slots",
    "subgoal",
    "next_action",
    "goal_progress",
)


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse the first JSON object embedded in model output.

    Models may wrap JSON in whitespace or accidentally add stray characters;
    we try a whole-string parse first, then a balanced-brace slice.
    """
    if not text or not text.strip():
        return None
    raw = text.strip()
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = raw[start : end + 1]
    try:
        val = json.loads(snippet)
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        return None


def validate_structured_plan(plan: Any) -> Tuple[bool, List[str]]:
    """
    Validate planner JSON shape and types.

    Returns (ok, issues) where issues are machine-readable tags for logging
    and retry prompts.
    """
    issues: List[str] = []
    if not isinstance(plan, dict):
        return False, ["not_a_json_object"]
    for key in _PLAN_REQUIRED_KEYS:
        if key not in plan:
            issues.append(f"missing_key:{key}")
    if issues:
        return False, issues
    for list_key in ("identified_slots", "missing_slots"):
        v = plan[list_key]
        if v is None:
            issues.append(f"null_list:{list_key}")
        elif not isinstance(v, list):
            issues.append(f"not_a_list:{list_key}")
        else:
            for i, item in enumerate(v):
                if not isinstance(item, (str, int, float, bool)):
                    issues.append(f"bad_slot_item:{list_key}:{i}")
    for str_key in ("current_state", "subgoal", "next_action", "goal_progress"):
        if plan[str_key] is not None and not isinstance(plan[str_key], str):
            issues.append(f"not_a_string:{str_key}")
    return (len(issues) == 0), issues


def normalize_structured_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce values into the contract types (defensive for partially valid JSON)."""
    out: Dict[str, Any] = {}
    for key in _PLAN_REQUIRED_KEYS:
        val = plan.get(key)
        if key in ("identified_slots", "missing_slots"):
            if not isinstance(val, list):
                out[key] = []
            else:
                out[key] = [str(x).strip() for x in val if str(x).strip()]
        else:
            out[key] = "" if val is None else str(val).strip()
    hier = plan.get("subgoal_hierarchy")
    if isinstance(hier, list):
        out["subgoal_hierarchy"] = [str(x).strip() for x in hier if str(x).strip()]
    else:
        out["subgoal_hierarchy"] = []
    notes = plan.get("decomposition_notes")
    out["decomposition_notes"] = "" if notes is None else str(notes).strip()
    return out


def default_fallback_plan(reason: str) -> Dict[str, Any]:
    """Safe default when parsing/validation fails after all retries."""
    return {
        "current_state": "planner_fallback",
        "identified_slots": [],
        "missing_slots": [],
        "subgoal": "Continue the dialogue toward the stated user goal.",
        "next_action": "Address the last visible turn in history coherently.",
        "goal_progress": f"fallback:{reason}",
        "subgoal_hierarchy": [],
        "decomposition_notes": "",
    }


def format_structured_plan_for_prompt(plan: Dict[str, Any]) -> str:
    """
    Turn validated planner JSON into a natural-language block for speaker prompts.

    The speaker model reads this *before* generating dialogue; it is not shown
    to end users in real deployments—only to the simulating LLM.
    """
    p = normalize_structured_plan(plan)
    lines = [
        "## Structured planner (reason before you speak — follow this analysis)",
        f"- Current dialogue state: {p['current_state']}",
        f"- Identified slots (grounded facts): {', '.join(p['identified_slots']) if p['identified_slots'] else '(none)'}",
        f"- Missing slots / info gaps: {', '.join(p['missing_slots']) if p['missing_slots'] else '(none)'}",
        f"- Active subgoal: {p['subgoal']}",
        f"- Next action (your move): {p['next_action']}",
        f"- Goal progress: {p['goal_progress']}",
    ]
    if p.get("subgoal_hierarchy"):
        lines.append("- Subgoal hierarchy (root → leaf): " + " → ".join(p["subgoal_hierarchy"]))
    if (p.get("decomposition_notes") or "").strip():
        lines.append(f"- Decomposition notes: {p['decomposition_notes']}")
    lines.extend(
        [
            "",
            "Compose your next utterance to advance the subgoal; do not contradict identified slots.",
        ]
    )
    return "\n".join(lines)


class PlannerAgent(BaseDialogueAgent):
    """
    Produces a structured JSON plan from dialogue history, goal, state, and memory.

    Generation uses low temperature and modest top-p for near-deterministic
    Mistral-style behavior (configurable via ``Config.planner_temperature``).
    """

    def generate(
        self,
        goal: str = "",
        context: str = "",
        domain: str = "general",
        dialogue_history: Optional[List[Dict[str, Any]]] = None,
        dialogue_state: Optional[Dict[str, Any]] = None,
        memory_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AgentGenerationResult:
        """
        Run the structured planner LLM and return JSON text plus parsed metadata.

        When ``structured_planner_enabled`` is False but ``agent_planning_enabled``
        is True, returns the legacy inline PLAN/REPLY instruction string for
        backward compatibility (no LLM call).
        """
        if not getattr(self.config, "agent_planning_enabled", True):
            logger.info("PlannerAgent: agent_planning_enabled=False — skipping planning.")
            return AgentGenerationResult(
                text="",
                metadata={"enabled": False, "skipped": True, "mode": "disabled"},
            )

        if not getattr(self.config, "structured_planner_enabled", True):
            instruction = build_planning_instruction()
            logger.info("PlannerAgent: structured planner off — using inline PLAN/REPLY scaffold only.")
            return AgentGenerationResult(
                text=instruction,
                metadata={"enabled": True, "structured": False, "mode": "inline_scaffold"},
            )

        history = dialogue_history or []
        k = getattr(self.config, "prompt_last_k_turns", 6)
        conv = [h for h in history if h.get("role") != "System"]
        recent = conv[-k:] if len(conv) > k else conv
        history_text = format_conversation_history(recent) if recent else "(no turns yet)"

        dialogue_state_json = json.dumps(dialogue_state or {}, ensure_ascii=False)
        memory_state_json = json.dumps(memory_state or {}, ensure_ascii=False)

        vg = (vertical_guidance_for_domain(domain) or "").strip()
        if not vg:
            vg = "General task-oriented dialogue — prioritize the stated goal and domain."

        system = STRUCTURED_DIALOGUE_PLANNER_PROMPTS["system"]
        user_msg = STRUCTURED_DIALOGUE_PLANNER_PROMPTS["user"].format(
            domain=domain,
            vertical_guidance=vg,
            goal=goal or "(unspecified)",
            context=context or "(none)",
            dialogue_state_json=dialogue_state_json,
            memory_state_json=memory_state_json,
            history=history_text,
        )
        rsfx = structured_planner_research_suffix(self.config)
        if rsfx:
            user_msg = user_msg + "\n\n" + rsfx
        full_prompt = f"{system}\n\n{user_msg}"

        temp = float(getattr(self.config, "planner_temperature", 0.1))
        top_p = float(getattr(self.config, "planner_top_p", 0.85))
        max_tokens = int(getattr(self.config, "max_tokens_structured_planner", 320))
        max_retries = max(1, int(getattr(self.config, "planner_json_max_retries", 3)))

        last_issues: List[str] = []
        raw_last = ""
        parsed: Optional[Dict[str, Any]] = None

        for attempt in range(1, max_retries + 1):
            prompt = full_prompt
            if attempt > 1 and last_issues:
                repair = (
                    "\n\nYour previous answer was invalid: "
                    + "; ".join(last_issues)
                    + ". Output ONLY one JSON object with ALL required keys "
                    + "(current_state, identified_slots, missing_slots, subgoal, next_action, goal_progress). "
                    "Optional keys when applicable: subgoal_hierarchy (string array), decomposition_notes (string). "
                    "No markdown fences, no commentary."
                )
                prompt = full_prompt + repair
                logger.warning(
                    "PlannerAgent: retry %s/%s after validation failure: %s",
                    attempt,
                    max_retries,
                    last_issues,
                )

            raw_last = self.llm_client.generate_completion(
                prompt,
                temperature=temp,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            logger.debug(
                "PlannerAgent: raw generation (truncated): %s",
                (raw_last or "")[:800],
            )

            candidate = extract_first_json_object(raw_last or "")
            if candidate is None:
                last_issues = ["json_parse_failed"]
                logger.warning("PlannerAgent: attempt %s JSON parse failed.", attempt)
                continue

            ok, issues = validate_structured_plan(candidate)
            if ok:
                parsed = normalize_structured_plan(candidate)
                logger.info(
                    "PlannerAgent: valid structured plan (attempt %s/%s) — %s",
                    attempt,
                    max_retries,
                    json.dumps(parsed, ensure_ascii=False)[:500],
                )
                body = json.dumps(parsed, ensure_ascii=False)
                return AgentGenerationResult(
                    text=body,
                    metadata={
                        "enabled": True,
                        "structured": True,
                        "mode": "json_plan",
                        "parsed_plan": parsed,
                        "attempts_used": attempt,
                        "planner_temperature": temp,
                    },
                )

            last_issues = issues
            logger.warning(
                "PlannerAgent: attempt %s/%s validation failed: %s",
                attempt,
                max_retries,
                issues,
            )

        fb = default_fallback_plan("max_retries_exhausted")
        logger.error(
            "PlannerAgent: exhausted retries (%s); using fallback plan. Last issues: %s",
            max_retries,
            last_issues,
        )
        body = json.dumps(fb, ensure_ascii=False)
        return AgentGenerationResult(
            text=body,
            metadata={
                "enabled": True,
                "structured": True,
                "mode": "fallback",
                "parsed_plan": fb,
                "attempts_used": max_retries,
                "last_raw_truncated": (raw_last or "")[:400],
                "last_validation_issues": last_issues,
                "planner_temperature": temp,
            },
        )

    def validate(self, result: AgentGenerationResult, ctx: Optional[Dict[str, Any]] = None) -> ValidationResult:
        if result.metadata.get("skipped"):
            return ValidationResult(ok=True, issues=[])
        if result.metadata.get("structured") is False:
            return ValidationResult(ok=True, issues=[]) if (result.text or "").strip() else ValidationResult(
                ok=False, issues=["empty_inline_scaffold"]
            )
        parsed = result.metadata.get("parsed_plan")
        if not isinstance(parsed, dict):
            return ValidationResult(ok=False, issues=["missing_parsed_plan"])
        ok, issues = validate_structured_plan(parsed)
        return ValidationResult(ok=ok, issues=issues)

    @staticmethod
    def format_planning_prompt_block(plan: Optional[Dict[str, Any]]) -> str:
        """Public helper for pipelines: validated dict → speaker prompt section."""
        if not plan:
            return ""
        return format_structured_plan_for_prompt(plan)

    def update_memory(self, memory: Any, recent_turns: List[Dict[str, Any]], **ctx: Any) -> None:
        """Optionally log planner steps on ``DialogueMemory.reflection_trace`` when present."""
        if memory is None:
            return
        parsed = ctx.get("parsed_plan")
        if parsed is not None and hasattr(memory, "record_reflection"):
            try:
                memory.record_reflection(
                    {
                        "type": "structured_planner",
                        "plan": parsed,
                        "turn_tail": len(recent_turns),
                    }
                )
            except Exception as e:
                logger.debug("PlannerAgent.update_memory: record_reflection skipped: %s", e)
