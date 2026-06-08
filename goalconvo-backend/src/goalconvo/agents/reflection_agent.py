"""
Goal-satisfaction heuristics and utterance-level **reflection** for GoalConvo.

Reflection-based generation
---------------------------
Classic autoregressive decoding commits to tokens as they are sampled. **Reflection-based
generation** adds one or more *critique* passes: after (or while) producing a candidate
reply, a verifier scores it against explicit criteria (goal fit, faithfulness, style).
Only candidates that pass the bar are emitted; otherwise the system may **regenerate**
with corrective hints. This mirrors human drafting (write → review → revise) and reduces
surface errors without retraining the base model.

Self-correction pipelines in modern LLM systems
-----------------------------------------------
Production stacks often chain: **generate → verify → repair** (sometimes iteratively).
Examples include tool-use *guardrails*, RAG *groundedness* checks, constitutional / rule
layers, and *LLM-as-judge* JSON scorers. These **self-correction pipelines** trade extra
latency and cost for higher reliability: the verifier can be smaller, colder, or a
specialized model. Important limitations remain—verifiers can be *wrong* or *aligned* with
the generator's mistakes—so we combine learned judgments with simple structural rules
(e.g. minimum score, blocking labels for hallucination or memory contradiction) and a
hard **max attempts** cap to avoid infinite loops.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..prompts.reflection_prompts import REFLECTION_PROMPTS
from ..utils import format_conversation_history
from .base import AgentGenerationResult, BaseDialogueAgent, ValidationResult

logger = logging.getLogger(__name__)


class ReflectionAgent(BaseDialogueAgent):
    """LLM goal checks, keyword completion signals, and per-utterance critique + regen loop."""

    def generate(self, task: str = "goal_probe", goal: str = "", history: Optional[List[Dict[str, str]]] = None, **_: Any) -> AgentGenerationResult:
        if task != "goal_probe" or not history:
            return AgentGenerationResult(text="", metadata={"skipped": True})
        history_text = format_conversation_history(history)
        prompt = REFLECTION_PROMPTS["goal_check"].format(goal=goal, history=history_text)
        response = self.llm_client.generate_completion(prompt, temperature=0.1, max_tokens=3)
        return AgentGenerationResult(text=(response or "").strip(), metadata={"task": task})

    def validate(self, result: AgentGenerationResult, ctx: Optional[Dict[str, Any]] = None) -> ValidationResult:
        if result.metadata.get("skipped"):
            return ValidationResult(ok=True, issues=[])
        return super().validate(result, ctx)

    def evaluate_utterance(
        self,
        *,
        goal: str,
        dialogue_history: Optional[List[Dict[str, str]]] = None,
        memory_state: Optional[Dict[str, Any]] = None,
        generated_response: str,
        role: str,
        domain: str = "general",
    ) -> Dict[str, Any]:
        """
        LLM-as-judge over one draft. Returns JSON-shaped dict:
        accepted, issues, score (0–5), reason.

        Checks (via the critique prompt): goal alignment, hallucination, repetition,
        coherence, task progress, contradiction with memory.
        """
        dialogue_history = dialogue_history or []
        memory_state = memory_state or {}
        k = getattr(self.config, "prompt_last_k_turns", 6)
        recent = dialogue_history[-k:] if dialogue_history else []
        history_text = format_conversation_history(recent)
        try:
            memory_json = json.dumps(memory_state, indent=2, ensure_ascii=False)[:8000]
        except (TypeError, ValueError):
            memory_json = "{}"

        prompt = REFLECTION_PROMPTS["response_critique"].format(
            role=role,
            goal=goal or "",
            domain=domain or "general",
            history=history_text or "(empty)",
            memory_json=memory_json,
            generated_response=generated_response or "",
        )

        verdict: Dict[str, Any]
        try:
            raw_text = self.llm_client.generate_completion(
                prompt,
                temperature=getattr(self.config, "reflection_temperature", 0.15),
                max_tokens=getattr(self.config, "max_tokens_reflection", 320),
            ) or ""
            parsed = self._extract_json_object(raw_text)
            if not parsed:
                logger.warning("Utterance reflection: JSON parse failed; preview=%r", (raw_text or "")[:400])
                verdict = self._heuristic_verdict(
                    generated_response=generated_response,
                    goal=goal,
                    history=dialogue_history,
                    role=role,
                    _memory_state=memory_state,
                )
                verdict["parse_error"] = True
            else:
                verdict = self._verdict_from_parsed(parsed)
        except Exception as e:
            logger.exception("Utterance reflection LLM failed: %s", e)
            verdict = self._heuristic_verdict(
                generated_response=generated_response,
                goal=goal,
                history=dialogue_history,
                role=role,
                _memory_state=memory_state,
            )
            verdict["llm_error"] = str(e)

        verdict = self._finalize_acceptance(verdict)
        logger.debug(
            "evaluate_utterance: role=%s accepted=%s score=%s issues=%s",
            role,
            verdict.get("accepted"),
            verdict.get("score"),
            verdict.get("issues"),
        )
        return verdict

    def run_reflected_generation(
        self,
        *,
        role: str,
        goal: str,
        dialogue_history: List[Dict[str, str]],
        memory_state: Dict[str, Any],
        domain: str,
        generate_one: Callable[[Optional[str]], AgentGenerationResult],
        reflection_attempt_bonus: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        **Regeneration loop**: generate → reflect → if rejected, regenerate with repair hint.

        Stops early when the reflection accepts the draft or when ``reflection_max_attempts``
        is reached (last draft is kept; caller metadata flags exhaustion).

        ``reflection_attempt_bonus`` adds extra regeneration tries (e.g. when a consistency
        pass flagged likely contradictions) without changing global config.
        """
        max_attempts = max(1, int(getattr(self.config, "reflection_max_attempts", 3)))
        max_attempts += max(0, int(reflection_attempt_bonus))
        trace: List[Dict[str, Any]] = []
        hint: Optional[str] = None
        last_result: Optional[AgentGenerationResult] = None

        for attempt in range(1, max_attempts + 1):
            logger.info(
                "Reflection pipeline: role=%s attempt=%s/%s has_repair_hint=%s",
                role,
                attempt,
                max_attempts,
                bool(hint),
            )
            last_result = generate_one(hint)
            text = (last_result.text or "").strip()
            verdict = self.evaluate_utterance(
                goal=goal,
                dialogue_history=dialogue_history,
                memory_state=memory_state,
                generated_response=text,
                role=role,
                domain=domain,
            )
            trace.append(
                {
                    "attempt": attempt,
                    "verdict": verdict,
                    "draft_preview": text[:240],
                }
            )
            logger.info(
                "Utterance reflection: accepted=%s score=%s issues=%s reason=%s",
                verdict.get("accepted"),
                verdict.get("score"),
                verdict.get("issues"),
                verdict.get("reason"),
            )
            if verdict.get("accepted"):
                meta = dict(last_result.metadata)
                meta["utterance_reflection"] = {"attempts": trace, "exhausted": False}
                return text, meta

            hint = (verdict.get("reason") or "").strip() or "; ".join(verdict.get("issues") or [])
            if not hint:
                hint = (
                    "Improve goal alignment, avoid unsupported facts, avoid repeating prior lines, "
                    "stay coherent with the last turn, advance the task, and stay consistent with memory."
                )

        assert last_result is not None
        logger.warning(
            "Reflection max attempts exhausted (%s); using last draft for role=%s",
            max_attempts,
            role,
        )
        meta = dict(last_result.metadata)
        meta["utterance_reflection"] = {"attempts": trace, "exhausted": True}
        return (last_result.text or "").strip(), meta

    @staticmethod
    def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
        if not raw or not str(raw).strip():
            return None
        s = str(raw).strip()
        if "```" in s:
            chunks = s.split("```")
            for ch in chunks:
                t = ch.strip()
                if t.lower().startswith("json"):
                    t = t[4:].lstrip()
                if t.startswith("{"):
                    s = t
                    break
        try:
            start = s.index("{")
            end = s.rindex("}") + 1
            return json.loads(s[start:end])
        except (ValueError, json.JSONDecodeError):
            return None

    def _verdict_from_parsed(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        issues_raw = parsed.get("issues")
        if isinstance(issues_raw, list):
            issues = [str(x) for x in issues_raw]
        elif issues_raw is None or issues_raw == "":
            issues = []
        else:
            issues = [str(issues_raw)]
        reason = str(parsed.get("reason") or "")
        try:
            score = int(parsed.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(5, score))
        return {
            "accepted": bool(parsed.get("accepted")),
            "issues": issues,
            "score": score,
            "reason": reason,
        }

    def _finalize_acceptance(self, verdict: Dict[str, Any]) -> Dict[str, Any]:
        """Merge model judgment with configurable score floor and blocking issue tags."""
        score = int(verdict.get("score", 0))
        score = max(0, min(5, score))
        min_s = int(getattr(self.config, "reflection_min_accept_score", 4))
        min_s = max(1, min(5, min_s))
        issues = verdict.get("issues") or []
        joined = " ".join(str(i).lower() for i in issues)
        blocking = ("hallucination" in joined or "memory_contradiction" in joined) and score < 5
        accepted = (score >= min_s) and (not blocking)
        out = {**verdict, "score": score, "accepted": accepted}
        return out

    def _heuristic_verdict(
        self,
        *,
        generated_response: str,
        goal: str,
        history: List[Dict[str, str]],
        role: str,
        _memory_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Cheap fallback when the judge LLM fails: repetition + length + goal overlap."""
        issues: List[str] = []
        score = 4
        text = (generated_response or "").strip()
        if not text:
            return {"accepted": False, "issues": ["coherence"], "score": 0, "reason": "Empty utterance."}

        last_same = ""
        for turn in reversed(history):
            if turn.get("role") == role:
                last_same = (turn.get("text") or "").strip()
                break
        if last_same and text.lower() == last_same.lower():
            issues.append("repetition")
            score -= 2

        if len(text.split()) < 3:
            issues.append("coherence")
            score -= 1

        gw = set(re.findall(r"[a-z0-9']+", (goal or "").lower())) - {"the", "a", "an", "to", "i", "and", "for"}
        cw = set(re.findall(r"[a-z0-9']+", text.lower()))
        if gw and not (gw & cw) and len(text.split()) < 8:
            issues.append("goal_alignment")
            score -= 1

        score = max(0, min(5, score))
        return {
            "accepted": False,
            "issues": issues,
            "score": score,
            "reason": "Heuristic fallback because reflection JSON or LLM was unavailable.",
        }

    def goal_satisfied_llm(self, goal: str, history: List[Dict[str, str]]) -> bool:
        user_turns = [h for h in history if h.get("role") == "User"]
        supportbot_turns = [h for h in history if h.get("role") == "SupportBot"]
        if len(user_turns) + len(supportbot_turns) < self.config.min_turns:
            return False
        if len(user_turns) < 2 or len(supportbot_turns) < 2:
            return False
        try:
            res = self.generate(task="goal_probe", goal=goal, history=history)
            response_upper = res.text.strip().upper()
            if response_upper.startswith("YES"):
                return True
            if "YES" in response_upper and "NO" not in response_upper:
                return True
            return False
        except Exception as e:
            logger.error("Error checking goal satisfaction: %s", e)
            return self.goal_satisfied_keywords(goal, history)

    @staticmethod
    def goal_satisfied_keywords(goal: str, history: List[Dict[str, str]]) -> bool:
        user_satisfaction_keywords = [
            "thank you",
            "thanks",
            "perfect",
            "great",
            "excellent",
            "that's great",
            "that works",
            "sounds good",
            "all set",
            "i'm all set",
            "that's exactly what I needed",
            "that'll work",
            "appreciate it",
            "good, thank",
        ]
        assistant_completion_keywords = [
            "booked",
            "confirmed",
            "reserved",
            "reservation",
            "reference",
            "reference number",
            "pickup at",
            "at ",
            "pm",
            "am",
            "from ",
            "to ",
            "venue",
            "restaurant",
            "hotel",
            "taxi",
            "booking",
            "confirmation",
            "address",
            "phone number",
            "time is",
            "scheduled",
        ]
        recent_turns = history[-6:] if len(history) >= 6 else history
        if len(recent_turns) < 2:
            return False
        has_user_satisfaction = False
        has_assistant_evidence = False
        for turn in recent_turns:
            text = (turn.get("text") or "").lower()
            role = turn.get("role", "")
            if role == "User":
                if any(kw in text for kw in user_satisfaction_keywords):
                    has_user_satisfaction = True
            elif role == "SupportBot":
                if any(kw in text for kw in assistant_completion_keywords):
                    has_assistant_evidence = True
        return has_user_satisfaction and has_assistant_evidence

    def update_memory(self, memory, recent_turns: List[Dict[str, Any]], **ctx: Any) -> None:
        if memory is not None and ctx.get("log_reflection"):
            memory.record_reflection({"turns": len(recent_turns), **ctx})
        return None
