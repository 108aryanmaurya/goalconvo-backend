"""
Automatic goal-completion evaluation: rule-based checks plus optional LLM judge.

Determines whether the original task appears completed, required slots (by domain)
are reflected in the dialogue, and the final state aligns with the stated goal.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..config import Config

logger = logging.getLogger(__name__)

_TECH_ALIASES = frozenset({"technical_support", "tech_support", "support", "it_support"})


def _normalize_domain(domain: str) -> str:
    d = (domain or "general").strip().lower()
    if d in _TECH_ALIASES or d == "technical support":
        return "technical_support"
    return d


def _dialogue_blob(dialogue: Dict[str, Any]) -> str:
    parts: List[str] = []
    for t in dialogue.get("turns", []) or []:
        parts.append((t.get("text") or "").strip())
    goal = dialogue.get("goal") or ""
    ctx = dialogue.get("context") or ""
    return " ".join(parts + [goal, ctx]).lower()


def _goal_lower(dialogue: Dict[str, Any]) -> str:
    return (dialogue.get("goal") or "").lower()


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    if "```" in s:
        for ch in s.split("```"):
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


def _user_closure(blob: str) -> bool:
    return any(
        p in blob
        for p in (
            "thank you",
            "thanks",
            "perfect",
            "great",
            "that works",
            "all set",
            "resolved",
            "that fixed",
            "it's working",
            "appreciate it",
        )
    )


def _support_commitment(blob: str) -> bool:
    return bool(
        re.search(
            r"\b(confirmed|booked|reserved|reservation|reference|"
            r"pickup|your taxi|table for|i've arranged|all set on our side|"
            r"ticket|case (?:number|id)|escalat|patch|update available|"
            r"reset|restart|network adapter|settings)\b",
            blob,
            re.I,
        )
    )


class GoalCompletionEvaluator:
    """
    Hybrid goal completion: deterministic slot + task signals, optional LLM JSON judge.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config
        self._threshold = float(
            getattr(config, "goal_completion_score_threshold", 0.72) if config else 0.72
        )

    def evaluate(
        self,
        dialogue: Dict[str, Any],
        *,
        llm_client: Any = None,
        use_llm_judge: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
            goal_completed (bool), completion_score (0-1), missing_requirements (list[str]),
            plus diagnostic keys ``rule_score``, ``task_signals`` when useful.
        """
        cfg = self.config
        use_llm = use_llm_judge
        if use_llm is None and cfg is not None:
            use_llm = bool(getattr(cfg, "goal_completion_llm_judge", False))
        elif use_llm is None:
            use_llm = False

        rule = self._rule_based(dialogue)
        missing = list(rule["missing_requirements"])
        rule_score = float(rule["rule_score"])

        completion_score = rule_score
        llm_diag: Dict[str, Any] = {}

        if use_llm and llm_client is not None:
            lj = self._llm_judge(dialogue, llm_client)
            llm_diag = lj
            if lj.get("parsed"):
                p = lj["parsed"]
                ls = float(p.get("completion_score", rule_score))
                ls = max(0.0, min(1.0, ls))
                completion_score = 0.5 * rule_score + 0.5 * ls
                for m in p.get("missing_requirements") or []:
                    if isinstance(m, str) and m.strip() and m.strip() not in missing:
                        missing.append(m.strip())

        goal_completed = bool(completion_score >= self._threshold and rule["task_complete"])
        if use_llm and llm_client is not None and llm_diag.get("parsed"):
            goal_completed = goal_completed and bool(llm_diag["parsed"].get("goal_completed"))

        hard_constraints = [m for m in missing if m.startswith("constraint_")]
        if hard_constraints:
            goal_completed = False

        out: Dict[str, Any] = {
            "goal_completed": goal_completed,
            "completion_score": round(max(0.0, min(1.0, completion_score)), 4),
            "missing_requirements": missing,
            "rule_score": round(rule_score, 4),
            "task_complete": rule["task_complete"],
        }
        if llm_diag:
            out["llm_judge"] = {k: v for k, v in llm_diag.items() if k != "raw_text"}
        return out

    def _rule_based(self, dialogue: Dict[str, Any]) -> Dict[str, Any]:
        domain = _normalize_domain(str(dialogue.get("domain", "")))
        blob = _dialogue_blob(dialogue)
        goal = _goal_lower(dialogue)

        missing: List[str] = []
        checks_passed = 0
        checks_total = 0

        def need(cond: bool, label: str) -> None:
            nonlocal checks_passed, checks_total
            checks_total += 1
            if cond:
                checks_passed += 1
            else:
                missing.append(label)

        if domain == "hotel":
            need(
                bool(
                    re.search(
                        r"\b(book|booked|confirm|reserved|reservation|reference|check[- ]?in|"
                        r"hold the room|availability|nights?)\b",
                        blob,
                    )
                ),
                "hotel_booking_or_confirmation_evidence",
            )
            if re.search(r"\b(night|tonight|tomorrow|check[- ]?in|check[- ]?out|guests?|star|parking|wifi)\b", goal):
                need(
                    bool(
                        re.search(
                            r"\b(night|tonight|tomorrow|date|pm|am|check|guest|star|park|wifi|room)\b",
                            blob,
                        )
                    ),
                    "hotel_stay_constraints_addressed",
                )
        elif domain == "restaurant":
            need(
                bool(
                    re.search(
                        r"\b(reserv|table|booked|confirm|dinner|lunch|party|people|guests|italian|indian|vegetarian|cuisine)\b",
                        blob,
                    )
                ),
                "restaurant_reservation_or_booking_details",
            )
            if re.search(r"\b(time|dinner|lunch|pm|am|party|people|guest|vegetarian|vegan|italian)\b", goal):
                need(
                    bool(re.search(r"\b(time|dinner|lunch|pm|am|table|seat|people|party|vegetarian|italian)\b", blob)),
                    "restaurant_time_or_party_or_diet_addressed",
                )
        elif domain == "taxi":
            need(
                bool(re.search(r"\b(from|pickup|collect|to |destination|airport|station|drop)\b", blob)),
                "taxi_route_or_locations",
            )
            need(
                bool(
                    re.search(
                        r"\b(?:\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm)|pickup|leave at|depart)\b",
                        blob,
                        re.I,
                    )
                ),
                "taxi_pickup_time",
            )
            need(
                bool(
                    re.search(
                        r"\b(confirmed|booked|reservation|reference|driver|assigned|"
                        r"cab is on the way|on the way|your ride is)\b",
                        blob,
                        re.I,
                    )
                ),
                "taxi_booking_confirmation",
            )
        elif domain == "technical_support":
            need(
                bool(
                    re.search(
                        r"\b(error|password|login|wifi|network|crash|install|update|"
                        r"browser|account|sync|vpn|driver|blue screen)\b",
                        blob,
                    )
                ),
                "support_problem_identified_or_described",
            )
            need(
                bool(
                    re.search(
                        r"\b(try|step|setting|reset|restart|clear cache|update|"
                        r"reinstall|ticket|escalat|workaround|patch|version)\b",
                        blob,
                    )
                ),
                "support_troubleshooting_or_resolution_steps",
            )
        else:
            # General / train / attraction: softer checklist
            need(_support_commitment(blob) or _user_closure(blob), "dialogue_progress_or_closure")

        constraints = dialogue.get("constraints")
        if isinstance(constraints, dict) and constraints:
            for k, v in constraints.items():
                if v is None or str(v).strip() == "":
                    continue
                fragment = str(v).lower().strip()[:80]
                if len(fragment) >= 3 and fragment not in blob:
                    missing.append(f"constraint_{k}_not_reflected_in_dialogue")

        task_complete = self._task_complete(blob)
        slot_fraction = checks_passed / checks_total if checks_total else 1.0
        task_bonus = 1.0 if task_complete else 0.35
        rule_score = max(0.0, min(1.0, 0.65 * slot_fraction + 0.35 * task_bonus))

        return {
            "missing_requirements": missing,
            "rule_score": rule_score,
            "task_complete": task_complete,
            "domain": domain,
        }

    def _task_complete(self, blob: str) -> bool:
        """User shows closure AND assistant showed concrete progress (booking domains)."""
        u_ok = _user_closure(blob)
        s_ok = _support_commitment(blob)
        return bool(u_ok and s_ok)

    def _llm_judge(self, dialogue: Dict[str, Any], llm_client: Any) -> Dict[str, Any]:
        cfg = self.config
        max_tok = int(getattr(cfg, "max_tokens_goal_completion_judge", 220)) if cfg else 220
        temp = float(getattr(cfg, "goal_completion_judge_temperature", 0.1)) if cfg else 0.1

        turns_txt = []
        for t in dialogue.get("turns", []) or []:
            turns_txt.append(f"{t.get('role','?')}: {t.get('text','')}")
        transcript = "\n".join(turns_txt[-40:])
        domain = _normalize_domain(str(dialogue.get("domain", "")))
        goal = dialogue.get("goal", "")

        prompt = f"""You are evaluating whether a support-style dialogue successfully completed the user's task.

Domain: {domain}
User goal: {goal}

Transcript (most recent last):
{transcript}

Return ONLY valid JSON (no markdown) with this exact shape:
{{"goal_completed": true, "completion_score": 0.85, "missing_requirements": []}}

Rules:
- goal_completed: true only if the user's original task appears resolved or fully arranged (not just generic help offers).
- completion_score: float 0.0-1.0 (1.0 = fully satisfied with evidence in the last turns).
- missing_requirements: short snake_case strings listing concrete gaps (e.g. missing_confirmation, no_pickup_time), or [] if none.

For hotel/restaurant/taxi bookings require evidence of confirmation or concrete arrangement.
For technical_support require troubleshooting steps and a plausible resolution or clear escalation.
"""
        try:
            raw = llm_client.generate_completion(prompt, temperature=temp, max_tokens=max_tok)
        except Exception as e:
            logger.warning("Goal completion LLM judge failed: %s", e)
            return {"error": str(e), "parsed": None}

        parsed = _extract_json_object(raw or "")
        if not isinstance(parsed, dict):
            return {"error": "parse_failed", "raw_text": (raw or "")[:400], "parsed": None}
        return {"parsed": parsed, "raw_text": None}
