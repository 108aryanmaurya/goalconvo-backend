"""
Dialogue state tracking for GoalConvo.

Maintains entities, per-domain slots, subtasks, open goals, user preferences,
and a bounded dialogue history. Updates can be driven by an LLM merge pass
(``update_memory`` / ``refresh_from_recent_turns``) so slots are extracted from
generated turns with explicit conflict resolution when values change.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..agent_capabilities import format_memory_lines, parse_memory_json
from ..utils import format_conversation_history

if TYPE_CHECKING:
    from ..config import Config
    from ..llm_client import LLMClient

logger = logging.getLogger(__name__)

_MAX_HISTORY = 80
_MAX_FACTS = 12
_MAX_CONFLICTS = 30


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text or not str(text).strip():
        return None
    raw = text.strip()
    if "```" in raw:
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I | re.M)
        raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        val = json.loads(raw[start : end + 1])
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        return None


@dataclass
class DialogueMemory:
    """
    Structured memory for multi-domain, goal-oriented dialogue simulation.

    Fields
    ------
    entities :
        Cross-domain named items (venues, confirmation refs, etc.), flat key → value.
    slots :
        ``{domain_id: {slot_name: value}}`` e.g. ``{"restaurant": {"venue": "Spice Garden", "people": 4}}``.
    completed_subtasks :
        Finished steps toward the overall goal.
    unresolved_goals :
        Open issues or remaining user intents not yet closed in the transcript.
    user_preferences :
        Short preference strings (tone, budget, dietary, etc.).
    dialogue_history :
        Recent turns ``{role, text, timestamp?}`` for retrieval and re-simulation.
    user_facts / support_facts :
        Legacy string buffers (kept for prompt compatibility with older prompts).
    reflection_trace :
        Optional planner/reflection audit entries.
    slot_conflicts :
        Ring buffer of slot/entity contradictions resolved in favour of newer values.
    """

    entities: Dict[str, Any] = field(default_factory=dict)
    slots: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    completed_subtasks: List[str] = field(default_factory=list)
    unresolved_goals: List[str] = field(default_factory=list)
    user_preferences: List[str] = field(default_factory=list)
    dialogue_history: List[Dict[str, Any]] = field(default_factory=list)
    user_facts: List[str] = field(default_factory=list)
    support_facts: List[str] = field(default_factory=list)
    reflection_trace: List[Dict[str, Any]] = field(default_factory=list)
    slot_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    rolling_dialogue_summary: str = ""
    research_consistency_hint: str = ""

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Full state as a JSON-serializable dict (multi-domain safe)."""
        return {
            "entities": dict(self.entities),
            "slots": {k: dict(v) for k, v in self.slots.items()},
            "completed_subtasks": list(self.completed_subtasks),
            "unresolved_goals": list(self.unresolved_goals),
            "user_preferences": list(self.user_preferences),
            "dialogue_history": list(self.dialogue_history),
            "user_facts": list(self.user_facts),
            "support_facts": list(self.support_facts),
            "reflection_trace": list(self.reflection_trace),
            "slot_conflicts": list(self.slot_conflicts),
            "rolling_dialogue_summary": self.rolling_dialogue_summary,
            "research_consistency_hint": self.research_consistency_hint,
        }

    def to_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        """Serialize memory to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=ensure_ascii)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueMemory":
        """Hydrate memory from ``to_dict`` / persisted JSON."""
        mem = cls()
        if not isinstance(data, dict):
            return mem
        mem.entities = dict(data.get("entities") or {})
        raw_slots = data.get("slots") or {}
        if isinstance(raw_slots, dict):
            mem.slots = {str(d): dict(sv) if isinstance(sv, dict) else {} for d, sv in raw_slots.items()}
        mem.completed_subtasks = [str(x) for x in (data.get("completed_subtasks") or []) if str(x).strip()]
        mem.unresolved_goals = [str(x) for x in (data.get("unresolved_goals") or []) if str(x).strip()]
        mem.user_preferences = [str(x) for x in (data.get("user_preferences") or []) if str(x).strip()]
        hist = data.get("dialogue_history") or []
        mem.dialogue_history = [dict(x) for x in hist if isinstance(x, dict)]
        if len(mem.dialogue_history) > _MAX_HISTORY:
            mem.dialogue_history = mem.dialogue_history[-_MAX_HISTORY:]
        mem.user_facts = [str(x) for x in (data.get("user_facts") or []) if str(x).strip()][-_MAX_FACTS:]
        mem.support_facts = [str(x) for x in (data.get("support_facts") or []) if str(x).strip()][-_MAX_FACTS:]
        mem.reflection_trace = list(data.get("reflection_trace") or [])
        mem.slot_conflicts = list(data.get("slot_conflicts") or [])[-_MAX_CONFLICTS:]
        mem.rolling_dialogue_summary = str(data.get("rolling_dialogue_summary") or "").strip()
        mem.research_consistency_hint = str(data.get("research_consistency_hint") or "").strip()
        return mem

    @classmethod
    def from_json(cls, payload: str) -> "DialogueMemory":
        data = json.loads(payload)
        if not isinstance(data, dict):
            return cls()
        return cls.from_dict(data)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def clear_memory(self) -> None:
        """Reset all tracked state (new dialogue)."""
        self.entities.clear()
        self.slots.clear()
        self.completed_subtasks.clear()
        self.unresolved_goals.clear()
        self.user_preferences.clear()
        self.dialogue_history.clear()
        self.user_facts.clear()
        self.support_facts.clear()
        self.reflection_trace.clear()
        self.slot_conflicts.clear()
        self.rolling_dialogue_summary = ""
        self.research_consistency_hint = ""
        logger.info("DialogueMemory: cleared all state.")

    def record_reflection(self, payload: Dict[str, Any]) -> None:
        """Append an audit record (planner / reflection)."""
        self.reflection_trace.append(dict(payload))

    # ------------------------------------------------------------------ #
    # History append
    # ------------------------------------------------------------------ #
    def _append_dialogue_history(self, turns: List[Dict[str, Any]]) -> None:
        for t in turns:
            if not isinstance(t, dict):
                continue
            role = t.get("role")
            if role not in ("User", "SupportBot"):
                continue
            rec = {
                "role": role,
                "text": (t.get("text") or "").strip(),
                "timestamp": t.get("timestamp"),
                "metadata": t.get("metadata") if isinstance(t.get("metadata"), dict) else {},
            }
            if self.dialogue_history:
                last = self.dialogue_history[-1]
                if last.get("role") == rec["role"] and last.get("text") == rec["text"]:
                    continue
            self.dialogue_history.append(rec)
        if len(self.dialogue_history) > _MAX_HISTORY:
            self.dialogue_history = self.dialogue_history[-_MAX_HISTORY:]

    # ------------------------------------------------------------------ #
    # Conflict resolution + merges
    # ------------------------------------------------------------------ #
    def _record_slot_conflict(
        self,
        scope: str,
        key: str,
        old_val: Any,
        new_val: Any,
        domain: Optional[str] = None,
    ) -> None:
        entry = {
            "scope": scope,
            "domain": domain,
            "key": key,
            "old": old_val,
            "new": new_val,
            "resolution": "prefer_newer",
        }
        self.slot_conflicts.append(entry)
        self.slot_conflicts = self.slot_conflicts[-_MAX_CONFLICTS:]
        logger.info(
            "DialogueMemory: resolved conflict %s[%s]: %r -> %r",
            scope,
            key,
            old_val,
            new_val,
        )

    def _merge_entities(self, incoming: Dict[str, Any]) -> None:
        if not isinstance(incoming, dict):
            return
        for k, v in incoming.items():
            key = str(k).strip()
            if not key:
                continue
            if key in self.entities and self.entities[key] != v:
                self._record_slot_conflict("entity", key, self.entities[key], v, None)
            self.entities[key] = v

    def _merge_slots_domain(self, domain: str, incoming: Dict[str, Any]) -> None:
        if not isinstance(incoming, dict):
            return
        d = (domain or "general").strip().lower() or "general"
        if d not in self.slots:
            self.slots[d] = {}
        bucket = self.slots[d]
        for k, v in incoming.items():
            sk = str(k).strip()
            if not sk:
                continue
            if sk in bucket and bucket[sk] != v:
                self._record_slot_conflict("slot", sk, bucket[sk], v, d)
            bucket[sk] = v

    def _merge_slots_multi(self, incoming: Dict[str, Any]) -> None:
        """``incoming`` maps domain -> slot dict."""
        if not isinstance(incoming, dict):
            return
        for domain, slotmap in incoming.items():
            if isinstance(slotmap, dict):
                self._merge_slots_domain(str(domain), slotmap)

    def _extend_unique(self, target: List[str], items: List[Any], cap: int) -> None:
        for x in items:
            s = str(x).strip()
            if s and s not in target:
                target.append(s)
        del target[cap:]

    # ------------------------------------------------------------------ #
    # LLM merge (slot extraction + structured update)
    # ------------------------------------------------------------------ #
    def _build_merge_prompt(
        self,
        recent_turns: List[Dict[str, Any]],
        domain: str,
        goal: str,
    ) -> str:
        hist = format_conversation_history(
            [t for t in recent_turns if t.get("role") in ("User", "SupportBot")]
        )
        prior = {
            "entities": self.entities,
            "slots": self.slots,
            "completed_subtasks": self.completed_subtasks,
            "unresolved_goals": self.unresolved_goals,
            "user_preferences": self.user_preferences,
            "user_facts": self.user_facts,
            "support_facts": self.support_facts,
        }
        return f"""You update structured dialogue memory for a multi-domain goal-oriented simulator.
Output ONLY valid JSON (no markdown fences) with exactly these keys:
{{
  "user_facts": ["string", ...],
  "support_facts": ["string", ...],
  "entities": {{}},
  "slots": {{}},
  "completed_subtasks": ["string", ...],
  "unresolved_goals": ["string", ...],
  "user_preferences": ["string", ...]
}}

Rules:
- "slots" MUST be an object whose keys are domain ids (e.g. hotel, restaurant, taxi, train, attraction, general).
  Each value is a flat object of slot names to values (strings, numbers, or booleans). Example restaurant domain:
  {{"restaurant": {{"restaurant": "Spice Garden", "people": 4, "booking_time": "7 PM", "status": "pending"}}}}
- "entities" holds cross-domain named anchors (venue, hotel_name, confirmation_ref, etc.).
- Merge with PRIOR_JSON; extract new slots/entities only from evidence in NEW DIALOGUE.
- On contradiction for the same slot key within the same domain OR the same entity key, prefer the NEWEST dialogue evidence (we log conflicts server-side).
- completed_subtasks: append newly finished items (dedupe strings).
- unresolved_goals: set to the current open issues relative to the user goal (max 8 strings); if none, [].
- user_preferences: short atomic user-stated preferences (max 10 new strings merged with prior style facts).
- user_facts / support_facts: short atomic strings (max 8 each after merge).

Primary domain for this episode: {domain}
User goal: {goal}

PRIOR_JSON:
{json.dumps(prior, ensure_ascii=False)}

NEW DIALOGUE:
{hist}
"""

    def _apply_llm_merge(self, data: Dict[str, Any]) -> None:
        # Legacy fact lists (also used by memory_section_text)
        uf = data.get("user_facts")
        sf = data.get("support_facts")
        if isinstance(uf, list):
            self._extend_unique(self.user_facts, uf, _MAX_FACTS)
        if isinstance(sf, list):
            self._extend_unique(self.support_facts, sf, _MAX_FACTS)

        ent = data.get("entities")
        if isinstance(ent, dict):
            self._merge_entities(ent)

        sl = data.get("slots")
        if isinstance(sl, dict) and sl:
            self._merge_slots_multi(sl)

        cs = data.get("completed_subtasks")
        if isinstance(cs, list) and cs:
            self._extend_unique(self.completed_subtasks, cs, 24)

        ug = data.get("unresolved_goals")
        if isinstance(ug, list):
            merged = [str(x).strip() for x in ug if str(x).strip()]
            if merged:
                self.unresolved_goals = merged[:12]

        up = data.get("user_preferences")
        if isinstance(up, list):
            self._extend_unique(self.user_preferences, up, 16)

    # ------------------------------------------------------------------ #
    # Public update API
    # ------------------------------------------------------------------ #
    def update_memory(
        self,
        *,
        recent_turns: Optional[List[Dict[str, Any]]] = None,
        llm_client: Optional["LLMClient"] = None,
        config: Optional["Config"] = None,
        domain: str = "general",
        goal: str = "",
    ) -> None:
        """
        Automatic memory update after a simulator step.

        1. Appends ``recent_turns`` into ``dialogue_history`` (User/SupportBot only).
        2. If ``llm_client`` and ``config`` are provided, runs one structured merge for
           slot extraction from generated lines and conflict resolution.
        3. If no LLM is provided, only history is updated (offline / tests).
        """
        if recent_turns:
            self._append_dialogue_history(recent_turns)
        if not recent_turns or llm_client is None or config is None:
            return
        hist = format_conversation_history(
            [t for t in recent_turns if t.get("role") in ("User", "SupportBot")]
        )
        if not hist.strip():
            return
        prompt = self._build_merge_prompt(recent_turns, domain, goal)
        try:
            raw = llm_client.generate_completion(
                prompt,
                temperature=0.12,
                max_tokens=getattr(config, "max_tokens_memory_refresh", 512),
            )
            parsed = _extract_json_object(raw or "")
            if parsed:
                self._apply_llm_merge(parsed)
                logger.debug(
                    "DialogueMemory: LLM merge applied keys=%s",
                    list(parsed.keys()),
                )
            else:
                # Fallback: legacy user/support fact lists only
                nu, ns = parse_memory_json(raw or "")
                for x in nu:
                    if x not in self.user_facts:
                        self.user_facts.append(x)
                for x in ns:
                    if x not in self.support_facts:
                        self.support_facts.append(x)
                self.user_facts = self.user_facts[-_MAX_FACTS:]
                self.support_facts = self.support_facts[-_MAX_FACTS:]
                logger.warning("DialogueMemory: full JSON parse failed; used legacy fact parser.")
        except Exception as e:
            logger.warning("DialogueMemory: update_memory LLM merge failed: %s", e)

    def refresh_from_recent_turns(
        self,
        llm_client: "LLMClient",
        config: "Config",
        recent_turns: List[Dict[str, Any]],
        domain: str,
        goal: str,
    ) -> None:
        """Backward-compatible alias: forwards to :meth:`update_memory`."""
        self.update_memory(
            recent_turns=recent_turns,
            llm_client=llm_client,
            config=config,
            domain=domain,
            goal=goal,
        )

    # ------------------------------------------------------------------ #
    # Prompt helpers + retrieval
    # ------------------------------------------------------------------ #
    def memory_section_text(self, enabled: bool = True) -> str:
        if not enabled:
            return ""
        parts: List[str] = []
        if self.rolling_dialogue_summary.strip():
            parts.append(
                "Rolling dialogue summary (faithful; do not contradict earlier facts):\n"
                + self.rolling_dialogue_summary.strip()
                + "\n"
            )
        if self.research_consistency_hint.strip():
            parts.append(
                "Consistency / contradiction repair (address explicitly if still relevant):\n"
                + self.research_consistency_hint.strip()
                + "\n"
            )
        summary = self.get_context_summary(max_turns=0)
        if summary.strip():
            parts.append("State summary:\n" + summary.strip() + "\n")
        u = format_memory_lines("User facts (remembered)", self.user_facts)
        s = format_memory_lines("SupportBot facts (remembered)", self.support_facts)
        parts.extend([x for x in (u, s) if x])
        if not parts:
            return ""
        return "Tracked dialogue memory (stay consistent):\n" + "\n".join(parts)

    def get_context_summary(self, max_turns: int = 4) -> str:
        """
        Compact natural-language summary for prompts and debugging.

        ``max_turns`` includes up to that many recent User/SupportBot lines.
        """
        lines: List[str] = []
        if self.entities:
            pairs = [f"{k}={v!r}" for k, v in list(self.entities.items())[:12]]
            lines.append("Entities: " + "; ".join(pairs))
        if self.slots:
            for dom, sd in list(self.slots.items())[:8]:
                if not sd:
                    continue
                inner = ", ".join(f"{k}={v!r}" for k, v in list(sd.items())[:10])
                lines.append(f"Slots[{dom}]: {inner}")
        if self.user_preferences:
            lines.append("User preferences: " + "; ".join(self.user_preferences[:8]))
        if self.completed_subtasks:
            lines.append("Completed subtasks: " + "; ".join(self.completed_subtasks[:8]))
        if self.unresolved_goals:
            lines.append("Unresolved goals/issues: " + "; ".join(self.unresolved_goals[:8]))
        if max_turns > 0 and self.dialogue_history:
            tail = [t for t in self.dialogue_history if t.get("role") in ("User", "SupportBot")][-max_turns:]
            rh = format_conversation_history(tail)
            if rh.strip():
                lines.append("Recent turns:\n" + rh)
        return "\n".join(lines) if lines else ""

    def get_slots(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """Return slots for one domain, or a shallow copy of all domains."""
        if domain:
            d = domain.strip().lower()
            return dict(self.slots.get(d, {}))
        return {k: dict(v) for k, v in self.slots.items()}

    def get_entities(self) -> Dict[str, Any]:
        return dict(self.entities)

    def get_completed_subtasks(self) -> List[str]:
        return list(self.completed_subtasks)

    def get_unresolved_goals(self) -> List[str]:
        return list(self.unresolved_goals)

    def get_user_preferences(self) -> List[str]:
        return list(self.user_preferences)

    def get_dialogue_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        if last_n is None or last_n <= 0:
            return list(self.dialogue_history)
        return list(self.dialogue_history[-last_n:])

    def get_memory_state_for_planner(self) -> Dict[str, Any]:
        """Small JSON-safe blob for planner prompts (token-aware)."""
        out: Dict[str, Any] = {
            "user_facts": list(self.user_facts),
            "support_facts": list(self.support_facts),
            "slots": {k: dict(v) for k, v in self.slots.items()},
            "entity_keys": list(self.entities.keys())[:20],
            "unresolved_goals": list(self.unresolved_goals)[:6],
        }
        if self.rolling_dialogue_summary.strip():
            out["rolling_dialogue_summary"] = self.rolling_dialogue_summary.strip()[:2000]
        if self.research_consistency_hint.strip():
            out["consistency_repair_hint"] = self.research_consistency_hint.strip()[:1200]
        return out
