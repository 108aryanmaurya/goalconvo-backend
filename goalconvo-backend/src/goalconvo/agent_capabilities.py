"""
Shared helpers for memory, tool stubs, and planning/reward plumbing used by DialogueSimulator.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def tool_search_stub(query: str, domain: str, goal: str) -> str:
    """Lightweight search simulation (no external network). Returns a short snippet the model can ground on."""
    q = (query or "").strip() or goal[:120]
    d = (domain or "general").lower()
    snippets = {
        "hotel": f"[search] Top matches near user request ({q[:80]}): mid-range central options; weekend availability likely; typical rate band £60–90/night (illustrative).",
        "restaurant": f"[search] Venues matching ({q[:80]}): dinner slots 18:00–21:00; vegetarian-friendly options listed; reserve 24h ahead recommended.",
        "taxi": f"[search] Local operators for ({q[:80]}): estimated pickup 8–15 min; city-center fare band £20–35 (illustrative).",
        "train": f"[search] Services for ({q[:80]}): hourly departures; express vs standard; seat reservation available on busy legs.",
        "attraction": f"[search] Attractions for ({q[:80]}): museum hours 10:00–18:00; walking tours morning/afternoon slots.",
    }
    return snippets.get(d, f"[search] General results for ({q[:100]}): several relevant options; confirm dates and party size with the user.")


def tool_db_lookup_stub(domain: str, goal: str, context: str) -> str:
    """Simulated structured DB lookup from domain + goal (no real DB)."""
    d = (domain or "general").lower()
    g = (goal or "")[:100].strip()
    ctx = (context or "")[:80].strip()
    refs = {
        "hotel": "HOTEL-REF",
        "restaurant": "REST-REF",
        "taxi": "TAXI-REF",
        "train": "TRN-REF",
        "attraction": "ATT-REF",
    }
    tag = refs.get(d, "GEN-REF")
    return (
        f"[db_lookup] domain={d}; goal_hint={g!r}; context_hint={ctx!r}; "
        f"status=ok; placeholder_confirmation_tag={tag}-SIM; "
        "note=Use a concrete reference number in your reply when confirming bookings."
    )


def format_tool_block_for_prompt(search_result: str, db_result: str) -> str:
    return (
        "Tool outputs (simulated — use when relevant, do not claim real-time web data):\n"
        f"{search_result}\n{db_result}"
    )


def parse_plan_and_reply(raw: str) -> Tuple[Optional[str], str]:
    """
    Expect format:
        PLAN:
        - ...
        REPLY:
        <dialogue>
    If missing, returns (None, stripped raw).
    """
    if not raw:
        return None, ""
    text = raw.strip()
    m = re.search(r"(?is)PLAN\s*:\s*(.*?)\s*REPLY\s*:\s*(.+)\s*$", text)
    if m:
        plan = m.group(1).strip()
        reply = m.group(2).strip()
        reply = re.sub(r"^(SupportBot|User|Assistant)\s*:\s*", "", reply, flags=re.I).strip()
        return plan or None, reply or text
    return None, text


def build_planning_instruction() -> str:
    return (
        "Before your spoken line, write a short private plan (2–4 bullets), then your message.\n"
        "Use exactly this format:\n"
        "PLAN:\n"
        "- ...\n"
        "REPLY:\n"
        "<only your dialogue line, no role label>\n"
    )


def parse_memory_json(response: str) -> Tuple[List[str], List[str]]:
    """Parse JSON with user_facts and support_facts lists."""
    if not response:
        return [], []
    text = response.strip()
    # Strip markdown code fence if present
    if "```" in text:
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        data = json.loads(text)
        uf = data.get("user_facts") or []
        sf = data.get("support_facts") or []
        if not isinstance(uf, list):
            uf = []
        if not isinstance(sf, list):
            sf = []
        return [str(x).strip() for x in uf if str(x).strip()][:10], [str(x).strip() for x in sf if str(x).strip()][:10]
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug("Memory JSON parse failed: %s", e)
        return [], []


def format_memory_lines(title: str, facts: List[str]) -> str:
    if not facts:
        return ""
    lines = "\n".join(f"- {f}" for f in facts[:10])
    return f"{title}:\n{lines}\n"
