"""
Macro-vertical guidance (healthcare, booking, customer support, education).

Simulator ``domain`` strings (hotel, technical_support, …) map to a vertical for
extra safety and style instructions layered into prompts.
"""

from __future__ import annotations

from typing import Dict

# Rich addenda injected into planner / user / support when vertical != general.
VERTICAL_ADDENDA: Dict[str, str] = {
    "healthcare": """
Vertical: **Healthcare / clinical-adjacent support**
- Use non-diagnostic language unless the scenario explicitly positions you as a licensed clinician (default: you are NOT diagnosing).
- Encourage appropriate in-person or emergency care when red-flag symptoms appear; never give medication dosing as definitive medical advice.
- Protect privacy: do not ask for unnecessary PHI; reference only details the user already shared.
- Prefer evidence-aligned general education, care navigation, scheduling, billing, or portal help over speculative treatment plans.
""".strip(),
    "booking": """
Vertical: **Bookings & reservations (travel, dining, mobility, events)**
- Separate **availability intent** from **confirmed booking**; only claim confirmation when the transcript or tool line supports it.
- Always surface key constraints the user cares about (dates, party size, accessibility, fare rules) before closing.
- Use concrete but grounded details: times, areas, confirmation-style references consistent with prior turns or explicit placeholders.
""".strip(),
    "customer_support": """
Vertical: **Customer / technical support**
- Troubleshoot in ordered steps; acknowledge impact; avoid blaming the user.
- Do not invent ticket IDs, internal team names, or SLA guarantees unless provided.
- For account/security topics, prefer safe verification flows and official channels over credential collection in chat.
""".strip(),
    "education": """
Vertical: **Education & tutoring**
- Teach stepwise: check understanding, scaffold, then optionally extend — stay at the learner's stated level.
- Do not fabricate citations, syllabus items, or institutional policies; distinguish general pedagogy from institution-specific rules.
- Academic integrity: do not complete graded assessments for the learner; guide with hints and analogous practice instead when asked.
""".strip(),
    "general": """
Vertical: **General task-oriented support**
- Stay within the user goal and domain; prefer clarity and safe defaults over speculative richness.
""".strip(),
}


def map_domain_to_vertical(domain: str) -> str:
    """Map a simulator domain label to a macro vertical key."""
    d = (domain or "").strip().lower().replace(" ", "_").replace("-", "_")

    healthcare_aliases = frozenset(
        {
            "healthcare",
            "health",
            "medical",
            "clinical",
            "clinic",
            "pharmacy",
            "patient",
            "telehealth",
        }
    )
    education_aliases = frozenset(
        {
            "education",
            "edtech",
            "tutoring",
            "tutor",
            "course",
            "school",
            "university",
            "learning",
            "student",
        }
    )
    support_aliases = frozenset(
        {
            "technical_support",
            "tech_support",
            "it_support",
            "customer_support",
            "support",
            "helpdesk",
            "billing_support",
        }
    )
    booking_aliases = frozenset(
        {
            "hotel",
            "restaurant",
            "taxi",
            "train",
            "attraction",
            "booking",
            "flight",
            "car_rental",
            "event",
        }
    )

    if d in healthcare_aliases:
        return "healthcare"
    if d in education_aliases:
        return "education"
    if d in support_aliases:
        return "customer_support"
    if d in booking_aliases:
        return "booking"
    return "general"


def vertical_guidance_for_domain(domain: str) -> str:
    """Return macro-vertical instructions, or empty string for unrecognized / general domains."""
    v = map_domain_to_vertical(domain)
    if v == "general":
        return ""
    return VERTICAL_ADDENDA.get(v, "")
