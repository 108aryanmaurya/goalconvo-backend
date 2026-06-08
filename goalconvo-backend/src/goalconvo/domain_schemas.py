"""
Per-domain schemas for SupportBot grounding.

Instructs the model to use only these slots/options and not invent
prices, addresses, or names. Reduces hallucination and keeps answers consistent.
"""

from typing import Dict, Any

# Slots and allowed-style options per domain (placeholder wording for natural replies)
DOMAIN_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "hotel": {
        "slots": [
            "area (e.g. city centre, north, near station)",
            "price_range (budget, moderate, expensive)",
            "stars (e.g. 3-star, 4-star)",
            "parking (yes/no)",
            "internet (yes/no)",
            "number of nights",
            "number of guests",
            "check-in/check-out",
        ],
        "instruction": (
            "Only give information that fits the hotel domain. "
            "Do not invent specific prices, addresses, or hotel names. "
            "Use placeholders like 'a mid-range option', 'one of our central hotels', "
            "'we have options in your budget', or 'availability for your dates'."
        ),
    },
    "restaurant": {
        "slots": [
            "area (e.g. city centre, near the hotel)",
            "food_type / cuisine (e.g. Italian, Indian, vegetarian)",
            "price_range (budget, moderate, expensive)",
            "number of people",
            "time (lunch, dinner, specific time)",
        ],
        "instruction": (
            "Only give information that fits the restaurant domain. "
            "Do not invent specific restaurant names, addresses, or prices. "
            "Use placeholders like 'several Italian options in the centre', "
            "'restaurants that can seat your party', or 'availability around that time'."
        ),
    },
    "taxi": {
        "slots": [
            "departure (address or place)",
            "destination (address or place)",
            "time (pickup time)",
            "number of passengers",
        ],
        "instruction": (
            "Only give information that fits the taxi/ride domain. "
            "Do not invent specific addresses or prices. "
            "Use placeholders like 'pickup from your location', 'fare to the airport', "
            "or 'availability at that time'."
        ),
    },
    "train": {
        "slots": [
            "departure (station or city)",
            "destination (station or city)",
            "date / time",
            "number of passengers",
        ],
        "instruction": (
            "Only give information that fits the train domain. "
            "Do not invent specific train numbers, times, or prices. "
            "Use placeholders like 'trains running on that route', 'departure times', "
            "or 'ticket options for your journey'."
        ),
    },
    "attraction": {
        "slots": [
            "area (e.g. city centre)",
            "type (museum, tour, park, etc.)",
            "opening hours / date",
        ],
        "instruction": (
            "Only give information that fits the attraction/tourism domain. "
            "Do not invent specific attraction names, addresses, or ticket prices. "
            "Use placeholders like 'popular attractions in the area', "
            "'opening times', or 'tickets can be booked'."
        ),
    },
    "technical_support": {
        "slots": [
            "problem / symptom (e.g. login error, slow wifi, crash)",
            "diagnostic or troubleshooting steps",
            "resolution or workaround (or escalation path)",
            "user acknowledgment or confirmation that the issue is addressed",
        ],
        "instruction": (
            "Stay factual; do not invent product versions or internal ticket IDs. "
            "Give clear steps, links or settings paths as placeholders when specifics are unknown."
        ),
    },
}

# Fallback for unknown domains
DEFAULT_SCHEMA = {
    "slots": ["relevant details the user asked for"],
    "instruction": (
        "Stay on topic. Do not invent specific names, prices, or addresses. "
        "Use general placeholders when giving information."
    ),
}


def get_domain_schema_text(domain: str) -> str:
    """Return a short text block to inject into the SupportBot prompt for domain grounding."""
    d = DOMAIN_SCHEMAS.get(domain.lower(), DEFAULT_SCHEMA)
    slots = d.get("slots", [])
    instr = d.get("instruction", DEFAULT_SCHEMA["instruction"])
    slots_text = "; ".join(slots) if isinstance(slots, list) else str(slots)
    return (
        f"Slots you may use when giving information: {slots_text}. "
        f"{instr}"
    )
