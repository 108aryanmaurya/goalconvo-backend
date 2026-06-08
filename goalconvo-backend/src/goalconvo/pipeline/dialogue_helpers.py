"""Shared helpers for prompt assembly, cleaning, and fallbacks (pipeline + agents)."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..agent_capabilities import (
    format_tool_block_for_prompt,
    tool_db_lookup_stub,
    tool_search_stub,
)
from ..utils import calculate_similarity

logger = logging.getLogger(__name__)


def format_structured_goal(experience_data: Optional[Dict[str, Any]] = None) -> str:
    if not experience_data:
        return ""
    parts = []
    dyn = experience_data.get("dynamic_subgoals")
    if dyn and isinstance(dyn, list):
        parts.append("Dynamic milestones (decomposed goal): " + "; ".join(str(s) for s in dyn))
    subgoals = experience_data.get("subgoals")
    if subgoals and isinstance(subgoals, list):
        parts.append("Subgoals: " + "; ".join(str(s) for s in subgoals))
    constraints = experience_data.get("constraints")
    if constraints and isinstance(constraints, dict):
        parts.append("Constraints: " + ", ".join(f"{k}={v}" for k, v in constraints.items()))
    if not parts:
        return ""
    return "\n".join(parts) + "\n"


def get_domain_grounding(domain: str) -> str:
    hints = {
        "hotel": "Relevant slots: area, price_range, stars, parking, room type. Do not invent specific prices or addresses; use placeholders like 'a mid-range option', 'one of our central hotels', or 'reference number X' when not given.",
        "restaurant": "Relevant slots: area, food type, price range, party size, time. Do not invent specific addresses or phone numbers; use placeholders like 'a restaurant in the centre', 'table for four' when not given.",
        "taxi": "Relevant: pickup location, dropoff, time, car type. When confirming, use placeholders for company name if needed (e.g. 'your taxi at 10:30 from X to Y, reference TAXI-123').",
        "train": "Relevant: destination, leave/arrive time, day. When confirming, include a reference or booking code; use placeholders for specific train IDs if needed.",
        "attraction": "Relevant: type, area. Do not invent specific addresses; use 'in the centre', 'one of our popular attractions' when not given.",
    }
    text = hints.get(domain.lower(), "").strip()
    if not text:
        return ""
    return f"Domain grounding: {text}"


def progress_hint_for_user(goal: str) -> str:
    return (
        "CRITICAL: Express satisfaction only when the assistant has given concrete details (time, place, reference, venue name). "
        "If they only said 'I've arranged it' with no specifics, ask for confirmation instead of thanking. "
        "Do not repeat the same thanking phrase twice; if you already said thanks/perfect, give a single brief closing or one follow-up."
    )


def last_k_turns(history: List[Dict[str, str]], k: int) -> List[Dict[str, str]]:
    conv = [h for h in history if h.get("role") != "System"]
    return conv[-k:] if len(conv) > k else conv


def truncate_prompt(prompt: str, max_words: int, instruction_words: int) -> str:
    words = prompt.split()
    if len(words) <= max_words:
        return prompt
    first_part = " ".join(words[:instruction_words])
    last_part = " ".join(words[-(max_words - instruction_words) :])
    return f"{first_part}... [truncated] ...{last_part}"


def clean_response(response: str, role: str = "User") -> str:
    if not response:
        return ""
    response = response.strip()
    role_prefixes = [
        f"{role}:",
        f"{role.lower()}:",
        f"{role.upper()}:",
        "User:",
        "user:",
        "USER:",
        "SupportBot:",
        "supportbot:",
        "SUPPORTBOT:",
        "System:",
        "system:",
        "SYSTEM:",
        "Assistant:",
        "assistant:",
        "ASSISTANT:",
    ]
    for prefix in role_prefixes:
        if response.startswith(prefix):
            response = response[len(prefix) :].strip()
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1].strip()
    if response.startswith("'") and response.endswith("'"):
        response = response[1:-1].strip()
    lines = response.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if any(line.startswith(prefix) for prefix in role_prefixes):
            continue
        if line.lower() in ["user", "supportbot", "system", "assistant"]:
            continue
        if line:
            cleaned_lines.append(line)
    response = " ".join(cleaned_lines).strip()
    for prefix in role_prefixes:
        response = response.replace(prefix, "").strip()
    if not response:
        return "I need help with this." if role == "User" else "I can help you with that."
    return response


def vary_user_response(original: str, goal: str, domain: str) -> str:
    variations = {
        "I think": "I believe",
        "that could work": "that might work",
        "What's the": "Can you tell me the",
        "Do you have": "Are there any",
        "I need": "I'm looking for",
        "I'd like": "I want",
        "Can I": "Is it possible to",
    }
    varied = original
    for old, new in variations.items():
        if old.lower() in varied.lower():
            varied = varied.replace(old, new)
            break
    if varied == original:
        varied = varied + " Can you help?" if "?" not in varied else "Also, " + varied.lower()
    return varied.strip()


def vary_supportbot_response(original: str, goal: str, domain: str, last_user_msg: str = "") -> str:
    variations = [
        ("I can help", "I'd be happy to help"),
        ("Let me check", "I'll check"),
        ("Would you like", "Would you prefer"),
        ("I can", "I'm able to"),
        ("Great!", "Perfect!"),
        ("Yes, I can", "Absolutely, I can"),
    ]
    varied = original
    for old, new in variations:
        if old in varied:
            varied = varied.replace(old, new)
            break
    if varied == original and last_user_msg:
        varied = "Regarding your question, " + varied.lower() if "?" in last_user_msg else "I understand. " + varied
    return varied.strip()


def venue_from_goal(goal: str, domain: str) -> str:
    if not goal or not goal.strip():
        return "our property" if domain == "hotel" else "that restaurant" if domain == "restaurant" else "that"
    g = goal.strip()
    for prefix in ("book a room at ", "find information about ", "make a reservation at ", "get info on ", "info about "):
        if g.lower().startswith(prefix):
            name = g[len(prefix) :].strip()
            return name.title() if name else ("our property" if domain == "hotel" else "that restaurant")
    if " from " in g.lower():
        part = g.lower().split(" from ", 1)[1].split(" to ")[0].strip()
        return part.title() if part else "your location"
    return g.title()


def supportbot_tool_outputs(
    history: List[Dict[str, str]], domain: str, goal: str, context: str, tools_enabled: bool
) -> Tuple[str, List[str]]:
    if not tools_enabled:
        return "", []
    last_user = ""
    for turn in reversed(history):
        if turn.get("role") == "User":
            last_user = (turn.get("text") or "").strip()
            break
    query = (last_user[:240] if last_user else goal[:240]).strip() or goal[:120]
    search_out = tool_search_stub(query, domain, goal)
    db_out = tool_db_lookup_stub(domain, goal, context)
    block = format_tool_block_for_prompt(search_out, db_out)
    return block, ["search", "db_lookup"]


def last_turn_is_open_request(turns: List[Dict[str, Any]]) -> bool:
    if not turns:
        return False
    last = turns[-1]
    if last.get("role") != "User":
        return False
    text = (last.get("text") or "").strip().lower()
    if not text:
        return False
    open_phrases = (
        "?",
        "please",
        "could you",
        "can you",
        "would you",
        "confirm",
        "provide",
        "let me know",
        "proceed with",
        "check the",
        "check availability",
        "share the",
        "give me",
        "tell me",
        "send me",
        "book ",
        "reserve",
        "availability",
    )
    return any(p in text for p in open_phrases)


def inject_ref_if_booking_claim_has_no_ref(turns: List[Dict[str, Any]], goal: str, domain: str) -> None:
    if not turns:
        return
    for i in range(len(turns) - 1, -1, -1):
        if turns[i].get("role") != "SupportBot":
            continue
        text = (turns[i].get("text") or "").strip()
        if not text:
            continue
        lower = text.lower()
        claim_phrases = (
            "successfully made the reservation",
            "have made the reservation",
            "made the reservation",
            "room is booked",
            "is booked",
            "reservation is confirmed",
            "booking is confirmed",
            "i have arranged",
            "have arranged for a taxi",
            "your taxi is",
            "taxi is all set",
            "secured a table",
            "table for you at",
            "reservation is set",
        )
        ref_phrases = (
            "reference",
            "confirmation number",
            "confirmation #",
            "ref #",
            "ref:",
            "reference number",
            "confirmation code",
            "booking reference",
        )
        if not any(p in lower for p in claim_phrases):
            continue
        if any(p in lower for p in ref_phrases):
            continue
        venue = venue_from_goal(goal, domain)
        if domain == "hotel":
            ref_sentence = f" Your confirmation number is {venue.replace(' ', '')[:8].upper()}-001."
        elif domain == "restaurant":
            ref_sentence = f" Reference: {venue.replace(' ', '')[:6].upper()}-RES."
        elif domain == "taxi":
            ref_sentence = f" Reference: TAXI-{venue.replace(' ', '')[:4].upper()}-001."
        elif domain == "train":
            ref_sentence = f" Reference: TRN-{venue.replace(' ', '')[:4].upper()}-001."
        else:
            ref_sentence = f" Reference: {venue.replace(' ', '')[:6].upper()}-001."
        turns[i]["text"] = text.rstrip() + ref_sentence
        logger.info("Injected reference number into last SupportBot message (dialogue had claimed booking without ref).")
        return


def fallback_user_response(history: List[Dict[str, str]], goal: str, domain: str) -> str:
    if goal and goal.strip():
        g = goal.strip()
        if "train-leaveat:" in g.lower():
            leaveat = g.split("train-leaveat:")[-1].split(";")[0].strip()
            return f"I need to catch a train leaving at {leaveat}. Can you help with that?"
        if "taxi-" in g.lower():
            return "I need to book a taxi. Can you help me with that?"
        if "attraction" in g.lower():
            return "I'm looking for things to do or attractions. Can you help?"
        return f"I still need help with {g}."
    if not history or len(history) < 2:
        return "I need help with this."
    return "I still need a bit more help with this, please."


def fallback_supportbot_response(goal: str, history: List[Dict[str, str]], domain: str = "general") -> str:
    venue = venue_from_goal(goal, domain)
    if not history or len(history) < 2:
        if domain == "hotel":
            return f"Great! I can help you book at {venue}. What dates do you need the room for?"
        if domain == "restaurant":
            return f"I'd be happy to help you with {venue}. How many people and what time are you looking for?"
        return f"I'd be happy to help you with {goal}. How can I assist you?"
    last_user = None
    for turn in reversed(history):
        if turn.get("role") == "User":
            last_user = turn.get("text", "").lower()
            break
    if not last_user:
        return f"I can help you with {goal}. What would you like to know?"
    if any(
        p in last_user
        for p in (
            "confirmation",
            "confirm the",
            "reference number",
            "reference #",
            "booking reference",
            "provide the",
            "let me know the",
            "share the",
        )
    ):
        if domain == "hotel":
            return f"Your booking at {venue} is confirmed. Reference: {venue.replace(' ', '')[:8].upper()}-001. Is there anything else?"
        if domain == "restaurant":
            return f"Your reservation at {venue} is confirmed. Reference: {venue.replace(' ', '')[:6].upper()}-RES. Is there anything else?"
        if domain == "taxi":
            return f"Your taxi with Swift Cabs from {venue} is confirmed for 3:00 PM pickup at the main entrance. Reference: TAXI-{venue.replace(' ', '')[:4].upper()}-001. Is there anything else?"
        if domain == "train":
            return f"Your train booking is confirmed. Reference: TR-{venue.replace(' ', '')[:6].upper()}. Is there anything else?"
        return "Your request is confirmed. Is there anything else I can help with?"
    all_user_text = " ".join([h.get("text", "").lower() for h in history if h.get("role") == "User"])
    all_supportbot_text = " ".join([h.get("text", "").lower() for h in history if h.get("role") == "SupportBot"])
    has_dates_info = any(word in all_user_text for word in ["night", "nights", "2 night", "weekend", "friday", "saturday"])
    has_price_asked = any(word in all_user_text for word in ["price", "cost", "rate", "how much"])
    has_amenity_asked = any(
        word in all_user_text for word in ["wifi", "breakfast", "amenity", "amenities", "include", "what's included", "parking"]
    )
    has_availability_asked = any(word in all_user_text for word in ["available", "availability", "book", "reserve"])
    price_answered = any(word in all_supportbot_text for word in ["£65", "£130", "per night", "total"])
    amenity_answered = any(word in all_supportbot_text for word in ["wifi", "breakfast", "includes"])
    availability_answered = any(word in all_supportbot_text for word in ["available", "book", "reservation"])
    if domain == "hotel":
        if ("price" in last_user or "cost" in last_user or "rate" in last_user or "how much" in last_user) and not price_answered:
            if has_dates_info:
                return f"{venue} is £65 per night. For 2 nights, that's £130 total including WiFi and breakfast. Would you like to proceed with the booking?"
            return f"{venue} offers rooms at £65 per night. How many nights do you need?"
        if ("available" in last_user or "weekend" in last_user) and not availability_answered:
            return f"Yes, I have availability for this weekend at {venue}. The rate is £65 per night. Would you like to book?"
        if (
            "amenity" in last_user
            or "wifi" in last_user
            or "breakfast" in last_user
            or "include" in last_user
            or "what's included" in last_user
        ) and not amenity_answered:
            return f"{venue} includes free WiFi, continental breakfast, and is located in the city center. The rate is £65 per night. Would you like to make a reservation?"
        if ("night" in last_user or "nights" in last_user or "2 night" in last_user) and has_dates_info:
            if not price_answered:
                return f"Perfect! For 2 nights at {venue}, the total is £130 including WiFi and breakfast. Would you like me to proceed with the booking?"
            return f"Great! I can confirm your booking for 2 nights at {venue} for £130. Would you like me to complete the reservation?"
        if "?" in last_user:
            if "what" in last_user and ("amenity" in last_user or "include" in last_user):
                return f"{venue} includes free WiFi, continental breakfast, and city center location. The rate is £65 per night."
            if "what" in last_user and ("price" in last_user or "cost" in last_user):
                return "The rate is £65 per night. For 2 nights, the total is £130 including all amenities."
            if "do you" in last_user or "can you" in last_user:
                if has_dates_info:
                    return f"Yes, I can book a room for 2 nights at {venue} for £130. Shall I proceed?"
                return "Yes, I can help you book. What dates do you need?"
            if has_dates_info and has_price_asked:
                return f"I can confirm your booking for 2 nights at {venue} for £130. Would you like to proceed?"
            if has_dates_info:
                return "For 2 nights, the total is £130 including WiFi and breakfast. Would you like to book?"
            return f"I can help you book at {venue}. What dates do you need?"
        if has_dates_info and price_answered:
            return f"Perfect! I have all the details. Your booking for 2 nights at {venue} for £130 is ready. Would you like me to confirm the reservation?"
        if has_dates_info:
            return "Great! For 2 nights, the total cost is £130 including WiFi and breakfast. Would you like to complete the booking?"
        if price_answered:
            return "The rate is £65 per night. How many nights would you like to book?"
        return f"I can help you book at {venue}. What dates do you need the room for?"
    if domain == "restaurant":
        if "vegetarian" in last_user or "vegan" in last_user or "gluten free" in last_user:
            return (
                f"Yes, {venue} offers vegetarian options on the menu, including mains and sides. "
                "Would you like me to focus on specific dishes or help you with a reservation?"
            )
        if "menu" in last_user:
            return (
                f"{venue} has a varied menu with starters, mains, and desserts. "
                "I can highlight price ranges or vegetarian options if you tell me what you're interested in."
            )
        if "parking" in last_user or "car park" in last_user or "carpark" in last_user:
            return (
                f"{venue} has nearby parking options for guests. "
                "If you tell me your arrival time, I can check what is likely to be available."
            )
        if "price" in last_user or "cost" in last_user or "expensive" in last_user or "cheap" in last_user:
            return f"{venue} typically ranges from £25-50 per person. Would you like me to check availability in your price range?"
        if "available" in last_user or "reservation" in last_user or "book" in last_user:
            return f"I can check availability for {venue}. What date, time, and how many people?"
        if "?" in last_user:
            return (
                f"For {venue}, I can help with details like menu highlights, typical prices, opening hours, "
                "and reservation options. What would you like to know first?"
            )
        return f"I can help you with {venue}. What time are you looking to dine and how many people?"
    if domain == "taxi":
        if "price" in last_user or "fare" in last_user or "cost" in last_user:
            return "The estimated fare to the city center is £25-30. Would you like me to book the taxi?"
        if "time" in last_user or "when" in last_user:
            return "I can arrange pickup in about 10-15 minutes. What's your pickup location?"
        if "?" in last_user:
            return "I can help you book a taxi. Where would you like to go?"
        return "I can arrange a taxi for you. What's your destination and preferred pickup time?"
    if domain == "train":
        if "price" in last_user or "cost" in last_user or "ticket" in last_user:
            return "Train tickets range from £30-45 depending on the service. Express trains are £45, regular trains are £30. Which would you prefer?"
        if "time" in last_user or "schedule" in last_user or "when" in last_user:
            return "Express trains depart at 9 AM and 2 PM. Regular trains have more frequent departures. What time works for you?"
        if "?" in last_user:
            return "I can help you book train tickets. What's your destination and travel date?"
        return "I can help with train bookings. Where are you traveling to and when?"
    if domain == "attraction":
        if "price" in last_user or "cost" in last_user or "ticket" in last_user:
            return "Attraction tickets range from £15-25. The museum is £15, city tours are £25. Which interests you?"
        if "time" in last_user or "open" in last_user or "when" in last_user:
            return "The museum is open 10 AM-6 PM daily. City tours run at 11 AM and 3 PM. Which would you prefer?"
        if "?" in last_user:
            return "I can help you find attractions. What type of activities are you interested in?"
        return "I can help with attractions. Are you interested in museums, tours, or outdoor activities?"
    if "?" in last_user:
        return "Let me check that information for you."
    if "thank" in last_user:
        return "You're welcome! Is there anything else I can help with?"
    if "need" in last_user or "want" in last_user:
        return f"I can help you with that. Let me provide some options for {goal}."
    return f"I understand. Let me help you with {goal}. What specific information do you need?"


def detect_repetition_loop(turns: List[Dict[str, Any]], window: int = 4, threshold: float = 0.45) -> bool:
    thanks_phrases = ("thank you", "thanks", "perfect", "all set", "appreciate", "that's perfect", "that's all i needed")
    closing_phrases = (
        "you're welcome",
        "glad i could",
        "anything else",
        "feel free to let me know",
        "feel free to ask",
        "if you need",
        "further assistance",
        "have a safe trip",
        "need any more",
        "any more questions",
    )
    if len(turns) >= 4:
        last_four = turns[-4:]
        user_texts_4 = [t.get("text", "").strip().lower() for t in last_four if t.get("role") == "User"]
        bot_texts_4 = [t.get("text", "").strip().lower() for t in last_four if t.get("role") == "SupportBot"]
        if len(user_texts_4) >= 1 and len(bot_texts_4) >= 1:
            if all(any(p in t for p in thanks_phrases) or len(t) < 60 for t in user_texts_4):
                if all(any(p in t for p in closing_phrases) for t in bot_texts_4):
                    return True
    if len(turns) >= 6:
        last_six = turns[-6:]
        user_texts = [t.get("text", "").strip().lower() for t in last_six if t.get("role") == "User"]
        bot_texts = [t.get("text", "").strip().lower() for t in last_six if t.get("role") == "SupportBot"]
        if len(user_texts) >= 2 and len(bot_texts) >= 2:
            if all(any(p in t for p in thanks_phrases) or len(t) < 50 for t in user_texts[-2:]):
                if all(any(p in t for p in closing_phrases) for t in bot_texts[-2:]):
                    return True
    if len(turns) < 2 * window:
        return False
    recent = [t.get("text", "") for t in turns[-window:]]
    previous = [t.get("text", "") for t in turns[-(2 * window) : -window]]
    similarities = [calculate_similarity(recent[i], previous[i]) for i in range(window)]
    return sum(similarities) / window >= threshold


def iter_dialogue_turns(max_turns: int, show_progress: bool = False, desc: str = "Dialogue") -> Any:
    """
    Iterate turn indices 1..max_turns, optionally wrapped with tqdm for a progress bar.
    """
    rng = range(1, max_turns + 1)
    if not show_progress:
        return rng
    try:
        from tqdm import tqdm

        return tqdm(rng, desc=desc, unit="turn")
    except ImportError:
        return rng
