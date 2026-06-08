"""
Core helpers (IDs, text similarity, dialogue validation).

JSON helpers live in ``goalconvo.utils.json_utils``; import via ``goalconvo.utils``.
"""

import hashlib
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_dialogue_id() -> str:
    return str(uuid.uuid4())


def generate_goal_hash(goal: str) -> str:
    return hashlib.md5(goal.encode()).hexdigest()[:8]


def format_conversation_history(turns: List[Dict[str, str]]) -> str:
    history = []
    for turn in turns:
        role = turn.get("role", "Unknown")
        text = turn.get("text", "")
        history.append(f"{role}: {text}")
    return "\n".join(history)


def extract_domain_from_goal(goal: str) -> str:
    goal_lower = goal.lower()
    domain_keywords = {
        "hotel": ["hotel", "accommodation", "room", "booking", "reservation"],
        "restaurant": ["restaurant", "food", "dining", "meal", "cuisine", "eat"],
        "taxi": ["taxi", "cab", "ride", "transport", "pickup"],
        "train": ["train", "railway", "station", "ticket", "journey"],
        "attraction": ["attraction", "sightseeing", "museum", "tour", "visit", "place"],
    }
    for domain, keywords in domain_keywords.items():
        if any(keyword in goal_lower for keyword in keywords):
            return domain
    return "unknown"


def calculate_similarity(text1: str, text2: str) -> float:
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union if union > 0 else 0.0


def detect_repeated_utterances(turns: List[Dict[str, str]], threshold: float = 0.7) -> bool:
    if len(turns) < 2:
        return False
    for i in range(1, len(turns)):
        current_text = turns[i].get("text", "")
        previous_text = turns[i - 1].get("text", "")
        if calculate_similarity(current_text, previous_text) > threshold:
            return True
    return False


def validate_dialogue_format(dialogue: Dict[str, Any]) -> bool:
    required_fields = ["dialogue_id", "goal", "domain", "turns"]
    for field in required_fields:
        if field not in dialogue:
            logger.error("Missing required field: %s", field)
            return False
    if not isinstance(dialogue["turns"], list):
        logger.error("Turns must be a list")
        return False
    for i, turn in enumerate(dialogue["turns"]):
        if not isinstance(turn, dict):
            logger.error("Turn %s must be a dictionary", i)
            return False
        if "role" not in turn or "text" not in turn:
            logger.error("Turn %s missing role or text", i)
            return False
        if turn["role"] not in ["User", "SupportBot"]:
            logger.error("Turn %s has invalid role: %s", i, turn["role"])
            return False
    return True


def create_metadata(
    dialogue_id: str,
    goal: str,
    domain: str,
    quality_score: Optional[float] = None,
    generation_time: Optional[float] = None,
    model_version: str = "mistral-7b",
) -> Dict[str, Any]:
    return {
        "dialogue_id": dialogue_id,
        "goal": goal,
        "domain": domain,
        "quality_score": quality_score,
        "generation_time": generation_time,
        "model_version": model_version,
        "created_at": datetime.now().isoformat(),
        "num_turns": 0,
    }


def update_metadata_turns(metadata: Dict[str, Any], num_turns: int) -> Dict[str, Any]:
    metadata["num_turns"] = num_turns
    return metadata


def get_timestamp() -> str:
    return datetime.now().isoformat()


def truncate_text(text: str, max_length: int = 100) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def clean_text(text: str) -> str:
    return " ".join(text.split())


def is_profane(text: str, profanity_list: Optional[List[str]] = None) -> bool:
    if profanity_list is None:
        profanity_list = ["damn", "hell", "crap", "stupid", "idiot"]
    text_lower = text.lower()
    return any(word in text_lower for word in profanity_list)
