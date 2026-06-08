"""Tests for DialogueMemory state tracking."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).parent.parent / "src"))

from goalconvo.memory import DialogueMemory


def test_append_history_dedupes_consecutive_duplicate():
    m = DialogueMemory()
    t = {"role": "User", "text": "Hi", "timestamp": "1"}
    m.update_memory(recent_turns=[t])
    m.update_memory(recent_turns=[t])
    assert len(m.dialogue_history) == 1


def test_clear_memory():
    m = DialogueMemory()
    m.entities["x"] = 1
    m.slots["restaurant"] = {"people": 4}
    m.dialogue_history.append({"role": "User", "text": "a", "timestamp": None, "metadata": {}})
    m.clear_memory()
    assert m.entities == {} and m.slots == {} and m.dialogue_history == []
    assert m.rolling_dialogue_summary == "" and m.research_consistency_hint == ""


def test_serialization_roundtrip():
    m = DialogueMemory()
    m.slots["restaurant"] = {"venue": "Spice Garden", "people": 4, "status": "pending"}
    m.completed_subtasks.append("chose venue")
    raw = m.to_json()
    m2 = DialogueMemory.from_json(raw)
    assert m2.get_slots("restaurant") == m.get_slots("restaurant")


def test_get_context_summary_includes_slots():
    m = DialogueMemory()
    m.slots["restaurant"] = {"restaurant": "Spice Garden", "people": 4}
    s = m.get_context_summary(max_turns=0)
    assert "Spice Garden" in s and "Slots[restaurant]" in s


def test_conflict_logged_on_slot_change():
    m = DialogueMemory()
    m._merge_slots_domain("hotel", {"date": "Fri"})
    m._merge_slots_domain("hotel", {"date": "Sat"})
    assert m.get_slots("hotel")["date"] == "Sat"
    assert len(m.slot_conflicts) == 1


def test_update_memory_llm_merge():
    m = DialogueMemory()
    cfg = MagicMock()
    cfg.max_tokens_memory_refresh = 256
    llm = MagicMock()
    llm.generate_completion.return_value = json.dumps(
        {
            "user_facts": ["prefers window seat"],
            "support_facts": [],
            "entities": {"venue": "Spice Garden"},
            "slots": {"restaurant": {"people": 4, "booking_time": "7 PM", "status": "pending"}},
            "completed_subtasks": [],
            "unresolved_goals": ["need confirmation number"],
            "user_preferences": ["vegetarian"],
        }
    )
    m.update_memory(
        recent_turns=[{"role": "User", "text": "Table for 4 at 7pm", "timestamp": None}],
        llm_client=llm,
        config=cfg,
        domain="restaurant",
        goal="Book dinner",
    )
    assert m.get_entities().get("venue") == "Spice Garden"
    assert m.get_slots("restaurant").get("people") == 4
    assert "vegetarian" in m.get_user_preferences()
