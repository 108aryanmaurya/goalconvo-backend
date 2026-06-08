"""Unit tests for structured planner JSON parsing and validation."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))  # noqa: E402

from goalconvo.agents.planner_agent import (
    extract_first_json_object,
    validate_structured_plan,
    normalize_structured_plan,
)


def test_extract_json_with_noise():
    raw = (
        'Sure.\n{"current_state": "x", "identified_slots": [], "missing_slots": [], '
        '"subgoal": "s", "next_action": "n", "goal_progress": "p"}\n'
    )
    d = extract_first_json_object(raw)
    assert d is not None
    assert d["current_state"] == "x"


def test_validate_ok():
    plan = {
        "current_state": "negotiating",
        "identified_slots": ["dates: Fri-Sun"],
        "missing_slots": ["confirmation number"],
        "subgoal": "confirm booking",
        "next_action": "Ask for reference",
        "goal_progress": "mid",
    }
    ok, issues = validate_structured_plan(plan)
    assert ok and issues == []


def test_validate_missing_key():
    plan = {"current_state": "x"}
    ok, issues = validate_structured_plan(plan)
    assert not ok
    assert any("missing_key" in i for i in issues)


def test_normalize_coerces_lists():
    plan = {
        "current_state": "a",
        "identified_slots": [1, " b "],
        "missing_slots": None,
        "subgoal": "s",
        "next_action": "n",
        "goal_progress": "p",
    }
    n = normalize_structured_plan(plan)
    assert n["identified_slots"] == ["1", "b"]
    assert n["missing_slots"] == []
