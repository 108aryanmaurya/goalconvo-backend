"""Unit tests for prompt-only research helpers (no live LLM)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).parent.parent / "src"))

from goalconvo.agents.planner_agent import normalize_structured_plan, format_structured_plan_for_prompt
from goalconvo.memory import DialogueMemory
from goalconvo.research.adaptive import both_planners_near_complete
from goalconvo.research.contradictions import analyze_contradictions
from goalconvo.research.decomposition import run_dynamic_goal_decomposition
from goalconvo.research.json_util import extract_json_dict
from goalconvo.research.planner_augment import structured_planner_research_suffix


def test_normalize_preserves_hierarchy_fields():
    plan = {
        "current_state": "x",
        "identified_slots": [],
        "missing_slots": [],
        "subgoal": "leaf",
        "next_action": "act",
        "goal_progress": "mid",
        "subgoal_hierarchy": [" root ", "leaf"],
        "decomposition_notes": " sync ",
    }
    n = normalize_structured_plan(plan)
    assert n["subgoal_hierarchy"] == ["root", "leaf"]
    assert n["decomposition_notes"] == "sync"


def test_format_structured_plan_includes_hierarchy():
    plan = {
        "current_state": "c",
        "identified_slots": [],
        "missing_slots": [],
        "subgoal": "s",
        "next_action": "n",
        "goal_progress": "early",
        "subgoal_hierarchy": ["A", "B"],
        "decomposition_notes": "note",
    }
    text = format_structured_plan_for_prompt(plan)
    assert "Subgoal hierarchy" in text and "A → B" in text
    assert "Decomposition notes: note" in text


def test_both_planners_near_complete():
    sb = {"structured_planner": {"goal_progress": "near_complete"}}
    u = {"structured_planner": {"goal_progress": "near_complete"}}
    assert both_planners_near_complete(sb, u) is True
    assert both_planners_near_complete(sb, {"structured_planner": {"goal_progress": "mid"}}) is False


def test_planner_augment_suffix_empty_by_default():
    cfg = MagicMock()
    cfg.research_hierarchical_subgoals = False
    cfg.research_dynamic_goal_decomposition = False
    assert structured_planner_research_suffix(cfg) == ""


def test_planner_augment_suffix_when_enabled():
    cfg = MagicMock()
    cfg.research_hierarchical_subgoals = True
    s = structured_planner_research_suffix(cfg)
    assert "subgoal_hierarchy" in s


def test_extract_json_dict():
    assert extract_json_dict('{"a": 1}') == {"a": 1}
    assert extract_json_dict("x") is None


def test_decomposition_parses_milestones():
    llm = MagicMock()
    llm.generate_completion.return_value = json.dumps(
        {"milestones": ["m1", "m2"], "notes": "n"}
    )
    cfg = MagicMock()
    cfg.max_tokens_research_aux = 128
    out = run_dynamic_goal_decomposition(
        llm, cfg, goal="g", context="c", domain="hotel"
    )
    assert out["milestones"] == ["m1", "m2"]
    assert out["notes"] == "n"


def test_contradictions_parses():
    llm = MagicMock()
    llm.generate_completion.return_value = json.dumps(
        {
            "contradictions": [{"description": "time mismatch", "severity": "high"}],
            "repair_hint": "Clarify the time with the user.",
        }
    )
    cfg = MagicMock()
    cfg.max_tokens_research_aux = 128
    out = analyze_contradictions(
        llm, cfg, goal="book", transcript="U: 7pm\nS: 8pm", memory_blob={"user_facts": []}
    )
    assert out["repair_hint"].startswith("Clarify")
    assert out["contradictions"][0]["severity"] == "high"


def test_memory_planner_blob_includes_summary_and_hint():
    m = DialogueMemory()
    m.rolling_dialogue_summary = "User wants Friday."
    m.research_consistency_hint = "Fix venue name."
    blob = m.get_memory_state_for_planner()
    assert "Friday" in blob["rolling_dialogue_summary"]
    assert "venue" in blob["consistency_repair_hint"]
