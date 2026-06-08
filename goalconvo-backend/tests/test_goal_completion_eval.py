"""Tests for automatic goal completion (rules + optional LLM)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.append(str(Path(__file__).parent.parent / "src"))

from goalconvo.config import Config
from goalconvo.evaluation.goal_completion_eval import GoalCompletionEvaluator


@pytest.fixture
def cfg():
    c = Config()
    c.goal_completion_score_threshold = 0.65
    return c


def _hotel_ok():
    return {
        "domain": "hotel",
        "goal": "Book a hotel for tonight in the city centre with parking",
        "turns": [
            {"role": "User", "text": "I need a hotel tonight centre parking"},
            {"role": "SupportBot", "text": "I can confirm a reservation at Central Inn for 2 nights with parking. Reference HOT-992."},
            {"role": "User", "text": "Thank you, perfect!"},
        ],
    }


def test_hotel_completion_rules(cfg):
    ev = GoalCompletionEvaluator(cfg)
    r = ev.evaluate(_hotel_ok(), use_llm_judge=False)
    assert r["goal_completed"] is True
    assert r["completion_score"] >= cfg.goal_completion_score_threshold
    assert r["missing_requirements"] == []


def test_taxi_missing_confirmation(cfg):
    ev = GoalCompletionEvaluator(cfg)
    d = {
        "domain": "taxi",
        "goal": "Taxi from hotel to airport at 7am",
        "turns": [
            {"role": "User", "text": "I need a taxi from the hotel to the airport at 7am"},
            {"role": "SupportBot", "text": "I can look into options for you."},
            {"role": "User", "text": "Ok thanks"},
        ],
    }
    r = ev.evaluate(d, use_llm_judge=False)
    assert r["goal_completed"] is False
    assert "taxi_booking_confirmation" in r["missing_requirements"]


def test_technical_support_rules(cfg):
    ev = GoalCompletionEvaluator(cfg)
    d = {
        "domain": "technical_support",
        "goal": "Fix wifi login error on laptop",
        "turns": [
            {"role": "User", "text": "Wifi login error on my laptop"},
            {"role": "SupportBot", "text": "Try resetting the network adapter in settings, then restart. If it persists we can escalate."},
            {"role": "User", "text": "That fixed it, thanks!"},
        ],
    }
    r = ev.evaluate(d, use_llm_judge=False)
    assert r["goal_completed"] is True


def test_constraints_surface_missing(cfg):
    ev = GoalCompletionEvaluator(cfg)
    d = {
        "domain": "restaurant",
        "goal": "Book Italian dinner",
        "constraints": {"area": "riverside"},
        "turns": [
            {"role": "User", "text": "Italian dinner please"},
            {"role": "SupportBot", "text": "Confirmed table at Pasta House reference RES-1 for 7pm."},
            {"role": "User", "text": "Great thanks!"},
        ],
    }
    r = ev.evaluate(d, use_llm_judge=False)
    assert any("constraint_area" in m for m in r["missing_requirements"])
    assert r["goal_completed"] is False


def test_llm_judge_merges(cfg):
    ev = GoalCompletionEvaluator(cfg)
    llm = MagicMock()
    llm.generate_completion.return_value = (
        '{"goal_completed": true, "completion_score": 0.9, "missing_requirements": []}'
    )
    r = ev.evaluate(_hotel_ok(), llm_client=llm, use_llm_judge=True)
    assert llm.generate_completion.called
    assert "completion_score" in r
    assert r["goal_completed"] is True
