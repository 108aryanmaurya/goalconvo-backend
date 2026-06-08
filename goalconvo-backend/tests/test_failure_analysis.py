"""Tests for goalconvo.analysis.failure_analysis."""

from __future__ import annotations

import json
from pathlib import Path

from goalconvo.analysis.failure_analysis import (
    EARLY_TASK_TERMINATION,
    FORGOTTEN_SLOTS,
    GOAL_DRIFT,
    HALLUCINATED_ENTITIES,
    INCOHERENT_TRANSITIONS,
    REPEATED_QUESTIONS,
    FailureAnalyzer,
    FailureCorpusAnalyzer,
    export_failure_bundle,
)


def _base(**kwargs):
    d = {
        "dialogue_id": "d1",
        "goal": "Book a moderate hotel in the centre for two nights.",
        "domain": "hotel",
        "turns": [],
    }
    d.update(kwargs)
    return d


def test_repeated_questions_detected():
    fa = FailureAnalyzer()
    d = _base(
        turns=[
            {"role": "User", "text": "Can you find a hotel near the station with wifi?"},
            {"role": "SupportBot", "text": "I can help; do you have dates in mind?"},
            {"role": "User", "text": "Can you find a hotel near the station with wifi please?"},
            {"role": "SupportBot", "text": "Sure, what dates?"},
        ]
    )
    r = fa.analyze(d)
    assert REPEATED_QUESTIONS in r["failure_categories"]


def test_hallucinated_entity_detected():
    fa = FailureAnalyzer()
    d = _base(
        turns=[
            {"role": "User", "text": "I need a hotel downtown."},
            {"role": "SupportBot", "text": "We have availability at the Grand Plaza Hotel Paris for you."},
        ]
    )
    r = fa.analyze(d)
    assert HALLUCINATED_ENTITIES in r["failure_categories"]


def test_forgotten_slots_parking_in_goal_not_in_dialogue():
    fa = FailureAnalyzer()
    d = _base(
        goal="Book a hotel with parking for Friday night.",
        turns=[
            {"role": "User", "text": "I need a hotel for Friday."},
            {"role": "SupportBot", "text": "We have central options for that night."},
            {"role": "User", "text": "What about wifi?"},
            {"role": "SupportBot", "text": "Most properties offer wifi."},
        ],
    )
    r = fa.analyze(d)
    assert FORGOTTEN_SLOTS in r["failure_categories"]
    assert "parking" in r["categories"][FORGOTTEN_SLOTS]["detail"]


def test_incoherent_transitions():
    fa = FailureAnalyzer()
    d = _base(
        turns=[
            {"role": "User", "text": "I need a hotel booking for tomorrow evening please."},
            {"role": "SupportBot", "text": "Quantum chromodynamics explains quark confinement at low energy."},
        ]
    )
    r = fa.analyze(d)
    assert INCOHERENT_TRANSITIONS in r["failure_categories"]


def test_early_task_termination():
    fa = FailureAnalyzer()
    d = _base(
        turns=[
            {"role": "User", "text": "Book a hotel with parking for two nights."},
            {"role": "SupportBot", "text": "I can look at options; what area do you prefer?"},
            {"role": "User", "text": "Thanks anyway, bye."},
        ]
    )
    r = fa.analyze(d)
    assert EARLY_TASK_TERMINATION in r["failure_categories"]


def test_goal_drift_synthetic():
    fa = FailureAnalyzer()
    goal = "reserve central london hotel three nights wifi parking two guests tomorrow"
    # Opening mirrors goal; second half unrelated (drops goal tokens)
    t_open = [
        {"role": "User", "text": goal},
        {"role": "SupportBot", "text": "I can help with central london hotel three nights wifi parking two guests."},
    ]
    noise = [
        {"role": "User", "text": "Actually tell me about pizza toppings and jazz history instead."},
        {"role": "SupportBot", "text": "Mozzarella pairs well with basil; jazz evolved from blues in new orleans."},
        {"role": "User", "text": "Interesting, what about saxophone players?"},
        {"role": "SupportBot", "text": "Charlie parker and john coltrane are influential saxophonists."},
    ]
    d = _base(goal=goal, turns=t_open + noise)
    r = fa.analyze(d)
    assert GOAL_DRIFT in r["failure_categories"]


def test_corpus_aggregate_and_export(tmp_path: Path):
    dialogues = [
        _base(dialogue_id="ok", turns=[{"role": "User", "text": "Hi"}, {"role": "SupportBot", "text": "Hello"}]),
        _base(
            dialogue_id="bad_rep",
            turns=[
                {"role": "User", "text": "Need a taxi to airport tomorrow morning please help"},
                {"role": "SupportBot", "text": "What time?"},
                {"role": "User", "text": "Need a taxi to airport tomorrow morning please help"},
            ],
        ),
    ]
    agg = FailureCorpusAnalyzer().analyze(dialogues)
    assert agg["total_dialogues"] == 2
    assert agg["failed_dialogues"] >= 1
    paths = export_failure_bundle(tmp_path, dialogues, agg)
    assert Path(paths["json"]).is_file()
    assert Path(paths["csv"]).is_file()
    assert Path(paths["markdown"]).is_file()
    data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert "aggregate" in data and "per_dialogue" in data
    failed_files = list((tmp_path / "failed_dialogues").glob("*.json"))
    assert len(failed_files) >= 1
