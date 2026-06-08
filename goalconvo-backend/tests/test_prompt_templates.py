"""Sanity checks for modular prompt assembly."""

from goalconvo.prompts import (
    STRUCTURED_DIALOGUE_PLANNER_PROMPTS,
    map_domain_to_vertical,
    vertical_guidance_for_domain,
)
from goalconvo.prompts.reflection_prompts import REFLECTION_PROMPTS


def test_structured_planner_template_has_vertical_placeholder():
    s = STRUCTURED_DIALOGUE_PLANNER_PROMPTS["user"]
    assert "{vertical_guidance}" in s
    text = s.format(
        domain="hotel",
        vertical_guidance="test-vertical-block",
        goal="g",
        context="c",
        dialogue_state_json="{}",
        memory_state_json="{}",
        history="(none)",
    )
    assert "test-vertical-block" in text


def test_vertical_mapping_booking_and_healthcare():
    assert map_domain_to_vertical("hotel") == "booking"
    assert map_domain_to_vertical("healthcare") == "healthcare"
    assert map_domain_to_vertical("education") == "education"
    assert map_domain_to_vertical("technical_support") == "customer_support"


def test_vertical_guidance_empty_for_unknown_domain():
    assert vertical_guidance_for_domain("unknown_xyz") == ""


def test_reflection_critique_format_roundtrip():
    p = REFLECTION_PROMPTS["response_critique"].format(
        role="User",
        goal="book taxi",
        domain="taxi",
        history="U: hi",
        memory_json="{}",
        generated_response="Thanks!",
    )
    assert "User" in p
    assert '{"accepted": true' in p
