"""Unit tests for utterance-level reflection (judge JSON, regen loop, scoring rules)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.append(str(Path(__file__).parent.parent / "src"))

from goalconvo.agents.reflection_agent import ReflectionAgent
from goalconvo.agents.base import AgentGenerationResult
from goalconvo.config import Config


@pytest.fixture
def cfg():
    c = Config()
    c.reflection_on_utterances_enabled = True
    c.reflection_max_attempts = 3
    c.reflection_min_accept_score = 4
    return c


def test_extract_json_object_with_markdown_fence():
    raw = 'Here is the verdict:\n```json\n{"accepted": true, "issues": [], "score": 5, "reason": "ok"}\n```'
    out = ReflectionAgent._extract_json_object(raw)
    assert out["accepted"] is True
    assert out["score"] == 5


def test_finalize_acceptance_blocking_hallucination(cfg):
    agent = ReflectionAgent(cfg, MagicMock())
    v = agent._finalize_acceptance(
        {"accepted": True, "issues": ["hallucination"], "score": 4, "reason": "x"}
    )
    assert v["accepted"] is False


def test_finalize_acceptance_passes_at_threshold(cfg):
    agent = ReflectionAgent(cfg, MagicMock())
    v = agent._finalize_acceptance(
        {"accepted": True, "issues": [], "score": 4, "reason": "ok"}
    )
    assert v["accepted"] is True


def test_run_reflected_generation_retries_then_accepts(cfg):
    llm = MagicMock()
    llm.generate_completion.side_effect = [
        '{"accepted": false, "issues": ["repetition"], "score": 2, "reason": "too repetitive"}',
        '{"accepted": true, "issues": [], "score": 5, "reason": "good"}',
    ]
    agent = ReflectionAgent(cfg, llm)
    calls = {"n": 0}

    def gen(hint):
        calls["n"] += 1
        return AgentGenerationResult(text=f"draft-{calls['n']}", metadata={"k": calls["n"]})

    text, meta = agent.run_reflected_generation(
        role="User",
        goal="book hotel",
        dialogue_history=[{"role": "SupportBot", "text": "Hi"}],
        memory_state={"user_facts": []},
        domain="hotel",
        generate_one=gen,
    )
    assert text == "draft-2"
    assert meta["k"] == 2
    assert meta["utterance_reflection"]["exhausted"] is False
    assert len(meta["utterance_reflection"]["attempts"]) == 2


def test_run_reflected_generation_exhausted(cfg):
    cfg.reflection_max_attempts = 2
    llm = MagicMock()
    llm.generate_completion.return_value = '{"accepted": false, "issues": ["coherence"], "score": 2, "reason": "bad"}'
    agent = ReflectionAgent(cfg, llm)

    def gen(_hint=None):
        return AgentGenerationResult(text="same", metadata={})

    text, meta = agent.run_reflected_generation(
        role="SupportBot",
        goal="g",
        dialogue_history=[],
        memory_state={},
        domain="general",
        generate_one=gen,
    )
    assert text == "same"
    assert meta["utterance_reflection"]["exhausted"] is True
    assert len(meta["utterance_reflection"]["attempts"]) == 2
