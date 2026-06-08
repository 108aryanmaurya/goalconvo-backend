"""
Tests for Multi-Agent Simulator module.
"""

import pytest
import unittest.mock as mock
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from goalconvo.config import Config
from goalconvo.multi_agent_simulator import DialogueSimulator
from goalconvo.llm_client import LLMClient

class TestDialogueSimulator:
    """Test cases for Dialogue Simulator."""
    
    def setup_method(self):
        """Setup test configuration."""
        self.config = Config()
        self.config.max_turns = 5
        self.config.temperature = 0.7
        self.config.top_p = 0.9
        # Avoid extra planner LLM calls in unit tests (mock side_effect length).
        self.config.structured_planner_enabled = False
        self.config.agent_memory_enabled = False
        self.config.reflection_on_utterances_enabled = False

        # Mock LLM client
        self.mock_llm_client = MagicMock()
        self.mock_llm_client.api_config = {"provider": "test", "model": "mock-model"}
        
        self.simulator = DialogueSimulator(self.config, self.mock_llm_client)
    
    def test_simulate_dialogue_basic(self):
        """Test basic dialogue simulation."""
        experience_data = {
            "goal": "Book a hotel room",
            "domain": "hotel",
            "context": "Need accommodation for tonight",
            "first_utterance": "Hi, I need to book a hotel room",
            "user_persona": "Business traveler"
        }
        
        # Mock LLM responses
        self.mock_llm_client.generate_completion.side_effect = [
            "I'd be happy to help you book a hotel room. What's your preferred location?",
            "Great! What's your budget range?",
            "Perfect! I have a few options for you. Would you like to see them?"
        ]
        
        result = self.simulator.simulate_dialogue(experience_data)

        assert "pipeline_turns" in result
        assert len(result["pipeline_turns"]) >= 1
        assert "stop_reason" in result["metadata"]
        assert "reproducibility" in result["metadata"]
        assert result["metadata"]["reproducibility"].get("llm_model") == "mock-model"
        assert result["goal"] == "Book a hotel room"
        assert result["domain"] == "hotel"
        assert "turns" in result
        assert len(result["turns"]) > 0
        assert all("role" in turn for turn in result["turns"])
        assert all("text" in turn for turn in result["turns"])
    
    def test_simulate_dialogue_max_turns(self):
        """Test dialogue simulation with max turns reached."""
        self.config.min_turns = 4  # allow early cap when max_turns is small
        experience_data = {
            "goal": "Test goal",
            "domain": "hotel",
            "context": "Test context",
            "first_utterance": "Test utterance"
        }
        
        # Mock LLM responses (will hit max turns)
        self.mock_llm_client.generate_completion.return_value = "Test response"
        
        result = self.simulator.simulate_dialogue(experience_data, max_turns=2)
        
        # 1 initial user + max_turns pairs of (SupportBot, User) = 1 + 2*2 = 5
        assert len(result["turns"]) <= 5
        assert result["metadata"]["max_turns_reached"] is True
        assert result["metadata"]["stop_reason"] == "max_turns"

    def test_export_dialogue_json(self, tmp_path):
        """Round-trip export of dialogue JSON including pipeline_turns."""
        import json as jsonlib

        from goalconvo.pipeline import export_dialogue_json

        experience_data = {
            "goal": "X",
            "domain": "hotel",
            "context": "Y",
            "first_utterance": "Hello",
        }
        self.mock_llm_client.generate_completion.return_value = "Bot reply"
        self.config.min_turns = 4
        self.config.max_turns = 1
        result = self.simulator.simulate_dialogue(experience_data)
        out = tmp_path / "dialogue.json"
        export_dialogue_json(result, str(out))
        loaded = jsonlib.loads(out.read_text(encoding="utf-8"))
        assert loaded["dialogue_id"] == result["dialogue_id"]
        assert loaded["pipeline_turns"] == result["pipeline_turns"]

    def test_check_goal_satisfied_yes(self):
        """Test goal satisfaction check with YES response."""
        self.config.min_turns = 4
        goal = "Book a hotel room"
        history = [
            {"role": "User", "text": "I need a hotel room"},
            {"role": "SupportBot", "text": "I can help with that"},
            {"role": "User", "text": "I'd like two nights in the centre"},
            {"role": "SupportBot", "text": "Your booking at the Grand Hotel is confirmed. Reference: GH-001."},
            {"role": "User", "text": "Thank you, that's perfect!"},
        ]
        
        # Mock LLM response indicating goal satisfaction
        self.mock_llm_client.generate_completion.return_value = "YES"
        
        result = self.simulator._check_goal_satisfied(goal, history)
        
        assert result is True
    
    def test_check_goal_satisfied_no(self):
        """Test goal satisfaction check with NO response."""
        self.config.min_turns = 4
        goal = "Book a hotel room"
        history = [
            {"role": "User", "text": "I need a hotel room"},
            {"role": "SupportBot", "text": "I can help with that"}
        ]
        
        # Mock LLM response indicating goal not satisfied
        self.mock_llm_client.generate_completion.return_value = "NO"
        
        result = self.simulator._check_goal_satisfied(goal, history)
        
        assert result is False
    
    def test_check_completion_keywords(self):
        """Test keyword-based goal completion check requires both user satisfaction and assistant-side concrete evidence."""
        goal = "Book a hotel room"
        # User thanks alone is not enough; need assistant to have given concrete completion detail
        history_no_evidence = [
            {"role": "User", "text": "I need a hotel room"},
            {"role": "SupportBot", "text": "I can help with that"},
            {"role": "User", "text": "Thank you, that's perfect!"}
        ]
        result = self.simulator._check_completion_keywords(goal, history_no_evidence)
        assert result is False

        # With assistant-side completion evidence (e.g. "confirmed", "booking"), returns True
        history_with_evidence = [
            {"role": "User", "text": "I need a hotel room"},
            {"role": "SupportBot", "text": "Your booking at the Grand Hotel is confirmed for 2 nights."},
            {"role": "User", "text": "Thank you, that's perfect!"}
        ]
        result = self.simulator._check_completion_keywords(goal, history_with_evidence)
        assert result is True
    
    def test_simulate_batch_dialogues(self):
        """Test batch dialogue simulation."""
        experience_data_list = [
            {
                "goal": "Book a hotel room",
                "domain": "hotel",
                "context": "Test context 1",
                "first_utterance": "Test utterance 1"
            },
            {
                "goal": "Find a restaurant",
                "domain": "restaurant", 
                "context": "Test context 2",
                "first_utterance": "Test utterance 2"
            }
        ]
        
        # Mock LLM responses
        self.mock_llm_client.generate_completion.return_value = "Test response"
        
        results = self.simulator.simulate_batch_dialogues(experience_data_list)
        
        assert len(results) == 2
        assert all("dialogue_id" in result for result in results)
        assert all("turns" in result for result in results)

if __name__ == "__main__":
    pytest.main([__file__])
