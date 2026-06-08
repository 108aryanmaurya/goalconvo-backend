"""
Integration tests for GoalConvo framework.

Tests the complete pipeline from generation to evaluation.
"""

import pytest
import unittest.mock as mock
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from goalconvo.config import Config
from goalconvo.llm_client import LLMClient
from goalconvo.experience_generator import ExperienceGenerator
from goalconvo.multi_agent_simulator import DialogueSimulator
from goalconvo.quality_judge import QualityJudge
from goalconvo.dataset_store import DatasetStore
from goalconvo.evaluator import Evaluator

class TestGoalConvoIntegration:
    """Integration tests for the complete GoalConvo pipeline."""
    
    def setup_method(self):
        """Setup test configuration."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        self.config = Config()
        self.config.data_dir = self.temp_dir
        self.config.synthetic_dir = f"{self.temp_dir}/synthetic"
        self.config.few_shot_hub_dir = f"{self.temp_dir}/few_shot_hub"
        self.config.multiwoz_dir = f"{self.temp_dir}/multiwoz"
        self.config.domains = ["hotel", "restaurant"]
        self.config.max_turns = 4  # Shorter for testing
        
        # Mock LLM client
        self.mock_llm_client = MagicMock()
        
        # Initialize components
        self.dataset_store = DatasetStore(self.config)
        self.experience_generator = ExperienceGenerator(self.config, self.mock_llm_client, self.dataset_store)
        self.dialogue_simulator = DialogueSimulator(self.config, self.mock_llm_client)
        self.quality_judge = QualityJudge(self.config, self.mock_llm_client)
        self.evaluator = Evaluator(self.config)
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_pipeline_single_dialogue(self):
        """Test complete pipeline with a single dialogue."""
        # Mock LLM responses for experience generation
        self.mock_llm_client.generate_completion.side_effect = [
            '{"goal": "book a hotel room", "context": "Need accommodation for tonight", "first_utterance": "Hi, I need to book a hotel room", "user_persona": "Business traveler"}',
            "I'd be happy to help you book a hotel room. What's your preferred location?",
            "Great! What's your budget range?",
            "Perfect! I have a few options for you. Would you like to see them?",
            "YES"  # Goal satisfaction check
        ]
        
        # Generate experience
        experience_data = self.experience_generator.generate_experience("book a hotel room", "hotel")
        
        assert experience_data["goal"] == "book a hotel room"
        assert experience_data["domain"] == "hotel"
        assert "context" in experience_data
        assert "first_utterance" in experience_data
        
        # Simulate dialogue
        dialogue = self.dialogue_simulator.simulate_dialogue(experience_data)
        
        assert "dialogue_id" in dialogue
        assert dialogue["goal"] == "book a hotel room"
        assert dialogue["domain"] == "hotel"
        assert "turns" in dialogue
        assert len(dialogue["turns"]) > 0
        
        # Judge quality
        quality_assessment = self.quality_judge.judge_dialogue(dialogue)
        
        assert "passed_filters" in quality_assessment
        assert "overall_score" in quality_assessment
        
        # Save dialogue
        dialogue_id = self.dataset_store.save_dialogue(dialogue)
        
        assert dialogue_id == dialogue["dialogue_id"]
        
        # Load dialogue back
        loaded_dialogue = self.dataset_store.load_dialogue(dialogue_id, "hotel")
        
        assert loaded_dialogue is not None
        assert loaded_dialogue["dialogue_id"] == dialogue_id
    
    def test_batch_generation_pipeline(self):
        """Test batch generation pipeline."""
        # Mock LLM responses for multiple dialogues
        responses = []
        for i in range(3):  # 3 dialogues
            responses.extend([
                f'{{"goal": "book hotel {i}", "context": "Test context {i}", "first_utterance": "Test utterance {i}", "user_persona": "Test user {i}"}}',
                f"SupportBot response {i}",
                f"User response {i}",
                "YES"  # Goal satisfaction
            ])
        
        self.mock_llm_client.generate_completion.side_effect = responses
        
        # Generate multiple dialogues
        dialogues = []
        for i in range(3):
            experience_data = self.experience_generator.generate_experience(f"book hotel {i}", "hotel")
            dialogue = self.dialogue_simulator.simulate_dialogue(experience_data)
            dialogues.append(dialogue)
        
        assert len(dialogues) == 3
        
        # Filter dialogues
        accepted, rejected = self.quality_judge.filter_dialogues(dialogues)
        
        # Should have some accepted dialogues
        assert len(accepted) + len(rejected) == 3
        
        # Save accepted dialogues
        for dialogue in accepted:
            self.dataset_store.save_dialogue(dialogue)
        
        # Check statistics
        stats = self.dataset_store.get_statistics()
        assert stats["total_dialogues"] == len(accepted)
    
    def test_few_shot_hub_update(self):
        """Test few-shot hub update mechanism."""
        # Create some high-quality dialogues
        high_quality_dialogue = {
            "dialogue_id": "high_quality_1",
            "goal": "book a luxury hotel room",
            "domain": "hotel",
            "turns": [
                {"role": "User", "text": "I need a luxury hotel room"},
                {"role": "SupportBot", "text": "I can help you find a luxury hotel"},
                {"role": "User", "text": "Perfect, thank you!"}
            ],
            "metadata": {"quality_score": 0.9}
        }
        
        low_quality_dialogue = {
            "dialogue_id": "low_quality_1",
            "goal": "book a hotel room",
            "domain": "hotel",
            "turns": [
                {"role": "User", "text": "Hotel"},
                {"role": "SupportBot", "text": "Ok"}
            ],
            "metadata": {"quality_score": 0.3}
        }
        
        # Save dialogues
        self.dataset_store.save_dialogue(high_quality_dialogue)
        self.dataset_store.save_dialogue(low_quality_dialogue)
        
        # Update few-shot hub
        added_count = self.dataset_store.update_few_shot_hub(top_percentage=0.5)
        
        # Should add the high-quality dialogue
        assert added_count == 1
        
        # Load few-shot examples
        examples = self.dataset_store.load_few_shot_examples("hotel", num_examples=5)
        
        assert len(examples) == 1
        assert examples[0]["dialogue_id"] == "high_quality_1"
    
    def test_evaluation_pipeline(self):
        """Test evaluation pipeline with synthetic and real data."""
        # Create synthetic dialogues
        synthetic_dialogues = [
            {
                "dialogue_id": "synth_1",
                "goal": "book a hotel room",
                "domain": "hotel",
                "turns": [
                    {"role": "User", "text": "I need a hotel room"},
                    {"role": "SupportBot", "text": "I can help with that"},
                    {"role": "User", "text": "Thank you!"}
                ]
            }
        ]
        
        # Create mock real dialogues (MultiWOZ format)
        real_dialogues = [
            {
                "dialogue_id": "real_1",
                "goal": "book a hotel room",
                "domain": "hotel",
                "turns": [
                    {"role": "User", "text": "I need to book a hotel"},
                    {"role": "SupportBot", "text": "I can help you book a hotel"},
                    {"role": "User", "text": "Great, thanks!"}
                ]
            }
        ]
        
        # Run evaluation
        with patch("goalconvo.evaluation.bertscore_eval.bert_score") as mock_bert_score:
            # Mock BERTScore response
            mock_bert_score.return_value = ([0.8], [0.8], [0.8])
            
            results = self.evaluator.evaluate_synthetic_vs_real(synthetic_dialogues, real_dialogues)
            
            assert "semantic_similarity" in results
            assert "diversity_metrics" in results
            assert "goal_relevance" in results
            assert "domain_analysis" in results
            assert "statistical_analysis" in results
    
    def test_error_handling(self):
        """Test error handling in the pipeline."""
        # Mock LLM client to raise errors
        self.mock_llm_client.generate_completion.side_effect = Exception("API Error")
        
        # Experience generation should handle errors gracefully
        experience_data = self.experience_generator.generate_experience("test goal", "hotel")
        
        # Should return fallback experience
        assert experience_data["goal"] == "test goal"
        assert experience_data["domain"] == "hotel"
        assert "context" in experience_data
        assert "first_utterance" in experience_data
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test invalid configuration
        invalid_config = Config()
        invalid_config.temperature = 5.0  # Invalid temperature
        
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            invalid_config.__post_init__()
        
        # Test missing API key
        no_key_config = Config()
        no_key_config.mistral_api_key = ""
        no_key_config.openai_api_key = ""
        
        with pytest.raises(ValueError, match="Either MISTRAL_API_KEY or OPENAI_API_KEY must be set"):
            no_key_config.__post_init__()

if __name__ == "__main__":
    pytest.main([__file__])
