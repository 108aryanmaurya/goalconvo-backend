#!/usr/bin/env python3
"""
Main Generation Pipeline for GoalConvo Framework

Generates synthetic goal-oriented dialogues using multi-agent simulation.
"""

import json
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from goalconvo.config import Config
from goalconvo.llm_client import LLMClient
from goalconvo.experience_generator import ExperienceGenerator
from goalconvo.multi_agent_simulator import DialogueSimulator
from goalconvo.quality_judge import QualityJudge
from goalconvo.dataset_store import DatasetStore
from goalconvo.dataset_versioning import DatasetVersionManager
from goalconvo.utils import load_json, save_json, ensure_dir

logger = logging.getLogger(__name__)

class GoalConvoGenerator:
    """Main generator class that orchestrates the entire pipeline."""
    
    def __init__(self, config: Config):
        """Initialize the generator with all components."""
        self.config = config
        
        # Initialize components
        self.generation_llm_client = LLMClient(config, api_config=config.get_generation_api_config())
        self.evaluation_llm_client = LLMClient(config, api_config=config.get_evaluation_api_config())
        # Backwards-compatible alias for places that expect a single client.
        self.llm_client = self.generation_llm_client
        self.dataset_store = DatasetStore(config)
        self.experience_generator = ExperienceGenerator(config, self.generation_llm_client, self.dataset_store)
        self.dialogue_simulator = DialogueSimulator(config, self.generation_llm_client)
        self.quality_judge = QualityJudge(config, self.evaluation_llm_client)
        
        # Generation statistics
        self.stats = {
            "total_generated": 0,
            "total_accepted": 0,
            "total_rejected": 0,
            "by_domain": {},
            "start_time": None,
            "end_time": None
        }
    
    def generate_dialogues(
        self, 
        num_dialogues: int, 
        domains: Optional[List[str]] = None,
        resume: bool = False,
        emit_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate synthetic dialogues using the complete pipeline.
        
        Args:
            num_dialogues: Number of dialogues to generate
            domains: List of domains to generate for (None for all)
            resume: Whether to resume from existing progress
            overrides: Optional ablation/experiment overrides: quality_judge (bool), few_shot_examples (int)
            
        Returns:
            Generation statistics and results
        """
        logger.info(f"Starting generation of {num_dialogues} dialogues")
        self.stats["start_time"] = datetime.now()
        
        # Emit start event
        if emit_callback:
            emit_callback('pipeline_start', {
                'message': f'Starting pipeline for {num_dialogues} dialogues',
                'num_dialogues': num_dialogues,
                'timestamp': datetime.now().isoformat()
            })
        
        # Determine domains to use
        target_domains = domains or self.config.domains
        logger.info(f"Target domains: {target_domains}")
        
        # Load existing progress if resuming
        if resume:
            self._load_generation_progress()
        
        # Generate dialogues
        generated_count = 0
        accepted_count = 0
        run_accepted_dialogues = []  # This run's accepted dialogues (for evaluation)

        # Calculate dialogues per domain
        num_domains = len(target_domains)
        if num_domains == 0:
            logger.error("No domains specified for generation")
            return self.stats
        
        # Distribute exactly num_dialogues across domains (no extra dialogues)
        dialogues_per_domain = num_dialogues // num_domains if num_dialogues > 0 else 0
        remainder = num_dialogues % num_domains if num_dialogues > 0 else 0
        
        logger.info(f"Generating {num_dialogues} dialogues across {num_domains} domain(s)")
        logger.info(f"Base dialogues per domain: {dialogues_per_domain}, remainder: {remainder}")
        
        for idx, domain in enumerate(target_domains):
            # Add one extra dialogue for first 'remainder' domains
            domain_dialogue_count = dialogues_per_domain + (1 if idx < remainder else 0)
            logger.info(f"Generating {domain_dialogue_count} dialogues for domain: {domain}")
            
            if emit_callback:
                emit_callback('step_start', {
                    'step': 'experience_generation',
                    'step_name': 'Experience Generation',
                    'domain': domain,
                    'message': f'Starting experience generation for domain: {domain}'
                })
            
            domain_dialogues = self._generate_domain_dialogues(
                domain, 
                domain_dialogue_count,
                emit_callback,
                overrides
            )
            
            logger.info(f"\n{'='*80}")
            logger.info(f"STEP 3: Quality Filtering - Domain: {domain}")
            logger.info(f"{'='*80}")
            use_quality_judge = overrides is None or overrides.get("quality_judge", True)
            if use_quality_judge:
                logger.info(f"Filtering {len(domain_dialogues)} generated dialogues...")
            else:
                logger.info("Quality judge disabled (ablation): accepting all dialogues")
            
            if emit_callback:
                emit_callback('step_start', {
                    'step': 'quality_filtering',
                    'step_name': 'Quality Filtering',
                    'domain': domain,
                    'message': f'Filtering {len(domain_dialogues)} dialogues for quality...' if use_quality_judge else 'Accepting all (quality judge off)'
                })
            
            # Process dialogues through quality filter (or accept all if ablation)
            if use_quality_judge:
                improve_on_fail = (overrides or {}).get("quality_improve_on_fail")
                accepted, rejected = self.quality_judge.filter_dialogues(
                    domain_dialogues, improve_on_fail=improve_on_fail
                )
            else:
                accepted, rejected = list(domain_dialogues), []
            
            if emit_callback:
                emit_callback('step_data', {
                    'step': 'quality_filtering',
                    'data': {
                        'accepted': len(accepted),
                        'rejected': len(rejected),
                        'accepted_dialogues': [
                            {
                                'dialogue_id': d.get('dialogue_id'),
                                'quality_score': d.get('metadata', {}).get('quality_score', 0.0),
                                'turns': len(d.get('turns', []))
                            }
                            for d in accepted
                        ]
                    }
                })
            
            logger.info(f"Quality filtering completed:")
            logger.info(f"  - Accepted: {len(accepted)}")
            logger.info(f"  - Rejected: {len(rejected)}")
            
            # Log details of accepted dialogues
            for idx, dialogue in enumerate(accepted, 1):
                dialogue_id = dialogue.get('dialogue_id', 'unknown')
                quality_score = dialogue.get('metadata', {}).get('quality_score', 0.0)
                num_turns = len(dialogue.get('turns', []))
                logger.info(f"  ✓ Accepted #{idx}: {dialogue_id} (quality: {quality_score:.2f}, turns: {num_turns})")
            
            # Log details of rejected dialogues
            for idx, dialogue in enumerate(rejected, 1):
                dialogue_id = dialogue.get('dialogue_id', 'unknown')
                quality_score = dialogue.get('metadata', {}).get('quality_score', 0.0)
                num_turns = len(dialogue.get('turns', []))
                logger.info(f"  ✗ Rejected #{idx}: {dialogue_id} (quality: {quality_score:.2f}, turns: {num_turns})")
            
            logger.info(f"\n{'='*80}")
            logger.info(f"STEP 4: Saving Dialogues - Domain: {domain}")
            logger.info(f"{'='*80}")
            
            if emit_callback:
                emit_callback('step_start', {
                    'step': 'saving',
                    'step_name': 'Saving Dialogues',
                    'message': f'Saving {len(accepted)} accepted dialogues...'
                })
            
            # Accumulate this run's accepted dialogues for evaluation (so backend evaluates this run, not arbitrary load from disk)
            run_accepted_dialogues.extend(accepted)

            # Save accepted dialogues
            for dialogue in accepted:
                dialogue_id = dialogue.get('dialogue_id', 'unknown')
                try:
                    saved_id = self.dataset_store.save_dialogue(dialogue)
                    logger.info(f"  ✓ Saved dialogue: {saved_id}")
                    accepted_count += 1
                    
                    if emit_callback:
                        emit_callback('log', {
                            'level': 'success',
                            'message': f'Saved dialogue: {saved_id}',
                            'step': 'saving'
                        })
                except Exception as e:
                    logger.error(f"  ✗ Failed to save dialogue {dialogue_id}: {e}")
                    if emit_callback:
                        emit_callback('log', {
                            'level': 'error',
                            'message': f'Failed to save dialogue {dialogue_id}: {str(e)}',
                            'step': 'saving'
                        })
            
            # Update statistics
            generated_count += len(domain_dialogues)
            self.stats["by_domain"][domain] = {
                "generated": len(domain_dialogues),
                "accepted": len(accepted),
                "rejected": len(rejected)
            }
            
            logger.info(f"Domain {domain}: {len(accepted)}/{len(domain_dialogues)} dialogues accepted")
            
            # Update few-shot hub periodically
            if generated_count % 100 == 0:
                self._update_few_shot_hub()
        
        # Final statistics
        self.stats["total_generated"] = generated_count
        self.stats["total_accepted"] = accepted_count
        self.stats["total_rejected"] = generated_count - accepted_count
        self.stats["end_time"] = datetime.now()
        
        # Update few-shot hub with final high-quality dialogues
        logger.info(f"\n{'='*80}")
        logger.info("STEP 5: Updating Few-Shot Hub")
        logger.info(f"{'='*80}")
        logger.info("Updating few-shot hub with final high-quality dialogues...")
        self._update_few_shot_hub()
        
        # Save final progress
        logger.info(f"\n{'='*80}")
        logger.info("STEP 6: Saving Generation Progress")
        logger.info(f"{'='*80}")
        self._save_generation_progress()

        # Attach this run's accepted dialogues to stats so backend evaluates this run (not re-load from disk)
        self.stats["accepted_dialogues"] = run_accepted_dialogues

        # Compute evaluation metrics from this run's accepted dialogues
        logger.info(f"\n{'='*80}")
        logger.info("STEP 7: Computing Evaluation Metrics")
        logger.info(f"{'='*80}")

        if emit_callback:
            emit_callback('step_start', {
                'step': 'evaluation',
                'step_name': 'Evaluation',
                'message': 'Computing evaluation metrics...'
            })

        # Compute basic evaluation metrics (for logging/summary only) from this run's dialogues
        evaluation_metrics = self._compute_evaluation_metrics(run_accepted_dialogues)
        
        logger.info(f"\n{'='*80}")
        logger.info("GENERATION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Generation completed: {accepted_count}/{generated_count} dialogues accepted")
        logger.info(f"Acceptance rate: {(accepted_count/generated_count*100) if generated_count > 0 else 0:.1f}%")
        logger.info(f"{'='*80}\n")
        
        # Do NOT emit pipeline_complete here. When run from backend_server, the server runs
        # comprehensive evaluation and emits pipeline_complete once with full metrics, so the
        # frontend only shows the detailed evaluation panel (no duplicate / less-detailed first).
        
        return self.stats
    
    def _generate_domain_dialogues(self, domain: str, num_dialogues: int, emit_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None, overrides: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate dialogues for a specific domain."""
        logger.info(f"Generating {num_dialogues} dialogues for domain: {domain}")
        few_shot_override = (overrides or {}).get("few_shot_examples")
        
        dialogues = []
        
        for i in range(num_dialogues):
            try:
                logger.info(f"\n{'='*80}")
                logger.info(f"STEP 1: Experience Generation - Dialogue {i+1}/{num_dialogues}")
                logger.info(f"{'='*80}")
                
                # Generate experience (uses config.few_shot_examples, seeded hub via load_few_shot_examples)
                goal = self.experience_generator.get_random_goal(domain)
                logger.info(f"Selected goal: {goal}")
                
                if emit_callback:
                    emit_callback('log', {
                        'level': 'info',
                        'message': f'Generating experience {i+1}/{num_dialogues} for goal: {goal}',
                        'step': 'experience_generation'
                    })
                
                def _on_experience_error(msg: str) -> None:
                    if emit_callback:
                        emit_callback('log', {
                            'level': 'error',
                            'message': f'Experience generation failed (dialogue {i+1}/{num_dialogues}, domain {domain}): {msg}',
                            'step': 'experience_generation',
                        })

                experience_data = self.experience_generator.generate_experience(
                    goal, domain, few_shot_override=few_shot_override, on_error=_on_experience_error
                )
                
                if emit_callback:
                    emit_callback('step_data', {
                        'step': 'experience_generation',
                        'data': {
                            'experience': {
                                'experience_id': f'exp_{i+1}',
                                'domain': experience_data.get('domain'),
                                'goal': experience_data.get('goal'),
                                'context': experience_data.get('context'),
                                'first_utterance': experience_data.get('first_utterance'),
                                'user_persona': experience_data.get('user_persona')
                            }
                        }
                    })
                logger.info(f"Experience generated successfully:")
                logger.info(f"  - Domain: {experience_data.get('domain', 'N/A')}")
                logger.info(f"  - Goal: {experience_data.get('goal', 'N/A')}")
                logger.info(f"  - Context: {experience_data.get('context', 'N/A')[:100]}...")
                logger.info(f"  - First utterance: {experience_data.get('first_utterance', 'N/A')[:100]}...")
                logger.info(f"  - User persona: {experience_data.get('user_persona', 'N/A')}")
                
                logger.info(f"\n{'='*80}")
                logger.info(f"STEP 2: Dialogue Simulation - Dialogue {i+1}/{num_dialogues}")
                logger.info(f"{'='*80}")
                
                if emit_callback:
                    emit_callback('step_start', {
                        'step': 'dialogue_simulation',
                        'step_name': 'Dialogue Simulation',
                        'message': f'Simulating dialogue {i+1}/{num_dialogues}...'
                    })
                
                def on_live_progress(turns_so_far: list, step_message: str) -> None:
                    if emit_callback:
                        emit_callback('live_dialogue', {
                            'current_turns': turns_so_far,
                            'step_message': step_message,
                            'dialogue_index': i + 1,
                            'total_dialogues': num_dialogues,
                            'goal': experience_data.get('goal', '')[:80],
                        })

                def _on_simulate_error(msg: str) -> None:
                    if emit_callback:
                        emit_callback('log', {
                            'level': 'error',
                            'message': f'Dialogue simulation error (dialogue {i+1}/{num_dialogues}, domain {domain}): {msg}',
                            'step': 'dialogue_simulation',
                        })

                # Simulate dialogue (uses last-K-turns context, domain schema, progress hint, stricter goal-check, config truncation)
                dialogue = self.dialogue_simulator.simulate_dialogue(
                    experience_data, progress_callback=on_live_progress, on_error=_on_simulate_error
                )
                dialogue_id = dialogue.get('dialogue_id', 'unknown')
                num_turns = len(dialogue.get('turns', []))
                
                if emit_callback:
                    # Send the complete dialogue object for frontend display
                    emit_callback('step_data', {
                        'step': 'dialogue_simulation',
                        'data': {
                            'dialogue': dialogue  # Send the complete dialogue object
                        }
                    })
                
                logger.info(f"Dialogue simulated successfully:")
                logger.info(f"  - Dialogue ID: {dialogue_id}")
                logger.info(f"  - Total turns: {num_turns}")
                logger.info(f"  - Domain: {dialogue.get('domain', 'N/A')}")
                logger.info(f"  - Goal: {dialogue.get('goal', 'N/A')[:80]}...")
                
                # Log first few turns
                turns = dialogue.get('turns', [])
                if turns:
                    logger.info(f"  - First turn ({turns[0].get('role', 'N/A')}): {turns[0].get('text', 'N/A')[:80]}...")
                    if len(turns) > 1:
                        logger.info(f"  - Second turn ({turns[1].get('role', 'N/A')}): {turns[1].get('text', 'N/A')[:80]}...")
                    if len(turns) > 2:
                        logger.info(f"  - Last turn ({turns[-1].get('role', 'N/A')}): {turns[-1].get('text', 'N/A')[:80]}...")
                
                dialogues.append(dialogue)
                
                logger.info(f"\n✓ Dialogue {i+1}/{num_dialogues} generated successfully")
                
            except Exception as e:
                logger.error(f"\n✗ Error generating dialogue {i+1} for domain {domain}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                if emit_callback:
                    emit_callback('log', {
                        'level': 'error',
                        'message': f'Error generating dialogue {i+1}/{num_dialogues} for domain {domain}: {str(e)}',
                        'step': 'dialogue_simulation',
                    })
                continue
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Generated {len(dialogues)} dialogues for domain {domain}")
        logger.info(f"{'='*80}\n")
        return dialogues
    
    def _update_few_shot_hub(self) -> None:
        """Update the few-shot hub with high-quality examples."""
        try:
            added_count = self.dataset_store.update_few_shot_hub()
            logger.info(f"✓ Added {added_count} examples to few-shot hub")
        except Exception as e:
            logger.error(f"✗ Error updating few-shot hub: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _load_generation_progress(self) -> None:
        """Load generation progress from file."""
        progress_file = Path(self.config.data_dir) / "generation_progress.json"
        
        if progress_file.exists():
            try:
                progress_data = load_json(str(progress_file))
                self.stats.update(progress_data)
                logger.info("Loaded generation progress from file")
            except Exception as e:
                logger.error(f"Error loading progress: {e}")
    
    def _save_generation_progress(self) -> None:
        """Save generation progress to file."""
        progress_file = Path(self.config.data_dir) / "generation_progress.json"
        
        try:
            save_json(self.stats, str(progress_file))
            logger.info(f"✓ Saved generation progress to: {progress_file}")
            logger.info(f"  - Total generated: {self.stats.get('total_generated', 0)}")
            logger.info(f"  - Total accepted: {self.stats.get('total_accepted', 0)}")
            logger.info(f"  - Total rejected: {self.stats.get('total_rejected', 0)}")
        except Exception as e:
            logger.error(f"✗ Error saving progress: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _compute_evaluation_metrics(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute basic evaluation metrics from generated dialogues."""
        if not dialogues:
            return {
                "overall_score": 0.0,
                "diversity_score": 0.0,
                "coherence_score": 0.0,
                "task_success_rate": 0.0,
                "fluency_score": 0.0,
                "groundedness_score": 0.0,
                "categories": {
                    "lexical_diversity": 0.0,
                    "conversation_length": {
                        "avg_turns": 0.0,
                        "std_dev": 0.0
                    },
                    "domain_distribution": {},
                    "task_success_by_domain": {}
                }
            }
        
        # Compute basic statistics
        total_dialogues = len(dialogues)
        total_turns = sum(len(d.get('turns', [])) for d in dialogues)
        avg_turns = total_turns / total_dialogues if total_dialogues > 0 else 0.0
        
        # Compute turn count standard deviation
        turn_counts = [len(d.get('turns', [])) for d in dialogues]
        if len(turn_counts) > 1:
            import statistics
            std_dev_turns = statistics.stdev(turn_counts) if len(turn_counts) > 1 else 0.0
        else:
            std_dev_turns = 0.0
        
        # Compute task success rate (based on quality scores)
        quality_scores = [
            d.get('metadata', {}).get('quality_score', 0.0) 
            for d in dialogues
        ]
        task_success_count = sum(1 for score in quality_scores if score > 0.7)
        task_success_rate = task_success_count / total_dialogues if total_dialogues > 0 else 0.0
        
        # Compute average quality score
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Compute domain distribution
        domain_counts = {}
        domain_quality = {}
        for dialogue in dialogues:
            domain = dialogue.get('domain', 'unknown')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            if domain not in domain_quality:
                domain_quality[domain] = []
            domain_quality[domain].append(dialogue.get('metadata', {}).get('quality_score', 0.0))
        
        domain_distribution = {domain: count / total_dialogues for domain, count in domain_counts.items()}
        task_success_by_domain = {
            domain: sum(scores) / len(scores) if scores else 0.0
            for domain, scores in domain_quality.items()
        }
        
        # Compute lexical diversity (simple word count diversity)
        all_texts = []
        for dialogue in dialogues:
            for turn in dialogue.get('turns', []):
                text = turn.get('text', '')
                if text:
                    all_texts.append(text)
        
        unique_words = set()
        total_words = 0
        for text in all_texts:
            words = text.lower().split()
            unique_words.update(words)
            total_words += len(words)
        
        lexical_diversity = (len(unique_words) / total_words * 100) if total_words > 0 else 0.0
        
        # Compute overall scores (normalized to 0-1 range)
        overall_score = avg_quality_score
        diversity_score = min(lexical_diversity / 100.0, 1.0)  # Normalize to 0-1
        coherence_score = avg_quality_score * 0.9  # Approximate
        fluency_score = avg_quality_score * 0.95  # Approximate
        groundedness_score = task_success_rate  # Use task success as proxy
        
        logger.info(f"Evaluation metrics computed:")
        logger.info(f"  - Overall score: {overall_score:.2f}")
        logger.info(f"  - Task success rate: {task_success_rate:.2%}")
        logger.info(f"  - Average turns: {avg_turns:.1f}")
        logger.info(f"  - Lexical diversity: {lexical_diversity:.1f}%")
        
        return {
            "overall_score": round(overall_score, 3),
            "diversity_score": round(diversity_score, 3),
            "coherence_score": round(coherence_score, 3),
            "task_success_rate": round(task_success_rate, 3),
            "fluency_score": round(fluency_score, 3),
            "groundedness_score": round(groundedness_score, 3),
            "categories": {
                "lexical_diversity": round(lexical_diversity, 2),
                "conversation_length": {
                    "avg_turns": round(avg_turns, 2),
                    "std_dev": round(std_dev_turns, 2)
                },
                "domain_distribution": domain_distribution,
                "task_success_by_domain": task_success_by_domain
            }
        }
    
    def test_connection(self) -> bool:
        """Test LLM connection before starting generation."""
        logger.info("Testing LLM connection...")
        
        try:
            if self.llm_client.test_connection():
                logger.info("LLM connection test successful")
                return True
            else:
                logger.error("LLM connection test failed")
                return False
        except Exception as e:
            logger.error(f"LLM connection test error: {e}")
            return False
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get current generation statistics."""
        return self.stats.copy()
    
    def estimate_cost(self, num_dialogues: int) -> Dict[str, Any]:
        """Estimate API costs for generation."""
        # Rough estimates based on typical usage
        avg_turns_per_dialogue = 6
        avg_tokens_per_turn = 100
        
        total_tokens = num_dialogues * avg_turns_per_dialogue * avg_tokens_per_turn
        
        # Rough cost estimates (these would need to be updated based on actual API pricing)
        cost_estimates = {
            "total_tokens": total_tokens,
            "estimated_cost_usd": total_tokens * 0.00002,  # Rough estimate
            "dialogues": num_dialogues,
            "avg_turns": avg_turns_per_dialogue,
            "avg_tokens_per_turn": avg_tokens_per_turn
        }
        
        return cost_estimates

def main():
    """Main function for dialogue generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic goal-oriented dialogues")
    parser.add_argument("--num-dialogues", type=int, default=1000, help="Number of dialogues to generate")
    parser.add_argument("--domains", nargs="+", help="Domains to generate for", 
                       choices=["hotel", "restaurant", "taxi", "train", "attraction"])
    parser.add_argument("--resume", action="store_true", help="Resume from existing progress")
    parser.add_argument("--test-connection", action="store_true", help="Test LLM connection only")
    parser.add_argument("--estimate-cost", action="store_true", help="Estimate API costs")
    parser.add_argument("--run-evaluation", action="store_true", help="Run comprehensive evaluation after generation")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('generation.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load configuration
    config = Config()
    
    # Initialize generator
    generator = GoalConvoGenerator(config)
    
    try:
        # Test connection if requested
        if args.test_connection:
            if generator.test_connection():
                logger.info("Connection test passed")
                return 0
            else:
                logger.error("Connection test failed")
                return 1
        
        # Estimate costs if requested
        if args.estimate_cost:
            cost_estimate = generator.estimate_cost(args.num_dialogues)
            logger.info(f"Cost estimate for {args.num_dialogues} dialogues:")
            for key, value in cost_estimate.items():
                logger.info(f"  {key}: {value}")
            return 0
        
        # Generate dialogues
        logger.info(f"Starting generation of {args.num_dialogues} dialogues")
        
        stats = generator.generate_dialogues(
            num_dialogues=args.num_dialogues,
            domains=args.domains,
            resume=args.resume
        )
        
        # Print final statistics
        logger.info("Generation completed!")
        logger.info(f"Total generated: {stats['total_generated']}")
        logger.info(f"Total accepted: {stats['total_accepted']}")
        logger.info(f"Total rejected: {stats['total_rejected']}")
        logger.info(f"Acceptance rate: {stats['total_accepted']/stats['total_generated']:.2%}")
        
        for domain, domain_stats in stats['by_domain'].items():
            logger.info(f"Domain {domain}: {domain_stats['accepted']}/{domain_stats['generated']} accepted")
        
        # Create dataset version snapshot with config
        try:
            all_dialogues = generator.dataset_store.load_dialogues(domain=None, limit=None)
            if all_dialogues:
                version_manager = DatasetVersionManager(config.data_dir)
                api_config = config.get_api_config()
                version_id = version_manager.create_version(
                    dialogues=all_dialogues,
                    description=f"Script run: {len(all_dialogues)} dialogues, domains: {args.domains or config.domains}",
                    generation_config={
                        "num_dialogues": args.num_dialogues,
                        "domains": args.domains or config.domains,
                        "temperature": config.temperature,
                        "max_turns": config.max_turns,
                        "min_turns": config.min_turns,
                        "few_shot_examples": config.few_shot_examples,
                        "model": api_config.get("model", config.mistral_model),
                    },
                    tags=["script", "auto-generated"]
                )
                logger.info(f"Created dataset version {version_id}")
        except Exception as version_error:
            logger.warning(f"Version creation failed: {version_error}")
        
        # Run evaluation if requested
        if args.run_evaluation:
            logger.info("Running comprehensive evaluation...")
            try:
                # Import comprehensive evaluator
                import subprocess
                import sys
                
                eval_script = Path(__file__).parent / "comprehensive_evaluate.py"
                if eval_script.exists():
                    result = subprocess.run(
                        [sys.executable, str(eval_script)],
                        cwd=Path(__file__).parent.parent,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        logger.info("Evaluation completed successfully!")
                        logger.info("Check data/results/comprehensive_evaluation_report_latest.txt for detailed report")
                    else:
                        logger.error(f"Evaluation failed: {result.stderr}")
                else:
                    logger.warning("Comprehensive evaluation script not found")
                    logger.info("You can run evaluation manually: python scripts/comprehensive_evaluate.py")
            except Exception as e:
                logger.error(f"Error running evaluation: {e}")
                logger.info("You can run evaluation manually: python scripts/comprehensive_evaluate.py")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in dialogue generation: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
