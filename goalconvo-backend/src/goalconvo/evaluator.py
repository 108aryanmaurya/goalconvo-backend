"""
Evaluator for computing metrics on generated dialogues.

Implements BERTScore, diversity metrics, goal relevance, and domain-wise analysis
as described in the research paper.
"""

import json
import logging
import math
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .evaluation import (
    BertScoreEvaluator,
    CoherenceEvaluator,
    DiversityEvaluator,
    GoalCompletionEvaluator,
    GoalEvaluator,
)
from .evaluation.research_evaluator import (
    ResearchEvaluationReport,
    ResearchEvaluationSuite,
    evaluate_batch_chunks,
)
from .utils import load_json, save_json, ensure_dir

logger = logging.getLogger(__name__)

class Evaluator:
    """Evaluates synthetic dialogues using metrics from the research paper."""
    
    def __init__(self, config: Config):
        """Initialize the evaluator."""
        self.config = config
        self.results_dir = Path(config.data_dir) / "results"
        ensure_dir(str(self.results_dir))
        
        # Initialize BERTScore model
        self.bertscore_model = config.bertscore_model
        
        # Cache for computed metrics
        self.metrics_cache = {}
        self._bert_eval = BertScoreEvaluator(self.bertscore_model)
        self._diversity_eval = DiversityEvaluator()
        self._goal_eval = GoalEvaluator()
        self._coherence_eval = CoherenceEvaluator()
        self._goal_completion_eval = GoalCompletionEvaluator(config)

    def evaluate_goal_completion(
        self,
        dialogue: Dict[str, Any],
        llm_client: Any = None,
        *,
        use_llm_judge: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Automatic goal completion: rule-based slot/task checks plus optional LLM judge.

        Returns keys ``goal_completed``, ``completion_score``, ``missing_requirements``
        (and optional diagnostics).
        """
        return self._goal_completion_eval.evaluate(
            dialogue,
            llm_client=llm_client,
            use_llm_judge=use_llm_judge,
        )

    def research_report(
        self,
        synthetic_dialogues: List[Dict[str, Any]],
        multiwoz_dialogues: List[Dict[str, Any]],
        *,
        confidence: float = 0.95,
    ) -> ResearchEvaluationReport:
        """
        Run the full research metric suite (BERTScore, Distinct-n, goal completion,
        coherence, TF–IDF semantic similarity) vs MultiWOZ references.

        Use ``report.save_csv(prefix)`` for CSV + viz JSON exports.
        """
        suite = ResearchEvaluationSuite(self.config, confidence=confidence)
        return suite.evaluate(synthetic_dialogues, multiwoz_dialogues)

    @staticmethod
    def evaluate_in_batches(
        config: Config,
        synthetic_dialogues: List[Dict[str, Any]],
        multiwoz_dialogues: List[Dict[str, Any]],
        *,
        chunk_size: int = 256,
        confidence: float = 0.95,
    ) -> ResearchEvaluationReport:
        """Large-corpus entry point: chunked evaluation with merged statistics."""
        suite = ResearchEvaluationSuite(config, confidence=confidence)
        return evaluate_batch_chunks(
            suite, synthetic_dialogues, multiwoz_dialogues, chunk_size=chunk_size
        )
    
    def evaluate_synthetic_vs_real(
        self, 
        synthetic_dialogues: List[Dict[str, Any]], 
        real_dialogues: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate synthetic dialogues against real MultiWOZ dialogues.
        
        Args:
            synthetic_dialogues: List of synthetic dialogue data
            real_dialogues: List of real MultiWOZ dialogue data
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating {len(synthetic_dialogues)} synthetic vs {len(real_dialogues)} real dialogues")
        
        results = {
            "semantic_similarity": self._compute_semantic_similarity(synthetic_dialogues, real_dialogues),
            "diversity_metrics": self._compute_diversity_metrics(synthetic_dialogues, real_dialogues),
            "goal_relevance": self._goal_eval.corpus_goal_relevance(synthetic_dialogues),
            "coherence_metrics": {
                "synthetic": self._coherence_eval.corpus_mean(synthetic_dialogues),
                "real": self._coherence_eval.corpus_mean(real_dialogues),
            },
            "domain_analysis": self._compute_domain_analysis(synthetic_dialogues, real_dialogues),
            "statistical_analysis": self._compute_statistical_analysis(synthetic_dialogues, real_dialogues)
        }
        
        # Save results
        self._save_evaluation_results(results)
        
        return results
    
    def _bertscore_one_pair(self, cand_text: str, ref_text: str, max_chars: int = 1000) -> Optional[float]:
        """Compute BERTScore for one pair; retry with shorter text, then fallback model on overflow."""
        return self._bert_eval.score_one_pair(cand_text, ref_text, max_chars)

    def _compute_semantic_similarity(
        self, 
        synthetic_dialogues: List[Dict[str, Any]], 
        real_dialogues: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute BERTScore semantic similarity between synthetic and real dialogues."""
        logger.info("Computing semantic similarity (BERTScore)...")
        
        # Group real dialogues by domain for matching
        real_by_domain = {}
        for dialogue in real_dialogues:
            domain = dialogue.get("domain", "unknown")
            if domain not in real_by_domain:
                real_by_domain[domain] = []
            real_by_domain[domain].append(dialogue)
        
        bert_scores = []
        domain_scores = {}
        
        for synthetic_dialogue in synthetic_dialogues:
            domain = synthetic_dialogue.get("domain", "unknown")
            
            if domain not in real_by_domain or not real_by_domain[domain]:
                continue
            
            # Find closest real dialogue in same domain
            synthetic_text = self._extract_dialogue_text(synthetic_dialogue)
            best_score = 0.0
            # Truncate to avoid "int too big to convert" / overflow (DeBERTa max 512 tokens)
            max_chars = 1000
            syn_trunc = synthetic_text[:max_chars] if len(synthetic_text) > max_chars else synthetic_text
            
            for real_dialogue in real_by_domain[domain]:
                real_text = self._extract_dialogue_text(real_dialogue)
                ref_trunc = real_text[:max_chars] if len(real_text) > max_chars else real_text
                
                score = self._bertscore_one_pair(syn_trunc, ref_trunc, max_chars)
                if score is not None:
                    best_score = max(best_score, score)
            
            bert_scores.append(best_score)
            
            # Track by domain
            if domain not in domain_scores:
                domain_scores[domain] = []
            domain_scores[domain].append(best_score)
        
        # Compute statistics
        overall_bertscore = np.mean(bert_scores) if bert_scores else 0.0
        
        domain_bertscores = {}
        for domain, scores in domain_scores.items():
            domain_bertscores[domain] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "count": len(scores)
            }
        
        return {
            "overall_bertscore": overall_bertscore,
            "domain_bertscores": domain_bertscores,
            "individual_scores": bert_scores,
            "target_score": 0.71  # From research paper
        }
    
    def _compute_diversity_metrics(
        self, 
        synthetic_dialogues: List[Dict[str, Any]], 
        real_dialogues: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute lexical diversity metrics."""
        logger.info("Computing diversity metrics...")
        
        # Extract all text from dialogues
        synthetic_texts = [self._extract_dialogue_text(d) for d in synthetic_dialogues]
        real_texts = [self._extract_dialogue_text(d) for d in real_dialogues]
        
        # Compute diversity metrics
        synthetic_diversity = self._diversity_eval.dialogue_diversity(synthetic_texts)
        real_diversity = self._diversity_eval.dialogue_diversity(real_texts)
        
        return {
            "synthetic_diversity": synthetic_diversity,
            "real_diversity": real_diversity,
            "diversity_ratio": synthetic_diversity["combined"] / real_diversity["combined"] if real_diversity["combined"] > 0 else 0.0,
            "target_diversity": 0.46  # From research paper
        }
    
    def _compute_domain_analysis(
        self, 
        synthetic_dialogues: List[Dict[str, Any]], 
        real_dialogues: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute domain-wise analysis."""
        logger.info("Computing domain-wise analysis...")
        
        # Group dialogues by domain
        synthetic_by_domain = self._group_dialogues_by_domain(synthetic_dialogues)
        real_by_domain = self._group_dialogues_by_domain(real_dialogues)
        
        domain_analysis = {}
        
        for domain in self.config.domains:
            synth_domain = synthetic_by_domain.get(domain, [])
            real_domain = real_by_domain.get(domain, [])
            
            if not synth_domain or not real_domain:
                continue
            
            # Compute metrics for this domain
            domain_metrics = {
                "synthetic_count": len(synth_domain),
                "real_count": len(real_domain),
                "avg_turns_synthetic": np.mean([len(d.get("turns", [])) for d in synth_domain]),
                "avg_turns_real": np.mean([len(d.get("turns", [])) for d in real_domain]),
                "avg_length_synthetic": np.mean([len(self._extract_dialogue_text(d)) for d in synth_domain]),
                "avg_length_real": np.mean([len(self._extract_dialogue_text(d)) for d in real_domain])
            }
            
            domain_analysis[domain] = domain_metrics
        
        return domain_analysis
    
    def _compute_statistical_analysis(
        self, 
        synthetic_dialogues: List[Dict[str, Any]], 
        real_dialogues: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute statistical analysis of dialogues."""
        logger.info("Computing statistical analysis...")
        
        # Extract statistics
        synth_stats = self._compute_dialogue_statistics(synthetic_dialogues)
        real_stats = self._compute_dialogue_statistics(real_dialogues)
        
        return {
            "synthetic": synth_stats,
            "real": real_stats,
            "comparison": {
                "turn_ratio": synth_stats["avg_turns"] / real_stats["avg_turns"] if real_stats["avg_turns"] > 0 else 0.0,
                "length_ratio": synth_stats["avg_length"] / real_stats["avg_length"] if real_stats["avg_length"] > 0 else 0.0
            }
        }
    
    def _compute_dialogue_statistics(self, dialogues: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute basic statistics for a set of dialogues."""
        if not dialogues:
            return {"avg_turns": 0.0, "avg_length": 0.0, "total_dialogues": 0}
        
        turn_counts = [len(d.get("turns", [])) for d in dialogues]
        lengths = [len(self._extract_dialogue_text(d)) for d in dialogues]
        
        return {
            "avg_turns": np.mean(turn_counts),
            "std_turns": np.std(turn_counts),
            "avg_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "total_dialogues": len(dialogues)
        }
    
    def _group_dialogues_by_domain(self, dialogues: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group dialogues by domain."""
        grouped = {}
        for dialogue in dialogues:
            domain = dialogue.get("domain", "unknown")
            if domain not in grouped:
                grouped[domain] = []
            grouped[domain].append(dialogue)
        return grouped
    
    def _extract_dialogue_text(self, dialogue: Dict[str, Any]) -> str:
        """Extract text content from a dialogue."""
        turns = dialogue.get("turns", [])
        texts = []
        for turn in turns:
            texts.append(turn.get("text", ""))
        return " ".join(texts)
    
    def _save_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to file."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"evaluation_results_{timestamp}.json"
        
        save_json(results, str(results_file))
        logger.info(f"Saved evaluation results to {results_file}")
    
    def generate_evaluation_report(
        self, 
        results: Dict[str, Any], 
        output_path: Optional[str] = None
    ) -> str:
        """Generate a human-readable evaluation report."""
        if output_path is None:
            output_path = str(self.results_dir / "evaluation_report.txt")
        
        report_lines = [
            "GoalConvo Evaluation Report",
            "=" * 50,
            "",
            "SEMANTIC SIMILARITY (BERTScore)",
            f"Overall BERTScore: {results['semantic_similarity']['overall_bertscore']:.3f}",
            f"Target Score: {results['semantic_similarity']['target_score']:.3f}",
            "",
            "Domain-wise BERTScore:",
        ]
        
        for domain, scores in results['semantic_similarity']['domain_bertscores'].items():
            report_lines.append(f"  {domain}: {scores['mean']:.3f} ± {scores['std']:.3f} (n={scores['count']})")
        
        report_lines.extend([
            "",
            "DIVERSITY METRICS",
            f"Synthetic Diversity: {results['diversity_metrics']['synthetic_diversity']['combined']:.3f}",
            f"Real Diversity: {results['diversity_metrics']['real_diversity']['combined']:.3f}",
            f"Target Diversity: {results['diversity_metrics']['target_diversity']:.3f}",
            "",
            "GOAL RELEVANCE",
            f"Overall Goal Relevance: {results['goal_relevance']['overall_goal_relevance']:.3f}",
            f"Target Goal Relevance: {results['goal_relevance']['target_goal_relevance']:.3f}",
            "",
            "Domain-wise Goal Relevance:",
        ])
        
        for domain, stats in results['goal_relevance']['domain_goal_relevance'].items():
            report_lines.append(f"  {domain}: {stats['percentage']:.3f} ({stats['satisfied']}/{stats['total']})")
        
        report_lines.extend([
            "",
            "STATISTICAL ANALYSIS",
            f"Synthetic Dialogues: {results['statistical_analysis']['synthetic']['total_dialogues']}",
            f"Real Dialogues: {results['statistical_analysis']['real']['total_dialogues']}",
            f"Avg Turns (Synthetic): {results['statistical_analysis']['synthetic']['avg_turns']:.1f}",
            f"Avg Turns (Real): {results['statistical_analysis']['real']['avg_turns']:.1f}",
            f"Avg Length (Synthetic): {results['statistical_analysis']['synthetic']['avg_length']:.1f}",
            f"Avg Length (Real): {results['statistical_analysis']['real']['avg_length']:.1f}",
        ])
        
        report_text = "\n".join(report_lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Generated evaluation report: {output_path}")
        return report_text
