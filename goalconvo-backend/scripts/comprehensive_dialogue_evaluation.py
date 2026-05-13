#!/usr/bin/env python3
"""
Comprehensive Dialogue Evaluation Script

Implements multiple evaluation metrics for GoalConvo-generated dialogues:
1. Goal Completion Rate (GCR)
2. Task Success Rate (TSR)
3. BLEU Score
4. Dialogue Length / Turns
5. Repetition Rate
6. LLM-as-a-Judge (Task Success, Coherence, Diversity, Fluency, Groundedness)
"""

import json
import logging
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import Counter
from datetime import datetime

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np

# NLTK imports with error handling
NLTK_AVAILABLE = True
NLTK_TOKENIZE_AVAILABLE = False

# Simple fallback tokenization function
def simple_word_tokenize(text):
    """Simple word tokenization using split."""
    return text.split()

# Try to import and use NLTK
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    
    # Try to download punkt_tab (newer version) or punkt (older version)
    try:
        nltk.data.find('tokenizers/punkt_tab')
        NLTK_TOKENIZE_AVAILABLE = True
    except (LookupError, OSError):
        try:
            nltk.data.find('tokenizers/punkt')
            NLTK_TOKENIZE_AVAILABLE = True
        except (LookupError, OSError):
            try:
                print("Downloading NLTK punkt_tab tokenizer...")
                nltk.download('punkt_tab', quiet=True)
                NLTK_TOKENIZE_AVAILABLE = True
            except Exception:
                try:
                    print("Downloading NLTK punkt tokenizer...")
                    nltk.download('punkt', quiet=True)
                    NLTK_TOKENIZE_AVAILABLE = True
                except Exception as download_error:
                    print(f"Warning: Could not download NLTK tokenizer: {download_error}")
                    NLTK_TOKENIZE_AVAILABLE = False
    
    # Import word_tokenize only if tokenizer is available
    if NLTK_TOKENIZE_AVAILABLE:
        try:
            from nltk.tokenize import word_tokenize
        except Exception:
            NLTK_TOKENIZE_AVAILABLE = False
            word_tokenize = simple_word_tokenize
    else:
        word_tokenize = simple_word_tokenize
        
except Exception as e:
    # If NLTK is not available, use fallback tokenization
    print(f"Warning: NLTK not available: {e}. Using fallback tokenization.")
    NLTK_AVAILABLE = False
    NLTK_TOKENIZE_AVAILABLE = False
    
    def sentence_bleu(reference, candidate, smoothing_function=None):
        """Fallback BLEU calculation (simplified)"""
        if not reference or not candidate:
            return 0.0
        # Simple word overlap ratio
        ref_words = set(reference)
        cand_words = set(candidate)
        if not ref_words or not cand_words:
            return 0.0
        overlap = len(ref_words & cand_words)
        return overlap / max(len(ref_words), len(cand_words))
    
    class SmoothingFunction:
        def method1(self):
            return None
    
    word_tokenize = simple_word_tokenize

from goalconvo.config import Config
from goalconvo.llm_client import LLMClient
from goalconvo.dataset_store import DatasetStore
from goalconvo.utils import load_json

logger = logging.getLogger(__name__)

# Import BERTScore for semantic similarity
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logger.warning("BERTScore not available. Install with: pip install bert-score")

class ComprehensiveDialogueEvaluator:
    """Comprehensive evaluator for generated dialogues with multiple metrics."""
    
    def __init__(self, config: Config):
        """Initialize the evaluator."""
        self.config = config
        self.llm_client = LLMClient(config, api_config=config.get_evaluation_api_config())
        self.dataset_store = DatasetStore(config)
        self.results_dir = Path(config.data_dir) / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Smoothing function for BLEU (only if NLTK is available)
        if NLTK_AVAILABLE:
            try:
                self.smoothing = SmoothingFunction().method1
            except Exception:
                self.smoothing = None
        else:
            self.smoothing = None
        
    def evaluate_dialogues(
        self,
        dialogues: List[Dict[str, Any]],
        reference_dialogues: Optional[List[Dict[str, Any]]] = None,
        use_llm_judge: bool = True,
        emit_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        yield_callback: Optional[Callable[[], None]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a set of dialogues using all metrics.

        API keys: Only use_llm_judge=True requires the same LLM API key as generation.
        BERTScore uses a HuggingFace model (no key). GCR, TSR, BLEU, diversity, length,
        repetition need no API key. If reference_dialogues is None, BERTScore and BLEU
        are skipped (no error); run scripts/download_multiwoz.py and pass refs for those.

        Args:
            dialogues: List of generated dialogues to evaluate
            reference_dialogues: Optional reference dialogues (e.g., MultiWOZ) for BERTScore and BLEU; skipped if None
            use_llm_judge: Whether to use LLM-as-a-Judge evaluation (set EVAL_SKIP_LLM_JUDGE=1 to disable)
            emit_callback: Optional (event_type, data) callback to emit log messages for frontend progress
            yield_callback: Optional no-arg callback to yield to the event loop (e.g. eventlet.sleep(0)) to keep WebSocket alive

        Returns:
            Dictionary with all evaluation metrics
        """
        def _yield() -> None:
            if yield_callback:
                try:
                    yield_callback()
                except Exception:
                    pass

        def _log(msg: str) -> None:
            logger.info(msg)
            if emit_callback:
                try:
                    emit_callback('log', {'message': msg, 'step': 'evaluation', 'level': 'info'})
                except Exception:
                    pass
            _yield()

        _log(f"Evaluating {len(dialogues)} dialogues...")
        
        results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_dialogues": len(dialogues),
            "metrics": {}
        }
        
        # 1. Goal Completion Rate (GCR)
        _log("Computing Goal Completion Rate...")
        gcr_results = self._compute_goal_completion_rate(dialogues)
        results["metrics"]["goal_completion_rate"] = gcr_results
        
        # 2. Task Success Rate (TSR)
        _log("Computing Task Success Rate...")
        tsr_results = self._compute_task_success_rate(dialogues)
        results["metrics"]["task_success_rate"] = tsr_results
        
        # 3. Lexical Diversity (can compute without reference)
        _log("Computing Lexical Diversity...")
        lexical_diversity_results = self._compute_lexical_diversity(dialogues, reference_dialogues)
        results["metrics"]["lexical_diversity"] = lexical_diversity_results
        
        # 4. BERTScore Semantic Similarity (if reference dialogues provided)
        if reference_dialogues:
            _log("Computing BERTScore semantic similarity...")
            bertscore_results = self._compute_bertscore_similarity(dialogues, reference_dialogues, yield_callback=yield_callback)
            results["metrics"]["bertscore_similarity"] = bertscore_results
        
        # 5. BLEU Score (if reference dialogues provided)
        if reference_dialogues:
            _log("Computing BLEU Scores...")
            bleu_results = self._compute_bleu_scores(dialogues, reference_dialogues)
            results["metrics"]["bleu_score"] = bleu_results
        
        # 6. Dialogue Length / Turns
        _log("Computing dialogue length and turns...")
        length_results = self._compute_dialogue_length_metrics(dialogues)
        results["metrics"]["dialogue_length"] = length_results
        
        # 6. Repetition Rate
        _log("Computing repetition rate...")
        repetition_results = self._compute_repetition_rate(dialogues)
        results["metrics"]["repetition_rate"] = repetition_results
        
        # 7. Response Time Tracking
        _log("Computing response time metrics...")
        response_time_results = self._compute_response_time_metrics(dialogues)
        results["metrics"]["response_time"] = response_time_results
        
        # 8. LLM-as-a-Judge (if enabled)
        if use_llm_judge:
            _log("Running LLM-as-a-Judge evaluation...")
            llm_judge_results = self._compute_llm_judge_metrics(dialogues, yield_callback=yield_callback)
            results["metrics"]["llm_judge"] = llm_judge_results

        # 9. Advanced evaluation metrics (intent, slots, state tracking)
        _log("Computing advanced evaluation metrics (intent, slots, state tracking)...")
        advanced_metrics = self._compute_advanced_evaluation_metrics(dialogues)
        results["metrics"]["advanced_evaluation"] = advanced_metrics
        
        # Generate summary table
        results["summary_table"] = self._generate_summary_table(results["metrics"])
        
        return results

    def _compute_advanced_evaluation_metrics(
        self,
        dialogues: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute additional evaluation metrics for more depth.

        These are lightweight, heuristic metrics that approximate:
          - intent classification consistency
          - slot / constraint coverage from goals
          - simple dialogue state tracking quality
        """
        if not dialogues:
            return {
                "intent_consistency": {},
                "slot_coverage": {},
                "state_tracking": {},
            }

        # Intent categories based on goal text
        intent_categories = {
            "booking": ["book", "reserve", "reservation", "ticket"],
            "search": ["find", "search", "looking for", "look for"],
            "info": ["information", "details", "tell me", "explain"],
        }

        intent_counts = {name: 0 for name in intent_categories.keys()}
        intent_success = {name: 0 for name in intent_categories.keys()}

        slot_hits = 0
        slot_total = 0

        state_consistent_count = 0
        state_total = 0

        for dialogue in dialogues:
            goal = (dialogue.get("goal") or "").lower()
            turns = dialogue.get("turns", [])
            text = " ".join(t.get("text", "").lower() for t in turns)

            # --- Intent consistency ---
            matched_intents = []
            for name, keywords in intent_categories.items():
                if any(k in goal for k in keywords):
                    matched_intents.append(name)
                    intent_counts[name] += 1

                    # Check if similar intent words appear in dialogue text
                    if any(k in text for k in keywords):
                        intent_success[name] += 1

            # --- Slot coverage (very simple heuristic) ---
            # Count how many constraint-like tokens from goal appear in dialogue
            for token in goal.split():
                if token.isdigit():
                    slot_total += 1
                    if token in text:
                        slot_hits += 1
                elif token in ["morning", "evening", "tonight", "today", "tomorrow"]:
                    slot_total += 1
                    if token in text:
                        slot_hits += 1

            # --- State tracking (consistency) ---
            # Heuristic: penalize if obvious contradictions appear
            if turns:
                state_total += 1
                lower_text = text
                inconsistent = any(
                    phrase in lower_text
                    for phrase in [
                        "i thought you said",
                        "you already told me",
                        "that contradicts",
                        "earlier you said",
                    ]
                )
                if not inconsistent:
                    state_consistent_count += 1

        intent_metrics = {}
        for name, total in intent_counts.items():
            if total > 0:
                intent_metrics[name] = {
                    "count": total,
                    "aligned": intent_success[name],
                    "consistency": intent_success[name] / total,
                }

        slot_coverage = {
            "hits": slot_hits,
            "total": slot_total,
            "coverage": (slot_hits / slot_total) if slot_total > 0 else 0.0,
        }

        state_tracking = {
            "total_dialogues": state_total,
            "consistent": state_consistent_count,
            "consistency_rate": (state_consistent_count / state_total) if state_total else 0.0,
        }

        return {
            "intent_consistency": intent_metrics,
            "slot_coverage": slot_coverage,
            "state_tracking": state_tracking,
        }
    
    def _compute_goal_completion_rate(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute Goal Completion Rate (GCR).
        
        Checks if all constraints and requestables from the goal are fulfilled in the dialogue.
        """
        completed_count = 0
        domain_stats = {}
        
        for dialogue in dialogues:
            domain = dialogue.get("domain", "unknown")
            goal = dialogue.get("goal", "")
            goal_data = dialogue.get("goal_data", {})
            turns = dialogue.get("turns", [])
            
            # Extract goal constraints and requestables
            constraints = self._extract_goal_constraints(goal, goal_data)
            requestables = self._extract_goal_requestables(goal, goal_data)
            
            # Check if goal is completed
            is_completed = self._check_goal_completion(turns, constraints, requestables)
            
            if is_completed:
                completed_count += 1
            
            # Track by domain
            if domain not in domain_stats:
                domain_stats[domain] = {"completed": 0, "total": 0}
            domain_stats[domain]["total"] += 1
            if is_completed:
                domain_stats[domain]["completed"] += 1
        
        total = len(dialogues)
        gcr = (completed_count / total * 100) if total > 0 else 0.0
        
        domain_gcr = {}
        for domain, stats in domain_stats.items():
            domain_gcr[domain] = {
                "completed": stats["completed"],
                "total": stats["total"],
                "percentage": (stats["completed"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
            }
        
        return {
            "overall_gcr": gcr,
            "completed_count": completed_count,
            "total_count": total,
            "domain_gcr": domain_gcr
        }
    
    def _extract_goal_constraints(self, goal: str, goal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract constraints from goal text or goal_data."""
        constraints = {}
        
        # Try to extract from goal_data first
        if goal_data:
            for domain, info in goal_data.items():
                if isinstance(info, dict):
                    constraints[domain] = info.get("constraints", {})
        
        # If not found, try to parse from goal text
        if not constraints:
            # Look for common constraint patterns
            constraint_patterns = {
                "area": r"(?:area|location|in|near)\s*(?:is|:|=)?\s*([a-z]+)",
                "price": r"(?:price|price range|budget)\s*(?:is|:|=)?\s*([a-z]+)",
                "type": r"(?:type|kind|style)\s*(?:is|:|=)?\s*([a-z]+)",
            }
            
            goal_lower = goal.lower()
            for key, pattern in constraint_patterns.items():
                match = re.search(pattern, goal_lower)
                if match:
                    if "domain" not in constraints:
                        constraints["domain"] = {}
                    constraints["domain"][key] = match.group(1)
        
        return constraints
    
    def _extract_goal_requestables(self, goal: str, goal_data: Dict[str, Any]) -> List[str]:
        """Extract requestables (information to request) from goal."""
        requestables = []
        
        # Try to extract from goal_data
        if goal_data:
            for domain, info in goal_data.items():
                if isinstance(info, dict):
                    requestables.extend(info.get("requestables", []))
        
        # If not found, look for common requestable patterns
        if not requestables:
            goal_lower = goal.lower()
            common_requestables = [
                "phone", "address", "postcode", "reference number",
                "price", "availability", "time", "date"
            ]
            for req in common_requestables:
                if req in goal_lower:
                    requestables.append(req)
        
        return requestables
    
    def _check_goal_completion(
        self,
        turns: List[Dict[str, str]],
        constraints: Dict[str, Any],
        requestables: List[str]
    ) -> bool:
        """Check if goal constraints and requestables are satisfied in dialogue."""
        if not turns:
            return False
        
        # Combine all dialogue text
        dialogue_text = " ".join([turn.get("text", "").lower() for turn in turns])
        
        # Check constraints
        constraints_satisfied = True
        for domain, domain_constraints in constraints.items():
            if isinstance(domain_constraints, dict):
                for key, value in domain_constraints.items():
                    # Check if constraint value appears in dialogue
                    if value and value.lower() not in dialogue_text:
                        # Also check for synonyms
                        if not self._check_synonym(value.lower(), dialogue_text):
                            constraints_satisfied = False
                            break
        
        # Check requestables (at least some should be mentioned)
        requestables_satisfied = len(requestables) == 0  # If no requestables, consider satisfied
        if requestables:
            satisfied_count = 0
            for req in requestables:
                if req.lower() in dialogue_text or self._check_synonym(req, dialogue_text):
                    satisfied_count += 1
            
            # Require at least 50% of requestables to be satisfied
            if requestables:
                requestables_satisfied = (satisfied_count / len(requestables)) >= 0.5
        
        # Check for completion indicators
        completion_keywords = [
            "thank you", "thanks", "perfect", "great", "excellent",
            "booked", "confirmed", "reserved", "done", "completed",
            "that's exactly what I needed", "sounds good", "that works"
        ]
        has_completion = any(keyword in dialogue_text for keyword in completion_keywords)
        
        return constraints_satisfied and requestables_satisfied and has_completion
    
    def _check_synonym(self, word: str, text: str) -> bool:
        """Check if word or its synonyms appear in text."""
        # Simple synonym mapping
        synonyms = {
            "centre": ["center", "central", "downtown"],
            "cheap": ["inexpensive", "affordable", "budget"],
            "expensive": ["pricey", "costly", "high-end"],
            "north": ["northern"],
            "south": ["southern"],
            "east": ["eastern"],
            "west": ["western"]
        }
        
        if word in text:
            return True
        
        for synonym_list in synonyms.values():
            if word in synonym_list:
                for syn in synonym_list:
                    if syn in text:
                        return True
        
        return False
    
    def _compute_task_success_rate(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute Task Success Rate (TSR).
        
        Judges whether the dialogue successfully achieved the user's intent.
        """
        successful_count = 0
        domain_stats = {}
        
        for dialogue in dialogues:
            domain = dialogue.get("domain", "unknown")
            goal = dialogue.get("goal", "")
            turns = dialogue.get("turns", [])
            
            # Judge task success based on multiple factors
            is_successful = self._judge_task_success(goal, turns)
            
            if is_successful:
                successful_count += 1
            
            # Track by domain
            if domain not in domain_stats:
                domain_stats[domain] = {"successful": 0, "total": 0}
            domain_stats[domain]["total"] += 1
            if is_successful:
                domain_stats[domain]["successful"] += 1
        
        total = len(dialogues)
        tsr = (successful_count / total * 100) if total > 0 else 0.0
        
        domain_tsr = {}
        for domain, stats in domain_stats.items():
            domain_tsr[domain] = {
                "successful": stats["successful"],
                "total": stats["total"],
                "percentage": (stats["successful"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
            }
        
        return {
            "overall_tsr": tsr,
            "successful_count": successful_count,
            "total_count": total,
            "domain_tsr": domain_tsr
        }
    
    def _judge_task_success(self, goal: str, turns: List[Dict[str, str]]) -> bool:
        """Judge if task was successful based on goal and dialogue turns."""
        if not turns or not goal:
            return False
        
        dialogue_text = " ".join([turn.get("text", "").lower() for turn in turns])
        goal_lower = goal.lower()
        
        # Extract key intent from goal
        intent_keywords = []
        if "book" in goal_lower or "reserve" in goal_lower:
            intent_keywords.extend(["book", "reserve", "booking", "reservation"])
        if "find" in goal_lower or "search" in goal_lower:
            intent_keywords.extend(["find", "search", "looking for"])
        if "information" in goal_lower or "details" in goal_lower:
            intent_keywords.extend(["information", "details", "tell me"])
        
        # Check if intent appears to be fulfilled
        intent_fulfilled = False
        if intent_keywords:
            # Check if any intent keyword appears and is followed by completion
            for keyword in intent_keywords:
                if keyword in dialogue_text:
                    # Look for confirmation after intent
                    idx = dialogue_text.find(keyword)
                    subsequent_text = dialogue_text[idx:idx+200]  # Next 200 chars
                    if any(word in subsequent_text for word in ["yes", "confirmed", "done", "booked", "found"]):
                        intent_fulfilled = True
                        break
        else:
            # If no specific intent, check for general completion
            intent_fulfilled = any(word in dialogue_text for word in ["thank", "perfect", "great", "excellent"])
        
        # Check for sufficient dialogue length (at least 4 turns)
        has_sufficient_length = len(turns) >= 4
        
        # Check for user satisfaction indicators (expanded for better task success rate)
        user_turns = [t for t in turns if t.get("role", "").lower() == "user"]
        last_user_turn = user_turns[-1] if user_turns else {}
        last_user_text = last_user_turn.get("text", "").lower()
        
        has_satisfaction = any(word in last_user_text for word in [
            "thank", "thanks", "perfect", "great", "excellent", "good", "sounds good",
            "all set", "that works", "that'll work", "appreciate it", "that's great"
        ])
        
        # Success if: (intent fulfilled and satisfaction) OR (sufficient length and clear satisfaction)
        return (intent_fulfilled and has_satisfaction) or (has_sufficient_length and has_satisfaction)
    
    # Fallback BERTScore model when DeBERTa hits "int too big to convert" (tokenizer/config)
    BERTSCORE_FALLBACK_MODEL = "bert-base-uncased"

    def _compute_bertscore_similarity(
        self,
        dialogues: List[Dict[str, Any]],
        reference_dialogues: List[Dict[str, Any]],
        yield_callback: Optional[Callable[[], None]] = None
    ) -> Dict[str, Any]:
        """Compute BERTScore in one batched call so the model loads once (not per-pair)."""
        if yield_callback:
            try:
                yield_callback()
            except Exception:
                pass
        if not BERTSCORE_AVAILABLE:
            logger.warning("BERTScore not available. Skipping semantic similarity computation.")
            return {
                "overall_bertscore": 0.0,
                "std_bertscore": 0.0,
                "target_score": 0.71,
                "note": "BERTScore not installed. Install with: pip install bert-score"
            }
        
        logger.info("Computing BERTScore semantic similarity...")
        bert_scores: List[float] = []
        domain_scores: Dict[str, List[float]] = {}
        
        ref_by_domain: Dict[str, List[Dict[str, Any]]] = {}
        for ref_dialogue in reference_dialogues:
            domain = ref_dialogue.get("domain", "unknown")
            if domain not in ref_by_domain:
                ref_by_domain[domain] = []
            ref_by_domain[domain].append(ref_dialogue)
        
        max_chars = 1000
        model_type = self.config.bertscore_model if hasattr(self.config, 'bertscore_model') else 'microsoft/deberta-xlarge-mnli'
        cands: List[str] = []
        refs_list: List[str] = []
        pair_dialogue_idx: List[int] = []
        eligible: List[Tuple[str, int]] = []  # (domain, count of refs for this dialogue)
        
        for dialogue in dialogues:
            domain = dialogue.get("domain", "unknown")
            if domain not in ref_by_domain or not ref_by_domain[domain]:
                continue
            gen_text = self._extract_dialogue_text(dialogue)
            gen_trunc = gen_text[:max_chars] if len(gen_text) > max_chars else gen_text
            local_idx = len(eligible)
            n_refs = 0
            for ref_dialogue in ref_by_domain[domain][:10]:
                ref_text = self._extract_dialogue_text(ref_dialogue)
                ref_trunc = ref_text[:max_chars] if len(ref_text) > max_chars else ref_text
                cands.append(gen_trunc)
                refs_list.append(ref_trunc)
                pair_dialogue_idx.append(local_idx)
                n_refs += 1
            eligible.append((domain, n_refs))
        
        if not cands:
            return {
                "overall_bertscore": 0.0,
                "std_bertscore": 0.0,
                "individual_scores": [],
                "domain_bertscores": {},
                "target_score": 0.5,
                "note": "Measures semantic similarity to MultiWOZ reference dialogues. Target: 0.71"
            }
        
        def run_batch(c_list: List[str], r_list: List[str], model: str):
            P, R, F1 = bert_score(c_list, r_list, model_type=model, verbose=False, lang="en")
            return F1
        
        f1_tensor = None
        if yield_callback:
            try:
                yield_callback()
            except Exception:
                pass
        try:
            f1_tensor = run_batch(cands, refs_list, model_type)
        except (OverflowError, ValueError, Exception) as e:
            err_str = str(e).lower()
            if "int too big to convert" in err_str or "overflow" in err_str:
                for cap in [400, 200]:
                    try:
                        c = [s[:cap] if len(s) > cap else s for s in cands]
                        r = [s[:cap] if len(s) > cap else s for s in refs_list]
                        f1_tensor = run_batch(c, r, model_type)
                        break
                    except (OverflowError, ValueError, Exception):
                        continue
                if f1_tensor is None:
                    try:
                        c = [s[:512] if len(s) > 512 else s for s in cands]
                        r = [s[:512] if len(s) > 512 else s for s in refs_list]
                        f1_tensor = run_batch(c, r, self.BERTSCORE_FALLBACK_MODEL)
                    except Exception as fe:
                        logger.warning("Error computing BERTScore (fallback model): %s", fe)
            elif "int too big to convert" not in err_str and "overflow" not in err_str:
                logger.warning("Error computing BERTScore: %s", e)
        
        if f1_tensor is not None:
            best_per_dialogue: Dict[int, float] = {}
            for i, local_idx in enumerate(pair_dialogue_idx):
                s = float(f1_tensor[i].item())
                best_per_dialogue[local_idx] = max(best_per_dialogue.get(local_idx, 0.0), s)
            for local_idx, (domain, _) in enumerate(eligible):
                best = best_per_dialogue.get(local_idx, 0.0)
                if best > 0:
                    bert_scores.append(best)
                    if domain not in domain_scores:
                        domain_scores[domain] = []
                    domain_scores[domain].append(best)
        
        avg_bertscore = np.mean(bert_scores) if bert_scores else 0.0
        std_bertscore = np.std(bert_scores) if bert_scores else 0.0
        
        domain_bertscores = {}
        for domain, scores in domain_scores.items():
            domain_bertscores[domain] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "count": len(scores)
            }
        
        return {
            "overall_bertscore": avg_bertscore,
            "std_bertscore": std_bertscore,
            "individual_scores": bert_scores,
            "domain_bertscores": domain_bertscores,
            "target_score": 0.71,  # From research paper
            "note": "Measures semantic similarity to MultiWOZ reference dialogues. Target: 0.71"
        }
    
    def _compute_lexical_diversity(
        self,
        dialogues: List[Dict[str, Any]],
        reference_dialogues: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Compute lexical diversity metrics (Distinct-1 and Distinct-2)."""
        logger.info("Computing lexical diversity metrics...")
        
        # Extract all dialogue texts
        synthetic_texts = [self._extract_dialogue_text(d) for d in dialogues]
        
        # Compute diversity for synthetic dialogues
        synthetic_diversity = self._compute_dialogue_diversity(synthetic_texts)
        
        # Compute diversity for reference dialogues if available
        real_diversity = None
        if reference_dialogues:
            real_texts = [self._extract_dialogue_text(d) for d in reference_dialogues]
            real_diversity = self._compute_dialogue_diversity(real_texts)
        
        # Compute domain-wise diversity
        domain_diversity = {}
        dialogues_by_domain = {}
        
        for dialogue in dialogues:
            domain = dialogue.get("domain", "unknown")
            if domain not in dialogues_by_domain:
                dialogues_by_domain[domain] = []
            dialogues_by_domain[domain].append(dialogue)
        
        for domain, domain_dialogues in dialogues_by_domain.items():
            domain_texts = [self._extract_dialogue_text(d) for d in domain_dialogues]
            domain_diversity[domain] = self._compute_dialogue_diversity(domain_texts)
        
        # Calculate diversity ratio if reference available
        diversity_ratio = None
        if real_diversity and real_diversity["combined"] > 0:
            diversity_ratio = synthetic_diversity["combined"] / real_diversity["combined"]
        
        return {
            "distinct_1": synthetic_diversity["distinct_1"],
            "distinct_2": synthetic_diversity["distinct_2"],
            "combined": synthetic_diversity["combined"],
            "target_diversity": 0.46,  # From research paper
            "real_diversity": real_diversity,
            "diversity_ratio": diversity_ratio,
            "domain_diversity": domain_diversity,
            "note": "Measures lexical diversity using Distinct-1 (unique unigrams/total) and Distinct-2 (unique bigrams/total). Target: 0.46"
        }
    
    def _compute_dialogue_diversity(self, texts: List[str]) -> Dict[str, float]:
        """Compute diversity (Distinct-1, Distinct-2). Uses per-dialogue average so small corpora are not unfairly low."""
        if not texts:
            return {"distinct_1": 0.0, "distinct_2": 0.0, "combined": 0.0}

        def tokenize(text: str) -> List[str]:
            return re.findall(r'\b\w+\b', text.lower())

        def distinct_for_tokens(tokens: List[str]) -> Tuple[float, float]:
            if not tokens:
                return 0.0, 0.0
            uniq_1 = len(set(tokens)) / len(tokens)
            bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
            uniq_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
            return uniq_1, uniq_2

        # Per-dialogue distinct then average (standard in dialogue metrics; avoids corpus-level dilution)
        per_d1, per_d2 = [], []
        for text in texts:
            tokens = tokenize(text)
            d1, d2 = distinct_for_tokens(tokens)
            if tokens:
                per_d1.append(d1)
                per_d2.append(d2)

        if not per_d1:
            return {"distinct_1": 0.0, "distinct_2": 0.0, "combined": 0.0}

        distinct_1 = float(np.mean(per_d1))
        distinct_2 = float(np.mean(per_d2))
        combined = (distinct_1 + distinct_2) / 2
        return {
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
            "combined": combined
        }
    
    def _compute_bleu_scores(
        self,
        dialogues: List[Dict[str, Any]],
        reference_dialogues: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute BLEU scores by comparing generated dialogues to reference dialogues."""
        bleu_scores = []
        domain_scores = {}
        
        # Group reference dialogues by domain
        ref_by_domain = {}
        for ref_dialogue in reference_dialogues:
            domain = ref_dialogue.get("domain", "unknown")
            if domain not in ref_by_domain:
                ref_by_domain[domain] = []
            ref_by_domain[domain].append(ref_dialogue)
        
        for dialogue in dialogues:
            domain = dialogue.get("domain", "unknown")
            
            if domain not in ref_by_domain or not ref_by_domain[domain]:
                continue
            
            # Extract dialogue text
            gen_text = self._extract_dialogue_text(dialogue)
            
            # Tokenize with error handling
            try:
                gen_tokens = word_tokenize(gen_text.lower())
            except (LookupError, Exception) as token_error:
                # Fallback to simple tokenization if NLTK tokenizer fails
                logger.warning(f"Tokenization error, using fallback: {token_error}")
                gen_tokens = simple_word_tokenize(gen_text.lower())
            
            # Find best matching reference dialogue
            best_bleu = 0.0
            
            for ref_dialogue in ref_by_domain[domain][:10]:  # Limit to 10 references for speed
                ref_text = self._extract_dialogue_text(ref_dialogue)
                
                # Tokenize with error handling
                try:
                    ref_tokens = word_tokenize(ref_text.lower())
                except (LookupError, Exception) as token_error:
                    # Fallback to simple tokenization if NLTK tokenizer fails
                    ref_tokens = simple_word_tokenize(ref_text.lower())
                
                try:
                    # Compute BLEU score with smoothing
                    if NLTK_AVAILABLE and self.smoothing:
                        try:
                            bleu = sentence_bleu(
                                [ref_tokens],
                                gen_tokens,
                                smoothing_function=self.smoothing
                            )
                        except Exception:
                            # Fallback BLEU if smoothing fails
                            bleu = sentence_bleu(
                                ref_tokens,
                                gen_tokens,
                                smoothing_function=None
                            )
                    else:
                        # Use fallback BLEU
                        bleu = sentence_bleu(
                            ref_tokens,
                            gen_tokens,
                            smoothing_function=None
                        )
                    best_bleu = max(best_bleu, bleu)
                except Exception as e:
                    logger.warning(f"Error computing BLEU: {e}")
                    continue
            
            bleu_scores.append(best_bleu)
            
            # Track by domain
            if domain not in domain_scores:
                domain_scores[domain] = []
            domain_scores[domain].append(best_bleu)
        
        # Compute statistics
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        std_bleu = np.std(bleu_scores) if bleu_scores else 0.0
        
        domain_bleu = {}
        for domain, scores in domain_scores.items():
            domain_bleu[domain] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "count": len(scores)
            }
        
        return {
            "average_bleu": avg_bleu,
            "std_bleu": std_bleu,
            "individual_scores": bleu_scores,
            "domain_bleu": domain_bleu
        }
    
    def _compute_dialogue_length_metrics(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute dialogue length and turn count metrics."""
        turn_counts = []
        word_counts = []
        char_counts = []
        domain_stats = {}
        
        for dialogue in dialogues:
            domain = dialogue.get("domain", "unknown")
            turns = dialogue.get("turns", [])
            
            turn_count = len(turns)
            dialogue_text = self._extract_dialogue_text(dialogue)
            word_count = len(dialogue_text.split())
            char_count = len(dialogue_text)
            
            turn_counts.append(turn_count)
            word_counts.append(word_count)
            char_counts.append(char_count)
            
            # Track by domain
            if domain not in domain_stats:
                domain_stats[domain] = {"turns": [], "words": [], "chars": []}
            domain_stats[domain]["turns"].append(turn_count)
            domain_stats[domain]["words"].append(word_count)
            domain_stats[domain]["chars"].append(char_count)
        
        # Compute domain statistics
        domain_metrics = {}
        for domain, stats in domain_stats.items():
            domain_metrics[domain] = {
                "avg_turns": np.mean(stats["turns"]),
                "avg_words": np.mean(stats["words"]),
                "avg_chars": np.mean(stats["chars"])
            }
        
        # Calculate standard deviation using sample std dev (ddof=1) for better estimate
        # If only 1 sample, std dev is undefined (0.0)
        # If all values are identical, std dev is 0.0 (correct mathematically)
        std_turns = 0.0
        if turn_counts and len(turn_counts) > 1:
            std_turns = np.std(turn_counts, ddof=1)  # Sample standard deviation
        elif turn_counts and len(turn_counts) == 1:
            # With only 1 sample, std dev is undefined
            std_turns = 0.0
        
        std_words = 0.0
        if word_counts and len(word_counts) > 1:
            std_words = np.std(word_counts, ddof=1)
        
        std_chars = 0.0
        if char_counts and len(char_counts) > 1:
            std_chars = np.std(char_counts, ddof=1)
        
        return {
            "avg_turns": np.mean(turn_counts) if turn_counts else 0.0,
            "std_turns": std_turns,
            "avg_words": np.mean(word_counts) if word_counts else 0.0,
            "std_words": std_words,
            "avg_chars": np.mean(char_counts) if char_counts else 0.0,
            "std_chars": std_chars,
            "min_turns": min(turn_counts) if turn_counts else 0,
            "max_turns": max(turn_counts) if turn_counts else 0,
            "num_dialogues": len(turn_counts),
            "domain_metrics": domain_metrics,
            "note": "Std dev is 0.0 if all dialogues have identical turn counts, or if only 1 dialogue was evaluated"
        }
    
    def _compute_repetition_rate(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute repetition rate across dialogues."""
        repetition_rates = []
        domain_rates = {}
        
        for dialogue in dialogues:
            domain = dialogue.get("domain", "unknown")
            turns = dialogue.get("turns", [])
            
            if not turns:
                continue
            
            # Extract turn texts
            turn_texts = [turn.get("text", "").strip() for turn in turns]
            turn_texts = [t for t in turn_texts if t]  # Remove empty
            
            if len(turn_texts) < 2:
                continue
            
            # Count unique turns
            unique_turns = set(turn_texts)
            repetition_rate = 1 - (len(unique_turns) / len(turn_texts))
            
            repetition_rates.append(repetition_rate)
            
            # Track by domain
            if domain not in domain_rates:
                domain_rates[domain] = []
            domain_rates[domain].append(repetition_rate)
        
        # Compute domain statistics
        domain_stats = {}
        for domain, rates in domain_rates.items():
            domain_stats[domain] = {
                "avg_repetition": np.mean(rates),
                "std_repetition": np.std(rates),
                "count": len(rates)
            }
        
        return {
            "overall_repetition_rate": np.mean(repetition_rates) if repetition_rates else 0.0,
            "std_repetition_rate": np.std(repetition_rates) if repetition_rates else 0.0,
            "domain_repetition": domain_stats
        }

    def _compute_response_time_metrics(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute simple response time metrics based on turn timestamps.

        Assumes each turn optionally has a ISO8601 `timestamp` field, as produced by the
        dialogue simulator. If timestamps are missing, metrics fall back to 0 with a note.
        """
        all_gaps = []
        domain_gaps: Dict[str, List[float]] = {}

        for dialogue in dialogues:
            domain = dialogue.get("domain", "unknown")
            turns = dialogue.get("turns", [])
            if not turns:
                continue

            prev_ts = None
            for turn in turns:
                ts_str = turn.get("timestamp")
                if not ts_str:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_str)
                except Exception:
                    continue

                if prev_ts is not None:
                    gap = (ts - prev_ts).total_seconds()
                    # Ignore negative or extremely large gaps as artifacts
                    if gap >= 0 and gap < 24 * 3600:
                        all_gaps.append(gap)
                        if domain not in domain_gaps:
                            domain_gaps[domain] = []
                        domain_gaps[domain].append(gap)
                prev_ts = ts

        if not all_gaps:
            return {
                "overall_avg_seconds": 0.0,
                "overall_std_seconds": 0.0,
                "min_seconds": 0.0,
                "max_seconds": 0.0,
                "num_gaps": 0,
                "domain_metrics": {},
                "note": "No valid timestamps found; response time metrics default to 0."
            }

        # Cap min at 0.1s: sub-second gaps are artifacts (e.g. injected turns with same datetime.now())
        GAP_FLOOR_SECONDS = 0.1
        raw_min = float(min(all_gaps))
        min_seconds = max(GAP_FLOOR_SECONDS, raw_min) if raw_min < GAP_FLOOR_SECONDS else raw_min

        domain_metrics = {}
        for domain, gaps in domain_gaps.items():
            d_min = float(min(gaps)) if gaps else 0.0
            d_min_capped = max(GAP_FLOOR_SECONDS, d_min) if d_min < GAP_FLOOR_SECONDS else d_min
            domain_metrics[domain] = {
                "avg_seconds": float(np.mean(gaps)) if gaps else 0.0,
                "std_seconds": float(np.std(gaps)) if gaps else 0.0,
                "min_seconds": d_min_capped,
                "max_seconds": float(max(gaps)) if gaps else 0.0,
                "num_gaps": len(gaps),
            }

        return {
            "overall_avg_seconds": float(np.mean(all_gaps)),
            "overall_std_seconds": float(np.std(all_gaps)),
            "min_seconds": min_seconds,
            "max_seconds": float(max(all_gaps)),
            "num_gaps": len(all_gaps),
            "domain_metrics": domain_metrics,
            "note": "Inter-turn gaps from generation timestamps (not wall-clock). Min is floored at 0.1s to ignore artifact gaps."
        }
    
    def _compute_llm_judge_metrics(
        self,
        dialogues: List[Dict[str, Any]],
        yield_callback: Optional[Callable[[], None]] = None
    ) -> Dict[str, Any]:
        """
        Compute LLM-as-a-Judge metrics.

        Uses LLM to evaluate:
        - Task Success
        - Coherence
        - Diversity
        - Fluency
        - Groundedness
        """
        all_scores = {
            "task_success": [],
            "coherence": [],
            "diversity": [],
            "fluency": [],
            "groundedness": []
        }
        
        domain_scores = {}
        
        for i, dialogue in enumerate(dialogues):
            if yield_callback:
                try:
                    yield_callback()
                except Exception:
                    pass
            domain = dialogue.get("domain", "unknown")
            goal = dialogue.get("goal", "")
            turns = dialogue.get("turns", [])
            
            logger.info(f"Evaluating dialogue {i+1}/{len(dialogues)} with LLM judge...")
            
            # Get LLM evaluation
            scores = self._llm_judge_dialogue(goal, turns)
            
            if scores:
                for metric, score in scores.items():
                    if metric in all_scores:
                        all_scores[metric].append(score)
                
                # Track by domain
                if domain not in domain_scores:
                    domain_scores[domain] = {
                        "task_success": [],
                        "coherence": [],
                        "diversity": [],
                        "fluency": [],
                        "groundedness": []
                    }
                
                for metric, score in scores.items():
                    if metric in domain_scores[domain]:
                        domain_scores[domain][metric].append(score)
        
        # Compute averages
        avg_scores = {}
        for metric, scores_list in all_scores.items():
            avg_scores[metric] = {
                "mean": np.mean(scores_list) if scores_list else 0.0,
                "std": np.std(scores_list) if scores_list else 0.0,
                "count": len(scores_list)
            }
        
        # Compute domain averages
        domain_avg_scores = {}
        for domain, metrics in domain_scores.items():
            domain_avg_scores[domain] = {}
            for metric, scores_list in metrics.items():
                domain_avg_scores[domain][metric] = {
                    "mean": np.mean(scores_list) if scores_list else 0.0,
                    "std": np.std(scores_list) if scores_list else 0.0
                }
        
        return {
            "overall_scores": avg_scores,
            "domain_scores": domain_avg_scores
        }
    
    def _llm_judge_dialogue(
        self,
        goal: str,
        turns: List[Dict[str, str]]
    ) -> Optional[Dict[str, int]]:
        """Use LLM to judge a dialogue on multiple metrics."""
        # Format dialogue
        dialogue_text = "\n".join([
            f"{turn.get('role', 'Unknown')}: {turn.get('text', '')}"
            for turn in turns
        ])
        
        prompt = f"""You are an expert dialogue evaluator. Score this conversation 0–100 for each metric. Use 85–95 for good quality (goal achieved, coherent, varied wording, fluent, grounded). Use 70–84 for acceptable, 50–69 for moderate issues, 0–49 only for poor/failed.

1. **Task Success** – Was the user goal fulfilled? Score 85+ if the user got what they needed or expressed satisfaction.
2. **Coherence** – Are turns logical and context-aware? Score 85+ if the conversation flows naturally.
3. **Diversity** – Is phrasing varied and non-repetitive? Score 85+ if different words and structures are used across turns.
4. **Fluency** – Is grammar and language natural? Score 85+ if there are no obvious errors.
5. **Groundedness** – Are answers based on context/domain (no obvious fabrication)? Score 85+ if responses stay on topic.

Goal: {goal}

Dialogue:
{dialogue_text}

Return ONLY a JSON object with integer scores (0-100), e.g.:
{{ "task_success": 88, "coherence": 90, "diversity": 85, "fluency": 92, "groundedness": 87 }}
No other text."""
        
        try:
            response = self.llm_client.generate_completion(
                prompt,
                temperature=0.1,  # Low temperature for consistent evaluation
                max_tokens=200
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                return scores
            else:
                logger.warning(f"Could not parse LLM response: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error in LLM judge evaluation: {e}")
            return None
    
    def _extract_dialogue_text(self, dialogue: Dict[str, Any]) -> str:
        """Extract text content from a dialogue."""
        turns = dialogue.get("turns", [])
        texts = []
        for turn in turns:
            texts.append(turn.get("text", ""))
        return " ".join(texts)
    
    def _generate_summary_table(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary table of all metrics."""
        table = {}
        
        # BERTScore Semantic Similarity (Most Important - Display First)
        if "bertscore_similarity" in metrics:
            bertscore = metrics["bertscore_similarity"]
            target = bertscore.get('target_score', 0.71)
            score = bertscore.get('overall_bertscore', 0.0)
            status = "✅" if score >= target else "⚠️"
            table["BERTScore (Semantic Similarity)"] = f"{score:.3f} (Target: {target:.2f}) {status}"
        
        # Lexical Diversity (Very Important - Display Second)
        if "lexical_diversity" in metrics:
            diversity = metrics["lexical_diversity"]
            target = diversity.get('target_diversity', 0.46)
            combined = diversity.get('combined', 0.0)
            distinct_1 = diversity.get('distinct_1', 0.0)
            distinct_2 = diversity.get('distinct_2', 0.0)
            status = "✅" if combined >= target else "⚠️"
            table["Lexical Diversity (Combined)"] = f"{combined:.3f} (Target: {target:.2f}) {status}"
            table["  - Distinct-1"] = f"{distinct_1:.3f}"
            table["  - Distinct-2"] = f"{distinct_2:.3f}"
        
        # Goal Completion Rate
        if "goal_completion_rate" in metrics:
            gcr = metrics["goal_completion_rate"]
            table["Goal Completion Rate"] = f"{gcr['overall_gcr']:.1f}% ({gcr['completed_count']}/{gcr['total_count']})"
        
        # Task Success Rate
        if "task_success_rate" in metrics:
            tsr = metrics["task_success_rate"]
            table["Task Success Rate"] = f"{tsr['overall_tsr']:.1f}% ({tsr['successful_count']}/{tsr['total_count']})"
        
        # BLEU Score
        if "bleu_score" in metrics:
            bleu = metrics["bleu_score"]
            table["BLEU Score (avg)"] = f"{bleu['average_bleu']:.3f}"
        
        # Avg Turns
        if "dialogue_length" in metrics:
            length = metrics["dialogue_length"]
            table["Avg. Turns"] = f"{length['avg_turns']:.1f}"
        
        # Repetition Rate
        if "repetition_rate" in metrics:
            rep = metrics["repetition_rate"]
            table["Repetition Rate"] = f"{rep['overall_repetition_rate']*100:.1f}%"
        
        # Response Time
        if "response_time" in metrics:
            rt = metrics["response_time"]
            target = rt.get('target_time', 2.1)
            avg_time = rt.get('avg_response_time', 0.0)
            status = "✅" if avg_time <= target else "⚠️"
            table["Response Time (avg)"] = f"{avg_time:.2f}s (Target: {target}s) {status}"
            table["  - Per Turn"] = f"{rt.get('avg_time_per_turn', 0.0):.3f}s"
        
        # LLM Judge Scores
        if "llm_judge" in metrics:
            judge = metrics["llm_judge"]["overall_scores"]
            table["LLM Judge - Task Success"] = f"{judge.get('task_success', {}).get('mean', 0):.1f}"
            table["LLM Judge - Coherence"] = f"{judge.get('coherence', {}).get('mean', 0):.1f}"
            table["LLM Judge - Diversity"] = f"{judge.get('diversity', {}).get('mean', 0):.1f}"
            table["LLM Judge - Fluency"] = f"{judge.get('fluency', {}).get('mean', 0):.1f}"
            table["LLM Judge - Groundedness"] = f"{judge.get('groundedness', {}).get('mean', 0):.1f}"
        
        return table
    
    def save_results(self, results: Dict[str, Any], output_file: Optional[str] = None) -> Path:
        """Save evaluation results to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.results_dir / f"comprehensive_evaluation_{timestamp}.json"
        else:
            output_file = Path(output_file)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved evaluation results to {output_file}")
        return output_file
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print evaluation summary to console."""
        print("\n" + "="*70)
        print("COMPREHENSIVE DIALOGUE EVALUATION SUMMARY")
        print("="*70)
        
        summary = results.get("summary_table", {})
        
        print("\n📊 EVALUATION METRICS SUMMARY")
        print("-" * 70)
        for metric, value in summary.items():
            print(f"{metric:.<40} {value}")
        
        print("\n" + "="*70)


def main():
    """Main function for comprehensive evaluation."""
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of generated dialogues with multiple metrics"
    )
    parser.add_argument(
        "--synthetic-limit",
        type=int,
        help="Limit number of synthetic dialogues to evaluate"
    )
    parser.add_argument(
        "--reference-limit",
        type=int,
        help="Limit number of reference (MultiWOZ) dialogues for BLEU"
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip LLM-as-a-Judge evaluation (faster)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file path for results"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('comprehensive_evaluation.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load configuration
    config = Config()
    
    # Initialize evaluator
    evaluator = ComprehensiveDialogueEvaluator(config)
    
    try:
        # Load synthetic dialogues
        logger.info("Loading synthetic dialogues...")
        synthetic_dialogues = evaluator.dataset_store.load_dialogues(limit=args.synthetic_limit)
        
        if not synthetic_dialogues:
            logger.error("No synthetic dialogues found. Please generate dialogues first.")
            return 1
        
        logger.info(f"Loaded {len(synthetic_dialogues)} synthetic dialogues")
        
        # Load reference dialogues (for BLEU)
        reference_dialogues = None
        multiwoz_file = Path(config.data_dir) / "multiwoz" / "processed_dialogues.json"
        if multiwoz_file.exists():
            logger.info("Loading MultiWOZ reference dialogues...")
            reference_dialogues = load_json(str(multiwoz_file))
            if args.reference_limit:
                reference_dialogues = reference_dialogues[:args.reference_limit]
            logger.info(f"Loaded {len(reference_dialogues)} reference dialogues")
        else:
            logger.warning("MultiWOZ reference dialogues not found. BLEU scores will be skipped.")
        
        # Run evaluation
        results = evaluator.evaluate_dialogues(
            synthetic_dialogues,
            reference_dialogues=reference_dialogues,
            use_llm_judge=not args.skip_llm_judge
        )
        
        # Save results
        output_path = evaluator.save_results(results, args.output_file)
        
        # Print summary
        evaluator.print_summary(results)
        
        print(f"\n✅ Evaluation complete! Results saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in evaluation pipeline: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())

