"""
Complete GoalConvo dialogue pipeline: goal → planner (per agent) → user → support →
reflection (utterance-level, summarized per turn) → memory update → goal completion check.

Turn-by-turn state is accumulated in ``pipeline_turns`` for export and structured logs.
"""

import copy
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime

from ..config import Config
from ..llm_client import LLMClient
from ..utils import generate_dialogue_id, format_conversation_history, calculate_similarity
from ..agents import PlannerAgent, ReflectionAgent, SupportAgent, UserAgent
from ..experiments.tracking import merge_repro_metadata
from ..memory import DialogueMemory
from ..research.adaptive import both_planners_near_complete
from ..research.contradictions import analyze_contradictions
from ..research.decomposition import run_dynamic_goal_decomposition
from ..research.summarization import update_rolling_summary
from .dialogue_helpers import (
    detect_repetition_loop,
    fallback_supportbot_response,
    fallback_user_response,
    inject_ref_if_booking_claim_has_no_ref,
    iter_dialogue_turns,
    last_turn_is_open_request,
    venue_from_goal,
)

logger = logging.getLogger(__name__)


class DialoguePipeline:
    """Simulates goal-oriented dialogues via composable research agents."""

    def __init__(self, config: Config, llm_client: LLMClient):
        self.config = config
        self.llm_client = llm_client
        self.planner_agent = PlannerAgent(config, llm_client)
        self.user_agent = UserAgent(config, llm_client)
        self.support_agent = SupportAgent(config, llm_client)
        self.reflection_agent = ReflectionAgent(config, llm_client)

    @staticmethod
    def export_dialogue_json(dialogue_data: Dict[str, Any], path: str, indent: int = 2) -> None:
        """Persist a full dialogue document (``turns``, ``pipeline_turns``, ``metadata``) to JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fp:
            json.dump(dialogue_data, fp, indent=indent, ensure_ascii=False, default=str)

    @staticmethod
    def _planner_turn_payload(
        sb_meta: Optional[Dict[str, Any]] = None,
        u_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if sb_meta and sb_meta.get("structured_planner") is not None:
            out["support_phase"] = sb_meta.get("structured_planner")
        if u_meta and u_meta.get("structured_planner") is not None:
            out["user_phase"] = u_meta.get("structured_planner")
        return out

    @staticmethod
    def _single_reflection_side(meta: Optional[Dict[str, Any]]) -> str:
        if not meta:
            return "—"
        ur = meta.get("utterance_reflection") or {}
        attempts = ur.get("attempts") or []
        if not attempts:
            return "—"
        verdict = (attempts[-1] or {}).get("verdict") or {}
        return f"score={verdict.get('score')} accepted={verdict.get('accepted')}"

    def _reflection_score_string(
        self,
        sb_meta: Optional[Dict[str, Any]],
        u_meta: Optional[Dict[str, Any]],
    ) -> str:
        parts: List[str] = []
        if sb_meta is not None:
            parts.append("support:" + self._single_reflection_side(sb_meta))
        if u_meta is not None:
            parts.append("user:" + self._single_reflection_side(u_meta))
        return "; ".join(parts) if parts else ""

    @staticmethod
    def _reflection_attempt_bonus(config: Config, mem: DialogueMemory) -> int:
        if not getattr(config, "research_retry_correction", False):
            return 0
        return 1 if (mem.research_consistency_hint or "").strip() else 0

    def _pipeline_debug(self, event: str, **fields: Any) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        payload: Dict[str, Any] = {"event": event, **fields}
        try:
            logger.debug("%s", json.dumps(payload, default=str))
        except (TypeError, ValueError):
            logger.debug("%s", payload)

    def simulate_dialogue(
        self,
        experience_data: Dict[str, Any],
        max_turns: Optional[int] = None,
        progress_callback: Optional[Callable[..., None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        show_progress: bool = False,
        experiment_run: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Full GoalConvo pipeline: goal input → planner (inside agent calls) → Support → User →
        reflection summary → memory update → goal completion check each iteration.

        ``pipeline_turns`` stores one record per iteration (plus turn 0 for the initial user).

        Args:
            experience_data: Must include ``goal`` and ``context``.
            max_turns: Max dialogue iterations (each adds SupportBot + User).
            progress_callback: Optional ``(turns_list, message)`` after substantive steps.
            on_error: Optional hook for recoverable errors mid-loop.
            show_progress: If True (or ``Config.pipeline_show_progress``), tqdm progress bar.
            experiment_run: Optional :class:`goalconvo.experiments.tracking.ExperimentRun`;
                when set, persists each completed dialogue and artifact slices under the run directory.
        """
        generation_start_time = time.time()

        max_turns = max_turns or self.config.max_turns
        goal = experience_data["goal"]
        context = experience_data["context"]
        domain = experience_data.get("domain", "general")
        user_persona = experience_data.get("user_persona", "General user")
        first_utterance = experience_data.get("first_utterance", "")

        dialogue_id = generate_dialogue_id()
        turns: List[Dict[str, Any]] = []
        conversation_history: List[Dict[str, Any]] = []
        dialogue_memory = DialogueMemory()
        rl_trajectory: List[float] = []
        memory_turn_cursor = 0
        pipeline_turns: List[Dict[str, Any]] = []
        stop_reason: Optional[str] = "in_progress"

        conversation_history.append({
            "role": "System",
            "text": f"Domain: {domain}\nUser Goal: {goal}",
        })

        self._pipeline_debug(
            "pipeline.dialogue.start",
            dialogue_id=dialogue_id,
            domain=domain,
            max_turns=max_turns,
            goal_preview=(goal or "")[:160],
        )

        experience_data = dict(experience_data)
        research_meta: Dict[str, Any] = {}
        if getattr(self.config, "research_dynamic_goal_decomposition", False):
            try:
                decomp = run_dynamic_goal_decomposition(
                    self.llm_client,
                    self.config,
                    goal=goal,
                    context=context,
                    domain=domain,
                )
                experience_data["dynamic_subgoals"] = list(decomp.get("milestones") or [])
                research_meta["decomposition"] = decomp
            except Exception as e:
                logger.warning("research_dynamic_goal_decomposition failed: %s", e)
                research_meta["decomposition_error"] = str(e)

        turn_cap = max_turns
        if getattr(self.config, "research_adaptive_dialogue_length", False):
            turn_cap = max_turns + max(0, int(getattr(self.config, "research_adaptive_extend_turns", 0)))

        gen_boot_ms = 0.0
        u_meta_boot: Dict[str, Any] = {}
        if first_utterance:
            user_turn = {
                "role": "User",
                "text": first_utterance,
                "timestamp": datetime.now().isoformat(),
                "metadata": {},
            }
            turns.append(user_turn)
            conversation_history.append(user_turn)
            if progress_callback:
                progress_callback(list(turns), "First user utterance")
        else:
            t_u0 = time.perf_counter()
            user_response, u_meta_boot = self._generate_user_turn(
                goal, context, user_persona, conversation_history, domain, experience_data,
                dialogue_memory,
            )
            gen_boot_ms = (time.perf_counter() - t_u0) * 1000
            user_turn = {
                "role": "User",
                "text": user_response,
                "timestamp": datetime.now().isoformat(),
                "metadata": u_meta_boot,
            }
            turns.append(user_turn)
            conversation_history.append(user_turn)
            if progress_callback:
                progress_callback(list(turns), "Generated initial user turn")

        t_m0 = time.perf_counter()
        if getattr(self.config, "agent_memory_enabled", True) and turns:
            dialogue_memory.refresh_from_recent_turns(
                self.llm_client, self.config, turns[memory_turn_cursor:], domain, goal
            )
        mem0_ms = (time.perf_counter() - t_m0) * 1000
        memory_turn_cursor = len(turns)

        pipeline_turns.append({
            "turn": 0,
            "planner_output": self._planner_turn_payload(None, u_meta_boot),
            "user_response": user_turn["text"],
            "support_response": "",
            "reflection_score": self._reflection_score_string(None, u_meta_boot),
            "memory_snapshot": copy.deepcopy(dialogue_memory.get_memory_state_for_planner()),
            "timings_ms": {
                "user_generation": round(gen_boot_ms, 2),
                "memory_update": round(mem0_ms, 2),
            },
        })
        self._pipeline_debug(
            "pipeline.turn.bootstrap",
            dialogue_id=dialogue_id,
            timings_ms=pipeline_turns[-1]["timings_ms"],
        )

        last_goal_check_turn = 0
        min_turns_required = self.config.min_turns
        min_iterations = max(1, (min_turns_required - 1 + 1) // 2)
        logger.info(
            "Dialogue %s: target ≥ %s utterance turns (≈%s iterations, pair-cap %s)",
            dialogue_id,
            min_turns_required,
            min_iterations,
            max_turns,
        )
        if turn_cap != max_turns:
            logger.info(
                "Dialogue %s: research adaptive turn_cap=%s (base max_turns=%s)",
                dialogue_id,
                turn_cap,
                max_turns,
            )

        use_progress = show_progress or getattr(self.config, "pipeline_show_progress", False)
        progress_desc = f"dialogue {dialogue_id[:8]}"

        for turn_num in iter_dialogue_turns(turn_cap, use_progress, desc=progress_desc):
            try:
                dialogue_memory.research_consistency_hint = ""
                if getattr(self.config, "research_contradiction_detection", False) and len(turns) >= 2:
                    tail_hist = [t for t in turns[-12:] if t.get("role") in ("User", "SupportBot")]
                    tail_txt = format_conversation_history(tail_hist)
                    if tail_txt.strip():
                        try:
                            cand = analyze_contradictions(
                                self.llm_client,
                                self.config,
                                goal=goal,
                                transcript=tail_txt,
                                memory_blob=dialogue_memory.get_memory_state_for_planner(),
                            )
                            hint = (cand.get("repair_hint") or "").strip()
                            lines: List[str] = []
                            if hint:
                                lines.append(hint)
                            for c in cand.get("contradictions") or []:
                                if isinstance(c, dict) and (c.get("description") or "").strip():
                                    sev = str(c.get("severity") or "?")
                                    lines.append(f"[{sev}] {c['description'].strip()}")
                            if lines:
                                dialogue_memory.research_consistency_hint = "\n".join(lines)[:2000]
                            research_meta.setdefault("contradiction_passes", []).append(
                                {"turn": turn_num, "result": cand}
                            )
                        except Exception as e:
                            logger.warning("research_contradiction_detection failed: %s", e)

                self._pipeline_debug(
                    "pipeline.phase",
                    dialogue_id=dialogue_id,
                    turn=turn_num,
                    phase="support_generate",
                )
                t_s = time.perf_counter()
                supportbot_response, sb_meta = self._generate_supportbot_turn(
                    goal, context, conversation_history, domain, experience_data,
                    dialogue_memory,
                )
                support_ms = (time.perf_counter() - t_s) * 1000

                supportbot_turn = {
                    "role": "SupportBot",
                    "text": supportbot_response,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": sb_meta,
                }
                turns.append(supportbot_turn)
                conversation_history.append(supportbot_turn)
                if progress_callback:
                    progress_callback(list(turns), f"Generating SupportBot turn {len(turns)}")

                self._pipeline_debug(
                    "pipeline.phase",
                    dialogue_id=dialogue_id,
                    turn=turn_num,
                    phase="user_generate",
                    support_ms=round(support_ms, 2),
                )
                t_u = time.perf_counter()
                user_response, u_meta = self._generate_user_turn(
                    goal, context, user_persona, conversation_history, domain, experience_data,
                    dialogue_memory,
                )
                user_ms = (time.perf_counter() - t_u) * 1000

                user_turn = {
                    "role": "User",
                    "text": user_response,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": u_meta,
                }
                turns.append(user_turn)
                conversation_history.append(user_turn)
                if progress_callback:
                    progress_callback(list(turns), f"Generating User turn {len(turns)}")

                t_mem = time.perf_counter()
                if getattr(self.config, "agent_memory_enabled", True):
                    new_turns = turns[memory_turn_cursor:]
                    memory_turn_cursor = len(turns)
                    dialogue_memory.refresh_from_recent_turns(
                        self.llm_client, self.config, new_turns, domain, goal
                    )
                mem_ms = (time.perf_counter() - t_mem) * 1000

                if getattr(self.config, "research_memory_summarization", False):
                    every = max(1, int(getattr(self.config, "research_summarize_every_n_iters", 2)))
                    if turn_num % every == 0:
                        chunk = format_conversation_history(
                            [t for t in turns[-12:] if t.get("role") in ("User", "SupportBot")]
                        )
                        if chunk.strip():
                            try:
                                dialogue_memory.rolling_dialogue_summary = update_rolling_summary(
                                    self.llm_client,
                                    self.config,
                                    goal=goal,
                                    prior_summary=dialogue_memory.rolling_dialogue_summary,
                                    new_transcript_chunk=chunk,
                                )
                            except Exception as e:
                                logger.warning("research_memory_summarization failed: %s", e)

                refl_score = self._reflection_score_string(sb_meta, u_meta)
                self._pipeline_debug(
                    "pipeline.turn.memory_reflection",
                    dialogue_id=dialogue_id,
                    turn=turn_num,
                    reflection_score=refl_score,
                    memory_refresh_ms=round(mem_ms, 2),
                )

                pipeline_turns.append({
                    "turn": turn_num,
                    "planner_output": self._planner_turn_payload(sb_meta, u_meta),
                    "user_response": user_response,
                    "support_response": supportbot_response,
                    "reflection_score": refl_score,
                    "memory_snapshot": copy.deepcopy(dialogue_memory.get_memory_state_for_planner()),
                    "timings_ms": {
                        "support_generation": round(support_ms, 2),
                        "user_generation": round(user_ms, 2),
                        "memory_update": round(mem_ms, 2),
                    },
                })
                logger.info(
                    "pipeline_turn=%s dialogue=%s reflection=%s gen_ms support/user=%s/%s mem_ms=%s",
                    turn_num,
                    dialogue_id[:12],
                    (refl_score or "—")[:80],
                    round(support_ms, 1),
                    round(user_ms, 1),
                    round(mem_ms, 1),
                )

                if getattr(self.config, "rl_lite_enabled", True):
                    rl_step = self._compute_rl_lite_step(turns, goal)
                    u_meta["rl_lite"] = rl_step
                    rl_trajectory.append(rl_step["reward"])

                if len(turns) < min_turns_required:
                    logger.debug(
                        "Dialogue %s: %s/%s utterance turns — below minimum; continue",
                        dialogue_id,
                        len(turns),
                        min_turns_required,
                    )
                    continue

                if (
                    getattr(self.config, "research_adaptive_dialogue_length", False)
                    and both_planners_near_complete(sb_meta, u_meta)
                ):
                    stop_reason = "adaptive_planner_near_complete"
                    logger.info(
                        "Dialogue %s: adaptive early stop (dual planner near_complete) at pipeline turn %s",
                        dialogue_id,
                        turn_num,
                    )
                    break

                if detect_repetition_loop(turns):
                    logger.info(
                        "Dialogue %s: repetition loop detected at %s turns; forcing completion.",
                        dialogue_id,
                        len(turns),
                    )
                    stop_reason = "repetition_loop"
                    venue = venue_from_goal(goal, domain)
                    if domain == "hotel":
                        confirm = f"Your booking at {venue} is confirmed for 2 nights. Your confirmation number is {venue.replace(' ', '')[:8].upper()}-001. Is there anything else?"
                    elif domain == "restaurant":
                        confirm = f"Your reservation at {venue} is confirmed for dinner. Reference: {venue.replace(' ', '')[:6].upper()}-RES. Is there anything else?"
                    elif domain == "taxi":
                        confirm = f"Your taxi with Swift Cabs from {venue} is confirmed for 3:00 PM pickup at the main entrance. Reference: TAXI-{venue.replace(' ', '')[:4].upper()}-001. Anything else?"
                    elif domain == "train":
                        confirm = f"Your train at 16:30 is confirmed. Reference: TRN-{venue.replace(' ', '')[:4].upper()}-001. Anything else I can help with?"
                    else:
                        confirm = f"Your request for {goal[:60]}{'...' if len(goal) > 60 else ''} is all set. Anything else I can help with?"
                    turns.append({
                        "role": "SupportBot",
                        "text": confirm,
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {"forced_completion": True},
                    })
                    conversation_history.append(turns[-1])
                    turns.append({
                        "role": "User",
                        "text": "That's all, thanks!",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {"forced_completion": True},
                    })
                    conversation_history.append(turns[-1])
                    if progress_callback:
                        progress_callback(list(turns), "Completed (loop broken)")
                    break
                
                goal_check_interval = 3
                if (
                    len(turns) - last_goal_check_turn >= goal_check_interval
                    and turn_num < turn_cap
                ):
                    t_gc = time.perf_counter()
                    if self.reflection_agent.goal_satisfied_keywords(goal, conversation_history):
                        stop_reason = "goal_satisfied_keywords"
                        logger.info(
                            "Goal satisfied (keyword check) after %s turns dialogue=%s",
                            len(turns),
                            dialogue_id,
                        )
                        gc_ms = (time.perf_counter() - t_gc) * 1000
                        pipeline_turns[-1].setdefault("timings_ms", {})[
                            "goal_completion_check"
                        ] = round(gc_ms, 2)
                        self._pipeline_debug(
                            "pipeline.goal_check",
                            dialogue_id=dialogue_id,
                            turn=turn_num,
                            result=stop_reason,
                            ms=round(gc_ms, 2),
                        )
                        break
                    try:
                        if self.reflection_agent.goal_satisfied_llm(goal, conversation_history):
                            stop_reason = "goal_satisfied_llm"
                            logger.info(
                                "Goal satisfied (LLM check) after %s turns dialogue=%s",
                                len(turns),
                                dialogue_id,
                            )
                            gc_ms = (time.perf_counter() - t_gc) * 1000
                            pipeline_turns[-1].setdefault("timings_ms", {})[
                                "goal_completion_check"
                            ] = round(gc_ms, 2)
                            self._pipeline_debug(
                                "pipeline.goal_check",
                                dialogue_id=dialogue_id,
                                turn=turn_num,
                                result=stop_reason,
                                ms=round(gc_ms, 2),
                            )
                            break
                    except Exception as e:
                        logger.warning(
                            "Goal satisfaction LLM check failed: %s. Using keywords only.",
                            e,
                        )
                    gc_ms = (time.perf_counter() - t_gc) * 1000
                    last_goal_check_turn = len(turns)
                    pipeline_turns[-1].setdefault("timings_ms", {})[
                        "goal_completion_check"
                    ] = round(gc_ms, 2)
                    self._pipeline_debug(
                        "pipeline.goal_check",
                        dialogue_id=dialogue_id,
                        turn=turn_num,
                        result="continue",
                        ms=round(gc_ms, 2),
                    )
                
            except Exception as e:
                logger.error(f"Error in turn {turn_num} for dialogue {dialogue_id}: {e}")
                if on_error:
                    try:
                        on_error(str(e))
                    except Exception:
                        pass
                # CRITICAL: Don't break if we haven't reached min_turns yet
                # Add only the missing turn (avoid double SupportBot or double User)
                if len(turns) < min_turns_required:
                    logger.warning(f"Error occurred but only {len(turns)}/{min_turns_required} turns generated. Using fallback to continue.")
                    last_role = turns[-1].get("role") if turns else None
                    # Exception happened either during SupportBot or User generation in this iteration.
                    # If we just appended SupportBot, we failed on User -> add only User.
                    # If last turn is User (from previous iteration), we failed on SupportBot -> add only SupportBot.
                    if last_role == "SupportBot":
                        user_turn = {
                            "role": "User",
                            "text": fallback_user_response(conversation_history, goal, domain),
                            "timestamp": datetime.now().isoformat(),
                            "metadata": {"fallback": True},
                        }
                        turns.append(user_turn)
                        conversation_history.append(user_turn)
                    else:
                        supportbot_turn = {
                            "role": "SupportBot",
                            "text": fallback_supportbot_response(goal, conversation_history, domain),
                            "timestamp": datetime.now().isoformat(),
                            "metadata": {"fallback": True},
                        }
                        turns.append(supportbot_turn)
                        conversation_history.append(supportbot_turn)
                    continue  # Continue loop instead of breaking
                else:
                    logger.warning("Error after reaching min_turns. Stopping dialogue generation.")
                    stop_reason = "error"
                    break

        if stop_reason == "in_progress":
            stop_reason = "max_turns"
        # Even if goal was satisfied early, we need minimum turns for quality
        while len(turns) < min_turns_required:
            logger.warning(f"Dialogue {dialogue_id}: Only {len(turns)}/{min_turns_required} turns generated. Adding fallback turns to reach minimum.")
            
            # Add SupportBot turn if needed
            if len([t for t in turns if t.get("role") == "SupportBot"]) < min_turns_required // 2:
                supportbot_turn = {
                    "role": "SupportBot",
                    "text": fallback_supportbot_response(goal, conversation_history, domain),
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {"fallback": True},
                }
                turns.append(supportbot_turn)
                conversation_history.append(supportbot_turn)
            
            # Add User turn if needed
            if len(turns) < min_turns_required:
                user_turn = {
                    "role": "User",
                    "text": fallback_user_response(conversation_history, goal, domain),
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {"fallback": True},
                }
                turns.append(user_turn)
                conversation_history.append(user_turn)
        
        # Final check - log warning if we still don't have enough turns (shouldn't happen)
        if len(turns) < min_turns_required:
            logger.error(f"CRITICAL: Dialogue {dialogue_id} has only {len(turns)} turns, required {min_turns_required}. This should not happen!")

        # CRITICAL: Never end on an open user question—add SupportBot answer then user satisfaction so dialogue closes properly
        if last_turn_is_open_request(turns) and len(turns) < turn_cap * 2:
            try:
                closing_bot, closing_meta = self._generate_supportbot_turn(
                    goal, context, conversation_history, domain, experience_data,
                    dialogue_memory,
                )
            except Exception as e:
                logger.warning(f"Final SupportBot turn failed: {e}; using fallback.")
                closing_bot = fallback_supportbot_response(goal, conversation_history, domain)
                closing_meta = {"fallback": True}
            turns.append({
                "role": "SupportBot",
                "text": closing_bot,
                "timestamp": datetime.now().isoformat(),
                "metadata": closing_meta,
            })
            conversation_history.append(turns[-1])
            turns.append({
                "role": "User",
                "text": "Thank you, that's perfect!",
                "timestamp": datetime.now().isoformat(),
                "metadata": {"closing_template": True},
            })
            conversation_history.append(turns[-1])
            logger.info(f"Dialogue {dialogue_id}: added closing SupportBot answer and user satisfaction so dialogue does not end on open user question.")

        # Post-process: if any SupportBot turn claims booking/reservation but has no ref number, inject one
        inject_ref_if_booking_claim_has_no_ref(turns, goal, domain)
        
        generation_end_time = time.time()
        generation_duration = generation_end_time - generation_start_time

        max_turns_reached = stop_reason == "max_turns"
        api_model = self.llm_client.api_config.get("model", self.config.mistral_model)
        api_provider = self.llm_client.api_config.get("provider", "")
        metadata = {
            "num_turns": len(turns),
            "generated_at": datetime.now().isoformat(),
            "model_version": api_model,
            "llm_provider": api_provider,
            "max_turns_reached": max_turns_reached,
            "min_turns_met": len(turns) >= min_turns_required,
            "generation_time_seconds": round(generation_duration, 3),
            "generation_start_time": datetime.fromtimestamp(generation_start_time).isoformat(),
            "generation_end_time": datetime.fromtimestamp(generation_end_time).isoformat(),
            "stop_reason": stop_reason,
            "pipeline_iterations": max_turns,
            "research_turn_cap": turn_cap,
        }
        rf = {
            "hierarchical_subgoals": bool(getattr(self.config, "research_hierarchical_subgoals", False)),
            "dynamic_goal_decomposition": bool(getattr(self.config, "research_dynamic_goal_decomposition", False)),
            "memory_summarization": bool(getattr(self.config, "research_memory_summarization", False)),
            "adaptive_dialogue_length": bool(getattr(self.config, "research_adaptive_dialogue_length", False)),
            "contradiction_detection": bool(getattr(self.config, "research_contradiction_detection", False)),
            "retry_correction": bool(getattr(self.config, "research_retry_correction", False)),
        }
        if any(rf.values()) or research_meta:
            metadata["research"] = {**research_meta, "flags": rf}
        if experience_data.get("subgoals") or experience_data.get("constraints"):
            metadata["goal_complexity"] = len(experience_data.get("subgoals") or []) + len(experience_data.get("constraints") or {})
        if experience_data.get("user_persona_traits"):
            metadata["user_persona_traits"] = experience_data["user_persona_traits"]
        if experience_data.get("supportbot_style"):
            metadata["supportbot_style"] = experience_data["supportbot_style"]
        if rl_trajectory:
            metadata["rl_lite"] = {
                "step_rewards": rl_trajectory,
                "mean_reward": round(sum(rl_trajectory) / len(rl_trajectory), 4),
                "weights": {
                    "goal": getattr(self.config, "rl_goal_weight", 0.6),
                    "coherence": getattr(self.config, "rl_coherence_weight", 0.4),
                },
            }
        dialogue_data = {
            "dialogue_id": dialogue_id,
            "goal": goal,
            "domain": experience_data.get("domain", "unknown"),
            "context": context,
            "user_persona": user_persona,
            "turns": turns,
            "pipeline_turns": pipeline_turns,
            "metadata": metadata,
        }

        pfv = getattr(experiment_run, "_prompt_versions", None) if experiment_run is not None else None
        exp_rid = getattr(experiment_run, "run_id", None) if experiment_run is not None else None
        merge_repro_metadata(
            dialogue_data,
            self.config,
            self.llm_client,
            prompt_versions=pfv,
            experiment_run_id=exp_rid,
        )
        if experiment_run is not None:
            experiment_run.record_dialogue(dialogue_data)

        logger.info(
            "Generated dialogue %s utterance_turns=%s pipeline_records=%s stop=%s in %.2fs",
            dialogue_id,
            len(turns),
            len(pipeline_turns),
            stop_reason,
            generation_duration,
        )
        self._pipeline_debug(
            "pipeline.dialogue.end",
            dialogue_id=dialogue_id,
            stop_reason=stop_reason,
            generation_time_seconds=round(generation_duration, 3),
            pipeline_turns=len(pipeline_turns),
        )
        return dialogue_data

    def _structured_planning_block(
        self,
        goal: str,
        context: str,
        domain: str,
        history: List[Dict[str, Any]],
        mem: DialogueMemory,
        acting_role: str,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Run ``PlannerAgent`` to obtain JSON state, then format it for speaker prompts.

        Returns (prompt_block, parsed_plan_dict_or_None).
        """
        if not getattr(self.config, "agent_planning_enabled", True):
            return "", None
        if not getattr(self.config, "structured_planner_enabled", True):
            return "", None
        last_sp = history[-1].get("role") if history else None
        conv_turns = [t for t in history if t.get("role") in ("User", "SupportBot")]
        dialogue_state = {
            "acting_role": acting_role,
            "last_speaker": last_sp,
            "turn_count": len(conv_turns),
            "domain": domain,
        }
        memory_state = mem.get_memory_state_for_planner()
        pres = self.planner_agent.generate(
            goal=goal,
            context=context,
            domain=domain,
            dialogue_history=history,
            dialogue_state=dialogue_state,
            memory_state=memory_state,
        )
        parsed = pres.metadata.get("parsed_plan")
        if isinstance(parsed, dict):
            self.planner_agent.update_memory(
                mem,
                history[-2:] if len(history) >= 2 else history,
                parsed_plan=parsed,
            )
            block = self.planner_agent.format_planning_prompt_block(parsed)
            return block, parsed
        return "", None

    def _generate_user_turn(
        self,
        goal: str,
        context: str,
        user_persona: str,
        history: List[Dict[str, str]],
        domain: str = "general",
        experience_data: Optional[Dict[str, Any]] = None,
        dialogue_memory: Optional[DialogueMemory] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        mem = dialogue_memory or DialogueMemory()
        if getattr(self.config, "single_agent_support_only", False):
            conv_users = len([t for t in history if t.get("role") == "User"])
            templates = [
                "I understand. What would you recommend next?",
                "Yes, that works for me.",
                "Please go ahead with that.",
                "Could you share confirmation details?",
                "That sounds good—thank you.",
            ]
            text = templates[conv_users % len(templates)]
            return text, {"single_agent_template": True, "ablation": True}

        mem_sec = mem.memory_section_text(getattr(self.config, "agent_memory_enabled", True))
        plan_block, planner_plan = self._structured_planning_block(
            goal, context, domain, history, mem, acting_role="User"
        )
        mem_state = mem.get_memory_state_for_planner()
        bonus = self._reflection_attempt_bonus(self.config, mem)

        def _user_once(hint: Optional[str] = None):
            return self.user_agent.generate(
                goal=goal,
                context=context,
                user_persona=user_persona,
                history=history,
                domain=domain,
                experience_data=experience_data,
                user_memory=mem.user_facts,
                support_memory=mem.support_facts,
                memory_section=mem_sec,
                structured_planning_block=plan_block or None,
                reflection_repair_hint=hint,
            )

        if getattr(self.config, "reflection_on_utterances_enabled", True):
            text, u_meta = self.reflection_agent.run_reflected_generation(
                role="User",
                goal=goal,
                dialogue_history=history,
                memory_state=mem_state,
                domain=domain,
                generate_one=_user_once,
                reflection_attempt_bonus=bonus,
            )
        else:
            out = _user_once(None)
            text, u_meta = out.text, dict(out.metadata)
        if planner_plan is not None:
            u_meta["structured_planner"] = planner_plan
        self.user_agent.update_memory(mem, history[-2:] if len(history) >= 2 else history, goal=goal)
        return text, u_meta

    def _generate_supportbot_turn(
        self,
        goal: str,
        context: str,
        history: List[Dict[str, str]],
        domain: str = "general",
        experience_data: Optional[Dict[str, Any]] = None,
        dialogue_memory: Optional[DialogueMemory] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        mem = dialogue_memory or DialogueMemory()
        mem_sec = mem.memory_section_text(getattr(self.config, "agent_memory_enabled", True))
        plan_block, planner_plan = self._structured_planning_block(
            goal, context, domain, history, mem, acting_role="SupportBot"
        )
        mem_state = mem.get_memory_state_for_planner()
        bonus = self._reflection_attempt_bonus(self.config, mem)

        def _support_once(hint: Optional[str] = None):
            return self.support_agent.generate(
                goal=goal,
                context=context,
                history=history,
                domain=domain,
                experience_data=experience_data,
                memory_section=mem_sec,
                structured_planning_block=plan_block or None,
                reflection_repair_hint=hint,
            )

        if getattr(self.config, "reflection_on_utterances_enabled", True):
            text, sb_meta = self.reflection_agent.run_reflected_generation(
                role="SupportBot",
                goal=goal,
                dialogue_history=history,
                memory_state=mem_state,
                domain=domain,
                generate_one=_support_once,
                reflection_attempt_bonus=bonus,
            )
        else:
            out = _support_once(None)
            text, sb_meta = out.text, dict(out.metadata)
        if planner_plan is not None:
            sb_meta["structured_planner"] = planner_plan
        self.support_agent.update_memory(mem, history[-2:] if len(history) >= 2 else history, goal=goal)
        return text, sb_meta

    def _check_goal_satisfied(self, goal: str, history: List[Dict[str, str]]) -> bool:
        """Backward-compatible wrapper for tests and callers."""
        return self.reflection_agent.goal_satisfied_llm(goal, history)

    def _check_completion_keywords(self, goal: str, history: List[Dict[str, str]]) -> bool:
        return self.reflection_agent.goal_satisfied_keywords(goal, history)

    def _compute_rl_lite_step(self, turns: List[Dict[str, Any]], goal: str) -> Dict[str, float]:
        """Lite scalar signal: goal overlap + pairwise turn coherence (not full RL training)."""
        conv = [t.get("text", "") for t in turns[-4:] if t.get("text")]
        if len(conv) < 2:
            return {"reward": 0.0, "goal_signal": 0.0, "coherence_signal": 0.0}
        goal_words = set(re.findall(r"[a-z0-9']+", (goal or "").lower()))
        joined = " ".join(conv).lower()
        doc_words = set(re.findall(r"[a-z0-9']+", joined))
        if not goal_words:
            g = 0.0
        else:
            g = len(goal_words & doc_words) / len(goal_words)
        coh_vals = []
        for i in range(1, len(conv)):
            coh_vals.append(calculate_similarity(conv[i - 1], conv[i]))
        c = sum(coh_vals) / len(coh_vals) if coh_vals else 0.0
        wg = float(getattr(self.config, "rl_goal_weight", 0.6))
        wc = float(getattr(self.config, "rl_coherence_weight", 0.4))
        r = wg * g + wc * c
        return {"reward": round(r, 4), "goal_signal": round(g, 4), "coherence_signal": round(c, 4)}

    def simulate_batch_dialogues(
        self,
        experience_data_list: List[Dict[str, Any]],
        experiment_run: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Simulate multiple dialogues in batch.
        
        Args:
            experience_data_list: List of experience data for each dialogue
            
        Returns:
            List of dialogue data
        """
        dialogues = []
        
        for i, experience_data in enumerate(experience_data_list):
            try:
                dialogue = self.simulate_dialogue(experience_data, experiment_run=experiment_run)
                dialogues.append(dialogue)
                
                # Add small delay between dialogues
                if i < len(experience_data_list) - 1:
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error simulating dialogue {i}: {e}")
                continue
        
        logger.info(f"Simulated {len(dialogues)} out of {len(experience_data_list)} dialogues")
        return dialogues
