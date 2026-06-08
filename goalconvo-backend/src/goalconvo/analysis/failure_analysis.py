"""
Automated failure analysis for goal-oriented dialogues.

Heuristic detectors (no LLM required): goal drift, hallucinated-looking entities,
repeated user questions, forgotten goal slots, incoherent transitions, early termination.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..domain_schemas import DOMAIN_SCHEMAS, DEFAULT_SCHEMA
from ..evaluation.goal_completion_eval import GoalCompletionEvaluator
from ..utils import calculate_similarity

logger = logging.getLogger(__name__)

# Canonical failure category keys
GOAL_DRIFT = "goal_drift"
HALLUCINATED_ENTITIES = "hallucinated_entities"
REPEATED_QUESTIONS = "repeated_questions"
FORGOTTEN_SLOTS = "forgotten_slots"
INCOHERENT_TRANSITIONS = "incoherent_transitions"
EARLY_TASK_TERMINATION = "early_task_termination"

CATEGORY_LABELS: Dict[str, str] = {
    GOAL_DRIFT: "Goal drift (dialogue moves away from stated goal)",
    HALLUCINATED_ENTITIES: "Hallucinated or ungrounded entities (proper names, codes)",
    REPEATED_QUESTIONS: "Repeated or near-duplicate user questions",
    FORGOTTEN_SLOTS: "Goal constraints / slots never addressed in dialogue",
    INCOHERENT_TRANSITIONS: "Incoherent adjacent turns (very low lexical overlap)",
    EARLY_TASK_TERMINATION: "User closes but task lacks completion evidence",
}

_STOP = frozenset(
    "a an the to of in on for and or is it at be as by not we you i are was were "
    "this that with from have has had can could would should need please hi hello".split()
)


def _meaningful_tokens(text: str) -> Set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", (text or "").lower()) if w not in _STOP and len(w) > 1}


def _concat_turns(turns: List[Dict[str, Any]], end: int) -> str:
    parts: List[str] = []
    for t in turns[: max(0, end)]:
        parts.append((t.get("text") or "").strip())
    return " ".join(parts).lower()


def _detect_goal_drift(goal: str, turns: List[Dict[str, Any]]) -> Tuple[bool, str, List[int]]:
    """Later dialogue retains much less goal vocabulary than the opening segment."""
    if len(turns) < 4:
        return False, "", []
    gw = _meaningful_tokens(goal)
    if len(gw) < 3:
        return False, "", []
    mid = max(2, len(turns) // 2)
    first = " ".join((t.get("text") or "").strip().lower() for t in turns[:mid])
    rest = " ".join((t.get("text") or "").strip().lower() for t in turns[mid:])
    fw = set(_meaningful_tokens(first))
    rw = set(_meaningful_tokens(rest))
    overlap_first = len(gw & fw) / max(len(gw), 1)
    overlap_rest = len(gw & rw) / max(len(gw), 1)
    if overlap_first < 0.12:
        return False, "", []
    if overlap_rest < 0.22 * overlap_first and overlap_rest < 0.35:
        return (
            True,
            f"goal_overlap_opening={overlap_first:.2f} goal_overlap_late={overlap_rest:.2f}",
            [mid],
        )
    return False, "", []


def _detect_hallucinated_entities(
    goal: str, turns: List[Dict[str, Any]]
) -> Tuple[bool, str, List[int]]:
    """Proper-noun-like tokens in SupportBot turns absent from prior context + goal."""
    flagged: List[int] = []
    details: List[str] = []
    goal_l = (goal or "").lower()
    for i, t in enumerate(turns):
        if t.get("role") != "SupportBot":
            continue
        text = t.get("text") or ""
        prior = (goal_l + " " + _concat_turns(turns, i)).lower()
        # Multi-word Title Case sequences (likely invented venue/person)
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
            ent = m.group(1)
            if len(ent) < 8:
                continue
            if ent.lower() not in prior:
                flagged.append(i)
                details.append(f"turn_{i}: '{ent}'")
        # Reference codes not echoed from user
        for m in re.finditer(r"\b([A-Z]{2,}-\d{2,}|[A-Z]{3,}\d{3,})\b", text):
            code = m.group(1)
            if code.lower() not in prior:
                flagged.append(i)
                details.append(f"turn_{i}: code {code}")
    if not flagged:
        return False, "", []
    return True, "; ".join(details[:4]), sorted(set(flagged))


def _detect_repeated_questions(turns: List[Dict[str, Any]]) -> Tuple[bool, str, List[int]]:
    user_texts: List[Tuple[int, str]] = []
    for i, t in enumerate(turns):
        if t.get("role") == "User":
            user_texts.append((i, (t.get("text") or "").strip()))
    for j in range(1, len(user_texts)):
        i2, t2 = user_texts[j]
        for k in range(j):
            i1, t1 = user_texts[k]
            if len(t1) < 12 or len(t2) < 12:
                continue
            if calculate_similarity(t1, t2) >= 0.72:
                return True, f"user_turns {i1} vs {i2} similarity≥0.72", [i1, i2]
    return False, "", []


def _slot_keywords_for_domain(domain: str) -> List[str]:
    schema = DOMAIN_SCHEMAS.get(domain.lower(), DEFAULT_SCHEMA)
    keys: List[str] = []
    for line in schema.get("slots", []) or []:
        m = re.match(r"^\s*([a-z_ /]+)", str(line).lower())
        if m:
            chunk = m.group(1)
            for part in re.split(r"[/,]", chunk):
                p = part.strip().split()
                if p:
                    keys.append(p[0].strip("()"))
    return list(dict.fromkeys(keys))[:12]


def _detect_forgotten_slots(
    goal: str, domain: str, turns: List[Dict[str, Any]]
) -> Tuple[bool, str, List[int]]:
    """Goal mentions a slot keyword (parking, vegan, airport, …) never appears in dialogue."""
    blob = " ".join((t.get("text") or "").lower() for t in turns)
    g = (goal or "").lower()
    missing: List[str] = []
    for kw in _slot_keywords_for_domain(domain):
        if len(kw) < 4:
            continue
        if kw in g and kw not in blob:
            missing.append(kw)
    constraints = turns[0].get("_constraints_ctx") if turns else None  # optional inject
    if isinstance(constraints, dict):
        for _k, v in constraints.items():
            vs = str(v).lower().strip()
            if len(vs) >= 4 and vs in g and vs not in blob:
                if vs not in missing:
                    missing.append(vs[:40])
    if not missing:
        return False, "", []
    return True, "missing: " + ", ".join(missing[:6]), []


def _detect_incoherent_transitions(turns: List[Dict[str, Any]]) -> Tuple[bool, str, List[int]]:
    bad: List[int] = []
    for i in range(1, len(turns)):
        a = (turns[i - 1].get("text") or "").strip()
        b = (turns[i].get("text") or "").strip()
        if len(a) < 14 or len(b) < 14:
            continue
        sim = calculate_similarity(a, b)
        if sim < 0.07:
            bad.append(i)
    if not bad:
        return False, "", []
    return True, f"low_overlap_at_indices={bad[:6]}", bad[:8]


def _detect_early_termination(
    goal: str, domain: str, turns: List[Dict[str, Any]], config: Optional[Any]
) -> Tuple[bool, str, List[int]]:
    """User thanks / closes without completion signals (heuristic + goal completion)."""
    if not turns:
        return False, "", []
    last_user_idx = None
    for i in range(len(turns) - 1, -1, -1):
        if turns[i].get("role") == "User":
            last_user_idx = i
            break
    if last_user_idx is None:
        return False, "", []
    last_u = (turns[last_user_idx].get("text") or "").lower()
    if not any(x in last_u for x in ("thank", "thanks", "perfect", "bye", "that's all", "all set")):
        return False, "", []
    dialogue = {"goal": goal, "domain": domain, "turns": turns}
    gc = GoalCompletionEvaluator(config) if config else GoalCompletionEvaluator()
    r = gc.evaluate(dialogue, use_llm_judge=False)
    if r.get("goal_completed"):
        return False, "", []
    return (
        True,
        "user_closure_without_task_completion",
        [last_user_idx],
    )


class FailureAnalyzer:
    """Per-dialogue failure detection."""

    def __init__(self, config: Optional[Any] = None):
        self.config = config

    def analyze(self, dialogue: Dict[str, Any]) -> Dict[str, Any]:
        goal = dialogue.get("goal") or ""
        domain = dialogue.get("domain") or "general"
        turns_in = dialogue.get("turns") or []
        # Strip system turns
        turns: List[Dict[str, Any]] = [t for t in turns_in if t.get("role") in ("User", "SupportBot")]
        for t in turns:
            t["_goal_ctx"] = goal
        if turns:
            turns[0]["_constraints_ctx"] = dialogue.get("constraints")

        checks = [
            (GOAL_DRIFT, lambda: _detect_goal_drift(goal, turns)),
            (HALLUCINATED_ENTITIES, lambda: _detect_hallucinated_entities(goal, turns)),
            (REPEATED_QUESTIONS, lambda: _detect_repeated_questions(turns)),
            (FORGOTTEN_SLOTS, lambda: _detect_forgotten_slots(goal, domain, turns)),
            (INCOHERENT_TRANSITIONS, lambda: _detect_incoherent_transitions(turns)),
            (EARLY_TASK_TERMINATION, lambda: _detect_early_termination(goal, domain, turns, self.config)),
        ]

        categories: Dict[str, Any] = {}
        triggered: List[str] = []
        for key, fn in checks:
            hit, detail, idxs = fn()
            categories[key] = {"detected": hit, "detail": detail, "turn_indices": idxs}
            if hit:
                triggered.append(key)

        for t in turns:
            t.pop("_goal_ctx", None)
            t.pop("_constraints_ctx", None)

        return {
            "dialogue_id": dialogue.get("dialogue_id", ""),
            "goal": goal,
            "domain": domain,
            "failure": len(triggered) > 0,
            "failure_categories": triggered,
            "categories": categories,
        }


class FailureCorpusAnalyzer:
    """Aggregate failures, percentages, and example snippets across many dialogues."""

    def __init__(self, config: Optional[Any] = None):
        self._analyzer = FailureAnalyzer(config)

    def analyze(self, dialogues: List[Dict[str, Any]], *, max_examples_per_category: int = 5) -> Dict[str, Any]:
        total = len(dialogues)
        cat_counts: Dict[str, int] = defaultdict(int)
        examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        failed_dialogues: List[Dict[str, Any]] = []
        failed_ids: List[str] = []

        for d in dialogues:
            r = self._analyzer.analyze(d)
            if r["failure"]:
                failed_dialogues.append(d)
                did = str(r.get("dialogue_id") or "")
                if did:
                    failed_ids.append(did)
            for cat in r["failure_categories"]:
                cat_counts[cat] += 1
                if len(examples[cat]) < max_examples_per_category:
                    snippet = ""
                    turns = d.get("turns") or []
                    if turns:
                        snippet = (turns[-1].get("text") or "")[:200]
                    examples[cat].append(
                        {
                            "dialogue_id": d.get("dialogue_id"),
                            "domain": d.get("domain"),
                            "detail": r["categories"][cat].get("detail", ""),
                            "snippet": snippet,
                        }
                    )

        failed_count = len(failed_dialogues)
        by_category: Dict[str, Any] = {}
        for cat, label in CATEGORY_LABELS.items():
            c = cat_counts.get(cat, 0)
            by_category[cat] = {
                "label": label,
                "count": c,
                "percentage_of_dialogues": (100.0 * c / total) if total else 0.0,
                "percentage_of_failed_dialogues": (100.0 * c / failed_count) if failed_count else 0.0,
                "examples": examples.get(cat, []),
            }

        return {
            "total_dialogues": total,
            "failed_dialogues": failed_count,
            "failure_rate": (failed_count / total) if total else 0.0,
            "by_category": by_category,
            "failed_dialogue_ids": failed_ids,
        }


def export_failure_bundle(
    output_dir: str | Path,
    dialogues: List[Dict[str, Any]],
    aggregate: Dict[str, Any],
    *,
    per_dialogue_results: Optional[List[Dict[str, Any]]] = None,
    analyzer: Optional[FailureAnalyzer] = None,
) -> Dict[str, str]:
    """
    Write failed dialogues, JSON report, CSV summary, and Markdown report under ``output_dir``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    failed_dir = out / "failed_dialogues"
    failed_dir.mkdir(exist_ok=True)

    fa = analyzer or FailureAnalyzer()
    per = per_dialogue_results
    if per is None:
        per = [fa.analyze(d) for d in dialogues]

    for d, pr in zip(dialogues, per):
        if not pr.get("failure"):
            continue
        did = str(d.get("dialogue_id") or pr.get("dialogue_id") or "unknown")
        safe = re.sub(r"[^\w\-]+", "_", did)[:120]
        with (failed_dir / f"{safe}.json").open("w", encoding="utf-8") as fp:
            json.dump({"dialogue": d, "failure_analysis": pr}, fp, indent=2, ensure_ascii=False, default=str)

    report = {
        "aggregate": aggregate,
        "per_dialogue": per,
        "category_labels": CATEGORY_LABELS,
    }
    json_path = out / "failure_report.json"
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False, default=str)

    csv_path = out / "failure_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(
            [
                "category",
                "label",
                "count",
                "pct_of_all_dialogues",
                "pct_of_failed_dialogues",
            ]
        )
        for cat, block in aggregate.get("by_category", {}).items():
            w.writerow(
                [
                    cat,
                    block.get("label", ""),
                    block.get("count", 0),
                    f"{block.get('percentage_of_dialogues', 0):.2f}",
                    f"{block.get('percentage_of_failed_dialogues', 0):.2f}",
                ]
            )

    md_path = out / "failure_report.md"
    with md_path.open("w", encoding="utf-8") as fp:
        fp.write("# Failure analysis report\n\n")
        fp.write(f"- Total dialogues: **{aggregate.get('total_dialogues', 0)}**\n")
        fp.write(f"- Failed (≥1 category): **{aggregate.get('failed_dialogues', 0)}** ")
        fp.write(f"({100 * aggregate.get('failure_rate', 0):.1f}%)\n\n")
        fp.write("## By category\n\n")
        fp.write("| Category | Count | % of all | % of failed |\n")
        fp.write("|----------|------:|---------:|------------:|\n")
        for cat, block in aggregate.get("by_category", {}).items():
            fp.write(
                f"| `{cat}` | {block.get('count', 0)} | "
                f"{block.get('percentage_of_dialogues', 0):.1f}% | "
                f"{block.get('percentage_of_failed_dialogues', 0):.1f}% |\n"
            )
        fp.write("\n## Examples\n\n")
        for cat, block in aggregate.get("by_category", {}).items():
            if not block.get("examples"):
                continue
            fp.write(f"### {cat}\n\n")
            for ex in block["examples"]:
                fp.write(f"- **{ex.get('dialogue_id')}** ({ex.get('domain')}): {ex.get('detail')}\n")
                fp.write(f"  - _Last turn snippet_: {ex.get('snippet', '')[:180]!r}\n")
            fp.write("\n")
        fp.write("\n## Failed dialogue files\n\n")
        fp.write(f"Individual JSON files under `{failed_dir.name}/`.\n")

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "failed_dialogues_dir": str(failed_dir),
    }
