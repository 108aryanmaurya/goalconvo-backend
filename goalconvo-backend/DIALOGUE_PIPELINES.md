## GoalConvo Dialogue Generation and Evaluation Pipelines

This document summarizes how GoalConvo generates synthetic task‑oriented dialogues and how it evaluates them. It is written to be suitable for explaining the system to a supervisor or professor.

---

## 1. Dialogue Generation Pipeline (excluding evaluation)

The generation pipeline creates synthetic goal‑oriented dialogues in several stages. The main orchestrator is `GoalConvoGenerator` (in `scripts/generate_dialogues.py`), which is also invoked by the backend endpoint `POST /api/run-pipeline`.

### 1.1 Inputs and Orchestration

- **Entry points**:
  - CLI: `python scripts/generate_dialogues.py --num-dialogues N --domains ...`
  - API: `POST /api/run-pipeline` (used by the frontend dashboard)
- **Key parameters**:
  - `num_dialogues`: total number of dialogues to generate.
  - `domains`: list of domains (e.g. `hotel`, `restaurant`, `taxi`, `train`, `attraction`); if omitted, `Config.domains` is used.
  - Optional overrides: `quality_judge` (on/off), `few_shot_examples` (int), `temperature`, etc.
- `GoalConvoGenerator.generate_dialogues()`:
  - Instantiates `LLMClient`, `DatasetStore`, `ExperienceGenerator`, `DialogueSimulator`, and `QualityJudge`.
  - Splits `num_dialogues` across the target domains (approximately evenly).
  - For each domain, runs the **experience → simulation → quality → saving** process, then updates the few‑shot hub and saves generation progress.

The backend API (`backend_server.py`) wraps this, runs it in a background task, and streams progress and final stats to the frontend via WebSocket events.

### 1.2 Step 1 – Experience Generation

**Component**: `ExperienceGenerator` (`src/goalconvo/experience_generator.py`)

**Objective**: Turn a simple goal into a rich, evaluable scenario:

- A normalized natural‑language goal.
- Domain (e.g. hotel, restaurant).
- Detailed context (background story).
- First user utterance.
- User persona and optional structured fields (subgoals, constraints, requestables).

**Sub‑steps**:

1. **Goal selection**
   - Randomly sample a goal for the domain from `seed_goals.json`.
   - These seed goals are derived from MultiWOZ (via `download_multiwoz.py`) or default templates.

2. **Goal normalization**
   - Convert MultiWOZ‑style goals (e.g. `hotel-name: Alpha-Milton guest house`) into natural language (e.g. “Book a room at Alpha‑Milton guest house”).
   - Infer domain from the normalized goal if domain is not explicitly provided.

3. **Few‑shot examples**
   - Call `DatasetStore.load_few_shot_examples(domain, num_examples)`:
     - Ensures there are at least a minimum number of high‑quality seed dialogues in the few‑shot hub (`data/few_shot_hub/{domain}/`).
     - Returns a small set of example dialogues to serve as in‑context examples for the LLM.
   - Number of examples is controlled by `Config.few_shot_examples` (or an override).

4. **Prompt construction and LLM call**
   - System prompt: describes how to create **evaluable** scenarios (clear constraints, requestables, realistic context, varied first utterances).
   - User prompt: provides the normalized goal and the retrieved few‑shot examples.
   - `LLMClient.generate_completion()` is called with this prompt to produce JSON.
   - The JSON is parsed into:
     - `goal`, `context`, `first_utterance`, `user_persona`.
     - Optional `subgoals`, `constraints`, `requestables`, etc.

The result is an `experience_data` object that provides everything needed to start a realistic dialogue simulation.

### 1.3 Step 2 – Multi‑Agent Dialogue Simulation

**Component**: `DialogueSimulator` (`src/goalconvo/multi_agent_simulator.py`)

**Objective**: Use a user agent and a support agent (both LLM‑driven) to simulate a complete conversation that tries to achieve the goal.

**Setup**:

- Input: a single `experience_data` instance.
- Initialize a new `dialogue_id`.
- Create an initial **System** message describing the domain and goal (e.g. “Domain: hotel\nUser Goal: Book a budget hotel tonight in the city centre…”).
- First user utterance:
  - If `first_utterance` is provided by the experience generator, use it.
  - Otherwise, generate it via `_generate_user_turn()`.

**Turn loop**:

- Alternating turns:
  - `_generate_supportbot_turn()` creates a `SupportBot` utterance based on the goal, context, and conversation history.
  - `_generate_user_turn()` creates the next `User` utterance given the persona and history.
- Each turn is appended to `turns` and `conversation_history`.
- A `progress_callback` can be invoked after each turn to stream an incremental view to the frontend (live dialogue preview).

**Stopping criteria**:

- Respect `Config.min_turns` and `Config.max_turns`.
- Only consider stopping after the minimum number of turns is reached.
- Two main stopping mechanisms:
  - **Goal satisfaction**:
    - Quick keyword‑based check for satisfaction and confirmation (e.g. “booked”, “reference number”, “that’s all, thanks”).
    - Optional LLM‑based goal check `_check_goal_satisfied()` for ambiguous cases.
  - **Repetition loop detection**:
    - Detect when the last N turns are effectively repeating an earlier pattern.
    - If so, force a grounded closing exchange (e.g. assistant provides a concrete confirmation with reference number) and end with a user “thanks” turn.

**Output**:

- A `dialogue` object with:
  - `dialogue_id`, `domain`, `goal`, `context`, `user_persona`.
  - `turns`: list of `{ role: "User" | "SupportBot", text, timestamp }`.
  - Optional `metadata` (e.g. generation time).

### 1.4 Step 3 – Quality Filtering

**Component**: `QualityJudge` (`src/goalconvo/quality_judge.py`)

**Objective**: Filter out low‑quality dialogues and keep only those suitable for downstream use (and for seeding the few‑shot hub).

**Heuristic filters** (cheap, rule‑based):

- **Length check**: `min_turns ≤ num_turns ≤ max_turns`.
- **Repetition check**: detect repeated utterances above a similarity threshold.
- **Profanity check**: scan turns against a profanity lexicon.
- **Coherence check**: basic consistency heuristics (e.g. no empty or trivial turns).
- **Goal mention check**: goal terms appear in the dialogue.
- **Empty response check**: no empty or placeholder turns.

Each check returns a boolean and a message; a combined heuristic score is computed from the fraction of passed checks.

**LLM‑based evaluation**:

- The full dialogue (and goal) is passed to the LLM via carefully designed prompts to rate:
  - **Coherence** (1–5).
  - **Goal relevance / completion** (YES/NO).
  - **Overall quality** (1–5), considering task success, coherence, diversity, fluency, groundedness, and appropriate length.
- Optionally, if `improve_on_fail` is enabled, the LLM is also prompted to propose a corrected version of failed dialogues, but generally only the original high‑quality ones are kept.

**Decision**:

- Combine heuristic and LLM results to assign:
  - `quality_assessment.overall_score` (0–1 or 1–5 scaled).
  - `passed_filters` (boolean).
- `QualityJudge.filter_dialogues()` returns two lists:
  - `accepted`: dialogues that pass the combined threshold.
  - `rejected`: dialogues that do not.

For each accepted dialogue, a `metadata.quality_score` is stored so later steps can pick the best ones.

### 1.5 Step 4 – Saving Dialogues and Updating the Few‑Shot Hub

**Component**: `DatasetStore` (`src/goalconvo/dataset_store.py`)

**Saving accepted dialogues**:

- `DatasetStore.save_dialogue(dialogue)`:
  - Validates the dialogue structure.
  - Ensures a `dialogue_id` exists.
  - Updates metadata with turn counts.
  - Writes JSON files to `data/synthetic/{domain}/{dialogue_id}.json`.

**Tracking statistics**:

- `GoalConvoGenerator` keeps `self.stats`:
  - Total generated, accepted, rejected.
  - Per‑domain counts.
  - Start/end times.
  - For the current run, it also stores `accepted_dialogues` in memory for downstream use.

**Few‑shot hub update**:

- `DatasetStore.update_few_shot_hub()`:
  - Loads dialogues with non‑zero `quality_score`.
  - Sorts them by quality score.
  - Selects the top percentage (e.g. top 10%).
  - Writes these as few‑shot examples to `data/few_shot_hub/{domain}/{dialogue_id}.json`, adding `hub_metadata`.
  - This gradually improves the quality of examples used by the experience generator and simulator.

**Versioning**:

- `DatasetVersionManager.create_version()`:
  - Builds a dataset snapshot containing the dialogues generated/accepted in a run.
  - Stores metadata (timestamp, description, generation configuration, domain distribution, tags).
  - Saves to `data/versions/{version_id}/dialogues.json`.

**End of pipeline**:

- Generation progress is written to `data/generation_progress.json`.
- The backend attaches `accepted_dialogues` to the stats so later evaluation runs can focus on this run, though full evaluation is handled by a separate pipeline.

---

## 2. Dialogue Evaluation Pipeline

The evaluation pipeline *does not generate dialogues*. It analyzes existing synthetic dialogues (optionally comparing them to human MultiWOZ dialogues) and computes a rich set of metrics. It is implemented primarily in `scripts/comprehensive_dialogue_evaluation.py` and wired into the backend via `/api/run-evaluation`.

### 2.1 Entry Point and Data Selection

**API endpoint**: `POST /api/run-evaluation`

**Request body**:

- `session_id` (optional): frontend WebSocket room identifier for streaming logs and results.
- `limit`: number of dialogues to evaluate (e.g. last 10 or last 20).
- `domains` (optional): list of domain names (`["hotel", "restaurant", ...]`); if omitted, all synthetic domain folders are considered.

**Dialogue loading logic**:

1. Determine the set of domains to load from:
   - If `domains` is provided and non‑empty, use those.
   - Otherwise, inspect `config.synthetic_dir` (e.g. `data/synthetic`) and collect all directory names present there; if nothing is found, fall back to `Config.domains`.
2. Compute `pool_size = max(limit * 20, 500)` to ensure a large pool from which to select the latest N.
3. Use `DatasetStore.load_dialogues(limit=pool_size, domains_override=all_domain_dirs)` to load dialogues across the chosen domains.
4. Sort all loaded dialogues by generation timestamp (either `metadata.generated_at` or `provenance.timestamp`).
5. Take the **latest N**:
   - `generated_dialogues = all_dialogues[-limit:]`
6. If no dialogues are found, emit an `evaluation_error` and stop.

In other words, the evaluation runs on the most recent N synthetic dialogues from the selected domain(s), not on arbitrary older ones.

### 2.2 Reference Dialogues from MultiWOZ (Optional)

**Reference file**: `processed_dialogues.json` under `config.multiwoz_dir` (e.g. `data/multiwoz/processed_dialogues.json`).

**Usage**:

- If this file exists:
  - Load it as `reference_dialogues`.
  - Trim to at most 100 dialogues for performance.
  - These are used for **BERTScore** and **BLEU** and as an optional comparison for lexical diversity.
- If it does not exist:
  - BERTScore and BLEU are simply skipped (with a log message).
  - All other metrics still run.

Reference dialogues use the same structure as synthetic ones: they have `domain` and `turns` with `text` fields.

### 2.3 Comprehensive Evaluation: Metric Suite

The central method is `ComprehensiveDialogueEvaluator.evaluate_dialogues(dialogues, reference_dialogues, use_llm_judge, emit_callback, yield_callback)`. It computes metrics in a fixed order, logging progress between steps.

#### 2.3.1 Goal Completion Rate (GCR)

**What it measures**: percentage of dialogues in which all **constraints** and **requestables** implied by the goal are fulfilled in the conversation.

**How**:

- For each dialogue:
  - Extract constraints (e.g. area, price range, type) from structured `goal_data` or parse the goal text with regexes.
  - Extract requestables (e.g. phone number, address, reference number).
  - Check the dialogue text to see whether all constraints and requestables are satisfied (using simple keyword and synonym matching).
- Aggregate:
  - Compute overall GCR (`completed_count / total_count`).
  - Compute per‑domain completion statistics (`domain_gcr`).

**Output fields**: `overall_gcr`, `completed_count`, `total_count`, `domain_gcr`.

#### 2.3.2 Task Success Rate (TSR)

**What it measures**: percentage of dialogues where the user’s **intent** was successfully fulfilled and the conversation ended satisfactorily.

**How**:

- For each dialogue:
  - Extract goal and turns, build a combined dialogue text.
  - Use heuristic signals such as:
    - Intent keywords from the goal (e.g. “book”, “reserve”, “find”).
    - Markers of satisfaction and completion in the dialogue (e.g. “thanks, that helps a lot”, “perfect, that’s all”).
  - Optionally, use an LLM‑based assessment to refine success vs. failure.
- Aggregate:
  - Overall TSR (`successful_count / total_count`).
  - Per‑domain TSR (`domain_tsr`).

**Output fields**: `overall_tsr`, `successful_count`, `total_count`, `domain_tsr`.

#### 2.3.3 Lexical Diversity

**What it measures**: how varied the language is across dialogues, using Distinct‑1 and Distinct‑2 metrics.

**How**:

- For each dialogue:
  - Concatenate all turn texts into a single string.
  - Tokenize using a simple regex (word tokens).
  - Compute:
    - Distinct‑1: unique unigrams / total unigrams.
    - Distinct‑2: unique bigrams / total bigrams.
- Average Distinct‑1 and Distinct‑2 across dialogues.
- Compute a **combined** score `(distinct_1 + distinct_2) / 2`.
- Optionally, compute the same metrics for reference (MultiWOZ) dialogues and a **diversity ratio**.
- Compute per‑domain diversity (same metrics, restricted to dialogues of that domain).

**Target**: Combined diversity ≈ 0.46.

**Output fields**: `distinct_1`, `distinct_2`, `combined`, `target_diversity`, `real_diversity`, `diversity_ratio`, `domain_diversity`.

#### 2.3.4 BERTScore (Semantic Similarity)

**What it measures**: how semantically similar synthetic dialogues are to real MultiWOZ dialogues in the same domain.

**How** (only if `reference_dialogues` is provided):

- Group reference dialogues by domain.
- For each synthetic dialogue:
  - Take its domain.
  - Collect up to 10 reference dialogues from the same domain.
  - Extract texts from both (concatenated turn texts).
  - Compute BERTScore F1 using a large transformer (e.g. DeBERTa), possibly truncating long texts and using fallbacks if needed.
  - For that synthetic dialogue, keep the **max** F1 across its reference pairs.
- Average the per‑dialogue best scores to get the overall BERTScore; collect per‑domain averages.

**Target**: F1 ≈ 0.71.

**Output fields**: `overall_bertscore`, `std_bertscore`, `individual_scores`, `domain_bertscores`, `target_score`.

#### 2.3.5 BLEU

**What it measures**: n‑gram overlap between synthetic and reference dialogues.

**How** (only if `reference_dialogues` is provided):

- Group references by domain.
- For each synthetic dialogue:
  - Tokenize its text (prefer NLTK `word_tokenize`, fallback to simple split).
  - For up to 10 references from the same domain:
    - Tokenize reference text.
    - Compute sentence‑level BLEU with smoothing (or a simple fallback if NLTK is unavailable).
  - Keep the **best** BLEU score across references.
- Aggregate:
  - Overall average BLEU and standard deviation.
  - Per‑domain BLEU stats.

**Output fields**: `average_bleu`, `std_bleu`, `individual_scores`, `domain_bleu`.

#### 2.3.6 Dialogue Length and Turn Statistics

**What it measures**: basic structural properties of the dialogues (length in turns, words, characters).

**How**:

- For each dialogue:
  - Count number of turns.
  - Concatenate text and count words and characters.
- Aggregate:
  - Mean and sample standard deviation of turns, words, characters.
  - Min and max turns.
  - Per‑domain averages.

**Output fields**: `avg_turns`, `std_turns`, `avg_words`, `std_words`, `min_turns`, `max_turns`, `num_dialogues`, `domain_metrics`.

#### 2.3.7 Repetition Rate

**What it measures**: how often dialogues reuse exactly the same turns (a sign of low diversity or loops).

**How**:

- For each dialogue:
  - Extract non‑empty turn texts.
  - Compute: `repetition_rate = 1 − (number_of_unique_turn_texts / total_turn_texts)`.
- Aggregate:
  - Mean and std of repetition rate.
  - Per‑domain repetition statistics.

**Output fields**: `overall_repetition_rate`, `std_repetition_rate`, `domain_repetition`.

#### 2.3.8 Response Time Metrics

**What it measures**: inter‑turn time gaps based on timestamps added during simulation (useful as a proxy for pacing, not for wall‑clock performance).

**How**:

- For each dialogue:
  - Parse `timestamp` fields on turns (ISO8601).
  - Compute time differences between consecutive timestamps.
  - Ignore negative or extremely large gaps; floor very small gaps at 0.1s to avoid artifacts.
- Aggregate:
  - Overall mean and std gap.
  - Min and max gap.
  - Per‑domain response time stats.

**Output fields**: `overall_avg_seconds`, `overall_std_seconds`, `min_seconds`, `max_seconds`, `num_gaps`, `domain_metrics`, `note`.

#### 2.3.9 LLM‑as‑a‑Judge

**What it measures**: human‑style quality judgements using an LLM, across five dimensions:

- Task Success
- Coherence
- Diversity (stylistic)
- Fluency
- Groundedness

**How**:

- For each dialogue:
  - Format as a readable transcript with roles (`User: ...`, `SupportBot: ...`).
  - Provide the goal and dialogue to the LLM in a single prompt, asking it to return JSON with five integer scores from 0–100.
- Validate and parse the JSON.
- Aggregate:
  - Overall mean and std for each dimension.
  - Per‑domain mean scores.

**Output fields**: `overall_scores` (per metric), `domain_scores`.

This step can be disabled via the `EVAL_SKIP_LLM_JUDGE` environment variable to save API cost.

#### 2.3.10 Advanced Heuristic Metrics

**What they approximate**:

- Intent consistency (does the dialogue behavior align with the goal’s inferred intent?).
- Slot coverage (do key constraint values like times, dates, numeric tokens appear in the dialogue?).
- Simple state consistency (absence of obvious contradictions like “earlier you said…”).

**Output fields**: `intent_consistency`, `slot_coverage`, `state_tracking`.

### 2.4 Summary Table and Raw Results

After computing all metrics, the evaluator:

- Builds a **summary table** mapping metric names to formatted strings (used in CLI logs).
- Packages all metrics in a dictionary: `{ "evaluation_timestamp", "total_dialogues", "metrics": { ... }, "summary_table": { ... } }`.

This full structure is passed to `convert_evaluation_to_frontend_format()`.

### 2.5 Conversion to Frontend Format and Delivery

**Conversion**:

- `convert_evaluation_to_frontend_format(eval_results)` turns the comprehensive metrics into a single object used by the dashboard:
  - `overall_score` in [0, 1], calibrated so typical good runs show ≈ 0.85–0.95.
  - Per‑dimension scores (task success, coherence, diversity, fluency, groundedness).
  - `total_dialogues_evaluated`.
  - `categories`:
    - `lexical_diversity` (0–100 scale).
    - `conversation_length` (avg turns and std dev).
    - `domain_distribution` (dialogue counts per domain).
    - `task_success_by_domain`.
    - `goal_completion_by_domain`.
  - `comprehensive_metrics`: raw metric dictionaries so the UI can show detailed panels.

**Delivery**:

- The backend emits:
  - `log` events describing progress (e.g. “Computing Task Success Rate…”).
  - `evaluation_complete` with `{ evaluation: frontend_metrics }`.
- The frontend’s `Evaluator` component receives this and renders:
  - Overall score gauge.
  - BERTScore, Lexical Diversity, and other metric cards.
  - Domain breakdowns (distribution, success rates, goal completion).

This completes a single evaluation run over the chosen set of synthetic dialogues.

