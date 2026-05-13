from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

EXPERIENCE_SYSTEM = """Expand the goal into one evaluable scenario. Output JSON only.
Required keys: goal, context, first_utterance, user_persona.
Optional: subgoals (array), constraints (object), domain.
Rules: specific goal with constraints; realistic context; natural varied first line."""

# -------------------------
# OPENROUTER — experience JSON only (dialogue stays local)
# -------------------------


def openrouter_chat(
    api_key: str,
    model: str,
    user_prompt: str,
    *,
    max_tokens: int = 512,
    temperature: float = 0.35,
    top_p: float = 0.9,
    timeout: int = 120,
) -> str:
    """Single user message → assistant text. Returns empty string on failure."""
    key = (api_key or "").strip()
    if not key:
        return ""
    payload = json.dumps(
        {
            "model": model.strip(),
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://github.com/goalconvo",
            "X-Title": "goalconvo-experience",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        logger.warning("OpenRouter request failed: %s", e)
        return ""


def _parse_experience_json(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {}
    t = text.strip()
    if "```" in t:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, re.I)
        if m:
            t = m.group(1).strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(t[start : end + 1])
    except json.JSONDecodeError:
        return {}


def generate_experience(
    domain: str,
    task: str,
    *,
    openrouter_api_key: Optional[str] = None,
    openrouter_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build scenario JSON via OpenRouter. If no key or call fails, returns a minimal local fallback.
    """
    dom = (domain or "general").strip() or "general"
    seed = (task or "").strip() or "Complete the task"
    key = (openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "") or "").strip()
    model = (openrouter_model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini") or "").strip()

    def _fallback() -> Dict[str, Any]:
        return {
            "goal": seed,
            "context": "",
            "first_utterance": f"I need help with {seed}",
            "user_persona": "General user",
            "domain": dom,
        }

    if not key:
        return _fallback()

    user_part = f"Domain: {dom}\nSeed task: {seed}\n\nReturn one JSON object only."
    prompt = f"{EXPERIENCE_SYSTEM}\n\n{user_part}"
    raw = openrouter_chat(
        key,
        model,
        prompt,
        max_tokens=512,
        temperature=0.35,
        top_p=0.9,
        timeout=int(os.getenv("OPENROUTER_TIMEOUT", "120")),
    )
    parsed = _parse_experience_json(raw)
    if not isinstance(parsed, dict) or not parsed.get("goal"):
        return _fallback()

    out = {
        "goal": str(parsed.get("goal", seed)).strip() or seed,
        "context": str(parsed.get("context", "") or "").strip(),
        "first_utterance": str(parsed.get("first_utterance", "") or "").strip()
        or f"I need help with {seed}",
        "user_persona": str(parsed.get("user_persona", "General user") or "General user").strip(),
        "domain": str(parsed.get("domain", dom) or dom).strip(),
    }
    if parsed.get("subgoals") is not None:
        out["subgoals"] = parsed["subgoals"]
    if parsed.get("constraints") is not None:
        out["constraints"] = parsed["constraints"]
    return out


# -------------------------
# GENERATE FUNCTION
# -------------------------

def generate(model, tokenizer, prompt, max_tokens=80):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 🔥 IMPORTANT: only take new generated part
    text = text[len(prompt):]

    return text.strip()


# -------------------------
# CLEAN RESPONSE
# -------------------------

def clean(text):
    text = text.strip()

    # remove prompt leakage
    bad_phrases = [
        "You are a helpful assistant",
        "You are a user",
        "Goal:",
        "Conversation:"
    ]

    for p in bad_phrases:
        if p in text:
            text = text.split(p)[0]

    # 🔥 CRITICAL FIX
    text = extract_response(text)

    return text.strip()


# -------------------------
# HISTORY LIMIT
# -------------------------

def truncate_history(history, max_chars=600):
    return history[-max_chars:]


# -------------------------
# USER TURN
# -------------------------

def user_turn(model, tokenizer, goal, history):
    prompt = f"""User is trying to complete a task.

Goal: {goal}

Conversation:
{history}

User:"""  # 🔥 IMPORTANT END

    response = generate(model, tokenizer, prompt, 50)
    return clean(response)


# -------------------------
# ASSISTANT TURN
# -------------------------

def bot_turn(model, tokenizer, goal, history):
    prompt = f"""You are a helpful assistant.

Goal: {goal}

Conversation:
{history}

Rules:
- Help complete the task
- Ask for missing details
- Confirm with user before taking any step
- Give specific details (time/place/reference)

Assistant:"""  # 🔥 CRITICAL FIX

    response = generate(model, tokenizer, prompt, 80)
    return clean(response)


# -------------------------
# STOP CONDITION
# -------------------------

def is_done(text):
    text = text.lower()
    return any(x in text for x in [
        "thank you", "thanks", "perfect", "that's all", "all set"
    ])


# -------------------------
# MAIN PIPELINE
# -------------------------

def run_dialogue_pipeline(
    model,
    tokenizer,
    domain,
    task,
    num_dialogues=5,
    openrouter_api_key=None,
    openrouter_model=None,
):
    results = []

    for _ in range(num_dialogues):

        experience = generate_experience(
            domain,
            task,
            openrouter_api_key=openrouter_api_key,
            openrouter_model=openrouter_model,
        )
        goal = (experience.get("goal") or task or "").strip() or task
        ctx = (experience.get("context") or "").strip()
        dialogue_goal = goal if not ctx else f"{goal}. {ctx}"
        first_line = (experience.get("first_utterance") or "").strip() or f"I need help with {task}"

        history = f"User: {first_line}"
        turns = [{"role": "User", "text": first_line}]

        for _ in range(8):

            # ---- Assistant ----
            bot = bot_turn(model, tokenizer, dialogue_goal, history)

            if not bot:
                break

            turns.append({"role": "Assistant", "text": bot})
            history += f"\nAssistant: {bot}"
            history = truncate_history(history)

            # ---- User ----
            user = user_turn(model, tokenizer, dialogue_goal, history)

            if not user:
                break

            turns.append({"role": "User", "text": user})
            history += f"\nUser: {user}"
            history = truncate_history(history)

            # ---- Stop ----
            if is_done(user):
                break

        results.append(turns)

    return results



def extract_response(text):
    # stop when next role starts
    stop_tokens = ["User:", "Assistant:"]

    for token in stop_tokens:
        if token in text:
            text = text.split(token)[0]

    return text.strip()