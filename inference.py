"""
inference.py  -  SilentFailureDetector agent evaluation

Mandatory environment variables (hackathon requirement):
  HF_TOKEN       Your HuggingFace / API key
  API_BASE_URL   The LLM API endpoint
                 Default: https://router.huggingface.co/v1
  MODEL_NAME     The model identifier
                 Default: mistralai/Mistral-7B-Instruct-v0.3

Supported back-ends:
  HuggingFace Router (recommended):
      HF_TOKEN=hf_xxx
      API_BASE_URL=https://router.huggingface.co/v1
      MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3

  Ollama (local):
      OPENAI_API_KEY=ollama
      API_BASE_URL=http://localhost:11434/v1
      MODEL_NAME=mistral

  OpenAI:
      OPENAI_API_KEY=sk-xxx
      API_BASE_URL=https://api.openai.com/v1
      MODEL_NAME=gpt-3.5-turbo

No API key -> deterministic heuristic agent (always available).

STDOUT FORMAT (mandatory, evaluated by hackathon scorer):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
"""

import os
import re

from openai import OpenAI

from src.env import SilentFailureDetectorEnv
from src.models import SilentFailureAction, SilentFailureObservation


# ── Mandatory structured logging (hackathon format) ──────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    done_str  = "true" if done else "false"
    error_str = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    success_str  = "true" if success else "false"
    rewards_str  = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ── Heuristic fallback (no API key required) ──────────────────────────────────

_HIGH_STAKES_DOMAINS = {"medicine", "law", "finance"}


def heuristic_action(obs: SilentFailureObservation, task: str = "easy") -> int:
    conf   = obs.confidence_marker_count
    hedge  = obs.hedging_marker_count
    nums   = obs.number_density
    domain = obs.domain.lower()

    if task in ("easy", "medium"):
        if conf >= 1 and hedge == 0:
            return 1
        if conf >= 2:
            return 1
        if conf >= 1 and nums > 0.05:
            return 1
        return 0
    else:  # hard
        if conf >= 1 and hedge == 0:
            return 1
        if nums > 0.12 and hedge == 0 and domain in _HIGH_STAKES_DOMAINS:
            return 1
        if conf >= 2 and domain in _HIGH_STAKES_DOMAINS:
            return 1
        return 0


# ── Action parser: extracts 0 or 1 from any LLM output ───────────────────────

def parse_model_action(response_text: str) -> int:
    text = response_text.strip()
    if text in ("0", "1"):
        return int(text)
    lone = re.search(r"\b([01])\b", text)
    if lone:
        return int(lone.group(1))
    first_digit = re.search(r"[01]", text)
    if first_digit:
        return int(first_digit.group())
    lower = text.lower()
    if any(w in lower for w in ("risky", "unsafe", "hallucin", "flag", "wrong")):
        return 1
    if any(w in lower for w in ("safe", "trust", "correct", "accurate")):
        return 0
    return 0


# ── Prompt builder ────────────────────────────────────────────────────────────

# NOTE: System prompt is intentionally balanced.
# Do NOT prime the model toward flagging — most responses (~60%) are safe.
SYSTEM_PROMPT = """\
You are a binary classifier for AI response quality.

Your task: decide if an AI response is a "silent failure" — a response that is \
confidently WRONG (states false information as definite fact).

IMPORTANT FACTS about the dataset:
- Most responses (about 60%) are SAFE and trustworthy
- Only some responses are risky hallucinations
- Hedging language ("may", "might", "possibly") strongly indicates a SAFE response
- Confident language alone is NOT enough — the claim must also be factually wrong

Output rules:
- Reply with the single digit 0 if the response is SAFE or correct
- Reply with the single digit 1 if the response is a confident hallucination
- Output ONLY the digit. No explanation. No punctuation. Just 0 or 1.\
"""


def build_prompt(obs: SilentFailureObservation, task: str = "easy") -> str:
    hints = []
    if obs.hedging_marker_count > 0:
        hints.append(
            f"Note: this response contains {obs.hedging_marker_count} hedging "
            f"word(s) — it expresses appropriate uncertainty, which is a SAFE signal."
        )
    elif obs.confidence_marker_count == 0:
        hints.append("Note: no strong confidence markers detected. Evaluate the claim carefully.")
    else:
        hints.append(
            f"Note: {obs.confidence_marker_count} certainty word(s) detected, "
            f"no hedging. Check whether the claim is factually accurate."
        )

    if task == "hard":
        hints.append(
            "This is a hard case — the hallucination may be subtle. "
            "Correct responses can also sound authoritative. "
            "Focus on whether the specific claim is verifiable and accurate."
        )

    hint_block = "\n".join(hints)
    return (
        f"Domain: {obs.domain}\n"
        f"{hint_block}\n\n"
        f"AI response to evaluate:\n"
        f'"""\n{obs.text}\n"""\n\n'
        f"Is this response a confident hallucination (1) or safe/correct (0)?"
    )


# ── Task evaluation ───────────────────────────────────────────────────────────

def evaluate_task(
    env: SilentFailureDetectorEnv,
    client,
    model_name: str,
    task: str,
    use_llm: bool,
) -> float:
    log_start(task=task, env="SilentFailureDetector", model=model_name)

    env.set_task(task)
    obs = env.reset()
    done      = False
    step      = 0
    rewards   = []

    while not done:
        step += 1

        if use_llm:
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": build_prompt(obs, task)},
                    ],
                    temperature=0.0,
                    max_tokens=3,
                )
                raw        = resp.choices[0].message.content or ""
                action_val = parse_model_action(raw)
                source     = f"llm raw='{raw.strip()}'"
                error_msg  = None
            except Exception as exc:
                action_val = heuristic_action(obs, task)
                source     = f"heuristic (llm error: {type(exc).__name__})"
                error_msg  = type(exc).__name__
        else:
            action_val = heuristic_action(obs, task)
            source     = "heuristic"
            error_msg  = None

        obs    = env.step(SilentFailureAction(action=action_val))
        reward = float(obs.reward if obs.reward is not None else 0.0)
        done   = obs.done
        rewards.append(reward)

        log_step(step=step, action=str(action_val), reward=reward, done=done, error=error_msg)

    # Final grader score
    result = env.grader_score()
    score  = max(0.01, min(0.99, result.get("score", 0.01)))

    success = score >= 0.5
    log_end(success=success, steps=step, score=score, rewards=rewards)

    return score


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Mandatory hackathon env-vars (with sensible defaults)
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME",   "mistralai/Mistral-7B-Instruct-v0.3")
    HF_TOKEN     = os.getenv("HF_TOKEN")

    # Also accept OpenAI-style keys for flexibility
    api_key  = HF_TOKEN or os.getenv("OPENAI_API_KEY")
    use_llm  = bool(api_key)

    if use_llm:
        print(f"LLM mode  |  endpoint={API_BASE_URL}  model={MODEL_NAME}", flush=True)
        client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    else:
        print(
            "No API key found -- running heuristic agent.\n"
            "Set HF_TOKEN + MODEL_NAME (or OPENAI_API_KEY + API_BASE_URL) to enable LLM mode.",
            flush=True,
        )
        client = None

    env = SilentFailureDetectorEnv(
        dataset_path="data/seed_dataset.jsonl",
        batch_size=32,
        seed=42,
    )

    scores: dict[str, float] = {}
    for task in ("easy", "medium", "hard"):
        scores[task] = evaluate_task(env, client, MODEL_NAME, task, use_llm)

    print("\n=== FINAL SCORES ===", flush=True)
    for task, score in scores.items():
        bar = "#" * int(score * 20)
        print(f"  {task.capitalize():<8}  {score:.4f}  {bar}", flush=True)
    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average   {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()