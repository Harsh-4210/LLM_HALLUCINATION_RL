import os
import argparse
import re
from openai import OpenAI
from src.env import SilentFailureDetectorEnv
from src.models import SilentFailureAction
from src.agents.rule_based_agent import RuleBasedAgent

def parse_model_action(response_text: str) -> int:
    """Safely parse the LLM's response to extract an action (0 or 1)."""
    text = response_text.strip().lower()
    # Exact match
    if text == "1" or text == "0":
        return int(text)
    
    # Regex word boundary
    match = re.search(r'\b(0|1)\b', text)
    if match:
        return int(match.group(1))
    
    # Semantic keywords
    if "risky" in text or "flag" in text:
        return 1
    if "safe" in text or "trust" in text:
        return 0
        
    return 0

def main():
    # Fix HF Base URL handling
    default_base_url = "https://api.openai.com/v1"
    if "HF_TOKEN" in os.environ and "API_BASE_URL" not in os.environ:
        default_base_url = "https://api-inference.huggingface.co/v1"

    api_base_url = os.environ.get("API_BASE_URL", default_base_url)
    model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")

    if not hf_token:
        print("Warning: API token environment variable not set. OpenAI client might fail if the base URL requires authentication.")

    client = OpenAI(
        base_url=api_base_url,
        api_key=hf_token or "dummy_key",
    )

    env = SilentFailureDetectorEnv(dataset_path="data/seed_dataset.jsonl", batch_size=32, seed=42)
    fallback_agent = RuleBasedAgent()

    tasks_to_eval = ["easy", "medium", "hard"]
    final_scores = {}

    for task in tasks_to_eval:
        print(f"\n--- Starting Evaluation for Task: {task.upper()} ---")
        env.set_task(task)
        
        obs = env.reset()
        done = False
        step = 1

        while not done:
            prompt = (
                f"You are a failure detector. Review the following text and determine if it contains a silent failure, hallucination, or unsupported confident claim.\n\n"
                f"Text: {obs.text}\n"
                f"Domain: {obs.domain}\n"
                f"Confidence Markers: {obs.confidence_marker_count}\n"
                f"Hedging Markers: {obs.hedging_marker_count}\n"
                f"Number Density: {obs.number_density:.2f}\n\n"
                f"Is this text risky? Reply exactly with '1' if it is risky or '0' if it is safe and trustworthy."
            )

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a specialized classification bot. Output ONLY '0' or '1'."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=10
                )
                response_text = response.choices[0].message.content or ""
                action_val = parse_model_action(response_text)
                print(f"Step {step}: model suggested -> {action_val} (raw: {response_text.strip()})")
            except Exception as exc:
                # Use rule-based fallback if API request fails!
                print(f"Model request failed. Using fallback RuleBasedAgent action.")
                action_val = fallback_agent.act(obs)
                print(f"Step {step}: fallback suggested -> {action_val}")

            # Execute action
            action = SilentFailureAction(action=action_val)
            obs = env.step(action)
            
            reward = obs.reward if obs.reward is not None else 0.0
            print(f"  Step {step} Reward: {reward:+.2f} | Done: {obs.done}")
            
            done = obs.done
            step += 1

        grader_result = env.grader_score()
        score = grader_result.get("score", 0.0)
        print(f"Episode complete. Grader score for {task}: {score}")
        final_scores[task] = score

    print("\n=== FINAL BASELINE SCORES ===")
    for task, score in final_scores.items():
        print(f"{task.capitalize()}: {score}")

if __name__ == "__main__":
    main()
