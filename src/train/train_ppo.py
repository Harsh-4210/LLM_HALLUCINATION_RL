"""MVP training hook — threshold search over the feature-based risk score."""

import argparse
import json
import random
from pathlib import Path

from src.env import SilentFailureDetectorEnv
from src.models import SilentFailureAction


class ThresholdAgent:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def predict(self, obs) -> int:
        confidence = obs.confidence_marker_count if hasattr(obs, "confidence_marker_count") else obs.get("confidence_marker_count", 0)
        hedging = obs.hedging_marker_count if hasattr(obs, "hedging_marker_count") else obs.get("hedging_marker_count", 0)
        density = obs.number_density if hasattr(obs, "number_density") else obs.get("number_density", 0.0)
        score = 0.7 * confidence - 0.5 * hedging + 2.0 * density
        return 1 if score >= self.threshold else 0


def run_episode(env: SilentFailureDetectorEnv, agent: ThresholdAgent) -> tuple[float, dict]:
    obs = env.reset()
    done = False
    total = 0.0
    info = {}
    while not done:
        action_val = agent.predict(obs)
        action = SilentFailureAction(action=action_val)
        obs = env.step(action)
        reward_val = obs.reward if obs.reward is not None else 0.0
        total += reward_val
        done = obs.done
        info = obs.metadata
    return total, info


def train_threshold_search(dataset_path: str, iterations: int, batch_size: int, seed: int) -> dict:
    rng = random.Random(seed)
    env = SilentFailureDetectorEnv(dataset_path=dataset_path, batch_size=batch_size, seed=seed)

    best_threshold = 1.0
    best_reward = float("-inf")
    history = []

    for step in range(iterations):
        candidate = rng.uniform(0.2, 2.0)
        agent = ThresholdAgent(candidate)
        reward, info = run_episode(env, agent)

        history.append({"step": step, "threshold": candidate, "reward": reward})
        if reward > best_reward:
            best_reward = reward
            best_threshold = candidate

    final_agent = ThresholdAgent(best_threshold)
    final_reward, final_info = run_episode(env, final_agent)

    return {
        "trainer": "threshold_search_hook",
        "note": "MVP training hook. Replace with PPO/GRPO in next phase.",
        "best_threshold": best_threshold,
        "best_reward": best_reward,
        "final_reward": final_reward,
        "final_metrics": final_info.get("metrics", {}),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="silent_failure_detector")
    parser.add_argument("--data", default="data/seed_dataset.jsonl")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/train_metrics.json")
    args = parser.parse_args()

    result = train_threshold_search(
        dataset_path=args.data,
        iterations=args.iterations,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result["final_metrics"], indent=2))
    print(f"Saved training metrics to {output_path}")


if __name__ == "__main__":
    main()
