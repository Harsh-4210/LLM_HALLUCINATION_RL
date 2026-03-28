"""
evaluate.py  –  Reproducible baseline evaluation for SilentFailureDetector.

Usage (CLI):
    python -m src.eval.evaluate --agent rule_based --data data/seed_dataset.jsonl

Usage (import):
    from src.eval.evaluate import evaluate_rule_based
    result = evaluate_rule_based("data/seed_dataset.jsonl", task_name="easy")
"""

import argparse
import json
from pathlib import Path
from typing import Literal

from src.agents.rule_based_agent import RuleBasedAgent
from src.env import SilentFailureDetectorEnv
from src.grader import compute_confusion, compute_metrics, compute_reward
from src.models import SilentFailureAction


TaskName = Literal["easy", "medium", "hard"]


def evaluate_rule_based(
    dataset_path: str | Path = "data/seed_dataset.jsonl",
    batch_size: int = 32,
    episodes: int = 5,
    task_name: TaskName = "easy",
    threshold: float = 0.8,
) -> dict:
    """Run the rule-based agent for `episodes` episodes and return averaged metrics.

    Returns:
        {
            "task": str,
            "episodes": int,
            "reward_total": float,      # averaged grader score across episodes
            "confusion": dict,          # summed confusion matrix
            "metrics": dict,            # metrics computed from summed confusion
        }
    """
    env = SilentFailureDetectorEnv(
        dataset_path=dataset_path,
        batch_size=batch_size,
        seed=42,
    )
    env.set_task(task_name)
    agent = RuleBasedAgent(threshold=threshold)

    all_y_true: list[int] = []
    all_y_pred: list[int] = []
    episode_scores: list[float] = []

    for ep in range(episodes):
        obs = env.reset(seed=42 + ep)
        done = False

        while not done:
            action_val = agent.act(obs)
            obs = env.step(SilentFailureAction(action=action_val))
            done = obs.done

        result = env.grader_score()
        episode_scores.append(result.get("score", 0.0))
        all_y_true.extend(env.y_true)
        all_y_pred.extend(env.y_pred)

    # Aggregate
    confusion = compute_confusion(all_y_true, all_y_pred)
    metrics = compute_metrics(confusion)
    avg_score = sum(episode_scores) / len(episode_scores)

    return {
        "task": task_name,
        "episodes": episodes,
        "reward_total": round(avg_score, 4),
        "confusion": confusion,
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SilentFailureDetector baseline")
    parser.add_argument("--agent", choices=["rule_based"], default="rule_based")
    parser.add_argument("--data", default="data/seed_dataset.jsonl")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="artifacts/baseline_metrics.json")
    args = parser.parse_args()

    print(f"Evaluating {args.agent} agent across all 3 tasks...")

    results: dict[str, dict] = {}
    for task in ("easy", "medium", "hard"):
        print(f"\n  Task: {task.upper()}")
        r = evaluate_rule_based(
            dataset_path=args.data,
            batch_size=args.batch_size,
            episodes=args.episodes,
            task_name=task,  # type: ignore[arg-type]
        )
        results[task] = r
        m = r["metrics"]
        print(
            f"    score={r['reward_total']:.4f}  "
            f"recall={m['recall']:.2f}  "
            f"specificity={m['specificity']:.2f}  "
            f"f1={m['f1']:.2f}"
        )

    # Save for dashboard
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # dashboard.py reads the "easy" task result by default
    easy_result = results["easy"]
    out_path.write_text(
        json.dumps(
            {
                "metrics": easy_result["metrics"],
                "reward_total": easy_result["reward_total"],
                "all_tasks": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()