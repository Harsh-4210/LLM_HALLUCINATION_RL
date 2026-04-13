"""
baseline.py  —  Root-level baseline runner for SilentFailureDetector.

Runs the rule-based heuristic agent across all three task levels (easy / medium / hard)
and prints structured results. Evaluators and reviewers run this to verify the
environment produces meaningful scores without any API key or GPU.

Usage:
    python baseline.py
    python baseline.py --data data/seed_dataset.jsonl --episodes 5
"""

import argparse
import json
from pathlib import Path

from src.eval.evaluate import evaluate_rule_based


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SilentFailureDetector rule-based baseline.")
    parser.add_argument("--data", default="data/seed_dataset.jsonl", help="Path to JSONL dataset.")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per task level.")
    parser.add_argument("--batch-size", type=int, default=32, help="Samples per episode.")
    parser.add_argument("--output", default="artifacts/baseline_metrics.json", help="Output JSON path.")
    args = parser.parse_args()

    print("=" * 60)
    print("SilentFailureDetector  —  Rule-Based Baseline Evaluation")
    print("=" * 60)

    all_results: dict[str, dict] = {}

    for task in ("easy", "medium", "hard"):
        print(f"\n[{task.upper()}]  Running {args.episodes} episodes...")
        result = evaluate_rule_based(
            dataset_path=args.data,
            batch_size=args.batch_size,
            episodes=args.episodes,
            task_name=task,  # type: ignore[arg-type]
        )
        all_results[task] = result

        m = result["metrics"]
        score = result["reward_total"]
        bar = "#" * int(score * 30)
        print(f"  score      = {score:.4f}  {bar}")
        print(f"  recall     = {m['recall']:.4f}   (catches hallucinations)")
        print(f"  specificity= {m['specificity']:.4f}   (avoids false alarms)")
        print(f"  precision  = {m['precision']:.4f}")
        print(f"  f1         = {m['f1']:.4f}")
        print(f"  miss_rate  = {m['miss_rate']:.4f}   (lower is better)")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    scores = {t: r["reward_total"] for t, r in all_results.items()}
    for task, score in scores.items():
        print(f"  {task.capitalize():<8}  {score:.4f}")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'Average':<8}  {avg:.4f}")
    print()

    # Persist for dashboard
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": all_results["easy"]["metrics"],
        "reward_total": all_results["easy"]["reward_total"],
        "all_tasks": all_results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
