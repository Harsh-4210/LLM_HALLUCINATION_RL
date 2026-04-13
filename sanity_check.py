"""
sanity_check.py  —  Pre-submission validation for SilentFailureDetector.

Runs a series of checks to verify the environment is working correctly
before pushing to the hackathon platform.

Usage:
    python sanity_check.py
    python sanity_check.py --url http://localhost:7860   # test live server

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed.
"""

import argparse
import json
import sys
from pathlib import Path

# ── Output helpers ─────────────────────────────────────────────────────────
PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def check(name: str, condition: bool, warning: bool = False) -> bool:
    tag = (WARN if warning else FAIL) if not condition else PASS
    print(f"  {tag}  {name}")
    return condition


def section(title: str) -> None:
    print(f"\n{'-' * 55}")
    print(f"  {title}")
    print(f"{'-' * 55}")


def run_local_checks() -> list[bool]:
    results: list[bool] = []

    # ── 1. File structure ────────────────────────────────────────────────
    section("1. File Structure")
    critical_files = [
        "main.py", "inference.py", "baseline.py", "requirements.txt",
        "Dockerfile", "openenv.yaml", "agent_policy.json", "LICENSE",
        "data/seed_dataset.jsonl",
        "src/env.py", "src/grader.py", "src/features.py",
        "src/models.py", "src/dataset.py",
        "src/agents/rule_based_agent.py",
        "src/eval/evaluate.py",
        "src/train/train_ppo.py",
        "ui/index.html",
    ]
    for f in critical_files:
        results.append(check(f, Path(f).exists()))

    # ── 2. Dataset validation ────────────────────────────────────────────
    section("2. Dataset")
    dataset_path = Path("data/seed_dataset.jsonl")
    if dataset_path.exists():
        lines = [l for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        results.append(check(f"Dataset non-empty ({len(lines)} samples)", len(lines) > 0))

        valid_labels = {"correct_cautious", "correct_confident", "wrong_cautious", "wrong_confident"}
        label_counts: dict[str, int] = {}
        parse_errors = 0
        for line in lines:
            try:
                row = json.loads(line)
                lbl = row.get("label", "MISSING")
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
            except Exception:
                parse_errors += 1

        results.append(check("No JSONL parse errors", parse_errors == 0))
        results.append(check("All labels are valid", all(k in valid_labels for k in label_counts)))
        results.append(check(
            f"Contains wrong_confident (target class): {label_counts.get('wrong_confident', 0)} samples",
            label_counts.get("wrong_confident", 0) > 0,
        ))
        print(f"    Label distribution: {label_counts}")
    else:
        results.append(check("Dataset file found", False))

    # ── 3. Environment smoke test ────────────────────────────────────────
    section("3. Environment Smoke Test")
    try:
        from src.env import SilentFailureDetectorEnv
        from src.models import SilentFailureAction

        for task in ("easy", "medium", "hard"):
            env = SilentFailureDetectorEnv(dataset_path="data/seed_dataset.jsonl", batch_size=8, seed=99)
            env.set_task(task)
            obs = env.reset()
            step_count = 0
            done = False
            while not done:
                obs = env.step(SilentFailureAction(action=1))
                done = obs.done
                step_count += 1
            score = env.grader_score()["score"]
            results.append(check(
                f"Task '{task}': ran {step_count} steps, score={score:.4f}, in (0.01, 0.99)",
                0.01 <= score <= 0.99,
            ))
    except Exception as exc:
        results.append(check(f"Environment smoke test — EXCEPTION: {exc}", False))

    # ── 4. Grader bounds ────────────────────────────────────────────────
    section("4. Grader Score Bounds")
    try:
        from src.grader import compute_confusion, compute_metrics, compute_reward

        # Always-flag degenerate agent
        y_true = [1, 0, 1, 0] * 8
        y_pred_all1 = [1] * 32
        y_pred_all0 = [0] * 32
        y_pred_perfect = y_true

        for label, y_pred in [("always-flag", y_pred_all1), ("always-trust", y_pred_all0), ("perfect", y_pred_perfect)]:
            s = compute_reward(compute_metrics(compute_confusion(y_true, y_pred)))
            results.append(check(f"Score '{label}' in (0.01, 0.99): {s:.4f}", 0.01 <= s <= 0.99))
    except Exception as exc:
        results.append(check(f"Grader bounds — EXCEPTION: {exc}", False))

    # ── 5. Features bug check ───────────────────────────────────────────
    section("5. Features Module")
    try:
        from src.features import count_confidence_markers, count_hedging_markers, CERTAINTY_TERMS

        # "safe" must NOT be a certainty term (it was a bug)
        results.append(check('"safe" is NOT in CERTAINTY_TERMS (bug fix)', "safe" not in CERTAINTY_TERMS))

        safe_text = "This medication is safe and may help patients."
        conf = count_confidence_markers(safe_text)
        hedge = count_hedging_markers(safe_text)
        results.append(check(f'Safe hedging sentence: confidence={conf}, hedging={hedge} (want conf<2, hedge>0)', conf < 2 and hedge > 0))

        risky_text = "This drug is definitely always safe and guaranteed to work."
        conf2 = count_confidence_markers(risky_text)
        results.append(check(f'Risky sentence: confidence={conf2} (want ≥2)', conf2 >= 2))
    except Exception as exc:
        results.append(check(f"Features module — EXCEPTION: {exc}", False))

    # ── 6. openenv.yaml ─────────────────────────────────────────────────
    section("6. OpenEnv YAML")
    yaml_path = Path("openenv.yaml")
    if yaml_path.exists():
        content = yaml_path.read_text(encoding="utf-8")
        results.append(check("Has 'reset' endpoint", "reset" in content))
        results.append(check("Has 'step' endpoint", "step" in content))
        results.append(check("Has 'state' endpoint", "state" in content))
        results.append(check("Has 'grader' endpoint", "grader" in content))
    else:
        results.append(check("openenv.yaml exists", False))

    return results


def run_server_checks(base_url: str) -> list[bool]:
    try:
        import requests
    except ImportError:
        print(f"  {WARN}  'requests' not installed — skipping server checks")
        return []

    results: list[bool] = []
    section(f"7. Live Server Checks  ({base_url})")

    endpoints = ["/health", "/info", "/runtime-config", "/tasks", "/grader"]
    for ep in endpoints:
        try:
            r = requests.get(f"{base_url}{ep}", timeout=5)
            results.append(check(f"GET {ep}  →  HTTP {r.status_code}", r.status_code == 200))
        except Exception as exc:
            results.append(check(f"GET {ep}  →  {exc}", False))

    # Full reset/step cycle
    try:
        r = requests.post(f"{base_url}/reset", json={}, timeout=10)
        results.append(check(f"POST /reset  →  HTTP {r.status_code}", r.status_code == 200))
        if r.status_code == 200:
            r2 = requests.post(f"{base_url}/step", json={"action": 0}, timeout=10)
            results.append(check(f"POST /step  →  HTTP {r2.status_code}", r2.status_code == 200))
    except Exception as exc:
        results.append(check(f"reset/step cycle  →  {exc}", False))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-submission sanity check for SilentFailureDetector.")
    parser.add_argument("--url", default=None, help="Base URL of live server to test (e.g. http://localhost:7860).")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  SilentFailureDetector  —  Sanity Check")
    print("=" * 55)

    results = run_local_checks()

    if args.url:
        results += run_server_checks(args.url)

    total = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"\n{'=' * 55}")
    print(f"  Results:  {passed}/{total} checks passed")
    if failed:
        print(f"  *** {failed} check(s) FAILED -- fix before submitting! ***")
    else:
        print(f"  All checks passed. Ready to submit!")
    print(f"{'=' * 55}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
