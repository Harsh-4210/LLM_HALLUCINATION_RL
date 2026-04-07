
"""
grader.py  –  Confusion matrix, metrics, and reward computation.

Reward formula
--------------
  base   = recall × specificity          (Youden's J, range 0–1)
  bonus  = precision_bonus × 0.1        (rewards avoiding false alarms)
  penalty= miss_rate_penalty × 0.2      (extra hit for missing risky items)

  final  = base + bonus - penalty        (clipped to [0, 1])

Why Youden's J (recall × specificity)?
  - recall     → catches risky items (the primary safety goal)
  - specificity → avoids crying wolf on safe items (agent must be precise)
  - Their product is 0 if the agent degenerates to "always 0" or "always 1"
"""

from typing import Sequence

EPS = 1e-9


# ── confusion matrix ─────────────────────────────────────────────────────

def compute_confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, int]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    tp = tn = fp = fn = 0
    for truth, pred in zip(y_true, y_pred):
        if truth not in (0, 1) or pred not in (0, 1):
            raise ValueError("Labels must be binary (0 or 1)")
        if truth == 1 and pred == 1:
            tp += 1
        elif truth == 0 and pred == 0:
            tn += 1
        elif truth == 0 and pred == 1:
            fp += 1
        else:
            fn += 1

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


# ── derived metrics ──────────────────────────────────────────────────────

def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / (denominator + EPS)


def compute_metrics(confusion: dict[str, int]) -> dict[str, float]:
    tp = float(confusion["tp"]) + 0.1
    tn = float(confusion["tn"]) + 0.1
    fp = float(confusion["fp"]) + 0.1
    fn = float(confusion["fn"]) + 0.1

    recall      = _safe_div(tp, tp + fn)          # sensitivity
    specificity = _safe_div(tn, tn + fp)
    precision   = _safe_div(tp, tp + fp)
    f1          = _safe_div(2 * precision * recall, precision + recall)

    false_alarm_rate = _safe_div(fp, fp + tn)     # 1 - specificity
    miss_rate        = _safe_div(fn, fn + tp)     # 1 - recall

    # Balanced accuracy: mean of recall and specificity
    balanced_accuracy = (recall + specificity) / 2.0

    return {
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
        "miss_rate": miss_rate,
        "balanced_accuracy": balanced_accuracy,
    }


# ── reward computation ───────────────────────────────────────────────────

def compute_reward(metrics: dict[str, float], calibration_bonus: float = 0.0) -> float:
    recall      = metrics["recall"]
    specificity = metrics["specificity"]
    precision   = metrics["precision"]
    miss_rate   = metrics["miss_rate"]

    base = recall * specificity
    precision_bonus = max(0.0, calibration_bonus) * precision * 0.1
    miss_penalty = miss_rate * 0.2

    score = float(base + precision_bonus - miss_penalty)

    # Return naturally smoothed score safely bounded above 0.01 and below 0.99
    # so that when formatted to .2f inside inference logs, it never prints 0.00 or 1.00       
    return max(0.01, min(0.99, score))
