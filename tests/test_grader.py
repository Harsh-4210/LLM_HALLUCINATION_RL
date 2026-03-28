from src.grader import compute_confusion, compute_metrics, compute_reward


def test_confusion_counts() -> None:
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]
    confusion = compute_confusion(y_true, y_pred)
    assert confusion == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}


def test_metrics_range() -> None:
    confusion = {"tp": 8, "tn": 12, "fp": 3, "fn": 2}
    metrics = compute_metrics(confusion)
    for key, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"{key} is out of range"


def test_reward_zero_when_recall_or_specificity_zero() -> None:
    metrics_zero_recall = {
        "recall": 0.0,
        "specificity": 1.0,
        "precision": 0.0,
        "f1": 0.0,
        "false_alarm_rate": 0.0,
        "miss_rate": 1.0,
    }
    assert compute_reward(metrics_zero_recall) == 0.0
