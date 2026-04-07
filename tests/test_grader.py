import math
from src.grader import compute_confusion, compute_metrics, compute_reward

def test_perfect_recall():
    conf = compute_confusion([1,1,0,0], [1,1,0,0])
    m = compute_metrics(conf)
    assert math.isclose(m["recall"], 1.0, rel_tol=1e-5)
    assert math.isclose(m["specificity"], 1.0, rel_tol=1e-5)

def test_degenerate_always_flag():
    conf = compute_confusion([1,0,1,0], [1,1,1,1])
    m = compute_metrics(conf)
    score = compute_reward(m)
    assert score == 0.001  # specificity = 0 â†’ collapses to near 0

def test_degenerate_always_trust():
    conf = compute_confusion([1,0,1,0], [0,0,0,0])
    m = compute_metrics(conf)
    score = compute_reward(m)
    assert score == 0.001  # recall = 0 â†’ collapses to near 0
