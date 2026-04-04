from src.grader import compute_confusion, compute_metrics, compute_reward

def test_perfect_recall():
    conf = compute_confusion([1,1,0,0], [1,1,0,0])
    m = compute_metrics(conf)
    assert m["recall"] == 1.0
    assert m["specificity"] == 1.0

def test_degenerate_always_flag():
    conf = compute_confusion([1,0,1,0], [1,1,1,1])
    m = compute_metrics(conf)
    score = compute_reward(m)
    assert score == 0.0  # specificity = 0 → collapses to 0

def test_degenerate_always_trust():
    conf = compute_confusion([1,0,1,0], [0,0,0,0])
    m = compute_metrics(conf)
    score = compute_reward(m)
    assert score == 0.0  # recall = 0 → collapses to 0
