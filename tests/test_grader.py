import math
from src.grader import compute_confusion, compute_metrics, compute_reward

def test_perfect_recall():
    conf = compute_confusion([1,1,0,0], [1,1,0,0])
    m = compute_metrics(conf)
    score = compute_reward(m)
    assert 0.9 < score < 1.0

def test_degenerate_always_flag():
    conf = compute_confusion([1,0,1,0], [1,1,1,1])
    m = compute_metrics(conf)
    score = compute_reward(m)
    assert 0.0 < score < 0.1  # specificity poor, score low 

def test_degenerate_always_trust():
    conf = compute_confusion([1,0,1,0], [0,0,0,0])
    m = compute_metrics(conf)
    score = compute_reward(m)
    assert 0.0 < score < 0.1  # recall poor, score low
