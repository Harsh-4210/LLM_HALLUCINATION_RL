"""
tests/test_features.py  —  Unit tests for src/features.py

Critical regression test: "safe" must NOT be in CERTAINTY_TERMS.
That was a bug causing safe responses to be incorrectly flagged.
"""

from src.features import (
    CERTAINTY_TERMS,
    HEDGING_TERMS,
    count_confidence_markers,
    count_hedging_markers,
    number_density,
    simple_risk_score,
)


# ── CERTAINTY_TERMS bug regression ───────────────────────────────────────────

def test_safe_not_in_certainty_terms():
    """REGRESSION: 'safe' was previously in CERTAINTY_TERMS causing mis-classifications."""
    assert "safe" not in CERTAINTY_TERMS, (
        "'safe' must NOT be in CERTAINTY_TERMS — it is a positive/safety word, "
        "not a hallucination risk marker."
    )


def test_certainty_terms_are_risk_words():
    """Every word in CERTAINTY_TERMS should intuitively signal strong (risky) confidence."""
    benign_safe_words = {"safe", "okay", "fine", "good", "helpful"}
    overlap = CERTAINTY_TERMS & benign_safe_words
    assert not overlap, f"Benign safe words should not be in CERTAINTY_TERMS: {overlap}"


# ── count_confidence_markers ─────────────────────────────────────────────────

def test_confidence_markers_on_risky_text():
    text = "This drug is definitely always guaranteed to cure cancer."
    count = count_confidence_markers(text)
    assert count >= 3, f"Expected ≥3 certainty markers, got {count}"


def test_confidence_markers_zero_on_neutral_text():
    text = "The patient might benefit from further evaluation."
    count = count_confidence_markers(text)
    assert count == 0, f"Expected 0 certainty markers in hedged text, got {count}"


def test_confidence_markers_text_with_safe_word():
    """'safe' appearing in text must NOT contribute to confidence marker count."""
    text = "This medication is safe and well-tolerated in adults."
    count = count_confidence_markers(text)
    assert count == 0, (
        f"'safe' should not count as a certainty marker. Got {count}. "
        "This was a bug — regression check."
    )


# ── count_hedging_markers ────────────────────────────────────────────────────

def test_hedging_markers_detected():
    text = "This treatment may possibly help, but results could vary."
    count = count_hedging_markers(text)
    assert count >= 2, f"Expected ≥2 hedging markers, got {count}"


def test_hedging_markers_zero_on_confident_text():
    text = "This is definitely the correct and absolute answer."
    count = count_hedging_markers(text)
    assert count == 0, f"Expected 0 hedging markers in confident text, got {count}"


# ── number_density ────────────────────────────────────────────────────────────

def test_number_density_zero_for_no_numbers():
    text = "The patient should rest and drink water."
    assert number_density(text) == 0.0


def test_number_density_nonzero_for_numbers():
    text = "The dose is 500mg taken 3 times daily."
    density = number_density(text)
    assert density > 0.0, f"Expected non-zero density, got {density}"


def test_number_density_empty_string():
    assert number_density("") == 0.0


# ── simple_risk_score ─────────────────────────────────────────────────────────

def test_risk_score_higher_for_risky_text():
    risky = "This drug definitely always cures cancer with a 100% success rate every time."
    safe  = "This medication may help reduce symptoms in some patients."
    assert simple_risk_score(risky) > simple_risk_score(safe)


def test_risk_score_non_negative():
    texts = [
        "may might possibly could",
        "",
        "definitely always guaranteed proven",
    ]
    for t in texts:
        assert simple_risk_score(t) >= 0.0, f"Risk score should be non-negative for: {t!r}"
