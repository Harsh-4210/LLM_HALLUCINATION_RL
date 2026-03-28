import re

CERTAINTY_TERMS = {
    "always",
    "never",
    "definitely",
    "guaranteed",
    "certainly",
    "undoubtedly",
    "proven",
    "safe",
}

HEDGING_TERMS = {
    "may",
    "might",
    "possibly",
    "can",
    "could",
    "often",
    "sometimes",
    "likely",
}


_TOKEN_RE = re.compile(r"\b\w+\b")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def count_confidence_markers(text: str) -> int:
    tokens = set(tokenize(text))
    return len(tokens.intersection(CERTAINTY_TERMS))


def count_hedging_markers(text: str) -> int:
    tokens = set(tokenize(text))
    return len(tokens.intersection(HEDGING_TERMS))


def number_density(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    numeric = sum(1 for token in tokens if token.isdigit())
    return numeric / len(tokens)


def simple_risk_score(text: str) -> float:
    confidence_count = count_confidence_markers(text)
    hedging_count = count_hedging_markers(text)
    density = number_density(text)

    score = 0.7 * confidence_count - 0.5 * hedging_count + 2.0 * density
    return max(0.0, score)
