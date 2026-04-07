"""Rule-based agent for the SilentFailureDetector environment.

This agent uses the pre-built heuristics from features.py:
  - confidence marker count  (certainty words → higher risk)
  - hedging marker count     (hedging words   → lower risk)
  - number density           (many numbers    → higher risk of fabrication)
  - combined simple_risk_score()

No LLM or training required — useful as a reproducible baseline.
"""

from src.features import simple_risk_score
from src.models import SilentFailureObservation


class RuleBasedAgent:
    """Deterministic heuristic agent.

    Flags a response as risky when simple_risk_score() ≥ threshold.
    The threshold is tunable; default (0.8) was chosen to maximise
    F1 on the seed dataset's easy split.
    """

    def __init__(self, threshold: float = 0.4) -> None:
        self.threshold = threshold

    def act(self, obs: SilentFailureObservation) -> int:
        """Return 1 (risky) or 0 (safe) based on the observation text."""
        score = simple_risk_score(obs.text)
        return 1 if score >= self.threshold else 0

    def __repr__(self) -> str:
        return f"RuleBasedAgent(threshold={self.threshold})"