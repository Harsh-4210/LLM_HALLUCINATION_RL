"""Typed Pydantic models for the SilentFailureDetector OpenEnv environment."""

from typing import Any, Dict, List, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class SilentFailureAction(Action):
    """Agent action: flag a response as risky or trust it."""

    action: int = Field(ge=0, le=1, description="0 = trust, 1 = flag as risky")


class SilentFailureObservation(Observation):
    """Observation returned to the agent each step."""

    id: str = Field(default="", description="Sample identifier")
    text: str = Field(default="", description="The AI response text to evaluate")
    domain: str = Field(default="", description="Domain: medicine/law/finance/coding/science")
    step_idx: int = Field(default=0, description="Current step index in the episode")
    confidence_marker_count: int = Field(default=0, description="Count of certainty terms")
    hedging_marker_count: int = Field(default=0, description="Count of hedging terms")
    number_density: float = Field(default=0.0, description="Fraction of numeric tokens")


class SilentFailureState(State):
    """Internal environment state."""

    index: int = Field(default=0, description="Current step index")
    batch_size: int = Field(default=0, description="Episode batch size")
    predictions_made: int = Field(default=0, description="Predictions completed")
    episode_reward: float = Field(default=0.0, description="Cumulative episode reward")
    task_name: str = Field(default="easy", description="Current task difficulty")
