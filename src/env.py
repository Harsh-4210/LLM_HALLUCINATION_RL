"""SilentFailureDetector RL environment implementing the OpenEnv spec."""

import random
import uuid
from pathlib import Path
from typing import Any, Optional

from openenv.core.env_server import Environment

from src.dataset import Sample, load_dataset
from src.features import count_confidence_markers, count_hedging_markers, number_density
from src.grader import compute_confusion, compute_metrics, compute_reward
from src.models import SilentFailureAction, SilentFailureObservation, SilentFailureState


class SilentFailureDetectorEnv(
    Environment[SilentFailureAction, SilentFailureObservation, SilentFailureState]
):
    """OpenEnv environment for detecting confident-but-wrong AI outputs."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        dataset_path: str | Path = "data/seed_dataset.jsonl",
        batch_size: int = 32,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.all_samples = load_dataset(dataset_path)
        self.batch_size = min(batch_size, len(self.all_samples))
        self._rng = random.Random(seed)

        self.episode_samples: list[Sample] = []
        self.index = 0
        self.y_true: list[int] = []
        self.y_pred: list[int] = []
        self.total_reward = 0.0
        self._task_name = "easy"
        self._episode_id: str | None = None

    # ── task filtering ──────────────────────────────────────────────────
    def set_task(self, task_name: str) -> None:
        """Filter dataset by difficulty level for the next episode."""
        valid = {"easy", "medium", "hard"}
        if task_name not in valid:
            raise ValueError(f"task_name must be one of {valid}")
        self._task_name = task_name

    def _filtered_samples(self) -> list[Sample]:
        """Return samples matching the current task difficulty."""
        filtered = [
            s for s in self.all_samples
            if s.metadata.get("difficulty") == self._task_name
        ]
        return filtered if filtered else self.all_samples

    # ── observation builder ──────────────────────────────────────────────
    def _build_observation(
        self, sample: Sample, done: bool = False, reward: float | None = None,
    ) -> SilentFailureObservation:
        return SilentFailureObservation(
            id=sample.id,
            text=sample.response,
            domain=sample.domain,
            step_idx=self.index,
            confidence_marker_count=count_confidence_markers(sample.response),
            hedging_marker_count=count_hedging_markers(sample.response),
            number_density=number_density(sample.response),
            done=done,
            reward=reward,
        )

    # ── OpenEnv interface ────────────────────────────────────────────────
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SilentFailureObservation:
        if seed is not None:
            self._rng = random.Random(seed)
        self._episode_id = episode_id or str(uuid.uuid4())

        pool = self._filtered_samples()
        shuffled = list(pool)
        self._rng.shuffle(shuffled)
        self.episode_samples = shuffled[: min(self.batch_size, len(shuffled))]
        self.index = 0
        self.y_true = []
        self.y_pred = []
        self.total_reward = 0.0
        return self._build_observation(self.episode_samples[0])

    def _step_reward(self, truth: int, pred: int) -> float:
        if truth == 1 and pred == 1:
            return 1.0
        if truth == 0 and pred == 0:
            return 0.5
        if truth == 0 and pred == 1:
            return -1.0
        return -1.5  # missed a risky one

    def step(
        self,
        action: SilentFailureAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SilentFailureObservation:
        pred = action.action
        if pred not in (0, 1):
            raise ValueError("Action must be 0 or 1")
        if self.index >= len(self.episode_samples):
            raise RuntimeError("Episode is done. Call reset() before step().")

        sample = self.episode_samples[self.index]
        truth = sample.is_risky

        self.y_true.append(truth)
        self.y_pred.append(pred)

        reward = self._step_reward(truth, pred)
        self.total_reward += reward

        self.index += 1
        done = self.index >= len(self.episode_samples)

        if done:
            confusion = compute_confusion(self.y_true, self.y_pred)
            metrics = compute_metrics(confusion)
            final_bonus = compute_reward(metrics, calibration_bonus=0.0)
            reward += final_bonus
            self.total_reward += final_bonus
            # Return a terminal observation
            return SilentFailureObservation(
                id=sample.id,
                text=sample.response,
                domain=sample.domain,
                step_idx=self.index,
                confidence_marker_count=count_confidence_markers(sample.response),
                hedging_marker_count=count_hedging_markers(sample.response),
                number_density=number_density(sample.response),
                done=True,
                reward=reward,
                metadata={
                    "confusion": confusion,
                    "metrics": metrics,
                    "final_bonus": final_bonus,
                    "episode_reward": self.total_reward,
                    "sample_id": sample.id,
                    "label": sample.label,
                    "is_risky": truth,
                },
            )
        else:
            next_sample = self.episode_samples[self.index]
            obs = self._build_observation(next_sample, done=False, reward=reward)
            obs.metadata = {
                "sample_id": sample.id,
                "label": sample.label,
                "is_risky": truth,
            }
            return obs

    @property
    def state(self) -> SilentFailureState:
        return SilentFailureState(
            episode_id=self._episode_id,
            step_count=self.index,
            index=self.index,
            batch_size=len(self.episode_samples),
            predictions_made=len(self.y_pred),
            episode_reward=self.total_reward,
            task_name=self._task_name,
        )

    # ── hackathon helpers (used by custom endpoints) ─────────────────────
    def tasks(self) -> list[dict]:
        return [
            {
                "name": "easy",
                "description": "Detect obvious confident wrong claims with certainty terms.",
                "action_schema": {
                    "action": "int",
                    "values": [0, 1],
                    "meaning": {"0": "trust", "1": "flag_risky"},
                },
            },
            {
                "name": "medium",
                "description": "Detect mixed claims with subtle confidence markers.",
                "action_schema": {
                    "action": "int",
                    "values": [0, 1],
                    "meaning": {"0": "trust", "1": "flag_risky"},
                },
            },
            {
                "name": "hard",
                "description": "Handle adversarial phrasing and low lexical cues.",
                "action_schema": {
                    "action": "int",
                    "values": [0, 1],
                    "meaning": {"0": "trust", "1": "flag_risky"},
                },
            },
        ]

    def grader_score(self) -> dict:
        """Return grader result with score in 0.0–1.0 range."""
        if not self.y_true or not self.y_pred:
            return {"score": 0.0, "message": "No episode data yet."}
        confusion = compute_confusion(self.y_true, self.y_pred)
        metrics = compute_metrics(confusion)
        score = compute_reward(metrics, calibration_bonus=0.0)
        return {
            "score": round(score, 4),
            "confusion": confusion,
            "metrics": metrics,
        }
