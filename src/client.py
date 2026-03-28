"""OpenEnv client for the SilentFailureDetector environment."""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from src.models import SilentFailureAction, SilentFailureObservation, SilentFailureState


class SilentFailureEnv(
    EnvClient[SilentFailureAction, SilentFailureObservation, SilentFailureState]
):
    """Client for interacting with a deployed SilentFailureDetector environment."""

    def _step_payload(self, action: SilentFailureAction) -> dict:
        return {"action": action.action}

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=SilentFailureObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                id=obs_data.get("id", ""),
                text=obs_data.get("text", ""),
                domain=obs_data.get("domain", ""),
                step_idx=obs_data.get("step_idx", 0),
                confidence_marker_count=obs_data.get("confidence_marker_count", 0),
                hedging_marker_count=obs_data.get("hedging_marker_count", 0),
                number_density=obs_data.get("number_density", 0.0),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> SilentFailureState:
        return SilentFailureState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            index=payload.get("index", 0),
            batch_size=payload.get("batch_size", 0),
            predictions_made=payload.get("predictions_made", 0),
            episode_reward=payload.get("episode_reward", 0.0),
            task_name=payload.get("task_name", "easy"),
        )
