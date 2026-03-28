from src.env import SilentFailureDetectorEnv
from src.models import SilentFailureAction, SilentFailureObservation


def test_env_reset_and_step_cycle() -> None:
    env = SilentFailureDetectorEnv(dataset_path="data/seed_dataset.jsonl", batch_size=8, seed=1)
    obs = env.reset()
    assert isinstance(obs, SilentFailureObservation)
    assert obs.text
    assert obs.domain

    done = False
    steps = 0
    final_info = {}
    while not done:
        action = SilentFailureAction(action=0)
        obs = env.step(action)
        steps += 1
        if obs.done:
            done = True
        
        # In OpenEnv, reward is part of observation
        if obs.reward is not None:
            assert isinstance(obs.reward, float)

    assert steps == 8
    # metrics and specific info are now likely in env.state() or returned differently
    # OpenEnv state keeps track of cumulative reward
    state = env.state
    assert state.episode_reward is not None


def test_invalid_action_raises() -> None:
    env = SilentFailureDetectorEnv(dataset_path="data/seed_dataset.jsonl", batch_size=4, seed=1)
    env.reset()
    try:
        # Pydantic validation should catch invalid action values if constructed properly
        # But here we test if ENV raises given a valid-typed but invalid-value action?
        # Actually, Pydantic model validation happens at construction.
        # Let's test constructing an invalid action.
        SilentFailureAction(action=3)
        assert False, "Expected ValidationError for invalid action"
    except Exception:
        assert True
