"""FastAPI server combining OpenEnv core endpoints with hackathon custom endpoints."""

from pathlib import Path

from fastapi import FastAPI
from openenv.core.env_server import create_fastapi_app

from src.agents.rule_based_agent import RuleBasedAgent
from src.env import SilentFailureDetectorEnv
from src.eval.evaluate import evaluate_rule_based
from src.models import SilentFailureAction, SilentFailureObservation

DATA_PATH = Path("data/seed_dataset.jsonl")


def _env_factory() -> SilentFailureDetectorEnv:
    return SilentFailureDetectorEnv(dataset_path=DATA_PATH, batch_size=32, seed=42)


# Create the OpenEnv-compliant app with /reset, /step, /state, /health, /docs etc.
app = create_fastapi_app(
    env=_env_factory,
    action_cls=SilentFailureAction,
    observation_cls=SilentFailureObservation,
)

app.title = "SilentFailureDetector OpenEnv API"
app.version = "0.1.0"

# Keep a shared instance for the custom hackathon endpoints
_shared_env = _env_factory()
_agent = RuleBasedAgent()


# ── Hackathon-required custom endpoints ─────────────────────────────────


@app.get("/")
def health() -> dict:
    return {"status": "ok", "env": "SilentFailureDetector"}


@app.get("/tasks")
def tasks() -> dict:
    return {"tasks": _shared_env.tasks()}


@app.get("/baseline")
def baseline() -> dict:
    """Run baseline agent and return reproducible scores for all 3 tasks."""
    results = {}
    for task_name in ("easy", "medium", "hard"):
        result = evaluate_rule_based(
            dataset_path=str(DATA_PATH),
            batch_size=32,
            episodes=5,
            task_name=task_name,
        )
        results[task_name] = {
            "score": result["reward_total"],
            "metrics": result["metrics"],
            "confusion": result["confusion"],
        }
    return {"baseline": results}


@app.get("/grader")
def grader() -> dict:
    """Return grader score (0.0–1.0) for the last completed episode."""
    return _shared_env.grader_score()
