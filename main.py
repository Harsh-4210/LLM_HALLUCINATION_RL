"""FastAPI server combining OpenEnv core endpoints with hackathon custom endpoints."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
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


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to the interactive reviewer dashboard."""
    return RedirectResponse(url="/ui")


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui_dashboard():
    """Interactive HTML reviewer dashboard — inspect live episode state, scores and task config."""
    ui_path = Path("ui/index.html")
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))
    # Minimal inline fallback if file is missing
    return HTMLResponse(content="""
    <!DOCTYPE html><html><head><title>SilentFailureDetector</title>
    <meta charset='utf-8'/></head><body>
    <h1>SilentFailureDetector Dashboard</h1>
    <p>UI file not found. Run: <code>uvicorn main:app</code> from the project root.</p>
    </body></html>""")


@app.get("/info")
def info() -> dict:
    """Service metadata and runtime diagnostics."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
    model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
    return {
        "name": "SilentFailureDetector",
        "version": "0.1.0",
        "description": "OpenEnv RL environment for detecting confident-but-wrong AI outputs.",
        "tasks": ["easy", "medium", "hard"],
        "action_space": {"type": "discrete", "values": [0, 1], "meaning": {"0": "trust", "1": "flag_risky"}},
        "runtime": {
            "llm_mode": bool(api_key),
            "model": model_name if api_key else "heuristic_rule_based",
            "api_key_set": bool(api_key),
        },
    }


@app.get("/runtime-config")
def runtime_config() -> dict:
    """Expose exactly which backend is active — useful for debugging evaluator runs."""
    api_key    = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
    api_base   = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME",   "mistralai/Mistral-7B-Instruct-v0.3")
    return {
        "mode":        "llm" if api_key else "heuristic",
        "model":       model_name if api_key else None,
        "api_base_url": api_base  if api_key else None,
        "fallback":    "rule_based_heuristic",
        "note":        "Set HF_TOKEN + MODEL_NAME env vars to enable LLM mode.",
    }


@app.get("/health")
def health() -> dict:
    """Health check endpoint — returns 200 when service is up."""
    return {"status": "ok", "env": "SilentFailureDetector", "version": "0.1.0"}


@app.get("/tasks")
def tasks() -> dict:
    return {"tasks": _shared_env.tasks()}


@app.post("/baseline")
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
            "score":       result["reward_total"],
            "recall":      result["metrics"].get("recall", 0.0),
            "specificity": result["metrics"].get("specificity", 0.0),
            "f1":          result["metrics"].get("f1", 0.0),
            "miss_rate":   result["metrics"].get("miss_rate", 0.0),
        }
    return {"baseline": results}


@app.get("/grader")
def grader() -> dict:
    """Return grader score (0.0–1.0) for the last completed episode."""
    return _shared_env.grader_score()
