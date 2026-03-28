# Audit & Fix: RL_ENVIRONMENT_HALLUCINATION vs Hackathon Requirements

## Background
The project is an RL environment for detecting "confidently wrong" AI outputs (silent failures). We need to align it with the **Scaler Meta PyTorch OpenEnv Hackathon** Round 1 requirements.

---

## Hackathon Requirements Checklist vs Current State

| # | Requirement | Status | Gap |
|---|---|---|---|
| 1 | **Real-world task** (not games/toys) | ✅ Pass | Hallucination detection is a valid real-world task |
| 2 | **OpenEnv spec: typed Pydantic models** (`models.py` with Action, Observation, State extending `openenv.core`) | ❌ Fail | No `models.py`. Using raw dicts for obs/action/state instead of typed Pydantic models |
| 3 | **OpenEnv spec: [step()/reset()/state()](file:///c:/Users/Harsh%20Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/src/app/main.py#33-43) endpoints** | ⚠️ Partial | Endpoints exist but return raw dicts, not typed model responses |
| 4 | **[openenv.yaml](file:///c:/Users/Harsh%20Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/openenv.yaml) manifest** | ✅ Pass | Exists and has correct fields |
| 5 | **3+ tasks with graders** (easy → medium → hard, scores 0.0–1.0) | ⚠️ Partial | `/tasks` returns 3 tasks, but tasks aren't selectable — env doesn't filter by difficulty. Grader doesn't return per-task scores in 0.0–1.0 |
| 6 | **Meaningful reward with partial progress** | ⚠️ Partial | Has step-level rewards + episode bonus, but step rewards aren't normalized 0.0–1.0 |
| 7 | **Baseline inference script with reproducible scores** | ✅ Pass | [evaluate.py](file:///c:/Users/Harsh%20Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/src/eval/evaluate.py) works deterministically with seed |
| 8 | **Deploy to Hugging Face Spaces + Dockerfile** | ⚠️ Partial | Dockerfile exists but no HF Space config ([README.md](file:///c:/Users/Harsh%20Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/README.md) YAML frontmatter for Spaces) |
| 9 | **README with env description, action/obs spaces, setup** | ⚠️ Partial | README exists but lacks action/observation space formal docs |
| 10 | **`/baseline` endpoint** | ✅ Pass | Returns baseline scores |
| 11 | **`/grader` endpoint** — returns score 0.0–1.0 after episode | ⚠️ Partial | Returns metrics dict, but not a clean normalized [score](file:///c:/Users/Harsh%20Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/src/features.py#51-58) field per task |
| 12 | **`/tasks` endpoint** — returns task list + action schema | ✅ Pass | Works |
| 13 | **Pre-submission: HF Space returns 200, responds to [reset()](file:///c:/Users/Harsh%20Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/src/env.py#37-46)** | ❌ Fail | Not deployed yet |
| 14 | **Pre-submission: Docker builds** | ❓ Untested | Dockerfile exists, not verified |
| 15 | **`openenv-core` package usage** | ❌ Fail | Not installed or used. Custom env class instead of extending `openenv.core.env_server.Environment` |
| 16 | **Client module** (`client.py`) | ❌ Fail | Missing entirely |

---

## User Review Required

> [!CAUTION]
> **Critical decision: Should we adopt `openenv-core` framework fully?**
> The hackathon course (Module 4) shows environments extending `openenv.core.env_server.Environment` with `create_fastapi_app()`. Your current code uses a custom FastAPI app. Adopting `openenv-core` would require restructuring the server but ensures spec compliance. **I strongly recommend adopting it.**

> [!IMPORTANT]
> **HF Spaces deployment is mandatory.** Without it, the submission is auto-disqualified. We need to set up the Space configuration and verify deployment.

---

## Proposed Changes

### 1. Add openenv-core dependency & typed models

#### [NEW] [models.py](file:///c:/Users/Harsh Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/src/models.py)
- Define `SilentFailureAction(Action)` with `action: int` (0 or 1)
- Define `SilentFailureObservation(Observation)` with [id](file:///c:/Users/Harsh%20Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/src/dataset.py#28-45), `text`, `domain`, `step_idx`, `confidence_marker_count`, `hedging_marker_count`, [number_density](file:///c:/Users/Harsh%20Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/src/features.py#43-49), plus inherited `done: bool`, `reward: Optional[float]`
- Define `SilentFailureState(State)` with `index`, `batch_size`, `predictions_made`, `episode_reward`, plus inherited `episode_id`, `step_count`

---

### 2. Refactor environment to extend OpenEnv base class

#### [MODIFY] [env.py](file:///c:/Users/Harsh Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/src/env.py)
- Extend `openenv.core.env_server.Environment` instead of plain class
- [reset()](file:///c:/Users/Harsh%20Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/src/env.py#37-46) → return `SilentFailureObservation`
- [step(action: SilentFailureAction)](file:///c:/Users/Harsh%20Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/src/app/main.py#33-43) → return `SilentFailureObservation`
- [state](file:///c:/Users/Harsh%20Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/src/env.py#101-108) property → return `SilentFailureState`
- Support task filtering: accept `task_name` param to filter dataset by difficulty (easy/medium/hard)

---

### 3. Refactor FastAPI server to use `create_fastapi_app`

#### [MODIFY] [main.py](file:///c:/Users/Harsh Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/src/app/main.py)
- Use `create_fastapi_app(SilentFailureDetectorEnv)` for core endpoints
- Add custom `/tasks`, `/baseline`, `/grader` endpoints on top
- `/grader` must return `{"score": float}` with value in 0.0–1.0 per task

---

### 4. Add per-task difficulty filtering and grading

#### [MODIFY] [env.py](file:///c:/Users/Harsh Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/src/env.py)
- Add `set_task(task_name: str)` method to filter dataset samples by `metadata.difficulty`
- Episode only uses samples matching the selected task difficulty

#### [MODIFY] [grader.py](file:///c:/Users/Harsh Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/src/grader.py)
- Ensure [compute_reward()](file:///c:/Users/Harsh%20Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/src/grader.py#55-60) always returns a value in 0.0–1.0 range (it already does since it's recall × specificity × (1+bonus), all ≤1 when bonus=0)

---

### 5. Add OpenEnv client

#### [NEW] [client.py](file:///c:/Users/Harsh Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/src/client.py)
- Implement `SilentFailureEnv(EnvClient)` with `_step_payload`, `_parse_result`, `_parse_state`

---

### 6. HF Spaces deployment config

#### [MODIFY] [Dockerfile](file:///c:/Users/Harsh Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/Dockerfile)
- Expose port 7860 (HF Spaces default)
- Use `--port 7860` in CMD

#### [MODIFY] [README.md](file:///c:/Users/Harsh Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/README.md)
- Add HF Spaces YAML frontmatter (`title`, `emoji`, `sdk: docker`, `app_port: 7860`)
- Add full action/observation space documentation

---

### 7. Update requirements & dependencies

#### [MODIFY] [requirements.txt](file:///c:/Users/Harsh Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/requirements.txt)
- Add `openenv-core`

---

### 8. Fix tests for new typed interface

#### [MODIFY] [test_env_step.py](file:///c:/Users/Harsh Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/tests/test_env_step.py)
- Update to work with typed models

#### [MODIFY] [test_grader.py](file:///c:/Users/Harsh Jain/Desktop/PROJECTS/RL_ENVIRONMENT_HALLUCINATION/tests/test_grader.py)
- Add test verifying grader scores are in 0.0–1.0 range for all edge cases

---

## Verification Plan

### Automated Tests
1. `pytest tests/ -v` — all existing + updated tests pass
2. `docker build -t silent-failure-detector .` — Dockerfile builds without error
3. Start server locally: `uvicorn src.app.main:app --port 8000` and verify:
   - `GET /` returns 200
   - `POST /reset` returns typed observation
   - `POST /step` with `{"action": 1}` returns typed response
   - `GET /tasks` returns 3 tasks
   - `GET /baseline` returns scores for all tasks
   - `GET /grader` returns score in 0.0–1.0

### Manual Verification
- Deploy to HF Spaces and verify the Space URL returns 200 and responds to `/reset` (this is the pre-submission automated check)
- Submit HF Spaces URL to hackathon validator if available
