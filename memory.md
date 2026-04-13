# SilentFailureDetector — Project Memory

## 1. Project Identity & Context
- **Name:** SilentFailureDetector
- **Type:** Reinforcement Learning (RL) Environment (OpenEnv Compliant)
- **Context:** Built for the Meta × PyTorch × Scaler Hackathon (Round 1 Submission).
- **Core Mission:** Train and evaluate agents (LLMs or rule-based) to detect **silent failures** — AI-generated responses that are factually incorrect but stated with high confidence.
- **License:** BSD-3-Clause (`LICENSE` file present)

---

## 2. Problem Statement
The environment tackles AI hallucinations by classifying responses into four states:
1. Correct + Cautious → `is_risky = 0`
2. Correct + Confident → `is_risky = 0`
3. Wrong + Cautious → `is_risky = 0`
4. **Wrong + Confident (Silent Failure)** → `is_risky = 1` ← *primary target class*

---

## 3. Project File Structure
```
RL_ENVIRONMENT_HALLUCINATION/
├── main.py                    # FastAPI server (OpenEnv + custom endpoints)
├── inference.py               # LLM/heuristic agent evaluation runner
├── baseline.py                # Root-level baseline runner (required by evaluators)
├── dashboard.py               # Legacy Streamlit dashboard
├── sanity_check.py            # Pre-submission validation (35 checks)
├── agent_policy.json          # Structured agent policy description
├── openenv.yaml               # OpenEnv spec (all endpoints declared)
├── LICENSE                    # BSD-3-Clause
├── Dockerfile                 # Docker image for HuggingFace Spaces (port 7860)
├── requirements.txt           # Pinned deps (includes python-dotenv)
├── pyproject.toml             # Build config (hatchling)
├── data/
│   └── seed_dataset.jsonl     # 43 samples: 22 wrong_confident, 15 correct_confident, 6 correct_cautious
├── src/
│   ├── env.py                 # SilentFailureDetectorEnv (OpenEnv-compliant)
│   ├── grader.py              # Confusion matrix → metrics → Youden's J reward
│   ├── features.py            # Lexical feature extraction
│   ├── models.py              # Pydantic schemas (Action/Observation/State)
│   ├── dataset.py             # JSONL loader + validator
│   ├── client.py              # OpenEnv HTTP client
│   ├── agents/
│   │   └── rule_based_agent.py   # Deterministic heuristic baseline
│   ├── eval/
│   │   └── evaluate.py           # Multi-episode evaluation harness
│   └── train/
│       └── train_ppo.py          # Threshold search training hook
├── ui/
│   └── index.html             # Interactive HTML reviewer dashboard
└── tests/
    ├── test_grader.py         # Grader bounds + degenerate policy tests
    ├── test_env_step.py       # Full reset/step cycle tests
    └── test_features.py       # Feature extraction + bug regression tests
```

---

## 4. Environment Design (`src/env.py`)
**Class:** `SilentFailureDetectorEnv` — extends `openenv.core.env_server.Environment`

### Observation Fields
| Field | Type | Description |
|---|---|---|
| `text` | str | The AI response to evaluate |
| `domain` | str | medicine / law / finance / coding / science |
| `confidence_marker_count` | int | Count of certainty words |
| `hedging_marker_count` | int | Count of hedging words |
| `number_density` | float | Fraction of numeric tokens |
| `step_idx` | int | Current position in episode |

### Action Space
Binary: `0` (trust/safe) or `1` (flag as risky).

### Episode Flow
1. `reset()` → shuffles dataset filtered to task difficulty, returns first observation
2. `step(action)` → records prediction, emits per-step reward + optionally final bonus
3. `state` property → returns `SilentFailureState` with cumulative metrics
4. `grader_score()` → returns final score clamped to `(0.01, 0.99)`

### Task Levels
| Level | Description |
|---|---|
| `easy` | Obvious certainty terms, clear patterns |
| `medium` | Subtle markers, contextual reasoning required |
| `hard` | Adversarial phrasing, domain jargon, no obvious lexical cues |

---

## 5. Grading & Reward System (`src/grader.py`)

### Per-Step Rewards (asymmetric, intentional)
| Outcome | Reward | Rationale |
|---|---|---|
| True Positive (caught hallucination) | **+1.0** | Primary safety goal |
| True Negative (correctly trusted) | **+0.5** | Good precision |
| False Positive (false alarm) | **−1.0** | Disrupts workflow |
| False Negative (missed hallucination) | **−1.5** | Most dangerous outcome |

### Episode-End Score Formula
```
base    = recall × specificity          (Youden's J — collapses to 0 for degenerate agents)
penalty = miss_rate × 0.2
final   = max(0.01, min(0.99, base - penalty))
```

### Smoothing Note
`compute_metrics()` applies Laplace smoothing (+0.1 to all confusion matrix cells) to prevent division by zero. This is intentional — it slightly flattens perfect scores, which is expected.

### Critical Score Bounds
The final score is **always clamped to `[0.01, 0.99]`** — this satisfies the OpenEnv validator which rejects exact `0.0` or `1.0` values.

---

## 6. Feature Extraction (`src/features.py`)

### CERTAINTY_TERMS (high-risk signal)
`always, never, definitely, guaranteed, certainly, undoubtedly, proven, absolute, absolutely, conclusively, clearly, invariably, unquestionably, categorically`

> ⚠️ **Bug Fixed:** `"safe"` was previously incorrectly included in `CERTAINTY_TERMS`. It has been removed. "safe" is a positive semantic word, not a hallucination risk indicator.

### HEDGING_TERMS (low-risk signal)
`may, might, possibly, can, could, often, sometimes, likely, perhaps, generally, usually, typically, approximately, suggest, suggests, indicate, indicates, appears, seems`

### Risk Score Formula
```python
score = 0.7 * confidence_count - 0.5 * hedging_count + 2.0 * number_density
```

---

## 7. API Endpoints (`main.py`)

| Method | Path | Description |
|---|---|---|
| GET | `/` | Redirects to `/ui` |
| GET | `/ui` | **Interactive HTML reviewer dashboard** |
| GET | `/info` | Service metadata + runtime mode info |
| GET | `/runtime-config` | Whether LLM or heuristic backend is active |
| GET | `/health` | Health check |
| GET | `/tasks` | List task levels with action schema |
| GET | `/grader` | Grader score for last completed episode |
| GET | `/baseline` | Run rule-based baseline for all 3 tasks |
| POST | `/reset` | Start a new episode |
| POST | `/step` | Submit action, get next observation + reward |
| GET | `/state` | Inspect current episode state |
| GET | `/docs` | Interactive Swagger UI (auto-generated) |

---

## 8. Inference (`inference.py`)
Evaluates agents across all 3 task levels in sequence.

### LLM Mode (if API key set)
- **Backends:** OpenAI-compatible (also works with HuggingFace Inference API and Ollama)
- **System Prompt:** Explicitly balanced — states ~60% of responses are safe to prevent over-flagging bias
- **Action Parser:** Multi-strategy fallback (exact digit → word boundary → first digit → semantic keywords)
- **Max tokens:** 3 (strictly constrained to get "0" or "1")

### Heuristic Mode (no API key)
- Uses `confidence_marker_count`, `hedging_marker_count`, `number_density` directly from observation
- Different thresholds per task level (easy/medium vs hard)

### Environment Variables
| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI or Ollama key (use `ollama` for Ollama) |
| `HF_TOKEN` | HuggingFace Inference API token |
| `API_BASE_URL` | Custom API endpoint (default: HF inference URL) |
| `MODEL_NAME` | Model name (default: `mistralai/Mistral-7B-Instruct-v0.3`) |

---

## 9. Dataset (`data/seed_dataset.jsonl`)
- **Total samples:** 43
- **Label distribution:** `wrong_confident: 22`, `correct_confident: 15`, `correct_cautious: 6`
- **Note:** Dataset has no `wrong_cautious` examples currently — class imbalance to be aware of

### JSONL Schema
```json
{
  "id": "med_001",
  "domain": "medicine",
  "response": "Ibuprofen is completely safe at any dose.",
  "label": "wrong_confident",
  "confidence_markers": ["completely", "any"],
  "metadata": {
    "difficulty": "easy",
    "source": "synthetic"
  }
}
```

---

## 10. Key Files Added/Fixed (April 2026)

| File | Status | Change |
|---|---|---|
| `src/features.py` | **BUG FIX** | Removed `"safe"` from `CERTAINTY_TERMS`; expanded both term sets |
| `src/env.py` | **FIXED** | Implemented asymmetric per-step rewards (+1.0/+0.5/−1.0/−1.5) |
| `main.py` | **ENHANCED** | Added `/ui`, `/info`, `/runtime-config` endpoints; `/` now redirects to `/ui` |
| `ui/index.html` | **NEW** | Interactive HTML reviewer dashboard with live episode runner |
| `baseline.py` | **NEW** | Root-level baseline runner for evaluators |
| `sanity_check.py` | **NEW** | 35-check pre-submission validator |
| `agent_policy.json` | **NEW** | Structured policy description for judges/evaluators |
| `LICENSE` | **NEW** | BSD-3-Clause license file (was claimed in README but didn't exist) |
| `requirements.txt` | **FIXED** | Added `python-dotenv`, pinned `openenv-core>=0.2.1` |
| `openenv.yaml` | **UPDATED** | Added `/info`, `/runtime-config`, `/ui` endpoint declarations |
| `tests/test_features.py` | **NEW** | 12 tests including regression test for `"safe"` bug |

---

## 11. Test Suite (`tests/`)
**Total: 17 tests — all passing**
| File | Tests | Coverage |
|---|---|---|
| `test_grader.py` | 3 | Perfect recall, always-flag degenerate, always-trust degenerate |
| `test_env_step.py` | 2 | Full reset/step cycle, invalid action validation |
| `test_features.py` | 12 | Certainty/hedging detection, `"safe"` bug regression, number density, risk score |

---

## 12. Docker & Deployment
- **Image:** `python:3.11-slim`
- **Port:** `7860` (HuggingFace Spaces standard)
- **CMD:** `uvicorn main:app --host 0.0.0.0 --port 7860`
- **HF Space tags:** `openenv` (in README frontmatter)

---

## 13. Comparison to PrivacyGuard (Competitor)
| Criterion | SilentFailureDetector | PrivacyGuard |
|---|---|---|
| Task design | Binary action, 3 levels | Multi-dim JSON action + regulation bonus |
| Grading | Youden's J + miss penalty | Youden's J + regulation bonus |
| API completeness | `/ui`, `/info`, `/runtime-config` ✓ | Same |
| `sanity_check.py` | ✓ (35 checks) | ✓ |
| `agent_policy.json` | ✓ | ✓ |
| `LICENSE` | BSD-3-Clause ✓ | MIT ✓ |
| `baseline.py` at root | ✓ | ✓ |
| Interactive UI | HTML dashboard ✓ | HTML dashboard ✓ |
| Test count | 17 | ~10 est. |
| Dataset size | 43 | Larger est. |
| Per-step rewards | Asymmetric (+1/+0.5/-1/-1.5) ✓ | Non-negative |
