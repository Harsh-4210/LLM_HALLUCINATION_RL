---
title: SilentFailureDetector
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---
# SilentFailureDetector — OpenEnv RL Environment

> **Meta × PyTorch × Scaler Hackathon** · Round 1 Submission

An RL environment that trains agents to detect **silent failures** in LLM outputs: responses that are factually wrong but stated with high confidence — the most dangerous class of AI hallucination.

---

## Problem Statement

LLMs frequently produce outputs that *sound* authoritative but are incorrect. Unlike obvious failures ("I don't know"), **silent failures** are confident and fluent — they bypass human skepticism. This is especially dangerous in high-stakes domains: medicine, law, finance, and science.

**The agent's task:** Given an AI-generated response and lightweight text features, classify whether the response is a silent failure (risky) or trustworthy (safe).

```
Action space:  {0 = safe/trust,  1 = risky/flag}
Label types:   correct_confident | correct_cautious | wrong_confident* | wrong_cautious
                                                      ↑ the hard ones to catch
```

---

## Environment Design

### State Space

Each step presents the agent with an `observation`:

| Field | Type | Description |
|---|---|---|
| `text` | str | The AI response to evaluate |
| `domain` | str | medicine / law / finance / coding / science |
| `confidence_marker_count` | int | Count of certainty words (always, definitely, proven…) |
| `hedging_marker_count` | int | Count of hedging words (may, might, possibly…) |
| `number_density` | float | Fraction of numeric tokens in the response |
| `step_idx` | int | Current position in the episode |

### Action Space

Binary: `0` (trust) or `1` (flag as risky).

### Reward Structure

#### Per-step rewards (immediate feedback):

| Outcome | Reward | Rationale |
|---|---|---|
| True Positive (caught risky) | **+1.0** | Primary goal |
| True Negative (correctly trusted) | **+0.5** | Good precision |
| False Positive (cried wolf) | **−1.0** | Disrupts workflow |
| False Negative (missed risky) | **−1.5** | Most dangerous outcome |

The asymmetry is intentional: missing a harmful hallucination is worse than a false alarm.

#### Episode-end bonus:

```
score = recall × specificity − (miss_rate × 0.2)
```

- `recall × specificity` is **Youden's J statistic** — it collapses to 0 if the agent degenerates to "always flag" or "always trust", forcing genuine discrimination.
- The miss-rate penalty further penalises agents that ignore risky items.

### Task Levels

| Level | Description | Key challenge |
|---|---|---|
| **Easy** | Obvious confident wrong claims with clear certainty terms | Lexical pattern matching |
| **Medium** | Mixed claims with subtle confidence markers | Context-sensitive reasoning |
| **Hard** | Adversarial phrasing, low lexical cues, domain jargon | Semantic understanding |

---

## Architecture

```
SilentFailureDetectorEnv
├── dataset.py          Load + validate JSONL dataset (4-label schema)
├── features.py         Lightweight text feature extraction (no ML required)
├── grader.py           Confusion matrix → metrics → episode reward
├── models.py           Pydantic schemas for Action / Observation / State
└── env.py              OpenEnv-compliant Environment class
```

The environment is served over HTTP via FastAPI (`main.py`) and exposed with standard OpenEnv endpoints (`/reset`, `/step`, `/state`, `/health`).

---

## Quick Start

### Prerequisites

```bash
python -m pip install openenv-core fastapi uvicorn
```

### Run locally (rule-based agent, no API key needed)

```bash
python inference.py
# Uses deterministic heuristics from features.py as fallback
```

### Run with an LLM (HuggingFace, free)

```bash
export HF_TOKEN=hf_your_token_here
export MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
python inference.py
```

### Run with OpenAI

```bash
export OPENAI_API_KEY=sk-xxx
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-3.5-turbo
python inference.py
```

### Start the HTTP server

```bash
uvicorn main:app --reload
# API docs at http://localhost:8000/docs
```

### Run baseline evaluation

```bash
python -m src.eval.evaluate --agent rule_based --data data/seed_dataset.jsonl
```

### View dashboard

```bash
streamlit run dashboard.py
```

---

## Dataset Format

The environment reads a JSONL file where each line is:

```json
{
  "id": "med_001",
  "domain": "medicine",
  "response": "Ibuprofen is completely safe at any dose and has no side effects.",
  "label": "wrong_confident",
  "confidence_markers": ["completely", "any", "no"],
  "metadata": {
    "difficulty": "easy",
    "source": "synthetic"
  }
}
```

**Labels:**
- `wrong_confident` — silent failure (the target class, `is_risky = 1`)
- `correct_confident`, `correct_cautious`, `wrong_cautious` — all `is_risky = 0`

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/reset` | Start a new episode |
| POST | `/step` | Submit action, get next observation + reward |
| GET | `/state` | Inspect current episode state |
| GET | `/tasks` | List available task levels |
| GET | `/grader` | Get grader score (0–1) for last episode |
| GET | `/baseline` | Run full baseline evaluation |
| GET | `/health` | Service health check |
| GET | `/docs` | Interactive Swagger UI |

---

## Baseline Scores

Run `python inference.py` (heuristic mode — no API key needed) to reproduce these numbers:

| Task   | Score  | Recall | Specificity | F1     | Miss Rate |
|--------|--------|--------|-------------|--------|-----------|
| Easy   | 0.8325 | 0.8731 | 0.9985      | 0.9310 | 0.1269    |
| Medium | 0.5635 | 0.6656 | 0.9902      | 0.7976 | 0.3344    |
| Hard   | 0.0100 | 0.1269 | 0.9967      | 0.2247 | 0.8731    |

**Average (heuristic baseline):** 0.47

### Difficulty Curve

The steep drop Easy → Medium → Hard is intentional and validates the environment:

- **Easy (0.83):** Obvious certainty terms (`always`, `definitely`, `guaranteed`) — lexical pattern matching succeeds.
- **Medium (0.56):** Subtler markers, context-dependent — heuristic partially degrades. Requires stronger reasoning.
- **Hard (0.01):** Adversarial phrasing with low lexical cues — heuristic collapses. Requires semantic understanding that only capable LLMs can provide.

This design forces genuine discrimination. An LLM achieving >0.50 on Hard demonstrates real hallucination detection, not lexical hacking.

```
Heuristic agent:  Easy=0.83  Medium=0.56  Hard=0.01
Target LLM goal:  Easy=0.95  Medium=0.80  Hard=0.50+
```

---

## Evaluation Criteria

| Criterion | How we address it |
|---|---|
| **Runtime correctness** | `python inference.py` runs end-to-end; no errors; reproducible scores |
| **Interface compliance** | Full OpenEnv spec: typed Pydantic Action/Observation/State, step()/reset()/state() |
| **Task design** | 3 difficulty levels, 5 real-world domains, automated grading 0.01→0.99 |
| **Grading logic** | Youden's J + miss-rate penalty; collapses to minimum for degenerate policies |
| **Reward shaping** | Asymmetric per-step: TP=+1.0, TN=+0.5, FP=−1.0, FN=−1.5 |
| **Meaningful difficulty** | Hard task score of 0.01 with heuristic — requires LLM semantic reasoning to improve |

---

## Spec Compliance Checklist

| Requirement | Status |
|---|---|
| Real-world task (not games/toys) | ✅ Hallucination detection in medicine, law, finance, coding, science |
| Full OpenEnv spec | ✅ Typed models, step()/reset()/state(), openenv.yaml |
| Minimum 3 tasks (easy→hard) | ✅ Easy 0.83 → Medium 0.56 → Hard 0.01 |
| Scores in (0.0, 1.0) | ✅ Clamped to (0.01, 0.99) |
| Meaningful reward function | ✅ Asymmetric per-step rewards + Youden's J episode score |
| `inference.py` at root | ✅ Uses `openai.OpenAI` client, reads `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME` |
| `[START]/[STEP]/[END]` log format | ✅ Exact mandatory format with all fields |
| HuggingFace Spaces deploy | ✅ Dockerfile + `openenv` tag in README frontmatter |
| Working Dockerfile + HEALTHCHECK | ✅ Python 3.11-slim, HEALTHCHECK on /health, port 7860 |
| `openenv validate` passes | ✅ openenv.yaml declares all endpoints |
| Baseline reproduces | ✅ Same seed (42) → same scores every run |

---

## Real-World Application

Silent failure detection is critical for:
- **Clinical decision support** — wrong confident drug recommendations
- **Legal AI** — hallucinated case citations
- **Financial advisory** — fabricated statistics presented as facts
- **Code review** — confidently wrong explanations of buggy code

A trained agent from this environment could serve as a lightweight, API-callable **hallucination guard** to wrap any LLM deployment.

---

## Project Structure

```
.
├── data/
│   └── seed_dataset.jsonl
├── src/
│   ├── agents/
│   │   └── rule_based_agent.py
│   ├── eval/
│   │   └── evaluate.py
│   ├── train/
│   │   └── train_ppo.py
│   ├── env.py
│   ├── dataset.py
│   ├── features.py
│   ├── grader.py
│   ├── models.py
│   └── client.py
├── ui/
│   └── index.html
├── tests/
│   ├── test_env_step.py
│   ├── test_features.py
│   └── test_grader.py
├── inference.py
├── baseline.py
├── main.py
├── sanity_check.py
├── agent_policy.json
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## License

BSD-3-Clause — see [LICENSE](LICENSE)