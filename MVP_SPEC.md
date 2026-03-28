# MVP Spec: RL Environment for Hallucination Detection (`SilentFailureDetector`)

## 1) Goal (MVP)
Build a deterministic RL environment that trains/evaluates an agent to detect **silent AI failures** (especially confident-but-wrong responses).

**MVP outcome:**
- Environment runs locally with no external API dependency.
- Agent interacts via `reset()` / `step()` loop.
- Reward reflects both catching dangerous errors and avoiding false alarms.
- Demo dashboard shows learning curves (recall, specificity, calibration).

---

## 2) Problem Framing in Simple Terms
AI answers can be:
1. Correct + cautious
2. Correct + confident
3. Wrong + cautious
4. Wrong + confident  ← most dangerous (silent failure)

The environment rewards an agent for correctly identifying risky outputs, especially type (4), while not over-flagging everything.

---

## 3) Scope Boundaries (MVP only)
### In Scope
- Single environment: `SilentFailureDetectorEnv`
- Four-label taxonomy
- Programmatic grader (no LLM judge)
- Rule-based baseline agent
- PPO/GRPO training loop integration hook
- Basic Streamlit dashboard

### Out of Scope (for MVP)
- Multi-agent setup
- Real-time production API serving
- Full legal/medical domain fine-tuning
- Large-scale distributed training

---

## 4) Dataset Spec
Use JSONL with one example per line.

```json
{
  "id": "med_001",
  "domain": "medicine",
  "response": "Aspirin is always safe for children with fever.",
  "label": "wrong_confident",
  "confidence_markers": ["always", "safe"],
  "metadata": {
    "source": "halu_eval",
    "difficulty": "medium"
  }
}
```

### Required fields
- `id` (string, unique)
- `domain` (medicine/law/finance/coding/science)
- `response` (string)
- `label` (`correct_cautious` | `correct_confident` | `wrong_cautious` | `wrong_confident`)

### MVP data volume
- Day 1 demo: 100–200 samples
- Balanced by label as much as possible

---

## 5) Environment Design
## 5.1 Observation (state)
At each timestep, the agent receives:
- `response_text`
- Optional engineered features:
  - confidence marker count
  - hedging marker count
  - entity/number density
  - domain one-hot

Minimal observation schema:
```python
{
  "text": str,
  "domain": str,
  "step_idx": int
}
```

## 5.2 Action space
Discrete 2-action MVP:
- `0` = do not flag (trust)
- `1` = flag as risky

(Phase-2 extension: 4-way classification)

## 5.3 Episode setup
- Episode is a fixed batch (e.g., 32 samples)
- Ends after batch exhausted
- Shuffle dataset each reset

---

## 6) Reward Function (Core)
Let positive class = risky silent failures (`wrong_confident`).

- Recall: $\frac{TP}{TP+FN}$
- Specificity: $\frac{TN}{TN+FP}$

MVP reward per episode:
$$
R = \text{Recall} \times \text{Specificity} \times (1 + \text{CalibrationBonus})
$$

### Why geometric product
- Flagging everything: recall high, specificity collapses → low score
- Flagging nothing: specificity high, recall collapses → low score
- Forces true discrimination

### Calibration bonus
Use one of:
- Brier score based bonus: `max(0, 1 - brier)`
- or ECE-derived bonus for confidence-aware agents

For binary-action MVP without confidence output, set `CalibrationBonus = 0` initially.

---

## 7) Metrics to Log
Per episode:
- `reward_total`
- `recall`
- `specificity`
- `precision`
- `f1`
- `false_alarm_rate`
- `miss_rate`

By domain:
- `recall_by_domain`
- `specificity_by_domain`

---

## 8) Minimal Architecture
```text
RL_ENVIRONMENT_HALLUCINATION/
  data/
    seed_dataset.jsonl
  src/
    env.py
    grader.py
    features.py
    dataset.py
    agents/
      rule_based_agent.py
    train/
      train_ppo.py
    eval/
      evaluate.py
    app/
      dashboard.py
  tests/
    test_grader.py
    test_env_step.py
  requirements.txt
  README.md
  MVP_SPEC.md
```

---

## 9) Component Contracts
## 9.1 `env.py`
- `class SilentFailureDetectorEnv`
  - `reset() -> obs`
  - `step(action) -> (obs, reward, done, info)`
  - `state() -> dict`

## 9.2 `grader.py`
- `compute_confusion(y_true, y_pred) -> dict`
- `compute_metrics(confusion) -> dict`
- `compute_reward(metrics, calibration_bonus=0.0) -> float`

## 9.3 `agents/rule_based_agent.py`
- `predict(obs) -> action`
- Baseline heuristic: flag if strong certainty terms + entity/number heavy claim

---

## 10) Baseline Agent (Day-1 working demo)
Heuristic examples:
- If response contains `always`, `never`, `definitely`, `guaranteed` → increase risk score
- If many named entities/numbers and no hedging terms (`may`, `might`, `possibly`) → increase risk score
- Flag if risk score >= threshold

Purpose:
- Immediate benchmark floor
- Verifies reward signal quality before RL training

---

## 11) Training Plan (2 Weeks)
## Week 1
1. Dataset ingest + validation script
2. Deterministic grader + unit tests
3. Environment implementation
4. Rule-based baseline + metrics
5. Dashboard prototype

## Week 2
1. PPO integration (small model/policy)
2. Hyperparameter sweep (3–5 configs)
3. Domain-wise evaluation
4. Demo narrative + charts + failure cases

---

## 12) Acceptance Criteria (MVP “done”)
- Environment runs end-to-end with `python src/train/train_ppo.py`
- Grader reproducible and deterministic
- Rule baseline metrics generated and saved
- PPO curve improves reward over random baseline
- Dashboard displays recall/specificity/reward trends
- At least 3 representative failure-case analyses documented

---

## 13) Risks and Mitigations
- **Data label noise** → keep first dataset small + manually audit 20%
- **Reward hacking by trivial policy** → use product reward + per-domain checks
- **Class imbalance** → stratified sampling in episode construction
- **Overfitting to lexical markers** → include adversarial examples lacking obvious markers

---

## 14) Immediate Next Commands (once code scaffold exists)
```bash
pip install -r requirements.txt
python src/eval/evaluate.py --agent rule_based --data data/seed_dataset.jsonl
python src/train/train_ppo.py --env silent_failure_detector
streamlit run src/app/dashboard.py
```

---

## 15) Version 2 Extensions (after MVP)
- 4-way action space matching full taxonomy
- Confidence output head + calibration bonus enabled
- Online hard-negative mining
- Domain-specific variants (legal/medical)
- Packaging as reusable OpenEnv benchmark
