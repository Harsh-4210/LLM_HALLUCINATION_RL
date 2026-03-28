import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="SilentFailureDetector Dashboard", layout="wide")
st.title("SilentFailureDetector MVP Dashboard")

baseline_path = Path("artifacts/baseline_metrics.json")
train_path = Path("artifacts/train_metrics.json")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Baseline Metrics")
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        st.json(baseline.get("metrics", {}))
        st.write(f"Reward: {baseline.get('reward_total', 0.0):.4f}")
    else:
        st.info("Run evaluation first: python src/eval/evaluate.py --agent rule_based --data data/seed_dataset.jsonl")

with col2:
    st.subheader("Training Curve")
    if train_path.exists():
        train_data = json.loads(train_path.read_text(encoding="utf-8"))
        history = pd.DataFrame(train_data.get("history", []))
        if not history.empty:
            st.line_chart(history.set_index("step")["reward"])
            st.write(f"Best threshold: {train_data.get('best_threshold', 0.0):.3f}")
            st.write(f"Best reward: {train_data.get('best_reward', 0.0):.4f}")
        st.json(train_data.get("final_metrics", {}))
    else:
        st.info("Run training first: python src/train/train_ppo.py --data data/seed_dataset.jsonl")
