import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


BASELINE_PATH = Path("artifacts/baseline_metrics.json")
TRAIN_PATH = Path("artifacts/train_metrics.json")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_num(value: float, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def fmt_pct(value: float) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}%"


st.set_page_config(
    page_title="SilentFailureDetector - Metrics Dashboard",
    page_icon="SFD",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

      .stApp {
        background: radial-gradient(circle at 10% -10%, rgba(255, 214, 153, 0.35), transparent 35%),
                    radial-gradient(circle at 100% 0%, rgba(130, 196, 255, 0.25), transparent 40%),
                    linear-gradient(180deg, #f6f8fb 0%, #f2f6ff 100%);
        font-family: 'Space Grotesk', sans-serif;
      }

      .main-card {
        padding: 1rem 1.15rem;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(28, 57, 108, 0.08);
        box-shadow: 0 8px 26px rgba(27, 48, 94, 0.08);
      }

      .headline {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.35rem;
        color: #17244f;
      }

      .subhead {
        color: #2f3d63;
        font-size: 1.02rem;
        margin-bottom: 1.1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

baseline = load_json(BASELINE_PATH)
train_data = load_json(TRAIN_PATH)

baseline_metrics = baseline.get("metrics", {})
all_tasks = baseline.get("all_tasks", {})
history = pd.DataFrame(train_data.get("history", []))
final_metrics = train_data.get("final_metrics", {})

st.markdown('<div class="headline">SilentFailureDetector Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subhead">Operational view of hallucination-risk detection performance across baseline and training runs.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Data Availability")
    st.write(f"Baseline artifact: {'Loaded' if baseline else 'Missing'}")
    st.write(f"Training artifact: {'Loaded' if train_data else 'Missing'}")

    if not baseline:
        st.info("Run evaluation: python -m src.eval.evaluate --agent rule_based --data data/seed_dataset.jsonl")
    if not train_data:
        st.info("Run training: python -m src.train.train_ppo --data data/seed_dataset.jsonl")

tab_overview, tab_baseline, tab_training = st.tabs([
    "Overview",
    "Baseline Deep Dive",
    "Training Analysis",
])

with tab_overview:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline Reward", fmt_num(baseline.get("reward_total", 0.0)))
    c2.metric("Baseline F1", fmt_pct(baseline_metrics.get("f1", 0.0)))
    c3.metric("Recall", fmt_pct(baseline_metrics.get("recall", 0.0)))
    c4.metric("Miss Rate", fmt_pct(baseline_metrics.get("miss_rate", 0.0)))
    st.markdown('</div>', unsafe_allow_html=True)

    if all_tasks:
        task_rows = []
        for task_name, task_data in all_tasks.items():
            metrics = task_data.get("metrics", {})
            task_rows.append(
                {
                    "task": task_name,
                    "reward_total": task_data.get("reward_total", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "specificity": metrics.get("specificity", 0.0),
                    "f1": metrics.get("f1", 0.0),
                    "miss_rate": metrics.get("miss_rate", 0.0),
                }
            )

        task_df = pd.DataFrame(task_rows).sort_values("reward_total", ascending=False)

        left, right = st.columns([1.2, 1])
        with left:
            st.subheader("Reward by Task")
            reward_chart = (
                alt.Chart(task_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("task:N", title="Task"),
                    y=alt.Y("reward_total:Q", title="Reward"),
                    color=alt.Color("task:N", legend=None),
                    tooltip=["task", alt.Tooltip("reward_total:Q", format=".4f")],
                )
                .properties(height=280)
            )
            st.altair_chart(reward_chart, use_container_width=True)

        with right:
            st.subheader("Task Scorecard")
            scorecard_df = task_df.copy()
            for col in ["recall", "specificity", "f1", "miss_rate"]:
                scorecard_df[col] = scorecard_df[col].map(lambda v: f"{100.0 * float(v):.2f}%")
            scorecard_df["reward_total"] = scorecard_df["reward_total"].map(lambda v: f"{float(v):.4f}")
            st.dataframe(scorecard_df, hide_index=True, use_container_width=True)
    else:
        st.warning("No task-level baseline data found in artifacts/baseline_metrics.json.")

with tab_baseline:
    if not all_tasks:
        st.warning("Baseline deep dive is unavailable until baseline metrics are generated.")
    else:
        task_options = sorted(all_tasks.keys())
        selected_task = st.selectbox("Select task", options=task_options, index=0)
        selected = all_tasks[selected_task]
        selected_metrics = selected.get("metrics", {})
        confusion = selected.get("confusion", {})

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Reward", fmt_num(selected.get("reward_total", 0.0)))
        k2.metric("Precision", fmt_pct(selected_metrics.get("precision", 0.0)))
        k3.metric("Balanced Accuracy", fmt_pct(selected_metrics.get("balanced_accuracy", 0.0)))
        k4.metric("False Alarm Rate", fmt_pct(selected_metrics.get("false_alarm_rate", 0.0)))

        metric_rows = []
        for task_name, task_data in all_tasks.items():
            task_metrics = task_data.get("metrics", {})
            for metric_name in ["recall", "specificity", "precision", "f1"]:
                metric_rows.append(
                    {
                        "task": task_name,
                        "metric": metric_name,
                        "value": task_metrics.get(metric_name, 0.0),
                    }
                )

        grouped_df = pd.DataFrame(metric_rows)
        grouped_chart = (
            alt.Chart(grouped_df)
            .mark_bar()
            .encode(
                x=alt.X("metric:N", title="Metric"),
                y=alt.Y("value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("task:N", title="Task"),
                xOffset="task:N",
                tooltip=["task", "metric", alt.Tooltip("value:Q", format=".4f")],
            )
            .properties(height=320)
        )
        st.subheader("Metric Distribution Across Tasks")
        st.altair_chart(grouped_chart, use_container_width=True)

        st.subheader(f"Confusion Matrix Values: {selected_task.title()}")
        confusion_df = pd.DataFrame(
            [
                {"label": "TP", "count": int(confusion.get("tp", 0))},
                {"label": "TN", "count": int(confusion.get("tn", 0))},
                {"label": "FP", "count": int(confusion.get("fp", 0))},
                {"label": "FN", "count": int(confusion.get("fn", 0))},
            ]
        )
        st.dataframe(confusion_df, hide_index=True, use_container_width=True)

with tab_training:
    if not train_data:
        st.warning("Training analysis is unavailable until artifacts/train_metrics.json exists.")
    else:
        t1, t2, t3 = st.columns(3)
        t1.metric("Best Threshold", fmt_num(train_data.get("best_threshold", 0.0), digits=3))
        t2.metric("Best Reward", fmt_num(train_data.get("best_reward", 0.0)))
        t3.metric("Final Reward", fmt_num(train_data.get("final_reward", 0.0)))

        if not history.empty:
            st.subheader("Reward Trajectory")
            reward_line = (
                alt.Chart(history)
                .mark_line(point=True)
                .encode(
                    x=alt.X("step:Q", title="Step"),
                    y=alt.Y("reward:Q", title="Reward"),
                    tooltip=[
                        alt.Tooltip("step:Q", format=".0f"),
                        alt.Tooltip("threshold:Q", format=".4f"),
                        alt.Tooltip("reward:Q", format=".4f"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(reward_line, use_container_width=True)

            st.subheader("Threshold Search History")
            history_table = history.copy()
            history_table["threshold"] = history_table["threshold"].map(lambda v: f"{float(v):.4f}")
            history_table["reward"] = history_table["reward"].map(lambda v: f"{float(v):.4f}")
            st.dataframe(history_table, hide_index=True, use_container_width=True)

        if final_metrics:
            st.subheader("Final Training Metrics")
            fm_cols = st.columns(6)
            fm_cols[0].metric("Recall", fmt_pct(final_metrics.get("recall", 0.0)))
            fm_cols[1].metric("Specificity", fmt_pct(final_metrics.get("specificity", 0.0)))
            fm_cols[2].metric("Precision", fmt_pct(final_metrics.get("precision", 0.0)))
            fm_cols[3].metric("F1", fmt_pct(final_metrics.get("f1", 0.0)))
            fm_cols[4].metric("False Alarm", fmt_pct(final_metrics.get("false_alarm_rate", 0.0)))
            fm_cols[5].metric("Miss Rate", fmt_pct(final_metrics.get("miss_rate", 0.0)))
