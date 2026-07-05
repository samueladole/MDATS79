"""
RAGScope Dashboard — Page 4: Benchmark Results
================================================
Visualises the results of the 200-query 2×2 factorial experiment.
Loads from the CSV output of ``experiments/run_2x2_factorial.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Benchmark Results · RAGScope", layout="wide", page_icon="📈")
st.title("📈 Benchmark Results")
st.caption(
    "Results of the 200-query, 2×2 factorial experiment "
    "(Llama3 vs Mistral × Dense vs Hybrid retrieval)."
)

# ── Load results CSV ──────────────────────────────────────────────────────────
RESULTS_DIR = Path("experiments/results")
csv_files = sorted(RESULTS_DIR.glob("*.csv"), reverse=True) if RESULTS_DIR.exists() else []

if not csv_files:
    st.info(
        "No benchmark results found. Run the experiment first:\n\n"
        "```bash\n"
        "uv run python experiments/run_2x2_factorial.py --dataset all --conditions all\n"
        "```"
    )
    st.stop()

selected_file = st.selectbox(
    "Select results file",
    options=[f.name for f in csv_files],
    index=0,
)
df = pd.read_csv(RESULTS_DIR / selected_file)
st.markdown(f"Loaded **{len(df):,}** records from `{selected_file}`.")
st.divider()

# ── Summary statistics ────────────────────────────────────────────────────────
st.subheader("Summary Statistics by Condition")
df["condition"] = df["llm_model"] + " + " + df["retrieval_strategy"]

METRICS = [
    "context_relevance",
    "answer_faithfulness",
    "answer_correctness",
    "hallucination_risk",
    "e2e_ms",
    "total_tokens",
]

agg = df.groupby("condition")[METRICS].agg(["mean", "std", "median"]).round(3)
st.dataframe(agg, width="stretch")
st.divider()

# ── Score distributions ───────────────────────────────────────────────────────
st.subheader("Score Distributions")
metric_choice = st.selectbox(
    "Metric",
    options=METRICS,
    format_func=lambda x: x.replace("_", " ").title(),
)

fig = px.box(
    df.dropna(subset=[metric_choice]),
    x="condition",
    y=metric_choice,
    color="condition",
    color_discrete_sequence=px.colors.qualitative.Set2,
    points="outliers",
    labels={"condition": "Condition", metric_choice: metric_choice.replace("_", " ").title()},
    title=f"{metric_choice.replace('_', ' ').title()} Distribution by Condition",
)
fig.update_layout(
    height=420,
    showlegend=False,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, width="stretch")

# ── Correlation: latency vs faithfulness ──────────────────────────────────────
st.subheader("Latency vs. Answer Faithfulness")
if "answer_faithfulness" in df.columns and "e2e_ms" in df.columns:
    fig2 = px.scatter(
        df.dropna(subset=["answer_faithfulness", "e2e_ms"]),
        x="e2e_ms",
        y="answer_faithfulness",
        color="condition",
        opacity=0.6,
        trendline="ols",
        labels={"e2e_ms": "End-to-End Latency (ms)", "answer_faithfulness": "Answer Faithfulness"},
        title="Latency vs. Faithfulness (all conditions)",
    )
    fig2.update_layout(
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig2, width="stretch")

# ── Per-dataset breakdown ─────────────────────────────────────────────────────
if "dataset" in df.columns:
    st.subheader("Performance by Dataset")
    dataset_agg = (
        df.groupby(["dataset", "condition"])[
            ["context_relevance", "answer_faithfulness", "hallucination_risk"]
        ]
        .mean()
        .round(3)
        .reset_index()
    )
    st.dataframe(dataset_agg, width="stretch")

# ── Raw data table ────────────────────────────────────────────────────────────
with st.expander("📋 Full Results Table"):
    st.dataframe(df, width="stretch")
    csv = df.to_csv(index=False)
    st.download_button(
        "⬇ Download CSV",
        data=csv,
        file_name=selected_file,
        mime="text/csv",
    )
