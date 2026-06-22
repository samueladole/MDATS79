"""
RAGScope Dashboard — Page 2: Comparison
=========================================
Side-by-side RAGAS score and latency heatmaps across the
4 experimental conditions (2 LLMs × 2 retrieval strategies).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Comparison · RAGScope", layout="wide", page_icon="📊")
st.title("📊 Condition Comparison")
st.caption(
    "Comparing RAGAS evaluation scores and telemetry across the "
    "2×2 factorial experimental conditions."
)

# ── Load telemetry ────────────────────────────────────────────────────────────
from dashboard.components.heatmap import ragas_heatmap, token_cost_heatmap
from dashboard.components.latency_chart import latency_box_by_condition
from evaluation.metrics import compare_conditions
from telemetry.logger import get_telemetry_logger

records = get_telemetry_logger().load_all()

if not records:
    st.info(
        "No telemetry records found. Run some queries on the **Live Monitor** page "
        "or execute the benchmark experiment first."
    )
    st.stop()

df = pd.DataFrame(records)

st.markdown(f"Showing **{len(records):,}** telemetry records.")

# ── RAGAS heatmap ─────────────────────────────────────────────────────────────
st.subheader("RAGAS Score Heatmap")
ragas_heatmap(records)

st.subheader("Token Usage Heatmap")
token_cost_heatmap(records)

st.subheader("Latency Distribution by Condition")
latency_box_by_condition(records)

st.divider()

# ── Pairwise Cohen's d table ──────────────────────────────────────────────────
st.subheader("Pairwise Effect Sizes (Cohen's d)")
st.caption(
    "Computed for all RAGAS metrics between condition pairs. "
    "Interpretation: |d|<0.2 negligible · 0.2–0.5 small · 0.5–0.8 medium · >0.8 large"
)

if "llm_model" in df.columns and "retrieval_strategy" in df.columns:
    conditions = {
        "Llama3+Dense": df[(df.llm_model == "llama3") & (df.retrieval_strategy == "dense")].to_dict(
            "records"
        ),
        "Llama3+Hybrid": df[
            (df.llm_model == "llama3") & (df.retrieval_strategy == "hybrid")
        ].to_dict("records"),
        "Mistral+Dense": df[
            (df.llm_model == "mistral") & (df.retrieval_strategy == "dense")
        ].to_dict("records"),
        "Mistral+Hybrid": df[
            (df.llm_model == "mistral") & (df.retrieval_strategy == "hybrid")
        ].to_dict("records"),
    }

    pairs = [
        ("Llama3+Dense", "Mistral+Dense", "LLM effect (Dense)"),
        ("Llama3+Hybrid", "Mistral+Hybrid", "LLM effect (Hybrid)"),
        ("Llama3+Dense", "Llama3+Hybrid", "Retrieval effect (Llama3)"),
        ("Mistral+Dense", "Mistral+Hybrid", "Retrieval effect (Mistral)"),
    ]

    rows = []
    for a, b, label in pairs:
        comp = compare_conditions(conditions[a], conditions[b], label_a=a, label_b=b)
        for metric, data in comp.items():
            d_data = data["cohens_d"]
            rows.append(
                {
                    "Comparison": label,
                    "Metric": metric.replace("_", " ").title(),
                    "Cohen's d": d_data["d"],
                    "Magnitude": d_data["magnitude"],
                    f"Mean {a}": d_data["mean_a"],
                    f"Mean {b}": d_data["mean_b"],
                }
            )

    if rows:
        effect_df = pd.DataFrame(rows)
        st.dataframe(
            effect_df.style.background_gradient(subset=["Cohen's d"], cmap="RdYlGn"),
            width="stretch",
        )
else:
    st.info("Condition metadata not present in telemetry records.")
