"""
RAGScope — Heatmap Components
================================
Plotly heatmaps for comparing RAGAS metric scores across
the 2×2 factorial experiment conditions.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

METRICS = [
    "context_relevance",
    "answer_faithfulness",
    "answer_correctness",
    "hallucination_risk",
]

CONDITIONS = [
    ("llama3", "dense"),
    ("llama3", "hybrid"),
    ("mistral", "dense"),
    ("mistral", "hybrid"),
]

CONDITION_LABELS = {
    ("llama3", "dense"): "Llama3 + Dense",
    ("llama3", "hybrid"): "Llama3 + Hybrid",
    ("mistral", "dense"): "Mistral + Dense",
    ("mistral", "hybrid"): "Mistral + Hybrid",
}


def ragas_heatmap(records: list[dict]) -> None:
    """
    Render a heatmap of mean RAGAS scores across the 4 experimental conditions.
    Rows = conditions, Columns = metrics.
    """
    if not records:
        st.info("No benchmark results to visualise.")
        return

    df = pd.DataFrame(records)
    if "llm_model" not in df.columns:
        st.warning("Records do not contain condition metadata.")
        return

    rows = []
    col_labels = [m.replace("_", " ").title() for m in METRICS]

    for llm, strategy in CONDITIONS:
        subset = df[(df["llm_model"] == llm) & (df["retrieval_strategy"] == strategy)]
        row = []
        for metric in METRICS:
            vals = subset[metric].dropna() if metric in subset.columns else pd.Series(dtype=float)
            row.append(round(vals.mean(), 3) if not vals.empty else None)
        rows.append(row)

    row_labels = [CONDITION_LABELS[(llm, s)] for llm, s in CONDITIONS]
    z = [[v if v is not None else 0.0 for v in row] for row in rows]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=col_labels,
            y=row_labels,
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            text=[[f"{v:.3f}" if v else "—" for v in row] for row in rows],
            texttemplate="%{text}",
            hovertemplate="Condition: %{y}<br>Metric: %{x}<br>Mean Score: %{z:.3f}<extra></extra>",
            showscale=True,
            colorbar=dict(title="Score"),
        )
    )

    fig.update_layout(
        title="Mean RAGAS Scores by Experimental Condition",
        height=320,
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis=dict(side="top"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")


def token_cost_heatmap(records: list[dict]) -> None:
    """Render a heatmap of mean token usage and cost per condition."""
    if not records:
        return

    df = pd.DataFrame(records)
    if "llm_model" not in df.columns:
        return

    cost_metrics = ["prompt_tokens", "completion_tokens", "total_tokens"]
    col_labels = ["Prompt Tokens", "Completion Tokens", "Total Tokens"]
    rows, row_labels = [], []

    for llm, strategy in CONDITIONS:
        subset = df[(df["llm_model"] == llm) & (df["retrieval_strategy"] == strategy)]
        row = []
        for metric in cost_metrics:
            vals = subset[metric].dropna() if metric in subset.columns else pd.Series(dtype=float)
            row.append(int(vals.mean()) if not vals.empty else 0)
        rows.append(row)
        row_labels.append(CONDITION_LABELS[(llm, strategy)])

    max_val = max((v for row in rows for v in row if v), default=1)
    fig = go.Figure(
        go.Heatmap(
            z=rows,
            x=col_labels,
            y=row_labels,
            colorscale="Blues",
            zmin=0,
            zmax=max_val,
            text=[[f"{v:,}" for v in row] for row in rows],
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(title="Tokens"),
        )
    )

    fig.update_layout(
        title="Mean Token Usage by Experimental Condition",
        height=320,
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis=dict(side="top"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")
