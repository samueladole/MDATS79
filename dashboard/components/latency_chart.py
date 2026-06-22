"""
RAGScope — Latency Chart Components
======================================
Plotly-based charts for visualising pipeline latency distributions
and time series on the Streamlit dashboard.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PURPLE = "#522D80"
TEAL = "#17a2b8"
AMBER = "#f39c12"


def latency_time_series(records: list[dict]) -> None:
    """
    Render a time-series line chart of end-to-end latency per query.
    Overlays retrieval and generation latency as stacked areas.
    """
    if not records:
        st.info("No telemetry records to display.")
        return

    df = pd.DataFrame(records)
    if "timestamp_utc" not in df.columns:
        st.warning("Telemetry records missing timestamps.")
        return

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df = df.sort_values("timestamp_utc").tail(200)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["timestamp_utc"],
            y=df.get("retrieval_ms", []),
            name="Retrieval",
            mode="lines",
            line=dict(color=TEAL, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(23,162,184,0.15)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp_utc"],
            y=df.get("generation_ms", []),
            name="Generation",
            mode="lines",
            line=dict(color=PURPLE, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(82,45,128,0.15)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp_utc"],
            y=df.get("e2e_ms", []),
            name="End-to-End",
            mode="lines",
            line=dict(color=AMBER, width=2, dash="dot"),
        )
    )

    fig.update_layout(
        title="Pipeline Latency Over Time",
        xaxis_title="Time (UTC)",
        yaxis_title="Latency (ms)",
        legend=dict(orientation="h", y=1.02),
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")


def latency_histogram(records: list[dict], metric: str = "e2e_ms") -> None:
    """Render a histogram of a latency metric."""
    if not records:
        return

    values = [r[metric] for r in records if r.get(metric) is not None]
    if not values:
        return

    label_map = {
        "e2e_ms": "End-to-End Latency (ms)",
        "retrieval_ms": "Retrieval Latency (ms)",
        "generation_ms": "Generation Latency (ms)",
    }

    fig = go.Figure(
        go.Histogram(
            x=values,
            nbinsx=30,
            marker_color=PURPLE,
            opacity=0.8,
        )
    )
    fig.update_layout(
        title=f"Distribution — {label_map.get(metric, metric)}",
        xaxis_title=label_map.get(metric, metric),
        yaxis_title="Count",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")


def latency_box_by_condition(records: list[dict]) -> None:
    """
    Render a grouped box plot of e2e latency split by condition
    (llm_model × retrieval_strategy).
    """
    if not records:
        return

    df = pd.DataFrame(records)
    if "llm_model" not in df.columns:
        return

    df["condition"] = df["llm_model"] + " + " + df["retrieval_strategy"]
    conditions = df["condition"].unique()
    colours = [PURPLE, TEAL, AMBER, "#e74c3c"]

    fig = go.Figure()
    for i, cond in enumerate(conditions):
        subset = df[df["condition"] == cond]["e2e_ms"].dropna()
        fig.add_trace(
            go.Box(
                y=subset,
                name=cond,
                marker_color=colours[i % len(colours)],
                boxmean=True,
            )
        )

    fig.update_layout(
        title="End-to-End Latency by Experimental Condition",
        yaxis_title="Latency (ms)",
        height=380,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")
