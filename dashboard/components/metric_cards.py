"""
RAGScope — Metric Card Components
====================================
Reusable Streamlit components for displaying RAGAS scores,
telemetry values, and hallucination risk in a consistent visual style.
"""

from __future__ import annotations

import streamlit as st

from evaluation.hallucination_score import risk_band, risk_colour


def ragas_scorecard(
    context_relevance: float | None,
    answer_faithfulness: float | None,
    answer_correctness: float | None,
    hallucination_risk: float | None,
) -> None:
    """
    Render a 4-column RAGAS scorecard row.
    Each metric is shown as a ``st.metric`` with a colour-coded delta hint.
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Context Relevance",
            value=_fmt_score(context_relevance),
            delta=_quality_delta(context_relevance),
            delta_color="normal",
            help="Proportion of retrieved context relevant to the query (RAGAS).",
        )

    with col2:
        st.metric(
            label="Answer Faithfulness",
            value=_fmt_score(answer_faithfulness),
            delta=_quality_delta(answer_faithfulness),
            delta_color="normal",
            help="Degree to which the answer is grounded in the retrieved context (RAGAS).",
        )

    with col3:
        st.metric(
            label="Answer Correctness",
            value=_fmt_score(answer_correctness),
            delta=_quality_delta(answer_correctness),
            delta_color="normal",
            help="Semantic match with the ground-truth answer (RAGAS). Requires ground truth.",
        )

    with col4:
        band = risk_band(hallucination_risk)
        color = risk_colour(hallucination_risk)
        value = f"{hallucination_risk:.3f}" if hallucination_risk is not None else "N/A"
        st.metric(
            label="Hallucination Risk",
            value=value,
            delta=band,
            delta_color="inverse" if band in ("HIGH", "MEDIUM") else "off",
            help="Composite risk score: 1 − (0.6×faithfulness + 0.4×context_relevance).",
        )
        # Colour indicator using markdown
        st.markdown(
            f'<div style="width:100%; height:4px; background:{color}; border-radius:2px;"></div>',
            unsafe_allow_html=True,
        )


def telemetry_row(
    retrieval_ms: float,
    generation_ms: float,
    e2e_ms: float,
    total_tokens: int,
    estimated_cost: float,
) -> None:
    """Render a 5-column telemetry row."""
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Retrieval Latency", f"{retrieval_ms:.0f} ms")
    c2.metric("Generation Latency", f"{generation_ms:.0f} ms")
    c3.metric("End-to-End Latency", f"{e2e_ms:.0f} ms")
    c4.metric("Total Tokens", f"{total_tokens:,}")
    c5.metric("Estimated Cost", f"${estimated_cost:.6f}")


def risk_gauge(score: float | None) -> None:
    """Render a simple horizontal risk gauge using Streamlit progress."""
    band = risk_band(score)
    color = risk_colour(score)
    value = score if score is not None else 0.0

    label_html = (
        f'<span style="color:{color}; font-weight:700; font-size:1.1rem;">'
        f"{band} ({value:.3f})"
        f"</span>"
    )
    st.markdown(f"**Hallucination Risk:** {label_html}", unsafe_allow_html=True)
    st.progress(value, text="")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fmt_score(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


def _quality_delta(v: float | None) -> str | None:
    """Return a descriptive delta label based on score band."""
    if v is None:
        return None
    if v >= 0.75:
        return "Good"
    if v >= 0.50:
        return "Fair"
    return "Poor"
