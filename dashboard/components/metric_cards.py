"""
RAGScope — Metric Card Components
====================================
Reusable Streamlit components for displaying RAGAS scores,
telemetry values, and hallucination risk in a consistent visual style.
"""

from __future__ import annotations

import streamlit as st

from evaluation.hallucination_score import risk_band, risk_colour

# Same red/amber/green triplet as risk_colour(), so a score reads consistently
# whether it's shown as a banded risk colour or this continuous gradient.
_RED = (231, 76, 60)
_AMBER = (243, 156, 18)
_GREEN = (46, 204, 113)
_GREY = "#95a5a6"


def score_gradient_color(value: float | None, invert: bool = False) -> str:
    """
    Interpolate a red -> amber -> green colour for a score in [0, 1].

    Parameters
    ----------
    value  : Score in [0, 1]. Returns a neutral grey if None.
    invert : Set True for metrics where *lower* is better (e.g. risk), so
             the colour still reads red=bad / green=good rather than
             red=low / green=high.
    """
    if value is None:
        return _GREY
    v = max(0.0, min(1.0, value))
    if invert:
        v = 1.0 - v
    stops = [(0.0, _RED), (0.5, _AMBER), (1.0, _GREEN)]
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if p0 <= v <= p1:
            t = (v - p0) / (p1 - p0)
            r, g, b = (round(c0[i] + (c1[i] - c0[i]) * t) for i in range(3))
            return f"rgb({r},{g},{b})"
    return _GREY


def similarity_meter_html(score: float, label: str = "Similarity") -> str:
    """
    Return an HTML snippet: a slim colour-graded meter bar for a [0, 1]
    score — green high, red low, with the numeric score alongside so the
    reading never depends on colour alone.
    """
    color = score_gradient_color(score)
    pct = max(0.0, min(1.0, score)) * 100
    return (
        '<div style="margin:2px 0 6px 0;">'
        '<div style="display:flex;justify-content:space-between;font-size:0.75rem;'
        f'color:#666;margin-bottom:2px;"><span>{label}</span>'
        f'<span style="font-weight:600;color:{color};">{score:.3f}</span></div>'
        '<div style="background:#e9ecef;border-radius:6px;height:8px;width:100%;overflow:hidden;">'
        f'<div style="background:{color};height:100%;width:{pct:.1f}%;border-radius:6px;"></div>'
        "</div></div>"
    )


def _score_card_html(
    icon: str,
    label: str,
    value: float | None,
    help_text: str,
    invert: bool = False,
) -> str:
    """Build one modern scorecard tile: icon, label, big value, meter, quality tag."""
    color = score_gradient_color(value, invert=invert)
    display_value = f"{value:.3f}" if value is not None else "N/A"
    pct = (1.0 - value if invert else value) * 100 if value is not None else 0.0
    # Hallucination risk already has an established LOW/MEDIUM/HIGH banding
    # (risk_band, used by the heatmaps and risk-band charts elsewhere) — reuse
    # it here instead of introducing a second, differently-thresholded scheme.
    quality = risk_band(value) if invert else (_quality_delta(value) or "No data")
    return f"""
    <div title="{help_text}" style="background:rgba(127,127,127,0.06);
                border:1px solid rgba(127,127,127,0.18);border-radius:12px;
                padding:0.9rem 1rem;height:100%;">
      <div style="display:flex;align-items:center;gap:0.4rem;margin-bottom:0.35rem;">
        <span style="font-size:1.05rem;">{icon}</span>
        <span style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                     letter-spacing:0.03em;opacity:0.65;">{label}</span>
      </div>
      <div style="font-size:1.65rem;font-weight:700;margin-bottom:0.5rem;">{display_value}</div>
      <div style="background:rgba(127,127,127,0.2);border-radius:6px;height:7px;
                  width:100%;overflow:hidden;margin-bottom:0.4rem;">
        <div style="background:{color};height:100%;width:{pct:.1f}%;border-radius:6px;"></div>
      </div>
      <div style="font-size:0.72rem;font-weight:700;color:{color};">{quality}</div>
    </div>
    """


def ragas_scorecard(
    context_relevance: float | None,
    answer_faithfulness: float | None,
    answer_correctness: float | None,
    hallucination_risk: float | None,
) -> None:
    """Render the 4-metric RAGAS scorecard as modern colour-graded tiles."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            _score_card_html(
                "🎯",
                "Context Relevance",
                context_relevance,
                "Proportion of retrieved context relevant to the query (RAGAS).",
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            _score_card_html(
                "🔗",
                "Answer Faithfulness",
                answer_faithfulness,
                "Degree to which the answer is grounded in the retrieved context (RAGAS).",
            ),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            _score_card_html(
                "✅",
                "Answer Correctness",
                answer_correctness,
                "Semantic match with the ground-truth answer (RAGAS). Requires ground truth.",
            ),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            _score_card_html(
                "⚠️",
                "Hallucination Risk",
                hallucination_risk,
                "Composite risk score: 1 − (0.6×faithfulness + 0.4×context_relevance).",
                invert=True,
            ),
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
