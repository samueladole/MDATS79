"""
RAGScope — Streamlit Dashboard Entry Point
============================================
Multi-page Streamlit application that provides real-time observability
of the RAG pipeline, comparative analysis across experimental conditions,
query-level drill-down inspection, and benchmark experiment visualisation.

Pages
-----
  01_live_monitor.py     Live query execution with real-time metric display
  02_comparison.py       Cross-condition heatmap comparison
  03_query_explorer.py   Per-query telemetry and chunk inspection
  04_benchmark_results.py Benchmark experiment results and analysis

Run
---
    uv run streamlit run dashboard/app.py
    # or via Docker (auto-started by entrypoint.sh)
"""

from __future__ import annotations

import streamlit as st

from config.settings import settings

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAGScope — RAG Observability",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**RAGScope** · Real-Time Observability Platform for RAG Systems\n\n"
            "MSc Data Science Dissertation · Leeds Beckett University · 2026"
        ),
    },
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
"""
<style>
    /* Sidebar brand header */
    .sidebar-brand {
        font-size: 1.4rem;
        font-weight: 700;
        color: #522D80;
        letter-spacing: -0.02em;
        padding-bottom: 0.2rem;
    }
    .sidebar-sub {
        font-size: 0.75rem;
        color: #666;
        margin-bottom: 1rem;
    }
    /* Metric card improvements */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700;
    }
    /* Risk colour bands */
    .risk-low    { color: #2ecc71; font-weight: 700; }
    .risk-medium { color: #f39c12; font-weight: 700; }
    .risk-high   { color: #e74c3c; font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🔭 RAGScope</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-sub">Real-Time RAG Observability · Leeds Beckett University</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**Configuration**")
    st.caption(f"Embedding model: `{settings.embedding_model}`")
    st.caption(f"Default LLM: `{settings.default_llm}`")
    st.caption(f"Retrieval: `{settings.retrieval_strategy}`")
    st.caption(f"Top-k: `{settings.top_k}`")
    st.divider()

    # Live telemetry count
    try:
        from telemetry.logger import get_telemetry_logger

        n = get_telemetry_logger().count()
        st.metric("Telemetry Records", f"{n:,}")
    except Exception:
        st.caption("Telemetry logger unavailable")

    st.divider()
    st.markdown(
        "<small>MSc Data Science Dissertation · 2026</small>",
        unsafe_allow_html=True,
    )


# ── Home page ─────────────────────────────────────────────────────────────────
st.title("🔭 RAGScope")
st.subheader("Real-Time Observability and Evaluation Platform for RAG Systems")
st.markdown(
    """
    Navigate using the **sidebar pages** to access:

    | Page | Description |
    |---|---|
    | 📡 Live Monitor | Submit queries and observe real-time RAGAS scores and telemetry |
    | 📊 Comparison | Compare RAGAS metrics across the 4 experimental conditions |
    | 🔍 Query Explorer | Inspect individual query telemetry, retrieved chunks, and scores |
    | 📈 Benchmark Results | Visualise the 200-query experiment results and statistics |
    | 🛠️ Metrics Explorer | Dive into the details of each RAGAS metric |
    | ⚙️ Settings | Configure the RAG pipeline parameters and defaults |

    ---
    **Research context:** This platform was developed as the primary artefact for an
    MSc Data Science dissertation at Leeds Beckett University (2026), following the
    Design Science Research Methodology (Peffers et al., 2007).
    """
)
