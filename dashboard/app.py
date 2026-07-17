"""
RAGScope — Streamlit Dashboard Entry Point
============================================
Multi-page Streamlit application that provides real-time observability
of the RAG pipeline, comparative analysis across experimental conditions,
query-level drill-down inspection, and benchmark experiment visualisation.

Pages
-----
  01_live_monitor.py      Live query execution with real-time metric display
  02_comparison.py        Cross-condition heatmap comparison
  03_query_explorer.py    Per-query telemetry and chunk inspection
  04_benchmark_results.py Benchmark experiment results and analysis
  05_knowledge_base.py    ChromaDB collection browser and semantic search preview

Run
---
    uv run streamlit run dashboard/app.py
    # or via Docker (auto-started by entrypoint.sh)
"""

from __future__ import annotations

import streamlit as st

from config.settings import settings
from dashboard.components.theme import ICONS, apply_theme

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAGScope — RAG Observability",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**RAGScope** · Real-Time Observability Platform for RAG Systems\n\n"
            "MSc Data Science Dissertation · Leeds Beckett University · 2026"
        ),
    },
)

apply_theme()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div class="sidebar-brand">{ICONS["monitoring"]} RAGScope</div>',
        unsafe_allow_html=True,
    )
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
st.title(":material/monitoring: RAGScope")
st.subheader("Real-Time Observability and Evaluation Platform for RAG Systems")
st.markdown(
    """
    Navigate using the **sidebar pages** to access:

    | Page | Description |
    |---|---|
    | :material/monitor_heart: Live Monitor | Submit queries and observe real-time RAGAS scores and telemetry |
    | :material/compare_arrows: Comparison | Compare RAGAS metrics across the 4 experimental conditions |
    | :material/manage_search: Query Explorer | Inspect individual query telemetry, retrieved chunks, and scores |
    | :material/analytics: Benchmark Results | Visualise the 200-query experiment results and statistics |
    | :material/database: Knowledge Base | Browse the ChromaDB collection and preview semantic search |

    ---
    **Research context:** This platform was developed as the primary artefact for an
    MSc Data Science dissertation at Leeds Beckett University (2026), following the
    Design Science Research Methodology (Peffers et al., 2007).
    """
)
