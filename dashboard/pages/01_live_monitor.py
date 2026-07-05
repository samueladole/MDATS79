"""
RAGScope Dashboard — Page 1: Live Monitor
===========================================
Allows the user to submit queries interactively and observe
real-time RAGAS scores, telemetry, and retrieved chunks.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Live Monitor · RAGScope", layout="wide", page_icon="📡")
st.title("📡 Live Monitor")
st.caption("Submit a query and watch the full RAG pipeline execute in real time.")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Pipeline Configuration")
    llm_choice = st.selectbox("LLM Model", ["llama3", "mistral"], index=0)
    retrieval_choice = st.selectbox("Retrieval Strategy", ["hybrid", "dense"], index=0)
    top_k = st.slider("Top-k Chunks", min_value=1, max_value=10, value=5)
    run_eval = st.toggle("Run RAGAS Evaluation", value=True)
    st.divider()
    ground_truth = st.text_area(
        "Ground-Truth Answer (optional)",
        height=180,
        placeholder="Used for answer_correctness computation.",
    )

# ── Query form ────────────────────────────────────────────────────────────────
query_text = st.text_area(
    "Enter your question",
    height=100,
    placeholder="e.g. What are the main causes of hallucination in large language models?",
)

run_btn = st.button("▶  Run Query", type="primary", width="stretch")

if run_btn and query_text.strip():
    from dashboard.components.latency_chart import latency_time_series
    from dashboard.components.metric_cards import ragas_scorecard, telemetry_row
    from pipeline.rag import RAGPipeline
    from telemetry.logger import get_telemetry_logger

    try:
        with st.spinner("Running RAG pipeline …"):
            pipeline = RAGPipeline(
                llm_model=llm_choice,
                retrieval_strategy=retrieval_choice,
                top_k=top_k,
                run_evaluation=run_eval,
            )
            result = pipeline.query(
                query_text=query_text,
                ground_truth=ground_truth.strip() or None,
            )
    except Exception as exc:
        st.error(f"Pipeline error: {exc}")
        st.stop()

    st.success("Query complete.")
    st.divider()

    # ── Answer ────────────────────────────────────────────────────────────────
    st.subheader("Generated Answer")
    st.markdown(
        f'<div style="background:#f8f9fa;padding:1rem;border-left:4px solid #522D80;'
        f'border-radius:4px;">{result.answer}</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── RAGAS Scores ──────────────────────────────────────────────────────────
    st.subheader("Evaluation Scores")
    ragas_scorecard(
        result.context_relevance,
        result.answer_faithfulness,
        result.answer_correctness,
        result.hallucination_risk,
    )
    st.divider()

    # ── Telemetry ─────────────────────────────────────────────────────────────
    st.subheader("Telemetry")
    telemetry_row(
        result.retrieval_telemetry.get("retrieval_ms", 0),
        result.generation_response.generation_ms,
        result.e2e_ms,
        result.token_usage.total_tokens,
        result.token_usage.estimated_cost_usd,
    )
    st.divider()

    # ── Retrieved Chunks ──────────────────────────────────────────────────────
    st.subheader(f"Retrieved Chunks (top {len(result.retrieved_chunks)})")
    for i, chunk in enumerate(result.retrieved_chunks, 1):
        with st.expander(
            f"Chunk {i} — score: {chunk.score:.3f} "
            f"| dataset: {chunk.metadata.get('dataset', '?')} "
            f"| {chunk.chunk_id}"
        ):
            st.markdown(chunk.text)

    st.divider()

    # ── Historical latency chart ───────────────────────────────────────────────
    st.subheader("Session Latency History")
    records = get_telemetry_logger().load_all(limit=50)
    latency_time_series(records)

elif run_btn:
    st.warning("Please enter a query before running.")
