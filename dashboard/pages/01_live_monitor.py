"""
RAGScope Dashboard — Page 1: Live Monitor
===========================================
Allows the user to submit queries interactively and observe
real-time RAGAS scores, telemetry, and retrieved chunks.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from dashboard.components.latency_chart import latency_time_series
from dashboard.components.metric_cards import (
    ragas_scorecard,
    similarity_meter_html,
    telemetry_row,
)

st.set_page_config(page_title="Live Monitor · RAGScope", layout="wide", page_icon="📡")
st.title("📡 Live Monitor")
st.caption("Submit a query and watch the full RAG pipeline execute in real time.")

st.session_state.setdefault("session_history", [])
st.session_state.setdefault("last_result", None)

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
    if st.session_state.session_history:
        st.divider()
        if st.button("🗑 Clear Session History", width="stretch"):
            st.session_state.session_history = []
            st.session_state.last_result = None
            st.rerun()

# ── Query form ────────────────────────────────────────────────────────────────
query_text = st.text_area(
    "Enter your question",
    height=100,
    placeholder="e.g. What are the main causes of hallucination in large language models?",
)

run_btn = st.button("▶  Run Query", type="primary", width="stretch")

if run_btn and query_text.strip():
    from pipeline.rag import RAGPipeline

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

    st.session_state.last_result = {
        "result": result,
        "query_text": query_text,
        "llm_choice": llm_choice,
        "retrieval_choice": retrieval_choice,
        "top_k": top_k,
        "run_eval": run_eval,
    }
    st.session_state.session_history.append(
        {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "retrieval_ms": result.retrieval_telemetry.get("retrieval_ms", 0),
            "generation_ms": result.generation_response.generation_ms,
            "e2e_ms": result.e2e_ms,
        }
    )
    st.success("Query complete.")
elif run_btn:
    st.warning("Please enter a query before running.")

# ── Render the most recent result ─────────────────────────────────────────────
# Reads from session_state rather than gating on run_btn, so the result stays
# on screen across reruns triggered by other widgets (e.g. expanding a chunk
# below) instead of disappearing the instant anything else is clicked.
if st.session_state.last_result:
    data = st.session_state.last_result
    result = data["result"]

    st.divider()
    st.caption(
        f"Condition: **{data['llm_choice'].title()} + {data['retrieval_choice'].title()}** "
        f"· top_k={data['top_k']} · RAGAS eval: {'on' if data['run_eval'] else 'off'} "
        f"· Query: _{data['query_text'][:100]}_"
    )

    # ── Answer ────────────────────────────────────────────────────────────────
    st.subheader("Generated Answer")
    st.markdown(
        f'<div style="background:rgba(82,45,128,0.08);padding:1rem;'
        f'border-left:4px solid #522D80;border-radius:4px;">{result.answer}</div>',
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
        st.markdown(
            similarity_meter_html(chunk.score, label=f"Chunk {i} similarity"),
            unsafe_allow_html=True,
        )
        with st.expander(
            f"Chunk {i} — dataset: {chunk.metadata.get('dataset', '?')} | {chunk.chunk_id}"
        ):
            st.markdown(chunk.text)

    st.divider()
else:
    st.info("Run a query above to see the generated answer, evaluation scores, and retrieved chunks.")
    st.divider()

# ── Session latency history ───────────────────────────────────────────────────
st.subheader("Pipeline Latency Over Time")
st.caption(
    "Latency for queries run in **this dashboard session** only — not the full "
    "telemetry store, which also contains past benchmark experiment runs."
)
if st.session_state.session_history:
    hist_df = pd.DataFrame(st.session_state.session_history)
    st.caption(
        f"{len(hist_df)} quer{'y' if len(hist_df) == 1 else 'ies'} this session · "
        f"mean end-to-end latency {hist_df['e2e_ms'].mean():.0f} ms"
    )
latency_time_series(st.session_state.session_history)
