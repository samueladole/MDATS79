"""
RAGScope Dashboard — Page 3: Query Explorer
=============================================
Inspect individual telemetry records in detail:
retrieved chunks, scores, latency decomposition, and answers.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Query Explorer · RAGScope", layout="wide", page_icon="🔍")
st.title("🔍 Query Explorer")
st.caption("Select any logged query to inspect its full telemetry trace.")

from dashboard.components.metric_cards import (
    ragas_scorecard,
    similarity_meter_html,
    telemetry_row,
)
from telemetry.logger import get_telemetry_logger

records = get_telemetry_logger().load_all()

if not records:
    st.info("No telemetry records yet. Run queries on the Live Monitor page first.")
    st.stop()

# ── Query selector ────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df["label"] = df["timestamp_utc"].str[:19] + " | " + df["query"].str[:60]

selected_label = st.selectbox(
    "Select a query",
    options=df["label"].tolist(),
    index=0,
)

record = records[df["label"].tolist().index(selected_label)]

st.divider()

# ── Metadata row ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"**Dataset:** `{record.get('dataset', '?')}`")
col2.markdown(f"**Query type:** `{record.get('query_type', '?')}`")
col3.markdown(f"**LLM:** `{record.get('llm_model', '?')}`")
col4.markdown(f"**Retrieval:** `{record.get('retrieval_strategy', '?')}`")

st.divider()

# ── Query & Answer ────────────────────────────────────────────────────────────
st.subheader("Query")
st.markdown(f"> {record.get('query', '')}")

if record.get("ground_truth"):
    st.subheader("Ground Truth")
    st.markdown(f"> {record['ground_truth']}")

st.subheader("Generated Answer")
st.markdown(
    f'<div style="background:#f8f9fa;padding:1rem;border-left:4px solid #522D80;'
    f'border-radius:4px;">{record.get("answer", "")}</div>',
    unsafe_allow_html=True,
)
st.divider()

# ── RAGAS Scores ──────────────────────────────────────────────────────────────
st.subheader("Evaluation Scores")
ragas_scorecard(
    record.get("context_relevance"),
    record.get("answer_faithfulness"),
    record.get("answer_correctness"),
    record.get("hallucination_risk"),
)
st.divider()

# ── Telemetry ─────────────────────────────────────────────────────────────────
st.subheader("Telemetry")
telemetry_row(
    record.get("retrieval_ms", 0),
    record.get("generation_ms", 0),
    record.get("e2e_ms", 0),
    record.get("total_tokens", 0),
    record.get("estimated_cost_usd", 0.0),
)
st.divider()

# ── Retrieved Chunks ──────────────────────────────────────────────────────────
chunks = record.get("retrieved_chunks", [])
st.subheader(f"Retrieved Chunks ({len(chunks)})")
for i, chunk in enumerate(chunks, 1):
    score = chunk.get("score", 0.0)
    st.markdown(
        similarity_meter_html(score, label=f"Chunk {i} similarity"),
        unsafe_allow_html=True,
    )
    with st.expander(f"Chunk {i} — dataset: {chunk.get('dataset', '?')} | {chunk.get('chunk_id', '')}"):
        st.markdown(chunk.get("text", ""))

# ── Raw JSON ──────────────────────────────────────────────────────────────────
with st.expander("🗂 Raw Telemetry Record (JSON)"):
    display = {k: v for k, v in record.items() if k != "retrieved_chunks"}
    st.json(display)
