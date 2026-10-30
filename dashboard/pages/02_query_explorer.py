"""
RAGScope Dashboard — Page 2: Query Explorer
=============================================
Inspect individual telemetry records in detail:
retrieved chunks, scores, latency decomposition, and answers.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

st.title(":material/manage_search: Query Explorer")
st.caption("Select any logged query to inspect its full telemetry trace.")

from dashboard.components.graph_view import render_retrieval_graph
from dashboard.components.latency_chart import pipeline_flow_sankey
from dashboard.components.metric_cards import ragas_scorecard, telemetry_row
from dashboard.components.passages import answer_html, render_passages
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
    f'<div style="background:rgba(82,45,128,0.08);padding:1rem;'
    f'border-left:4px solid #522D80;border-radius:4px;">{answer_html(record.get("answer"))}</div>',
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
    record.get("evaluation_ms", 0),
    record.get("e2e_ms", 0),
    record.get("total_tokens", 0),
    record.get("estimated_cost_usd", 0.0),
)
st.divider()

# ── Pipeline flow ─────────────────────────────────────────────────────────────
st.subheader("Pipeline Flow")
st.caption(
    "Where this query's time went, read left to right: the total, its phases, "
    "then retrieval's own steps. Band width is time taken; hover any band for "
    "exact figures. Very small stages are drawn slightly wider than their "
    "true share so they stay visible; the numbers on each label are exact. "
    "'Other overhead' is time not attributed to a named stage (token "
    "counting, writing the telemetry record)."
)
retrieval_telemetry = {
    "strategy": record.get("retrieval_strategy", "dense"),
    "embed_query_ms": record.get("embed_query_ms", 0.0),
    "vector_search_ms": record.get("vector_search_ms", 0.0),
    "dense_search_ms": record.get("dense_search_ms", 0.0),
    "bm25_search_ms": record.get("bm25_search_ms", 0.0),
    "rrf_fusion_ms": record.get("rrf_fusion_ms", 0.0),
    "graph_seed_ms": record.get("graph_seed_ms", 0.0),
    "graph_traverse_ms": record.get("graph_traverse_ms", 0.0),
    "graph_score_ms": record.get("graph_score_ms", 0.0),
}
pipeline_flow_sankey(
    retrieval_telemetry,
    record.get("generation_ms", 0.0),
    record.get("evaluation_ms", 0.0),
    record.get("e2e_ms", 0.0),
    ran_evaluation=record.get("context_relevance") is not None,
)
if record.get("evaluation_ms", 0.0) == 0.0 and record.get("context_relevance") is not None:
    st.caption(
        ":material/warning: This record predates per-stage evaluation timing — RAGAS ran (scores are "
        "present) but its duration wasn't captured separately, so 'Other overhead' absorbs it."
    )
st.divider()

# ── Graph retrieval path ──────────────────────────────────────────────────────
# Redraws the walk from the persisted record, so a past graph query can be
# inspected exactly as it ran, not only at the moment it was submitted.
if record.get("retrieval_strategy") == "graph":
    st.subheader(":material/hub: Graph Retrieval Path")
    if record.get("graph_path"):
        render_retrieval_graph(
            record["graph_path"],
            chunk_texts={
                c.get("chunk_id", ""): c.get("text", "")
                for c in record.get("retrieved_chunks", [])
            },
            key=f"qe_graph_{record.get('record_id', selected_label)}",
        )
    elif record.get("graph_seeds_found", 0) == 0:
        st.info("No entity in this query matched the knowledge graph, so there was no walk to draw.")
    else:
        st.info(
            "This record predates path capture: the graph was traversed, but the "
            "path itself was not persisted. Re-run the query in Live Monitor to "
            "record one."
        )
    st.divider()

# ── Retrieved passages ────────────────────────────────────────────────────────


def _graph_details(passage: dict) -> None:
    """Why graph retrieval returned this passage, which a score cannot express."""
    info = passage["graph"]
    origin = "graph only" if info.get("from_graph_leg") else "graph and dense retrieval"
    st.caption(f"Graph rank {info.get('graph_rank', '?')}, found by {origin}.")
    via = info.get("via", []) or []
    if via:
        st.caption("Reached via these graph entities:")
        st.dataframe(
            [{"Entity": v.get("node", ""), "PageRank mass": v.get("mass", 0.0)} for v in via],
            hide_index=True, width="stretch",
        )
    else:
        st.caption("Reached via the document node rather than a named entity.")


render_passages(
    record.get("retrieved_chunks", []),
    answer=record.get("answer"),
    strategy=record.get("retrieval_strategy", "dense"),
    excerpts=True,
    graph_details=_graph_details,
)

# ── Raw JSON ──────────────────────────────────────────────────────────────────
with st.expander("Raw Telemetry Record (JSON)", icon=":material/data_object:"):
    display = {k: v for k, v in record.items() if k != "retrieved_chunks"}
    st.json(display)
