"""
RAGScope Dashboard — Page 1: Live Monitor
===========================================
Allows the user to submit queries interactively and observe
real-time RAGAS scores, telemetry, and retrieved chunks.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from dashboard.components.graph_view import render_retrieval_graph
from dashboard.components.latency_chart import latency_time_series, pipeline_flow_sankey
from dashboard.components.metric_cards import (
    ragas_scorecard,
    similarity_meter_html,
    telemetry_row,
)
st.title(":material/monitor_heart: Live Monitor")
st.caption("Submit a query and watch the full RAG pipeline execute in real time.")

st.session_state.setdefault("session_history", [])
st.session_state.setdefault("last_result", None)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Pipeline Configuration")
    llm_choice = st.selectbox("LLM Model", ["llama3", "mistral"], index=0)
    retrieval_choice = st.selectbox(
        "Retrieval Strategy", ["hybrid", "dense", "graph"], index=0,
        help="graph: seeds entities from the query, traverses the knowledge "
             "graph, and fuses the result with dense retrieval.",
    )
    query_type_choice = st.selectbox(
        "Query Type",
        ["factoid", "single_relevant", "multi_relevant", "yesno", "summary", "unknown"],
        index=0,
        help="Label recorded with this query's telemetry. Matches the BioASQ role "
        "taxonomy used by the benchmark loader (data/loaders/bioasq.py) — pick "
        "'unknown' if none apply.",
    )
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
        if st.button("Clear Session History", icon=":material/delete_sweep:", width="stretch"):
            st.session_state.session_history = []
            st.session_state.last_result = None
            st.rerun()

# ── Query form ────────────────────────────────────────────────────────────────
query_text = st.text_area(
    "Enter your question",
    height=100,
    placeholder="e.g. What are the main causes of hallucination in large language models?",
    key="query_text",
)

# The Run Query button is disabled for as long as a pipeline run is in
# progress, to prevent accidental double-submission. A run takes 35-60 s,
# almost all of it the RAGAS judge, which is ample time to click twice.
#
# ``pipeline_running`` means exactly "a run is in progress": the click sets it,
# and it is cleared when the run ends, however it ends. It must not be derived
# from whether a previous result exists — that leaves the first query
# unprotected and locks the button permanently after it.
st.session_state.setdefault("pipeline_running", False)


def _start_run() -> None:
    """Arm a run. on_click callbacks execute before the rerun the click
    triggers, so the button below is already rendered disabled by the time the
    pipeline starts. An empty query is not armed, so it cannot leave the
    button disabled."""
    if st.session_state.get("query_text", "").strip():
        st.session_state.pipeline_running = True


run_btn = st.button(
    "Run Query",
    icon=":material/play_arrow:",
    type="primary",
    width="stretch",
    key="run_query_btn",
    on_click=_start_run,
    disabled=st.session_state.pipeline_running,
)

if st.session_state.pipeline_running:
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
                query_type=query_type_choice,
            )
        st.session_state.last_result = {
            "result": result,
            "query_text": query_text,
            "llm_choice": llm_choice,
            "retrieval_choice": retrieval_choice,
            "query_type_choice": query_type_choice,
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
        st.session_state.pipeline_notice = "Query complete."
    except Exception as exc:
        st.session_state.pipeline_error = str(exc)
    finally:
        # Re-enable on every exit path: success, pipeline error, or a run
        # interrupted by Streamlit (a rerun or stop raised mid-script).
        st.session_state.pipeline_running = False
    # Rerun so the button redraws enabled. Outside the try: st.rerun() works by
    # raising, and must not be swallowed by the except above.
    st.rerun()
elif run_btn:
    st.warning("Please enter a query before running.")

# One-shot messages from the run that just finished, shown after the rerun.
if notice := st.session_state.pop("pipeline_notice", None):
    st.success(notice)
if error := st.session_state.pop("pipeline_error", None):
    st.error(f"Pipeline error: {error}")
    # As before: on failure, render nothing below the error, rather than the
    # previous query's result appearing as though it were this one's.
    st.stop()


# ── Graph retrieval panel ─────────────────────────────────────────────────────


def _render_graph_panel(telemetry: dict, chunks: list) -> None:
    """
    Show why the graph retriever returned what it did.

    This is the distinguishing affordance of graph retrieval: dense retrieval
    yields a similarity score and no account of itself, whereas this yields an
    inspectable path — which entities the query matched, how far traversal
    reached, and which of the returned chunks the graph found rather than the
    dense leg.
    """
    st.divider()
    st.subheader(":material/hub: Graph Retrieval Path")

    seeds = telemetry.get("graph_seed_labels", []) or []
    from_graph = telemetry.get("graph_chunks_from_graph_leg", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Seed entities", telemetry.get("graph_seeds_found", 0))
    c2.metric("Subgraph nodes", telemetry.get("graph_subgraph_nodes", 0))
    c3.metric("Subgraph edges", telemetry.get("graph_subgraph_edges", 0))
    c4.metric(
        "Chunks from graph", from_graph,
        help="Results the graph surfaced that dense retrieval had not already "
             "found. Zero means the graph contributed nothing for this query.",
    )

    if not seeds:
        st.warning(
            "No entities in this query matched the graph, so retrieval fell back "
            "entirely to the dense leg. Expected while the graph covers only part "
            "of the corpus — the entity simply isn't in it yet."
        )
        return

    graph_path = telemetry.get("graph_path")
    # Keyed on the specific result so a new query never inherits the previous
    # query's highlight or selection state.
    fingerprint = hashlib.md5(
        json.dumps(graph_path or {}, sort_keys=True, default=str).encode()
    ).hexdigest()[:10]
    render_retrieval_graph(
        graph_path,
        chunk_texts={c.chunk_id: c.text for c in chunks},
        key=f"live_graph_{fingerprint}",
    )

    with st.expander("Seed entity matches", icon=":material/target:"):
        st.dataframe(
            pd.DataFrame([
                {"Entity": s.get("label", ""),
                 "Matched by": s.get("matcher", ""),
                 "Score": round(float(s.get("score", 0.0)), 4)}
                for s in seeds
            ]),
            hide_index=True, width="stretch",
        )

    if from_graph == 0 and telemetry.get("fuse_dense", True):
        st.info(
            "No result here was found *only* by the graph. In fused mode the dense "
            "leg searches the whole corpus while the graph covers a small subset, "
            "and Reciprocal Rank Fusion cannot promote a graph-only chunk above "
            "dense's top results at current coverage. Set `GRAPH_FUSE_DENSE=false` "
            "to see graph-only retrieval."
        )


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
        f"· query_type={data.get('query_type_choice', 'unknown')} "
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
        result.evaluation_ms,
        result.e2e_ms,
        result.token_usage.total_tokens,
        result.token_usage.estimated_cost_usd,
    )
    st.divider()

    # ── Pipeline flow ─────────────────────────────────────────────────────────
    st.subheader("Pipeline Flow")
    st.caption(
        "Every instrumented stage this query passed through link width is that "
        "stage's actual wall-clock duration. 'Response' absorbs any small residual "
        "(token counting, telemetry write) not attributed to a named stage."
    )
    pipeline_flow_sankey(
        result.retrieval_telemetry,
        result.generation_response.generation_ms,
        result.evaluation_ms,
        result.e2e_ms,
        ran_evaluation=data["run_eval"],
    )
    if result.retrieval_telemetry.get("strategy") == "graph":
        _render_graph_panel(result.retrieval_telemetry, result.retrieved_chunks)
    if result.retrieval_telemetry.get("strategy") == "hybrid":
        st.caption(
            "Note: BM25 search doesn't depend on the query embedding, but the current "
            "implementation runs retrieval sub-stages sequentially rather than in "
            "parallel — this diagram reflects actual measured execution order."
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
