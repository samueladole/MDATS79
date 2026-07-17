"""
RAGScope Dashboard — Page 5: Knowledge Base
===============================================
Read-only view into the ChromaDB collection backing the RAG pipeline:
collection stats, per-dataset composition, a metadata-filterable chunk
browser, and a semantic search preview against the live index.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import settings
from dashboard.components.theme import apply_theme
from pipeline.vectorstore import get_vector_store

st.set_page_config(page_title="Knowledge Base · RAGScope", layout="wide", page_icon=":material/database:")
apply_theme()
st.title(":material/database: Knowledge Base")
st.caption("Read-only view of the ChromaDB collection backing the RAG pipeline.")

DATASET_LABELS = {
    "msmarco": "MS MARCO",
    "natural_questions": "Natural Questions",
    "hotpotqa": "HotpotQA",
}
DATASET_COLORS = {
    "MS MARCO": "#2a78d6",
    "Natural Questions": "#008300",
    "HotpotQA": "#e87ba4",
    "Other / Unlabelled": "#95a5a6",
}

try:
    vector_store = get_vector_store()
    total_chunks = vector_store.count()
except Exception as exc:
    st.error(
        f"Could not connect to ChromaDB at `{settings.chroma_host}:{settings.chroma_port}`.\n\n"
        f"**Error:** {exc}"
    )
    st.stop()

# ── Overview ───────────────────────────────────────────────────────────────────
st.subheader("Collection Overview")

if total_chunks == 0:
    st.info(
        "The collection is empty — no chunks have been ingested yet. Run:\n\n"
        "```bash\n"
        "uv run python pipeline/ingestion.py --corpus all --chunk-size 512 --chunk-overlap 64\n"
        "```"
    )
    st.stop()

embedding_dim = vector_store.embedding_dimension()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Chunks", f"{total_chunks:,}")
c2.metric("Collection", vector_store.collection_name)
c3.metric("Embedding Dim", embedding_dim if embedding_dim is not None else "—")
c4.metric("Chroma Endpoint", f"{settings.chroma_host}:{settings.chroma_port}")
st.caption(f"Embedding model (configured): `{settings.embedding_model}`")
st.divider()

# ── Corpus composition ────────────────────────────────────────────────────────
st.subheader("Corpus Composition")
st.caption("How the indexed chunks split across the three source benchmark datasets.")

dataset_counts = {ds_key: vector_store.count_where({"dataset": ds_key}) for ds_key in DATASET_LABELS}
other_count = total_chunks - sum(dataset_counts.values())

comp_rows = [
    {"dataset": DATASET_LABELS[k], "chunks": v} for k, v in dataset_counts.items() if v > 0
]
if other_count > 0:
    comp_rows.append({"dataset": "Other / Unlabelled", "chunks": other_count})
comp_df = pd.DataFrame(comp_rows)
comp_df["share_pct"] = (comp_df["chunks"] / total_chunks * 100).round(1)

col_chart, col_table = st.columns([2, 1])
with col_chart:
    fig = px.bar(
        comp_df,
        x="dataset",
        y="chunks",
        color="dataset",
        color_discrete_map=DATASET_COLORS,
        labels={"dataset": "Dataset", "chunks": "Chunk Count"},
        title="Chunks per Source Dataset",
    )
    fig.update_layout(
        height=360,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")
with col_table:
    st.dataframe(comp_df, width="stretch", hide_index=True)
st.divider()

# ── Chunk browser ──────────────────────────────────────────────────────────────
st.subheader("Chunk Browser")
st.caption(
    "A direct browse of stored chunks by metadata filter — rows are in storage order, "
    "not ranked by similarity. Use Semantic Search Preview below to see similarity-ranked results."
)

available_labels = ["All"] + [DATASET_LABELS[k] for k, v in dataset_counts.items() if v > 0]
reverse_label_map = {v: k for k, v in DATASET_LABELS.items()}

filter_col, size_col, page_col = st.columns([2, 1, 1])
with filter_col:
    dataset_filter = st.selectbox("Filter by dataset", options=available_labels)
with size_col:
    page_size = st.selectbox("Rows per page", options=[10, 25, 50, 100], index=1)

where_clause = {"dataset": reverse_label_map[dataset_filter]} if dataset_filter != "All" else None
filtered_count = dataset_counts[reverse_label_map[dataset_filter]] if where_clause else total_chunks
max_page = max(0, (filtered_count - 1) // page_size)

with page_col:
    page_num = st.number_input("Page", min_value=0, max_value=max_page, value=0, step=1)

page_chunks = vector_store.get_chunks(where=where_clause, limit=page_size, offset=page_num * page_size)

if page_chunks:
    browser_rows = [
        {
            "chunk_id": c.chunk_id,
            "dataset": DATASET_LABELS.get(c.metadata.get("dataset"), c.metadata.get("dataset") or "—"),
            "title": c.metadata.get("title") or "—",
            "text_preview": (c.text[:200] + "…") if c.text and len(c.text) > 200 else (c.text or ""),
        }
        for c in page_chunks
    ]
    st.dataframe(pd.DataFrame(browser_rows), width="stretch", hide_index=True)
    st.caption(f"Showing {len(browser_rows)} of {filtered_count:,} chunks (page {page_num + 1} of {max_page + 1}).")
else:
    st.info("No chunks match this filter.")
st.divider()

# ── Semantic search preview ───────────────────────────────────────────────────
st.subheader("Semantic Search Preview")
st.caption(
    "Embed a query and retrieve the nearest chunks — a quick way to spot-check "
    "what the pipeline would actually retrieve for a given question."
)

search_query = st.text_input("Query", placeholder="e.g. What are the main causes of climate change?")
search_top_k = st.slider("Top-K", min_value=1, max_value=20, value=5)

if search_query:
    from pipeline.embeddings import get_embedding_generator

    with st.spinner("Embedding query and searching…"):
        query_vector = get_embedding_generator().embed_query(search_query)
        results = vector_store.query(query_vector, top_k=search_top_k)

    if results:
        result_rows = [
            {
                "rank": i + 1,
                "similarity": r.score,
                "dataset": DATASET_LABELS.get(r.metadata.get("dataset"), r.metadata.get("dataset") or "—"),
                "chunk_id": r.chunk_id,
                "text_preview": (r.text[:200] + "…") if len(r.text) > 200 else r.text,
            }
            for i, r in enumerate(results)
        ]
        st.dataframe(pd.DataFrame(result_rows), width="stretch", hide_index=True)
    else:
        st.info("No results.")
