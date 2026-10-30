"""
RAGScope Dashboard — Page 5: Knowledge Base
===============================================
View of the ChromaDB collection backing the RAG pipeline: collection stats,
per-dataset composition, a metadata-filterable chunk browser, a semantic
search preview against the live index, and the knowledge graph built over the
corpus for graph retrieval.

The "My documents" section at the top is the one place that writes: it saves
uploaded PDF and text files and indexes them in their own collection, never
in the BioASQ benchmark collection shown below it.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import settings
from dashboard.components.vector_space import vector_space_scatter
from pipeline.vectorstore import get_vector_store

st.title(":material/database: Knowledge Base")
st.caption("The ChromaDB collections behind the RAG pipeline.")


def _documents_section() -> None:
    """Upload and index your own files, in a collection of their own."""
    from pipeline.ingestion import ingest_corpus
    from pipeline.knowledge_base import get_knowledge_base

    kb = get_knowledge_base("documents")
    folder = Path(settings.documents_dir)
    files = sorted(
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in {".pdf", ".txt", ".md"}
    ) if folder.is_dir() else []

    with st.expander(
        f"My documents ({len(files)} file{'s' if len(files) != 1 else ''})",
        icon=":material/upload_file:", expanded=not files,
    ):
        st.caption(
            f"Add PDF, .txt or .md files to ask questions about them. They are saved "
            f"to `{folder}` and indexed in their own collection (`{kb.collection}`), "
            f"so the BioASQ benchmark below is never changed. Query them on the "
            f"Live Monitor page under **Knowledge base → My documents**. PDFs need a "
            f"text layer: scanned pages are skipped."
        )
        uploads = st.file_uploader(
            "Add files", type=["pdf", "txt", "md"], accept_multiple_files=True,
            key="documents_upload",
        )
        if st.button("Save and index", icon=":material/library_add:", type="primary",
                     disabled=not (uploads or files)):
            folder.mkdir(parents=True, exist_ok=True)
            for upload in uploads or []:
                # Keep only the base name, so an upload cannot write outside the folder.
                (folder / Path(upload.name).name).write_bytes(upload.getbuffer())
            try:
                with st.spinner("Reading and indexing documents …"):
                    summary = ingest_corpus("documents", source=folder)
                st.success(
                    f"Indexed {summary.total_passages:,} pages and files as "
                    f"{summary.total_chunks:,} chunks."
                )
            except ValueError as exc:
                st.warning(str(exc))
            st.rerun()
        if files:
            st.dataframe(
                pd.DataFrame([
                    {"File": f.relative_to(folder).as_posix(),
                     "Size (KB)": round(f.stat().st_size / 1024, 1)}
                    for f in files
                ]),
                hide_index=True, width="stretch",
            )
            st.caption(
                "To remove a file, delete it from the folder and press **Save and "
                "index** again: the collection is rebuilt from the folder each time."
            )


_documents_section()
st.subheader("BioASQ Benchmark Collection")

DATASET_LABELS = {
    "bioasq": "BioASQ",
}
DATASET_COLORS = {
    "BioASQ": "#2a78d6",
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
        "uv run python pipeline/ingestion.py --corpus bioasq --chunk-size 512 --chunk-overlap 64\n"
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
st.caption("Indexed chunks by source dataset — a single BioASQ corpus, so this is a sanity-check total rather than a breakdown.")

dataset_counts = {ds_key: vector_store.count_where({"dataset": ds_key}) for ds_key in DATASET_LABELS}
other_count = total_chunks - sum(dataset_counts.values())

comp_rows = [
    {"dataset": DATASET_LABELS[k], "chunks": v} for k, v in dataset_counts.items() if v > 0
]
if other_count > 0:
    comp_rows.append({"dataset": "Other / Unlabelled", "chunks": other_count})
comp_df = pd.DataFrame(comp_rows)
comp_df["share_percentage"] = (comp_df["chunks"] / total_chunks * 100).round(1)

col_chart, col_table = st.columns([1, 1])
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
        showlegend=True,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")
with col_table:
    st.write(f"**Chunk counts and share per dataset**")
    st.dataframe(comp_df, width="stretch", hide_index=True)
st.divider()

# ── 3D embedding space ───────────────────────────────────────────────────────
st.subheader("3D Embedding Space")
st.caption(
    "A PCA projection of sampled embeddings down to 3 dimensions a coarse view of "
    "how the indexed corpus clusters in vector space, coloured by source dataset."
)

sample_per_dataset = st.slider(
    "Points to sample per dataset (capped by how many exist)",
    min_value=50,
    max_value=1000,
    value=300,
    step=50,
)

if st.button("Compute projection", icon=":material/scatter_plot:"):
    with st.spinner("Sampling embeddings and computing the PCA projection…"):
        sampled_chunks = []
        for ds_key, ds_count in dataset_counts.items():
            if ds_count == 0:
                continue
            sampled_chunks += vector_store.sample_embeddings(
                where={"dataset": ds_key}, sample_size=sample_per_dataset
            )
        st.session_state["vector_space_sample"] = sampled_chunks

sampled_chunks = st.session_state.get("vector_space_sample")
if sampled_chunks:
    vector_space_scatter(sampled_chunks, DATASET_LABELS, DATASET_COLORS)
else:
    st.info("Click **Compute projection** to sample embeddings and render the 3D view.")
st.divider()

# ── Chunk browser ──────────────────────────────────────────────────────────────
st.subheader("Chunk Browser")
st.caption(
    "A direct browse of stored chunks by metadata filter rows are in storage order, "
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
    "Embed a query and retrieve the nearest chunks a quick way to spot-check "
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


# ── Knowledge graph ───────────────────────────────────────────────────────────
st.divider()
st.subheader("Knowledge Graph")
st.caption(
    "Entities and relations extracted from the corpus, used by the `graph` "
    "retrieval strategy to reach chunks through entity provenance rather than "
    "vector similarity."
)


@st.cache_data(show_spinner=False)
def _load_graph_summary(graph_path: str, nc_path: str, manifest_path: str) -> dict | None:
    """Summarise the graph artefacts. Returns None if they haven't been built."""
    gp, ncp = Path(graph_path), Path(nc_path)
    if not gp.exists() or not ncp.exists():
        return None
    data = json.loads(gp.read_text())
    node_chunks = json.loads(ncp.read_text())
    manifest = {}
    mp = Path(manifest_path)
    if mp.exists():
        manifest = json.loads(mp.read_text())

    nodes = data.get("nodes", [])
    links = data.get("edges", data.get("links", []))
    is_doc = lambda n: str(n.get("id", "")).startswith("DOC::")  # noqa: E731
    classes = Counter(
        n.get("__class__", "?") for n in nodes if not is_doc(n)
    )
    labels = Counter(e.get("label", "?") for e in links)
    ent_ids = {n["id"] for n in nodes if not is_doc(n)}
    ent_ent = sum(
        1 for e in links
        if e.get("source") in ent_ids and e.get("target") in ent_ids
    )
    covered = {c for v in node_chunks.values() for c in v}
    return {
        "nodes": len(nodes),
        "edges": len(links),
        "entities": len(ent_ids),
        "documents": len(nodes) - len(ent_ids),
        "classes": dict(classes.most_common()),
        "edge_labels": dict(labels.most_common()),
        "entity_entity_edges": ent_ent,
        "chunks_covered": len(covered),
        "manifest": manifest,
    }


summary = _load_graph_summary(
    str(settings.graph_path), str(settings.graph_node_chunks_path),
    str(Path(settings.graph_path).parent / "manifest.json"),
)

if summary is None:
    st.info(
        f"No knowledge graph found at `{settings.graph_path}`. "
        "Build one with `graph/build_graph.py` to enable the `graph` "
        "retrieval strategy."
    )
else:
    coverage = 100 * summary["chunks_covered"] / total_chunks if total_chunks else 0.0

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Entities", f"{summary['entities']:,}")
    g2.metric("Relations", f"{summary['edges']:,}")
    g3.metric("Chunks covered", f"{summary['chunks_covered']:,}")
    g4.metric(
        "Corpus coverage", f"{coverage:.1f}%",
        help="Share of the indexed corpus the graph spans. The graph retrieval "
             "strategy can only reach chunks inside this slice.",
    )

    # Coverage is the single most important caveat on anything the graph does,
    # so it is stated here rather than left for a reader to infer.
    if coverage < 50:
        st.warning(
            f"**The graph covers {coverage:.1f}% of the corpus** "
            f"({summary['chunks_covered']:,} of {total_chunks:,} chunks). "
            "Dense retrieval searches the whole corpus, so in the fused `graph` "
            "strategy the dense leg will dominate: a chunk found only by the "
            "graph cannot currently displace dense's top results. Treat graph "
            "retrieval as a demonstration of the mechanism, not a quality "
            "comparison, until coverage matches."
        )

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**Entity types**")
        if summary["classes"]:
            df_cls = pd.DataFrame(
                [{"Type": k, "Count": v} for k, v in summary["classes"].items()]
            )
            st.plotly_chart(
                px.bar(df_cls, x="Count", y="Type", orientation="h",
                       color_discrete_sequence=["#2a78d6"]).update_layout(
                    height=240, margin=dict(l=0, r=0, t=10, b=0), showlegend=False),
                width="stretch",
            )
    with c_right:
        st.markdown("**Relation types**")
        if summary["edge_labels"]:
            df_lbl = pd.DataFrame(
                [{"Relation": k, "Count": v} for k, v in summary["edge_labels"].items()]
            )
            st.plotly_chart(
                px.bar(df_lbl, x="Count", y="Relation", orientation="h",
                       color_discrete_sequence=["#24a35a"]).update_layout(
                    height=240, margin=dict(l=0, r=0, t=10, b=0), showlegend=False),
                width="stretch",
            )

    # Entity-entity edges are the relational signal; edges to document nodes
    # only encode co-occurrence, which dense retrieval already approximates.
    ent_ent = summary["entity_entity_edges"]
    st.caption(
        f"Of {summary['edges']:,} relations, **{ent_ent:,} connect two entities** "
        f"(true relational structure); the remaining "
        f"{summary['edges'] - ent_ent:,} connect an entity to its source document "
        f"(co-occurrence). Entities: {summary['entities']:,} across "
        f"{summary['documents']:,} document nodes."
    )

    man = summary.get("manifest") or {}
    if man:
        with st.expander("Build provenance", icon=":material/history:"):
            st.json({
                "extractor": man.get("extractor"),
                "chunks_attempted": man.get("chunks_attempted"),
                "chunks_ok": man.get("chunks_ok"),
                "failures": len(man.get("failures", []) or []),
                "s_per_chunk": man.get("s_per_chunk"),
                "dedup_ratio": man.get("dedup_ratio"),
                "pct_nodes_cross_chunk": man.get("pct_nodes_cross_chunk"),
            })

    # ── Entity browser ────────────────────────────────────────────────────────
    st.markdown("**Entity Browser**")
    term = st.text_input(
        "Filter entities", placeholder="e.g. aspirin, kinase, diabetes",
        key="graph_entity_filter",
    )
    node_chunks = json.loads(Path(settings.graph_node_chunks_path).read_text())
    graph_raw = json.loads(Path(settings.graph_path).read_text())
    rows = []
    for n in graph_raw.get("nodes", []):
        nid = str(n.get("id", ""))
        if nid.startswith("DOC::"):
            continue
        name = n.get("name") or n.get("label") or ""
        if term and term.lower() not in str(name).lower():
            continue
        rows.append({
            "Entity": name,
            "Type": n.get("__class__", "?"),
            "Chunks": len(node_chunks.get(nid, [])),
        })
    rows.sort(key=lambda r: (-r["Chunks"], str(r["Entity"])))
    st.caption(f"{len(rows):,} entities" + (f" matching '{term}'" if term else ""))
    st.dataframe(pd.DataFrame(rows[:200]), hide_index=True, width="stretch")
