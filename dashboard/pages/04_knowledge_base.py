"""
RAGScope Dashboard — Page 4: Knowledge Base
===============================================
The data behind the RAG pipeline, in three tabs:

* **Chunks** — browse any ChromaDB collection, or search it by meaning, and
  open a chunk to read it with its metadata.
* **Upload documents** — add your own PDF and text files. They are indexed in
  their own collection, never in the BioASQ benchmark collection, and a
  knowledge graph is then built over them in the background.
* **Knowledge graph** — for either knowledge base, explore an entity's
  neighbourhood, or trace how graph retrieval walks the graph for a question,
  both as left-to-right DAGs.

A collection picker at the top selects which ChromaDB collection the Chunks
tab shows; every collection on the server is listed, not only BioASQ.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import settings
from dashboard.components.formatting import format_duration
from dashboard.components.graph_view import (
    DOC_PREFIX,
    TYPE_NAMES,
    entity_neighbourhood,
    render_entity_graph,
    render_retrieval_graph,
)
from pipeline.vectorstore import get_vector_store

st.title(":material/database: Knowledge Base")
st.caption("The ChromaDB collections and the knowledge graph behind the RAG pipeline.")

FRIENDLY = {
    settings.chroma_collection: "BioASQ benchmark",
    settings.documents_collection: "My documents",
}


GRAPH_REBUILD_HELP = """Build or rebuild the BioASQ graph from the command line (it needs Ollama
and `uv sync --extra graph`):

```bash
uv run python graph/build_graph.py            # the 300-chunk sample: about 1 hour
uv run python graph/build_graph.py --all      # whole corpus: about 100 hours, resumable
uv run python graph/build_graph.py --publish-only   # restore from saved extractions
```

A whole-corpus build saves every chunk as it goes: stop it with Ctrl-C and run
the same command again to resume. The graph is published as it grows, so this
page and graph retrieval work throughout. See the README, *Building the
Knowledge Graph*."""

DOCUMENTS_GRAPH_HELP = """Your documents' graph is built in the background each time you press
**Save and index**, with the same extractor and schema as the BioASQ graph
(docling-graph, `llama3.2`). Only new chunks are extracted, so adding a file
costs only that file's chunks. Expect 12-20 seconds per chunk. Command-line
equivalent:

```bash
uv run python graph/build_graph.py --kb documents
```"""


# ── Data access (cached) ──────────────────────────────────────────────────────


@st.cache_data(ttl=30, show_spinner=False)
def _collections() -> list[dict]:
    return get_vector_store().list_collections()


@st.cache_data(ttl=300, show_spinner=False)
def _metadata_values(collection: str, key: str, count: int) -> dict[str, int]:
    # ``count`` is part of the cache key, so re-ingesting refreshes the values.
    return get_vector_store(collection).metadata_values(key)


@st.cache_data(ttl=300, show_spinner=False)
def _embedding_dim(collection: str, count: int) -> int | None:
    return get_vector_store(collection).embedding_dimension()


@st.cache_resource(show_spinner=False)
def _graph(graph_path: str, nc_path: str, stamp: tuple):
    """Load the knowledge graph once per file version."""
    import networkx as nx

    digraph = nx.node_link_graph(json.loads(Path(graph_path).read_text()), edges="edges")
    node_chunks = json.loads(Path(nc_path).read_text())
    entities = [n for n in digraph.nodes if not str(n).startswith(DOC_PREFIX)]
    undirected = digraph.to_undirected(as_view=True)
    relations = {
        n: sum(1 for m in undirected.neighbors(n) if not str(m).startswith(DOC_PREFIX))
        for n in entities
    }
    rows = [{
        "id": n,
        "Entity": str(digraph.nodes[n].get("name") or digraph.nodes[n].get("label") or n),
        "Type": TYPE_NAMES.get(str(digraph.nodes[n].get("__class__", "")),
                               str(digraph.nodes[n].get("__class__", "Entity"))),
        "Relations": relations[n],
        "Chunks": len(node_chunks.get(n, [])),
    } for n in entities]
    rows.sort(key=lambda r: (-r["Relations"], -r["Chunks"], r["Entity"].lower()))
    return digraph, node_chunks, pd.DataFrame(rows)


def _store_label() -> str:
    if settings.chroma_mode == "embedded":
        return f"embedded · {settings.chroma_path}"
    return f"server · {settings.chroma_host}:{settings.chroma_port}"


def _source_of(meta: dict) -> str:
    return str(meta.get("source") or meta.get("title") or meta.get("dataset") or "—")


def _preview(text: str, n: int = 180) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


# ── Collection picker ─────────────────────────────────────────────────────────

try:
    collections = _collections()
except Exception as exc:  # the server is down or misconfigured
    st.error(f"Could not reach ChromaDB ({_store_label()}).\n\n**Error:** {exc}")
    st.stop()

if not collections:
    st.info(
        "ChromaDB has no collections yet. Fetch the prebuilt index with "
        "`uv run python scripts/fetch_index.py`, or build it with "
        "`uv run python pipeline/ingestion.py --corpus all`. Your own files can be "
        "added on the **Upload documents** tab below."
    )

names = [c["name"] for c in collections]
by_name = {c["name"]: c for c in collections}
default = names.index(settings.chroma_collection) if settings.chroma_collection in names else 0

if names:
    pick_col, *metric_cols = st.columns([2.2, 1, 1, 1, 1.6])
    selected = pick_col.selectbox(
        "Collection", names, index=default, key="kb_collection",
        format_func=lambda n: f"{FRIENDLY.get(n, n)} · {by_name[n]['count']:,} chunks",
        help="Every collection on this ChromaDB instance. The benchmark index and "
             "your own documents are kept in separate collections.",
    )
    info = by_name[selected]
    metric_cols[0].metric("Chunks", f"{info['count']:,}")
    dim = _embedding_dim(selected, info["count"]) if info["count"] else None
    metric_cols[1].metric("Embedding dim", dim if dim is not None else "—")
    metric_cols[2].metric("Distance", info["space"])
    metric_cols[3].metric("Store", _store_label().split(" · ")[0],
                          help=_store_label())
else:
    selected, info = None, {"count": 0}

tab_chunks, tab_upload, tab_graph = st.tabs([
    ":material/table_rows: Chunks",
    ":material/upload_file: Upload documents",
    ":material/hub: Knowledge graph",
])


# ── Chunks: browse and search in one table ────────────────────────────────────

with tab_chunks:
    if not selected or info["count"] == 0:
        st.info("This collection is empty.")
    else:
        store = get_vector_store(selected)
        # Filter on the file for your documents, on the dataset otherwise.
        field = "source" if _metadata_values(selected, "source", info["count"]) else "dataset"
        values = _metadata_values(selected, field, info["count"])

        q_col, f_col, n_col = st.columns([3, 2, 1])
        query = q_col.text_input(
            "Search", key=f"kb_search_{selected}",
            placeholder="Search by meaning, e.g. Which protein does empagliflozin inhibit? "
                        "Leave empty to browse.",
        )
        choice = f_col.selectbox(
            f"Filter by {'file' if field == 'source' else field}",
            ["All"] + list(values), key=f"kb_filter_{selected}",
            format_func=lambda v: v if v == "All" else f"{v} ({values[v]:,})",
        )
        size = n_col.selectbox("Rows", [10, 25, 50, 100], index=1, key="kb_rows")
        where = None if choice == "All" else {field: choice}

        if query.strip():
            from pipeline.embeddings import get_embedding_generator

            with st.spinner("Searching…"):
                vector = get_embedding_generator().embed_query(query)
                results = store.query(vector, top_k=size, where=where)
            rows = [{"Rank": i, "Similarity": r.score, "Chunk": r.chunk_id,
                     "Source": _source_of(r.metadata), "Page": r.metadata.get("page"),
                     "Text": _preview(r.text)} for i, r in enumerate(results, 1)]
            caption = (f"Top {len(rows)} by cosine similarity to the query"
                       + ("" if where is None else f", within {choice}") + ".")
        else:
            total = values.get(choice, 0) if where else info["count"]
            pages = max(1, -(-total // size))
            page = st.number_input(f"Page (of {pages:,})", 1, pages, 1, key=f"kb_page_{selected}")
            results = store.get_chunks(where=where, limit=size, offset=(page - 1) * size)
            rows = [{"Chunk": r.chunk_id, "Source": _source_of(r.metadata),
                     "Page": r.metadata.get("page"), "Text": _preview(r.text)}
                    for r in results]
            caption = f"{total:,} chunks in storage order. Type a query above to rank them by meaning."

        if not rows:
            st.info("No chunks match.")
        else:
            df = pd.DataFrame(rows)
            if df["Page"].isna().all():
                df = df.drop(columns="Page")
            table_col, detail_col = st.columns([3, 2])
            with table_col:
                event = st.dataframe(
                    df, hide_index=True, width="stretch", height=min(38 * len(df) + 40, 640),
                    on_select="rerun", selection_mode="single-row",
                    key=f"kb_table_{selected}_{hashlib.md5(query.encode()).hexdigest()[:6]}",
                    column_config={
                        "Similarity": st.column_config.ProgressColumn(
                            "Similarity", min_value=0.0, max_value=1.0, format="%.3f"),
                        "Text": st.column_config.TextColumn("Text", width="large"),
                    },
                )
                st.caption(caption + " Select a row to read the chunk.")
            with detail_col:
                picked = event.selection.rows if event and event.selection else []
                if not picked:
                    st.info("Select a row to see the full chunk and its metadata.", icon=":material/left_click:")
                else:
                    chunk = results[picked[0]]
                    with st.container(border=True):
                        st.markdown(f"**{chunk.chunk_id}**")
                        if chunk.score_kind == "cosine":
                            st.caption(f"Similarity {chunk.score:.3f}")
                        st.markdown(" ".join(f"`{k}: {v}`" for k, v in chunk.metadata.items()
                                             if v not in (None, "")))
                        st.text(chunk.text)


# ── Upload documents ──────────────────────────────────────────────────────────

with tab_upload:
    from pipeline.ingestion import ingest_corpus
    from pipeline.knowledge_base import get_knowledge_base

    kb = get_knowledge_base("documents")
    folder = Path(settings.documents_dir)
    files = sorted(
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in {".pdf", ".txt", ".md"}
    ) if folder.is_dir() else []

    st.markdown(
        f"Add PDF, .txt or .md files to ask questions about them. They are saved to "
        f"`{folder}` and indexed in the **{FRIENDLY[kb.collection]}** collection "
        f"(`{kb.collection}`), so the BioASQ benchmark is never changed. Query them on "
        f"the Live Monitor under **Knowledge base → My documents**, or pick the "
        f"collection above to browse them. PDFs need a text layer: scanned pages are skipped."
    )
    uploads = st.file_uploader(
        "Add files", type=["pdf", "txt", "md"], accept_multiple_files=True,
        key="documents_upload",
    )
    for level, message in st.session_state.pop("kb_upload_flash", []):
        getattr(st, level)(message)

    if st.button("Save and index", icon=":material/library_add:", type="primary",
                 disabled=not (uploads or files)):
        from pipeline.graph_jobs import start_build, stop_build

        folder.mkdir(parents=True, exist_ok=True)
        for upload in uploads or []:
            # Keep only the base name, so an upload cannot write outside the folder.
            (folder / Path(upload.name).name).write_bytes(upload.getbuffer())
        flash = []
        try:
            stop_build(kb)  # a build over the old chunks would miss the new ones
            with st.spinner("Reading and indexing documents …"):
                summary = ingest_corpus("documents", source=folder)
            flash.append(("success", f"Indexed {summary.total_passages:,} pages and files as "
                                     f"{summary.total_chunks:,} chunks."))
            started = start_build(kb)
            flash.append(("info", "Building the knowledge graph for your documents in the "
                                  "background. Follow it on the **Knowledge graph** tab.")
                         if started == "started" else ("warning", f"Knowledge graph: {started}"))
        except ValueError as exc:
            flash.append(("warning", str(exc)))
        st.session_state.kb_upload_flash = flash
        _collections.clear()
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
            "To remove a file, delete it from the folder and press **Save and index** "
            "again: the collection is rebuilt from the folder each time."
        )

    with st.expander("Clear my documents", icon=":material/delete_sweep:"):
        st.caption(
            "Empties the My documents collection (its chunks, BM25 file and manifest) "
            "and removes its knowledge graph (entities, relations and chunks covered), "
            "so you can index again from scratch. The BioASQ benchmark is not touched. "
            "Command-line equivalent: `uv run python scripts/reset_data.py --documents`."
        )
        also_files = st.checkbox("Also delete the uploaded files", key="clear_docs_files")
        sure = st.checkbox("I understand this cannot be undone", key="clear_docs_sure")
        if st.button("Clear my documents", type="primary", disabled=not sure,
                     icon=":material/delete:", key="clear_docs_btn"):
            from pipeline.maintenance import clear_documents

            report = clear_documents(delete_files=also_files)
            _collections.clear()
            _metadata_values.clear()
            _graph.clear()
            st.success("Removed: " + "; ".join(report.removed))


# ── Knowledge graph ───────────────────────────────────────────────────────────


def _build_progress(kb) -> None:
    """A build's progress, refreshed every few seconds while it runs."""
    from pipeline.graph_jobs import build_status, running_pid, stop_build

    flag = f"kg_was_running_{kb.name}"

    @st.fragment(run_every=5)
    def _panel():
        if running_pid(kb) is None:
            if st.session_state.get(flag):  # it just finished: show the new graph
                st.session_state[flag] = False
                st.rerun()
            return
        st.session_state[flag] = True
        status = build_status(kb)
        done = status.extracted / status.total if status.total else 0.0
        with st.container(border=True):
            st.progress(done, text=f"Building the knowledge graph: {status.extracted:,} of "
                                   f"{status.total:,} chunks extracted. It is published as it "
                                   "grows; reload the page to see the latest graph.")
            if status.log_tail:
                st.code(status.log_tail, language=None)
            if st.button("Stop the build", key=f"kg_stop_{kb.name}", icon=":material/stop:"):
                stop_build(kb)
                st.rerun()

    _panel()


with tab_graph:
    from pipeline.knowledge_base import get_knowledge_base

    _graph_kbs = {"BioASQ benchmark": "bioasq", "My documents": "documents"}
    kb_choice = st.segmented_control(
        "Graph for", list(_graph_kbs), default="BioASQ benchmark", key="kg_kb",
        help="Each knowledge base has its own graph. The BioASQ graph is the one behind "
             "the published results; your documents' graph is built when you index them.",
    ) or "BioASQ benchmark"
    kb = get_knowledge_base(_graph_kbs[kb_choice])
    help_text = GRAPH_REBUILD_HELP if kb.name == "bioasq" else DOCUMENTS_GRAPH_HELP

    _build_progress(kb)
    gp, ncp = kb.graph_path, kb.node_chunks_path
    if not kb.has_graph:
        st.info(f"No knowledge graph has been built for **{kb.label}** yet, so graph "
                "retrieval is unavailable for it."
                + (" Fetch the prebuilt index (`scripts/fetch_index.py`) or build one."
                   if kb.name == "bioasq" else
                   " Index your documents on the **Upload documents** tab, or build it now."))
        if kb.name == "documents" and kb.bm25_path.exists():
            from pipeline.graph_jobs import running_pid, start_build

            if not running_pid(kb) and st.button("Build the graph", type="primary",
                                                 icon=":material/hub:", key="kg_build_empty"):
                msg = start_build(kb)
                if msg != "started":
                    st.warning(msg)
                else:
                    st.rerun()
        st.markdown(help_text)
        st.stop()

    stamp = (gp.stat().st_mtime_ns, ncp.stat().st_mtime_ns)
    digraph, node_chunks, entities = _graph(str(gp), str(ncp), stamp)
    covered = len({c for v in node_chunks.values() for c in v})
    corpus = by_name.get(kb.collection, {}).get("count", 0)
    ent_ids = set(entities["id"])
    ent_ent = sum(1 for u, v in digraph.edges if u in ent_ids and v in ent_ids)
    coverage = 100 * covered / corpus if corpus else 0.0

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Entities", f"{len(entities):,}")
    g2.metric("Entity relations", f"{ent_ent:,}",
              help="Edges between two entities. The remaining edges link an entity "
                   "to the chunk it came from (co-occurrence).")
    g3.metric("Chunks covered", f"{covered:,}")
    g4.metric("Corpus coverage", f"{coverage:.1f}%",
              help=f"Share of the {kb.label} collection the graph spans. Graph retrieval "
                   "can only reach chunks inside it.")
    if coverage < 99.5 and not st.session_state.get(f"kg_was_running_{kb.name}"):
        st.caption(
            f":material/warning: The graph covers {coverage:.1f}% of the {kb.label} corpus, "
            "and graph retrieval can only reach chunks inside it. "
            + ("Build the whole corpus with `graph/build_graph.py --all` (about 100 hours; see "
               "*Manage the graph* below)." if kb.name == "bioasq" else
               "Chunks the extractor failed on are left out; *Manage the graph* below "
               "rebuilds it.")
        )

    mode = st.segmented_control(
        "View", ["Explore an entity", "Trace a question"], default="Explore an entity",
        key="kg_mode", label_visibility="collapsed",
    ) or "Explore an entity"

    if mode == "Explore an entity":
        if "kg_focus" not in st.session_state or st.session_state.kg_focus not in digraph:
            st.session_state.kg_focus = entities.iloc[0]["id"] if len(entities) else None
            st.session_state.kg_last_pick = None

        list_col, graph_col = st.columns([2, 5])
        with list_col:
            term = st.text_input("Find an entity", placeholder="e.g. aspirin, kinase, diabetes",
                                 key="kg_term")
            types = ["All types"] + sorted(entities["Type"].unique())
            type_choice = st.selectbox("Type", types, key="kg_type", label_visibility="collapsed")
            shown = entities
            if term:
                shown = shown[shown["Entity"].str.contains(term, case=False, regex=False)]
            if type_choice != "All types":
                shown = shown[shown["Type"] == type_choice]
            shown = shown.head(300).reset_index(drop=True)
            table = st.dataframe(
                shown[["Entity", "Type", "Relations", "Chunks"]], hide_index=True,
                width="stretch", height=520, on_select="rerun", selection_mode="single-row",
                key=f"kg_table_{kb.name}_{term}_{type_choice}",
            )
            st.caption(f"{len(shown):,} shown, most-connected first. Select one to explore it.")
            rows = table.selection.rows if table and table.selection else []
            picked = shown.iloc[rows[0]]["id"] if rows else None
            # Only a new table selection moves the focus, so clicking through the
            # graph is not undone by a row that stays selected.
            if picked is not None and picked != st.session_state.get("kg_last_pick"):
                st.session_state.kg_last_pick = picked
                st.session_state.kg_focus = picked

        with graph_col:
            focus = st.session_state.kg_focus
            if focus is None:
                st.info("The graph has no entities.")
            else:
                view = entity_neighbourhood(digraph, node_chunks, focus)
                name = view["nodes"][0]["label"] if view["nodes"] else focus
                kind = view["nodes"][0]["type"] if view["nodes"] else ""
                st.markdown(f"**{name}** · {TYPE_NAMES.get(kind, kind)}")
                clicked = render_entity_graph(view, key=f"kg_{hashlib.md5(focus.encode()).hexdigest()[:8]}")
                if clicked and clicked.startswith(DOC_PREFIX):
                    cid = clicked[len(DOC_PREFIX):]
                    found = get_vector_store(kb.collection).get_by_ids([cid]).get(cid)
                    with st.container(border=True):
                        st.markdown(f"**{cid}**")
                        in_chunk = [n["label"] for n in view["nodes"]
                                    if n["kind"] != "source" and cid in node_chunks.get(n["id"], [])]
                        if in_chunk:
                            st.caption("Entities shown that come from this chunk: " + ", ".join(in_chunk))
                        st.text(found.text if found else f"Chunk text not found in the {kb.label} collection.")
                elif clicked and clicked != focus and clicked in digraph:
                    st.session_state.kg_focus = clicked
                    st.rerun()

    else:
        st.caption(
            "Runs graph retrieval alone, with no language model and no dense leg, and "
            "draws how the walk reached each chunk: the same view as the Live Monitor."
        )
        question = st.text_input("Question", key="kg_question",
                                 placeholder="e.g. What does Cinryze inhibit?")
        if question.strip():
            from pipeline.retrieval.graph import GraphRetriever

            with st.spinner("Walking the graph…"):
                retriever = GraphRetriever(
                    graph_path=gp, node_chunks_path=ncp, top_k=5, fuse_dense=False,
                    vector_store=get_vector_store(kb.collection))
                chunks, tel = retriever.retrieve(question, top_k=5)
            m1, m2, m3 = st.columns(3)
            m1.metric("Seed entities", tel.get("graph_seeds_found", 0))
            m2.metric("Chunks reached", len(chunks))
            m3.metric("Retrieval", format_duration(tel.get("retrieval_ms", 0.0)))
            render_retrieval_graph(
                tel.get("graph_path"),
                chunk_texts={c.chunk_id: c.text for c in chunks},
                key=f"kg_trace_{kb.name}_{hashlib.md5(question.encode()).hexdigest()[:8]}",
            )

    # ── Graph statistics ──────────────────────────────────────────────────────
    with st.expander("Graph statistics and build provenance", icon=":material/query_stats:"):
        s_left, s_right = st.columns(2)
        types = entities["Type"].value_counts().rename_axis("Type").reset_index(name="Count")
        s_left.plotly_chart(
            px.bar(types, x="Count", y="Type", orientation="h", title="Entity types",
                   color_discrete_sequence=["#2a78d6"]).update_layout(
                height=240, margin=dict(l=0, r=0, t=40, b=0)),
            width="stretch",
        )
        labels = Counter(d.get("label", "?") for _, _, d in digraph.edges(data=True))
        rel = pd.DataFrame(labels.most_common(), columns=["Relation", "Count"])
        s_right.plotly_chart(
            px.bar(rel, x="Count", y="Relation", orientation="h", title="Relation types",
                   color_discrete_sequence=["#24a35a"]).update_layout(
                height=240, margin=dict(l=0, r=0, t=40, b=0)),
            width="stretch",
        )
        manifest = kb.graph_manifest_path
        if manifest.exists():
            man = json.loads(manifest.read_text())
            st.json({
                "scope": man.get("scope", "sample"),
                "chunks_in_scope": man.get("chunks_in_scope", man.get("chunks_attempted")),
                "extractor": man.get("extractor"),
                "chunks_attempted": man.get("chunks_attempted"),
                "chunks_ok": man.get("chunks_ok"),
                "chunks_still_to_extract": man.get("chunks_pending"),
                "failures": len(man.get("failures", []) or []),
                "corpus_coverage_pct": man.get("corpus_coverage_pct"),
                "mean_seconds_per_extraction": man.get("mean_seconds_per_extraction",
                                                       man.get("s_per_chunk")),
                "dedup_ratio": man.get("dedup_ratio"),
                "published_at": man.get("published_at"),
            })

    # ── Manage the graph ──────────────────────────────────────────────────────
    with st.expander("Manage the graph: rebuild or clear", icon=":material/build:"):
        st.markdown(help_text)
        if kb.name == "documents":
            from pipeline.graph_jobs import running_pid, start_build

            st.markdown("**Rebuild** extracts any chunk not yet in the graph, including "
                        "ones that failed before, and republishes it. **Clear my documents** "
                        "on the *Upload documents* tab removes this graph along with the chunks.")
            if st.button("Rebuild the graph", icon=":material/refresh:", key="kg_rebuild_docs",
                         disabled=running_pid(kb) is not None):
                msg = start_build(kb)
                if msg != "started":
                    st.warning(msg)
                else:
                    st.rerun()
        else:
            st.markdown("**Clear the graph** so you can rebuild it. Graph retrieval is "
                        "unavailable until a new graph is built.")
            drop_saved = st.checkbox(
                "Also delete the saved extractions (the next build re-extracts every chunk)",
                key="clear_graph_extractions")
            sure_graph = st.checkbox("I understand this cannot be undone", key="clear_graph_sure")
            if st.button("Clear the graph", type="primary", disabled=not sure_graph,
                         icon=":material/delete:", key="clear_graph_btn"):
                from pipeline.maintenance import clear_graph

                report = clear_graph(delete_extractions=drop_saved)
                _graph.clear()
                st.success("Removed: " + "; ".join(report.removed)
                           + ("" if not report.kept else ". Kept: " + "; ".join(report.kept)))
