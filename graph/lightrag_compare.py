"""
LightRAG compared with RAGScope's docling-graph retriever.

Builds LightRAG (default in-memory / local-file storage: JSON key-value store,
NanoVectorDB, NetworkX) over the same chunks our graph was built from, with the
same extraction model and the same embedder, then compares:

  * build cost: seconds per chunk, entities, relations, failed chunks
  * retrieval latency per question, in LightRAG's naive, local and mix modes,
    both
      (a) as shipped: an LLM call per query extracts keywords, and
      (b) with those keywords supplied, isolating the retrieval itself,
    against our graph-only retriever on the same chunks
  * agreement: how often each returns anything, and top-5 chunk overlap
  * hit rate, when the questions say which file (and page) holds the answer

Two knowledge bases:

  --kb bioasq (default)  the 50-chunk BioASQ run behind the published results,
                         timed on the 200 benchmark questions
  --kb documents         the files you uploaded (one is enough), timed on your
                         own questions (--questions FILE)

LightRAG is not a project dependency. Run it layered on the environment (the
OpenTelemetry pin keeps the version ChromaDB was locked with):

    uv run --with lightrag-hku==1.5.7 --with opentelemetry-api==1.40.0 \
        python graph/lightrag_compare.py build
    uv run --with lightrag-hku==1.5.7 --with opentelemetry-api==1.40.0 \
        python graph/lightrag_compare.py compare
    uv run python graph/lightrag_compare.py report      # tables to share
    uv run --with lightrag-hku==1.5.7 --with opentelemetry-api==1.40.0 \
        python graph/lightrag_compare.py query "What is amifostine used for?"

Add --kb documents to each command for your documents, and give compare your
questions with --questions my_questions.txt.

``build`` starts from an empty LightRAG store each time. For your documents it
first brings RAGScope's docling-graph graph up to date, so one command builds
both sides.

A questions file has one question per line. Blank lines and lines starting
with # are skipped. To score hits, add the file (and optionally the page) that
holds the answer after a |:

    Does empagliflozin reduce tissue sodium? | empagliflozin_trial.pdf, page 1

Outputs go to data/graph/lightrag/ (BioASQ) or data/graph/lightrag/documents/.
graph/lightrag_results.md is the written-up analysis of the 27 October 2026
BioASQ run.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from builder_config import EXTRACTOR_MODEL, GRAPH_DIR, REPO_ROOT, select_sample  # noqa: E402

from config.settings import settings  # noqa: E402
from pipeline.knowledge_base import KnowledgeBase, get_knowledge_base  # noqa: E402

BENCHMARK_CSV = REPO_ROOT / "experiments/results/benchmark_20260813_033105.csv"
BUILDER = Path(__file__).parent / "build_graph.py"

#: Chunks in the BioASQ comparison: the first 50 of the sample that our
#: builder extracted. Your documents use every extracted chunk.
BIOASQ_CHUNKS = 50
#: Concurrent LLM requests. Ollama serves one request at a time by default
#: (OLLAMA_NUM_PARALLEL=1), and LightRAG counts time spent queued against its
#: per-call timeout, so more than 2 only turns waiting into timeouts.
CONCURRENCY = 2
#: Output cap per extraction call. Without it llama3.2 sometimes never emits
#: its completion marker and runs until LightRAG's 8-minute timeout. Complete
#: extractions are ~400 tokens, so the cap only cuts off runaway output. Our
#: docling-graph builder has the same kind of cap (max_output_tokens 8192).
MAX_OUTPUT_TOKENS = 2048
#: LightRAG's default (1) re-prompts each chunk for missed entities. 0 makes
#: one extraction call per chunk, as our builder does.
GLEANING = 0
TOP_K = 5

# Set by use_knowledge_base(); BioASQ until then.
KB: KnowledgeBase
WORK = OURS_GRAPH = OURS_NODE_CHUNKS = BUILD_STATS = RESULTS = Path()


def use_knowledge_base(name: str) -> None:
    """Point every path at knowledge base ``name``."""
    global KB, WORK, OURS_GRAPH, OURS_NODE_CHUNKS, BUILD_STATS, RESULTS
    KB = get_knowledge_base(name)
    WORK = GRAPH_DIR / "lightrag" / ("" if name == "bioasq" else name)
    OURS_GRAPH = WORK / "ours_graph.json"
    OURS_NODE_CHUNKS = WORK / "ours_node_to_chunks.json"
    BUILD_STATS = WORK / "build_stats.json"
    RESULTS = WORK / "compare_results.json"


use_knowledge_base("bioasq")


# ── Input: chunks and questions ───────────────────────────────────────────────


def _extracted_chunks() -> set[str]:
    """Chunks our graph holds, i.e. the ones docling-graph extracted."""
    graph = json.loads(KB.graph_path.read_text())
    return {str(node["id"])[len("DOC::"):] for node in graph["nodes"]
            if str(node["id"]).startswith("DOC::")}


def _select_chunks(n: int | None) -> list[tuple[str, str]]:
    """The chunks both systems are built over, as (id, text).

    Only chunks our builder extracted successfully are used, so both systems
    are given exactly the same input. BioASQ: the first n of the 300-chunk
    sample. Your documents: every chunk, in file order (or the first n).
    """
    ok = _extracted_chunks()
    if KB.name == "bioasq":
        source = select_sample()
    else:
        with KB.bm25_path.open() as fh:
            source = [(r["id"], r["text"]) for r in
                      (json.loads(line) for line in fh if line.strip())]
    picked = [(cid, text) for cid, text in source if cid in ok]
    return picked[:n] if n else picked


def _benchmark_queries() -> list[str]:
    seen, queries = set(), []
    with BENCHMARK_CSV.open() as fh:
        for row in csv.DictReader(fh):
            if row["query"] not in seen:
                seen.add(row["query"])
                queries.append(row["query"])
    return queries


_EXPECTED = re.compile(r"^(?P<file>.*?)(?:\s*,?\s*(?:page|p\.?)\s*(?P<page>\d+))?\s*$", re.I)


def read_questions(path: Path) -> list[dict]:
    """Questions from a file: ``question`` or ``question | file[, page N]``."""
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        question, _, expected = line.partition("|")
        item = {"question": question.strip(), "expected": None}
        if expected.strip():
            m = _EXPECTED.match(expected.strip())
            item["expected"] = {"file": m["file"].strip(),
                                "page": int(m["page"]) if m["page"] else None}
        out.append(item)
    if not out:
        sys.exit(f"no questions in {path}")
    return out


def _chunk_sources(chunk_ids: list[str]) -> dict[str, tuple[str, int | None]]:
    """chunk id -> (file name, page) from the knowledge base's collection."""
    from pipeline.vectorstore import get_vector_store

    found = get_vector_store(KB.collection).get_by_ids(chunk_ids)
    return {cid: (str(c.metadata.get("source") or ""), c.metadata.get("page"))
            for cid, c in found.items()}


def _is_hit(chunk_ids: list[str], expected: dict, sources: dict) -> bool:
    want_file = expected["file"].lower()
    for cid in chunk_ids:
        file, page = sources.get(cid, ("", None))
        file = file.lower()
        if want_file and want_file not in (file, file.rsplit("/", 1)[-1]):
            continue
        if expected["page"] is not None and page != expected["page"]:
            continue
        return True
    return False


# ── LightRAG setup ────────────────────────────────────────────────────────────


def _make_rag(*, cache: bool):
    import numpy as np
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_model_complete
    from lightrag.utils import EmbeddingFunc

    from pipeline.embeddings import get_embedding_generator

    embedder = get_embedding_generator()

    async def embed(texts: list[str]) -> np.ndarray:
        return np.asarray(embedder.embed_batch(list(texts)), dtype=np.float32)

    return LightRAG(
        working_dir=str(WORK / "store"),
        llm_model_func=ollama_model_complete,
        llm_model_name=EXTRACTOR_MODEL,
        llm_model_kwargs={
            "host": settings.ollama_base_url,
            "options": {"num_ctx": 32768, "temperature": 0.0, "seed": 42,
                        "num_predict": MAX_OUTPUT_TOKENS},
            "timeout": 600,
        },
        llm_model_max_async=CONCURRENCY,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedder.embedding_dim, max_token_size=512, func=embed,
            model_name=embedder.model_name,
        ),
        # One input file is already one of our chunks (<= 512 tokens), so a
        # 4096-token chunk size keeps the 1:1 mapping, as in our builder.
        chunk_token_size=4096,
        chunk_overlap_token_size=0,
        entity_extract_max_gleaning=GLEANING,
        enable_llm_cache=cache,
        enable_llm_cache_for_entity_extract=cache,
    )


def _build_our_graph() -> None:
    """Bring our documents graph up to date (resumable: done chunks are skipped)."""
    if not KB.bm25_path.exists():
        sys.exit("No documents are indexed. Upload a file on the Knowledge Base page "
                 "(Save and index), or run: uv run python pipeline/ingestion.py --corpus documents")
    print("Building RAGScope's graph over your documents first (docling-graph; chunks "
          "already extracted are skipped) …", flush=True)
    done = subprocess.run([sys.executable, str(BUILDER), "--kb", KB.name], cwd=REPO_ROOT)
    if done.returncode != 0:
        sys.exit("RAGScope's graph build did not finish; see the message above.")


async def _build(n: int | None) -> dict:
    import networkx as nx

    if KB.name != "bioasq":
        _build_our_graph()
    if not KB.has_graph:
        sys.exit(f"No RAGScope graph for {KB.label}: build it with "
                 f"uv run python graph/build_graph.py --kb {KB.name}")
    chunks = _select_chunks(n)
    if not chunks:
        sys.exit("RAGScope's graph holds no extracted chunks, so there is nothing to compare.")

    # Start from an empty store: LightRAG skips documents it already holds, which
    # would leave a rebuilt store's timing meaningless.
    shutil.rmtree(WORK / "store", ignore_errors=True)
    for stale in (RESULTS, OURS_GRAPH, OURS_NODE_CHUNKS):
        stale.unlink(missing_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    print(f"Building LightRAG over {len(chunks)} chunks …", flush=True)
    rag = _make_rag(cache=True)
    await rag.initialize_storages()
    ids = [cid for cid, _ in chunks]
    t0 = time.perf_counter()
    await rag.ainsert([text for _, text in chunks], ids=ids, file_paths=ids)
    wall = time.perf_counter() - t0
    await rag.finalize_storages()

    status = json.loads((WORK / "store" / "kv_store_doc_status.json").read_text())
    failed = [k for k, v in status.items() if v.get("status") != "processed"]
    g = nx.read_graphml(WORK / "store" / "graph_chunk_entity_relation.graphml")
    stats = {
        "kb": KB.name,
        "chunks": len(chunks),
        "chunk_ids": ids,
        "extractor": EXTRACTOR_MODEL,
        "concurrency": CONCURRENCY,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "gleaning": GLEANING,
        "wall_s": round(wall, 1),
        "s_per_chunk": round(wall / len(chunks), 2),
        "failed_chunks": len(failed),
        "entities": g.number_of_nodes(),
        "relations": g.number_of_edges(),
    }
    BUILD_STATS.write_text(json.dumps(stats, indent=2))
    return stats


# ── Our retriever on the same chunks ──────────────────────────────────────────


def _restrict_our_graph(chunk_ids: list[str]) -> dict:
    """Write our graph restricted to ``chunk_ids`` and return its size."""
    import networkx as nx

    keep_chunks = set(chunk_ids)
    data = json.loads(KB.graph_path.read_text())
    g = nx.node_link_graph(data, edges="edges")
    node_chunks = json.loads(KB.node_chunks_path.read_text())

    sub_chunks = {
        nid: [c for c in cids if c in keep_chunks]
        for nid, cids in node_chunks.items()
    }
    sub_chunks = {nid: cids for nid, cids in sub_chunks.items() if cids}
    keep = set(sub_chunks) | {f"DOC::{c}" for c in keep_chunks}
    sub = g.subgraph(n for n in g if n in keep).copy()

    OURS_GRAPH.write_text(json.dumps(nx.node_link_data(sub, edges="edges"), default=str))
    OURS_NODE_CHUNKS.write_text(json.dumps(sub_chunks))
    return {"nodes": sub.number_of_nodes(), "edges": sub.number_of_edges(),
            "entity_nodes": len(sub_chunks)}


def _our_retriever(top_k: int):
    from pipeline.retrieval.graph import GraphRetriever
    from pipeline.vectorstore import get_vector_store

    # Graph-only, so it is compared with LightRAG's graph modes alone. Chunk
    # text comes from this knowledge base's collection.
    return GraphRetriever(graph_path=OURS_GRAPH, node_chunks_path=OURS_NODE_CHUNKS,
                          top_k=top_k, fuse_dense=False,
                          vector_store=get_vector_store(KB.collection))


# ── Comparison ────────────────────────────────────────────────────────────────


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def _summary(lat: list[float], hits: list[list[str]]) -> dict:
    return {
        "median_ms": round(statistics.median(lat), 1),
        "p95_ms": round(_pct(lat, 0.95), 1),
        "nonempty_pct": round(100 * sum(1 for h in hits if h) / len(hits), 1),
    }


def _overlap(a: list[list[str]], b: list[list[str]]) -> float | None:
    both = [(x, y) for x, y in zip(a, b, strict=True) if x and y]
    if not both:
        return None
    return round(statistics.mean(len(set(x) & set(y)) / TOP_K for x, y in both), 3)


def _hit_rate(hits: list[list[str]], questions: list[dict], sources: dict) -> dict | None:
    scored = [(h, q["expected"]) for h, q in zip(hits, questions, strict=True) if q["expected"]]
    if not scored:
        return None
    n = sum(_is_hit(h, exp, sources) for h, exp in scored)
    return {"hits": n, "questions": len(scored), "pct": round(100 * n / len(scored), 1)}


async def _compare(questions_file: Path | None) -> dict:
    from lightrag import QueryParam
    from loguru import logger

    logger.remove()
    build = json.loads(BUILD_STATS.read_text())
    ours_size = _restrict_our_graph(build["chunk_ids"])
    if questions_file:
        questions = read_questions(questions_file)
    elif KB.name == "bioasq":
        questions = [{"question": q, "expected": None} for q in _benchmark_queries()]
    else:
        sys.exit("Your documents have no benchmark questions: write some in a text file, "
                 "one per line, and pass it with --questions FILE.")
    queries = [q["question"] for q in questions]
    sources = (_chunk_sources(build["chunk_ids"])
               if any(q["expected"] for q in questions) else {})

    ours = _our_retriever(TOP_K)
    ours.retrieve(queries[0], top_k=TOP_K)  # warm-up, excluded
    ours_lat, ours_hits = [], []
    for q in queries:
        t0 = time.perf_counter()
        chunks, _ = ours.retrieve(q, top_k=TOP_K)
        ours_lat.append((time.perf_counter() - t0) * 1000)
        ours_hits.append([c.chunk_id for c in chunks])

    # LightRAG with its LLM cache off, so the per-query keyword call is paid.
    rag = _make_rag(cache=False)
    await rag.initialize_storages()

    async def run(q: str, mode: str, keywords: dict | None):
        param = QueryParam(mode=mode, top_k=40, chunk_top_k=TOP_K, enable_rerank=False)
        if keywords:
            param.hl_keywords = keywords.get("high_level") or []
            param.ll_keywords = keywords.get("low_level") or []
        t0 = time.perf_counter()
        out = await rag.aquery_data(q, param=param)
        ms = (time.perf_counter() - t0) * 1000
        chunks = (out.get("data") or {}).get("chunks") or []
        kw = (out.get("metadata") or {}).get("keywords")
        return ms, [c.get("file_path") for c in chunks][:TOP_K], kw

    await run(queries[0], "local", None)  # warm-up, excluded: loads the model
    lightrag, per_mode_hits = {}, {}
    for mode in ("naive", "local", "mix"):
        lat_a, hits_a, kws = [], [], []
        for q in queries:
            ms, hits, kw = await run(q, mode, None)
            lat_a.append(ms)
            hits_a.append(hits)
            kws.append(kw)
        per_mode_hits[mode] = hits_a
        entry = {"with_llm_keywords": _summary(lat_a, hits_a),
                 "overlap_with_ours": _overlap(hits_a, ours_hits),
                 "hit_rate": _hit_rate(hits_a, questions, sources)}
        if mode != "naive":  # naive mode makes no keyword call
            lat_b = []
            for q, kw in zip(queries, kws, strict=True):
                ms, _, _ = await run(q, mode, kw or {})
                lat_b.append(ms)
            entry["keywords_supplied"] = {
                "median_ms": round(statistics.median(lat_b), 1),
                "p95_ms": round(_pct(lat_b, 0.95), 1),
            }
        lightrag[mode] = entry
        print(mode, json.dumps(entry), flush=True)
    await rag.finalize_storages()

    results = {
        "kb": KB.name,
        "queries": len(queries),
        "questions_file": str(questions_file) if questions_file else None,
        "build": {k: v for k, v in build.items() if k != "chunk_ids"},
        "ours_subgraph": ours_size,
        "ours": {**_summary(ours_lat, ours_hits),
                 "hit_rate": _hit_rate(ours_hits, questions, sources)},
        "lightrag": lightrag,
        "per_question": [
            {"question": q["question"], "expected": q["expected"], "ours": ours_hits[i],
             **{f"lightrag_{m}": per_mode_hits[m][i] for m in per_mode_hits}}
            for i, q in enumerate(questions)
        ],
        "ollama_base_url": settings.ollama_base_url,
    }
    RESULTS.write_text(json.dumps(results, indent=2))
    return results


async def _query(question: str, modes: list[str], top_k: int) -> None:
    """Run one question through both systems and print what each returns."""
    from lightrag import QueryParam
    from loguru import logger

    logger.remove()
    build = json.loads(BUILD_STATS.read_text())
    if not OURS_GRAPH.exists():
        _restrict_our_graph(build["chunk_ids"])
    sources = _chunk_sources(build["chunk_ids"]) if KB.name != "bioasq" else {}

    def show(chunks: list[tuple[str, str]], other: set[str]) -> None:
        if not chunks:
            print("    (nothing returned)")
        for rank, (cid, text) in enumerate(chunks, 1):
            both = "  [both]" if cid in other else ""
            where = ""
            if cid in sources:
                file, page = sources[cid]
                where = f"  ({file}{f', page {page}' if page else ''})"
            snippet = " ".join((text or "").split())[:110]
            print(f"    {rank}. {cid}{where}{both}\n       {snippet}")

    ours = _our_retriever(top_k)
    ours.retrieve(question, top_k=top_k)  # warm-up: loads the graph and embedder
    t0 = time.perf_counter()
    chunks, tel = ours.retrieve(question, top_k=top_k)
    ours_ms = (time.perf_counter() - t0) * 1000
    ours_hits = [(c.chunk_id, c.text) for c in chunks]

    rag = _make_rag(cache=False)
    await rag.initialize_storages()
    # Warm-up, untimed: loads the extractor model so the first mode is not
    # charged for it.
    await rag.aquery_data(question, param=QueryParam(mode="local", top_k=40, chunk_top_k=top_k,
                                                     enable_rerank=False))
    lightrag = {}
    for mode in modes:
        param = QueryParam(mode=mode, top_k=40, chunk_top_k=top_k, enable_rerank=False)
        t0 = time.perf_counter()
        out = await rag.aquery_data(question, param=param)
        ms = (time.perf_counter() - t0) * 1000
        got = (out.get("data") or {}).get("chunks") or []
        lightrag[mode] = (ms, (out.get("metadata") or {}).get("keywords"),
                          [(c.get("file_path"), c.get("content")) for c in got][:top_k])
    await rag.finalize_storages()

    found_by_lightrag = {cid for _, _, hits in lightrag.values() for cid, _ in hits}
    print(f"\nQuestion: {question}\n")
    seeds = ", ".join(s["label"] for s in tel.get("graph_seed_labels", [])) or "none"
    print(f"RAGScope graph retriever (graph-only): {ours_ms:.1f} ms, seed entities: {seeds}")
    show(ours_hits, found_by_lightrag)
    ours_ids = {cid for cid, _ in ours_hits}
    for mode, (ms, kw, hits) in lightrag.items():
        print(f"\nLightRAG {mode}: {ms:,.1f} ms"
              + ("" if mode == "naive" else f" (includes the LLM keyword call), keywords: {kw}"))
        show(hits, ours_ids)
    n = build["chunks"]
    print(f"\n[both] marks a chunk the other system also returned. Both systems search the "
          f"same {n} chunk{'' if n == 1 else 's'} (listed in {BUILD_STATS}).")


def _report() -> str:
    """The saved build and comparison results as Markdown tables."""
    res = json.loads(RESULTS.read_text())
    b, ours, lr = res["build"], res["ours"], res["lightrag"]
    manifest = KB.graph_manifest_path
    ours_build = json.loads(manifest.read_text()) if manifest.exists() else {}
    ours_s = ours_build.get("mean_seconds_per_extraction", ours_build.get("s_per_chunk", "?"))
    hit_rates = ours.get("hit_rate") is not None

    def ms(x):
        return f"{x:,.1f} ms"

    def hit(h):
        return f"{h['pct']}% ({h['hits']} of {h['questions']})" if h else ""

    lines = [
        f"## LightRAG vs RAGScope graph retriever: {KB.label} ({b['chunks']} "
        f"chunk{'' if b['chunks'] == 1 else 's'}, "
        f"{res['queries']} questions, extractor {b['extractor']})",
        "",
        "### Build",
        "",
        "| | RAGScope (docling-graph) | LightRAG |",
        "|---|---:|---:|",
        f"| Seconds per chunk | {ours_s} | {b['s_per_chunk']} |",
        f"| Failed chunks | {len(ours_build.get('failures', []) or [])} of "
        f"{ours_build.get('chunks_attempted', '?')} | {b['failed_chunks']} of {b['chunks']} |",
        f"| Graph on these chunks | {res['ours_subgraph']['nodes']:,} nodes, "
        f"{res['ours_subgraph']['edges']:,} edges | {b['entities']:,} entities, "
        f"{b['relations']:,} relations |",
        "",
        "### Retrieval latency",
        "",
        "| | Median | 95th percentile |",
        "|---|---:|---:|",
        f"| RAGScope graph retriever (graph-only) | {ms(ours['median_ms'])} | {ms(ours['p95_ms'])} |",
    ]
    for mode in ("naive", "local", "mix"):
        if mode not in lr:
            continue
        a = lr[mode]["with_llm_keywords"]
        shipped = " (vector search, no graph)" if mode == "naive" else ", as shipped (LLM keyword call)"
        lines.append(f"| LightRAG `{mode}`{shipped} | {ms(a['median_ms'])} | {ms(a['p95_ms'])} |")
        if "keywords_supplied" in lr[mode]:
            k = lr[mode]["keywords_supplied"]
            lines.append(f"| LightRAG `{mode}`, keywords supplied | {ms(k['median_ms'])} | "
                         f"{ms(k['p95_ms'])} |")
    lines += [
        "",
        "### What each returns",
        "",
        "| | Questions with at least one chunk | Mean top-5 overlap with RAGScope"
        + (" | Hit rate (answer in the top 5) |" if hit_rates else " |"),
        "|---|---:|---:|" + ("---:|" if hit_rates else ""),
        f"| RAGScope graph retriever | {ours['nonempty_pct']}% | |"
        + (f" {hit(ours['hit_rate'])} |" if hit_rates else ""),
    ]
    for mode in ("naive", "local", "mix"):
        if mode in lr:
            ov = lr[mode].get("overlap_with_ours")
            lines.append(f"| LightRAG `{mode}` | {lr[mode]['with_llm_keywords']['nonempty_pct']}% | "
                         f"{'' if ov is None else ov} |"
                         + (f" {hit(lr[mode].get('hit_rate'))} |" if hit_rates else ""))
    if b["chunks"] < TOP_K:
        lines += ["", f"Overlap counts the chunks two systems share out of {TOP_K}. With "
                      f"{b['chunks']} chunk{'' if b['chunks'] == 1 else 's'} to search, it "
                      f"cannot exceed {b['chunks'] / TOP_K:.1f}, even when both return the same."]
    if hit_rates:
        lines += ["", "A hit is a question whose expected file (and page, where given) is "
                      "among the system's top 5 chunks."]
    if res.get("per_question") and len(res["per_question"]) <= 30:
        lines += ["", "### Per question (top-5 chunks returned)", "",
                  "| Question | RAGScope | LightRAG `naive` | LightRAG `local` | LightRAG `mix` |",
                  "|---|---:|---:|---:|---:|"]
        for row in res["per_question"]:
            cells = [len(row["ours"])] + [len(row.get(f"lightrag_{m}", [])) for m in
                                          ("naive", "local", "mix")]
            lines.append(f"| {row['question']} | " + " | ".join(map(str, cells)) + " |")
    if KB.name == "bioasq":
        lines += ["", "Non-empty rates describe behaviour, not quality: see "
                      "graph/lightrag_results.md."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare LightRAG with RAGScope's graph retriever")
    kb = argparse.ArgumentParser(add_help=False)
    kb.add_argument("--kb", choices=["bioasq", "documents"], default="bioasq",
                    help="BioASQ benchmark (default) or your uploaded documents")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", parents=[kb], help="build LightRAG over the chunks")
    b.add_argument("--n", type=int, default=None,
                   help=f"number of chunks (default: {BIOASQ_CHUNKS} for BioASQ, all your documents')")
    c = sub.add_parser("compare", parents=[kb], help="time and compare retrieval")
    c.add_argument("--questions", type=Path, default=None,
                   help="questions file, one per line, optionally 'question | file[, page N]'; "
                        "required for your documents, replaces the benchmark for BioASQ")
    q = sub.add_parser("query", parents=[kb], help="run one question through both systems")
    q.add_argument("question")
    q.add_argument("--modes", nargs="+", default=["naive", "local", "mix"],
                   choices=["naive", "local", "global", "hybrid", "mix"])
    q.add_argument("--top-k", type=int, default=TOP_K)
    r = sub.add_parser("report", parents=[kb], help="print the saved results as Markdown tables")
    r.add_argument("--out", type=Path, default=None, help="also write the tables to this file")
    args = parser.parse_args()
    use_knowledge_base(args.kb)
    flag = "" if args.kb == "bioasq" else f" --kb {args.kb}"

    if args.cmd == "report":
        if not RESULTS.exists():
            sys.exit(f"{RESULTS} not found: run the build and compare steps first{flag}")
        text = _report()
        print(text)
        if args.out:
            args.out.write_text(text)
        return 0

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.cmd in ("query", "compare") and not BUILD_STATS.exists():
        sys.exit(f"{BUILD_STATS} not found: run the build step first "
                 f"(lightrag_compare.py build{flag})")
    if args.cmd == "query":
        asyncio.run(_query(args.question, args.modes, args.top_k))
    elif args.cmd == "build":
        n = args.n or (BIOASQ_CHUNKS if args.kb == "bioasq" else None)
        print(json.dumps(asyncio.run(_build(n)), indent=2))
    else:
        if args.questions and not args.questions.exists():
            sys.exit(f"{args.questions} not found")
        results = asyncio.run(_compare(args.questions))
        print(json.dumps({k: v for k, v in results.items() if k != "per_question"}, indent=2))
        print(f"\nTables: uv run python graph/lightrag_compare.py report{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
