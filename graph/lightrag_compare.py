"""
LightRAG as a GraphRAG baseline for the docling-graph retriever.

Builds LightRAG (default in-memory / local-file storage: JSON key-value store,
NanoVectorDB, NetworkX) over the same 50 corpus chunks our graph was built
from, with the same extraction model and the same embedder, then compares:

  * build cost: seconds per chunk, entities, relations, failed chunks
  * retrieval latency over the 200 benchmark queries, in LightRAG's naive,
    local and mix modes, both
      (a) as shipped: an LLM call per query extracts keywords, and
      (b) with those keywords supplied, isolating the retrieval itself,
    against our graph-only retriever on the same subgraph
  * agreement: how often each returns anything, and top-5 chunk overlap

LightRAG is not a project dependency. Run it layered on the environment (the
OpenTelemetry pin keeps the version ChromaDB was locked with):

    uv run --with lightrag-hku==1.5.7 --with opentelemetry-api==1.40.0 \
        python graph/lightrag_compare.py build
    uv run --with lightrag-hku==1.5.7 --with opentelemetry-api==1.40.0 \
        python graph/lightrag_compare.py compare
    uv run python graph/lightrag_compare.py report      # tables to share
    uv run --with lightrag-hku==1.5.7 --with opentelemetry-api==1.40.0 \
        python graph/lightrag_compare.py query "What is amifostine used for?"

Outputs go to data/graph/lightrag/. ``report`` prints the results as Markdown
tables (``--out FILE`` also saves them); graph/lightrag_results.md is the
written-up analysis of the 27 October 2026 run.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from builder_config import (  # noqa: E402
    EXTRACTOR_MODEL,
    GRAPH_DIR,
    GRAPH_PATH,
    NODE_CHUNKS_PATH,
    REPO_ROOT,
    select_sample,
)

from config.settings import settings  # noqa: E402

WORK = GRAPH_DIR / "lightrag"
OURS_GRAPH = WORK / "ours_graph.json"
OURS_NODE_CHUNKS = WORK / "ours_node_to_chunks.json"
BUILD_STATS = WORK / "build_stats.json"
RESULTS = WORK / "compare_results.json"
REPORT = Path(__file__).parent / "lightrag_results.md"
BENCHMARK_CSV = REPO_ROOT / "experiments/results/benchmark_20260813_033105.csv"

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


def _select_chunks(n: int) -> list[tuple[str, str]]:
    """The first n sample chunks our graph extracted successfully, as (id, text).

    Restricting to chunks our builder succeeded on means both systems are
    given exactly the same input.
    """
    graph = json.loads(GRAPH_PATH.read_text())
    ok = {str(node["id"])[len("DOC::"):] for node in graph["nodes"]
          if str(node["id"]).startswith("DOC::")}
    picked = []
    for cid, text in select_sample():
        if cid in ok:
            picked.append((cid, text))
        if len(picked) == n:
            break
    return picked


def _benchmark_queries() -> list[str]:
    seen, queries = set(), []
    with BENCHMARK_CSV.open() as fh:
        for row in csv.DictReader(fh):
            if row["query"] not in seen:
                seen.add(row["query"])
                queries.append(row["query"])
    return queries


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


async def _build(n: int) -> dict:
    import networkx as nx

    chunks = _select_chunks(n)
    WORK.mkdir(parents=True, exist_ok=True)
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
    data = json.loads(GRAPH_PATH.read_text())
    g = nx.node_link_graph(data, edges="edges")
    node_chunks = json.loads(NODE_CHUNKS_PATH.read_text())

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
    return round(statistics.mean(len(set(x) & set(y)) / 5 for x, y in both), 3)


async def _compare() -> dict:
    from lightrag import QueryParam
    from loguru import logger

    from pipeline.retrieval.graph import GraphRetriever

    logger.remove()
    build = json.loads(BUILD_STATS.read_text())
    ours_size = _restrict_our_graph(build["chunk_ids"])
    queries = _benchmark_queries()

    # Ours: graph-only, so it is compared with LightRAG's graph modes alone.
    ours = GraphRetriever(graph_path=OURS_GRAPH, node_chunks_path=OURS_NODE_CHUNKS,
                          top_k=5, fuse_dense=False)
    ours.retrieve(queries[0], top_k=5)  # warm-up, excluded
    ours_lat, ours_hits = [], []
    for q in queries:
        t0 = time.perf_counter()
        chunks, _ = ours.retrieve(q, top_k=5)
        ours_lat.append((time.perf_counter() - t0) * 1000)
        ours_hits.append([c.chunk_id for c in chunks])

    # LightRAG with its LLM cache off, so the per-query keyword call is paid.
    rag = _make_rag(cache=False)
    await rag.initialize_storages()

    async def run(q: str, mode: str, keywords: dict | None):
        param = QueryParam(mode=mode, top_k=40, chunk_top_k=5, enable_rerank=False)
        if keywords:
            param.hl_keywords = keywords.get("high_level") or []
            param.ll_keywords = keywords.get("low_level") or []
        t0 = time.perf_counter()
        out = await rag.aquery_data(q, param=param)
        ms = (time.perf_counter() - t0) * 1000
        chunks = (out.get("data") or {}).get("chunks") or []
        kw = (out.get("metadata") or {}).get("keywords")
        return ms, [c.get("file_path") for c in chunks][:5], kw

    await run(queries[0], "naive", None)  # warm-up, excluded
    lightrag = {}
    for mode in ("naive", "local", "mix"):
        lat_a, hits_a, kws = [], [], []
        for q in queries:
            ms, hits, kw = await run(q, mode, None)
            lat_a.append(ms)
            hits_a.append(hits)
            kws.append(kw)
        entry = {"with_llm_keywords": _summary(lat_a, hits_a),
                 "overlap_with_ours": _overlap(hits_a, ours_hits)}
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
        "queries": len(queries),
        "build": {k: v for k, v in build.items() if k != "chunk_ids"},
        "ours_subgraph": ours_size,
        "ours": _summary(ours_lat, ours_hits),
        "lightrag": lightrag,
        "ollama_base_url": settings.ollama_base_url,
    }
    RESULTS.write_text(json.dumps(results, indent=2))
    return results


async def _query(question: str, modes: list[str], top_k: int) -> None:
    """Run one question through both systems and print what each returns."""
    from lightrag import QueryParam
    from loguru import logger

    from pipeline.retrieval.graph import GraphRetriever

    logger.remove()
    build = json.loads(BUILD_STATS.read_text())
    if not OURS_GRAPH.exists():
        _restrict_our_graph(build["chunk_ids"])

    def show(chunks: list[tuple[str, str]], other: set[str]) -> None:
        if not chunks:
            print("    (nothing returned)")
        for rank, (cid, text) in enumerate(chunks, 1):
            both = "  [both]" if cid in other else ""
            snippet = " ".join((text or "").split())[:110]
            print(f"    {rank}. {cid}{both}\n       {snippet}")

    ours = GraphRetriever(graph_path=OURS_GRAPH, node_chunks_path=OURS_NODE_CHUNKS,
                          top_k=top_k, fuse_dense=False)
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
    print("\n[both] marks a chunk the other system also returned. Only the 50 chunks in "
          "build_stats.json are searchable; ask about one of them for a meaningful result.")


def _report() -> str:
    """The saved build and comparison results as Markdown tables."""
    res = json.loads(RESULTS.read_text())
    b, ours, lr = res["build"], res["ours"], res["lightrag"]
    manifest = GRAPH_DIR / "manifest.json"
    ours_build = json.loads(manifest.read_text()) if manifest.exists() else {}
    ours_s = ours_build.get("mean_seconds_per_extraction", ours_build.get("s_per_chunk", "?"))

    def ms(x):
        return f"{x:,.1f} ms"

    lines = [
        f"## LightRAG vs RAGScope graph retriever ({b['chunks']} chunks, "
        f"{res['queries']} queries, extractor {b['extractor']})",
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
        "| | Queries with at least one chunk | Mean top-5 overlap with RAGScope |",
        "|---|---:|---:|",
        f"| RAGScope graph retriever | {ours['nonempty_pct']}% | |",
    ]
    for mode in ("naive", "local", "mix"):
        if mode in lr:
            ov = lr[mode].get("overlap_with_ours")
            lines.append(f"| LightRAG `{mode}` | {lr[mode]['with_llm_keywords']['nonempty_pct']}% | "
                         f"{'' if ov is None else ov} |")
    lines += ["", "Non-empty rates describe behaviour, not quality: see graph/lightrag_results.md."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="LightRAG baseline comparison")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build LightRAG over the sample chunks")
    b.add_argument("--n", type=int, default=50, help="number of chunks (default 50)")
    sub.add_parser("compare", help="time and compare retrieval")
    q = sub.add_parser("query", help="run one question through both systems")
    q.add_argument("question")
    q.add_argument("--modes", nargs="+", default=["naive", "local", "mix"],
                   choices=["naive", "local", "global", "hybrid", "mix"])
    q.add_argument("--top-k", type=int, default=5)
    r = sub.add_parser("report", help="print the saved results as Markdown tables")
    r.add_argument("--out", type=Path, default=None, help="also write the tables to this file")
    args = parser.parse_args()

    if args.cmd == "report":
        if not RESULTS.exists():
            sys.exit(f"{RESULTS} not found: run the build and compare steps first")
        text = _report()
        print(text)
        if args.out:
            args.out.write_text(text)
        return 0

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.cmd == "query":
        if not BUILD_STATS.exists():
            sys.exit(f"{BUILD_STATS} not found: run the build step first")
        asyncio.run(_query(args.question, args.modes, args.top_k))
        return 0
    if args.cmd == "build":
        print(json.dumps(asyncio.run(_build(args.n)), indent=2))
    else:
        print(json.dumps(asyncio.run(_compare()), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
