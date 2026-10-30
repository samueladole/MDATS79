"""
Build the knowledge graph used by graph retrieval.

Each corpus chunk is extracted on its own (one chunk == one file == one
docling-graph run: a directory source silently extracts the path string), and
the per-chunk graphs are composed into one corpus graph. Every node records
which chunks it came from; that provenance index is what lets the retriever
map graph hits back to passages.

Two scopes:

    uv run python graph/build_graph.py           # the 300-chunk sample (~1 h)
    uv run python graph/build_graph.py --all     # the whole corpus (~100 h)

A whole-corpus build takes days, so the builder is resumable. Each chunk's
result is saved to ``data/graph/extractions/`` as soon as it is extracted; a
rerun skips chunks already done, so an interrupted build (Ctrl-C, a crash, a
restart) continues where it stopped. The graph files the retriever reads are
re-published every ``--publish-every`` chunks and at the end of every run,
written atomically, so graph retrieval works throughout and coverage grows as
the build progresses. ``--limit N`` stops after N new chunks, for building in
batches (for example overnight).

Options
-------
    --all                 every chunk in data/bm25_corpus.jsonl
    --concurrency N       requests sent to Ollama at once (default 8)
    --limit N             extract at most N new chunks this run
    --retry-failed        retry chunks that failed in an earlier run
    --publish-every N     re-publish the graph every N chunks (default 500)
    --publish-only        rebuild the graph files from saved extractions, no extraction

A bare number is still accepted as the concurrency (``build_graph.py 8``).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import networkx as nx  # noqa: E402
from builder_config import (  # noqa: E402
    CORPUS_PATH,
    EXTRACTIONS_DIR,
    EXTRACTOR_MODEL,
    GRAPH_PATH,
    MANIFEST_PATH,
    NODE_CHUNKS_PATH,
    SAMPLE_DIR,
    WORK_DIR,
    build_config,
)

# Entity names the extractor emits when it finds nothing. Left in, these merge
# into a junk hub that dominates PageRank (the top traversal hits in the
# feasibility study were all "none" nodes).
JUNK_NAME = re.compile(
    r"^\s*(none|n/?a|not applicable|unknown|null|no items|not specified|none / .*)\s*$",
    re.IGNORECASE,
)
# The per-document root. docling-graph keys it on graph_id_fields=["source_id"],
# which the extractor leaves null, so every chunk's root would collapse into a
# shared node, producing super-hubs and fake connectivity. normalise() re-keys it.
ROOT_CLASS = "BiomedicalAbstract"


# ── Per-chunk extraction ──────────────────────────────────────────────────────


def node_name(data: dict) -> str:
    return str(data.get("name") or data.get("label") or "")


def normalise(g: nx.DiGraph, chunk_id: str) -> nx.DiGraph:
    """Make a per-chunk graph safe to compose into the corpus graph.

    1. Re-key the document root to this chunk, so roots never merge.
    2. Drop junk-named entities and their edges.
    """
    out = nx.DiGraph()
    remap: dict[str, str] = {}
    for nid, data in g.nodes(data=True):
        if data.get("__class__") == ROOT_CLASS:
            remap[nid] = f"DOC::{chunk_id}"
            out.add_node(remap[nid], **{**data, "source_id": chunk_id,
                                        "name": chunk_id, "__class__": ROOT_CLASS})
        elif JUNK_NAME.match(node_name(data)):
            continue
        else:
            remap[nid] = nid
            out.add_node(nid, **data)
    for u, v, data in g.edges(data=True):
        if u in remap and v in remap:
            out.add_edge(remap[u], remap[v], **data)
    return out


def stem(chunk_id: str) -> str:
    """'9797::chunk_0' -> '9797__chunk_0' (safe as a file name)."""
    return chunk_id.replace("::", "__")


def chunk_id_from_stem(name: str) -> str:
    return name.replace("__chunk_", "::chunk_")


def _saved(chunk_id: str) -> Path:
    return EXTRACTIONS_DIR / f"{stem(chunk_id)}.json"


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)  # readers see the old file or the new one, never half


def extract_one(chunk_id: str, text: str) -> dict:
    """Extract one chunk and save the result. Returns the saved record."""
    t0 = time.time()
    source = WORK_DIR / f"{stem(chunk_id)}.md"
    source.write_text(text)
    try:
        g = normalise(run_pipeline(build_config(source)).knowledge_graph, chunk_id)
        record = {"chunk_id": chunk_id, "ok": True, "seconds": round(time.time() - t0, 2),
                  "graph": nx.node_link_data(g, edges="edges")}
    except Exception as exc:  # noqa: BLE001 - a failed chunk is recorded, not fatal
        record = {"chunk_id": chunk_id, "ok": False, "seconds": round(time.time() - t0, 2),
                  "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    finally:
        source.unlink(missing_ok=True)
    _write_atomic(_saved(chunk_id), json.dumps(record, default=str))
    return record


# ── Scope ─────────────────────────────────────────────────────────────────────


def target_chunks(all_chunks: bool) -> list[tuple[str, str]]:
    """(chunk_id, text) for every chunk in scope, in corpus order."""
    if all_chunks:
        if not CORPUS_PATH.exists():
            sys.exit(f"{CORPUS_PATH} not found: run ingestion or scripts/fetch_index.py first")
        out = []
        with CORPUS_PATH.open() as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    out.append((rec["id"], rec["text"]))
        return out
    docs = sorted(SAMPLE_DIR.glob("*.md"))
    if not docs:
        sys.exit(f"no chunks in {SAMPLE_DIR}: run graph/sample_chunks.py first, or use --all")
    return [(chunk_id_from_stem(p.stem), p.read_text()) for p in docs]


# ── Publishing ────────────────────────────────────────────────────────────────


def _live_chunks() -> int:
    """Chunks in the graph the retriever currently reads (0 if none)."""
    try:
        return int(json.loads(MANIFEST_PATH.read_text()).get("chunks_ok") or 0)
    except (OSError, ValueError):
        return 0


def publish(scope: list[str], corpus_size: int, label: str, *, force: bool = False) -> dict:
    """Compose the saved extractions for ``scope`` into the graph files.

    Written atomically to where the retriever reads, so the graph is usable
    at every publish, including midway through a multi-day build. A build
    that is still under way never replaces a larger live graph (for example
    the 300-chunk sample while an --all build has done 40 chunks); a finished
    build, or ``force``, always publishes.
    """
    merged = nx.DiGraph()
    node_to_chunks: dict[str, set[str]] = defaultdict(set)
    failures, seconds, per_chunk_nodes = [], [], []
    pending = 0
    for chunk_id in scope:
        path = _saved(chunk_id)
        if not path.exists():
            pending += 1
            continue
        rec = json.loads(path.read_text())
        seconds.append(rec.get("seconds", 0.0))
        if not rec.get("ok"):
            failures.append({"chunk_id": chunk_id, "error": rec.get("error")})
            continue
        g = nx.node_link_graph(rec["graph"], edges="edges")
        per_chunk_nodes.append(g.number_of_nodes())
        for nid, data in g.nodes(data=True):
            if nid not in merged:  # the same entity from another chunk keeps one node
                merged.add_node(nid, **data)
            node_to_chunks[nid].add(chunk_id)
        merged.add_edges_from(g.edges(data=True))

    ok_now = len(scope) - pending - len(failures)
    live = _live_chunks()
    if pending and not force and ok_now < live:
        print(f"  kept the live graph ({live:,} chunks) until this build overtakes it "
              f"({ok_now:,} so far)", flush=True)
        return {"chunks_ok": ok_now, "corpus_coverage_pct":
                round(100 * ok_now / corpus_size, 2) if corpus_size else None,
                "chunks_in_scope": len(scope), "chunks_pending": pending,
                "entity_nodes": None, "entity_entity_edges": None, "published": False}

    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(GRAPH_PATH, json.dumps(nx.node_link_data(merged, edges="edges"), default=str))
    _write_atomic(NODE_CHUNKS_PATH,
                  json.dumps({k: sorted(v) for k, v in node_to_chunks.items()}))

    unique = merged.number_of_nodes()
    multi = sum(1 for v in node_to_chunks.values() if len(v) > 1)
    ok = len(scope) - pending - len(failures)
    is_doc = {n for n, d in merged.nodes(data=True) if d.get("__class__") == ROOT_CLASS}
    manifest = {
        "scope": label,
        "chunks_in_scope": len(scope),
        "chunks_attempted": len(scope) - pending,
        "chunks_ok": ok,
        "chunks_pending": pending,
        "corpus_chunks": corpus_size,
        "corpus_coverage_pct": round(100 * ok / corpus_size, 2) if corpus_size else None,
        "failures": failures,
        # Time per extraction call. With several calls in flight, chunks
        # complete faster than this; the build log reports that throughput.
        "mean_seconds_per_extraction": round(sum(seconds) / len(seconds), 2) if seconds else None,
        "nodes": unique,
        "edges": merged.number_of_edges(),
        "node_instances_before_merge": sum(per_chunk_nodes),
        "dedup_ratio": round(sum(per_chunk_nodes) / max(unique, 1), 2),
        "nodes_spanning_multiple_chunks": multi,
        "pct_nodes_cross_chunk": round(100 * multi / max(unique, 1), 1),
        "extractor": EXTRACTOR_MODEL,
        "doc_nodes": len(is_doc),
        "entity_nodes": unique - len(is_doc),
        "entity_entity_edges": sum(1 for u, v in merged.edges()
                                   if u not in is_doc and v not in is_doc),
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_atomic(MANIFEST_PATH, json.dumps(manifest, indent=2))
    return manifest


# ── Main ──────────────────────────────────────────────────────────────────────


def _fmt_hours(seconds: float) -> str:
    h = seconds / 3600
    return f"{h:.1f} h" if h >= 1 else f"{seconds / 60:.0f} min"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].isdigit():  # legacy form: build_graph.py 8
        argv = ["--concurrency", argv[0], *argv[1:]]
    parser = argparse.ArgumentParser(description="Build the knowledge graph for graph retrieval.")
    parser.add_argument("--all", action="store_true", help="every chunk in the corpus")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="at most N new chunks this run")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--publish-every", type=int, default=500)
    parser.add_argument("--publish-only", action="store_true")
    args = parser.parse_args(argv)

    chunks = target_chunks(args.all)
    scope = [cid for cid, _ in chunks]
    label = "all" if args.all else "sample"
    corpus_size = sum(1 for _ in CORPUS_PATH.open()) if CORPUS_PATH.exists() else 0

    if args.publish_only:
        m = publish(scope, corpus_size, label, force=True)
        print(json.dumps({k: v for k, v in m.items() if k != "failures"}, indent=2))
        return 0

    global run_pipeline
    try:
        from docling_graph import run_pipeline
    except ImportError:
        sys.exit("docling-graph is not installed. It is an optional extra:\n"
                 "  uv sync --extra graph")

    EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    def needs_work(cid: str) -> bool:
        path = _saved(cid)
        if not path.exists():
            return True
        return args.retry_failed and not json.loads(path.read_text()).get("ok")

    todo = [(cid, text) for cid, text in chunks if needs_work(cid)]
    done_before = len(chunks) - len(todo)
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"scope: {label}, {len(chunks):,} chunks; {done_before:,} already extracted; "
          f"{len(todo):,} to extract now (concurrency {args.concurrency})", flush=True)
    if not todo:
        m = publish(scope, corpus_size, label)
        print(f"nothing to extract; graph published ({m['chunks_ok']:,} chunks, "
              f"{m['corpus_coverage_pct']}% of the corpus)")
        return 0

    stop = {"now": False}

    def on_signal(signum, frame):  # noqa: ARG001
        if stop["now"]:
            raise KeyboardInterrupt
        stop["now"] = True
        print("\nstopping after the chunks in flight; press Ctrl-C again to abort", flush=True)

    signal.signal(signal.SIGINT, on_signal)

    t0, finished, failed = time.time(), 0, 0
    pool = ThreadPoolExecutor(max_workers=args.concurrency)
    try:
        # A bounded window of futures, so a 30,000-chunk run neither queues
        # everything up front nor loses more than the in-flight chunks on stop.
        queue = iter(todo)
        running = {pool.submit(extract_one, cid, text) for cid, text in
                   [next(queue) for _ in range(min(args.concurrency * 2, len(todo)))]}
        while running:
            done = next(as_completed(running))
            running.discard(done)
            rec = done.result()
            finished += 1
            failed += 0 if rec["ok"] else 1
            if not stop["now"]:
                nxt = next(queue, None)
                if nxt:
                    running.add(pool.submit(extract_one, *nxt))
            if finished % 25 == 0 or not running:
                rate = (time.time() - t0) / finished
                left = len(todo) - finished
                print(f"  {finished:,}/{len(todo):,} extracted, {failed} failed  "
                      f"{rate:.1f} s/chunk  ~{_fmt_hours(rate * left)} left", flush=True)
            if finished % args.publish_every == 0:
                m = publish(scope, corpus_size, label)
                print(f"  published: {m['chunks_ok']:,} chunks, "
                      f"{m['corpus_coverage_pct']}% of the corpus", flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        m = publish(scope, corpus_size, label)
        print(f"\n=== {'STOPPED' if stop['now'] else 'RUN COMPLETE'} ===")
        print(f"this run: {finished:,} chunks in {_fmt_hours(time.time() - t0)}, {failed} failed")
        if m.get("published") is False:
            print(f"extracted so far: {m['chunks_ok']:,} of {m['chunks_in_scope']:,} chunks "
                  f"({m['chunks_pending']:,} still to do); rerun the same command to resume")
        else:
            print(f"graph: {m['chunks_ok']:,} of {m['chunks_in_scope']:,} chunks in scope "
                  f"({m['chunks_pending']:,} still to do), {m['corpus_coverage_pct']}% of the "
                  f"corpus; {m['entity_nodes']:,} entities, "
                  f"{m['entity_entity_edges']:,} entity relations")
    return 0


run_pipeline = None  # imported in main(), so --publish-only works without docling-graph

if __name__ == "__main__":
    sys.exit(main())
