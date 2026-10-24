"""
Build a knowledge graph over a subset of the corpus.

One chunk == one file == one run_pipeline call (spike S1: directory sources
silently extract the path string). Per-chunk graphs are composed into one
corpus graph, accumulating provenance so every node knows which chunks it
came from — that index is what lets the retriever map graph hits back to
RetrievedChunk objects.

Critical assumption under test: docling-graph's stable node ids are derived
from entity identity, so the same entity appearing in two different chunks
collapses to ONE node. If that does not hold, the graph is N disconnected
islands and traversal buys nothing.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import re  # noqa: E402

import networkx as nx  # noqa: E402

from docling_graph import run_pipeline  # noqa: E402
from pipeline_config import (  # noqa: E402
    GRAPH_PATH,
    MANIFEST_PATH,
    NODE_CHUNKS_PATH,
    SAMPLE_DIR,
    build_config,
)

# Entity names the extractor emits when it finds nothing. Left in, these merge
# into a junk hub that dominates PageRank — in spike S3 the top traversal hits
# were all "none" nodes.
JUNK_NAME = re.compile(
    r"^\s*(none|n/?a|not applicable|unknown|null|no items|not specified|none / .*)\s*$",
    re.IGNORECASE,
)
# The per-document root. docling-graph keys it on graph_id_fields=["source_id"],
# which the extractor leaves null, so every chunk's root collapses into a shared
# node — producing degree-175 super-hubs and fake connectivity (spike S3).
ROOT_CLASS = "BiomedicalAbstract"


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


def chunk_id_from_stem(stem: str) -> str:
    """'9797__chunk_0' -> '9797::chunk_0' (the id used by Chroma and BM25)."""
    return stem.replace("__chunk_", "::chunk_")


def extract_one(path: Path) -> tuple[str, nx.DiGraph | None, float, str | None]:
    t0 = time.time()
    try:
        chunk_id = chunk_id_from_stem(path.stem)
        ctx = run_pipeline(build_config(path))
        return chunk_id, normalise(ctx.knowledge_graph, chunk_id), time.time() - t0, None
    except Exception as exc:  # noqa: BLE001
        return chunk_id_from_stem(path.stem), None, time.time() - t0, type(exc).__name__


def main(concurrency: int = 8) -> None:
    docs = sorted(SAMPLE_DIR.glob("*.md"))
    if not docs:
        sys.exit(f"no chunks in {SAMPLE_DIR} — run make_sample.py first")
    print(f"building over {len(docs)} chunks, concurrency={concurrency}", flush=True)

    merged = nx.DiGraph()
    node_to_chunks: dict[str, set[str]] = defaultdict(set)
    failures: list[dict] = []
    per_chunk_nodes: list[int] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for i, (chunk_id, g, secs, err) in enumerate(pool.map(extract_one, docs), 1):
            if g is None:
                failures.append({"chunk_id": chunk_id, "error": err})
            else:
                per_chunk_nodes.append(g.number_of_nodes())
                for nid, data in g.nodes(data=True):
                    if nid in merged:
                        # same entity seen in another chunk: keep the node,
                        # just record the extra provenance
                        pass
                    else:
                        merged.add_node(nid, **data)
                    node_to_chunks[nid].add(chunk_id)
                for u, v, data in g.edges(data=True):
                    merged.add_edge(u, v, **data)
            if i % 25 == 0:
                print(f"  {i}/{len(docs)}  nodes={merged.number_of_nodes()} "
                      f"edges={merged.number_of_edges()}  ({time.time()-t0:.0f}s)", flush=True)

    wall = time.time() - t0
    # Written straight to where the retriever reads, so the new graph is live
    # as soon as the build finishes. Nothing is written until extraction is
    # done, so an interrupted build leaves the previous graph intact.
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRAPH_PATH.write_text(
        json.dumps(nx.node_link_data(merged, edges="edges"), indent=2, default=str))
    NODE_CHUNKS_PATH.write_text(
        json.dumps({k: sorted(v) for k, v in node_to_chunks.items()}, indent=2))

    # The decisive number: do entities merge across chunks?
    total_node_instances = sum(per_chunk_nodes)
    unique_nodes = merged.number_of_nodes()
    multi = sum(1 for v in node_to_chunks.values() if len(v) > 1)

    manifest = {
        "chunks_attempted": len(docs),
        "chunks_ok": len(docs) - len(failures),
        "failures": failures,
        "wall_s": round(wall, 1),
        "s_per_chunk": round(wall / len(docs), 2),
        "nodes": unique_nodes,
        "edges": merged.number_of_edges(),
        "node_instances_before_merge": total_node_instances,
        "dedup_ratio": round(total_node_instances / max(unique_nodes, 1), 2),
        "nodes_spanning_multiple_chunks": multi,
        "pct_nodes_cross_chunk": round(100 * multi / max(unique_nodes, 1), 1),
        "extractor": "llama3.2:latest",
        "doc_nodes": sum(1 for _, d in merged.nodes(data=True)
                         if d.get("__class__") == ROOT_CLASS),
        "entity_nodes": sum(1 for _, d in merged.nodes(data=True)
                            if d.get("__class__") != ROOT_CLASS),
        "entity_entity_edges": sum(
            1 for u, v in merged.edges()
            if merged.nodes[u].get("__class__") != ROOT_CLASS
            and merged.nodes[v].get("__class__") != ROOT_CLASS),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print("\n=== BUILD COMPLETE ===")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
