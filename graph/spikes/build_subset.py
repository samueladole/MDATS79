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

import networkx as nx  # noqa: E402

from docling_graph import run_pipeline  # noqa: E402
from pipeline_config import build_config  # noqa: E402

HERE = Path(__file__).parent
DOCS = HERE / "docs"
OUT = HERE / "out"


def chunk_id_from_stem(stem: str) -> str:
    """'9797__chunk_0' -> '9797::chunk_0' (the id used by Chroma and BM25)."""
    return stem.replace("__chunk_", "::chunk_")


def extract_one(path: Path) -> tuple[str, nx.DiGraph | None, float, str | None]:
    t0 = time.time()
    try:
        ctx = run_pipeline(build_config(path))
        return chunk_id_from_stem(path.stem), ctx.knowledge_graph, time.time() - t0, None
    except Exception as exc:  # noqa: BLE001
        return chunk_id_from_stem(path.stem), None, time.time() - t0, type(exc).__name__


def main(concurrency: int = 8) -> None:
    docs = sorted(DOCS.glob("*.md"))
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
    OUT.mkdir(exist_ok=True)
    (OUT / "graph.json").write_text(
        json.dumps(nx.node_link_data(merged, edges="edges"), indent=2, default=str))
    (OUT / "node_to_chunks.json").write_text(
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
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\n=== BUILD COMPLETE ===")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
