"""
Spike S3 — graph quality and entity-seeding viability.

Runs in the MAIN project venv (not the builder venv) because the seeder will
eventually live there and must use the same all-MiniLM-L6-v2 embedder as the
dense retriever.

Two separate questions, deliberately not conflated:

  A. STRUCTURE  - is this one connected graph or N disconnected islands?
                  Islands mean traversal buys nothing.
  B. LINKING    - can a query be matched to graph entities?

For (B) the raw benchmark seed-rate is COVERAGE-BOUND at this scale: a graph
over 300 of 30,850 chunks covers ~1% of the corpus, so most benchmark queries
have no relevant entity present at any linking quality. The coverage-independent
measure is whether the matcher can recover entities KNOWN to be in the graph.
That is the number that actually tests the seeder.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import networkx as nx

HERE = Path(__file__).parent
OUT = HERE / "out"
BENCH = Path("/Users/samuel/Documents/GitHub/MDATS79/experiments/results/benchmark_20260813_033105.csv")


def load_graph() -> tuple[nx.DiGraph, dict]:
    data = json.loads((OUT / "graph.json").read_text())
    g = nx.node_link_graph(data, edges="edges")
    manifest = json.loads((OUT / "manifest.json").read_text())
    return g, manifest


def structure_report(g: nx.DiGraph, manifest: dict) -> dict:
    und = g.to_undirected()
    comps = sorted(nx.connected_components(und), key=len, reverse=True)
    degs = [d for _, d in und.degree()]
    types = Counter(d.get("type", "?") for _, d in g.nodes(data=True))
    labels = Counter(d.get("label", "?") for _, _, d in g.edges(data=True))
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "components": len(comps),
        "largest_component": len(comps[0]) if comps else 0,
        "pct_in_largest": round(100 * len(comps[0]) / max(g.number_of_nodes(), 1), 1) if comps else 0,
        "singleton_components": sum(1 for c in comps if len(c) == 1),
        "degree_max": max(degs) if degs else 0,
        "degree_median": statistics.median(degs) if degs else 0,
        "degree_mean": round(statistics.mean(degs), 2) if degs else 0,
        "hubs_over_50": sum(1 for d in degs if d > 50),
        "node_types": dict(types.most_common(10)),
        "edge_labels": dict(labels.most_common(12)),
        "dedup_ratio": manifest.get("dedup_ratio"),
        "pct_nodes_cross_chunk": manifest.get("pct_nodes_cross_chunk"),
    }


def node_labels(g: nx.DiGraph) -> dict[str, str]:
    """node_id -> display name, skipping unnamed/None nodes."""
    out = {}
    for nid, d in g.nodes(data=True):
        name = d.get("name") or d.get("label")
        if isinstance(name, str) and name.strip() and name.lower() != "none":
            out[nid] = name.strip()
    return out


def seed(query: str, names: list[str], embs, model, *, k: int = 10,
         cos_thresh: float = 0.55, fuzz_thresh: float = 0.88) -> list[tuple[str, str, float]]:
    """Union of embedding-ANN and fuzzy-token matching over node labels."""
    import numpy as np
    q = model.encode([query], normalize_embeddings=True)[0]
    sims = embs @ q
    hits: dict[str, tuple[str, float]] = {}
    for i in np.argsort(-sims)[:k]:
        if sims[i] >= cos_thresh:
            hits[names[i]] = ("embed", float(sims[i]))
    # fuzzy: catches gene symbols / acronyms where MiniLM is weak.
    # stdlib difflib stands in for rapidfuzz, which is not in this venv.
    toks = [t for t in re.findall(r"[A-Za-z0-9\-]{3,}", query)]
    low = [n.lower() for n in names]
    for t in toks:
        tl = t.lower()
        for j, n in enumerate(low):
            if tl == n or (len(tl) > 4 and SequenceMatcher(None, tl, n).ratio() >= fuzz_thresh):
                hits.setdefault(names[j], ("fuzzy", 1.0))
    return [(n, how, s) for n, (how, s) in hits.items()]


def main() -> None:
    import csv
    import numpy as np
    from sentence_transformers import SentenceTransformer

    g, manifest = load_graph()
    print("=" * 62)
    print("A. STRUCTURE")
    print("=" * 62)
    rep = structure_report(g, manifest)
    for k, v in rep.items():
        print(f"  {k:28} {v}")

    labels = node_labels(g)
    names = sorted(set(labels.values()))
    print(f"\n  distinct usable entity labels: {len(names)}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(names, normalize_embeddings=True, show_progress_bar=False)

    print("\n" + "=" * 62)
    print("B1. LINKING QUALITY (coverage-independent)")
    print("=" * 62)
    print("  Can the matcher recover entities KNOWN to be in the graph?")
    probes = names[:: max(1, len(names) // 100)][:100]
    recovered = sum(1 for p in probes
                    if any(n == p for n, _, _ in seed(f"What is known about {p}?", names, embs, model)))
    print(f"  probes: {len(probes)}   recovered: {recovered}  "
          f"({100*recovered/max(len(probes),1):.1f}%)")

    print("\n" + "=" * 62)
    print("B2. BENCHMARK SEED RATE  -- CONTEXT ONLY, NOT A KILL CRITERION")
    print("=" * 62)
    try:
        import pathlib as _pl
        from data.loaders import bioasq as _bq
        subset_docs = {q.stem.split("__chunk_")[0] for q in (HERE / "docs").glob("*.md")}
        allq = (_bq.load_phase_a_queries() + _bq.load_factoid_queries()
                + _bq.load_summary_queries())
        cov = sum(1 for r in allq
                  if set(map(str, r.get("relevant_passage_ids", []))) & subset_docs)
        print(f"  corpus coverage: {manifest['chunks_attempted']}/30850 = "
              f"{100*manifest['chunks_attempted']/30850:.2f}%")
        print(f"  queries with a gold passage in this subset: {cov}/{len(allq)} "
              f"({100*cov/len(allq):.1f}%)")
        print(f"  -> a high zero-seed rate here is EXPECTED and reflects coverage,")
        print(f"     not linking quality. B1 above is the real linking measure.")
    except Exception as exc:
        print(f"  (gold-coverage check unavailable: {type(exc).__name__})")
    print()
    rows = list(csv.DictReader(BENCH.open()))
    seen, queries = set(), []
    for r in rows:
        if r["query"] not in seen:
            seen.add(r["query"])
            queries.append((r["query"], r["dataset"]))
    counts, by_role = [], {}
    for q, role in queries:
        n = len(seed(q, names, embs, model))
        counts.append(n)
        by_role.setdefault(role, []).append(n)
    zero = sum(1 for c in counts if c == 0)
    print(f"  queries: {len(counts)}")
    print(f"  zero-seed: {zero} ({100*zero/len(counts):.1f}%)")
    print(f"  seeds/query: mean {statistics.mean(counts):.2f}, "
          f"median {statistics.median(counts)}, max {max(counts)}")
    for role, cs in sorted(by_role.items()):
        z = sum(1 for c in cs if c == 0)
        print(f"    {role:20} n={len(cs):3}  zero={z:3} ({100*z/len(cs):4.1f}%)  "
              f"mean={statistics.mean(cs):.2f}")

    (OUT / "s3_results.json").write_text(json.dumps({
        "structure": rep,
        "distinct_labels": len(names),
        "linking_recovery_pct": round(100 * recovered / max(len(probes), 1), 1),
        "benchmark_zero_seed_pct": round(100 * zero / len(counts), 1),
        "benchmark_mean_seeds": round(statistics.mean(counts), 2),
        "corpus_coverage_pct": round(100 * manifest["chunks_attempted"] / 30850, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
