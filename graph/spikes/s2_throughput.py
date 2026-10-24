"""
Spike S2 — extraction throughput and concurrency scaling.

Kill criterion from the plan: >48 h projected wall-clock for the full corpus
at the best achievable concurrency.

Each chunk is a separate run_pipeline call (S1: directory sources are unsafe),
so concurrency has to be driven client-side. litellm calls are HTTP-bound, so
threads are appropriate; Ollama's own OLLAMA_NUM_PARALLEL caps the real
server-side parallelism regardless of what we request here.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from docling_graph import run_pipeline  # noqa: E402
from pipeline_config import GRAPH_DIR, build_config  # noqa: E402
from pipeline_config import SAMPLE_DIR as DOCS_DIR  # noqa: E402

CORPUS_TOTAL = 30_850


def extract_one(path: Path) -> dict:
    t0 = time.time()
    try:
        ctx = run_pipeline(build_config(path))
        g = ctx.knowledge_graph
        return {"id": path.stem, "ok": True, "s": time.time() - t0,
                "nodes": g.number_of_nodes(), "edges": g.number_of_edges()}
    except Exception as exc:  # noqa: BLE001 - we want the failure rate, not a traceback
        return {"id": path.stem, "ok": False, "s": time.time() - t0,
                "err": type(exc).__name__, "nodes": 0, "edges": 0}


def run(concurrency: int, docs: list[Path]) -> dict:
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(extract_one, docs))
    wall = time.time() - t0
    ok = [r for r in results if r["ok"]]
    per_chunk = wall / len(docs)
    return {
        "concurrency": concurrency,
        "docs": len(docs),
        "wall_s": round(wall, 1),
        "s_per_chunk": round(per_chunk, 2),
        "throughput_per_min": round(60 / per_chunk, 1),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "mean_nodes": round(sum(r["nodes"] for r in ok) / max(len(ok), 1), 1),
        "mean_edges": round(sum(r["edges"] for r in ok) / max(len(ok), 1), 1),
        "projected_full_corpus_h": round(per_chunk * CORPUS_TOTAL / 3600, 1),
        "errors": sorted({r.get("err") for r in results if not r["ok"]} - {None}),
    }


if __name__ == "__main__":
    docs = sorted(DOCS_DIR.glob("*.md"))
    if not docs:
        sys.exit("no docs — run make_sample.py first")
    levels = [int(x) for x in sys.argv[1:]] or [1]
    out = []
    for c in levels:
        r = run(c, docs)
        out.append(r)
        print(json.dumps(r), flush=True)
    GRAPH_DIR.joinpath("s2_results.json").write_text(json.dumps(out, indent=2))
