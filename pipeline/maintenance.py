"""
Clearing derived data so it can be rebuilt.

Two independent resets, used by ``scripts/reset_data.py`` and the dashboard's
Knowledge Base page:

* **Your documents** — the ``ragscope_documents`` collection in ChromaDB, its
  BM25 file and its manifest. The uploaded files in ``data/documents/`` are
  kept unless asked for, so "Save and index" rebuilds the collection from them.
* **The knowledge graph** — the graph files graph retrieval reads (entities,
  relations and the chunks they came from). The saved per-chunk extractions
  are kept unless asked for: keeping them lets ``graph/build_graph.py
  --publish-only`` restore the graph in seconds, while deleting them forces a
  full re-extraction.

The BioASQ benchmark index is never touched here; rebuild that with
``pipeline/ingestion.py --corpus all --reset``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import settings
from pipeline.knowledge_base import PROJECT_ROOT, get_knowledge_base


def _in_repo(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass
class ResetReport:
    removed: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)

    def remove(self, path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path)
            self.removed.append(f"{path}/")
        elif path.exists():
            path.unlink()
            self.removed.append(str(path))


def graph_paths() -> dict[str, Path]:
    graph = _in_repo(settings.graph_path)
    return {
        "graph": graph,
        "node_chunks": _in_repo(settings.graph_node_chunks_path),
        "manifest": graph.parent / "manifest.json",
        "extractions": graph.parent / "extractions",
        "sample": graph.parent / "sample",
        "work": graph.parent / "work",
    }


def clear_documents(*, delete_files: bool = False) -> ResetReport:
    """Empty the documents collection and remove its BM25 file and manifest."""
    from pipeline.vectorstore import get_vector_store

    kb = get_knowledge_base("documents")
    report = ResetReport()
    store = get_vector_store(kb.collection)
    count = store.count()
    store.reset()  # delete and recreate empty, so open handles stay valid
    report.removed.append(f"ChromaDB collection '{kb.collection}' ({count:,} chunks)")
    report.remove(kb.bm25_path)
    report.remove(kb.manifest_path)
    folder = _in_repo(settings.documents_dir)
    if delete_files:
        report.remove(folder)
    elif folder.exists():
        report.kept.append(f"{folder}/ (your files; press Save and index to rebuild)")
    return report


def clear_graph(*, delete_extractions: bool = False, delete_sample: bool = False) -> ResetReport:
    """Remove the knowledge graph that graph retrieval reads."""
    p = graph_paths()
    report = ResetReport()
    for key in ("graph", "node_chunks", "manifest", "work"):
        report.remove(p[key])
    if delete_extractions:
        report.remove(p["extractions"])
    elif p["extractions"].exists():
        n = sum(1 for _ in p["extractions"].glob("*.json"))
        report.kept.append(f"{p['extractions']}/ ({n:,} saved extractions; "
                           "graph/build_graph.py --publish-only restores the graph from them)")
    if delete_sample:
        report.remove(p["sample"])
    return report
