"""
Knowledge bases: which collection and BM25 corpus a query searches.

``bioasq`` is the benchmark corpus behind the published results. ``documents``
holds your own PDF, text and Markdown files. They never share a collection, a
BM25 file or a knowledge graph, so ingesting documents cannot change the
benchmark index.

Each knowledge base has its own graph folder: ``data/graph/`` for BioASQ (the
graph behind the published results) and ``data/graph/documents/`` for your
documents. ``graph/build_graph.py --kb <name>`` builds either one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.settings import settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class KnowledgeBase:
    name: str
    label: str
    collection: str
    bm25_path: Path
    manifest_path: Path
    #: The knowledge graph graph retrieval reads, and its provenance index.
    graph_path: Path
    node_chunks_path: Path

    @property
    def graph_dir(self) -> Path:
        """Folder holding the graph, its manifest and the saved extractions."""
        return self.graph_path.parent

    @property
    def graph_manifest_path(self) -> Path:
        return self.graph_dir / "manifest.json"

    @property
    def has_graph(self) -> bool:
        """True once a graph has been built, so graph retrieval can run."""
        return self.graph_path.exists() and self.node_chunks_path.exists()


def get_knowledge_base(name: str) -> KnowledgeBase:
    """Return the knowledge base called ``name`` ("bioasq" or "documents")."""
    try:
        return KNOWLEDGE_BASES[name]()
    except KeyError:
        valid = ", ".join(sorted(KNOWLEDGE_BASES))
        raise ValueError(f"Unknown knowledge base {name!r}. Valid: {valid}.") from None


def _in_repo(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _bioasq() -> KnowledgeBase:
    return KnowledgeBase(
        name="bioasq",
        label="BioASQ benchmark",
        collection=settings.chroma_collection,
        bm25_path=DATA_DIR / "bm25_corpus.jsonl",
        manifest_path=DATA_DIR / "ingestion_manifest.json",
        graph_path=_in_repo(settings.graph_path),
        node_chunks_path=_in_repo(settings.graph_node_chunks_path),
    )


def _documents() -> KnowledgeBase:
    return KnowledgeBase(
        name="documents",
        label="My documents",
        collection=settings.documents_collection,
        bm25_path=DATA_DIR / "bm25_documents.jsonl",
        manifest_path=DATA_DIR / "documents_manifest.json",
        graph_path=DATA_DIR / "graph" / "documents" / "graph.json",
        node_chunks_path=DATA_DIR / "graph" / "documents" / "node_to_chunks.json",
    )


#: Built on access, so settings overrides (tests, .env) are always honoured.
KNOWLEDGE_BASES = {"bioasq": _bioasq, "documents": _documents}
