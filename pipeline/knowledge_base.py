"""
Knowledge bases: which collection and BM25 corpus a query searches.

``bioasq`` is the benchmark corpus behind the published results. ``documents``
holds your own PDF, text and Markdown files. They never share a collection or
a BM25 file, so ingesting documents cannot change the benchmark index.
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
    #: Graph retrieval needs a knowledge graph built over this corpus.
    supports_graph: bool


def get_knowledge_base(name: str) -> KnowledgeBase:
    """Return the knowledge base called ``name`` ("bioasq" or "documents")."""
    try:
        return KNOWLEDGE_BASES[name]()
    except KeyError:
        valid = ", ".join(sorted(KNOWLEDGE_BASES))
        raise ValueError(f"Unknown knowledge base {name!r}. Valid: {valid}.") from None


def _bioasq() -> KnowledgeBase:
    return KnowledgeBase(
        name="bioasq",
        label="BioASQ benchmark",
        collection=settings.chroma_collection,
        bm25_path=DATA_DIR / "bm25_corpus.jsonl",
        manifest_path=DATA_DIR / "ingestion_manifest.json",
        supports_graph=True,
    )


def _documents() -> KnowledgeBase:
    return KnowledgeBase(
        name="documents",
        label="My documents",
        collection=settings.documents_collection,
        bm25_path=DATA_DIR / "bm25_documents.jsonl",
        manifest_path=DATA_DIR / "documents_manifest.json",
        supports_graph=False,
    )


#: Built on access, so settings overrides (tests, .env) are always honoured.
KNOWLEDGE_BASES = {"bioasq": _bioasq, "documents": _documents}
