"""
Retrieval strategies and the registry that selects between them.

``RETRIEVERS`` is the single source of truth for which strategy names are valid.
``config.settings`` validates against it, so adding a strategy here makes it
valid everywhere rather than requiring two lists to be kept in sync by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from config.settings import RETRIEVAL_STRATEGIES
from pipeline.retrieval.base import BaseRetriever

if TYPE_CHECKING:  # pragma: no cover
    from pipeline.embeddings import EmbeddingGenerator
    from pipeline.vectorstore import VectorStore

__all__ = ["BaseRetriever", "RETRIEVERS", "build_retriever"]


def _build_dense(
    *,
    store: "VectorStore | None" = None,
    embedder: "EmbeddingGenerator | None" = None,
    top_k: int | None = None,
) -> BaseRetriever:
    from pipeline.retrieval.dense import DenseRetriever

    return DenseRetriever(
        vector_store=store,
        embedding_generator=embedder,
        top_k=top_k,
    )


#: corpus path -> (file stamp, (index, ids, texts)). Tokenising and indexing
#: 30,850 chunks takes ~0.8 s, and the dashboard builds a new retriever for
#: every query, so the index is built once per corpus file and shared.
_BM25_CACHE: dict[str, tuple[tuple[int, int], tuple]] = {}


def _bm25_index(corpus_path: "Path | None" = None) -> tuple:
    # Imported here rather than at module scope: loading the BM25 corpus is
    # hybrid-specific, and pulling pipeline.ingestion in at import time would
    # make every retriever pay for one strategy's dependency.
    from pipeline.ingestion import BM25_CORPUS_PATH, load_bm25_corpus
    from pipeline.retrieval.hybrid import build_bm25

    path = Path(corpus_path) if corpus_path is not None else BM25_CORPUS_PATH
    if not path.exists():
        load_bm25_corpus(path)  # raises the explanatory FileNotFoundError
    st = path.stat()
    stamp = (st.st_mtime_ns, st.st_size)
    key = str(path.resolve())
    cached = _BM25_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    ids, texts = load_bm25_corpus(path)
    entry = (build_bm25(texts), ids, texts)
    _BM25_CACHE[key] = (stamp, entry)
    return entry


def _build_hybrid(
    *,
    store: "VectorStore | None" = None,
    embedder: "EmbeddingGenerator | None" = None,
    top_k: int | None = None,
    bm25_corpus_path: "Path | None" = None,
) -> BaseRetriever:
    from pipeline.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever(
        vector_store=store,
        embedding_generator=embedder,
        top_k=top_k,
    )
    retriever.attach_bm25_index(*_bm25_index(bm25_corpus_path))
    return retriever


def _build_graph(
    *,
    store: "VectorStore | None" = None,
    embedder: "EmbeddingGenerator | None" = None,
    top_k: int | None = None,
) -> BaseRetriever:
    # Imported lazily: loading the graph artefacts is graph-specific and
    # networkx should not be imported by callers using another strategy.
    from pipeline.retrieval.graph import GraphRetriever

    from config.settings import settings

    return GraphRetriever(
        vector_store=store,
        embedding_generator=embedder,
        top_k=top_k,
        fuse_dense=settings.graph_fuse_dense,
    )


#: Strategy name -> factory. Each factory owns its own asset loading.
RETRIEVERS: dict[str, Callable[..., BaseRetriever]] = {
    "dense": _build_dense,
    "hybrid": _build_hybrid,
    "graph": _build_graph,
}


def build_retriever(
    strategy: str,
    *,
    store: "VectorStore | None" = None,
    embedder: "EmbeddingGenerator | None" = None,
    top_k: int | None = None,
) -> BaseRetriever:
    """
    Construct the retriever for ``strategy``, fully initialised.

    Raises
    ------
    ValueError
        If ``strategy`` is not a registered strategy. This is deliberate: the
        previous inline ``if/else`` dispatch fell through to hybrid retrieval
        for any unrecognised name, producing telemetry records labelled with a
        strategy that had not actually run.
    """
    try:
        factory = RETRIEVERS[strategy]
    except KeyError:
        valid = ", ".join(sorted(RETRIEVERS))
        raise ValueError(
            f"Unknown retrieval strategy {strategy!r}. Valid strategies: {valid}."
        ) from None

    return factory(store=store, embedder=embedder, top_k=top_k)


# Fail loudly at import time if the registry and the canonical name list drift
# apart, rather than letting a strategy be valid in one place and not the other.
if set(RETRIEVERS) != set(RETRIEVAL_STRATEGIES):
    raise RuntimeError(
        "Retrieval strategy drift: RETRIEVERS has "
        f"{sorted(RETRIEVERS)} but config.settings.RETRIEVAL_STRATEGIES has "
        f"{sorted(RETRIEVAL_STRATEGIES)}. Update both."
    )
