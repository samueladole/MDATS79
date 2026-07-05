"""
RAGScope — Dense Retrieval
============================
Pure vector-similarity retrieval using ChromaDB.

Dense retrieval encodes the query with the same embedding model used
during ingestion, then performs an approximate nearest-neighbour search
(HNSW) over the stored chunk embeddings using cosine similarity.

This module implements Condition A and C of the 2×2 factorial experiment:
    * Condition A: Llama 3  + Dense retrieval
    * Condition C: Mistral + Dense retrieval

Reference
---------
Karpukhin et al. (2020) — Dense Passage Retrieval for Open-Domain QA.
"""

from __future__ import annotations

from loguru import logger

from config.settings import settings
from pipeline.embeddings import EmbeddingGenerator, get_embedding_generator
from pipeline.vectorstore import RetrievedChunk, VectorStore, get_vector_store
from telemetry.timer import Timer


class DenseRetriever:
    """
    Retrieves the top-k most semantically similar chunks for a given query.

    Parameters
    ----------
    vector_store      : VectorStore instance. Defaults to the module singleton.
    embedding_generator : EmbeddingGenerator instance. Defaults to the module singleton.
    top_k             : Number of chunks to retrieve. Defaults to ``settings.top_k``.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        top_k: int | None = None,
    ) -> None:
        self._store = vector_store or get_vector_store()
        self._embedder = embedding_generator or get_embedding_generator()
        self.top_k = top_k or settings.top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], dict]:
        """
        Retrieve the most relevant chunks for ``query``.

        Parameters
        ----------
        query : Raw query string (not pre-embedded).
        top_k : Override the instance-level top_k for this call.

        Returns
        -------
        tuple[list[RetrievedChunk], dict]
            * Ordered list of retrieved chunks (descending similarity).
            * Telemetry dict with timing and embedding metadata.
        """
        k = top_k or self.top_k

        with Timer("embed_query") as embed_timer:
            query_embedding = self._embedder.embed_query(query)

        with Timer("vector_search") as search_timer:
            chunks = self._store.query(
                query_embedding=query_embedding,
                top_k=k,
            )

        telemetry = {
            "strategy": "dense",
            "top_k": k,
            "embed_query_ms": embed_timer.elapsed_ms,
            "vector_search_ms": search_timer.elapsed_ms,
            "retrieval_ms": embed_timer.elapsed_ms + search_timer.elapsed_ms,
            "chunks_retrieved": len(chunks),
            "top_score": chunks[0].score if chunks else 0.0,
            "mean_score": (sum(c.score for c in chunks) / len(chunks) if chunks else 0.0),
        }

        logger.debug(
            f"Dense retrieval: {len(chunks)} chunks in "
            f"{telemetry['retrieval_ms']:.1f} ms "
            f"(top score: {telemetry['top_score']:.3f})."
        )

        return chunks, telemetry
