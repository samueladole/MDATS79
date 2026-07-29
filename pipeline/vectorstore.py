"""
RAGScope — Vector Store Interface
===================================
Wraps ChromaDB to provide a clean, typed interface for:
    * Storing embedded document chunks (ingestion)
    * Performing similarity search (dense retrieval)
    * Inspecting collection statistics (telemetry / dashboard)

ChromaDB is run as a standalone HTTP service (see docker-compose.yml).
The client connects via the HTTP API so the store persists independently
of the Python process lifecycle.
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass, field

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError
from loguru import logger

from config.settings import settings
from data.preprocessing.chunker import Chunk

# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class RetrievedChunk:
    """
    A single document chunk returned by a similarity search.

    Attributes
    ----------
    chunk_id   : Unique identifier (``{doc_id}::chunk_{index}``).
    text       : Chunk text content.
    score      : Cosine similarity score in [0, 1]. Higher = more similar.
    metadata   : Arbitrary metadata stored alongside the chunk.
    embedding  : The stored dense vector, if fetched (``None`` otherwise —
                 most methods don't request it, since it's the most
                 expensive field to transfer and is only needed for
                 visualisation, not retrieval or browsing).
    """

    chunk_id: str
    text: str
    score: float
    metadata: dict
    embedding: list[float] | None = field(default=None, repr=False)


def _reconnect_on_stale_collection(method):
    """
    Reconnect and retry once if the cached collection handle has gone stale.

    ``get_or_create_collection`` caches a handle bound to the collection's
    UUID at connection time. If another process deletes and recreates the
    collection under that same name in the meantime — e.g. ``ingestion.py
    --reset``, or ``VectorStore.reset()`` called from a different session —
    the server assigns a new UUID, and the cached handle starts raising
    ``NotFoundError`` on every call even though a collection with the same
    name exists again. Re-fetching by name picks up the current UUID.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except NotFoundError:
            logger.warning(
                f"ChromaDB collection '{self.collection_name}' handle is stale "
                "(deleted and recreated elsewhere) — reconnecting."
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            return method(self, *args, **kwargs)

    return wrapper


# ── VectorStore ───────────────────────────────────────────────────────────────


class VectorStore:
    """
    Manages a ChromaDB collection of embedded document chunks.

    Parameters
    ----------
    collection_name : ChromaDB collection name. Defaults to ``settings.chroma_collection``.
    host            : ChromaDB server host. Defaults to ``settings.chroma_host``.
    port            : ChromaDB server port. Defaults to ``settings.chroma_port``.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self.collection_name = collection_name or settings.chroma_collection
        host = host or settings.chroma_host
        port = port or settings.chroma_port

        logger.info(f"Connecting to ChromaDB at {host}:{port} …")
        self._client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # get_or_create ensures idempotent initialisation across restarts.
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine distance for all-MiniLM
        )
        logger.info(
            f"ChromaDB collection '{self.collection_name}' ready. "
            f"Current count: {self._collection.count():,} chunks."
        )

    # ── Ingestion ──────────────────────────────────────────────────────────────

    @_reconnect_on_stale_collection
    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        batch_size: int = 500,
    ) -> int:
        """
        Add a list of chunks with their pre-computed embeddings.

        Chunks that already exist in the collection (by ``chunk_id``) are
        skipped via upsert semantics — safe to re-run on the same corpus.

        Parameters
        ----------
        chunks     : List of Chunk objects to store.
        embeddings : Corresponding embedding vectors (same order as chunks).
        batch_size : Number of records inserted per ChromaDB call. Default 500.

        Returns
        -------
        int
            Number of new chunks added (excludes existing ones).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have equal length."
            )

        added = 0
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            self._collection.upsert(
                ids=[c.chunk_id for c in batch_chunks],
                embeddings=batch_embeddings,
                documents=[c.text for c in batch_chunks],
                metadatas=[
                    {**c.metadata, "doc_id": c.doc_id, "chunk_index": c.chunk_index}
                    for c in batch_chunks
                ],
            )
            added += len(batch_chunks)
            logger.debug(
                f"Upserted batch {i // batch_size + 1}: "
                f"{i + len(batch_chunks):,} / {len(chunks):,} chunks."
            )

        logger.info(f"VectorStore: {added:,} chunks upserted.")
        return added

    # ── Retrieval ──────────────────────────────────────────────────────────────

    @_reconnect_on_stale_collection
    def query(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Perform a cosine similarity search against the stored embeddings.

        Parameters
        ----------
        query_embedding : Dense query vector.
        top_k           : Number of results to return. Defaults to ``settings.top_k``.
        where           : Optional ChromaDB metadata filter dict.

        Returns
        -------
        list[RetrievedChunk]
            Results ordered by descending similarity score.
        """
        top_k = top_k or settings.top_k
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        chunks: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance ∈ [0, 2]; convert to similarity ∈ [0, 1].
            similarity = max(0.0, 1.0 - distance / 2.0)
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    score=round(similarity, 4),
                    metadata=metadata or {},
                )
            )

        return chunks
    
    # ── Inspection ─────────────────────────────────────────────────────────────

    @_reconnect_on_stale_collection
    def count(self) -> int:
        """Return the total number of chunks stored in the collection."""
        return self._collection.count()

    @_reconnect_on_stale_collection
    def count_where(self, where: dict) -> int:
        """
        Count chunks matching a metadata filter, without transferring
        documents or embeddings over the wire.

        Parameters
        ----------
        where : ChromaDB metadata filter dict, e.g. ``{"dataset": "msmarco"}``.
        """
        result = self._collection.get(where=where, include=[])
        return len(result["ids"])

    @_reconnect_on_stale_collection
    def get_chunks(
        self,
        where: dict | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> list[RetrievedChunk]:
        """
        Fetch chunks directly by metadata filter and offset — a plain browse,
        not a similarity search. Order is whatever the store returns them in
        (not similarity-ranked), so ``score`` is fixed at 0.0 on every result.

        Parameters
        ----------
        where  : Optional ChromaDB metadata filter dict.
        limit  : Maximum number of chunks to return.
        offset : Number of matching chunks to skip (for pagination).
        """
        kwargs: dict = {"limit": limit, "offset": offset, "include": ["documents", "metadatas"]}
        if where:
            kwargs["where"] = where
        result = self._collection.get(**kwargs)
        return [
            RetrievedChunk(chunk_id=cid, text=text, score=0.0, metadata=meta or {})
            for cid, text, meta in zip(result["ids"], result["documents"], result["metadatas"])
        ]

    @_reconnect_on_stale_collection
    def sample_embeddings(
        self,
        where: dict | None = None,
        sample_size: int = 300,
        seed: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Return a random sample of chunks with their embeddings attached —
        for visualisation (e.g. a 3D projection of the vector space), not
        retrieval or browsing.

        ChromaDB's ``.get()`` only supports contiguous ``limit``/``offset``
        windows in storage order, not random sampling, and storage order is
        not shuffled across datasets — a naive "first N" fetch would be
        dominated by whichever dataset happened to be ingested first. This
        method instead fetches matching IDs only (cheap: no documents,
        metadata, or embeddings transferred), samples from those in Python,
        and only then fetches the full records for the sampled IDs.

        Parameters
        ----------
        where       : Optional ChromaDB metadata filter dict, e.g.
                      ``{"dataset": "msmarco"}``. ``None`` samples the whole
                      collection.
        sample_size : Maximum number of chunks to sample.
        seed        : Random seed for reproducible sampling. Defaults to
                      ``settings.experiment_random_seed``.

        Returns
        -------
        list[RetrievedChunk]
            Each with ``.embedding`` populated. Order is randomised, not
            similarity-ranked, so ``score`` is fixed at 0.0.
        """
        id_probe = self._collection.get(where=where or None, include=[])
        all_ids = id_probe["ids"]
        if not all_ids:
            return []

        rng = random.Random(seed if seed is not None else settings.experiment_random_seed)
        sampled_ids = rng.sample(all_ids, min(sample_size, len(all_ids)))

        result = self._collection.get(ids=sampled_ids, include=["documents", "metadatas", "embeddings"])
        return [
            RetrievedChunk(
                chunk_id=cid,
                text=text,
                score=0.0,
                metadata=meta or {},
                embedding=list(emb) if emb is not None else None,
            )
            for cid, text, meta, emb in zip(
                result["ids"], result["documents"], result["metadatas"], result["embeddings"]
            )
        ]

    @_reconnect_on_stale_collection
    def embedding_dimension(self) -> int | None:
        """Return the dimensionality of stored embeddings, or None if the collection is empty."""
        probe = self._collection.get(limit=1, include=["embeddings"])
        embeddings = probe.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None
        return len(embeddings[0])

    def collection_info(self) -> dict:
        """Return a summary dict for the dashboard and telemetry."""
        return {
            "collection_name": self.collection_name,
            "chunk_count": self.count(),
            "chroma_host": settings.chroma_host,
            "chroma_port": settings.chroma_port,
        }

    # ── Maintenance ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Delete and recreate the collection.
        Use with caution — this erases all stored embeddings.
        """
        logger.warning(
            f"Resetting ChromaDB collection '{self.collection_name}'. "
            "All stored embeddings will be deleted."
        )
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection reset complete.")


# ── Module-level singleton ─────────────────────────────────────────────────────
_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Return the module-level singleton, initialising it on first call."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
