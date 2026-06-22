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

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings
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
    """

    chunk_id: str
    text: str
    score: float
    metadata: dict


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
    
    def list_documents(self) -> list[dict]:
        """
        List all unique documents in the collection.

        Returns
        -------
        list[dict]
            Each dict contains 'doc_id', 'title', and 'source' keys.
        """
        # Query all chunks and extract unique document metadata
        results = self._collection.query(
            query_embeddings=[[0] * 384],  # Dummy embedding for full scan
            n_results=self.count(),
            include=["metadatas"],
        )

        unique_docs = {}
        for metadata in results["metadatas"][0]:
            if metadata:
                doc_id = metadata.get("doc_id")
                title = metadata.get("title", "Untitled")
                source = metadata.get("source", "Unknown")
                if doc_id not in unique_docs:
                    unique_docs[doc_id] = {"doc_id": doc_id, "title": title, "source": source}

        return list(unique_docs.values())

    # ── Inspection ─────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return the total number of chunks stored in the collection."""
        return self._collection.count()

    def collection_info(self) -> dict:
        """Return a summary dict for the dashboard and telemetry."""
        return {
            "collection_name": self.collection_name,
            "chunk_count": self.count(),
            "chroma_host": settings.chroma_host,
            "chroma_port": settings.chroma_port,
        }

    # ── Maintenance ─────────────────────────────────────────────────────────────

    def delete_chunks(self, chunk_ids: list[str]) -> int:
        """
        Delete chunks by their unique identifiers.

        Parameters
        ----------
        chunk_ids : List of chunk IDs to delete.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        self._collection.delete(ids=chunk_ids)
        logger.info(f"Deleted {len(chunk_ids):,} chunks from collection '{self.collection_name}'.")
        return len(chunk_ids)

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
