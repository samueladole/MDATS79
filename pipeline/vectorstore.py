"""
RAGScope — Vector Store Interface
===================================
Wraps ChromaDB to provide a clean, typed interface for:
    * Storing embedded document chunks (ingestion)
    * Performing similarity search (dense retrieval)
    * Inspecting collection statistics (telemetry / dashboard)

ChromaDB runs in one of two modes, chosen by ``CHROMA_MODE``:

* ``embedded`` — inside this Python process, reading and writing the index
  under ``CHROMA_PATH``. No database server to install or start.
* ``http`` — as a separate server (see docker-compose.yml), reached over its
  HTTP API, so several processes can share one store.

Both use the same collection layout and cosine HNSW index.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

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
    score      : Relevance score, higher = more relevant. **The scale depends
                 on the retrieval strategy and scores are NOT comparable
                 across strategies** — dense returns cosine similarity
                 (typically ~0.75), hybrid returns a Reciprocal Rank Fusion
                 score (typically ~0.016). Read ``score_kind`` before
                 comparing or aggregating.
    score_kind : Which scale ``score`` is on: ``"cosine"``, ``"rrf"`` or
                 ``"none"`` (a plain browse, where score is fixed at 0.0).
    metadata   : Arbitrary metadata stored alongside the chunk.
    """

    chunk_id: str
    text: str
    score: float
    metadata: dict
    score_kind: str = "cosine"


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
    host            : ChromaDB server host (http mode). Defaults to ``settings.chroma_host``.
    port            : ChromaDB server port (http mode). Defaults to ``settings.chroma_port``.
    mode            : ``"embedded"`` or ``"http"``. Defaults to ``settings.chroma_mode``.
    path            : Index directory (embedded mode). Defaults to ``settings.chroma_path``.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        host: str | None = None,
        port: int | None = None,
        mode: str | None = None,
        path: str | Path | None = None,
    ) -> None:
        self.collection_name = collection_name or settings.chroma_collection
        mode = mode or settings.chroma_mode
        chroma_settings = ChromaSettings(anonymized_telemetry=False)

        if mode == "embedded":
            path = Path(path or settings.chroma_path)
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Opening embedded ChromaDB at {path} …")
            self._client = chromadb.PersistentClient(path=str(path), settings=chroma_settings)
        else:
            host = host or settings.chroma_host
            port = port or settings.chroma_port
            logger.info(f"Connecting to ChromaDB at {host}:{port} …")
            self._client = chromadb.HttpClient(host=host, port=port, settings=chroma_settings)

        # get_or_create ensures idempotent initialisation across restarts.
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine distance for all-MiniLM
        )
        count = self._collection.count()
        logger.info(
            f"ChromaDB collection '{self.collection_name}' ready. "
            f"Current count: {count:,} chunks."
        )
        if count == 0:
            logger.warning(
                f"Collection '{self.collection_name}' is empty. Retrieval will return "
                "nothing until it is filled: run `uv run python scripts/fetch_index.py` "
                "for the prebuilt index, or `uv run python pipeline/ingestion.py` to "
                "build it from source."
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
        where : ChromaDB metadata filter dict, e.g. ``{"dataset": "bioasq"}``.
        """
        result = self._collection.get(where=where, include=[])
        return len(result["ids"])

    @_reconnect_on_stale_collection
    def get_by_ids(self, chunk_ids: list[str]) -> dict[str, RetrievedChunk]:
        """
        Fetch specific chunks by id, returning ``{chunk_id: RetrievedChunk}``.

        Needed by any retriever that identifies chunks by some means other than
        a vector search — the graph retriever reaches chunks through entity
        provenance, so most of its hits are absent from the dense result set.
        Without this, such chunks can only be constructed as stubs with empty
        metadata, which makes ``build_record`` log ``"dataset": ""`` for them
        and silently empties the per-dataset breakdown in the analysis.

        Missing ids are omitted rather than raising. ``score`` is 0.0 here:
        this is a hydration lookup, and the caller supplies the ranking.
        """
        if not chunk_ids:
            return {}
        result = self._collection.get(
            ids=list(chunk_ids), include=["documents", "metadatas"]
        )
        return {
            cid: RetrievedChunk(
                chunk_id=cid, text=text or "", score=0.0,
                metadata=meta or {}, score_kind="none",
            )
            for cid, text, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        }

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
            RetrievedChunk(chunk_id=cid, text=text, score=0.0, metadata=meta or {}, score_kind="none")
            for cid, text, meta in zip(result["ids"], result["documents"], result["metadatas"])
        ]

    @_reconnect_on_stale_collection
    def embedding_dimension(self) -> int | None:
        """Return the dimensionality of stored embeddings, or None if the collection is empty."""
        probe = self._collection.get(limit=1, include=["embeddings"])
        embeddings = probe.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None
        return len(embeddings[0])

    @_reconnect_on_stale_collection
    def metadata_values(self, key: str) -> dict[str, int]:
        """Count chunks per distinct value of metadata ``key``, most common first.

        Reads metadata only (no documents or embeddings). Used to offer
        filters such as the source file or dataset on the Knowledge Base page.
        """
        metas = self._collection.get(include=["metadatas"])["metadatas"] or []
        counts: dict[str, int] = {}
        for m in metas:
            value = (m or {}).get(key)
            if value not in (None, ""):
                counts[str(value)] = counts.get(str(value), 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def list_collections(self) -> list[dict]:
        """Every collection on this ChromaDB instance: name, chunk count, distance."""
        out = []
        for col in self._client.list_collections():
            meta = col.metadata or {}
            out.append({"name": col.name, "count": col.count(),
                        "space": meta.get("hnsw:space", "l2")})
        return sorted(out, key=lambda c: c["name"])

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


# ── Module-level singletons ────────────────────────────────────────────────────
_stores: dict[str, VectorStore] = {}


def get_vector_store(collection_name: str | None = None) -> VectorStore:
    """Return the shared store for ``collection_name``, opening it on first use.

    Defaults to the BioASQ collection, ``settings.chroma_collection``.
    """
    name = collection_name or settings.chroma_collection
    if name not in _stores:
        _stores[name] = VectorStore(collection_name=name)
    return _stores[name]
