"""
RAGScope — Hybrid Retrieval (BM25 + Dense via RRF)
====================================================
Combines sparse (BM25) and dense retrieval scores using
Reciprocal Rank Fusion (RRF), producing a ranked list that is more
robust than either method alone — particularly for domain-specific
terminology that may not be well-captured by the embedding space.

Algorithm
---------
1. Dense retrieval   → ranked list D (ChromaDB cosine similarity)
2. BM25 retrieval    → ranked list B (rank_bm25 over the in-memory corpus)
3. RRF fusion        → each chunk receives score:
                           rrf(r) = 1 / (k + rank_in_list)
                       The final score is the sum of RRF scores from
                       both lists. Ties broken by dense score.

RRF constant k=60 follows the original Cormack et al. (2009) recommendation.

This module implements Condition B and D of the 2×2 factorial experiment:
    * Condition B: Llama 3  + Hybrid retrieval
    * Condition D: Qwen  + Hybrid retrieval

References
----------
Cormack et al. (2009) — Reciprocal Rank Fusion outperforms Condorcet fusion.
Zhao et al. (2024)    — RRF for RAG survey.
"""

from __future__ import annotations

from collections import defaultdict

from loguru import logger
from rank_bm25 import BM25Okapi

from config.settings import settings
from pipeline.embeddings import EmbeddingGenerator, get_embedding_generator
from pipeline.vectorstore import RetrievedChunk, VectorStore, get_vector_store
from telemetry.timer import Timer

# RRF smoothing constant — Cormack et al. (2009) recommendation.
RRF_K = 60


class HybridRetriever:
    """
    Retrieves chunks by fusing BM25 and dense similarity rankings via RRF.

    The BM25 index is built lazily over the corpus text loaded into memory.
    For large corpora, consider using a pre-built index persisted to disk.

    Parameters
    ----------
    vector_store        : VectorStore instance.
    embedding_generator : EmbeddingGenerator instance.
    top_k               : Number of final fused results. Defaults to ``settings.top_k``.
    dense_candidates    : Candidate pool size for dense retrieval before fusion.
                          Should be > top_k. Default 50.
    bm25_candidates     : Candidate pool size for BM25 before fusion. Default 50.
    dense_weight        : Weight for dense RRF scores [0, 1].
                          BM25 weight = 1 - dense_weight.
                          Defaults to ``settings.hybrid_dense_weight``.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        top_k: int | None = None,
        dense_candidates: int = 50,
        bm25_candidates: int = 50,
        dense_weight: float | None = None,
    ) -> None:
        self._store = vector_store or get_vector_store()
        self._embedder = embedding_generator or get_embedding_generator()
        self.top_k = top_k or settings.top_k
        self.dense_candidates = max(dense_candidates, self.top_k)
        self.bm25_candidates = max(bm25_candidates, self.top_k)
        self.dense_weight = (
            dense_weight if dense_weight is not None else settings.hybrid_dense_weight
        )
        self.bm25_weight = 1.0 - self.dense_weight

        # BM25 index: built on first retrieval call (lazy)
        self._bm25: BM25Okapi | None = None
        self._corpus_ids: list[str] = []
        self._corpus_texts: list[str] = []

    # ── BM25 index ─────────────────────────────────────────────────────────────

    def build_bm25_index(
        self,
        corpus_ids: list[str],
        corpus_texts: list[str],
    ) -> None:
        """
        Build the in-memory BM25 index from the stored corpus.

        Call this explicitly after ingestion, or it will be called
        lazily on first retrieval (which may cause a noticeable delay).

        Parameters
        ----------
        corpus_ids   : List of chunk IDs matching the ChromaDB collection.
        corpus_texts : Corresponding text strings.
        """
        logger.info(f"Building BM25 index over {len(corpus_texts):,} chunks …")
        with Timer("bm25_index_build") as t:
            tokenised = [_tokenise(text) for text in corpus_texts]
            self._bm25 = BM25Okapi(tokenised)
            self._corpus_ids = corpus_ids
            self._corpus_texts = corpus_texts
        logger.info(f"BM25 index built in {t.elapsed_ms:.0f} ms.")

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], dict]:
        """
        Retrieve the top-k fused chunks for ``query``.

        Parameters
        ----------
        query : Raw query string.
        top_k : Override the instance-level top_k for this call.

        Returns
        -------
        tuple[list[RetrievedChunk], dict]
            * Ordered list of fused chunks (descending RRF score).
            * Telemetry dict with per-stage timing and score metadata.
        """
        k = top_k or self.top_k

        if self._bm25 is None:
            raise RuntimeError(
                "BM25 index has not been built. Call build_bm25_index() before retrieve()."
            )

        # ── Stage 1: Dense retrieval ─────────────────────────────────────────
        with Timer("embed_query") as embed_timer:
            query_embedding = self._embedder.embed_query(query)

        with Timer("dense_search") as dense_timer:
            dense_chunks = self._store.query(
                query_embedding=query_embedding,
                top_k=self.dense_candidates,
            )

        # ── Stage 2: BM25 retrieval ──────────────────────────────────────────
        with Timer("bm25_search") as bm25_timer:
            bm25_top = self._bm25_top_k(query, self.bm25_candidates)

        # ── Stage 3: RRF fusion ───────────────────────────────────────────────
        with Timer("rrf_fusion") as rrf_timer:
            fused = self._fuse(dense_chunks, bm25_top, k)

        retrieval_ms = (
            embed_timer.elapsed_ms
            + dense_timer.elapsed_ms
            + bm25_timer.elapsed_ms
            + rrf_timer.elapsed_ms
        )

        telemetry = {
            "strategy": "hybrid",
            "top_k": k,
            "dense_candidates": self.dense_candidates,
            "bm25_candidates": self.bm25_candidates,
            "dense_weight": self.dense_weight,
            "bm25_weight": self.bm25_weight,
            "embed_query_ms": embed_timer.elapsed_ms,
            "dense_search_ms": dense_timer.elapsed_ms,
            "bm25_search_ms": bm25_timer.elapsed_ms,
            "rrf_fusion_ms": rrf_timer.elapsed_ms,
            "retrieval_ms": retrieval_ms,
            "chunks_retrieved": len(fused),
            "top_score": fused[0].score if fused else 0.0,
            "mean_score": (sum(c.score for c in fused) / len(fused) if fused else 0.0),
        }

        logger.debug(
            f"Hybrid retrieval: {len(fused)} chunks in {retrieval_ms:.1f} ms "
            f"(top score: {telemetry['top_score']:.3f})."
        )

        return fused, telemetry

    # ── Private helpers ────────────────────────────────────────────────────────

    def _bm25_top_k(self, query: str, k: int) -> list[tuple[str, str, float]]:
        """
        Return the top-k BM25 results as (chunk_id, text, bm25_score) tuples.
        """
        tokens = _tokenise(query)
        scores = self._bm25.get_scores(tokens)

        # Pair scores with IDs and texts; sort descending
        scored = sorted(
            zip(self._corpus_ids, self._corpus_texts, scores),
            key=lambda x: x[2],
            reverse=True,
        )
        return scored[:k]

    def _fuse(
        self,
        dense_chunks: list[RetrievedChunk],
        bm25_top: list[tuple[str, str, float]],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        Apply Reciprocal Rank Fusion and return the top-k merged chunks.
        """
        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_store: dict[str, RetrievedChunk] = {}

        # Dense RRF contribution
        for rank, chunk in enumerate(dense_chunks, start=1):
            rrf_scores[chunk.chunk_id] += self.dense_weight * (1.0 / (RRF_K + rank))
            chunk_store[chunk.chunk_id] = chunk

        # BM25 RRF contribution
        for rank, (cid, text, _) in enumerate(bm25_top, start=1):
            rrf_scores[cid] += self.bm25_weight * (1.0 / (RRF_K + rank))
            if cid not in chunk_store:
                # BM25 found a chunk not in the dense results — create a stub.
                chunk_store[cid] = RetrievedChunk(chunk_id=cid, text=text, score=0.0, metadata={})

        # Sort by fused RRF score and return top_k
        ranked_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_k]

        result = []
        for cid in ranked_ids:
            chunk = chunk_store[cid]
            # Replace score with the RRF score for transparent telemetry
            result.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=round(rrf_scores[cid], 6),
                    metadata=chunk.metadata,
                )
            )
        return result


def _tokenise(text: str) -> list[str]:
    """Simple whitespace tokeniser for BM25 (no stemming required for RRF)."""
    return text.lower().split()
