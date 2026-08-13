"""
Tests for the retrieval layer.

All tests use lightweight mocks / stubs so they run without a live
ChromaDB instance or a loaded embedding model.  Integration tests that
require real services are marked with @pytest.mark.integration and are
excluded from the default `uv run pytest` run.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline.retrieval.hybrid import HybridRetriever, _tokenise
from pipeline.vectorstore import RetrievedChunk

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_chunk(chunk_id: str, score: float, text: str = "passage text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        metadata={"dataset": "bioasq"},
    )


def _mock_store(chunks: list[RetrievedChunk]) -> MagicMock:
    store = MagicMock()
    store.query.return_value = chunks
    return store


def _mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 384
    return embedder


# ── Dense retriever ───────────────────────────────────────────────────────────


class TestDenseRetriever:
    def test_returns_chunks_and_telemetry(self):
        from pipeline.retrieval.dense import DenseRetriever

        expected = [_make_chunk(f"c{i}", 0.9 - i * 0.1) for i in range(5)]
        retriever = DenseRetriever(
            vector_store=_mock_store(expected),
            embedding_generator=_mock_embedder(),
            top_k=5,
        )
        chunks, tel = retriever.retrieve("What is RAG?")

        assert chunks == expected
        assert tel["strategy"] == "dense"
        assert tel["top_k"] == 5
        assert "retrieval_ms" in tel
        assert tel["retrieval_ms"] >= 0

    def test_telemetry_contains_scores(self):
        from pipeline.retrieval.dense import DenseRetriever

        chunks = [_make_chunk("c0", 0.8), _make_chunk("c1", 0.6)]
        retriever = DenseRetriever(
            vector_store=_mock_store(chunks),
            embedding_generator=_mock_embedder(),
            top_k=2,
        )
        _, tel = retriever.retrieve("test query")

        assert tel["top_score"] == pytest.approx(0.8)
        assert tel["mean_score"] == pytest.approx(0.7)

    def test_empty_results(self):
        from pipeline.retrieval.dense import DenseRetriever

        retriever = DenseRetriever(
            vector_store=_mock_store([]),
            embedding_generator=_mock_embedder(),
            top_k=5,
        )
        chunks, tel = retriever.retrieve("empty query")

        assert chunks == []
        assert tel["top_score"] == 0.0
        assert tel["mean_score"] == 0.0

    def test_top_k_override(self):
        from pipeline.retrieval.dense import DenseRetriever

        store = _mock_store([_make_chunk("c0", 0.9)])
        retriever = DenseRetriever(
            vector_store=store,
            embedding_generator=_mock_embedder(),
            top_k=5,
        )
        retriever.retrieve("query", top_k=3)
        store.query.assert_called_once()
        call_kwargs = store.query.call_args.kwargs
        assert call_kwargs["top_k"] == 3


# ── Hybrid retriever — RRF logic ──────────────────────────────────────────────


class TestHybridRetrieverRRF:
    """Unit tests for the RRF fusion algorithm — no model loading required."""

    def _build_retriever(self) -> HybridRetriever:
        retriever = HybridRetriever(
            vector_store=_mock_store([]),
            embedding_generator=_mock_embedder(),
            top_k=3,
            dense_candidates=10,
            bm25_candidates=10,
        )
        # Build a tiny in-memory BM25 index
        corpus_ids = [f"chunk_{i}" for i in range(6)]
        corpus_texts = [
            "retrieval augmented generation systems",
            "large language model hallucination",
            "vector database embedding search",
            "natural language processing techniques",
            "knowledge base question answering",
            "transformer attention mechanism",
        ]
        retriever.build_bm25_index(corpus_ids, corpus_texts)
        return retriever

    def test_raises_without_bm25_index(self):
        retriever = HybridRetriever(
            vector_store=_mock_store([]),
            embedding_generator=_mock_embedder(),
        )
        with pytest.raises(RuntimeError, match="BM25 index has not been built"):
            retriever.retrieve("test query")

    def test_fuse_combines_both_lists(self):
        retriever = self._build_retriever()

        # Dense returns chunk_0 and chunk_1 with high scores
        dense_chunks = [_make_chunk("chunk_0", 0.95), _make_chunk("chunk_1", 0.80)]
        retriever._store.query.return_value = dense_chunks

        chunks, tel = retriever.retrieve("retrieval augmented generation")

        assert len(chunks) > 0
        assert tel["strategy"] == "hybrid"
        assert "bm25_search_ms" in tel
        assert "rrf_fusion_ms" in tel

    def test_fuse_merges_unique_chunks(self):
        """RRF should surface chunks from BM25 that dense missed."""
        retriever = self._build_retriever()
        # Dense only returns chunk_5 (unrelated to query)
        retriever._store.query.return_value = [_make_chunk("chunk_5", 0.3)]

        chunks, _ = retriever.retrieve("retrieval augmented generation")

        chunk_ids = [c.chunk_id for c in chunks]
        # BM25 should surface chunk_0 ("retrieval augmented generation systems")
        assert "chunk_0" in chunk_ids

    def test_top_k_limits_output(self):
        retriever = self._build_retriever()
        retriever._store.query.return_value = [
            _make_chunk(f"chunk_{i}", 0.9 - i * 0.1) for i in range(6)
        ]
        chunks, _ = retriever.retrieve("test", top_k=2)
        assert len(chunks) <= 2

    def test_rrf_scores_descending(self):
        retriever = self._build_retriever()
        retriever._store.query.return_value = [
            _make_chunk(f"chunk_{i}", 0.9 - i * 0.05) for i in range(5)
        ]
        chunks, _ = retriever.retrieve("retrieval augmented generation", top_k=5)

        scores = [c.score for c in chunks]
        assert scores == sorted(scores, reverse=True)

    def test_telemetry_all_stages_present(self):
        retriever = self._build_retriever()
        retriever._store.query.return_value = [_make_chunk("chunk_0", 0.9)]
        _, tel = retriever.retrieve("test")

        for key in [
            "strategy",
            "top_k",
            "embed_query_ms",
            "dense_search_ms",
            "bm25_search_ms",
            "rrf_fusion_ms",
            "retrieval_ms",
            "chunks_retrieved",
            "top_score",
            "mean_score",
        ]:
            assert key in tel, f"Missing telemetry key: {key}"


# ── Tokeniser ─────────────────────────────────────────────────────────────────


class TestTokenise:
    def test_lowercases(self):
        assert _tokenise("Hello World") == ["hello", "world"]

    def test_splits_on_whitespace(self):
        assert _tokenise("one two  three") == ["one", "two", "three"]

    def test_empty(self):
        assert _tokenise("") == []

    def test_single_word(self):
        assert _tokenise("retrieval") == ["retrieval"]


# ── VectorStore ───────────────────────────────────────────────────────────────


class TestVectorStore:
    """Unit tests for VectorStore helpers — mocks the ChromaDB client."""

    def test_add_chunks_validates_length_mismatch(self):
        from data.preprocessing.chunker import Chunk
        from pipeline.vectorstore import VectorStore

        with patch("pipeline.vectorstore.chromadb.HttpClient"):
            store = VectorStore.__new__(VectorStore)
            store._collection = MagicMock()
            store._collection.count.return_value = 0

            chunk = Chunk(
                text="test", doc_id="d1", chunk_index=0, start_token=0, end_token=5, metadata={}
            )
            with pytest.raises(ValueError, match="equal length"):
                store.add_chunks([chunk], [[0.1] * 384, [0.2] * 384])

    def test_retrieved_chunk_score_conversion(self):
        """ChromaDB cosine distance → similarity conversion."""
        from pipeline.vectorstore import VectorStore

        with patch("pipeline.vectorstore.chromadb.HttpClient"):
            store = VectorStore.__new__(VectorStore)
            store.collection_name = "test"
            store._collection = MagicMock()
            store._collection.count.return_value = 0
            store._collection.query.return_value = {
                "ids": [["c0"]],
                "documents": [["passage text"]],
                "metadatas": [[{}]],
                "distances": [[0.0]],  # distance=0 → similarity=1.0
            }

            chunks = store.query([0.1] * 384, top_k=1)
            assert len(chunks) == 1
            assert chunks[0].score == pytest.approx(1.0)

    def test_distance_to_similarity_edge_cases(self):
        """distance=2 → similarity=0.0; distance=1 → similarity=0.5."""
        from pipeline.vectorstore import VectorStore

        with patch("pipeline.vectorstore.chromadb.HttpClient"):
            store = VectorStore.__new__(VectorStore)
            store.collection_name = "test"
            store._collection = MagicMock()
            store._collection.count.return_value = 0

            for distance, expected_sim in [(0.0, 1.0), (1.0, 0.5), (2.0, 0.0)]:
                store._collection.query.return_value = {
                    "ids": [["c0"]],
                    "documents": [["text"]],
                    "metadatas": [[{}]],
                    "distances": [[distance]],
                }
                chunks = store.query([0.1] * 384, top_k=1)
                assert chunks[0].score == pytest.approx(expected_sim, abs=1e-4)

    def test_collection_info_returns_dict(self):
        from pipeline.vectorstore import VectorStore

        with patch("pipeline.vectorstore.chromadb.HttpClient"):
            store = VectorStore.__new__(VectorStore)
            store.collection_name = "ragscope_corpus"
            store._collection = MagicMock()
            store._collection.count.return_value = 42_000

            info = store.collection_info()
            assert info["chunk_count"] == 42_000
            assert info["collection_name"] == "ragscope_corpus"
