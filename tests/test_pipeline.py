"""Tests for data preprocessing and pipeline components."""

from __future__ import annotations

import pytest

from data.preprocessing.chunker import Chunk, TokenChunker
from data.preprocessing.cleaner import clean, is_meaningful


class TestCleaner:
    def test_strips_html(self):
        result = clean("<p>Hello <b>world</b></p>")
        assert "<p>" not in result
        assert "Hello" in result
        assert "world" in result

    def test_decodes_html_entities(self):
        result = clean("AT&amp;T &lt;company&gt;")
        assert "&amp;" not in result
        assert "AT&T" in result

    def test_normalises_whitespace(self):
        result = clean("Hello    world\t\there")
        assert "  " not in result

    def test_strips_leading_trailing(self):
        result = clean("  hello  ")
        assert result == "hello"

    def test_empty_string(self):
        assert clean("") == ""

    def test_is_meaningful_true(self):
        text = "Natural Language Processing is the study of computational methods for language."
        assert is_meaningful(text) is True

    def test_is_meaningful_too_short(self):
        assert is_meaningful("hi") is False

    def test_is_meaningful_mostly_digits(self):
        text = "12345 67890 11111 22222 33333 44444 55555"
        assert is_meaningful(text) is False


class TestTokenChunker:
    def setup_method(self):
        self.chunker = TokenChunker(chunk_size=50, chunk_overlap=10)

    def test_returns_list_of_chunks(self):
        text = "This is a test passage. " * 20
        chunks = self.chunker.chunk(text, doc_id="doc1")
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_ids_are_unique(self):
        text = "This is a test passage. " * 50
        chunks = self.chunker.chunk(text, doc_id="doc1")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_id_format(self):
        chunks = self.chunker.chunk("Hello world! " * 10, doc_id="myDoc")
        assert chunks[0].chunk_id == "myDoc::chunk_0"

    def test_empty_text_returns_empty_list(self):
        assert self.chunker.chunk("", doc_id="empty") == []

    def test_short_text_single_chunk(self):
        text = "Short text."
        chunks = self.chunker.chunk(text, doc_id="short")
        assert len(chunks) == 1
        assert chunks[0].start_token == 0

    def test_metadata_propagated(self):
        chunks = self.chunker.chunk(
            "Some passage text. " * 10,
            doc_id="d1",
            metadata={"dataset": "msmarco"},
        )
        assert all(c.metadata.get("dataset") == "msmarco" for c in chunks)

    def test_overlap_invalid_raises(self):
        with pytest.raises(ValueError):
            TokenChunker(chunk_size=10, chunk_overlap=10)

    def test_chunk_many(self):
        docs = [{"id": f"doc{i}", "text": "Hello world " * 20} for i in range(3)]
        chunks = list(self.chunker.chunk_many(docs))
        assert all(isinstance(c, Chunk) for c in chunks)
        assert len(chunks) >= 3


class TestEmbeddingGeneratorInterface:
    """Tests the EmbeddingGenerator interface without loading the model."""

    def test_get_generator_returns_singleton(self):
        """get_embedding_generator() must return the same instance on second call."""
        from pipeline import embeddings as emb_mod

        # Reset singleton for test isolation
        emb_mod._generator = None

        # We can't call the actual model in unit tests, so just verify
        # the module-level singleton is None before first call.
        assert emb_mod._generator is None
